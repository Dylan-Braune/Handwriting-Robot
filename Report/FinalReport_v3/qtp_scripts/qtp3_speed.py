"""
qtp3_speed.py -- QTP3 / requirement R3: reproduction speed (<= 4 s per character).

For 5 sentences of different length (~20, 40, 60, 80, 120 characters) and nTries in {1,3,14,30}
(repeats: 3,3,3,2) measure the wall-clock time of
    SynthesizeJointBestOf(...)  [exactly server.py /api/generate settings: mmPerXh=4, lineWidthMm=185,
                                 jitter 0.5, seed 0, production reader + writer classifier]
    WriteGcode(...)
(the server additionally renders a preview and reads it back; timed separately as 'extra').
Models are loaded once before timing (server loads them at start-up).  Default numpy/BLAS thread
settings (as the server would run).  Run on an IDLE machine.  DEV LAPTOP, NOT THE ODROID N2+.

PHYSICAL WRITING TIME of the emitted G-code, two models (both computed from the SAME .gcode):
  generator : SynthesizeHandwriting.WriteStepSchedule -- rest-to-rest trapezoid per segment
              (accel 400 mm/s^2), draw 15 mm/s, travel 40 mm/s, pen pulse 350 ms.
  driver    : odroid_direct_drive.run_gcode_file as written -- constant feed per move at the MODAL
              F (no acceleration; 'G1 F900' after the first stroke makes every later G0 travel run at
              900 mm/min too, i.e. 15 mm/s, not 40), moves from calibrate()'s park corner (0,0),
              coordinates clamped to the 194 x 252 mm usable area, whole-step rounding; G4 dwell
              (0.35 s) per pen toggle; set_pen() additionally runs the pen motor until the switch confirms
              (unmeasurable here: lower bound 0 s, and 0.26 s = generator's assumption) and +0.04 s
              overrun on pen-up.  Python/GPIO per-step overhead (4 end-stop reads + 2 writes per step)
              is NOT included -- only an ODROID measurement can give it.
The driver model is the real behaviour of the shipped controller; the generator model is the
generator's estimate.  Total = synthesis + G-code generation + writing.
"""
import os
import hashlib
import math
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtp_common as C
import SynthesizeHandwriting as SY
import odroid_direct_drive as DD
from np_inference.text_model import PaperCRNNNumpy, ReadText, CharAcc
from np_inference.author_model import AuthorClassifierCNNNumpy

SENTENCES = [
    "Bring the blue folder.",
    "The delivery van left before seven today",
    "Please charge both batteries overnight and label the cable",
    "Please charge both batteries overnight and label the spare cable before the meeting",
    "The delivery van left the depot just before seven, so please charge both batteries overnight and label the spare cable.",
]
NTRIES = {1: 3, 3: 3, 14: 3, 30: 2}   # nTries -> repeats
AUTHOR = "153"


def driver_writing_time(gcode_path, pen_motor_s=0.0):
    """Replays DD.run_gcode_file's time budget on a laptop (no GPIO).  Returns dict."""
    cal = DD.calibration
    spx = spy = DD.STEPS_PER_MM
    E = DD.EDGE_TOLERANCE_MM
    w, h = cal["usable_width_mm"], cal["usable_height_mm"]
    cx_steps = cy_steps = round(E * spx)       # calibrate() parks at mm (0,0) -> steps = E*spmm
    feed = 900.0
    abs_mode = True
    cur_x = cur_y = 0.0
    t_move = t_dwell = t_pen = 0.0
    draw_len = travel_len = 0.0
    pen_down = False
    n_toggle = n_moves = 0
    steps_total = 0
    clamped = 0
    t_travel = t_draw = 0.0
    for raw in open(gcode_path, encoding="utf-8"):
        parsed = DD.parse_gcode_line(raw)
        if parsed is None:
            continue
        cmd, p = parsed
        if cmd in ("G0", "G1"):
            if "F" in p:
                feed = p["F"]
            tx = p.get("X", cur_x) if abs_mode else cur_x + p.get("X", 0.0)
            ty = p.get("Y", cur_y) if abs_mode else cur_y + p.get("Y", 0.0)
            cur_x, cur_y = tx, ty
            ox, oy = tx, ty
            tx, ty = min(max(tx, 0.0), w), min(max(ty, 0.0), h)
            if (tx, ty) != (ox, oy):
                clamped += 1
            x0 = cx_steps / spx - E
            y0 = cy_steps / spy - E
            dist = math.hypot(tx - x0, ty - y0)
            tx_steps = round((tx + E) * spx)
            ty_steps = round((ty + E) * spy)
            dxs, dys = tx_steps - cx_steps, ty_steps - cy_steps
            if dxs == 0 and dys == 0:
                continue
            major = max(abs(dxs), abs(dys), 1)
            feed_s = max(feed / 60.0, DD.GCODE_MIN_FEED_MM_S)
            step_delay = max((dist / feed_s) / major, DD.CIRCLE_STEP_DELAY_S)
            dt = major * step_delay
            t_move += dt
            if pen_down:
                draw_len += dist; t_draw += dt
            else:
                travel_len += dist; t_travel += dt
            steps_total += abs(dxs) + abs(dys)
            n_moves += 1
            cx_steps, cy_steps = tx_steps, ty_steps
        elif cmd == "G4":
            t_dwell += p.get("P", 0.0)
        elif cmd in ("M3", "M5"):
            want_down = (cmd == "M3")
            if want_down != pen_down:                 # set_pen returns at once if already in that state
                t_pen += pen_motor_s + (DD.PEN_UP_OVERRUN_S if not want_down else 0.0)
                n_toggle += 1
                pen_down = want_down
        elif cmd in ("M2", "M30"):
            break
    return dict(total_s=t_move + t_dwell + t_pen, move_s=t_move, draw_move_s=t_draw, travel_move_s=t_travel,
                dwell_G4_s=t_dwell, pen_motor_and_overrun_s=t_pen, pen_toggles=n_toggle, moves=n_moves,
                draw_len_mm=draw_len, travel_len_mm=travel_len, steps=steps_total, vertices_clamped=clamped)


def main():
    cfg = SY.GantryConfig()
    t_load = time.perf_counter()
    reader = PaperCRNNNumpy()
    authm = AuthorClassifierCNNNumpy()
    profiles = SY.LoadAllProfiles()
    load_s = time.perf_counter() - t_load
    print(f"model+profile load {load_s:.2f} s (one-off, server start-up)", flush=True)
    tmp = tempfile.mkdtemp()
    rows = []
    # warm-up (not recorded): first call pays lazy imports / caches
    SY.SynthesizeJointBestOf(AUTHOR, "warm up line", profiles[AUTHOR], nTries=1, mmPerXh=4.0,
                             lineWidthMm=cfg.boundsMaxXmm - cfg.originXmm - 5, reader=reader,
                             authorModel=authm, authorMapping=authm.author_mapping)

    plan = []
    for si, text in enumerate(SENTENCES):
        for n, reps in NTRIES.items():
            for r in range(reps):
                plan.append((si, text, n, r, AUTHOR))
    # extra: writer dependence at 58 chars, nTries 14
    for w in ("150", "551"):
        for r in range(2):
            plan.append((2, SENTENCES[2], 14, r, w))
    print(f"{len(plan)} timed runs", flush=True)
    t_all = time.time()
    for k, (si, text, n, r, w) in enumerate(plan):
        prof = profiles[w]
        t0 = time.perf_counter()
        traj = SY.SynthesizeJointBestOf(w, text, prof, nTries=n, mmPerXh=4.0,
                                        lineWidthMm=cfg.boundsMaxXmm - cfg.originXmm - 5,
                                        reader=reader, authorModel=authm, authorMapping=authm.author_mapping)
        t1 = time.perf_counter()
        gp = os.path.join(tmp, "job.gcode")
        gres = SY.WriteGcode(traj, cfg, gp, title="qtp3")
        t2 = time.perf_counter()
        # extra work the server does after generation (not part of 'synthesis + G-code')
        img = SY.RenderTrajectory(traj, pxPerMm=18.0, profile=prof)
        t3 = time.perf_counter()
        back = ReadText(img, reader)
        t4 = time.perf_counter()
        sched = SY.WriteStepSchedule(traj, cfg, os.path.join(tmp, "steps.csv"))
        drv0 = driver_writing_time(gp, pen_motor_s=0.0)
        drv26 = driver_writing_time(gp, pen_motor_s=cfg.penPulseMs / 1000.0)
        nch = len(text)
        row = dict(sentence_idx=si, text=text, n_chars=nch, n_nonspace=len(text.replace(" ", "")), writer=w,
                   nTries=n, repeat=r, lines=int(traj.meta["lines"]), n_strokes=int(traj.meta["nStrokes"]),
                   synth_s=t1 - t0, gcode_s=t2 - t1, preview_render_s=t3 - t2, readback_s=t4 - t3,
                   joint_score=traj.meta.get("jointScore"), tries_drawn=traj.meta.get("jointTries"),
                   char_acc_readback=CharAcc(back, text),
                   gcode_hash=hashlib.sha256(open(gp, "rb").read()).hexdigest()[:12],
                   gcode_lines=gres["lines"], pen_pulses=gres["penPulses"],
                   write_generator_model_s=sched["seconds"],
                   write_driver_model_lower_s=drv0["total_s"], write_driver_model_pen026_s=drv26["total_s"],
                   driver_detail_pen0=drv0, driver_detail_pen026=drv26)
        rows.append(row)
        print(f"[{k+1}/{len(plan)}] {w} n={nch:3d} nTries={n:2d} rep{r}: synth {row['synth_s']:.1f}s gcode {row['gcode_s']*1000:.0f} ms "
              f"write(gen) {row['write_generator_model_s']:.0f}s write(driver) {row['write_driver_model_lower_s']:.0f}-{row['write_driver_model_pen026_s']:.0f}s "
              f"| {(time.time()-t_all)/60:.1f} min", flush=True)
        # incremental save
        C.save_json(dict(rows=rows, model_load_s=load_s, machine=C.machine_info()), "qtp3_results_partial.json")

    # ---- aggregate ----
    agg = []
    keys = sorted({(r["writer"], r["sentence_idx"], r["nTries"]) for r in rows})
    for (w, si, n) in keys:
        rr = [r for r in rows if (r["writer"], r["sentence_idx"], r["nTries"]) == (w, si, n)]
        nch = rr[0]["n_chars"]
        d = dict(writer=w, sentence_idx=si, n_chars=nch, nTries=n, repeats=len(rr), lines=rr[0]["lines"],
                 deterministic_gcode=(len({r["gcode_hash"] for r in rr}) == 1))
        for k in ("synth_s", "gcode_s", "preview_render_s", "readback_s", "write_generator_model_s",
                  "write_driver_model_lower_s", "write_driver_model_pen026_s"):
            v = [r[k] for r in rr]
            d[k] = dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)))
        d["synth_s_per_char"] = d["synth_s"]["mean"] / nch
        d["gcode_s_per_char"] = d["gcode_s"]["mean"] / nch
        d["write_generator_s_per_char"] = d["write_generator_model_s"]["mean"] / nch
        d["write_driver_s_per_char_lower"] = d["write_driver_model_lower_s"]["mean"] / nch
        d["write_driver_s_per_char_pen026"] = d["write_driver_model_pen026_s"]["mean"] / nch
        d["total_s_per_char_generator_model"] = (d["synth_s"]["mean"] + d["gcode_s"]["mean"] + d["write_generator_model_s"]["mean"]) / nch
        d["total_s_per_char_driver_model_lower"] = (d["synth_s"]["mean"] + d["gcode_s"]["mean"] + d["write_driver_model_lower_s"]["mean"]) / nch
        d["total_s_per_char_driver_model_pen026"] = (d["synth_s"]["mean"] + d["gcode_s"]["mean"] + d["write_driver_model_pen026_s"]["mean"]) / nch
        d["requirement_s_per_char"] = 4.0
        agg.append(d)
    C.save_json(dict(rows=rows, aggregate=agg, model_load_s=load_s, sentences=SENTENCES,
                     nTries_repeats=NTRIES, author=AUTHOR, machine=C.machine_info()), "qtp3_results.json")

    # ---- figure ----
    plt = C.mpl_style()
    fig, axs = plt.subplots(1, 2, figsize=(16 * C.CM, 7 * C.CM))
    ax = axs[0]
    for k, n in enumerate(NTRIES):
        pts = [(a["n_chars"], a["synth_s_per_char"]) for a in agg if a["nTries"] == n and a["writer"] == AUTHOR]
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=f"nTries={n}", color=C.PALETTE[k])
    ax.axhline(4.0, color="red", ls="--", lw=1)
    ax.text(5, 4.15, "R3: 4 s/char", color="red", fontsize=8)
    ax.set_xlabel("sentence length [characters]"); ax.set_ylabel("synthesis time [s / character]")
    ax.set_title(f"SynthesizeJointBestOf, writer {AUTHOR} (dev laptop)", fontsize=9)
    ax.legend(fontsize=8)
    ax = axs[1]
    n = 14
    sel = sorted([a for a in agg if a["nTries"] == n and a["writer"] == AUTHOR], key=lambda a: a["n_chars"])
    xs = np.arange(len(sel))
    syn = [a["synth_s_per_char"] for a in sel]
    gc = [a["gcode_s_per_char"] for a in sel]
    wr = [a["write_driver_s_per_char_pen026"] for a in sel]
    ax.bar(xs, syn, label="synthesis", color=C.PALETTE[0])
    ax.bar(xs, gc, bottom=syn, label="G-code generation", color=C.PALETTE[2])
    ax.bar(xs, wr, bottom=np.array(syn) + np.array(gc), label="physical writing (driver model,\npen motor 0.26 s assumed)", color=C.PALETTE[3])
    ax.axhline(4.0, color="red", ls="--", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels([str(a["n_chars"]) for a in sel])
    ax.set_xlabel("sentence length [characters]"); ax.set_ylabel("time [s / character]")
    ax.set_title("total reproduction time, nTries=14", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    C.savefig(fig, "qtp3_timing_breakdown.png"); plt.close(fig)
    print("done, total", (time.time() - t_all) / 60, "min")


if __name__ == "__main__":
    main()
