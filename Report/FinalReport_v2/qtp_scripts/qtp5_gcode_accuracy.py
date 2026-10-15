"""
qtp5_gcode_accuracy.py -- QTP5 / requirement R5, SOFTWARE part (no hardware involved).

(i)   theoretical step resolution from the constants in the generator (GantryConfig) and the driver
      (odroid_direct_drive.py: MICROSTEPS, STEPS_PER_MM, calibrate()).
(ii)  path fidelity: for 10 sentences (10 writers, one sentence each, production settings
      mmPerXh=4, lineWidthMm=185 wrap, jitter 0.5, legibility = profile lambda, seed fixed) the
      planned path (trajectory scaled/translated to the bed, y flipped, BEFORE decimation) is compared
      with the path the driver would execute from the emitted G-code:
        G-code text (3 decimals)  ->  driver parse -> absolute target -> clamp_to_usable_area
        (194 x 252 mm) -> target_steps = round((mm + 5) * steps_per_mm)  ->  Bresenham stepping
      Stages isolated: (a) decimation (minSegmentMm=0.12), (b) 3-decimal rounding, (c) step
      quantisation at the driver's nominal 80 steps/mm, (d) Bresenham stepping between vertices,
      (e) the driver's usable-area clamp (194 mm wide: G-code bounds are 200 mm!).
      Metric: distance in mm from every EXECUTED step position to the planned polyline (point-to-curve,
      polylines resampled at 0.005 mm) and from every planned vertex to the executed path (sampling
      error <= 0.0025 mm).
(iii) repeatability in software: same word, same seed, twice -> identical G-code (hash) and
      deviation 0 (plain SynthesizeText and SynthesizeJointBestOf nTries=3); different seeds -> the
      between-run deviation (symmetric chamfer, all seed pairs, start-aligned as the G-code origin is
      the same) for 3 words x 3 writers x 8 seeds.
Nominal steps/mm only: the real value is measured by calibrate() (travel steps / 204 or 262 mm) and
depends on the real belt/pulley; that is a HARDWARE quantity (see the protocol in QTP_RESULTS.md).
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import hashlib
import math
import re
import sys
import tempfile

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtp_common as C
import SynthesizeHandwriting as SY
import odroid_direct_drive as DD

SENTENCES = [
    "The gantry writes this sentence for the first time today",
    "My robot copies handwriting from ten different authors",
    "Please bring the blue notebook and a sharp pencil",
    "Every careful measurement makes the next result better",
    "We tested the machine on Monday and it worked quietly",
    "Nothing about this line appears in the training pages",
    "The delivery van left the depot just before seven this morning.",
    "Please charge both batteries overnight and label the spare cable.",
    "Our meeting moved to room 14 on the third floor at noon.",
    "A quick brown fox jumps over the lazy dog while it rains.",
]


def resample(poly, step):
    p = np.asarray(poly, float)
    if len(p) < 2:
        return p
    seg = np.hypot(*np.diff(p, axis=0).T)
    t = np.concatenate([[0], np.cumsum(seg)])
    if t[-1] < 1e-9:
        return p[:1]
    n = max(2, int(math.ceil(t[-1] / step)) + 1)
    s = np.linspace(0, t[-1], n)
    return np.stack([np.interp(s, t, p[:, 0]), np.interp(s, t, p[:, 1])], axis=1)


def dense(strokes, step=0.005):
    pts = [resample(s, step) for s in strokes if len(s) >= 1]
    return np.vstack(pts)


def bresenham_points(x0, y0, x1, y1):
    """Step positions visited by DD.bresenham_move from (x0,y0) to (x1,y1) (integer steps)."""
    dx, dy = x1 - x0, y1 - y0
    ax, ay = abs(dx), abs(dy)
    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1
    pts = []
    x, y = x0, y0
    if ax >= ay:
        err = ax // 2
        for _ in range(ax):
            x += sx
            err -= ay
            if err < 0:
                y += sy
                err += ax
            pts.append((x, y))
    else:
        err = ay // 2
        for _ in range(ay):
            y += sy
            err -= ax
            if err < 0:
                x += sx
                err += ay
            pts.append((x, y))
    return pts


def parse_driver_style(path):
    """Replays the .gcode like DD.run_gcode_file's parser: returns list of pen-down polylines in the
    driver's coordinate frame (mm, after _clamp_to_usable_area), the number of clamped vertices, and
    all (unclamped) G-code vertices."""
    usable_w, usable_h = DD.calibration["usable_width_mm"], DD.calibration["usable_height_mm"]
    strokes, cur, pen_down = [], [], False
    x = y = 0.0
    clamped = 0
    total = 0
    for raw in open(path, encoding="utf-8"):
        parsed = DD.parse_gcode_line(raw)
        if parsed is None:
            continue
        cmd, p = parsed
        if cmd in ("G0", "G1"):
            x = p.get("X", x)
            y = p.get("Y", y)
            cx, cy = min(max(x, 0.0), usable_w), min(max(y, 0.0), usable_h)
            total += 1
            if (cx, cy) != (x, y):
                clamped += 1
            if pen_down:
                cur.append((cx, cy))
        elif cmd == "M3":
            pen_down = True
            cur = [(min(max(x, 0.0), usable_w), min(max(y, 0.0), usable_h))]
        elif cmd == "M5":
            if pen_down and len(cur) >= 2:
                strokes.append(cur)
            cur, pen_down = [], False
    if pen_down and len(cur) >= 2:
        strokes.append(cur)
    return strokes, clamped, total


def planned_reference(traj, cfg):
    """The planned path in bed coordinates (what ToMachine does, WITHOUT decimation/ordering)."""
    x0, y0, x1, y1 = traj.Bounds()
    w, h = max(1e-6, x1 - x0), max(1e-6, y1 - y0)
    availW = cfg.boundsMaxXmm - cfg.originXmm
    availH = cfg.originYmm - cfg.boundsMinYmm
    scale = min(1.0, availW / w, availH / h)
    ref = [[(cfg.originXmm + (x - x0) * scale, cfg.originYmm - (y1 - y) * scale) for (x, y) in s]
           for s in traj.strokes if len(s) >= 2]
    return ref, scale


def stage_dev(a_pts, ref_dense_tree):
    d, _ = ref_dense_tree.query(a_pts)
    return d


def main():
    cfg = SY.GantryConfig()
    profiles = SY.LoadAllProfiles()
    res = {}
    # ------------------------------------------------------------ (i)
    spmm_gen = cfg.stepsPerMmX
    res["resolution"] = dict(
        generator_stepsPerRev=cfg.microstepsPerRev, generator_mmPerRev=cfg.mmPerRevX,
        generator_steps_per_mm_X=cfg.stepsPerMmX, generator_steps_per_mm_Y=cfg.stepsPerMmY,
        driver_MICROSTEPS=DD.MICROSTEPS, driver_STEPS_PER_REV=DD.STEPS_PER_REV,
        driver_STEPS_PER_MM_nominal=DD.STEPS_PER_MM,
        mm_per_step=1.0 / spmm_gen,
        max_quantisation_error_per_axis_mm=0.5 / spmm_gen,
        max_quantisation_error_2d_mm=math.sqrt(2) * 0.5 / spmm_gen,
        rms_quantisation_error_per_axis_mm=(1.0 / spmm_gen) / math.sqrt(12),
        gcode_decimals_mm=0.001, gcode_rounding_max_error_2d_mm=math.sqrt(2) * 0.0005,
        driver_usable_width_mm=DD.calibration["usable_width_mm"],
        driver_usable_height_mm=DD.calibration["usable_height_mm"],
        generator_bounds=[cfg.boundsMaxXmm, cfg.boundsMaxYmm],
        X_SWITCH_TRAVEL_MM=DD.X_SWITCH_TRAVEL_MM, Y_SWITCH_TRAVEL_MM=DD.Y_SWITCH_TRAVEL_MM,
        EDGE_TOLERANCE_MM=DD.EDGE_TOLERANCE_MM,
        requirement_mm=0.5,
        step_resolution_over_requirement=0.5 / (1.0 / spmm_gen),
        calibration_note=("calibrate() re-measures steps/mm = travel_steps / switch-to-switch distance "
                          "(204 mm X, 262 mm Y); the accuracy of that distance (a hardware measurement) "
                          "scales every coordinate: a 0.1 % error in 204 mm = 0.2 mm at the far end"),
        scale_error_for_half_mm_at_194mm_percent=100 * 0.5 / 194.0,
    )
    print("resolution:", res["resolution"]["mm_per_step"], "mm/step")

    # ------------------------------------------------------------ (ii)
    writers = C.ALL_AUTHORS
    rows = []
    allbins = {k: [] for k in ("total_vertex", "total_step")}
    stage_all = {k: [] for k in ("a_decimation", "b_rounding", "c_quantisation", "d_bresenham_steps", "e_total_exec_vs_planned", "f_planned_vertices_vs_exec")}
    tmpdir = tempfile.mkdtemp()
    for i, (w, text) in enumerate(zip(writers, SENTENCES)):
        prof = profiles[w]
        traj = SY.SynthesizeText(text, prof, mmPerXh=4.0, seed=0, lineWidthMm=cfg.boundsMaxXmm - cfg.originXmm - 5,
                                 jitter=0.5)
        gpath = os.path.join(tmpdir, f"s{i}.gcode")
        gres = SY.WriteGcode(traj, cfg, gpath, title=f"qtp5 {w}")
        ref, scale = planned_reference(traj, cfg)
        ref_dense = dense(ref, 0.005)
        tree = cKDTree(ref_dense)
        ref_pts = np.vstack([np.asarray(s) for s in ref])
        # stages ------------------------------------------------------------------
        machine, _scale2 = SY.ToMachine(traj, cfg)
        machine, nclip_gen = SY._Clamp(machine, cfg)
        # (a) decimation + ordering only
        mach_pts = np.vstack([np.asarray(s) for s in machine])
        dev_a = stage_dev(mach_pts, tree)
        # (b) after 3-decimal text rounding (parse our own text)
        strokes_txt = SY.ParseGcode(gpath)
        txt_pts = np.vstack([np.asarray(s) for s in strokes_txt])
        dev_b = stage_dev(txt_pts, tree)
        # (c) driver: clamp to 194x252, then absolute step quantisation at nominal 80 steps/mm
        dstrokes, clamped, total_v = parse_driver_style(gpath)
        spmm = DD.STEPS_PER_MM
        E = DD.EDGE_TOLERANCE_MM
        q_pts = []
        step_pts = []
        for s in dstrokes:
            qs = [(round((x + E) * spmm), round((y + E) * spmm)) for (x, y) in s]
            q_pts += [((a / spmm) - E, (b / spmm) - E) for a, b in qs]
            for (a0, b0), (a1, b1) in zip(qs[:-1], qs[1:]):
                bp = bresenham_points(a0, b0, a1, b1)
                step_pts += [((a / spmm) - E, (b / spmm) - E) for a, b in bp]
            step_pts.append(((qs[0][0] / spmm) - E, (qs[0][1] / spmm) - E))
        q_pts = np.asarray(q_pts)
        step_pts = np.asarray(step_pts)
        dev_c = stage_dev(q_pts, tree)
        dev_d = stage_dev(step_pts, tree)
        # unclamped version of the executed path (to separate quantisation from clamping)
        dev_c_noclamp = []
        for s in strokes_txt:
            for (x, y) in s:
                xq = round((x + E) * spmm) / spmm - E
                yq = round((y + E) * spmm) / spmm - E
                dev_c_noclamp.append(math.hypot(xq - x, yq - y))
        dev_c_noclamp = np.asarray(dev_c_noclamp)
        # reverse direction: planned vertices -> executed step path
        tree_exec = cKDTree(step_pts)
        dev_f = tree_exec.query(ref_pts)[0]
        # quantisation-only (unclamped) vertex deviation vs the text-rounded vertices
        rows.append(dict(
            writer=w, text=text, n_chars=len(text), lines=int(traj.meta["lines"]), scale=scale,
            n_strokes_planned=len(ref), n_strokes_gcode=len(strokes_txt), n_vertices_planned=int(len(ref_pts)),
            n_gcode_vertices=int(total_v), vertices_clamped_by_driver=int(clamped),
            vertices_clamped_by_generator=int(nclip_gen),
            max_x_mm=float(np.max(txt_pts[:, 0])),
            a_decimation_mean=float(dev_a.mean()), a_decimation_p95=float(np.percentile(dev_a, 95)), a_decimation_max=float(dev_a.max()),
            b_rounding_mean=float(dev_b.mean()), b_rounding_p95=float(np.percentile(dev_b, 95)), b_rounding_max=float(dev_b.max()),
            c_quant_vertices_mean=float(dev_c.mean()), c_quant_vertices_p95=float(np.percentile(dev_c, 95)), c_quant_vertices_max=float(dev_c.max()),
            c_quant_noclamp_mean=float(dev_c_noclamp.mean()), c_quant_noclamp_p95=float(np.percentile(dev_c_noclamp, 95)), c_quant_noclamp_max=float(dev_c_noclamp.max()),
            d_exec_steps_mean=float(dev_d.mean()), d_exec_steps_p95=float(np.percentile(dev_d, 95)), d_exec_steps_max=float(dev_d.max()),
            f_planned_to_exec_mean=float(dev_f.mean()), f_planned_to_exec_p95=float(np.percentile(dev_f, 95)), f_planned_to_exec_max=float(dev_f.max()),
            n_exec_steps=int(len(step_pts)), gcode_lines=gres["lines"],
        ))
        stage_all["a_decimation"].append(dev_a); stage_all["b_rounding"].append(dev_b)
        stage_all["c_quantisation"].append(dev_c); stage_all["d_bresenham_steps"].append(dev_d)
        stage_all["f_planned_vertices_vs_exec"].append(dev_f)
        stage_all.setdefault("c_noclamp", []).append(dev_c_noclamp)
        print(f"{w:8s} {text[:40]:40s} scale {scale:.3f} lines {traj.meta['lines']} clamped {clamped}/{total_v} "
              f"exec dev mean {dev_d.mean()*1000:.1f} um p95 {np.percentile(dev_d,95)*1000:.1f} um max {dev_d.max():.3f} mm", flush=True)
    pooled = {k: np.concatenate(v) for k, v in stage_all.items() if v}
    res["path_fidelity_rows"] = rows
    res["path_fidelity_pooled_mm"] = {k: dict(n=int(len(v)), mean=float(v.mean()), median=float(np.median(v)),
                                              p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)),
                                              max=float(v.max())) for k, v in pooled.items()}
    res["path_fidelity_over_sentences"] = {
        "exec_mean_of_means_mm": float(np.mean([r["d_exec_steps_mean"] for r in rows])),
        "exec_max_over_sentences_mm": float(np.max([r["d_exec_steps_max"] for r in rows])),
        "exec_max_without_clamp_effect_note": "see c_quant_noclamp for quantisation-only error",
        "sentences_with_clamped_vertices": int(sum(1 for r in rows if r["vertices_clamped_by_driver"] > 0)),
        "total_vertices_clamped": int(sum(r["vertices_clamped_by_driver"] for r in rows)),
        "total_vertices": int(sum(r["n_gcode_vertices"] for r in rows)),
    }

    # ------------------------------------------------------------ (iii)
    rep = {}
    word = "handwriting"
    text2 = "Please bring the blue notebook"
    prof = profiles["153"]
    def gc_hash(traj, tag):
        p = os.path.join(tmpdir, tag + ".gcode")
        SY.WriteGcode(traj, cfg, p, title="repeat")
        data = open(p, "rb").read()
        return hashlib.sha256(data).hexdigest(), SY.ParseGcode(p)

    t1 = SY.SynthesizeText(word, prof, mmPerXh=4.0, seed=5, lineWidthMm=185, jitter=0.5)
    t2 = SY.SynthesizeText(word, prof, mmPerXh=4.0, seed=5, lineWidthMm=185, jitter=0.5)
    h1, s1 = gc_hash(t1, "r1"); h2, s2 = gc_hash(t2, "r2")
    dmax = max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(s1, s2)) if len(s1) == len(s2) else float("inf")
    rep["same_seed_plain"] = dict(word=word, writer="153", seed=5, gcode_sha256_run1=h1, gcode_sha256_run2=h2,
                                  identical_file=(h1 == h2), n_strokes=len(s1), max_coordinate_difference_mm=dmax)
    from np_inference.text_model import PaperCRNNNumpy
    from np_inference.author_model import AuthorClassifierCNNNumpy
    R = PaperCRNNNumpy(); A = AuthorClassifierCNNNumpy()
    jb = []
    for _ in range(2):
        tj = SY.SynthesizeJointBestOf("153", word, prof, nTries=3, mmPerXh=4.0, lineWidthMm=185,
                                      reader=R, authorModel=A, authorMapping=A.author_mapping, seed=5)
        jb.append(gc_hash(tj, f"j{_}"))
    dmax = max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(jb[0][1], jb[1][1])) if len(jb[0][1]) == len(jb[1][1]) else float("inf")
    rep["same_seed_jointbestof_nTries3"] = dict(word=word, writer="153", seed=5, gcode_sha256_run1=jb[0][0],
                                                gcode_sha256_run2=jb[1][0], identical_file=(jb[0][0] == jb[1][0]),
                                                max_coordinate_difference_mm=dmax)
    # different seeds
    pair_stats = []
    seeds = list(range(8))
    for w in ("150", "153", "dylan"):
        for tx in (word, "robot", text2):
            trajs = []
            for sd in seeds:
                tr = SY.SynthesizeText(tx, profiles[w], mmPerXh=4.0, seed=sd, lineWidthMm=185, jitter=0.5)
                strokes, _sc = SY.ToMachine(tr, cfg)
                trajs.append(strokes)
            for a in range(len(seeds)):
                for b in range(a + 1, len(seeds)):
                    pa, pb = dense(trajs[a], 0.05), dense(trajs[b], 0.05)
                    ta, tb = cKDTree(pa), cKDTree(pb)
                    d_ab = tb.query(pa)[0]; d_ba = ta.query(pb)[0]
                    dd = np.concatenate([d_ab, d_ba])
                    pair_stats.append(dict(writer=w, text=tx, seeds=(seeds[a], seeds[b]), mean=float(dd.mean()),
                                           p95=float(np.percentile(dd, 95)), max=float(dd.max())))
    allm = np.array([p["mean"] for p in pair_stats]); allp = np.array([p["p95"] for p in pair_stats]); allx = np.array([p["max"] for p in pair_stats])
    rep["different_seeds"] = dict(n_pairs=len(pair_stats), words=[word, "robot", text2], writers=["150", "153", "dylan"],
                                  seeds=seeds, mean_dev_mm=dict(mean=float(allm.mean()), min=float(allm.min()), max=float(allm.max())),
                                  p95_dev_mm=dict(mean=float(allp.mean()), min=float(allp.min()), max=float(allp.max())),
                                  max_dev_mm=dict(mean=float(allx.mean()), min=float(allx.min()), max=float(allx.max())),
                                  pairs=pair_stats)
    res["repeatability_software"] = rep
    res["meta"] = dict(machine=C.machine_info(), settings=dict(mmPerXh=4.0, lineWidthMm=185, jitter=0.5, seed=0,
                                                                 resample_step_mm=0.005))
    C.save_json(res, "qtp5_results.json")

    # ------------------------------------------------------------ figures
    plt = C.mpl_style()
    fig, axs = plt.subplots(1, 2, figsize=(16 * C.CM, 6.5 * C.CM))
    ax = axs[0]
    v = pooled["d_bresenham_steps"] * 1000
    ax.hist(v, bins=60, color=C.PALETTE[0], alpha=0.85)
    ax.set_yscale("log")
    ax.axvline(500, color="red", ls="--", lw=1); ax.text(500, ax.get_ylim()[1] * 0.5, " R5 limit\n 500 um", color="red", fontsize=8, va="top")
    ax.set_xlabel("deviation executed step path vs planned path [um]")
    ax.set_ylabel("number of step positions")
    ax.set_title(f"10 sentences, {len(v)} steps\nmean {v.mean():.1f} um, p95 {np.percentile(v,95):.1f} um, max {v.max():.0f} um", fontsize=9)
    ax = axs[1]
    labels = ["decimation", "3-dec.\nrounding", "step quant.\n(vertices)", "executed\nsteps"]
    data = [pooled["a_decimation"] * 1000, pooled["b_rounding"] * 1000, pooled["c_quantisation"] * 1000, pooled["d_bresenham_steps"] * 1000]
    ax.boxplot(data, tick_labels=labels, showfliers=False, whis=(5, 95))
    ax.set_yscale("log"); ax.set_ylabel("deviation [um]")
    ax.set_title("per-stage deviation (whiskers 5-95 %)", fontsize=9)
    fig.tight_layout()
    C.savefig(fig, "qtp5_gcode_deviation_hist.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8 * C.CM, 6 * C.CM))
    ax.hist(allm, bins=30, color=C.PALETTE[1], alpha=0.85)
    ax.set_xlabel("mean between-run path deviation [mm]"); ax.set_ylabel("seed pairs")
    ax.set_title("different seeds, same text/writer", fontsize=9)
    fig.tight_layout()
    C.savefig(fig, "qtp5_seed_variability.png"); plt.close(fig)

    print("\npooled executed-vs-planned (mm):", res["path_fidelity_pooled_mm"]["d_bresenham_steps"])
    print("clamped vertices:", res["path_fidelity_over_sentences"])
    print("same seed plain identical:", rep["same_seed_plain"]["identical_file"], "| joint:", rep["same_seed_jointbestof_nTries3"]["identical_file"])
    print("different seeds mean dev (mm):", rep["different_seeds"]["mean_dev_mm"])


if __name__ == "__main__":
    main()
