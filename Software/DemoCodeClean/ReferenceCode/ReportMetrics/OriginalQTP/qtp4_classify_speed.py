"""
qtp4_classify_speed.py -- QTP4 / requirement R4: classification speed
(<= 25 ms per character plus 2 s for image capture/conditioning, i.e. total <= 2 + 0.025 x s).

For each of the 13 real photographed pages (3 repeats each, serial, idle machine, DEV LAPTOP, NOT
THE ODROID N2+) time, replicating server.py /api/classify:
   stage 0  server-side image handling : Image.open + convert('RGB') + np.array(convert('L')) +
                                          PNG write of the grey page
   stage 1  SegmentPage.ProcessPage(png) : illumination correction, binarisation, deskew, rule-line
                                          removal, line segmentation (the 'conditioning' budget)
   stage 2  text recognition of every TEXT line (np_inference ReadText, joint checkpoint)
   stage 3  writer classification of every line (np_inference ClassifyImage, ink-based model,
                                          apply_stroke_normalize=False) + probability averaging
x = number of characters on the page = sum of len(stripped label line), MESS lines excluded
(from *_labels.txt).  Requirement for the page = 2 + 0.025 x seconds; ratio = total / requirement.
(Image capture itself is not included: no camera on the laptop.)
Also: cProfile of ProcessPage on one representative photo (cumulative times).
Default numpy/BLAS thread settings.
"""
import os
import cProfile
import io
import pstats
import sys
import tempfile
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtp_common as C
import SegmentPage as PS
from np_inference.text_model import PaperCRNNNumpy, ReadText
from np_inference.author_model import AuthorClassifierCNNNumpy, ClassifyImage
from authors_config import author_classifier_weights_filename

REPEATS = 3
PROFILE_PAGE = "preprocessing_page_localization_illumination_binarisation.jpg"


def one_pass(path, text_model, ink_model, tmpdir):
    t0 = time.perf_counter()
    pil = Image.open(path).convert("RGB")
    gray = np.array(pil.convert("L"))
    tmp = os.path.join(tmpdir, "_classify_tmp.png")
    Image.fromarray(gray).save(tmp)
    t1 = time.perf_counter()
    results, _prev, _meta = PS.ProcessPage(tmp)
    t2 = time.perf_counter()
    crops = [r["raw_crop"] for r in results if r.get("tag") == "TEXT"]
    pils = [Image.fromarray(c).convert("L") for c in crops]
    rec_times, cls_times, cls_sum = [], [], None
    for pl in pils:
        a = time.perf_counter()
        ReadText(pl, text_model)
        rec_times.append(time.perf_counter() - a)
    for pl in pils:
        a = time.perf_counter()
        _i, probs = ClassifyImage(pl, ink_model, apply_stroke_normalize=False)
        cls_sum = probs if cls_sum is None else cls_sum + probs
        cls_times.append(time.perf_counter() - a)
    t3 = time.perf_counter()
    return dict(stage0_io_s=t1 - t0, stage1_processpage_s=t2 - t1, stage2_recognition_s=float(sum(rec_times)),
                stage3_classification_s=float(sum(cls_times)), n_lines=len(crops),
                rec_per_line_s=rec_times, cls_per_line_s=cls_times,
                total_s=(t1 - t0) + (t2 - t1) + sum(rec_times) + sum(cls_times))


def main():
    text_model = PaperCRNNNumpy()
    ink_model = AuthorClassifierCNNNumpy(checkpoint_path=C.WEIGHTS / author_classifier_weights_filename())
    pages = C.personal_pages()
    tmpdir = tempfile.mkdtemp()
    out_rows = []
    # warm-up of the two networks (first call pays lazy allocation); not recorded
    dummy = Image.fromarray(np.full((60, 600), 255, np.uint8))
    ReadText(dummy, text_model); ClassifyImage(dummy, ink_model, apply_stroke_normalize=False)
    t_all = time.time()
    for author, p in pages:
        labels = [g.strip() for g in C.read_labels(p) if g.strip() != "MESS"]
        x = sum(len(g) for g in labels)
        reps = []
        for r in range(REPEATS):
            res = one_pass(p, text_model, ink_model, tmpdir)
            res["repeat"] = r
            reps.append(res)
            print(f"{author}/{p.name[:40]:40s} rep{r}: ProcessPage {res['stage1_processpage_s']:.1f}s rec {res['stage2_recognition_s']:.1f}s "
                  f"cls {res['stage3_classification_s']:.1f}s total {res['total_s']:.1f}s | x={x} req {2+0.025*x:.1f}s "
                  f"| {(time.time()-t_all)/60:.1f} min", flush=True)
        pil = Image.open(p)
        row = dict(writer=author, page=p.name, x_chars=x, n_label_lines=len(labels), image_size=list(pil.size),
                   requirement_s=2 + 0.025 * x, repeats=reps)
        for k in ("stage0_io_s", "stage1_processpage_s", "stage2_recognition_s", "stage3_classification_s", "total_s"):
            v = [rr[k] for rr in reps]
            row[k] = dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)))
        row["n_lines"] = reps[0]["n_lines"]
        row["ratio_total_over_requirement"] = row["total_s"]["mean"] / row["requirement_s"]
        row["rec_ms_per_char"] = 1000 * row["stage2_recognition_s"]["mean"] / x
        row["cls_ms_per_char"] = 1000 * row["stage3_classification_s"]["mean"] / x
        row["rec_ms_per_line"] = 1000 * row["stage2_recognition_s"]["mean"] / row["n_lines"]
        row["cls_ms_per_line"] = 1000 * row["stage3_classification_s"]["mean"] / row["n_lines"]
        row["rec_plus_cls_ms_per_char"] = row["rec_ms_per_char"] + row["cls_ms_per_char"]
        row["processpage_ms_per_char"] = 1000 * row["stage1_processpage_s"]["mean"] / x
        out_rows.append(row)
        C.save_json(dict(rows=out_rows, machine=C.machine_info()), "qtp4_results_partial.json")

    # ---------------- cProfile ----------------
    prof_path = next(p for a, p in pages if p.name == PROFILE_PAGE)
    pil = Image.open(prof_path).convert("RGB")
    tmp = os.path.join(tmpdir, "_prof.png")
    Image.fromarray(np.array(pil.convert("L"))).save(tmp)
    pr = cProfile.Profile()
    t = time.perf_counter()
    pr.enable()
    PS.ProcessPage(tmp)
    pr.disable()
    prof_wall = time.perf_counter() - t
    st = pstats.Stats(pr)
    entries = []
    for func, (cc, nc, tt, ct, callers) in st.stats.items():
        entries.append(dict(file=os.path.basename(func[0]), line=func[1], name=func[2], ncalls=nc,
                            tottime_s=tt, cumtime_s=ct))
    entries.sort(key=lambda e: -e["cumtime_s"])
    pp_key = next(f for f in st.stats if f[2] == "ProcessPage")
    pp_total = st.stats[pp_key][3]
    direct = []
    for func, (cc, nc, tt, ct, callers) in st.stats.items():
        if pp_key in callers:
            c = callers[pp_key]
            direct.append(dict(name=func[2], file=os.path.basename(func[0]), line=func[1], calls=c[0], cumtime_s=c[3]))
    # aggregate direct callees by name (same helper may be called several times)
    agg = {}
    for d in direct:
        a = agg.setdefault(d["name"], dict(name=d["name"], calls=0, cumtime_s=0.0))
        a["calls"] += d["calls"]; a["cumtime_s"] += d["cumtime_s"]
    direct = sorted(agg.values(), key=lambda d: -d["cumtime_s"])
    top_cum = [e for e in entries if e["name"] not in ("<module>", "main") and "exec" not in e["name"]][:12]
    top_tot = sorted(entries, key=lambda e: -e["tottime_s"])[:10]
    prof_out = dict(page=PROFILE_PAGE, wall_s_with_profiler=prof_wall, processpage_cumtime_s=pp_total,
                    top_cumulative=top_cum, top_tottime=top_tot, direct_callees_of_ProcessPage=direct[:20],
                    note="cProfile adds overhead (pure-Python call counting), so absolute times exceed the unprofiled ProcessPage time; use the proportions")
    out = dict(rows=out_rows, profile=prof_out, machine=C.machine_info(), repeats=REPEATS, runtime_min=(time.time() - t_all) / 60)
    C.save_json(out, "qtp4_results.json")

    # ---------------- figures ----------------
    plt = C.mpl_style()
    fig, ax = plt.subplots(figsize=(16 * C.CM, 8 * C.CM))
    xs = np.arange(len(out_rows))
    bottoms = np.zeros(len(out_rows))
    for key, lab, col in (("stage0_io_s", "image decode + PNG write", C.PALETTE[7]),
                          ("stage1_processpage_s", "ProcessPage (conditioning + segmentation)", C.PALETTE[0]),
                          ("stage2_recognition_s", "text recognition (all lines)", C.PALETTE[2]),
                          ("stage3_classification_s", "writer classification (all lines)", C.PALETTE[3])):
        v = np.array([r[key]["mean"] for r in out_rows])
        ax.bar(xs, v, bottom=bottoms, label=lab, color=col)
        bottoms += v
    err_lo = np.array([r["total_s"]["mean"] - r["total_s"]["min"] for r in out_rows])
    err_hi = np.array([r["total_s"]["max"] - r["total_s"]["mean"] for r in out_rows])
    ax.errorbar(xs, bottoms, yerr=[err_lo, err_hi], fmt="none", ecolor="black", capsize=2, lw=0.8)
    ax.scatter(xs, [r["requirement_s"] for r in out_rows], marker="_", s=300, color="red", zorder=5, label="R4 budget 2 + 0.025 x")
    ax.set_xticks(xs); ax.set_xticklabels([f"{r['writer'][:1].upper()}{i+1}\n(x={r['x_chars']})" for i, r in enumerate(out_rows)], fontsize=7)
    ax.set_ylabel("time per page [s]  (dev laptop, 3 repeats)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("QTP4: classification time per photographed page vs the R4 budget", fontsize=9)
    fig.tight_layout()
    C.savefig(fig, "qtp4_stage_timing.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(16 * C.CM, 7.5 * C.CM))
    top = direct[:10][::-1]
    ax.barh([d["name"] for d in top], [100 * d["cumtime_s"] / pp_total for d in top], color=C.PALETTE[0])
    ax.set_xlabel("share of ProcessPage cumulative time [%] (cProfile)")
    ax.set_title(f"QTP4: ProcessPage stages (direct callees), page '{PROFILE_PAGE[:30]}...'", fontsize=9)
    fig.tight_layout()
    C.savefig(fig, "qtp4_processpage_profile.png"); plt.close(fig)
    print("done", out["runtime_min"], "min")


if __name__ == "__main__":
    main()
