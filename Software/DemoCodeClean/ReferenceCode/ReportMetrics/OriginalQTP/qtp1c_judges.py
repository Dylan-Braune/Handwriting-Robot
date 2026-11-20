"""
qtp1c_judges.py -- QTP1(c): character accuracy with a recogniser that was NOT used inside the
best-of-N search.

  production judge  = weights/paper_cnn_bilstm_ctc_joint_best.pt  (inside SynthesizeJointBestOf)
  held-out judge    = weights/paper_cnn_bilstm_ctc_hf_best.pt     (general base model, trained
                      on Teklia/IAM lines only; never used by the search)
Both through np_inference.text_model.PaperCRNNNumpy (as the server does).

This is only a PARTIAL-independent judge: same architecture, same data lineage (the joint model
was initialised from / trained after the hf model), same CTC decoder, and the synthesiser's
search selected candidates that look legible to a closely related network.  It removes the
winner's-curse of selecting with the very checkpoint that scores, not the shared biases.

Sets read
  A  : 60 synthesised lines (10 writers x 6 novel sentences), reference = the sentence
  B  : matched synthesis: for the 30 real held-out line texts, synth(A, T) in the writer A's own
       profile (only the matched writer), reference = T; plus the SAME recognisers on the REAL
       line with text T (ceiling on identical text)
  R  : all real held-out lines (held out of the style profile: IAM held-out page + personal lines
       outside the profile split): ceiling reference per writer
Metrics: CharAcc (1 - Levenshtein/len, case sensitive, = server.py's CharAcc), micro char
accuracy (1 - sum edits / sum ref chars), WordAcc (LCS over words, case-insensitive, = project's),
and case-insensitive char accuracy (Evaluate legibility convention).
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import io
import pickle
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtp_common as C
import qtp_lines as L
from np_inference.text_model import PaperCRNNNumpy, ReadText, CharAcc, WordAcc, levenshtein


def stats_for(items):
    """items: list of (pred, truth)."""
    if not items:
        return None
    edits = sum(levenshtein(p, t) for p, t in items)
    ref = sum(len(t) for _, t in items)
    edits_ci = sum(levenshtein(p.lower(), t.lower()) for p, t in items)
    return dict(n=len(items), char_acc_mean=float(np.mean([CharAcc(p, t) for p, t in items])),
                char_acc_micro=1 - edits / ref,
                char_acc_ci_mean=float(np.mean([CharAcc(p.lower().strip(), t.lower().strip()) for p, t in items])),
                char_acc_ci_micro=1 - edits_ci / ref,
                word_acc_mean=float(np.mean([WordAcc(p, t) for p, t in items])))


def main():
    t0 = time.time()
    models = {"production_joint": PaperCRNNNumpy(checkpoint_path=C.JOINT_CKPT),
              "heldout_hf": PaperCRNNNumpy(checkpoint_path=C.HF_CKPT)}
    syn_dir = C.DATA_OUT / "synth"
    jobs = C.load_json("qtp1_synth_jobs.json")
    W = C.ALL_AUTHORS

    def load(name):
        with open(syn_dir / (name + ".pkl"), "rb") as f:
            r = pickle.load(f)
        return r, Image.open(io.BytesIO(r["png"])).convert("L")

    reads = {m: dict(A=[], B=[], R=[], Rmatch=[]) for m in models}
    # A
    for si, s in enumerate(jobs["sentences"]):
        for w in W:
            p = syn_dir / f"A_{w}_s{si}.pkl"
            if not p.exists():
                continue
            r, img = load(f"A_{w}_s{si}")
            for m, mod in models.items():
                reads[m]["A"].append(dict(writer=w, truth=s, pred=ReadText(img, mod), score=r["meta"].get("jointScore")))
    print("A read", time.time() - t0, flush=True)
    # B matched + real line same text
    lines = L.load_real_lines()
    by_key = {(l["writer"], l["page"], l["line_idx"]): l for l in lines}
    for t in jobs["overlay_texts"]:
        w = t["real_writer"]
        p = syn_dir / f"B_t{t['tid']:02d}_{w}.pkl"
        if not p.exists():
            continue
        r, img = load(f"B_t{t['tid']:02d}_{w}")
        real = by_key[(w, t["page"], t["line_idx"])]
        rimg = Image.fromarray(real["gray"]).convert("L")
        for m, mod in models.items():
            reads[m]["B"].append(dict(writer=w, truth=t["text"], pred=ReadText(img, mod)))
            reads[m]["Rmatch"].append(dict(writer=w, truth=t["text"].strip(), pred=ReadText(rimg, mod)))
    print("B read", time.time() - t0, flush=True)
    # R: all held-out (profile) real lines, charset-clean truth (a recogniser cannot output other chars)
    from np_inference.text_model import CHAR_TO_IDX
    held = [l for l in lines if l["heldout_profile"] and 0 < len(l["text"].strip()) <= 120]
    for l in held:
        img = Image.fromarray(l["gray"]).convert("L")
        for m, mod in models.items():
            reads[m]["R"].append(dict(writer=l["writer"], truth=l["text"].strip(), pred=ReadText(img, mod),
                                      truth_f="".join(c for c in l["text"].strip() if c in CHAR_TO_IDX)))
    print("R read", len(held), time.time() - t0, flush=True)

    out = dict(meta=dict(machine=C.machine_info(), runtime_s=time.time() - t0,
                         production_ckpt=C.JOINT_CKPT.name, heldout_ckpt=C.HF_CKPT.name,
                         nTries_in_synthesis=jobs["nTries"],
                         note="hf checkpoint is a PARTIAL-independent judge (same architecture and data lineage)"))
    for m in models:
        out[m] = {}
        for k in ("A", "B", "Rmatch", "R"):
            rows = reads[m][k]
            items = [(r["pred"], r["truth"].strip()) for r in rows]
            blk = dict(overall=stats_for(items), per_writer={})
            for w in W:
                it = [(r["pred"], r["truth"].strip()) for r in rows if r["writer"] == w]
                if it:
                    blk["per_writer"][w] = stats_for(it)
            # mean over writers (each writer weighted equally; = how Evaluate.py aggregates)
            pw = [v["char_acc_mean"] for v in blk["per_writer"].values()]
            blk["char_acc_mean_over_writers"] = float(np.mean(pw)) if pw else None
            pw = [v["word_acc_mean"] for v in blk["per_writer"].values()]
            blk["word_acc_mean_over_writers"] = float(np.mean(pw)) if pw else None
            pw = [v["char_acc_ci_mean"] for v in blk["per_writer"].values()]
            blk["char_acc_ci_mean_over_writers"] = float(np.mean(pw)) if pw else None
            out[m][k] = blk
    for m in models:   # real lines scored against the charset-filtered truth (characters the net cannot output removed)
        it = [(r["pred"], r["truth_f"]) for r in reads[m]["R"] if r["truth_f"]]
        out[m]["R_charset_filtered_overall"] = stats_for(it)
    out["raw_reads"] = reads
    C.save_json(out, "qtp1c_results.json")
    for m in models:
        for k in ("A", "B", "Rmatch", "R"):
            b = out[m][k]
            print(f"{m:17s} {k:6s} n={b['overall']['n']:3d} char(mean over writers) {b['char_acc_mean_over_writers']*100:5.1f}  "
                  f"micro {b['overall']['char_acc_micro']*100:5.1f}  word {b['word_acc_mean_over_writers']*100:5.1f}")

    # ---------------- figure ----------------
    plt = C.mpl_style()
    fig, ax = plt.subplots(figsize=(16 * C.CM, 7.5 * C.CM))
    xs = np.arange(len(W))
    wd = 0.2
    series = [("synth, production judge (joint)", out["production_joint"]["A"], C.PALETTE[1]),
              ("synth, held-out judge (hf)", out["heldout_hf"]["A"], C.PALETTE[0]),
              ("REAL held-out lines, hf judge", out["heldout_hf"]["R"], C.PALETTE[2]),
              ("REAL held-out lines, joint judge", out["production_joint"]["R"], C.PALETTE[7])]
    for k, (lab, blk, col) in enumerate(series):
        v = [blk["per_writer"].get(w, {}).get("char_acc_mean", np.nan) * 100 for w in W]
        ax.bar(xs + (k - 1.5) * wd, v, wd, label=lab, color=col)
    ax.axhline(85, color="black", ls="--", lw=1)
    ax.text(len(W) - 0.5, 86, "R1: 85 %", ha="right", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(W)
    ax.set_xlabel("writer"); ax.set_ylabel("character accuracy [%]")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    C.savefig(fig, "qtp1_char_acc_judges.png"); plt.close(fig)


if __name__ == "__main__":
    main()
