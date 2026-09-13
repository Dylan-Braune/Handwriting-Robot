"""
compare_htr_weights.py -- run two text-recogniser checkpoints over the same
pages and report CER for each, split by domain:

  * DATASET   -- IAM Sentence-Database pages (segmented by ExtractLinePatches)
  * PERSONAL  -- your own photographed pages (segmented by SegmentPage)

Both live in NOGIT/holdout_test_pages/ next to their *_labels.txt. The IAM
pages are the numbered files (000_a05-125.png ...); the personal pages are
the named ones (baseline_model.png, gantry_planning.png, ...).

Usage:
    python Diagnostics/compare_htr_weights.py \
        --a NOGIT/weights/paper_cnn_bilstm_ctc_best.pt \
        --b C:/Users/braun/Downloads/paper_cnn_bilstm_ctc_hf_best.pt
    (run from Software/CNN/)
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
CNN_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CNN_DIR))

from ClassifyText import load_model, transcribe_page          # noqa: E402
from ExtractIAMLines import ReadLabelLines                    # noqa: E402
from TrainText import levenshtein                             # noqa: E402

HOLDOUT = CNN_DIR / "NOGIT" / "holdout_test_pages"
PERSONAL_STEMS = {"baseline_model", "gantry_planning", "repeatability_notes",
                  "ducm_cov_notes", "erp_prac2", "exam_page1"}


def list_pages():
    dataset, personal = [], []
    for p in sorted(HOLDOUT.iterdir()):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        if not p.with_name(p.stem + "_labels.txt").exists():
            continue
        (personal if p.stem in PERSONAL_STEMS else dataset).append(p)
    return dataset, personal


def _norm(s):
    return " ".join(s.split())


def score_page(model, img_path, device, is_dataset):
    preds = transcribe_page(model, img_path, device, is_dataset=is_dataset)
    gt = ReadLabelLines(str(img_path))
    if not is_dataset:
        gt = [g for g in gt if g.strip() != "MESS"]
    # (1) per-line by index -- punished hard when segmentation wraps lines
    #     at different points than the label file does
    li_c = li_e = 0
    for i, truth in enumerate(gt):
        pred = preds[i] if i < len(preds) else ""
        li_c += len(truth)
        li_e += levenshtein(pred, truth)
    # (2) page level -- join everything, ignore line breaks. isolates
    #     recognition from line-segmentation alignment.
    pj, gj = _norm(" ".join(preds)), _norm(" ".join(gt))
    pg_e = levenshtein(pj, gj)
    pg_c = len(gj)
    return (li_c, li_e), (pg_c, pg_e), len(preds), len(gt)


def run_set(model, pages, device, is_dataset, label):
    li_C = li_E = pg_C = pg_E = 0
    print(f"\n  {label}  ({len(pages)} pages)      per-line / page-level CER")
    for p in pages:
        (lc, le), (pc, pe), npred, ngt = score_page(model, p, device, is_dataset)
        li_C += lc; li_E += le; pg_C += pc; pg_E += pe
        flag = "" if npred == ngt else f"  [seg {npred}!={ngt}]"
        print(f"    {p.name:28s} {le/max(1,lc):6.1%} / {pe/max(1,pc):6.1%}{flag}")
    print(f"    {'-- overall':28s} {li_E/max(1,li_C):6.1%} / {pg_E/max(1,pg_C):6.1%}"
          f"   (page-level char-acc {1 - pg_E/max(1,pg_C):.1%})")
    return (li_C, li_E), (pg_C, pg_E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(CNN_DIR / "NOGIT" / "weights" /
                                      "paper_cnn_bilstm_ctc_best.pt"))
    ap.add_argument("--b", required=True)
    ap.add_argument("--max-dataset", type=int, default=0,
                    help="cap number of IAM pages (0 = all)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, personal = list_pages()
    if args.max_dataset:
        dataset = dataset[:args.max_dataset]

    summary = {}
    for tag, path in (("A", args.a), ("B", args.b)):
        path = Path(path)
        print("\n" + "=" * 74)
        print(f"[{tag}] {path.name}   ({path})")
        print("=" * 74)
        model = load_model(path, device)
        d_li, d_pg = run_set(model, dataset, device, True, "DATASET (IAM)")
        p_li, p_pg = run_set(model, personal, device, False, "PERSONAL (your pages)")
        summary[tag] = dict(
            name=path.name,
            d_li=d_li[1] / max(1, d_li[0]), d_pg=d_pg[1] / max(1, d_pg[0]),
            p_li=p_li[1] / max(1, p_li[0]), p_pg=p_pg[1] / max(1, p_pg[0]))

    print("\n" + "=" * 74)
    print("SUMMARY  (CER, lower is better)   per-line-by-index / page-level")
    print("=" * 74)
    for tag in ("A", "B"):
        s = summary[tag]
        print(f"  [{tag}] {s['name']:26s}  IAM {s['d_li']:.1%}/{s['d_pg']:.1%}"
              f"   PERSONAL {s['p_li']:.1%}/{s['p_pg']:.1%}")
    dg = summary["A"]["d_pg"] - summary["B"]["d_pg"]
    pgd = summary["A"]["p_pg"] - summary["B"]["p_pg"]
    print(f"\n  B vs A (page-level): IAM {'+' if dg>0 else ''}{dg*100:.1f} pts, "
          f"personal {'+' if pgd>0 else ''}{pgd*100:.1f} pts  (positive = B better)")
    print("\n  Note: per-line-by-index CER is inflated -- ExtractLinePatches /"
          " SegmentPage wrap lines at\n  different points than the label files,"
          " so predicted line N is scored against a\n  misaligned reference."
          " Page-level (join all lines) is the honest recognition number.")


if __name__ == "__main__":
    main()
