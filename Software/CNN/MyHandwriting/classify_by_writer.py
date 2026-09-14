"""Aggregate the per-page classification stats by IAM writer ID, and measure
how joined (cursive) each writer's real hand actually is, using the same
independent ink-run-per-character measure used earlier in this project for
the 10 AuthorReproductionStuff writers -- so "cursive" here means the same
thing it meant there, not a new definition.
"""
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np

import _env
from _env import OUT_DIR, NOGIT_DIR, REPO
from ExtractIAMLines import ExtractLinePatches
import BuildStyleProfile as SP

HOLDOUT_DIR = NOGIT_DIR / "holdout_test_pages"
FORMS_TXT = REPO / "Data" / "Datasets" / "IAMlines50" / "forms_for_parsing.txt"
STATS_CSV = OUT_DIR / "classify_perpage_stats.csv"


def load_forms():
    forms = {}
    for line in open(FORMS_TXT, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        forms[parts[0]] = parts[1]
    return forms


def ink_runs_per_char(page_path, label_path):
    text = "".join(l.strip() for l in open(label_path, encoding="utf-8") if l.strip())
    n = max(1, len(text.replace(" ", "")))
    try:
        line_samples, _, _ = ExtractLinePatches(
            str(page_path), targetHeight=32, maxWidth=1024,
            expectedLineCount=None, labelLines=None, is_dataset=True)
    except Exception:
        return None
    total_runs = 0
    for s in line_samples:
        ink = SP.BinarizeLine(s["raw_crop"])
        if not ink.any():
            continue
        col = ink.any(axis=0)
        prev = False
        for v in col:
            if v and not prev:
                total_runs += 1
            prev = v
    return total_runs / n


def main():
    forms = load_forms()
    rows = list(csv.DictReader(open(STATS_CSV, encoding="utf-8")))

    per_writer = defaultdict(list)   # writer -> list of (new_acc, old_acc, page)
    skipped = []
    for r in rows:
        stem = Path(r["page"]).stem
        form_id = stem.split("_", 1)[1] if "_" in stem else stem
        writer = forms.get(form_id)
        if writer is None:
            skipped.append(r["page"])
            continue
        per_writer[writer].append((float(r["new_char_acc_pct"]),
                                   float(r["old_char_acc_pct"]), r["page"], form_id))

    print(f"skipped (no writer mapping / not real IAM pages): {skipped}\n")

    # cursive-ness per writer: ink runs/char measured on each of their pages
    print("measuring ink-runs-per-char (cursive-ness) per writer's page(s)...")
    cursive_by_writer = {}
    for writer, items in per_writer.items():
        vals = []
        for _new, _old, page, _form in items:
            p = HOLDOUT_DIR / page
            lbl = p.with_name(p.stem + "_labels.txt")
            if not lbl.exists():
                continue
            v = ink_runs_per_char(p, lbl)
            if v is not None:
                vals.append(v)
        if vals:
            cursive_by_writer[writer] = float(np.median(vals))

    writer_summary = []
    for writer, items in per_writer.items():
        news = [x[0] for x in items]
        olds = [x[1] for x in items]
        writer_summary.append(dict(
            writer=writer, n_pages=len(items),
            new_acc=float(np.mean(news)), old_acc=float(np.mean(olds)),
            ink_runs_per_char=cursive_by_writer.get(writer),
        ))

    out_csv = OUT_DIR / "classify_by_writer.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["writer_id", "n_pages", "new_char_acc_pct", "old_char_acc_pct",
                   "ink_runs_per_char"])
        for row in sorted(writer_summary, key=lambda r: -r["new_acc"]):
            w.writerow([row["writer"], row["n_pages"], round(row["new_acc"], 1),
                       round(row["old_acc"], 1),
                       round(row["ink_runs_per_char"], 3) if row["ink_runs_per_char"] else ""])
    print(f"-> {out_csv}  ({len(writer_summary)} distinct writers)\n")

    ranked = sorted(writer_summary, key=lambda r: -r["new_acc"])
    print("=== TOP 10 MOST ACCURATE WRITERS (new model) ===")
    for r in ranked[:10]:
        cv = f"{r['ink_runs_per_char']:.3f}" if r["ink_runs_per_char"] else "n/a"
        print(f"  writer {r['writer']:>4}  new={r['new_acc']:5.1f}%  "
              f"old={r['old_acc']:5.1f}%  n_pages={r['n_pages']}  ink_runs/char={cv}")

    have_cv = [r for r in writer_summary if r["ink_runs_per_char"] is not None]
    cv_vals = sorted(r["ink_runs_per_char"] for r in have_cv)
    median_cv = cv_vals[len(cv_vals)//2] if cv_vals else 0.5
    cursive = [r for r in have_cv if r["ink_runs_per_char"] <= median_cv]
    cursive_ranked = sorted(cursive, key=lambda r: -r["new_acc"])
    print(f"\n=== TOP 10 MOST ACCURATE *CURSIVE* WRITERS ===")
    print(f"(cursive = ink_runs/char at or below this sample's median, {median_cv:.3f} -- "
          f"lower = more joined; for reference the AuthorReproductionStuff 10-author set's "
          f"clearly-cursive writers measured 0.25-0.29, its clearly-print writers 0.52-0.75)")
    for r in cursive_ranked[:10]:
        print(f"  writer {r['writer']:>4}  new={r['new_acc']:5.1f}%  "
              f"old={r['old_acc']:5.1f}%  n_pages={r['n_pages']}  "
              f"ink_runs/char={r['ink_runs_per_char']:.3f}")

    n1 = sum(1 for r in writer_summary if r["n_pages"] == 1)
    print(f"\n({n1} of {len(writer_summary)} writers have only 1 page in this sample -- "
          f"their accuracy numbers are single-page, not averaged)")


if __name__ == "__main__":
    main()
