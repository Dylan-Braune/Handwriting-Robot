"""
Regenerates _labels.txt for every page across the whole IAMpages671 dataset
using the new word-count-based line splitter (CountWordsFromInkColumns /
SplitWordsIntoLines in FullLineBoxMaker.py), REPLACING the old width-based-
guess labels in place, in your actual dataset folder.

WHAT THIS DOES
--------------
  - For every page NOT in the hand-verified set (author folder "150" --
    copied in from IAMpages10/150, manually verified, never touched by
    this script), deletes its existing _labels.txt (if any) and writes a
    fresh one using the same pipeline generate_all_labels.py uses for a
    page it's never labeled before: OCR the printed header, then split
    those words across detected lines with the new word-count-based logic.
  - Applies two quality gates before writing anything, either of which
    leaves a page with NO label file (its old one, if any, is still
    deleted, since it was built on the same guess-based logic this
    replaces and isn't more trustworthy than having no label):
      1. PageBoundsAreUncertain() (FullLineBoxMaker.py) -- True when no
         reliable rule line was found separating the printed header from
         the handwriting, meaning the extracted "handwriting" region risks
         actually starting inside the printed text. This is the "stealing
         from the typed header" failure mode -- a page can pass a pure
         line-COUNT check while still having this problem, since the count
         can come out right by coincidence even when the top "line" is
         really printed text. Caught here, before any label gets written.
      2. The same OCR-confidence gate generate_all_labels.py uses (header
         OCR confidence >= 85, at least 15 words).
  - Every N pages that actually get a fresh label written (default 40),
    saves a verification bundle to a review folder: the annotated preview
    image (line boxes, numbered to match the label list) plus a .txt with
    every line's freshly-generated text, numbered the same way, and the
    OLD label (if there was one) underneath for comparison.

WHAT THIS DOES NOT DO -- READ THIS FIRST
-----------------------------------------
There's no way to *prove* every line matches perfectly without either (a)
the official IAM per-line ground truth (lines.xml/lines.txt from the real
IAM database -- not present in this repo, would need downloading separately
with FKI Bern registration), or (b) a human reading every single line,
which defeats the point of automatic labeling. What this script gives you
instead, as the best available automatic proxies:
  1. A live count of how many pages end up with a DIFFERENT number of
     detected lines than their old label had -- printed in the summary.
     A big change there is worth a look either way.
  2. The periodic verification bundles, for actually reading a sample
     yourself.
None of this replaces occasionally opening the review folder and reading a
handful of pages.

AFTER RUNNING THIS: line_cache_raw/ (used by train_paper_cnn_bilstm_ctc.py)
was built from the OLD labels and will be stale. Delete it or set
force_rebuild=True before your next training run, or you'll silently keep
training on the old, replaced labels.

Usage:
    python regenerate_all_labels.py --dry-run       # report only, touches nothing
    python regenerate_all_labels.py                 # do it for real
    python regenerate_all_labels.py --review-every 20
"""

import argparse
from pathlib import Path

from FullLineBoxMaker import ExtractLinePatches, ReadLabelLines, PageBoundsAreUncertain
from generate_all_labels import HeaderOcrConfidence, MIN_AVG_CONF, MIN_WORD_COUNT

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[1] / "Data" / "Datasets" / "IAMpages671"
DEFAULT_REVIEW_DIR = SCRIPT_DIR / "label_regeneration_review"

# Author folder "150" holds the hand-verified labels copied in from
# IAMpages10/150 (see generate_all_labels.py's CopyVerifiedFolder150Labels).
# Those were actually read and checked by a person -- never regenerate them.
PROTECTED_AUTHORS = {"150"}


def resolve_dataset_root(root_dir):
    data_dir = root_dir / "data"
    return data_dir if data_dir.exists() and any(p.is_dir() for p in data_dir.iterdir()) else root_dir


def write_label_lines(label_path, text_lines):
    with open(label_path, "w", encoding="utf-8") as f:
        for line in text_lines:
            f.write(f"{line}\n")


def regenerate_page(img_path, dry_run):
    img_path = Path(img_path)
    label_path = img_path.with_name(img_path.stem + "_labels.txt")
    old_label_lines = ReadLabelLines(str(img_path))  # captured before any deletion

    # If the top handwriting-region boundary was a guess rather than a
    # detected rule line, the "handwriting" region can start inside the
    # printed header -- meaning the first detected "line" would actually be
    # printed text, mislabeled as if it were handwriting. Don't generate a
    # label at all for a page in that state; better to leave it unlabeled
    # than to confidently write a wrong one.
    try:
        if PageBoundsAreUncertain(str(img_path)):
            if not dry_run and label_path.exists():
                label_path.unlink()
            return {"status": "boundary_uncertain", "old_label_lines": old_label_lines}
    except Exception as e:
        return {"status": "error", "detail": str(e), "old_label_lines": old_label_lines}

    try:
        avg_conf, n_words_header = HeaderOcrConfidence(str(img_path))
    except Exception as e:
        return {"status": "error", "detail": str(e), "old_label_lines": old_label_lines}

    if avg_conf < MIN_AVG_CONF or n_words_header < MIN_WORD_COUNT:
        if not dry_run and label_path.exists():
            label_path.unlink()
        return {
            "status": "low_confidence",
            "avg_conf": avg_conf,
            "n_words": n_words_header,
            "old_label_lines": old_label_lines,
        }

    try:
        samples, text_lines, preview = ExtractLinePatches(str(img_path), expectedLineCount=None, labelLines=None)
    except Exception as e:
        return {"status": "error", "detail": str(e), "old_label_lines": old_label_lines}

    non_empty = [t for t in text_lines if t.strip()]
    if not non_empty:
        if not dry_run and label_path.exists():
            label_path.unlink()
        return {"status": "empty", "old_label_lines": old_label_lines}

    if not dry_run:
        if label_path.exists():
            label_path.unlink()
        write_label_lines(label_path, text_lines)

    return {
        "status": "written",
        "n_lines": len(text_lines),
        "text_lines": text_lines,
        "preview": preview,
        "old_label_lines": old_label_lines,
    }


def save_review_bundle(author_id, img_path, result, review_dir):
    page_name = Path(img_path).stem
    base = review_dir / f"{author_id}_{page_name}"
    result["preview"].save(base.with_suffix(".png"))

    with open(base.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write(f"Page: {author_id}/{Path(img_path).name}\n")
        f.write(f"Lines detected (NEW): {result['n_lines']}\n\n")
        for idx, line in enumerate(result["text_lines"]):
            f.write(f"{idx}: {line}\n")

        if result["old_label_lines"]:
            f.write(f"\n--- OLD labels (before this regeneration), for comparison ---\n")
            for idx, line in enumerate(result["old_label_lines"]):
                f.write(f"{idx}: {line}\n")
        else:
            f.write("\n--- No OLD label existed for this page before regeneration ---\n")


def main():
    parser = argparse.ArgumentParser(description="Regenerate _labels.txt for every page with the new word-count splitter.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--review-every", type=int, default=40,
                         help="Save a verification bundle every Nth page that gets a fresh label written (default: 40).")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--dry-run", action="store_true",
                         help="Report what would happen without deleting or writing any label files.")
    args = parser.parse_args()

    root_dir = resolve_dataset_root(Path(args.data_dir))
    author_folders = sorted(p.name for p in root_dir.iterdir() if p.is_dir())

    review_dir = Path(args.review_dir)
    if not args.dry_run:
        review_dir.mkdir(parents=True, exist_ok=True)

    mode = "DRY RUN -- no files will be touched" if args.dry_run else "LIVE -- deleting and writing real label files"
    print(f"[Regenerate] Mode: {mode}")
    print(f"[Regenerate] {len(author_folders)} author folder(s) under {root_dir}.")
    print(f"[Regenerate] Protected (never touched): {sorted(PROTECTED_AUTHORS)}\n")

    stats = {"written": 0, "boundary_uncertain": 0, "low_confidence": 0, "empty": 0, "error": 0}
    protected_page_count = 0
    total_pages = 0
    written_counter = 0
    line_count_changed = 0

    for author_id in author_folders:
        author_dir = root_dir / author_id
        page_paths = sorted(author_dir.glob("*.png"))

        if author_id in PROTECTED_AUTHORS:
            protected_page_count += len(page_paths)
            continue

        for img_path in page_paths:
            total_pages += 1
            result = regenerate_page(img_path, args.dry_run)
            stats[result["status"]] = stats.get(result["status"], 0) + 1

            if result["status"] == "written":
                old_count = len(result["old_label_lines"]) if result["old_label_lines"] else None
                if old_count is not None and old_count != result["n_lines"]:
                    line_count_changed += 1

                written_counter += 1
                if written_counter % args.review_every == 0:
                    if not args.dry_run:
                        save_review_bundle(author_id, img_path, result, review_dir)
                    print(f"  [Review bundle #{written_counter // args.review_every}] {author_id}/{img_path.name} "
                          f"({result['n_lines']} lines, old had {old_count if old_count is not None else 'no label'})")

            if total_pages % 200 == 0:
                print(f"  ... {total_pages} pages processed so far "
                      f"(written={stats.get('written', 0)}, boundary_uncertain={stats.get('boundary_uncertain', 0)}, "
                      f"low_conf={stats.get('low_confidence', 0)}, empty={stats.get('empty', 0)}, "
                      f"error={stats.get('error', 0)})")

    print("\n" + "=" * 75)
    print(f"[Regenerate] Total non-protected pages processed:     {total_pages}")
    print(f"[Regenerate] Labels written:                          {stats.get('written', 0)}")
    print(f"[Regenerate] Skipped (header/handwriting boundary uncertain -- see note below): "
          f"{stats.get('boundary_uncertain', 0)}")
    print(f"[Regenerate] Skipped (low OCR confidence on header):  {stats.get('low_confidence', 0)}")
    print(f"[Regenerate] Skipped (no usable text detected):       {stats.get('empty', 0)}")
    print(f"[Regenerate] Errors:                                  {stats.get('error', 0)}")
    print(f"[Regenerate] Protected pages left untouched (author 150): {protected_page_count}")
    print(f"[Regenerate] Pages where new line count differs from old label's line count: "
          f"{line_count_changed}/{stats.get('written', 0)}")

    if stats.get("boundary_uncertain", 0) > 0:
        print(f"\n[Regenerate] Note on the {stats['boundary_uncertain']} boundary-uncertain skip(s): these are "
              f"pages where no reliable rule line was found separating the printed header from the "
              f"handwriting, so the region PageBoundsAreUncertain() would extract risks starting inside "
              f"the printed text instead of below it. Left unlabeled on purpose rather than risk labeling "
              f"printed text as handwriting -- these need a look (see FindRuleLineClusters/DetectPageBounds "
              f"in FullLineBoxMaker.py if you want to loosen or improve that detection for your form layout).")

    if not args.dry_run:
        print(f"\n[Regenerate] Verification bundles saved to:\n  {review_dir}")
        print("[Regenerate] IMPORTANT: line_cache_raw/ was built from the OLD labels and is now stale. "
              "Delete it (or run training with force_rebuild=True) before your next training run.")
    else:
        print("\n[Regenerate] Dry run only -- no files were changed. Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
