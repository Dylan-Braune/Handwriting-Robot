"""
Exercises the ACTUAL auto-labeling path (the one generate_all_labels.py
uses on a page with no per-line ground truth yet: labelLines=None,
expectedLineCount=None -- OCR the printed header, then split those words
across detected lines via CountWordsFromInkColumns/SplitWordsIntoLines in
FullLineBoxMaker.py) on every Nth page (default: every 40th, across the
whole dataset in sorted order), and exports every resulting line as its own
PNG next to a .txt showing both the freshly-generated text and whatever is
currently saved in that page's _labels.txt, side by side.

This is deliberately NOT the same as just re-reading existing _labels.txt
files: if a page already has a label file, ExtractLinePatches will just
echo it back rather than re-running the splitter, which wouldn't tell you
anything about whether the new word-count-based logic actually improved
things. Forcing labelLines=None here guarantees the new splitter runs, so
you can judge it directly -- and the "OLD (saved)" line lets you compare
against what was there before without needing a separate run.

This is different from generate_spot_check_previews.py (whole-page preview
with bounding boxes, every 20th author) and from verify_line_cache.py's
review export (only the worst-scoring cached samples).

Usage:
    python export_line_quality_check.py                  # every 40th page
    python export_line_quality_check.py --step 20         # every 20th page
    python export_line_quality_check.py --out my_review   # custom output folder
"""

import argparse
from pathlib import Path

from FullLineBoxMaker import ExtractLinePatches, ReadLabelLines

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[1] / "Data" / "Datasets" / "IAMpages671"
DEFAULT_OUT_DIR = SCRIPT_DIR / "line_quality_check"


def resolve_dataset_root(root_dir):
    data_dir = root_dir / "data"
    return data_dir if data_dir.exists() and any(p.is_dir() for p in data_dir.iterdir()) else root_dir


def collect_labeled_pages(data_dir):
    """Flat, consistently-ordered list of (author_id, img_path) across every
    author folder -- sorted the same way the rest of the pipeline enumerates
    pages, so "every Nth page" means the same thing here as it does when the
    dataset is actually built. Only pages that already have SOME label file
    are included, purely so there's an "OLD (saved)" value to compare
    against -- the new text itself is always freshly generated, not read
    from that file."""
    root_dir = resolve_dataset_root(Path(data_dir))
    author_folders = sorted(p.name for p in root_dir.iterdir() if p.is_dir())

    pages = []
    for author_id in author_folders:
        author_dir = root_dir / author_id
        for img_path in sorted(author_dir.glob("*.png")):
            if ReadLabelLines(str(img_path)):
                pages.append((author_id, img_path))
    return pages


def export_page_lines(author_id, img_path, out_dir):
    existing_label_lines = ReadLabelLines(str(img_path))

    # Force the real auto-generation path -- labelLines=None,
    # expectedLineCount=None -- exactly like generate_all_labels.py does
    # for a page it hasn't labeled yet. This is what actually exercises
    # CountWordsFromInkColumns / SplitWordsIntoLines.
    samples, generated_text_lines, _preview = ExtractLinePatches(
        str(img_path), expectedLineCount=None, labelLines=None
    )

    page_name = img_path.stem
    page_dir = out_dir / f"{author_id}_{page_name}"
    page_dir.mkdir(parents=True, exist_ok=True)

    n = len(samples)
    for line_idx in range(n):
        line_png = page_dir / f"line_{line_idx:02d}.png"
        line_txt = page_dir / f"line_{line_idx:02d}.txt"
        samples[line_idx]["processed_patch"].save(line_png)

        new_text = generated_text_lines[line_idx] if line_idx < len(generated_text_lines) else "<no text generated>"
        old_text = (
            existing_label_lines[line_idx]
            if line_idx < len(existing_label_lines)
            else "<no saved label at this line index>"
        )

        with open(line_txt, "w", encoding="utf-8") as f:
            f.write(f"NEW (word-count splitter): {new_text}\n")
            f.write(f"OLD (currently saved):     {old_text}\n")

    return n, len(existing_label_lines)


def main():
    parser = argparse.ArgumentParser(description="Export every line from every Nth page for manual quality review.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--step", type=int, default=40, help="Export every Nth page (default: 40).")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = collect_labeled_pages(args.data_dir)
    print(f"[Export] Found {len(pages)} labeled page(s) total. Exporting every {args.step}th one.")

    selected = pages[::args.step]
    print(f"[Export] Selected {len(selected)} page(s) to export.\n")

    total_lines, total_old_labels, count_mismatches = 0, 0, 0
    for i, (author_id, img_path) in enumerate(selected):
        try:
            detected_count, old_label_count = export_page_lines(author_id, img_path, out_dir)
        except Exception as e:
            print(f"  [ERROR] {author_id}/{img_path.name}: {e}")
            continue

        # detected_count comes from the unconstrained (expectedLineCount=None)
        # segmentation pass, so this compares fresh detection against
        # whatever line count the page's currently-saved labels assumed --
        # not a pass/fail on its own, just a signal worth a look if they differ.
        same = detected_count == old_label_count
        if not same:
            count_mismatches += 1
        status = "SAME COUNT" if same else "DIFFERENT COUNT"
        print(f"  [{status}] ({i + 1}/{len(selected)}) {author_id}/{img_path.name}: "
              f"detected {detected_count} line(s) vs {old_label_count} in the currently-saved label file")
        total_lines += detected_count
        total_old_labels += old_label_count

    print(f"\n[Export] Done. Exported {total_lines} line image+label pair(s) from {len(selected)} page(s) to:\n  {out_dir}")
    print(f"[Export] {count_mismatches}/{len(selected)} page(s) had a different detected line count than "
          f"their currently-saved label file (not necessarily wrong -- just worth a look).")
    print("[Export] Each page got its own subfolder -- open a few line_NN.txt files and compare the "
          "NEW (word-count splitter) line against the OLD (currently saved) one, next to line_NN.png, "
          "to judge whether the new splitter is doing a better job of matching the actual handwriting.")


if __name__ == "__main__":
    main()
