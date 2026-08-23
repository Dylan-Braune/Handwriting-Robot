"""
Scans your real _labels.txt files (no image cache / torch needed -- just
reads text files) and flags any single "line" label that's suspiciously
long. A real handwritten line rarely holds more than ~60-90 characters;
anything much longer usually means line segmentation collapsed several
real handwritten lines into one detected region during label regeneration,
and the whole run of printed words got dumped into a single line label.

Training already filters these out automatically (see MAX_SAFE_LABEL_CHARS
in train_paper_cnn_bilstm_ctc.py), so this isn't required before training --
it's for finding and understanding which specific pages are affected, in
case you want to inspect/re-check their line segmentation later.

Usage:
    python scan_label_lengths.py
        (prompts: "How many author folders to scan?", blank = all)
    python scan_label_lengths.py --num-folders 50
    python scan_label_lengths.py --threshold 120
"""

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[1] / "Data" / "Datasets" / "IAMpages671"


def resolve_dataset_root(root_dir):
    data_dir = root_dir / "data"
    return data_dir if data_dir.exists() and any(p.is_dir() for p in data_dir.iterdir()) else root_dir


def main():
    parser = argparse.ArgumentParser(description="Scan _labels.txt files for suspiciously long single-line labels.")
    parser.add_argument("--num-folders", type=int, default=None)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--threshold", type=int, default=120,
                         help="Flag any line label longer than this many characters (default 120).")
    args = parser.parse_args()

    num_folders = args.num_folders
    if num_folders is None:
        raw = input("How many author folders to scan? (blank = all): ").strip()
        num_folders = int(raw) if raw.isdigit() and int(raw) > 0 else None

    root_dir = resolve_dataset_root(Path(args.data_dir))
    author_folders = sorted(p.name for p in root_dir.iterdir() if p.is_dir())
    if num_folders is not None:
        author_folders = author_folders[:num_folders]

    all_lengths = []
    flagged = []

    for author_id in author_folders:
        author_dir = root_dir / author_id
        for label_path in sorted(author_dir.glob("*_labels.txt")):
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [line.rstrip("\n") for line in f if line.strip()]
            for idx, line in enumerate(lines):
                all_lengths.append(len(line))
                if len(line) > args.threshold:
                    flagged.append((author_id, label_path.name, idx, len(line), line))

    if all_lengths:
        all_lengths.sort()
        median = all_lengths[len(all_lengths) // 2]
        print(f"Scanned {len(author_folders)} author folder(s), {len(all_lengths)} total line label(s).")
        print(f"Median line length: {median} chars | Max: {max(all_lengths)} chars | Threshold: {args.threshold} chars\n")
    else:
        print("No label lines found.")
        return

    if not flagged:
        print(f"No lines longer than {args.threshold} chars found -- nothing to flag.")
        return

    print(f"[FLAGGED] {len(flagged)} line(s) over {args.threshold} chars (likely collapsed/merged line segmentation):\n")
    for author_id, label_file, line_idx, length, text in flagged:
        print(f"  {author_id}/{label_file}  line {line_idx}  ({length} chars)")
        print(f"    {text[:100]}{'...' if len(text) > 100 else ''}")

    print(f"\nTo inspect any of these visually, run generate_boxes_and_labels.py and check that specific "
          f"author's preview image -- if the handwriting region shows several real lines merged into one "
          f"detected box, that confirms a segmentation issue for that page specifically.")


if __name__ == "__main__":
    main()
