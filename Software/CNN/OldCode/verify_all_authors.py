import os
import sys
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from FullLineBoxMaker import ExtractLinePatches, ReadLabelLines

REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATASET_DIR = os.path.join(REPO_ROOT, "Data", "Datasets", "IAMpages671", "data")
REPORT_PATH = os.path.join(SCRIPT_DIR, "handwriting_line_extraction_output", "line_box_report.json")


def main():
    author_folders = sorted([
        f for f in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, f))
    ])

    total = 0
    matched = 0
    mismatches = []
    per_author_mismatch_count = {}

    for a_idx, author_id in enumerate(author_folders):
        author_dir = os.path.join(DATASET_DIR, author_id)
        label_files = sorted(glob.glob(os.path.join(author_dir, "*_labels.txt")))

        author_mismatches = 0
        for label_path in label_files:
            img_path = label_path[: -len("_labels.txt")] + ".png"
            if not os.path.exists(img_path):
                continue

            label_lines = ReadLabelLines(img_path)
            expected = len(label_lines)

            try:
                samples, _texts, _ = ExtractLinePatches(img_path, expectedLineCount=None, labelLines=None)
            except Exception as e:
                mismatches.append((author_id, os.path.basename(img_path), expected, f"ERROR: {e}"))
                author_mismatches += 1
                continue

            detected = len(samples)
            total += 1
            if detected == expected:
                matched += 1
            else:
                mismatches.append((author_id, os.path.basename(img_path), expected, detected))
                author_mismatches += 1

        if author_mismatches > 0:
            per_author_mismatch_count[author_id] = author_mismatches

        if (a_idx + 1) % 50 == 0:
            print(f"  ... checked {a_idx + 1}/{len(author_folders)} authors "
                  f"(matched={matched}/{total} so far, {len(per_author_mismatch_count)} authors with a mismatch)")

    print("=" * 75)
    print(f"[Line-count verification] {matched}/{total} pages match exactly ({100*matched/max(1,total):.2f}%)")
    print(f"[Line-count verification] Authors with >=1 mismatch: {len(per_author_mismatch_count)}/{len(author_folders)}")

    if mismatches:
        print("\nMismatches:")
        for a, f, e, d in mismatches:
            print(f"  {a}/{f}: expected {e}, detected {d}")

    import json
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "total_pages_checked": total,
            "matched": matched,
            "match_rate": matched / max(1, total),
            "authors_with_mismatch": len(per_author_mismatch_count),
            "mismatches": [{"author": a, "page": f, "expected": e, "detected": d} for a, f, e, d in mismatches],
        }, f, indent=2)
    print(f"\nFull report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
