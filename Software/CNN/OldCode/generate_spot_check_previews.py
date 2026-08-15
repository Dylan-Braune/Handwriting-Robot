import os
import sys
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from FullLineBoxMaker import ExtractLinePatches, ReadLabelLines

REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATASET_DIR = os.path.join(REPO_ROOT, "Data", "Datasets", "IAMpages671", "data")
OUT_DIR = os.path.join(SCRIPT_DIR, "handwriting_line_extraction_output", "spot_check_previews")

STEP = 20


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    author_folders = sorted([
        f for f in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, f))
    ])

    picked = []
    for a_idx in range(0, len(author_folders), STEP):
        author_id = author_folders[a_idx]
        author_dir = os.path.join(DATASET_DIR, author_id)
        label_files = sorted(glob.glob(os.path.join(author_dir, "*_labels.txt")))
        if not label_files:
            print(f"[skip] author {author_id} (index {a_idx}): no labeled pages")
            continue

        label_path = label_files[0]
        img_path = label_path[: -len("_labels.txt")] + ".png"
        label_lines = ReadLabelLines(img_path)

        samples, _texts, preview = ExtractLinePatches(
            img_path, expectedLineCount=len(label_lines), labelLines=label_lines
        )

        out_path = os.path.join(OUT_DIR, f"{author_id}_{os.path.splitext(os.path.basename(img_path))[0]}.png")
        preview.save(out_path)

        label_txt_path = out_path.replace(".png", "_label.txt")
        with open(label_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        picked.append((author_id, os.path.basename(img_path), len(samples), len(label_lines)))
        print(f"[OK] author {author_id} (index {a_idx}): {os.path.basename(img_path)} "
              f"-> {len(samples)} boxes, {len(label_lines)} label lines -> {os.path.basename(out_path)}")

    print(f"\nGenerated {len(picked)} spot-check previews in: {OUT_DIR}")


if __name__ == "__main__":
    main()
