"""
Copies every page currently used as HELD-OUT validation data into its own
folder, alongside its _labels.txt, so you can test the trained model
against pages it has genuinely never trained on.

Uses the EXACT same selection rule as IAMLineDatasetRaw in
train_paper_cnn_bilstm_ctc.py: the last labeled page of any author with at
least MIN_PAGES_FOR_HOLDOUT labeled pages. That means the pages this script
exports are precisely the ones val loss/val char-accuracy were computed
against during training -- not a freshly-chosen, possibly-different sample.

Usage:
    python export_holdout_pages.py
        (writes to ./holdout_test_pages by default)
    python export_holdout_pages.py --out-dir my_test_folder
"""

import argparse
import shutil
from pathlib import Path

from FullLineBoxMaker import ReadLabelLines

SCRIPT_DIR = Path(__file__).resolve().parent
NOGIT_DIR = SCRIPT_DIR / "NOGIT"
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[1] / "Data" / "Datasets" / "IAMpages671"
DEFAULT_OUT_DIR = NOGIT_DIR / "holdout_test_pages"

# Must match IAMLineDatasetRaw's min_pages_for_holdout default in
# train_paper_cnn_bilstm_ctc.py -- if you ever change one, change both.
MIN_PAGES_FOR_HOLDOUT = 3


def resolve_dataset_root(root_dir):
    data_dir = root_dir / "data"
    return data_dir if data_dir.exists() and any(p.is_dir() for p in data_dir.iterdir()) else root_dir


def main():
    parser = argparse.ArgumentParser(description="Export the held-out validation pages to a folder for manual testing.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    root_dir = resolve_dataset_root(Path(args.data_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    author_folders = sorted(p.name for p in root_dir.iterdir() if p.is_dir())
    exported = 0

    for author_id in author_folders:
        author_dir = root_dir / author_id
        author_pages = sorted(author_dir.glob("*.png"))
        if not author_pages:
            continue
        labeled_pages = [p for p in author_pages if ReadLabelLines(str(p))]
        if len(labeled_pages) < MIN_PAGES_FOR_HOLDOUT:
            continue  # this author never contributed a holdout page during training

        holdout_page = labeled_pages[-1]
        label_path = holdout_page.with_name(holdout_page.stem + "_labels.txt")

        dest_img = out_dir / f"{author_id}_{holdout_page.name}"
        dest_label = out_dir / f"{author_id}_{holdout_page.stem}_labels.txt"

        shutil.copy2(holdout_page, dest_img)
        if label_path.exists():
            shutil.copy2(label_path, dest_label)

        exported += 1

    print(f"[Export] Exported {exported} held-out page(s) (+ their labels) to:\n  {out_dir.resolve()}")
    print("[Export] These are the exact same pages train_paper_cnn_bilstm_ctc.py computes val loss/accuracy "
          "against -- the model has never trained on a single line from any of them.")


if __name__ == "__main__":
    main()
