"""Reads SegmentationBest/Crops/<Author>/<Page>/*.png crops through the
trained text recognizer and reports char/word accuracy against the real
ground-truth labels in PersonalDataset/<Author>/Labels/, per personal author.

Run from anywhere:
    python SegmentationBest/verify_text_accuracy.py                      # all 5 authors
    python SegmentationBest/verify_text_accuracy.py --authors Owen Thiya # just these authors
    python SegmentationBest/verify_text_accuracy.py --authors Owen --pages Owen02 Owen03
    python SegmentationBest/verify_text_accuracy.py --crop path/to/some_crop.png  # one arbitrary image, no accuracy (no ground truth)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
CNN_DIR = THIS_DIR.parent
sys.path.insert(0, str(CNN_DIR))
sys.path.insert(0, str(CNN_DIR / "AuthorReproductionStuff"))

from TrainText import PaperCRNN, CHARSET, decode_ctc, resize_line_image_fixed, tensor_from_resized
from Evaluate import CharAcc, WordAcc

TEXT_WEIGHTS = CNN_DIR / "weights" / "paper_cnn_bilstm_ctc_joint_best.pt"
AUTHORS = ["Abhinav", "Dylan", "Owen", "Thiya", "Yeukita"]
CROPS_ROOT = THIS_DIR / "Crops"


def load_model(device):
    model = PaperCRNN(num_classes=len(CHARSET) + 1).to(device)
    sd = torch.load(TEXT_WEIGHTS, map_location=device, weights_only=False)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    return model


def read_text(model, pil_img, device):
    t = tensor_from_resized(resize_line_image_fixed(pil_img)).unsqueeze(0).to(device)
    with torch.no_grad():
        return decode_ctc(model(t))[0]


def read_labels(label_path):
    lines = [l for l in label_path.read_text(encoding="utf-8").splitlines()]
    while lines and lines[-1].strip() == "":
        lines.pop()
    return lines


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--authors", nargs="+", choices=AUTHORS, default=None,
                    help="Only test these authors (default: all 5). Case-sensitive, e.g. Owen Thiya.")
    p.add_argument("--pages", nargs="+", default=None,
                    help="Only test these page stems (e.g. Owen02 Owen03), across whichever --authors are selected.")
    p.add_argument("--crop", type=Path, default=None,
                    help="Classify one arbitrary crop image directly and print its predicted text -- "
                         "skips the whole Crops/ folder and accuracy scoring (no ground truth for a loose file).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Weights] {TEXT_WEIGHTS}")
    model = load_model(device)

    if args.crop:
        pred = read_text(model, Image.open(args.crop).convert("L"), device)
        print(f"[{args.crop}] predicted: {pred!r}")
        return

    authors = args.authors if args.authors else AUTHORS

    overall_char, overall_word, overall_n = 0.0, 0.0, 0
    overall_seconds, overall_chars_read = 0.0, 0
    per_author = {}
    rows = []  # author, page, line_idx, pred, truth, char_acc, word_acc, seconds

    for author in authors:
        page_dirs = sorted((CROPS_ROOT / author).iterdir()) if (CROPS_ROOT / author).exists() else []
        if args.pages:
            page_dirs = [d for d in page_dirs if d.name in args.pages]
        if not page_dirs:
            print(f"[Skip] {author}: no matching crops found at {CROPS_ROOT / author} -- run run_full_test.py first.")
            continue

        char_sum, word_sum, n = 0.0, 0.0, 0
        for page_dir in page_dirs:
            label_path = CNN_DIR / "PersonalDataset" / author / "Labels" / f"{page_dir.name}_labels.txt"
            if not label_path.exists():
                continue
            labels = read_labels(label_path)
            crops = sorted(page_dir.glob("[0-9][0-9].png"))
            n_use = min(len(crops), len(labels))
            if len(crops) != len(labels):
                print(f"  [WARN] {author}/{page_dir.name}: {len(crops)} crops vs {len(labels)} labels -- using first {n_use}")
            for crop_path, truth in zip(crops[:n_use], labels[:n_use]):
                t0 = time.perf_counter()
                pred = read_text(model, Image.open(crop_path).convert("L"), device)
                dt = time.perf_counter() - t0
                ca, wa = CharAcc(pred, truth), WordAcc(pred, truth)
                char_sum += ca
                word_sum += wa
                n += 1
                overall_seconds += dt
                overall_chars_read += len(truth)
                rows.append((author, page_dir.name, crop_path.stem, pred, truth, f"{ca:.3f}", f"{wa:.3f}", f"{dt:.4f}"))

        if n == 0:
            continue
        per_author[author] = (char_sum / n, word_sum / n, n)
        overall_char += char_sum
        overall_word += word_sum
        overall_n += n
        print(f"{author}: char-acc {char_sum/n:.3%} | word-acc {word_sum/n:.3%} | {n} lines")

    print()
    if overall_n:
        print(f"OVERALL: char-acc {overall_char/overall_n:.3%} | word-acc {overall_word/overall_n:.3%} | {overall_n} lines")
        print(f"TIMING: {overall_seconds:.1f}s total inference | "
              f"{1000*overall_seconds/overall_n:.1f} ms/line | "
              f"{1000*overall_seconds/max(1,overall_chars_read):.2f} ms/char "
              f"(device={device}, excludes model load + image I/O)")

    results_path = THIS_DIR / "text_accuracy_results.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["author", "page", "line", "predicted", "truth", "char_acc", "word_acc", "seconds"])
        w.writerows(rows)
    print(f"Per-line predictions saved to: {results_path}")


if __name__ == "__main__":
    main()
