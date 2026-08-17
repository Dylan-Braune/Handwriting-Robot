"""
Runs the trained CRNN (paper_cnn_bilstm_ctc_best.pt by default) on a full
page image -- or a whole folder of page images, e.g. the output of
export_holdout_pages.py -- and prints its transcription line by line.

If a matching <page>_labels.txt sits next to an image (as it will for
anything exported by export_holdout_pages.py), the ground-truth text is
printed alongside each predicted line and a per-page + overall CER is
computed, so you can see exactly how the model does on handwriting it has
never trained on.

Reuses the exact same model class, preprocessing, and CTC decode function
as training (imported directly from train_paper_cnn_bilstm_ctc.py), so
what you see here is a true reflection of what training/validation
measured -- not a separate reimplementation that could quietly drift out
of sync with it.

Usage:
    python classify_page.py
        (prompts for a path; blank defaults to ./holdout_test_pages)
    python classify_page.py --input holdout_test_pages
    python classify_page.py --input some_page.png
    python classify_page.py --input holdout_test_pages --weights paper_cnn_bilstm_ctc_best.pt
    python classify_page.py --input my_notebook_photo.png --personal-page
        (for a page that ISN'T an IAM Sentence Database scan -- e.g. a
        photo of your own ruled notebook paper. Skips the IAM-specific
        header/footer detection, which otherwise misfires badly on a page
        that doesn't have that printed-form layout, and uses a
        periodicity-based line splitter suited to densely-written/ruled
        pages instead. See ExtractLinePatches' docstring in
        FullLineBoxMaker.py for the full story.)
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from FullLineBoxMaker import ExtractLinePatches, ReadLabelLines
from train_paper_cnn_bilstm_ctc import (
    CHARSET,
    PaperCRNN,
    decode_ctc,
    levenshtein,
    resize_line_image_fixed,
    tensor_from_resized,
)

SCRIPT_DIR = Path(__file__).resolve().parent
NOGIT_DIR = SCRIPT_DIR / "NOGIT"
DEFAULT_WEIGHTS = NOGIT_DIR / "weights" / "paper_cnn_bilstm_ctc_best.pt"
DEFAULT_INPUT_DIR = NOGIT_DIR / "holdout_test_pages"


def load_model(weights_path, device):
    model = PaperCRNN(num_classes=len(CHARSET) + 1).to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def transcribe_page(model, img_path, device, is_dataset=True):
    """Segments the page into lines using the SAME ExtractLinePatches call
    the training image cache uses, then runs each line crop through the
    model and greedily decodes it. Returns predicted line strings in
    top-to-bottom order.

    is_dataset=True (default): IAM Sentence Database page layout.
    is_dataset=False: personal/non-IAM page (e.g. a photo of ruled
    notebook paper) -- see ExtractLinePatches' docstring."""
    line_samples, _, _ = ExtractLinePatches(
        str(img_path), targetHeight=32, maxWidth=1024,
        expectedLineCount=None, labelLines=None, is_dataset=is_dataset,
    )
    predictions = []
    for sample in line_samples:
        pil_line = Image.fromarray(sample["raw_crop"]).convert("L")
        pil_line = resize_line_image_fixed(pil_line)
        img_tensor = tensor_from_resized(pil_line).unsqueeze(0).to(device)
        with torch.no_grad():
            log_probs = model(img_tensor)
        pred_text = decode_ctc(log_probs)[0]
        predictions.append(pred_text)
    return predictions


def process_image(model, img_path, device, is_dataset=True):
    print("\n" + "=" * 78)
    print(f"Page: {img_path}")
    print("=" * 78)

    predictions = transcribe_page(model, img_path, device, is_dataset=is_dataset)
    ground_truth = ReadLabelLines(str(img_path))

    page_chars, page_errors = 0, 0
    for idx, pred in enumerate(predictions):
        truth = ground_truth[idx] if idx < len(ground_truth) else None
        print(f"  Line {idx}: PRED  -> {pred}")
        if truth is not None:
            err = levenshtein(pred, truth)
            page_chars += len(truth)
            page_errors += err
            match = "exact match" if pred == truth else f"{err} char diff(s)"
            print(f"          TRUTH -> {truth}   [{match}]")

    if ground_truth:
        if len(predictions) != len(ground_truth):
            print(f"  [NOTE] Detected {len(predictions)} line(s) but the label file has {len(ground_truth)} -- "
                  f"line segmentation didn't line up 1:1 for this page, so the CER below only covers the "
                  f"lines that were matched by index.")
        page_cer = page_errors / max(1, page_chars)
        print(f"  Page CER: {page_cer:.2%} ({page_errors}/{page_chars} chars)")
        return page_chars, page_errors
    else:
        print("  (No _labels.txt found next to this image -- nothing to compare against.)")
        return 0, 0


def main():
    parser = argparse.ArgumentParser(description="Transcribe a page (or folder of pages) using the trained model.")
    parser.add_argument("--input", default=None, help="Path to a page image or a folder of page images.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                         help="Path to model weights (default: paper_cnn_bilstm_ctc_best.pt).")
    parser.add_argument("--personal-page", action="store_true",
                         help="Set this for a page that is NOT an IAM Sentence Database scan (e.g. your own "
                              "ruled notebook paper). Skips IAM-specific header/footer detection and uses a "
                              "periodicity-based line splitter instead -- see FullLineBoxMaker.ExtractLinePatches.")
    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        raw = input(
            f"Path to a page image or a folder of page images to transcribe "
            f"(blank = {DEFAULT_INPUT_DIR.name}): "
        ).strip()
        input_path = raw if raw else str(DEFAULT_INPUT_DIR)

        # Only prompt for this when we're already asking interactively
        # (i.e. --input wasn't passed) -- an unattended/scripted run that
        # passes --input explicitly just uses --personal-page as given
        # (default: IAM dataset page).
        page_type = input(
            "Is this an IAM Sentence Database page, or your own page (e.g. ruled notebook paper)? "
            "[iam/personal, blank = iam]: "
        ).strip().lower()
        is_dataset = page_type not in ("personal", "p", "own", "mine")
    else:
        is_dataset = not args.personal_page

    input_path = Path(input_path)
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[Error] Weights file not found: {weights_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Running on: {device}")
    model = load_model(weights_path, device)

    if input_path.is_dir():
        img_paths = sorted(input_path.glob("*.png"))
    elif input_path.is_file():
        img_paths = [input_path]
    else:
        print(f"[Error] Path not found: {input_path}")
        return

    if not img_paths:
        print(f"[Error] No .png images found at {input_path}")
        return

    total_chars, total_errors = 0, 0
    for img_path in img_paths:
        chars, errors = process_image(model, img_path, device, is_dataset=is_dataset)
        total_chars += chars
        total_errors += errors

    if total_chars:
        print("\n" + "=" * 78)
        print(f"[Overall] {len(img_paths)} page(s) evaluated | Overall CER: {total_errors / total_chars:.2%} "
              f"({total_errors}/{total_chars} chars) | Overall char accuracy: {1 - total_errors / total_chars:.2%}")
        print("=" * 78)


if __name__ == "__main__":
    main()
