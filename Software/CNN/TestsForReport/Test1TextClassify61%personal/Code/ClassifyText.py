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
as training (imported directly from TrainText.py), so
what you see here is a true reflection of what training/validation
measured -- not a separate reimplementation that could quietly drift out
of sync with it.

Usage:
    python ClassifyText.py
    Just run it -- no flags. It prompts for everything it needs: the image
    (or folder of images) to transcribe, whether the page is an IAM
    Sentence Database scan or your own page (e.g. a photo of ruled
    notebook paper -- picking "personal" uses SegmentPage's
    component/chain-based segmenter instead of the IAM-specific line
    splitter, see transcribe_page's docstring), and which weights file to
    load. Blank answers fall back to sensible defaults (shown in each
    prompt).
"""

from pathlib import Path

import torch
from PIL import Image

from ExtractIAMLines import ExtractLinePatches, ReadLabelLines
import SegmentPage as PersonalSegmenter
from TrainText import (
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
DEFAULT_CROPS_DIR = NOGIT_DIR / "classify_crops"


def load_model(weights_path, device):
    model = PaperCRNN(num_classes=len(CHARSET) + 1).to(device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def transcribe_page(model, img_path, device, is_dataset=True, crops_dir=None):
    """Segments the page into lines, then runs each line crop through the
    model and greedily decodes it. Returns predicted line strings in
    top-to-bottom order.

    is_dataset=True (default): IAM Sentence Database page layout -- uses
    the SAME ExtractLinePatches call the training image cache uses.

    is_dataset=False: personal/non-IAM page (e.g. a photo of your own ruled
    notebook paper) -- uses SegmentPage.ProcessPage instead of
    ExtractLinePatches' older periodicity-based splitter. That's the
    first-principles component/chain-based segmenter that scores 100%
    (strict ordered TEXT/MESS match) against the labelled pages in
    NOGIT/NonDatasetImages -- see NonDatasetPreprocessing.py's docstring for
    why it replaced the old personal-page path. MESS-tagged regions
    (diagrams/sketches) are skipped here since there's no text in them to
    recognise.

    crops_dir: if given, writes two images per detected line so you can see
    exactly what happened at each stage --
      line_NN_raw.png       the segmented crop straight off the page, before
                             any model preprocessing
      line_NN_model_input.png  what actually gets fed to the network: resized
                             to the model's fixed INPUT_H x INPUT_W via
                             resize_line_image_fixed (the same call used
                             below, before the ink-invert + normalize step
                             that only exists as a tensor, not an image)."""
    if is_dataset:
        line_samples, _, preview = ExtractLinePatches(
            str(img_path), targetHeight=32, maxWidth=1024,
            expectedLineCount=None, labelLines=None, is_dataset=True,
        )
        crops = [sample["raw_crop"] for sample in line_samples]
    else:
        results, preview, _ = PersonalSegmenter.ProcessPage(str(img_path))
        crops = [r["raw_crop"] for r in results if r["tag"] == "TEXT"]

    if crops_dir is not None:
        crops_dir.mkdir(parents=True, exist_ok=True)
        preview.save(crops_dir / "page_preview.png")

    predictions = []
    for idx, crop in enumerate(crops):
        pil_line = Image.fromarray(crop).convert("L")
        pil_line = resize_line_image_fixed(pil_line)
        if crops_dir is not None:
            Image.fromarray(crop).save(crops_dir / f"line_{idx:02d}_raw.png")
            pil_line.save(crops_dir / f"line_{idx:02d}_model_input.png")
        img_tensor = tensor_from_resized(pil_line).unsqueeze(0).to(device)
        with torch.no_grad():
            log_probs = model(img_tensor)
        pred_text = decode_ctc(log_probs)[0]
        predictions.append(pred_text)
    return predictions


def process_image(model, img_path, device, is_dataset=True, crops_root=None):
    print("\n" + "=" * 78)
    print(f"Page: {img_path}")
    print("=" * 78)

    crops_dir = (crops_root / Path(img_path).stem) if crops_root is not None else None
    predictions = transcribe_page(model, img_path, device, is_dataset=is_dataset, crops_dir=crops_dir)
    ground_truth = ReadLabelLines(str(img_path))
    if not is_dataset:
        # SegmentPage's predictions only cover TEXT-tagged regions
        # (MESS/diagram regions are skipped, see transcribe_page) -- drop the
        # label file's MESS placeholder rows so indices line back up 1:1
        # with what was actually predicted.
        ground_truth = [g for g in ground_truth if g.strip() != "MESS"]

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

    if crops_dir is not None:
        print(f"  Line crops + preview saved to: {crops_dir}")

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
    raw = input(
        f"Path to a page image or a folder of page images to transcribe "
        f"(blank = {DEFAULT_INPUT_DIR.name}): "
    ).strip()
    input_path = Path(raw) if raw else DEFAULT_INPUT_DIR

    page_type = input(
        "Is this an IAM Sentence Database page, or your own page (e.g. ruled notebook paper)? "
        "[iam/personal, blank = iam]: "
    ).strip().lower()
    is_dataset = page_type not in ("personal", "p", "own", "mine")

    raw_weights = input(
        f"Path to model weights (blank = {DEFAULT_WEIGHTS.name}): "
    ).strip()
    weights_path = Path(raw_weights) if raw_weights else DEFAULT_WEIGHTS
    if not weights_path.exists():
        print(f"[Error] Weights file not found: {weights_path}")
        return

    save_crops = input(
        f"Save each line's crop + model-input image + a boxed page preview for inspection? "
        f"[Y/n, blank = yes, saved under {DEFAULT_CROPS_DIR.name}]: "
    ).strip().lower()
    crops_root = None if save_crops in ("n", "no") else DEFAULT_CROPS_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Running on: {device}")
    model = load_model(weights_path, device)

    if input_path.is_dir():
        img_paths = sorted(
            p for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
            for p in input_path.glob(ext)
        )
    elif input_path.is_file():
        img_paths = [input_path]
    else:
        print(f"[Error] Path not found: {input_path}")
        return

    if not img_paths:
        print(f"[Error] No image files found at {input_path}")
        return

    total_chars, total_errors = 0, 0
    for img_path in img_paths:
        chars, errors = process_image(model, img_path, device, is_dataset=is_dataset, crops_root=crops_root)
        total_chars += chars
        total_errors += errors

    if total_chars:
        print("\n" + "=" * 78)
        print(f"[Overall] {len(img_paths)} page(s) evaluated | Overall CER: {total_errors / total_chars:.2%} "
              f"({total_errors}/{total_chars} chars) | Overall char accuracy: {1 - total_errors / total_chars:.2%}")
        print("=" * 78)


if __name__ == "__main__":
    main()
