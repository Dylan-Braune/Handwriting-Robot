import os
import sys
import glob
import shutil
import traceback

import numpy as np
from PIL import Image
import pytesseract

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from FullLineBoxMaker import ExtractLinePatches, ReadLabelLines, OtsuThreshold, DetectPageBounds

MIN_AVG_CONF = 85.0
MIN_WORD_COUNT = 15


def HeaderOcrConfidence(imgPath):
    """
    Runs a throwaway OCR pass over the printed-header crop region to score
    how trustworthy the transcription is likely to be. Some IAM pages don't
    have the usual printed 'Sentence Database' header (continuation pages,
    misaligned scans), and OCR-ing handwriting or noise as if it were clean
    printed text produces garbage labels. Filtering on OCR confidence here
    is much cheaper than manually checking 1500+ generated label files.
    """
    rawImage = Image.open(imgPath)
    grayArr = np.array(rawImage.convert('L'))
    h, w = grayArr.shape

    binaryImg = OtsuThreshold(grayArr)
    topY, _botY = DetectPageBounds(binaryImg, grayArr)

    y1 = int(h * 0.115)
    y2 = max(y1 + 10, topY - 8)
    x1 = int(w * 0.06)
    x2 = int(w * 0.94)
    region = grayArr[y1:y2, x1:x2]

    try:
        data = pytesseract.image_to_data(region, config='--psm 6', output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data['conf'] if c not in ('-1',) and int(c) >= 0]
        avg_conf = sum(confs) / len(confs) if confs else -1.0
    except Exception:
        avg_conf, confs = -1.0, []

    return avg_conf, len(confs)

REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATASET_671_DIR = os.path.join(REPO_ROOT, "Data", "Datasets", "IAMpages671", "data")
DATASET_10_150_DIR = os.path.join(REPO_ROOT, "Data", "Datasets", "IAMpages10", "150")

REPORT_PATH = os.path.join(SCRIPT_DIR, "handwriting_line_extraction_output", "label_generation_report.txt")


def WriteLabelLines(labelPath, textLines):
    # Plain "one line of text per physical line" format, matching the original
    # hand-verified IAMpages10/150 labels (no leading "N\t" index).
    with open(labelPath, "w", encoding="utf-8") as f:
        for line in textLines:
            f.write(f"{line}\n")


def CopyVerifiedFolder150Labels():
    copied = 0
    for src_label in sorted(glob.glob(os.path.join(DATASET_10_150_DIR, "*_labels.txt"))):
        base = os.path.basename(src_label)
        dst_label = os.path.join(DATASET_671_DIR, "150", base)
        if os.path.exists(os.path.join(DATASET_671_DIR, "150")):
            shutil.copyfile(src_label, dst_label)
            copied += 1
    print(f"[Folder 150] Copied {copied} verified label files from IAMpages10/150 -> IAMpages671/data/150")
    return copied


def RemoveStaleGeneratedLabels():
    removed = 0
    for stale in glob.glob(os.path.join(DATASET_671_DIR, "*", "*_generated_labels.txt")):
        os.remove(stale)
        removed += 1
    print(f"[Cleanup] Removed {removed} stale '_generated_labels.txt' files")
    return removed


def GenerateLabelsForAllAuthors():
    author_folders = sorted([
        f for f in os.listdir(DATASET_671_DIR)
        if os.path.isdir(os.path.join(DATASET_671_DIR, f))
    ])

    total_pages = 0
    generated = 0
    already_had_labels = 0
    empty_ocr = 0
    low_confidence = 0
    errors = 0

    report_lines = []

    for a_idx, author_id in enumerate(author_folders):
        author_path = os.path.join(DATASET_671_DIR, author_id)
        image_files = sorted(glob.glob(os.path.join(author_path, "*.png")))

        for img_path in image_files:
            total_pages += 1
            base_name = os.path.splitext(img_path)[0]
            label_path = base_name + "_labels.txt"

            if os.path.exists(label_path):
                already_had_labels += 1
                continue

            try:
                avg_conf, n_words = HeaderOcrConfidence(img_path)
            except Exception as e:
                errors += 1
                report_lines.append(f"[ERROR-CONF] {img_path}: {e}")
                traceback.print_exc()
                continue

            if avg_conf < MIN_AVG_CONF or n_words < MIN_WORD_COUNT:
                low_confidence += 1
                report_lines.append(
                    f"[LOW_CONF] {img_path}: avg_conf={avg_conf:.1f} n_words={n_words} (threshold {MIN_AVG_CONF}/{MIN_WORD_COUNT})"
                )
                continue

            try:
                samples, textLines, _preview = ExtractLinePatches(img_path, expectedLineCount=None, labelLines=None)
            except Exception as e:
                errors += 1
                report_lines.append(f"[ERROR] {img_path}: {e}")
                traceback.print_exc()
                continue

            non_empty = [t for t in textLines if t.strip()]
            if len(textLines) == 0 or len(non_empty) == 0:
                empty_ocr += 1
                report_lines.append(f"[EMPTY] {img_path}: detected {len(textLines)} lines, 0 non-empty after OCR")
                continue

            WriteLabelLines(label_path, textLines)
            generated += 1

        if (a_idx + 1) % 25 == 0:
            print(f"  ... processed {a_idx + 1}/{len(author_folders)} author folders "
                  f"(generated={generated}, already_had={already_had_labels}, low_conf={low_confidence}, empty={empty_ocr}, errors={errors})")

    print("=" * 75)
    print(f"[Summary] Total pages seen:      {total_pages}")
    print(f"[Summary] Already had labels:    {already_had_labels}")
    print(f"[Summary] Newly generated:       {generated}")
    print(f"[Summary] Low OCR confidence:    {low_confidence}")
    print(f"[Summary] Empty OCR (skipped):   {empty_ocr}")
    print(f"[Summary] Errors:                {errors}")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Total pages seen: {total_pages}\n")
        f.write(f"Already had labels: {already_had_labels}\n")
        f.write(f"Newly generated: {generated}\n")
        f.write(f"Low OCR confidence (skipped): {low_confidence}\n")
        f.write(f"Empty OCR (skipped): {empty_ocr}\n")
        f.write(f"Errors: {errors}\n\n")
        f.write("\n".join(report_lines))
    print(f"\nFull report written to: {REPORT_PATH}")


if __name__ == "__main__":
    if not os.path.exists(DATASET_671_DIR):
        print(f"Error: dataset dir not found: {DATASET_671_DIR}")
        sys.exit(1)

    RemoveStaleGeneratedLabels()
    CopyVerifiedFolder150Labels()
    GenerateLabelsForAllAuthors()
