import os
import sys
import glob
import traceback

import numpy as np
from PIL import Image
import pytesseract

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from FullLineBoxMaker import ExtractLinePatches, OtsuThreshold, DetectPageBounds, DetectHeaderTopY

REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATASET_10_DIR = os.path.join(REPO_ROOT, "Data", "Datasets", "IAMpages10")

MIN_AVG_CONF = 85.0
MIN_WORD_COUNT = 15
TRUSTED_FOLDERS = {"150"}  # already has hand-verified _labels.txt; never touch


def HeaderOcrConfidence(imgPath):
    rawImage = Image.open(imgPath)
    grayArr = np.array(rawImage.convert('L'))
    h, w = grayArr.shape

    binaryImg = OtsuThreshold(grayArr)
    topY, _botY = DetectPageBounds(binaryImg, grayArr)
    headerTopY = DetectHeaderTopY(binaryImg, grayArr, topY)

    y1 = headerTopY
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


def WriteLabelLines(labelPath, textLines):
    # Plain "one line of text per physical line" format, matching the original
    # hand-verified IAMpages10/150 labels (no leading "N\t" index).
    with open(labelPath, "w", encoding="utf-8") as f:
        for line in textLines:
            f.write(f"{line}\n")


def main():
    folders = sorted([
        f for f in os.listdir(DATASET_10_DIR)
        if os.path.isdir(os.path.join(DATASET_10_DIR, f))
    ])

    print(f"[IAMpages10] Author folders: {folders}")

    for author_id in folders:
        if author_id in TRUSTED_FOLDERS:
            print(f"\n--- {author_id}: trusted hand-verified labels, skipping ---")
            continue

        author_path = os.path.join(DATASET_10_DIR, author_id)
        image_files = sorted(glob.glob(os.path.join(author_path, "*.png")))
        print(f"\n--- {author_id}: {len(image_files)} pages ---")

        for img_path in image_files:
            base_name = os.path.splitext(img_path)[0]
            label_path = base_name + "_labels.txt"
            fname = os.path.basename(img_path)

            try:
                avg_conf, n_words = HeaderOcrConfidence(img_path)
            except Exception as e:
                print(f"  [ERROR-CONF] {fname}: {e}")
                traceback.print_exc()
                continue

            if avg_conf < MIN_AVG_CONF or n_words < MIN_WORD_COUNT:
                print(f"  [LOW_CONF] {fname}: avg_conf={avg_conf:.1f} n_words={n_words}")
                continue

            try:
                samples, textLines, _preview = ExtractLinePatches(img_path, expectedLineCount=None, labelLines=None)
            except Exception as e:
                print(f"  [ERROR] {fname}: {e}")
                traceback.print_exc()
                continue

            non_empty = [t for t in textLines if t.strip()]
            if len(textLines) == 0 or len(non_empty) == 0:
                print(f"  [EMPTY] {fname}: no usable OCR text")
                continue

            maxlen = max(len(t) for t in textLines)
            if maxlen > 100:
                print(f"  [SUSPECT] {fname}: max line length {maxlen} chars (likely mis-segmented), skipping")
                continue

            WriteLabelLines(label_path, textLines)
            print(f"  [OK] {fname}: {len(textLines)} lines, avg_conf={avg_conf:.1f}")


if __name__ == "__main__":
    main()
