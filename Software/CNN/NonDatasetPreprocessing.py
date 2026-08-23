"""
NonDatasetPreprocessing.py

PERSONAL-mode (non-IAM-dataset) page segmentation entry point.

This file used to contain its own hand-built preprocessing/MESS-classifier
pipeline (row-projection valley splitting + a 5-signal weighted MESS score).
That pipeline hit a real ceiling: on the 3-page NOGIT/NonDatasetImages test
set, borderline diagram detections flipped from a sub-1-degree deskew change
alone, and a bold underlined title got misclassified as MESS because its
capital letters were legitimately taller than the page's median line height
-- the same geometric property the classifier used to catch diagrams. Both
are documented in this repo's history as things pure weighted-average
geometry couldn't resolve without a larger labelled dataset or a real
text-recognition-confidence signal.

NonDatasetSegmenterFP.py (first-principles: numpy + fp_ops.py only, no
OpenCV/scipy) replaces that with a component-level algorithm -- page
detection, chain-based line building along fitted baseline curves, and a
MESS score built on largest-single-enclosed-hole-size plus a neighbour veto
that specifically rules out "big letters in a heading" -- and it scores
100% (strict ordered TEXT/MESS match) against all 3 labelled test pages,
including the two failure modes above. See NonDatasetSegmenterFP.py's own
docstring for its 9 pipeline stages.

THIS FILE now just adapts that algorithm to the interactive,
menu-driven workflow the rest of this project's scripts use (per project
convention: run the script, press a number, no CLI flags) and writes
model-ready crops in the same aspect-preserving/padded format
NonDatasetSegmenterTest.py already produces, so a script that used to import
from NonDatasetPreprocessing.py (nothing in this repo currently does) has
one place to call. It deliberately does NOT reimplement, wrap, or modify
NonDatasetSegmenterCV.py / NonDatasetSegmenterFP.py / fp_ops.py /
NonDatasetSegmenterTest.py -- those are imported as-is.

If you want the OpenCV-backed version instead (faster, same algorithm,
needs opencv-python installed), swap the single import below from
NonDatasetSegmenterFP to NonDatasetSegmenterCV -- both expose the same
ProcessPage(imgPath) -> (results, previewPIL, meta) interface.
"""

import os
import sys
import glob
import json
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import NonDatasetSegmenterFP as Segmenter
from NonDatasetSegmenterTest import ReadLabels, Score

# Keep in sync with train_paper_cnn_bilstm_ctc.py INPUT_HEIGHT / INPUT_WIDTH
# (also mirrored in NonDatasetSegmenterTest.py -- kept here too so this file
# doesn't have to reach into that module's internals for it).
INPUT_H, INPUT_W = 64, 640

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "NOGIT", "NonDatasetImages")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "NOGIT", "NonDatasetTestOutput", "preprocessing")


def ProcessNonDatasetPage(imgPath):
    """Thin passthrough to the first-principles segmenter's ProcessPage, kept
    under this file's old function name in case anything expects it.
    Returns (results, previewPIL, meta) -- see NonDatasetSegmenterFP.ProcessPage
    for the exact shape of `results` (order, tag, bbox, raw_crop, n_components)
    and `meta` (skew, textH, nText, nMess)."""
    return Segmenter.ProcessPage(imgPath)


def SaveLineCrops(results, cropDir, modelDir):
    """Writes both the raw non-rectangular crop (white background, exactly
    the line's own ink -- see NonDatasetSegmenterFP's RenderLine) and a
    model-ready version scaled to INPUT_H x INPUT_W with aspect ratio kept
    and the remainder padded white, matching NonDatasetSegmenterTest.py's
    convention so crops from either script are trainer-compatible."""
    os.makedirs(cropDir, exist_ok=True)
    os.makedirs(modelDir, exist_ok=True)
    for r in results:
        fname = f"line_{r['order']:02d}_{r['tag']}.png"
        img = Image.fromarray(r['raw_crop'])
        img.save(os.path.join(cropDir, fname))

        g = img.convert('L')
        s = min(INPUT_W / g.width, INPUT_H / g.height)
        g = g.resize((max(1, int(g.width * s)), max(1, int(g.height * s))),
                      Image.Resampling.BILINEAR)
        canvas = Image.new('L', (INPUT_W, INPUT_H), 255)
        canvas.paste(g, (0, (INPUT_H - g.height) // 2))
        canvas.save(os.path.join(modelDir, fname))


def RunTestOnImage(imgPath, outputRoot=OUTPUT_ROOT):
    name = os.path.splitext(os.path.basename(imgPath))[0]
    results, preview, meta = ProcessNonDatasetPage(imgPath)

    previewDir = os.path.join(outputRoot, "previews")
    os.makedirs(previewDir, exist_ok=True)
    preview.save(os.path.join(previewDir, f"{name}_preview.png"))

    SaveLineCrops(
        results,
        os.path.join(outputRoot, "crops", name),
        os.path.join(outputRoot, "crops_model", name),
    )

    report = dict(
        page=name,
        detected=len(results),
        detected_mess=sum(1 for r in results if r['tag'] == 'MESS'),
        skew_deg=round(meta.get('skew', 0.0), 2),
        text_height_px=round(meta.get('textH', 0.0), 1),
    )

    labels = ReadLabels(imgPath)
    if labels is None:
        print(f"{name}: no label file -- segmented {len(results)} boxes (no accuracy score)")
        return report

    acc, expTags, detTags = Score(results, labels)
    report.update(
        expected=len(expTags),
        expected_mess=expTags.count('MESS'),
        accuracy=round(acc, 4),
    )
    print(
        f"{name}: acc={acc * 100:.1f}%  expected {len(expTags)} rows "
        f"({expTags.count('MESS')} MESS) | detected {len(detTags)} "
        f"({detTags.count('MESS')} MESS) | skew={report['skew_deg']}deg"
    )
    if acc < 1.0:
        for i in range(max(len(expTags), len(detTags))):
            e = expTags[i] if i < len(expTags) else '--'
            d = detTags[i] if i < len(detTags) else '--'
            if e != d:
                lbl = labels[i][:50] if i < len(labels) else ''
                print(f"   {i:2d}  exp={e:4s} det={d:4s}  {lbl}   <<< MISMATCH")
    return report


def WriteReport(reports, outputRoot=OUTPUT_ROOT):
    reportPath = os.path.join(outputRoot, "report.json")
    os.makedirs(outputRoot, exist_ok=True)
    with open(reportPath, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    return reportPath


def RunInteractiveMenu():
    imagePaths = sorted(
        p for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
        for p in glob.glob(os.path.join(IMAGES_DIR, ext))
    )

    print("=== NonDatasetPreprocessing (first-principles segmenter) test menu ===")
    if not imagePaths:
        print(f"No images found in {IMAGES_DIR}")
        return

    print(f"Found {len(imagePaths)} image(s) in NOGIT/NonDatasetImages:")
    for i, p in enumerate(imagePaths):
        print(f"  {i + 1}) {os.path.basename(p)}")
    print("  0) Run ALL")

    choice = input("Pick a page to process (number), or 0 for all: ").strip()
    try:
        choiceNum = int(choice)
    except ValueError:
        choiceNum = 0

    targets = [imagePaths[choiceNum - 1]] if choiceNum in range(1, len(imagePaths) + 1) else imagePaths

    reports = [RunTestOnImage(p) for p in targets]
    reportPath = WriteReport(reports)

    print(f"\nPreviews saved to:    {os.path.join(OUTPUT_ROOT, 'previews')}")
    print(f"Line crops saved to:  {os.path.join(OUTPUT_ROOT, 'crops')}")
    print(f"Model-ready crops to: {os.path.join(OUTPUT_ROOT, 'crops_model')}")
    print(f"Report saved to:      {reportPath}")


if __name__ == "__main__":
    RunInteractiveMenu()
