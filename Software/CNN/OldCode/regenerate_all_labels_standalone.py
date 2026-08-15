"""
STANDALONE, SELF-CONTAINED VERSION -- everything needed is in this one file.
No local imports of FullLineBoxMaker.py / generate_all_labels.py /
regenerate_all_labels.py -- only standard library + numpy/PIL/pytesseract
(third-party packages, still need to be installed, same as every other
script in this project). Safe to copy to a machine that doesn't have the
rest of the repo's CNN/ folder synced.

Regenerates _labels.txt for every page across the whole IAMpages671 dataset
using the word-count-based line splitter (CountWordsFromInkColumns /
SplitWordsIntoLines), REPLACING the old width-based-guess labels in place,
in your actual dataset folder.

WHAT THIS DOES
--------------
  - For every page NOT in the hand-verified set (author folder "150" --
    copied in from IAMpages10/150, manually verified, never touched by
    this script), deletes its existing _labels.txt (if any) and writes a
    fresh one: OCR the printed header, then split those words across
    detected lines with the word-count-based logic.
  - Applies two quality gates before writing anything, either of which
    leaves a page with NO label file (its old one, if any, is still
    deleted, since it was built on the same guess-based logic this
    replaces and isn't more trustworthy than having no label):
      1. PageBoundsAreUncertain() -- True when no reliable rule line was
         found separating the printed header from the handwriting,
         meaning the extracted "handwriting" region risks actually
         starting inside the printed text. This is the "stealing from the
         typed header" failure mode -- a page can pass a pure line-COUNT
         check while still having this problem, since the count can come
         out right by coincidence even when the top "line" is really
         printed text. Caught here, before any label gets written.
      2. Header OCR confidence >= 85, at least 15 words.
  - Every N pages that actually get a fresh label written (default 40),
    saves a verification bundle to a review folder: the annotated preview
    image (line boxes, numbered to match the label list) plus a .txt with
    every line's freshly-generated text, numbered the same way, and the
    OLD label (if there was one) underneath for comparison.

WHAT THIS DOES NOT DO -- READ THIS FIRST
-----------------------------------------
There's no way to *prove* every line matches perfectly without either (a)
the official IAM per-line ground truth (lines.xml/lines.txt from the real
IAM database -- not present in this repo, would need downloading separately
with FKI Bern registration), or (b) a human reading every single line,
which defeats the point of automatic labeling. What this script gives you
instead, as the best available automatic proxies:
  1. A live count of how many pages end up with a DIFFERENT number of
     detected lines than their old label had -- printed in the summary.
  2. The periodic verification bundles, for actually reading a sample
     yourself.
None of this replaces occasionally opening the review folder and reading a
handful of pages.

AFTER RUNNING THIS: line_cache_raw/ (used by train_paper_cnn_bilstm_ctc.py)
was built from the OLD labels and will be stale. Delete it or set
force_rebuild=True before your next training run, or you'll silently keep
training on the old, replaced labels.

Usage (this runs LIVE by default -- no --dry-run needed):
    python regenerate_all_labels_standalone.py
    python regenerate_all_labels_standalone.py --review-every 20
    python regenerate_all_labels_standalone.py --dry-run     # report only, touches nothing
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pytesseract

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# =====================================================================
# Otsu's Automatic Image Thresholding (First Principles)
# =====================================================================
def OtsuThreshold(grayImg):
    hist, _ = np.histogram(grayImg, bins=256, range=(0, 256))
    total = grayImg.size
    currentMax, bestThresh, sumB, sum1, wB = 0, 0, 0, np.dot(np.arange(256), hist), 0

    for t in range(256):
        wB += hist[t]
        if wB == 0: continue
        wF = total - wB
        if wF == 0: break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        varBetween = wB * wF * ((mB - mF) ** 2)
        if varBetween > currentMax:
            currentMax = varBetween
            bestThresh = t

    return (grayImg < bestThresh).astype(np.uint8) * 255


def LongestRun(boolRow):
    """Small helper for detecting straight printed divider lines."""
    inkCols = np.where(boolRow)[0]
    if len(inkCols) == 0:
        return 0
    gaps = np.where(np.diff(inkCols) > 1)[0]
    starts = np.concatenate(([0], gaps + 1))
    ends = np.concatenate((gaps, [len(inkCols) - 1]))
    return int(np.max(inkCols[ends] - inkCols[starts] + 1))


def ClusterRows(rows, maxGap=5):
    if not rows:
        return []

    clusters, currGroup = [], [rows[0]]
    for i in range(1, len(rows)):
        if rows[i] <= rows[i - 1] + maxGap:
            currGroup.append(rows[i])
        else:
            clusters.append(int(np.mean(currGroup)))
            currGroup = [rows[i]]
    clusters.append(int(np.mean(currGroup)))
    return clusters


# =====================================================================
# Page Bounds Detection
# =====================================================================
def FindRuleLineClusters(binaryImg, rawGrayscaleArray=None):
    h, w = binaryImg.shape
    ruleRows = []

    lineSource = rawGrayscaleArray if rawGrayscaleArray is not None else binaryImg
    for r in range(h):
        if rawGrayscaleArray is not None:
            row = lineSource[r, :] < 225
        else:
            row = lineSource[r, :] > 0

        if not np.any(row): continue
        if LongestRun(row) > w * 0.25 and np.sum(row) > w * 0.10:
            ruleRows.append(r)

    if not ruleRows:
        return []

    return ClusterRows(ruleRows, maxGap=8)


def DetectPageBounds(binaryImg, rawGrayscaleArray=None, return_confidence=False):
    """
    return_confidence=False (default): returns (topY, botY).
    return_confidence=True: also returns whether topY/botY had to fall back
    to a fixed percentage-of-height GUESS rather than an actual detected
    rule line. When topY falls back, the top of the "handwriting region"
    can end up sitting inside the printed prompt text instead of below it.
    See PageBoundsAreUncertain() below.
    """
    h, w = binaryImg.shape
    clusters = FindRuleLineClusters(binaryImg, rawGrayscaleArray)

    topCandidates = [c for c in clusters if h * 0.12 < c < h * 0.40]
    botCandidates = [c for c in clusters if h * 0.55 < c < h * 0.92]

    topY = topCandidates[-1] + 8 if topCandidates else int(h * 0.12)
    botY = botCandidates[0] - 8 if botCandidates else int(h * 0.80)

    if return_confidence:
        return topY, botY, len(topCandidates) == 0, len(botCandidates) == 0
    return topY, botY


def HeaderGapIsSafe(binaryImg, topY, min_gap_px=10, max_ink_fraction=0.05):
    """
    Independent, content-based check: looks at the min_gap_px rows
    immediately ABOVE topY and confirms it's mostly blank -- consistent
    with topY actually sitting in the gap after the header's last line of
    print, not on top of it.
    """
    h = binaryImg.shape[0]
    zoneStart = max(0, topY - min_gap_px)
    checkZone = binaryImg[zoneStart:topY, :]
    if checkZone.size == 0:
        return True

    inkFractionPerRow = np.mean(checkZone > 0, axis=1)
    return not np.any(inkFractionPerRow > max_ink_fraction)


def PageBoundsAreUncertain(imgPath):
    """
    True if this page's handwriting region should NOT be trusted for
    auto-labeling, for either of two independent reasons:
      1. The TOP boundary had to fall back to a fixed-percentage guess.
      2. Even where a rule line WAS detected, the rows immediately above
         topY aren't actually blank (HeaderGapIsSafe).
    """
    rawImage = Image.open(imgPath)
    grayArr = np.array(rawImage.convert('L'))
    binaryImg = OtsuThreshold(grayArr)
    topY, _botY, topWasFallback, _botWasFallback = DetectPageBounds(binaryImg, grayArr, return_confidence=True)

    if topWasFallback:
        return True

    return not HeaderGapIsSafe(binaryImg, topY)


def DetectHeaderTopY(binaryImg, rawGrayscaleArray, topY):
    """Finds the rule line bounding the TOP of the printed sentence-prompt box."""
    h, _w = binaryImg.shape
    clusters = FindRuleLineClusters(binaryImg, rawGrayscaleArray)

    candidates = [c for c in clusters if h * 0.04 < c < (topY - 8)]
    if candidates:
        return max(candidates) + 8

    return int(h * 0.115)


# =====================================================================
# Extract Printed Text Ground-Truth via PyTesseract
# =====================================================================
def ExtractPrintedGroundTruth(rawGrayscaleArray, topY, headerTopY=None):
    """Crops the top printed box region above topY and OCRs the expected printed words."""
    h, w = rawGrayscaleArray.shape
    y1 = headerTopY if headerTopY is not None else int(h * 0.115)
    y2 = max(y1 + 10, topY - 8)
    x1 = int(w * 0.06)
    x2 = int(w * 0.94)
    printedRegion = rawGrayscaleArray[y1:y2, x1:x2]

    try:
        rawText = pytesseract.image_to_string(printedRegion, config='--psm 6')
        cleanText = rawText.replace('\n', ' ').strip()
        cleanText = re.sub(r'\bSentence\s+Database\b', ' ', cleanText, flags=re.IGNORECASE)
        cleanText = re.sub(r'\b[A-Z]?\d{2}-\d{3}[a-z]?\b', ' ', cleanText)
        cleanText = cleanText.replace('|', ' ').replace(' j ', ' ')
        words = [w.strip() for w in cleanText.split() if w.strip()]
        return words
    except Exception as e:
        print(f"  [OCR Warning] Could not extract printed text: {e}")
        return []


def ReadLabelLines(imgPath):
    labelPath = os.path.splitext(imgPath)[0] + "_labels.txt"
    if not os.path.exists(labelPath):
        return []

    with open(labelPath, "r", encoding="utf-8") as f:
        rawLines = [line.strip() for line in f.readlines() if line.strip()]

    return [re.sub(r'^\d+\t', '', line) for line in rawLines]


def HeaderOcrConfidence(imgPath):
    """
    Runs a throwaway OCR pass over the printed-header crop region to score
    how trustworthy the transcription is likely to be.
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


MIN_AVG_CONF = 85.0
MIN_WORD_COUNT = 15


# =====================================================================
# Word-Count Estimation (from ink, not from assumed character width)
# =====================================================================
def CountWordsFromInkColumns(inkCols, lineHeight, minGapFactor=0.6, minRunWidth=3):
    """
    Estimates how many space-separated 'words' are physically present in a
    detected line, from the ink itself -- by clustering ink columns into
    contiguous runs and counting gaps wide enough to be word boundaries.
    """
    if len(inkCols) == 0:
        return 0

    gaps = np.where(np.diff(inkCols) > 1)[0]
    starts = np.concatenate(([0], gaps + 1))
    ends = np.concatenate((gaps, [len(inkCols) - 1]))
    runs = [(inkCols[s], inkCols[e]) for s, e in zip(starts, ends)]

    runs = [(s, e) for s, e in runs if (e - s + 1) >= minRunWidth]
    if not runs:
        return 0

    wordGapThreshold = max(1, lineHeight * minGapFactor)
    wordCount = 1
    for i in range(1, len(runs)):
        gapWidth = runs[i][0] - runs[i - 1][1]
        if gapWidth > wordGapThreshold:
            wordCount += 1

    return wordCount


# =====================================================================
# Proportional Word Distribution Engine
# =====================================================================
def SplitWordsIntoLines(words, lineWordCounts):
    """
    Distributes a continuous list of words across N line crops using a
    per-line word-COUNT estimate (from CountWordsFromInkColumns), scaled so
    the total matches the known transcript's actual word count exactly.
    """
    numLines = len(lineWordCounts)
    if numLines == 0 or not words:
        return [""] * numLines

    numWords = len(words)
    totalEstimated = sum(lineWordCounts)

    if totalEstimated <= 0:
        lineWordCounts = [1] * numLines
        totalEstimated = numLines

    scale = numWords / float(totalEstimated)

    allocatedLines = []
    wordIdx = 0
    for i in range(numLines):
        if wordIdx >= numWords:
            allocatedLines.append("")
            continue

        if i == numLines - 1:
            allocatedLines.append(" ".join(words[wordIdx:]))
            break

        count = max(1, round(lineWordCounts[i] * scale))
        count = min(count, numWords - wordIdx)
        allocatedLines.append(" ".join(words[wordIdx:wordIdx + count]))
        wordIdx += count

    return allocatedLines


def RemoveStraightFormLines(hwRegion):
    cleaned = hwRegion.copy()
    regionH, regionW = cleaned.shape

    for r in range(regionH):
        row = cleaned[r, :] > 0
        if LongestRun(row) > regionW * 0.25 and np.sum(row) > regionW * 0.10:
            cleaned[max(0, r - 2):min(regionH, r + 3), :] = 0

    return cleaned


def MergeUndersizedFragments(lines, minRelativeHeight=0.4):
    """Merge tiny fragments (descender tails, stray marks) into the nearest real line."""
    if len(lines) <= 1:
        return lines

    heights = [e - s for s, e in lines]
    medianHeight = sorted(heights)[len(heights) // 2]
    threshold = medianHeight * minRelativeHeight

    merged = list(lines)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, (s, e) in enumerate(merged):
            if (e - s) >= threshold:
                continue
            if i == 0:
                neighbor = 1
            elif i == len(merged) - 1:
                neighbor = i - 1
            else:
                gapPrev = merged[i][0] - merged[i - 1][1]
                gapNext = merged[i + 1][0] - merged[i][1]
                neighbor = i - 1 if gapPrev <= gapNext else i + 1

            lo, hi = min(i, neighbor), max(i, neighbor)
            combined = (min(merged[lo][0], merged[hi][0]), max(merged[lo][1], merged[hi][1]))
            merged = merged[:lo] + [combined] + merged[hi + 1:]
            changed = True
            break

    return merged


def MergeOrSplitToExpectedLineCount(lines, rowInkCount, expectedLineCount):
    if expectedLineCount is None or expectedLineCount <= 0 or len(lines) == 0:
        return lines

    adjusted = list(lines)

    while len(adjusted) > expectedLineCount:
        gapScores = []
        for i in range(len(adjusted) - 1):
            prevS, prevE = adjusted[i]
            currS, currE = adjusted[i + 1]
            gap = currS - prevE
            gapInk = np.mean(rowInkCount[prevE:currS]) if currS > prevE else 0
            gapScores.append((gap + gapInk, i))

        _, mergeIdx = min(gapScores)
        adjusted[mergeIdx] = (adjusted[mergeIdx][0], adjusted[mergeIdx + 1][1])
        del adjusted[mergeIdx + 1]

    while len(adjusted) < expectedLineCount:
        splittable = [(e - s, i, s, e) for i, (s, e) in enumerate(adjusted) if (e - s) >= 42]
        if not splittable:
            break

        _, splitIdx, s, e = max(splittable)
        lo, hi = s + 18, e - 18
        if hi > lo:
            splitY = lo + int(np.argmin(rowInkCount[lo:hi]))
        else:
            splitY = (s + e) // 2
        adjusted[splitIdx:splitIdx + 1] = [(s, splitY), (splitY, e)]

    return adjusted


# =====================================================================
# Line Extraction Pipeline
# =====================================================================
def ExtractLinePatches(imgPath, targetHeight=32, maxWidth=1024, expectedLineCount=None, labelLines=None):
    rawImage = Image.open(imgPath)
    RawGrayscaleArray = np.array(rawImage.convert('L'))
    h, w = RawGrayscaleArray.shape

    BinaryInvertedImage = OtsuThreshold(RawGrayscaleArray)

    topY, botY = DetectPageBounds(BinaryInvertedImage, RawGrayscaleArray)
    headerTopY = DetectHeaderTopY(BinaryInvertedImage, RawGrayscaleArray, topY)

    printedWords = ExtractPrintedGroundTruth(RawGrayscaleArray, topY, headerTopY)

    # HANDWRITING_TOP_MARGIN: hard buffer so header pixels structurally
    # cannot end up inside a handwriting line crop.
    HANDWRITING_TOP_MARGIN = 15
    handwritingStartY = min(botY, topY + HANDWRITING_TOP_MARGIN)

    hwRegion = BinaryInvertedImage[handwritingStartY:botY, :].copy()
    regionH, regionW = hwRegion.shape

    hwRegion = RemoveStraightFormLines(hwRegion)

    colInkHeights = np.sum(hwRegion > 0, axis=0)
    vertLineCols = np.where(colInkHeights > regionH * 0.40)[0]
    hwRegion[:, vertLineCols] = 0
    hwRegion[:, :int(w * 0.07)] = 0
    hwRegion[:, int(w * 0.97):] = 0

    rowInkCount = np.sum(hwRegion > 0, axis=1)
    hasInk = rowInkCount > max(10, int(w * 0.006))

    pad1D = np.pad(hasInk, (3, 3), mode='constant')
    smoothedInk = np.zeros_like(hasInk)
    for dy in range(7):
        smoothedInk = np.logical_or(smoothedInk, pad1D[dy:dy + len(hasInk)])

    diff = np.diff(np.concatenate(([0], smoothedInk.astype(np.int8), [0])))
    lineStarts, lineEnds = np.where(diff == 1)[0], np.where(diff == -1)[0]

    rawLines = [(s, e) for s, e in zip(lineStarts, lineEnds) if (e - s) >= 8]

    mergedLines = []
    for line in rawLines:
        if not mergedLines:
            mergedLines.append(line)
        else:
            prevS, prevE = mergedLines[-1]
            currS, currE = line
            if (currS - prevE) < 18:
                mergedLines[-1] = (prevS, currE)
            else:
                mergedLines.append(line)

    mergedLines = MergeUndersizedFragments(mergedLines)
    mergedLines = MergeOrSplitToExpectedLineCount(mergedLines, rowInkCount, expectedLineCount)

    drawCanvas = rawImage.convert('RGB')
    drawObj = ImageDraw.Draw(drawCanvas)
    drawObj.line([(0, topY), (w, topY)], fill=(0, 0, 255), width=2)
    drawObj.line([(0, handwritingStartY), (w, handwritingStartY)], fill=(0, 200, 255), width=1)
    drawObj.line([(0, botY), (w, botY)], fill=(0, 0, 255), width=2)

    ExtractedLineSamples = []
    lineWidths = []
    lineWordCounts = []

    for idx, (s, e) in enumerate(mergedLines):
        absY1 = handwritingStartY + max(0, s - 5)
        absY2 = handwritingStartY + min(regionH, e + 5)
        cropH = absY2 - absY1

        lineBinary = hwRegion[max(0, s - 3):min(regionH, e + 3), :]
        colSum = np.sum(lineBinary > 0, axis=0)
        inkCols = np.where(colSum > 0)[0]
        if len(inkCols) == 0: continue

        absX1 = max(0, inkCols[0] - 8)
        absX2 = min(w, inkCols[-1] + 8)
        cropW = absX2 - absX1
        lineWidths.append(cropW)
        lineWordCounts.append(CountWordsFromInkColumns(inkCols, cropH))

        drawObj.rectangle([absX1, absY1, absX2, absY2], outline=(255, 0, 0), width=2)
        drawObj.text((absX1, max(0, absY1 - 14)), str(idx), fill=(0, 128, 0))

        rawLineCrop = RawGrayscaleArray[absY1:absY2, absX1:absX2]
        scaleRatio = targetHeight / float(cropH)
        newWidth = min(maxWidth, max(1, int(cropW * scaleRatio)))

        cropPIL = Image.fromarray(rawLineCrop)
        resizedPIL = cropPIL.resize((newWidth, targetHeight), Image.Resampling.LANCZOS)

        paddedCanvas = Image.new('L', (maxWidth, targetHeight), color=255)
        paddedCanvas.paste(resizedPIL, (0, 0))

        ExtractedLineSamples.append({
            'line_idx': idx,
            'raw_crop': rawLineCrop,
            'processed_patch': paddedCanvas,
            'label_text': labelLines[idx] if labelLines is not None and idx < len(labelLines) else ""
        })

    formattedTextLines = labelLines if labelLines is not None and len(labelLines) == len(ExtractedLineSamples) else SplitWordsIntoLines(printedWords, lineWordCounts)

    return ExtractedLineSamples, formattedTextLines, drawCanvas


# =====================================================================
# Regeneration Driver (formerly regenerate_all_labels.py)
# =====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[1] / "Data" / "Datasets" / "IAMpages671"
DEFAULT_REVIEW_DIR = SCRIPT_DIR / "label_regeneration_review"

# Author folder "150" holds the hand-verified labels copied in from
# IAMpages10/150 -- never regenerate them.
PROTECTED_AUTHORS = {"150"}


def resolve_dataset_root(root_dir):
    data_dir = root_dir / "data"
    return data_dir if data_dir.exists() and any(p.is_dir() for p in data_dir.iterdir()) else root_dir


def write_label_lines(label_path, text_lines):
    with open(label_path, "w", encoding="utf-8") as f:
        for line in text_lines:
            f.write(f"{line}\n")


def regenerate_page(img_path, dry_run):
    img_path = Path(img_path)
    label_path = img_path.with_name(img_path.stem + "_labels.txt")
    old_label_lines = ReadLabelLines(str(img_path))  # captured before any deletion

    try:
        if PageBoundsAreUncertain(str(img_path)):
            if not dry_run and label_path.exists():
                label_path.unlink()
            return {"status": "boundary_uncertain", "old_label_lines": old_label_lines}
    except Exception as e:
        return {"status": "error", "detail": str(e), "old_label_lines": old_label_lines}

    try:
        avg_conf, n_words_header = HeaderOcrConfidence(str(img_path))
    except Exception as e:
        return {"status": "error", "detail": str(e), "old_label_lines": old_label_lines}

    if avg_conf < MIN_AVG_CONF or n_words_header < MIN_WORD_COUNT:
        if not dry_run and label_path.exists():
            label_path.unlink()
        return {
            "status": "low_confidence",
            "avg_conf": avg_conf,
            "n_words": n_words_header,
            "old_label_lines": old_label_lines,
        }

    try:
        samples, text_lines, preview = ExtractLinePatches(str(img_path), expectedLineCount=None, labelLines=None)
    except Exception as e:
        return {"status": "error", "detail": str(e), "old_label_lines": old_label_lines}

    non_empty = [t for t in text_lines if t.strip()]
    if not non_empty:
        if not dry_run and label_path.exists():
            label_path.unlink()
        return {"status": "empty", "old_label_lines": old_label_lines}

    if not dry_run:
        if label_path.exists():
            label_path.unlink()
        write_label_lines(label_path, text_lines)

    return {
        "status": "written",
        "n_lines": len(text_lines),
        "text_lines": text_lines,
        "preview": preview,
        "old_label_lines": old_label_lines,
    }


def save_review_bundle(author_id, img_path, result, review_dir):
    page_name = Path(img_path).stem
    base = review_dir / f"{author_id}_{page_name}"
    result["preview"].save(base.with_suffix(".png"))

    with open(base.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write(f"Page: {author_id}/{Path(img_path).name}\n")
        f.write(f"Lines detected (NEW): {result['n_lines']}\n\n")
        for idx, line in enumerate(result["text_lines"]):
            f.write(f"{idx}: {line}\n")

        if result["old_label_lines"]:
            f.write(f"\n--- OLD labels (before this regeneration), for comparison ---\n")
            for idx, line in enumerate(result["old_label_lines"]):
                f.write(f"{idx}: {line}\n")
        else:
            f.write("\n--- No OLD label existed for this page before regeneration ---\n")


def main():
    parser = argparse.ArgumentParser(description="Regenerate _labels.txt for every page with the word-count splitter.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--review-every", type=int, default=40,
                         help="Save a verification bundle every Nth page that gets a fresh label written (default: 40).")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--dry-run", action="store_true",
                         help="Report what would happen without deleting or writing any label files.")
    args = parser.parse_args()

    root_dir = resolve_dataset_root(Path(args.data_dir))
    author_folders = sorted(p.name for p in root_dir.iterdir() if p.is_dir())

    review_dir = Path(args.review_dir)
    if not args.dry_run:
        review_dir.mkdir(parents=True, exist_ok=True)

    mode = "DRY RUN -- no files will be touched" if args.dry_run else "LIVE -- deleting and writing real label files"
    print(f"[Regenerate] Mode: {mode}")
    print(f"[Regenerate] {len(author_folders)} author folder(s) under {root_dir}.")
    print(f"[Regenerate] Protected (never touched): {sorted(PROTECTED_AUTHORS)}\n")

    stats = {"written": 0, "boundary_uncertain": 0, "low_confidence": 0, "empty": 0, "error": 0}
    protected_page_count = 0
    total_pages = 0
    written_counter = 0
    line_count_changed = 0

    for author_id in author_folders:
        author_dir = root_dir / author_id
        page_paths = sorted(author_dir.glob("*.png"))

        if author_id in PROTECTED_AUTHORS:
            protected_page_count += len(page_paths)
            continue

        for img_path in page_paths:
            total_pages += 1
            result = regenerate_page(img_path, args.dry_run)
            stats[result["status"]] = stats.get(result["status"], 0) + 1

            if result["status"] == "written":
                old_count = len(result["old_label_lines"]) if result["old_label_lines"] else None
                if old_count is not None and old_count != result["n_lines"]:
                    line_count_changed += 1

                written_counter += 1
                if written_counter % args.review_every == 0:
                    if not args.dry_run:
                        save_review_bundle(author_id, img_path, result, review_dir)
                    print(f"  [Review bundle #{written_counter // args.review_every}] {author_id}/{img_path.name} "
                          f"({result['n_lines']} lines, old had {old_count if old_count is not None else 'no label'})")

            if total_pages % 200 == 0:
                print(f"  ... {total_pages} pages processed so far "
                      f"(written={stats.get('written', 0)}, boundary_uncertain={stats.get('boundary_uncertain', 0)}, "
                      f"low_conf={stats.get('low_confidence', 0)}, empty={stats.get('empty', 0)}, "
                      f"error={stats.get('error', 0)})")

    print("\n" + "=" * 75)
    print(f"[Regenerate] Total non-protected pages processed:     {total_pages}")
    print(f"[Regenerate] Labels written:                          {stats.get('written', 0)}")
    print(f"[Regenerate] Skipped (header/handwriting boundary uncertain -- see note below): "
          f"{stats.get('boundary_uncertain', 0)}")
    print(f"[Regenerate] Skipped (low OCR confidence on header):  {stats.get('low_confidence', 0)}")
    print(f"[Regenerate] Skipped (no usable text detected):       {stats.get('empty', 0)}")
    print(f"[Regenerate] Errors:                                  {stats.get('error', 0)}")
    print(f"[Regenerate] Protected pages left untouched (author 150): {protected_page_count}")
    print(f"[Regenerate] Pages where new line count differs from old label's line count: "
          f"{line_count_changed}/{stats.get('written', 0)}")

    if stats.get("boundary_uncertain", 0) > 0:
        print(f"\n[Regenerate] Note on the {stats['boundary_uncertain']} boundary-uncertain skip(s): these are "
              f"pages where no reliable rule line was found separating the printed header from the "
              f"handwriting, so the region PageBoundsAreUncertain() would extract risks starting inside "
              f"the printed text instead of below it. Left unlabeled on purpose rather than risk labeling "
              f"printed text as handwriting -- these need a look (see FindRuleLineClusters/DetectPageBounds "
              f"above if you want to loosen or improve that detection for your form layout).")

    if not args.dry_run:
        print(f"\n[Regenerate] Verification bundles saved to:\n  {review_dir}")
        print("[Regenerate] IMPORTANT: line_cache_raw/ was built from the OLD labels and is now stale. "
              "Delete it (or run training with force_rebuild=True) before your next training run.")
    else:
        print("\n[Regenerate] Dry run only -- no files were changed. Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
