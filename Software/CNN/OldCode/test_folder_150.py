"""
STEP 1 of the label-regeneration rollout: run the SAME preprocessing +
label-generation pipeline as regenerate_all_labels_standalone.py, but only
on author folder "150" (the hand-verified set), and write NOTHING back to
any _labels.txt. Purely for visual inspection.

For every page in folder 150 this produces, in a fresh output folder:
  <page>_preview.png          -- full page, numbered red boxes around each
                                  detected line (same boxes the real
                                  pipeline would crop)
  <page>/line_00.png ...       -- each individual detected line, cropped
                                  and preprocessed exactly like a real
                                  training sample would be
  <page>/line_00.txt ...       -- for that same line index: the label text
                                  the pipeline just GENERATED (from OCR +
                                  word-count splitting), and underneath it,
                                  the label text that's ALREADY SAVED in
                                  folder 150's real _labels.txt (the hand-
                                  verified one) -- so you can eyeball
                                  line_00.png against both and see whether
                                  the generation pipeline agrees with the
                                  hand-verified ground truth.

Nothing here touches any real label file -- fully read-only against your
dataset.

Standalone: no local imports, just numpy/PIL/pytesseract (needs installing,
same as any other script here).

Usage:
    python test_folder_150.py
    python test_folder_150.py --data-dir "C:\\path\\to\\IAMpages671\\data\\150"
    python test_folder_150.py --out my_folder150_check
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
    h, w = binaryImg.shape
    clusters = FindRuleLineClusters(binaryImg, rawGrayscaleArray)

    topCandidates = [c for c in clusters if h * 0.12 < c < h * 0.40]
    botCandidates = [c for c in clusters if h * 0.55 < c < h * 0.92]

    topY = topCandidates[-1] + 8 if topCandidates else int(h * 0.12)
    botY = botCandidates[0] - 8 if botCandidates else int(h * 0.80)

    if return_confidence:
        return topY, botY, len(topCandidates) == 0, len(botCandidates) == 0
    return topY, botY


def DetectHeaderTopY(binaryImg, rawGrayscaleArray, topY):
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


# =====================================================================
# Word-Count Estimation (from ink, not from assumed character width)
# =====================================================================
def CountWordsFromInkColumns(inkCols, lineHeight, minGapFactor=0.6, minRunWidth=3):
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
# Folder-150 visual check driver
# =====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[1] / "Data" / "Datasets" / "IAMpages671" / "data" / "150"
DEFAULT_OUT_DIR = SCRIPT_DIR / "folder150_visual_check"


def main():
    parser = argparse.ArgumentParser(description="Visual line/label check for author folder 150 -- read-only.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                         help="Folder containing folder 150's page PNGs + _labels.txt (default: IAMpages671/data/150).")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[Error] Data dir not found: {data_dir}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    page_paths = sorted(data_dir.glob("*.png"))
    print(f"[Folder150 Check] Found {len(page_paths)} page(s) in {data_dir}")
    print(f"[Folder150 Check] Output going to {out_dir}\n")
    print("[Folder150 Check] READ-ONLY -- no _labels.txt files are written or deleted.\n")

    for img_path in page_paths:
        page_name = img_path.stem
        old_label_lines = ReadLabelLines(str(img_path))

        try:
            # labelLines=None forces the real generation path (OCR + word-
            # count splitting) instead of just echoing the saved label back.
            samples, generated_text_lines, preview = ExtractLinePatches(
                str(img_path), expectedLineCount=None, labelLines=None
            )
        except Exception as e:
            print(f"  [ERROR] {page_name}: {e}")
            continue

        preview_path = out_dir / f"{page_name}_preview.png"
        preview.save(preview_path)

        page_dir = out_dir / page_name
        page_dir.mkdir(parents=True, exist_ok=True)

        for line_idx, sample in enumerate(samples):
            line_png = page_dir / f"line_{line_idx:02d}.png"
            line_txt = page_dir / f"line_{line_idx:02d}.txt"
            sample['processed_patch'].save(line_png)

            new_text = generated_text_lines[line_idx] if line_idx < len(generated_text_lines) else "<no text generated>"
            old_text = old_label_lines[line_idx] if line_idx < len(old_label_lines) else "<no saved label at this line index>"

            with open(line_txt, "w", encoding="utf-8") as f:
                f.write(f"GENERATED: {new_text}\n")
                f.write(f"SAVED (hand-verified 150 label): {old_text}\n")

        match = "SAME COUNT" if len(samples) == len(old_label_lines) else "DIFFERENT COUNT"
        print(f"  [{match}] {page_name}: generated {len(samples)} line(s) vs {len(old_label_lines)} saved -- "
              f"{preview_path.name} + {len(samples)} line PNG/TXT pair(s) in {page_dir.name}/")

    print(f"\n[Folder150 Check] Done. Open {out_dir} -- start with a page's _preview.png (numbered boxes), "
          f"then open that page's folder and compare line_NN.png against line_NN.txt (GENERATED vs SAVED).")


if __name__ == "__main__":
    main()
