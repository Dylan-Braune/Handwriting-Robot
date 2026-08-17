"""
WRITES to your real dataset. Regenerates _labels.txt for a number of
author folders YOU choose (e.g. the first 50), using the width-based DP
alignment method (proven on folder 150 at ~89% exact-line match, no LLM,
no handwriting OCR -- only OCR of the clean printed header, same as
before).

This is the write-enabled counterpart to generate_first50_lastpage.py /
generate_boxes_and_labels.py, which only ever wrote to a review folder.
THIS script deletes and replaces real _labels.txt files in your dataset.

NO SEPARATE SAFETY GATES: two earlier candidate gates (PageBoundsAreUncertain
via HeaderGapIsSafe, and a header-OCR-confidence check) were tried and both
turned out to be broken relative to the pipeline that actually generates
labels -- they were checking a DIFFERENT, cruder crop/heuristic than the one
ProcessPage really uses, and ended up rejecting pages already proven correct
(including all 10 hand-verified folder 150 pages). Both were removed. The
real safety net is simpler and matches what was actually validated: a page
only gets NO label written (old one deleted, nothing new written) if
ProcessPage itself produces no usable text at all ("empty" status below).

NOT PROTECTED: every author folder in your selection is regenerated,
including "150" if it falls within --num-folders -- by explicit choice,
since the alignment method has already been validated against 150's
hand-verified labels and treating it as untouchable was no longer needed.

NO ORIGINAL LABEL = SKIPPED, NOT REGENERATED: if a page never had a
_labels.txt to begin with (the old pipeline didn't write one when its
preprocessing didn't work out for that page), this script leaves it
alone -- no attempt is made to invent a label for it. That page simply
stays excluded from training, same as before.

Every N pages that get a fresh label written (default 40), a
verification bundle (annotated preview PNG + old-vs-new text) is saved
to a review folder, same as before.

Standalone: no local imports, only numpy/PIL/pytesseract (need
installing, same as your other scripts).

Usage:
    python regenerate_labels_with_alignment.py
        (prompts: "How many author folders? ", then asks you to
         type "yes" to confirm before writing anything)
    python regenerate_labels_with_alignment.py --num-folders 50
    python regenerate_labels_with_alignment.py --num-folders 50 --yes
        (skips the confirmation prompt -- for unattended runs)
    python regenerate_labels_with_alignment.py --num-folders 50 --dry-run
        (report only, touches nothing)
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
# Shared low-level pieces
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


# =====================================================================
# Word-run detection with a page-calibrated (Otsu on gap sizes) threshold
# =====================================================================
def GetRawInkRuns(inkCols, minRunWidth=3):
    if len(inkCols) == 0:
        return []
    gaps = np.where(np.diff(inkCols) > 1)[0]
    starts = np.concatenate(([0], gaps + 1))
    ends = np.concatenate((gaps, [len(inkCols) - 1]))
    rawRuns = [(int(inkCols[s]), int(inkCols[e])) for s, e in zip(starts, ends)]
    return [(s, e) for s, e in rawRuns if (e - s + 1) >= minRunWidth]


def OtsuThreshold1D(values, bins=48):
    if len(values) < 2:
        return max(values) if values else 1
    vmin, vmax = min(values), max(values)
    if vmax <= vmin:
        return vmax
    hist, edges = np.histogram(values, bins=bins, range=(vmin, vmax + 1))
    total = hist.sum()
    sum1 = np.dot(np.arange(bins), hist)
    currentMax, bestBin, sumB, wB = 0, bins // 2, 0, 0
    for t in range(bins):
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
            bestBin = t
    return float(edges[bestBin + 1])


def MergeRunsByThreshold(rawRuns, wordGapThreshold):
    if not rawRuns:
        return []
    wordRuns = [rawRuns[0]]
    for s, e in rawRuns[1:]:
        prevS, prevE = wordRuns[-1]
        if s - prevE <= wordGapThreshold:
            wordRuns[-1] = (prevS, e)
        else:
            wordRuns.append((s, e))
    return wordRuns


# =====================================================================
# Approximate per-character widths (word-width estimation + hyphen split)
# =====================================================================
_NARROW = {'i': 0.28, 'j': 0.32, 'l': 0.30, 'f': 0.38, 't': 0.38, 'r': 0.42}
_PUNCT_WIDTHS = {
    '.': 0.28, ',': 0.28, "'": 0.24, '-': 0.35, '"': 0.4,
    '\u2019': 0.24, '\u2018': 0.24, '\u201c': 0.4, '\u201d': 0.4,
    '(': 0.32, ')': 0.32, ';': 0.28, ':': 0.28,
}
_WIDE_LOWER = {'m': 0.92, 'w': 0.88}
_WIDE_UPPER = {'M': 1.0, 'W': 1.0}


def CharWidth(ch):
    if ch in _NARROW:
        return _NARROW[ch]
    if ch in _PUNCT_WIDTHS:
        return _PUNCT_WIDTHS[ch]
    if ch.isdigit():
        return 0.6
    if ch.isupper():
        return _WIDE_UPPER.get(ch, 0.72)
    if ch.islower():
        return _WIDE_LOWER.get(ch, 0.58)
    return 0.5


def WordWidthUnits(word):
    return sum(CharWidth(c) for c in word) if word else 0.0


def SplitWordByWidthFraction(word, frac):
    if len(word) <= 1:
        return word, ""
    widths = [CharWidth(c) for c in word]
    total = sum(widths)
    target = total * frac
    cumAtIdx = {}
    cum = 0.0
    for idx in range(1, len(word)):
        cum += widths[idx - 1]
        cumAtIdx[idx] = cum
    bestIdx, bestDiff = 1, None
    for idx in range(1, len(word)):
        diff = abs(cumAtIdx[idx] - target)
        if bestDiff is None or diff < bestDiff:
            bestDiff, bestIdx = diff, idx
    hyphenPositions = [idx + 1 for idx, c in enumerate(word[:-1]) if c == '-']
    tolerance = bestDiff + total * 0.15
    for hp in hyphenPositions:
        if hp in cumAtIdx and abs(cumAtIdx[hp] - target) <= tolerance:
            bestIdx = hp
            break
    bestIdx = max(1, min(len(word) - 1, bestIdx))
    return word[:bestIdx], word[bestIdx:]


# =====================================================================
# Forced alignment -- known printed-header words <-> detected word-runs
# =====================================================================
def AlignWordsToLines(linesRuns, knownWords, mergePenalty=0.5):
    runWidths = []
    runLineIdx = []
    for li, widths in enumerate(linesRuns):
        for w_ in widths:
            runWidths.append(w_)
            runLineIdx.append(li)

    M, N = len(runWidths), len(knownWords)
    if M == 0 or N == 0:
        return ["" for _ in linesRuns]

    charLens = [max(0.1, WordWidthUnits(w_)) for w_ in knownWords]
    totalCharUnits = sum(charLens)
    totalRunWidth = sum(runWidths)
    avgPxPerChar = totalRunWidth / totalCharUnits if totalCharUnits > 0 else 1.0
    expectedWidths = [c * avgPxPerChar for c in charLens]

    def cost(actual, expected):
        return ((actual - expected) / (expected + 1.0)) ** 2

    INF = float('inf')
    dp = [[INF] * (N + 1) for _ in range(M + 1)]
    choice = [[None] * (N + 1) for _ in range(M + 1)]
    dp[0][0] = 0.0

    for i in range(0, M + 1):
        for j in range(0, N + 1):
            if i == 0 and j == 0:
                continue
            best, bestChoice = INF, None

            if i >= 1 and j >= 1:
                c = dp[i - 1][j - 1] + cost(runWidths[i - 1], expectedWidths[j - 1])
                if c < best:
                    best, bestChoice = c, ('1:1', 1, 1)

            if i >= 2 and j >= 1:
                mergedWidth = runWidths[i - 2] + runWidths[i - 1]
                c = dp[i - 2][j - 1] + cost(mergedWidth, expectedWidths[j - 1]) + mergePenalty
                if c < best:
                    best, bestChoice = c, ('2:1', 2, 1)

            if i >= 1 and j >= 2:
                mergedExpected = expectedWidths[j - 2] + expectedWidths[j - 1]
                c = dp[i - 1][j - 2] + cost(runWidths[i - 1], mergedExpected) + mergePenalty
                if c < best:
                    best, bestChoice = c, ('1:2', 1, 2)

            dp[i][j] = best
            choice[i][j] = bestChoice

    if dp[M][N] == INF:
        buckets = [[] for _ in linesRuns]
        perLine = max(1, N // max(1, len(linesRuns)))
        wi = 0
        for li in range(len(linesRuns)):
            take = perLine if li < len(linesRuns) - 1 else N - wi
            buckets[li] = knownWords[wi:wi + take]
            wi += take
        return [" ".join(b) for b in buckets]

    steps = []
    i, j = M, N
    while i > 0 or j > 0:
        ch = choice[i][j]
        kind, di, dj = ch
        runsUsed = list(range(i - di, i))
        wordsUsed = list(range(j - dj, j))
        steps.append((kind, runsUsed, wordsUsed))
        i -= di
        j -= dj
    steps.reverse()

    buckets = [[] for _ in linesRuns]
    for kind, runsUsed, wordsUsed in steps:
        if kind == '1:1':
            r0, w0 = runsUsed[0], wordsUsed[0]
            buckets[runLineIdx[r0]].append(knownWords[w0])
        elif kind == '1:2':
            r0 = runsUsed[0]
            for wIdx in wordsUsed:
                buckets[runLineIdx[r0]].append(knownWords[wIdx])
        elif kind == '2:1':
            r0, r1 = runsUsed
            word = knownWords[wordsUsed[0]]
            line0, line1 = runLineIdx[r0], runLineIdx[r1]
            if line0 == line1:
                buckets[line0].append(word)
            else:
                w0w, w1w = runWidths[r0], runWidths[r1]
                frac = w0w / (w0w + w1w) if (w0w + w1w) > 0 else 0.5
                firstPart, secondPart = SplitWordByWidthFraction(word, frac)
                if not firstPart.endswith("-"):
                    firstPart = firstPart + "-"
                buckets[line0].append(firstPart)
                buckets[line1].append(secondPart)

    return [" ".join(words) for words in buckets]


# =====================================================================
# Full page processing: line boxes (preview image) + alignment labels
# =====================================================================
def ProcessPage(imgPath):
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

    perLineRawRuns = []
    allGaps = []
    lineBoxes = []
    for (s, e) in mergedLines:
        absY1 = handwritingStartY + max(0, s - 5)
        absY2 = handwritingStartY + min(regionH, e + 5)

        lineBinary = hwRegion[max(0, s - 3):min(regionH, e + 3), :]
        colSum = np.sum(lineBinary > 0, axis=0)
        inkCols = np.where(colSum > 0)[0]
        rawRuns = GetRawInkRuns(inkCols)
        perLineRawRuns.append(rawRuns)
        for i in range(1, len(rawRuns)):
            allGaps.append(rawRuns[i][0] - rawRuns[i - 1][1])

        if len(inkCols) == 0:
            lineBoxes.append(None)
            continue
        absX1 = max(0, inkCols[0] - 8)
        absX2 = min(w, inkCols[-1] + 8)
        lineBoxes.append((absX1, absY1, absX2, absY2))

    wordGapThreshold = OtsuThreshold1D(allGaps) if allGaps else 1

    linesRuns = []
    for rawRuns in perLineRawRuns:
        wordRuns = MergeRunsByThreshold(rawRuns, wordGapThreshold)
        widths = [(r[1] - r[0] + 1) for r in wordRuns]
        linesRuns.append(widths)

    textLines = AlignWordsToLines(linesRuns, printedWords)

    drawCanvas = rawImage.convert('RGB')
    drawObj = ImageDraw.Draw(drawCanvas)
    drawObj.line([(0, topY), (w, topY)], fill=(0, 0, 255), width=2)
    drawObj.line([(0, handwritingStartY), (w, handwritingStartY)], fill=(0, 200, 255), width=1)
    drawObj.line([(0, botY), (w, botY)], fill=(0, 0, 255), width=2)
    for idx, box in enumerate(lineBoxes):
        if box is None:
            continue
        absX1, absY1, absX2, absY2 = box
        drawObj.rectangle([absX1, absY1, absX2, absY2], outline=(255, 0, 0), width=2)
        drawObj.text((absX1, max(0, absY1 - 14)), str(idx), fill=(0, 128, 0))

    return drawCanvas, textLines


# =====================================================================
# Write-enabled regeneration driver
# =====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
NOGIT_DIR = SCRIPT_DIR / "NOGIT"
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[1] / "Data" / "Datasets" / "IAMpages671"
DEFAULT_REVIEW_DIR = NOGIT_DIR / "label_regeneration_review"


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

    # If this page never had a label to begin with, leave it alone. The old
    # pipeline simply never wrote a label for pages where preprocessing
    # didn't work out well, so "no original label" == "this page is meant
    # to be excluded from training" -- not something for us to attempt and
    # guess at. Skip it entirely, don't touch the (nonexistent) file.
    if not label_path.exists():
        return {"status": "skipped_no_original_label", "old_label_lines": []}

    old_label_lines = ReadLabelLines(str(img_path))

    # NOTE: two earlier gates were dropped after testing found them both
    # broken relative to the pipeline actually used to generate labels:
    #   - PageBoundsAreUncertain (HeaderGapIsSafe) misread the rule line's
    #     own ~12px of ink thickness as "header text bleeding through",
    #     flagging every single page -- including all 10 hand-verified
    #     folder 150 pages already known to be correct -- as uncertain.
    #   - HeaderOcrConfidence scored a DIFFERENT, cruder, fixed-percentage
    #     header crop than the one ProcessPage/ExtractPrintedGroundTruth
    #     actually uses (which is header-boundary-aware). On c03-000a.png
    #     (folder 150, verified perfect) the confidence gate's crop scored
    #     74% and got rejected, while the real crop cleanly read all 56
    #     words correctly -- the gate was measuring the wrong thing.
    # ProcessPage below is the same pipeline already proven page-by-page
    # against folder 150 and spot-checked on other authors; the "empty"
    # check right after it is the real safety net for a page that
    # genuinely can't be read.
    try:
        preview, text_lines = ProcessPage(str(img_path))
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
    parser = argparse.ArgumentParser(description="Regenerate _labels.txt (alignment method) for a chosen number of author folders. WRITES to your real dataset.")
    parser.add_argument("--num-folders", type=int, default=None,
                         help="How many author folders to process. If omitted, you'll be asked interactively.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--review-every", type=int, default=40,
                         help="Save a verification bundle every Nth page that gets a fresh label written (default: 40).")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--dry-run", action="store_true",
                         help="Report what would happen without deleting or writing any label files.")
    parser.add_argument("--yes", action="store_true",
                         help="Skip the 'type yes to continue' confirmation prompt (for unattended runs).")
    args = parser.parse_args()

    num_folders = args.num_folders
    if num_folders is None:
        while True:
            raw = input("How many author folders would you like to regenerate? ").strip()
            if raw.isdigit() and int(raw) > 0:
                num_folders = int(raw)
                break
            print("Please enter a positive whole number.")

    root_dir = resolve_dataset_root(Path(args.data_dir))
    author_folders = sorted(p.name for p in root_dir.iterdir() if p.is_dir())
    selected = author_folders[:num_folders]

    review_dir = Path(args.review_dir)

    print(f"[Regenerate] {'DRY RUN -- no files will be touched' if args.dry_run else 'LIVE -- will overwrite real label files'}")
    print(f"[Regenerate] {len(selected)} author folder(s) selected (out of {len(author_folders)} available): {selected}")
    print("[Regenerate] No folders are protected -- author 150 will be regenerated too if it falls within your selection.")

    if not args.dry_run and not args.yes:
        total_pages = sum(len(list((root_dir / a).glob('*.png'))) for a in selected)
        print(f"\n[Regenerate] This will DELETE and REWRITE _labels.txt for up to {total_pages} page(s) "
              f"across {len(selected)} author folder(s) in your real dataset:\n  {root_dir}")
        confirm = input("Type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("[Regenerate] Not confirmed -- exiting without touching anything.")
            return

    if not args.dry_run:
        review_dir.mkdir(parents=True, exist_ok=True)

    stats = {"written": 0, "empty": 0, "error": 0, "skipped_no_original_label": 0}
    total_pages, written_counter, line_count_changed = 0, 0, 0

    for author_id in selected:
        author_dir = root_dir / author_id
        page_paths = sorted(author_dir.glob("*.png"))

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

            if total_pages % 100 == 0:
                print(f"  ... {total_pages} pages processed so far "
                      f"(written={stats.get('written', 0)}, empty={stats.get('empty', 0)}, "
                      f"error={stats.get('error', 0)}, "
                      f"no_original_label={stats.get('skipped_no_original_label', 0)})")

    print("\n" + "=" * 75)
    print(f"[Regenerate] Total pages processed:                   {total_pages}")
    print(f"[Regenerate] Labels written:                          {stats.get('written', 0)}")
    print(f"[Regenerate] Skipped (no usable text detected):       {stats.get('empty', 0)}")
    print(f"[Regenerate] Skipped (no original label -- untouched):{stats.get('skipped_no_original_label', 0)}")
    print(f"[Regenerate] Errors:                                  {stats.get('error', 0)}")
    print(f"[Regenerate] Pages where new line count differs from old label's line count: "
          f"{line_count_changed}/{stats.get('written', 0)}")

    if not args.dry_run:
        print(f"\n[Regenerate] Verification bundles saved to:\n  {review_dir}")
        print("[Regenerate] IMPORTANT: line_image_cache/ (if it already exists) does not need rebuilding "
              "for label text changes -- only IAMLineDatasetRaw's fresh re-read of _labels.txt does, which "
              "happens automatically on your next training run. See train_paper_cnn_bilstm_ctc.py.")
    else:
        print("\n[Regenerate] Dry run only -- no files were changed. Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
