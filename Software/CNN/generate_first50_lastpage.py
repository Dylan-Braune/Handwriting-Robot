"""
PRE-CONFIGURED, single-file, standalone script: processes the first 50
author folders in your IAMpages671 dataset, and for each one saves ONLY
its LAST page (sorted order) as:
  <author>_<page>_preview.png  -- full page image with numbered red boxes
                                   around each detected handwriting line
  <author>_<page>_labels.txt   -- one line of TEXT per detected line
                                   (same order as the numbered boxes),
                                   generated using the width-based DP
                                   alignment method (proven on folder 150
                                   at ~89% exact-line match, no LLM, no
                                   handwriting OCR)

Just run it -- no arguments needed. It uses no imports of any other local
file in this repo, only numpy/PIL/pytesseract (need installing on this
machine, same as any other script here).

READ-ONLY against your real dataset -- nothing here overwrites or deletes
any existing _labels.txt in IAMpages671. Output goes to a separate review
folder (generated_boxes_and_labels/ next to this script) so you can
inspect results before deciding whether to run a full regeneration.

NOTE ON TIMING: each page needs a real OCR pass on its printed header,
roughly 3-6 seconds per page. Since this only processes 1 page per author
(the last one) for 50 authors, expect roughly 3-5 minutes total -- much
faster than processing every page in every folder.

NOTE ON CACHING FOR TRAINING: this script does not touch your training
cache (line_cache_raw/). If you go on to replace real _labels.txt files
with regenerated ones, the training script's cache tracks pages by "was
this page already cached", not by label content -- so it will NOT notice
the label text changed and will keep training on the OLD labels paired
with the old cached crops. You will need to delete line_cache_raw/ (or
pass force_rebuild=True) before your next training run whenever labels
change, even though the actual image preprocessing/cropping logic itself
is unchanged.

Usage:
    python generate_first50_lastpage.py
        (does exactly the above: first 50 authors, last page of each)
    python generate_first50_lastpage.py --num-folders 10
    python generate_first50_lastpage.py --all-pages
        (process every page of each author, not just the last)
    python generate_first50_lastpage.py --out my_review_folder
    python generate_first50_lastpage.py --include-150
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


def DetectPageBounds(binaryImg, rawGrayscaleArray=None):
    h, w = binaryImg.shape
    clusters = FindRuleLineClusters(binaryImg, rawGrayscaleArray)
    topCandidates = [c for c in clusters if h * 0.12 < c < h * 0.40]
    botCandidates = [c for c in clusters if h * 0.55 < c < h * 0.92]
    topY = topCandidates[-1] + 8 if topCandidates else int(h * 0.12)
    botY = botCandidates[0] - 8 if botCandidates else int(h * 0.80)
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
# Approximate per-character widths (for word-width estimation and for
# choosing exactly where to split a word that gets hyphenated across a
# line break)
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

    # Word-run detection, page-calibrated gap threshold.
    perLineRawRuns = []
    allGaps = []
    lineBoxes = []  # (absX1, absY1, absX2, absY2) per line, for the preview
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

    # Build the annotated preview image.
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
# Driver -- pre-configured: first 50 authors, last page of each
# =====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[1] / "Data" / "Datasets" / "IAMpages671" / "data"
DEFAULT_OUT_DIR = SCRIPT_DIR / "generated_boxes_and_labels"


def main():
    parser = argparse.ArgumentParser(description="Generate bounding-box preview images + labels.txt for the first N author folders (last page of each by default).")
    parser.add_argument("--num-folders", type=int, default=50,
                         help="How many author folders to process (default: 50).")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--include-150", action="store_true",
                         help="Also process author folder 150 (hand-verified reference -- skipped by default).")
    parser.add_argument("--all-pages", action="store_true",
                         help="Process EVERY page of each author folder, instead of just the last one (default: last page only).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[Error] Data dir not found: {data_dir}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    author_folders = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    if not args.include_150 and "150" in author_folders:
        author_folders.remove("150")

    selected = author_folders[:args.num_folders]
    last_page_only = not args.all_pages

    print(f"[Generate] Processing {len(selected)} author folder(s) "
          f"(out of {len(author_folders)} available)")
    print(f"[Generate] Mode: {'LAST PAGE ONLY per author' if last_page_only else 'ALL PAGES per author'}")
    print(f"[Generate] Output going to: {out_dir}")
    print("[Generate] READ-ONLY -- no files in your real dataset are touched.\n")

    total_pages, total_lines, errors = 0, 0, 0

    for author_id in selected:
        author_dir = data_dir / author_id
        page_paths = sorted(author_dir.glob("*.png"))
        if last_page_only and page_paths:
            page_paths = [page_paths[-1]]
        for img_path in page_paths:
            page_name = img_path.stem
            try:
                preview, textLines = ProcessPage(str(img_path))
            except Exception as e:
                print(f"  [ERROR] {author_id}/{page_name}: {e}")
                errors += 1
                continue

            preview_path = out_dir / f"{author_id}_{page_name}_preview.png"
            labels_path = out_dir / f"{author_id}_{page_name}_labels.txt"
            preview.save(preview_path)
            with open(labels_path, "w", encoding="utf-8") as f:
                for line in textLines:
                    f.write(f"{line}\n")

            total_pages += 1
            total_lines += len(textLines)
            print(f"  [OK] {author_id}/{page_name}: {len(textLines)} line(s) -> "
                  f"{preview_path.name} + {labels_path.name}")

    print(f"\n[Generate] Done. {total_pages} page(s) processed, {total_lines} line(s) total, {errors} error(s).")
    print(f"[Generate] Output folder: {out_dir}")
    print("\n[Generate] Reminder: if you go on to regenerate real _labels.txt files in your dataset, "
          "delete line_cache_raw/ (or pass force_rebuild=True) before your next training run -- the "
          "training cache tracks pages by 'already processed', not by label content, so it won't notice "
          "labels changed and will silently keep training on stale ones.")


if __name__ == "__main__":
    main()
