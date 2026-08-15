"""
PROOF-OF-CONCEPT: no-LLM, width-based forced alignment between the known
printed-header words and each line's DETECTED WORD RUNS (ink blobs
separated by word-sized gaps -- same detection already used by
CountWordsFromInkColumns), instead of the old "estimate a word COUNT per
line, then hand out words proportionally" approach.

WHY THIS SHOULD BE BETTER THAN THE COUNT-BASED SPLITTER
---------------------------------------------------------
The old method (CountWordsFromInkColumns + SplitWordsIntoLines) throws away
information: it reduces each line down to a single integer (how many word
gaps it found), then distributes words across lines using ratios of those
integers. Two different lines that "look like 9 words" get treated
identically even if their actual ink widths are quite different.

This version keeps the actual per-run PIXEL WIDTHS (not just counts) and
aligns them against the known words' WIDTHS (estimated from character
count, calibrated per page) using a small dynamic-programming forced
alignment -- the same kind of technique used to align text to audio/ink
when the transcript is known but the segmentation isn't. It allows for
the run detector occasionally over-splitting a word (2 runs matched to 1
word) or under-splitting two touching words (1 run matched to 2 words),
which a pure per-line proportional split can't correct for at all.

NO OCR of handwriting, NO vision LLM calls anywhere in this script --
100% local, deterministic, and free. The only OCR involved is on the
clean PRINTED header text, which is reliable already.

WHAT THIS SCRIPT DOES
----------------------
Runs this alignment method on author folder 150 (the one hand-verified
folder), and directly compares its generated line text against the
hand-verified _labels.txt for every page -- exact line match count, plus
a word-level diff for any line that doesn't match exactly. Fully
read-only, writes no label files anywhere.

Usage:
    python test_folder_150_alignment.py
"""

import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
import pytesseract

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# =====================================================================
# Shared low-level pieces (same as FullLineBoxMaker.py)
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
# NEW: word-RUN detection that keeps widths, not just a count
# =====================================================================
def GetRawInkRuns(inkCols, minRunWidth=3):
    """Every contiguous ink blob (letter or touching letter-cluster),
    with NO word-gap judgment applied yet -- just "where is there ink."""
    if len(inkCols) == 0:
        return []
    gaps = np.where(np.diff(inkCols) > 1)[0]
    starts = np.concatenate(([0], gaps + 1))
    ends = np.concatenate((gaps, [len(inkCols) - 1]))
    rawRuns = [(int(inkCols[s]), int(inkCols[e])) for s, e in zip(starts, ends)]
    return [(s, e) for s, e in rawRuns if (e - s + 1) >= minRunWidth]


def OtsuThreshold1D(values, bins=48):
    """
    Same variance-maximizing idea as OtsuThreshold above, generalized to
    an arbitrary 1D array of numbers instead of a 0-255 pixel histogram --
    used here to find the natural cut point in a page's GAP-SIZE
    distribution between "small gap, same word" and "large gap, new
    word", rather than guessing that threshold as a fixed fraction of
    line height (which doesn't actually track letter/word spacing and is
    what caused the first version of this alignment attempt to collapse
    whole lines into a single run).
    """
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
# NEW: page-level line/run extraction (positions only -- no image
# cropping needed for this proof-of-concept, since the box/preprocessing
# side is already verified correct)
# =====================================================================
def ExtractLineRuns(imgPath):
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

    # Pass 1: collect RAW ink runs (letter/letter-cluster blobs, no word
    # judgment yet) per line, plus every gap between consecutive raw runs
    # within a line -- gaps are collected across the WHOLE PAGE so the
    # word/letter cut point is calibrated from this page's own handwriting
    # size, not a fixed assumption.
    perLineRawRuns = []
    allGaps = []
    for (s, e) in mergedLines:
        lineBinary = hwRegion[max(0, s - 3):min(regionH, e + 3), :]
        colSum = np.sum(lineBinary > 0, axis=0)
        inkCols = np.where(colSum > 0)[0]
        rawRuns = GetRawInkRuns(inkCols)
        perLineRawRuns.append(rawRuns)
        for i in range(1, len(rawRuns)):
            allGaps.append(rawRuns[i][0] - rawRuns[i - 1][1])

    wordGapThreshold = OtsuThreshold1D(allGaps) if allGaps else 1

    # Pass 2: merge each line's raw runs into word-shaped runs using that
    # page-calibrated threshold.
    linesRuns = []
    for rawRuns in perLineRawRuns:
        wordRuns = MergeRunsByThreshold(rawRuns, wordGapThreshold)
        widths = [(r[1] - r[0] + 1) for r in wordRuns]
        linesRuns.append(widths)

    return linesRuns, printedWords


# =====================================================================
# NEW: forced alignment -- known words <-> detected word-runs, by width
# =====================================================================
# Rough relative character advance-widths (narrow letters like i/l/t vs
# wide ones like m/w), used INSTEAD of raw character count so word-width
# estimates -- and, critically, where inside a word to place a hyphen
# split -- account for real letterform width differences rather than
# assuming every character occupies the same pixel width. Approximate,
# proportional-font-style values; good enough to be much closer than
# uniform per-character width without needing to read any actual ink.
_NARROW = {'i': 0.28, 'j': 0.32, 'l': 0.30, 'f': 0.38, 't': 0.38, 'r': 0.42}
_PUNCT_WIDTHS = {
    '.': 0.28, ',': 0.28, "'": 0.24, '-': 0.35, '"': 0.4,
    '’': 0.24, '‘': 0.24, '“': 0.4, '”': 0.4,
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
    """
    Finds the character index inside `word` whose CUMULATIVE relative
    character width best matches `frac` of the word's total width --
    used to decide exactly where a hyphen-split word should be broken,
    instead of just cutting at len(word) * frac (which is wrong whenever
    the word mixes narrow and wide letters, e.g. splits "insensitively"
    after 'insensit' instead of the correct 'insensi').
    """
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

    # If the word already contains a real hyphen (compound words like
    # "father-in-law"), that's the split point a writer would actually
    # use -- prefer it over the raw width estimate whenever it's
    # reasonably close to that estimate, since our width table is only
    # an approximation and compound-word hyphens are exact ground truth.
    hyphenPositions = [idx + 1 for idx, c in enumerate(word[:-1]) if c == '-']
    tolerance = bestDiff + total * 0.15
    for hp in hyphenPositions:
        if hp in cumAtIdx and abs(cumAtIdx[hp] - target) <= tolerance:
            bestIdx = hp
            break

    bestIdx = max(1, min(len(word) - 1, bestIdx))
    return word[:bestIdx], word[bestIdx:]


def AlignWordsToLines(linesRuns, knownWords, mergePenalty=0.5):
    """
    linesRuns: list (per line) of lists of run widths, e.g.
        [[42, 55, 30], [61, 22, 48, 19], ...]
    knownWords: flat ordered list of the known correct words for the
        whole page (from the printed header OCR).

    Returns a list of strings, one per line -- the words assigned to
    that line by the alignment, in order.
    """
    # Flatten runs into one global sequence, remembering which line each
    # run belongs to.
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
    # dp[i][j] = best cost aligning first i runs with first j words
    dp = [[INF] * (N + 1) for _ in range(M + 1)]
    choice = [[None] * (N + 1) for _ in range(M + 1)]
    dp[0][0] = 0.0

    for i in range(0, M + 1):
        for j in range(0, N + 1):
            if i == 0 and j == 0:
                continue
            best = INF
            bestChoice = None

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

    # Backtrack from (M, N), collecting the step sequence, then replay it
    # forward so words get appended to line buckets in correct reading
    # order (needed because a hyphen-split word contributes ITS TWO
    # HALVES to two different, correctly-ordered buckets -- see below).
    if dp[M][N] == INF:
        # No valid path at all (shouldn't normally happen) -- fall back to
        # an even split across lines rather than crashing.
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
            # One run matched two words (touching/under-segmented words,
            # e.g. "ATaste" detected as one blob) -- both words belong to
            # that single run's line.
            r0 = runsUsed[0]
            for wIdx in wordsUsed:
                buckets[runLineIdx[r0]].append(knownWords[wIdx])

        elif kind == '2:1':
            r0, r1 = runsUsed
            word = knownWords[wordsUsed[0]]
            line0, line1 = runLineIdx[r0], runLineIdx[r1]
            if line0 == line1:
                # Two runs merged into one word within the SAME line --
                # e.g. a word whose middle gap got over-detected as a
                # word-boundary. Just use the whole word.
                buckets[line0].append(word)
            else:
                # The two runs straddle a LINE BOUNDARY -- this is the
                # hyphen-at-line-end case (writer split the word across
                # two lines with a hyphen, e.g. "insensi-" / "tively").
                # Split the known word at the same proportion as the two
                # ink runs' widths, and append a hyphen to the first half
                # -- matching how a human transcriber writes it.
                w0w, w1w = runWidths[r0], runWidths[r1]
                frac = w0w / (w0w + w1w) if (w0w + w1w) > 0 else 0.5
                firstPart, secondPart = SplitWordByWidthFraction(word, frac)
                if not firstPart.endswith("-"):
                    firstPart = firstPart + "-"
                buckets[line0].append(firstPart)
                buckets[line1].append(secondPart)

    lines_out = [" ".join(words) for words in buckets]
    return lines_out


# =====================================================================
# Folder 150 proof run
# =====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[1] / "Data" / "Datasets" / "IAMpages671" / "data" / "150"


def main():
    data_dir = DEFAULT_DATA_DIR
    if not data_dir.exists():
        print(f"[Error] Data dir not found: {data_dir}")
        return

    page_paths = sorted(data_dir.glob("*.png"))
    print(f"[Alignment Proof] Testing width-based DP alignment (no LLM, no handwriting OCR) "
          f"on {len(page_paths)} hand-verified page(s) in {data_dir}\n")

    total_lines, exact_matches = 0, 0
    per_page_results = []

    for img_path in page_paths:
        page_name = img_path.stem
        hand_verified = ReadLabelLines(str(img_path))

        try:
            linesRuns, knownWords = ExtractLineRuns(str(img_path))
            aligned_lines = AlignWordsToLines(linesRuns, knownWords)
        except Exception as e:
            print(f"  [ERROR] {page_name}: {e}")
            continue

        n = min(len(aligned_lines), len(hand_verified))
        page_exact = 0
        mismatches = []
        for idx in range(n):
            gen = aligned_lines[idx].strip()
            ref = hand_verified[idx].strip()
            # Compare ignoring apostrophe-curl/quote-character differences
            # (OCR of the printed header often normalizes ' vs \u2019),
            # which isn't a real alignment error.
            norm = lambda s: re.sub(r"[\u2019`]", "'", s)
            if norm(gen) == norm(ref):
                page_exact += 1
            else:
                mismatches.append((idx, gen, ref))

        total_lines += len(hand_verified)
        exact_matches += page_exact
        per_page_results.append((page_name, page_exact, len(hand_verified), len(aligned_lines), mismatches))

        status = "ALL MATCH" if page_exact == len(hand_verified) and len(aligned_lines) == len(hand_verified) else "MISMATCH(ES)"
        print(f"  [{status}] {page_name}: {page_exact}/{len(hand_verified)} lines exact "
              f"(generated {len(aligned_lines)} lines, hand-verified had {len(hand_verified)})")
        for idx, gen, ref in mismatches:
            print(f"      line {idx}:")
            print(f"        ALIGNED: {gen}")
            print(f"        HAND:    {ref}")

    print("\n" + "=" * 75)
    print(f"[Alignment Proof] TOTAL: {exact_matches}/{total_lines} lines matched the hand-verified "
          f"label exactly ({100.0 * exact_matches / total_lines:.1f}%)")
    print("[Alignment Proof] Method used: width-based DP forced alignment between printed-header "
          "OCR words and detected ink word-runs. No handwriting OCR, no LLM/vision API calls.")


if __name__ == "__main__":
    main()
