import os
import glob
import re
import numpy as np
from PIL import Image, ImageDraw
import pytesseract

# Set your Tesseract OCR path if running on Windows
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

    # The printed divider lines can be too faint after Otsu on some scans,
    # so use the raw gray pixels when available and keep the binary fallback.
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
    return_confidence=False (default, unchanged for every existing caller):
    returns (topY, botY) as before.

    return_confidence=True: also returns whether topY/botY had to fall back
    to a fixed percentage-of-height GUESS rather than an actual detected
    rule line. When topY falls back, the top of the "handwriting region"
    this function hands to ExtractLinePatches can end up sitting inside the
    printed prompt text instead of below it -- the segmenter then treats
    rows of PRINTED text as a handwritten line, and the auto-labeler
    happily assigns real transcript words to it as if it were handwriting.
    See PageBoundsAreUncertain() below, which wraps this specifically to
    flag that risk before trusting a page's generated labels.
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
    immediately ABOVE topY (i.e. the tail end of what DetectPageBounds
    thinks is the printed-header side of the boundary) and confirms it's
    mostly blank -- consistent with topY actually sitting in the gap after
    the header's last line of print, not on top of it.

    This does not depend on trusting that a rule line was detected in the
    right place; it independently verifies the boundary makes sense against
    the actual ink in the image. If real printed-text ink is found packed
    right up against topY, that's a sign topY is sitting too high (i.e. on
    top of header content, not below it) regardless of how confident the
    rule-line detection was.
    """
    h = binaryImg.shape[0]
    zoneStart = max(0, topY - min_gap_px)
    checkZone = binaryImg[zoneStart:topY, :]
    if checkZone.size == 0:
        return True  # nothing to check against (topY is at/near the top of the page)

    inkFractionPerRow = np.mean(checkZone > 0, axis=1)
    return not np.any(inkFractionPerRow > max_ink_fraction)


def PageBoundsAreUncertain(imgPath):
    """
    True if this page's handwriting region should NOT be trusted for
    auto-labeling, for either of two independent reasons:

      1. The TOP boundary had to fall back to a fixed-percentage guess
         rather than a real detected rule line.
      2. Even where a rule line WAS detected, the rows immediately above
         topY aren't actually blank (HeaderGapIsSafe) -- meaning topY may
         be sitting on top of real printed content rather than below it,
         which a "we found A line" check alone can't catch.

    Either condition means the page is at meaningfully higher risk of the
    extracted "handwriting" region actually starting inside the printed
    header text. This does not fix the underlying detection; it flags pages
    a caller should not trust (e.g. skip auto-generating a label for that
    page rather than risk labeling printed text as if it were handwriting).
    """
    rawImage = Image.open(imgPath)
    grayArr = np.array(rawImage.convert('L'))
    binaryImg = OtsuThreshold(grayArr)
    topY, _botY, topWasFallback, _botWasFallback = DetectPageBounds(binaryImg, grayArr, return_confidence=True)

    if topWasFallback:
        return True

    return not HeaderGapIsSafe(binaryImg, topY)


def DetectHeaderTopY(binaryImg, rawGrayscaleArray, topY):
    """
    Finds the rule line bounding the TOP of the printed sentence-prompt box
    (the one separating it from the "Sentence Database" title text), rather
    than assuming a fixed page-height percentage. Some form templates place
    this box higher than others, and a fixed percentage clips the first
    printed line on those templates, garbling its OCR (confirmed on several
    IAMpages671 "c03" pages where lines 3+ OCR'd perfectly but lines 1-2
    were consistently mangled).
    """
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
    """
    Crops the top printed box region above topY and uses PyTesseract
    to extract the expected printed words.
    """
    h, w = rawGrayscaleArray.shape
    # Crop below the "Sentence Database" header and page-id text. On IAMpages671
    # those header words otherwise leak into generated training labels.
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

    # Label files are written as "<line_number>\t<text>" (see generate_all_labels.py /
    # IAMpages10/150's hand-verified labels). Strip that leading index so it never leaks
    # into training targets as a literal digit character.
    return [re.sub(r'^\d+\t', '', line) for line in rawLines]


# =====================================================================
# Word-Count Estimation (from ink, not from assumed character width)
# =====================================================================
def CountWordsFromInkColumns(inkCols, lineHeight, minGapFactor=0.6, minRunWidth=3):
    """
    Estimates how many space-separated 'words' are physically present in a
    detected line, from the ink itself -- by clustering ink columns into
    contiguous runs (letters/letter-groups) and counting how many gaps
    between those runs are wide enough to be a word boundary rather than
    just the space between two letters within a word.

    The gap threshold is scaled to the line's own height rather than a
    fixed pixel count, since both letter width and inter-word spacing grow
    together with how large someone writes -- a line's height is a decent
    stand-in for "how big is this handwriting" without needing to measure
    individual letters.

    This exists as a more direct alternative to guessing word count from
    a line's total pixel WIDTH (see SplitWordsIntoLines): width-based
    apportioning assumes roughly constant character width across the page,
    which real handwriting violates often enough to cause word drift
    between adjacent lines. Counting actual word-shaped ink runs is a
    stronger, still fully-automatic signal -- no per-line manual labeling
    needed, which matters since this is also what has to work later on
    pages where only the full known transcript is available, not a
    hand-verified per-line breakdown.
    """
    if len(inkCols) == 0:
        return 0

    gaps = np.where(np.diff(inkCols) > 1)[0]
    starts = np.concatenate(([0], gaps + 1))
    ends = np.concatenate((gaps, [len(inkCols) - 1]))
    runs = [(inkCols[s], inkCols[e]) for s, e in zip(starts, ends)]

    # Drop tiny specks (stray marks, punctuation fragments, noise) that
    # aren't real letters, so they don't get miscounted as their own word.
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
    per-line word-COUNT estimate (from CountWordsFromInkColumns), rather
    than a pixel-width proportion. A line's ink-based word count is scaled
    so the total across all lines matches the known transcript's actual
    word count exactly (handles the estimate being systematically a little
    over or under), then words are handed out to each line in that
    proportion, in order.
    """
    numLines = len(lineWordCounts)
    if numLines == 0 or not words:
        return [""] * numLines

    numWords = len(words)
    totalEstimated = sum(lineWordCounts)

    # If ink-based word counting couldn't find anything usable on this page
    # (e.g. very faint scan), fall back to an even split across lines
    # rather than collapsing to all-blank lines.
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

    # Remove any long straight printed rules left inside the writing region.
    for r in range(regionH):
        row = cleaned[r, :] > 0
        if LongestRun(row) > regionW * 0.25 and np.sum(row) > regionW * 0.10:
            cleaned[max(0, r - 2):min(regionH, r + 3), :] = 0

    return cleaned


def MergeUndersizedFragments(lines, minRelativeHeight=0.4):
    """
    Descender tails, dangling punctuation, and stray marks sometimes sit far
    enough below their line's main ink band (>18px, see the gap-merge above)
    that they survive as their own tiny detected "line" instead of merging
    back in. These fragments are reliably distinguishable from real lines by
    HEIGHT alone (~15-20px vs ~100-130px for a real handwritten line here),
    regardless of how far the gap separates them, so merge on that basis
    into whichever real neighbor is closer.
    """
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

    # Binary Thresholding
    BinaryInvertedImage = OtsuThreshold(RawGrayscaleArray)

    # Detect Page Bounds
    topY, botY = DetectPageBounds(BinaryInvertedImage, RawGrayscaleArray)
    headerTopY = DetectHeaderTopY(BinaryInvertedImage, RawGrayscaleArray, topY)

    # Extract printed words from top box
    printedWords = ExtractPrintedGroundTruth(RawGrayscaleArray, topY, headerTopY)

    # HANDWRITING_TOP_MARGIN: the region actually scanned for handwriting
    # (and therefore cropped into training line images) starts this many
    # pixels BELOW topY, not exactly at it. topY marks where a detected
    # rule line sits, but rule lines have their own thickness/anti-aliasing
    # and detection can be off by a few pixels -- starting extraction
    # exactly at topY risks a training image's top edge grazing the last
    # row or two of PRINTED header text. This margin is a hard, always-on
    # buffer so header pixels structurally cannot end up inside a
    # handwriting line crop, at the cost of a few pixels of the true first
    # handwritten line's very top (negligible -- see MergeUndersizedFragments
    # for how thin slivers get handled anyway).
    HANDWRITING_TOP_MARGIN = 15
    handwritingStartY = min(botY, topY + HANDWRITING_TOP_MARGIN)

    hwRegion = BinaryInvertedImage[handwritingStartY:botY, :].copy()
    regionH, regionW = hwRegion.shape

    hwRegion = RemoveStraightFormLines(hwRegion)

    # Mask Left Vertical Margin Lines & Edge Artifacts
    colInkHeights = np.sum(hwRegion > 0, axis=0)
    vertLineCols = np.where(colInkHeights > regionH * 0.40)[0]
    hwRegion[:, vertLineCols] = 0
    hwRegion[:, :int(w * 0.07)] = 0
    hwRegion[:, int(w * 0.97):] = 0

    # 1D Horizontal Projection Profile
    rowInkCount = np.sum(hwRegion > 0, axis=1)
    hasInk = rowInkCount > max(10, int(w * 0.006))

    pad1D = np.pad(hasInk, (3, 3), mode='constant')
    smoothedInk = np.zeros_like(hasInk)
    for dy in range(7):
        smoothedInk = np.logical_or(smoothedInk, pad1D[dy:dy + len(hasInk)])

    diff = np.diff(np.concatenate(([0], smoothedInk.astype(np.int8), [0])))
    lineStarts, lineEnds = np.where(diff == 1)[0], np.where(diff == -1)[0]

    rawLines = [(s, e) for s, e in zip(lineStarts, lineEnds) if (e - s) >= 8]

    # Merge Descender Fragments ('y', 'p', 'g')
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
        # Offset from handwritingStartY, not topY -- hwRegion (and therefore
        # s/e, which are row indices WITHIN hwRegion) starts at
        # handwritingStartY now, not topY. Using topY here would shift
        # every line crop's absolute position up by HANDWRITING_TOP_MARGIN
        # pixels, silently re-introducing the exact header-overlap risk the
        # margin above exists to prevent.
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

        # Draw Red Bounding Box on visualization canvas, numbered to match
        # this line's index in the returned text-line list -- makes it easy
        # to cross-reference the preview image against a printed label list.
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
# Main Script Execution: Loop Through All Images in Folder '150'
# =====================================================================
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    AUTHOR_150_DIR = os.path.abspath(os.path.join(
        SCRIPT_DIR, "..", "..", "Data", "Datasets", "IAMpages10", "150"
    ))

    # Base output folder
    outputDir = os.path.join(SCRIPT_DIR, "handwriting_line_extraction_output")
    previewsDir = os.path.join(outputDir, "previews")
    cropsBaseDir = os.path.join(outputDir, "line_crops")

    os.makedirs(previewsDir, exist_ok=True)
    os.makedirs(cropsBaseDir, exist_ok=True)

    if os.path.exists(AUTHOR_150_DIR):
        # Gather all page images in author folder 150
        image_paths = sorted(glob.glob(os.path.join(AUTHOR_150_DIR, "*.png")))
        print(f"Found {len(image_paths)} images in '{AUTHOR_150_DIR}' for processing...\n")

        passedCount = 0

        for img_path in image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            labelLines = ReadLabelLines(img_path)
            expectedLineCount = len(labelLines) if labelLines else None
            print(f"-> Processing '{base_name}.png'...")

            samples, textLines, previewImg = ExtractLinePatches(
                img_path,
                expectedLineCount=expectedLineCount,
                labelLines=labelLines if labelLines else None
            )

            if previewImg is not None and len(samples) > 0:
                # 1. Save bounding box preview image for this specific page scan
                previewPath = os.path.join(previewsDir, f"{base_name}_preview.png")
                previewImg.save(previewPath)

                # 2. Save line crops into a folder named after the page scan
                pageCropsDir = os.path.join(cropsBaseDir, base_name)
                os.makedirs(pageCropsDir, exist_ok=True)

                for s in samples:
                    lineFilename = f"line_{s['line_idx']:02d}.png"
                    s['processed_patch'].save(os.path.join(pageCropsDir, lineFilename))

                    if s.get('label_text'):
                        labelFilename = f"line_{s['line_idx']:02d}.txt"
                        with open(os.path.join(pageCropsDir, labelFilename), "w", encoding="utf-8") as f:
                            f.write(s['label_text'] + "\n")

                if expectedLineCount is not None and len(samples) == expectedLineCount:
                    passedCount += 1
                    countStatus = "PASS"
                elif expectedLineCount is not None:
                    countStatus = "FAIL"
                else:
                    countStatus = "NO LABEL"

                expectedText = expectedLineCount if expectedLineCount is not None else "?"
                print(f"   [{countStatus}] Extracted {len(samples)} lines | Expected {expectedText} | Preview saved: {os.path.basename(previewPath)}")
            else:
                print(f"   [Warning] No lines detected in '{base_name}.png'.")

        print(f"\n[Finished!] Processed all {len(image_paths)} images.")
        print(f" Label-count matches: {passedCount}/{len(image_paths)}")
        print(f" All Bounding Box Previews saved to:\n   {previewsDir}")
        print(f" All Extracted Line Crops saved to:\n   {cropsBaseDir}")
    else:
        print(f"Error: Could not find directory at '{AUTHOR_150_DIR}'")
