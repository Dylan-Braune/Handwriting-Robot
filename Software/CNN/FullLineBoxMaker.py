import os
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


# =====================================================================
# Page Bounds Detection
# =====================================================================
def DetectPageBounds(binaryImg):
    h, w = binaryImg.shape
    ruleRows = []

    for r in range(h):
        row = binaryImg[r, :] > 0
        if not np.any(row): continue
        diff = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
        if len(starts) > 0 and np.max(ends - starts) > w * 0.25:
            ruleRows.append(r)

    if not ruleRows:
        return int(h * 0.12), int(h * 0.80)

    clusters, currGroup = [], [ruleRows[0]]
    for i in range(1, len(ruleRows)):
        if ruleRows[i] <= ruleRows[i - 1] + 3:
            currGroup.append(ruleRows[i])
        else:
            clusters.append(int(np.mean(currGroup)))
            currGroup = [ruleRows[i]]
    clusters.append(int(np.mean(currGroup)))

    topCandidates = [c for c in clusters if c < h * 0.35]
    botCandidates = [c for c in clusters if c > h * 0.65]

    topY = topCandidates[-1] + 2 if topCandidates else int(h * 0.12)
    botY = botCandidates[0] - 3 if botCandidates else int(h * 0.80)

    return topY, botY


# =====================================================================
# Extract Printed Text Ground-Truth via PyTesseract
# =====================================================================
def ExtractPrintedGroundTruth(rawGrayscaleArray, topY):
    """
    Crops the top printed box region above topY and uses PyTesseract
    to extract the expected printed words.
    """
    h, w = rawGrayscaleArray.shape
    printedRegion = rawGrayscaleArray[int(h * 0.05):topY, :]
    
    # Run PyTesseract OCR (PSM 6 = assume a single uniform block of text)
    try:
        rawText = pytesseract.image_to_string(printedRegion, config='--psm 6')
        cleanText = rawText.replace('\n', ' ').strip()
        words = [w.strip() for w in cleanText.split() if w.strip()]
        return words
    except Exception as e:
        print(f"  [OCR Warning] Could not extract printed text: {e}")
        return []


# =====================================================================
# Proportional Word Distribution Engine
# =====================================================================
def SplitWordsIntoLines(words, lineWidths):
    """
    Distributes a continuous list of words across N line crops based on
    the relative physical pixel width of each detected handwritten line.
    """
    numLines = len(lineWidths)
    if numLines == 0 or not words:
        return [""] * numLines

    totalWidth = float(sum(lineWidths))
    totalChars = sum(len(w) for w in words)
    
    allocatedLines = []
    wordIdx = 0
    numWords = len(words)

    for i in range(numLines):
        if wordIdx >= numWords:
            allocatedLines.append("")
            continue

        if i == numLines - 1:
            # Last line takes all remaining words
            allocatedLines.append(" ".join(words[wordIdx:]))
            break

        # Calculate target character budget for this specific line
        targetChars = totalChars * (lineWidths[i] / totalWidth)
        currentLineWords = []
        currentChars = 0

        while wordIdx < numWords:
            word = words[wordIdx]
            # Stop if adding this word significantly exceeds the line budget
            if currentLineWords and (currentChars + len(word)) > targetChars * 1.15:
                break
            currentLineWords.append(word)
            currentChars += len(word) + 1  # includes space
            wordIdx += 1

        allocatedLines.append(" ".join(currentLineWords))

    return allocatedLines


# =====================================================================
# Line Extraction Pipeline
# =====================================================================
def ExtractLinePatches(imgPath, targetHeight=32, maxWidth=1024):
    rawImage = Image.open(imgPath)
    RawGrayscaleArray = np.array(rawImage.convert('L'))
    h, w = RawGrayscaleArray.shape

    # Binary Thresholding
    BinaryInvertedImage = OtsuThreshold(RawGrayscaleArray)

    # Detect Page Bounds
    topY, botY = DetectPageBounds(BinaryInvertedImage)
    
    # Extract printed words from top box
    printedWords = ExtractPrintedGroundTruth(RawGrayscaleArray, topY)

    hwRegion = BinaryInvertedImage[topY:botY, :].copy()
    regionH, regionW = hwRegion.shape

    # Mask Left Vertical Margin Lines & Edge Artifacts
    colInkHeights = np.sum(hwRegion > 0, axis=0)
    vertLineCols = np.where(colInkHeights > regionH * 0.40)[0]
    hwRegion[:, vertLineCols] = 0
    hwRegion[:, :int(w * 0.07)] = 0

    # 1D Horizontal Projection Profile
    rowInkCount = np.sum(hwRegion > 0, axis=1)
    hasInk = rowInkCount > 10

    pad1D = np.pad(hasInk, (2, 2), mode='constant')
    smoothedInk = np.zeros_like(hasInk)
    for dy in range(5):
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
            if (currS - prevE) < 16:
                mergedLines[-1] = (prevS, currE)
            else:
                mergedLines.append(line)

    drawCanvas = rawImage.convert('RGB')
    drawObj = ImageDraw.Draw(drawCanvas)
    drawObj.line([(0, topY), (w, topY)], fill=(0, 0, 255), width=2)
    drawObj.line([(0, botY), (w, botY)], fill=(0, 0, 255), width=2)

    ExtractedLineSamples = []
    lineWidths = []

    for idx, (s, e) in enumerate(mergedLines):
        absY1 = topY + max(0, s - 3)
        absY2 = topY + min(regionH, e + 3)
        cropH = absY2 - absY1

        lineBinary = hwRegion[s:e, :]
        colSum = np.sum(lineBinary > 0, axis=0)
        inkCols = np.where(colSum > 0)[0]
        if len(inkCols) == 0: continue

        absX1 = max(0, inkCols[0] - 6)
        absX2 = min(w, inkCols[-1] + 6)
        cropW = absX2 - absX1
        lineWidths.append(cropW)

        drawObj.rectangle([absX1, absY1, absX2, absY2], outline=(255, 0, 0), width=2)

        rawLineCrop = RawGrayscaleArray[absY1:absY2, absX1:absX2]
        scaleRatio = targetHeight / float(cropH)
        newWidth = min(maxWidth, int(cropW * scaleRatio))

        cropPIL = Image.fromarray(rawLineCrop)
        resizedPIL = cropPIL.resize((newWidth, targetHeight), Image.Resampling.LANCZOS)

        paddedCanvas = Image.new('L', (maxWidth, targetHeight), color=255)
        paddedCanvas.paste(resizedPIL, (0, 0))

        ExtractedLineSamples.append({
            'line_idx': idx,
            'raw_crop': rawLineCrop,
            'processed_patch': paddedCanvas
        })

    # Divide extracted prompt text across N line crops proportionally
    formattedTextLines = SplitWordsIntoLines(printedWords, lineWidths)

    return ExtractedLineSamples, formattedTextLines, drawCanvas


# =====================================================================
# Main Script Execution
# =====================================================================
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAMPLE_IMAGE_PATH = os.path.abspath(os.path.join(
        SCRIPT_DIR, "..", "..", "Data", "Datasets", "IAMpages10", "150", "c03-000a.png"
    ))

    outputDir = os.path.join(SCRIPT_DIR, "handwriting_line_extraction_output")
    os.makedirs(outputDir, exist_ok=True)

    if os.path.exists(SAMPLE_IMAGE_PATH):
        print(f"Processing '{os.path.basename(SAMPLE_IMAGE_PATH)}'...")
        samples, textLines, previewImg = ExtractLinePatches(SAMPLE_IMAGE_PATH)

        if previewImg is not None and len(samples) > 0:
            previewPath = os.path.join(outputDir, "line_bounding_boxes_preview.png")
            previewImg.save(previewPath)

            linesDir = os.path.join(outputDir, "line_crops")
            os.makedirs(linesDir, exist_ok=True)

            # Save line crops
            for s in samples:
                lineFilename = f"line_{s['line_idx']:02d}.png"
                s['processed_patch'].save(os.path.join(linesDir, lineFilename))

            # Save matching line-by-line label file
            labelsPath = os.path.join(outputDir, "page_labels.txt")
            with open(labelsPath, "w", encoding="utf-8") as f:
                for line in textLines:
                    f.write(line + "\n")

            print(f"\n[Success] Extracted {len(samples)} lines cleanly!")
            print(f" Preview saved to:\n  {previewPath}")
            print(f" 32x1024 line tensors saved to:\n  {linesDir}")
            print(f" Aligned label text file saved to:\n  {labelsPath}")
            print("\n--- Formatted Text Labels ---")
            for idx, text in enumerate(textLines):
                print(f" Line {idx:02d}: \"{text}\"")
    else:
        print(f"Error: Could not find image at '{SAMPLE_IMAGE_PATH}'")