import os
import numpy as np
from PIL import Image, ImageDraw


# =====================================================================
# Otsu's Automatic Image Thresholding (First Principles)
# =====================================================================
def OtsuThreshold(grayImg):
    hist, _ = np.histogram(grayImg, bins=256, range=(0, 256))
    total = grayImg.size
    currentMax = 0
    bestThresh = 0
    sumB = 0
    sum1 = np.dot(np.arange(256), hist)
    wB = 0

    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        varBetween = wB * wF * ((mB - mF) ** 2)

        if varBetween > currentMax:
            currentMax = varBetween
            bestThresh = t

    return (grayImg < bestThresh).astype(np.uint8) * 255


# =====================================================================
# 2D Morphological Dilation (NumPy Slice Maxing)
# =====================================================================
def DilateImage(binaryImg, kWidth, kHeight):
    h, w = binaryImg.shape
    paddedImg = np.pad(
        binaryImg,
        ((kHeight // 2, kHeight // 2), (kWidth // 2, kWidth // 2)),
        mode='constant'
    )
    outImg = np.zeros_like(binaryImg)

    for dy in range(kHeight):
        for dx in range(kWidth):
            outImg = np.maximum(outImg, paddedImg[dy:dy + h, dx:dx + w])

    return outImg


# =====================================================================
# Two-Pass Union-Find Connected Component Analysis
# =====================================================================
def FindComponents(binaryImg):
    h, w = binaryImg.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = [0]
    nextLabel = 1

    def find(i):
        p = i
        while parent[p] != p:
            p = parent[p]
        curr = i
        while curr != p:
            nxt = parent[curr]
            parent[curr] = p
            curr = nxt
        return p

    def union(i, j):
        r1 = find(i)
        r2 = find(j)
        if r1 != r2:
            parent[r2] = r1

    for r in range(h):
        for c in range(w):
            if binaryImg[r, c] == 0:
                continue
            neighbors = []
            if r > 0 and labels[r - 1, c] > 0:
                neighbors.append(labels[r - 1, c])
            if c > 0 and labels[r, c - 1] > 0:
                neighbors.append(labels[r, c - 1])
            if r > 0 and c > 0 and labels[r - 1, c - 1] > 0:
                neighbors.append(labels[r - 1, c - 1])
            if r > 0 and c < w - 1 and labels[r - 1, c + 1] > 0:
                neighbors.append(labels[r - 1, c + 1])

            if not neighbors:
                parent.append(nextLabel)
                labels[r, c] = nextLabel
                nextLabel += 1
            else:
                minLbl = min(neighbors)
                labels[r, c] = minLbl
                for n in neighbors:
                    union(minLbl, n)

    boxesDict = {}
    for r in range(h):
        for c in range(w):
            lbl = labels[r, c]
            if lbl == 0:
                continue
            root = find(lbl)
            if root not in boxesDict:
                boxesDict[root] = [c, r, c, r]
            else:
                b = boxesDict[root]
                if c < b[0]: b[0] = c
                if r < b[1]: b[1] = r
                if c > b[2]: b[2] = c
                if r > b[3]: b[3] = r

    finalBoxes = []
    for b in boxesDict.values():
        x1, y1, x2, y2 = b
        finalBoxes.append([x1, y1, x2 - x1 + 1, y2 - y1 + 1])

    return finalBoxes


# =====================================================================
# Page Boundary & Horizontal Rule Line Detection
# =====================================================================
def DetectPageBounds(binaryImg):
    h, w = binaryImg.shape
    yLines = []

    for r in range(h):
        row = binaryImg[r, :]
        diff = np.diff(np.concatenate(([0], (row > 0).astype(np.int8), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        if len(starts) > 0:
            maxRun = np.max(ends - starts)
            if maxRun > w * 0.4:
                yLines.append(r)

    if not yLines:
        return int(h * 0.22), int(h * 0.82)

    lineCenters = []
    currGroup = [yLines[0]]

    for i in range(1, len(yLines)):
        if yLines[i] == yLines[i - 1] + 1:
            currGroup.append(yLines[i])
        else:
            lineCenters.append(int(np.mean(currGroup)))
            currGroup = [yLines[i]]
    lineCenters.append(int(np.mean(currGroup)))

    topY = lineCenters[0] if len(lineCenters) >= 1 else int(h * 0.22)
    botY = lineCenters[-1] if len(lineCenters) >= 2 else int(h * 0.82)

    if botY <= topY + 100:
        topY, botY = int(h * 0.22), int(h * 0.82)

    return topY, botY


# =====================================================================
# Main Handwriting Extraction & Segmentation Pipeline
# =====================================================================
def ProcessSegmentation(imgPath):
    rawImage = Image.open(imgPath)
    RawGrayscaleArray = np.array(rawImage.convert('L'))
    h, w = RawGrayscaleArray.shape

    # Binary Image Conversion
    BinaryInvertedImage = OtsuThreshold(RawGrayscaleArray)

    # Detect Page Bounds
    topY, botY = DetectPageBounds(BinaryInvertedImage)
    hwRegion = BinaryInvertedImage[topY:botY, :]

    # Estimate Median Character Scale
    rawComp = FindComponents(hwRegion)
    heights = [b[3] for b in rawComp if b[3] > 5]
    if not heights:
        return None
    MedianCharacterHeight = int(np.median(heights))

    # Line Separation
    lineKW = max(40, MedianCharacterHeight * 3)
    lineKH = max(3, MedianCharacterHeight // 4)
    dilatedLines = DilateImage(hwRegion, lineKW, lineKH)

    rawLineBoxes = FindComponents(dilatedLines)
    LineBoundingBoxes = [b for b in rawLineBoxes if b[2] > MedianCharacterHeight * 2]
    LineBoundingBoxes.sort(key=lambda b: b[1])

    # Canvas Drawing Setup
    drawCanvas = rawImage.convert('RGB')
    drawObj = ImageDraw.Draw(drawCanvas)
    drawObj.line([(0, topY), (w, topY)], fill=(0, 0, 255), width=2)
    drawObj.line([(0, botY), (w, botY)], fill=(0, 0, 255), width=2)

    for lx, ly, lw, lh in LineBoundingBoxes:
        drawObj.rectangle([lx, topY + ly, lx + lw, topY + ly + lh], outline=(255, 0, 0), width=2)

        lineBinary = hwRegion[ly:ly + lh, lx:lx + lw]

        # Extract Raw Undilated Ink Components
        rawInkBoxes = FindComponents(lineBinary)
        
        # Filter Noise
        validInk = [b for b in rawInkBoxes if b[2] >= 2 and b[3] >= 2 and b[0] > 15]

        PunctuationBlocks = []
        LetterBlocks = []

        # --- Stage 1: Explicit Punctuation Detection ---
        for b in validInk:
            wx, wy, ww, wh = b
            
            # Check upper half region directly above the ink blob
            UpperRegion = lineBinary[0:int(lh * 0.45), wx:wx + ww]
            IsEmptyAboveRegion = np.sum(UpperRegion) == 0

            # Punctuation Criteria: Sits in lower half + empty upper space + compact size
            IsPunctuationBlob = (wy > lh * 0.45) and IsEmptyAboveRegion and (wh < MedianCharacterHeight * 1.1)

            if IsPunctuationBlob:
                PunctuationBlocks.append(b)
            else:
                LetterBlocks.append(b)

        # --- Stage 2: Word Grouping (Punctuation Excluded) ---
        # Stitch vertical letter strokes ('p', 'b', 't', etc.)
        strokeKW = 3
        strokeKH = 5
        dilatedStrokes = DilateImage(lineBinary, strokeKW, strokeKH)
        rawStrokeBoxes = FindComponents(dilatedStrokes)
        
        # Filter stroke boxes to only keep those containing letter components
        LetterStrokeBoxes = []
        for sb in rawStrokeBoxes:
            sx, sy, sw, sh = sb
            if sx <= 15:
                continue
            # Check if this stroke overlaps with any classified letter component
            hasLetter = any(
                not (lx2 < sx or lx1 > sx + sw or ly2 < sy or ly1 > sy + sh)
                for lx1, ly1, lw1, lh1 in LetterBlocks
                for lx2, ly2 in [(lx1 + lw1, ly1 + lh1)]
            )
            if hasLetter:
                LetterStrokeBoxes.append(sb)

        LetterStrokeBoxes.sort(key=lambda b: b[0])

        MergedWordBlocks = []
        WordGapThreshold = int(MedianCharacterHeight * 0.55)

        for b in LetterStrokeBoxes:
            wx, wy, ww, wh = b

            if not MergedWordBlocks:
                MergedWordBlocks.append([wx, wy, ww, wh])
                continue

            lwx, lwy, lww, lwh = MergedWordBlocks[-1]
            gap = wx - (lwx + lww)

            if gap < WordGapThreshold:
                # Merge into current word bounding box
                MergedWordBlocks[-1] = [
                    min(lwx, wx), min(lwy, wy),
                    max(lwx + lww, wx + ww) - min(lwx, wx),
                    max(lwy + lwh, wy + wh) - min(lwy, wy)
                ]
            else:
                # Start new word bounding box
                MergedWordBlocks.append([wx, wy, ww, wh])

        # Draw Word Boxes in GREEN
        for b in MergedWordBlocks:
            x1, y1 = lx + b[0], topY + ly + b[1]
            x2, y2 = x1 + b[2], y1 + b[3]
            drawObj.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

        # Draw Punctuation Boxes in RED (Guaranteed Special Boxes)
        for b in PunctuationBlocks:
            x1, y1 = lx + b[0], topY + ly + b[1]
            x2, y2 = x1 + b[2], y1 + b[3]
            drawObj.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

    return drawCanvas


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAMPLE_IMAGE_PATH = os.path.abspath(os.path.join(
        SCRIPT_DIR, "..", "..", "Data", "Datasets", "IAMpages10", "150", "c03-000a.png"
    ))

    outputDir = os.path.join(SCRIPT_DIR, "handwriting_extraction_output")
    os.makedirs(outputDir, exist_ok=True)

    if os.path.exists(SAMPLE_IMAGE_PATH):
        print(f"Processing '{os.path.basename(SAMPLE_IMAGE_PATH)}' using pure NumPy...")
        outCanvas = ProcessSegmentation(SAMPLE_IMAGE_PATH)

        if outCanvas is not None:
            debugPath = os.path.join(outputDir, "numpy_smart_boxes_preview.png")
            outCanvas.save(debugPath)
            print(f"[Success] Saved first-principles preview to:\n  {debugPath}")
    else:
        print(f"Error: Could not find image at '{SAMPLE_IMAGE_PATH}'")