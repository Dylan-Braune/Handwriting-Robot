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

    # Returns inverted binary matrix (Ink = 255, Background = 0)
    return (grayImg < bestThresh).astype(np.uint8) * 255


# =====================================================================
# 2D Morphological Dilation (First Principles NumPy Slice-Maxing)
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
# Connected Components Analysis & Bounding Box Extraction (Two-Pass Union-Find)
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

    # First Pass: Labeling and Equivalence Recording
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

    # Second Pass: Bounding Box Aggregation
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
# Page Boundary & Horizontal Rule Detection
# =====================================================================
def DetectBounds(binaryImg):
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
# Main Segmentation Pipeline Execution
# =====================================================================
def ProcessSegmentation(imgPath):
    rawImage = Image.open(imgPath)
    grayArray = np.array(rawImage.convert('L'))
    h, w = grayArray.shape

    # Binary Thresholding
    binaryImg = OtsuThreshold(grayArray)

    # Detect Page Bounds
    topY, botY = DetectBounds(binaryImg)
    hwRegion = binaryImg[topY:botY, :]

    # Median Character Scale Estimate
    rawComp = FindComponents(hwRegion)
    heights = [b[3] for b in rawComp if b[3] > 5]
    if not heights:
        return None
    medianH = int(np.median(heights))

    # Line Extraction via Horizontal Dilation
    lineKW = max(40, medianH * 3)
    lineKH = max(3, medianH // 4)
    dilatedLines = DilateImage(hwRegion, lineKW, lineKH)

    rawLineBoxes = FindComponents(dilatedLines)
    lineBoxes = [b for b in rawLineBoxes if b[2] > medianH * 2]
    lineBoxes.sort(key=lambda b: b[1])

    # Visualization Setup
    drawCanvas = rawImage.convert('RGB')
    drawObj = ImageDraw.Draw(drawCanvas)
    drawObj.line([(0, topY), (w, topY)], fill=(0, 0, 255), width=2)
    drawObj.line([(0, botY), (w, botY)], fill=(0, 0, 255), width=2)

    for lx, ly, lw, lh in lineBoxes:
        drawObj.rectangle([lx, topY + ly, lx + lw, topY + ly + lh], outline=(255, 0, 0), width=2)

        lineBinary = hwRegion[ly:ly + lh, lx:lx + lw]

        # Vertical Pen Stroke Stitching Kernel (2x5)
        # Increases horizontal stroke smearing from 2px to 3px to bridge 'b' and 'p' gaps
        dilatedStrokes = DilateImage(lineBinary, 3, 5)
        rawBoxes = FindComponents(dilatedStrokes)

        validBoxes = []
        for b in rawBoxes:
            wx, wy, ww, wh = b
            if ww >= 2 and wh >= 2 and (ww * wh) >= 4:
                if wx > 15:  # Left Margin Filter
                    validBoxes.append([wx, wy, ww, wh])

        validBoxes.sort(key=lambda b: b[0])

        mergedBlocks = []
        WordGapThresh = int(medianH * 0.45)

        for box in validBoxes:
            wx, wy, ww, wh = box

         # Pushes the punctuation zone lower (65% down) and requires punctuation to be narrower
            isBottomOnly = wy > (lh * 0.65)
            isPunctSize = ww < (medianH * 0.5) and wh < (medianH * 0.8)
            isPunctuation = isPunctSize and isBottomOnly

            isDotSize = ww < (medianH * 0.5) and wh < (medianH * 0.5)
            isTopHalf = (wy + wh) < (lh * 0.5)
            isIDot = isDotSize and isTopHalf

            if not mergedBlocks:
                mergedBlocks.append({'box': [wx, wy, ww, wh], 'is_punct': isPunctuation})
                continue

            lastBlock = mergedBlocks[-1]
            lwx, lwy, lww, lwh = lastBlock['box']
            gap = wx - (lwx + lww)

            # Rule 1: 'i' dots expand upward
            if isIDot and gap < WordGapThresh:
                mergedBlocks[-1]['box'] = [
                    min(lwx, wx), min(lwy, wy),
                    max(lwx + lww, wx + ww) - min(lwx, wx),
                    max(lwy + lwh, wy + wh) - min(lwy, wy)
                ]
                continue

            # Rule 2: Commas & Full Stops break off
            if isPunctuation:
                mergedBlocks.append({'box': [wx, wy, ww, wh], 'is_punct': True})
                continue

            # Rule 3: Merge normal letters
            if gap < WordGapThresh and not lastBlock['is_punct']:
                mergedBlocks[-1]['box'] = [
                    min(lwx, wx), min(lwy, wy),
                    max(lwx + lww, wx + ww) - min(lwx, wx),
                    max(lwy + lwh, wy + wh) - min(lwy, wy)
                ]
            else:
                mergedBlocks.append({'box': [wx, wy, ww, wh], 'is_punct': False})

        for block in mergedBlocks:
            bx, by, bw, bh = block['box']
            color = (255, 0, 0) if block['is_punct'] else (0, 255, 0)
            x1, y1 = lx + bx, topY + ly + by
            x2, y2 = x1 + bw, y1 + bh
            drawObj.rectangle([x1, y1, x2, y2], outline=color, width=2)

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