#segments a scanned IAM sentence-database page into per-line crops. real
#per-line labels always come from ReadLabelLines's hand-verified transcript,
#matched to crops by count -- also supports personal (non-IAM) ruled-paper
#pages via isDataset=False, which skips the IAM-specific header/footer
#detection and switches to a periodicity-based line splitter instead of a
#blank-gap one
import os
import glob
import re
import numpy as np
from PIL import Image,ImageDraw
from scipy.signal import find_peaks


#otsu's method: the threshold that maximises between-class variance
def OtsuThreshold(grayImg):
    hist,_ = np.histogram(grayImg,bins=256,range=(0,256))
    total = grayImg.size
    currentMax,bestThresh,sumB,sum1,wB = 0,0,0,np.dot(np.arange(256),hist),0

    for t in range(256):
        wB+=hist[t]
        if wB==0: continue
        wF = total-wB
        if wF==0: break
        sumB+=t*hist[t]
        mB = sumB/wB
        mF = (sum1-sumB)/wF
        varBetween = wB*wF*((mB-mF)**2)
        if varBetween>currentMax:
            currentMax = varBetween
            bestThresh = t

    return (grayImg<bestThresh).astype(np.uint8)*255


#small helper for detecting straight printed divider lines
def LongestRun(boolRow):
    inkCols = np.where(boolRow)[0]
    if len(inkCols)==0:
        return 0
    gaps = np.where(np.diff(inkCols)>1)[0]
    starts = np.concatenate(([0],gaps+1))
    ends = np.concatenate((gaps,[len(inkCols)-1]))
    return int(np.max(inkCols[ends]-inkCols[starts]+1))


#groups nearby row indices (within maxGap) into single cluster centres
def ClusterRows(rows,maxGap=5):
    if not rows:
        return []

    clusters,currGroup = [],[rows[0]]
    for i in range(1,len(rows)):
        if rows[i]<=rows[i-1]+maxGap:
            currGroup.append(rows[i])
        else:
            clusters.append(int(np.mean(currGroup)))
            currGroup = [rows[i]]
    clusters.append(int(np.mean(currGroup)))
    return clusters


#rows containing a long, mostly-solid horizontal run: printed rule lines,
#using raw grayscale (less noisy than the binary mask) when available
def FindRuleLineClusters(binaryImg,rawGrayscaleArray=None):
    h,w = binaryImg.shape
    ruleRows = []

    lineSource = rawGrayscaleArray if rawGrayscaleArray is not None else binaryImg
    for r in range(h):
        if rawGrayscaleArray is not None:
            row = lineSource[r,:]<225
        else:
            row = lineSource[r,:]>0

        if not np.any(row): continue
        if LongestRun(row)>w*0.25 and np.sum(row)>w*0.10:
            ruleRows.append(r)

    if not ruleRows:
        return []

    return ClusterRows(ruleRows,maxGap=8)


#finds the handwriting region's top/bottom rule lines (returnConfidence=True
#also flags when a bound had to fall back to a fixed-percentage guess
#instead of a real detected rule line -- see PageBoundsAreUncertain)
def DetectPageBounds(binaryImg,rawGrayscaleArray=None,returnConfidence=False):
    h,w = binaryImg.shape
    clusters = FindRuleLineClusters(binaryImg,rawGrayscaleArray)

    topCandidates = [c for c in clusters if h*0.12<c<h*0.40]
    botCandidates = [c for c in clusters if h*0.55<c<h*0.92]

    topY = topCandidates[-1]+8 if topCandidates else int(h*0.12)
    botY = botCandidates[0]-8 if botCandidates else int(h*0.80)

    if returnConfidence:
        return topY,botY,len(topCandidates)==0,len(botCandidates)==0
    return topY,botY


#content-based check: confirms the rows just above topY are mostly blank,
#i.e. topY isn't sitting on top of real printed header text
def HeaderGapIsSafe(binaryImg,topY,minGapPx=10,maxInkFraction=0.05):
    h = binaryImg.shape[0]
    zoneStart = max(0,topY-minGapPx)
    checkZone = binaryImg[zoneStart:topY,:]
    if checkZone.size==0:
        return True   #nothing to check against -- topY is at/near the page top

    inkFractionPerRow = np.mean(checkZone>0,axis=1)
    return not np.any(inkFractionPerRow>maxInkFraction)


#true if this page's handwriting region should NOT be trusted for auto-
#labeling: either the top bound fell back to a fixed-percentage guess, or
#the gap just above it isn't actually blank (may sit over header text)
def PageBoundsAreUncertain(imgPath):
    rawImage = Image.open(imgPath)
    grayArr = np.array(rawImage.convert('L'))
    binaryImg = OtsuThreshold(grayArr)
    topY,_botY,topWasFallback,_botWasFallback = DetectPageBounds(binaryImg,grayArr,returnConfidence=True)

    if topWasFallback:
        return True

    return not HeaderGapIsSafe(binaryImg,topY)


#finds the rule line bounding the TOP of the printed prompt box, rather
#than assuming a fixed page-height percentage (some form templates place
#it higher, and a fixed percentage clips/garbles the first printed line's OCR)
def DetectHeaderTopY(binaryImg,rawGrayscaleArray,topY):
    h,_w = binaryImg.shape
    clusters = FindRuleLineClusters(binaryImg,rawGrayscaleArray)

    candidates = [c for c in clusters if h*0.04<c<(topY-8)]
    if candidates:
        return max(candidates)+8

    return int(h*0.115)


#used to OCR (tesseract) the printed prompt box above topY as a fallback
#transcript source. removed: the only live caller (ExtractLinePatches, via
#LineImageCache) always discards this function's result -- real per-line
#labels come from ReadLabelLines's hand-verified transcript instead, matched
#to crops by count, so this was pure OCR computation nothing ever consumed.
#kept as a stub (always empty) so ExtractLinePatches's fallback branch below
#still has a well-defined, harmless input if label counts ever mismatch
def ExtractPrintedGroundTruth(rawGrayscaleArray,topY,headerTopY=None):
    return []


#reads a page's per-line hand-verified transcript. labelPath defaults to
#the image's own "<stem>_labels.txt" sibling; pass it explicitly for a
#dataset that keeps images/labels in separate Pages/Labels folders
def ReadLabelLines(imgPath,labelPath=None):
    labelPath = labelPath or (os.path.splitext(imgPath)[0]+"_labels.txt")
    if not os.path.exists(labelPath):
        return []

    with open(labelPath,"r",encoding="utf-8") as f:
        rawLines = [line.strip() for line in f.readlines() if line.strip()]

    #label lines are "<line_number>\t<text>" -- strip the leading index so
    #it never leaks into training targets as a literal digit character
    return [re.sub(r'^\d+\t','',line) for line in rawLines]


#estimates word count from the ink itself: clusters ink columns into
#letter-runs and counts gaps wide enough (relative to line height) to be
#a word boundary rather than just the space within a word
def CountWordsFromInkColumns(inkCols,lineHeight,minGapFactor=0.6,minRunWidth=3):
    if len(inkCols)==0:
        return 0

    gaps = np.where(np.diff(inkCols)>1)[0]
    starts = np.concatenate(([0],gaps+1))
    ends = np.concatenate((gaps,[len(inkCols)-1]))
    runs = [(inkCols[s],inkCols[e]) for s,e in zip(starts,ends)]

    #drop tiny specks that aren't real letters
    runs = [(s,e) for s,e in runs if (e-s+1)>=minRunWidth]
    if not runs:
        return 0

    wordGapThreshold = max(1,lineHeight*minGapFactor)
    wordCount = 1
    for i in range(1,len(runs)):
        gapWidth = runs[i][0]-runs[i-1][1]
        if gapWidth>wordGapThreshold:
            wordCount+=1

    return wordCount


#distributes the page's known words across N line crops in proportion to
#each line's ink-based word-count estimate (scaled so the total matches
#the real word count exactly), rather than by pixel-width proportion
def SplitWordsIntoLines(words,lineWordCounts):
    numLines = len(lineWordCounts)
    if numLines==0 or not words:
        return [""]*numLines

    numWords = len(words)
    totalEstimated = sum(lineWordCounts)

    #ink-based counting found nothing usable (e.g. very faint scan) --
    #fall back to an even split rather than collapsing to all-blank lines
    if totalEstimated<=0:
        lineWordCounts = [1]*numLines
        totalEstimated = numLines

    scale = numWords/float(totalEstimated)

    allocatedLines = []
    wordIdx = 0
    for i in range(numLines):
        if wordIdx>=numWords:
            allocatedLines.append("")
            continue

        if i==numLines-1:
            allocatedLines.append(" ".join(words[wordIdx:]))
            break

        count = max(1,round(lineWordCounts[i]*scale))
        count = min(count,numWords-wordIdx)
        allocatedLines.append(" ".join(words[wordIdx:wordIdx+count]))
        wordIdx+=count

    return allocatedLines


#blanks out any long straight printed rule left inside the writing region
def RemoveStraightFormLines(hwRegion):
    cleaned = hwRegion.copy()
    regionH,regionW = cleaned.shape

    for r in range(regionH):
        row = cleaned[r,:]>0
        if LongestRun(row)>regionW*0.25 and np.sum(row)>regionW*0.10:
            cleaned[max(0,r-2):min(regionH,r+3),:] = 0

    return cleaned


#merges a tiny detected "line" (descender tail, stray mark -- reliably
#distinguishable from a real line by height alone) into its closer neighbor
def MergeUndersizedFragments(lines,minRelativeHeight=0.4):
    if len(lines)<=1:
        return lines

    heights = [e-s for s,e in lines]
    medianHeight = sorted(heights)[len(heights)//2]
    threshold = medianHeight*minRelativeHeight

    merged = list(lines)
    changed = True
    while changed and len(merged)>1:
        changed = False
        for i,(s,e) in enumerate(merged):
            if (e-s)>=threshold:
                continue
            if i==0:
                neighbor = 1
            elif i==len(merged)-1:
                neighbor = i-1
            else:
                gapPrev = merged[i][0]-merged[i-1][1]
                gapNext = merged[i+1][0]-merged[i][1]
                neighbor = i-1 if gapPrev<=gapNext else i+1

            lo,hi = min(i,neighbor),max(i,neighbor)
            combined = (min(merged[lo][0],merged[hi][0]),max(merged[lo][1],merged[hi][1]))
            merged = merged[:lo]+[combined]+merged[hi+1:]
            changed = True
            break

    return merged


#merges the cheapest-gap pair of lines down to expectedLineCount, or
#splits the tallest line (at its thinnest-ink row) up to it
def MergeOrSplitToExpectedLineCount(lines,rowInkCount,expectedLineCount):
    if expectedLineCount is None or expectedLineCount<=0 or len(lines)==0:
        return lines

    adjusted = list(lines)

    while len(adjusted)>expectedLineCount:
        gapScores = []
        for i in range(len(adjusted)-1):
            prevS,prevE = adjusted[i]
            currS,currE = adjusted[i+1]
            gap = currS-prevE
            gapInk = np.mean(rowInkCount[prevE:currS]) if currS>prevE else 0
            gapScores.append((gap+gapInk,i))

        _,mergeIdx = min(gapScores)
        adjusted[mergeIdx] = (adjusted[mergeIdx][0],adjusted[mergeIdx+1][1])
        del adjusted[mergeIdx+1]

    while len(adjusted)<expectedLineCount:
        splittable = [(e-s,i,s,e) for i,(s,e) in enumerate(adjusted) if (e-s)>=42]
        if not splittable:
            break

        _,splitIdx,s,e = max(splittable)
        lo,hi = s+18,e-18
        if hi>lo:
            splitY = lo+int(np.argmin(rowInkCount[lo:hi]))
        else:
            splitY = (s+e)//2
        adjusted[splitIdx:splitIdx+1] = [(s,splitY),(splitY,e)]

    return adjusted


#estimates the dominant line-to-line spacing via autocorrelation of the
#smoothed row-ink profile, for the periodicity-based splitter below
def EstimateLinePitch(smoothedRowInk,minPitch=40,maxPitch=400):
    sig = smoothedRowInk-smoothedRowInk.mean()
    autocorr = np.correlate(sig,sig,mode='full')[len(sig)-1:]
    maxPitch = min(maxPitch,len(autocorr))
    if maxPitch<=minPitch:
        return minPitch
    search = autocorr[minPitch:maxPitch]
    return minPitch+int(np.argmax(search))


#splits densely-written pages (no blank row between lines, e.g. cursive
#bridging ascenders/descenders) by the ink profile's repeating relative
#low points instead of a blank-gap detector. ruleLineHints lightly snaps
#a nearby valley to a detected ruled line if one exists close by
def SplitLinesByPeriodicity(binaryRegion,ruleLineHints=None,smoothWindow=9,minLineHeight=30):
    rowInk = np.sum(binaryRegion>0,axis=1).astype(np.float32)
    if rowInk.max()<=0:
        return []

    kernel = np.ones(smoothWindow)/smoothWindow
    smoothed = np.convolve(rowInk,kernel,mode='same')

    pitch = EstimateLinePitch(smoothed)
    minDistance = max(minLineHeight,int(pitch*0.55))
    peaks,_ = find_peaks(-smoothed,distance=minDistance,prominence=max(1.0,rowInk.max()*0.03))

    if ruleLineHints:
        snapWindow = max(6,int(pitch*0.12))
        peaks = list(peaks)
        for i,p in enumerate(peaks):
            nearby = [r for r in ruleLineHints if abs(r-p)<=snapWindow]
            if nearby:
                peaks[i] = min(nearby,key=lambda r: abs(r-p))
        peaks = np.array(sorted(set(peaks)))

    boundaries = [0]+list(peaks)+[len(rowInk)]
    lines = [(boundaries[i],boundaries[i+1]) for i in range(len(boundaries)-1)]
    #drop segments with no ink at all
    lines = [(s,e) for s,e in lines if np.sum(rowInk[s:e])>0]
    return lines


#full page-to-line-crops pipeline, the main entry point for this file.
#isDataset=True (default) assumes the IAM form layout (printed prompt box
#bounded by rule lines, OCR'd for ground truth); isDataset=False is for a
#personal non-IAM page (no printed prompt box) -- skips that header/footer
#detection entirely (it actively misfires on ordinary ruled paper) and
#uses almost the whole page, splitting lines by periodicity instead
def ExtractLinePatches(imgPath,targetHeight=32,maxWidth=1024,expectedLineCount=None,labelLines=None,
                        isDataset=True):
    rawImage = Image.open(imgPath)
    RawGrayscaleArray = np.array(rawImage.convert('L'))
    h,w = RawGrayscaleArray.shape

    BinaryInvertedImage = OtsuThreshold(RawGrayscaleArray)

    if not isDataset:
        #no IAM-style printed prompt box to bound against -- use almost the
        #whole page, trimming a small margin for scan/photo edge artifacts
        margin = int(h*0.01)
        topY,botY = margin,h-margin
        headerTopY = None
        printedWords = []
    else:
        topY,botY = DetectPageBounds(BinaryInvertedImage,RawGrayscaleArray)
        headerTopY = DetectHeaderTopY(BinaryInvertedImage,RawGrayscaleArray,topY)

        printedWords = ExtractPrintedGroundTruth(RawGrayscaleArray,topY,headerTopY)

    #handwriting extraction starts this many px below topY, a hard buffer
    #so a rule-line detection off by a few px can't pull header pixels into
    #a line crop (at the negligible cost of a sliver off the true first line)
    HANDWRITING_TOP_MARGIN = 15
    handwritingStartY = min(botY,topY+HANDWRITING_TOP_MARGIN)

    hwRegion = BinaryInvertedImage[handwritingStartY:botY,:].copy()
    regionH,regionW = hwRegion.shape

    hwRegion = RemoveStraightFormLines(hwRegion)

    #mask left vertical margin lines and edge artifacts
    colInkHeights = np.sum(hwRegion>0,axis=0)
    vertLineCols = np.where(colInkHeights>regionH*0.40)[0]
    hwRegion[:,vertLineCols] = 0
    hwRegion[:,:int(w*0.07)] = 0
    hwRegion[:,int(w*0.97):] = 0

    rowInkCount = np.sum(hwRegion>0,axis=1)

    if not isDataset:
        #densely-written page -- periodicity splitter, blank rows aren't safe to assume
        mergedLines = SplitLinesByPeriodicity(hwRegion)
        mergedLines = MergeUndersizedFragments(mergedLines)
    else:
        hasInk = rowInkCount>max(10,int(w*0.006))

        pad1D = np.pad(hasInk,(3,3),mode='constant')
        smoothedInk = np.zeros_like(hasInk)
        for dy in range(7):
            smoothedInk = np.logical_or(smoothedInk,pad1D[dy:dy+len(hasInk)])

        diff = np.diff(np.concatenate(([0],smoothedInk.astype(np.int8),[0])))
        lineStarts,lineEnds = np.where(diff==1)[0],np.where(diff==-1)[0]

        rawLines = [(s,e) for s,e in zip(lineStarts,lineEnds) if (e-s)>=8]

        #merge descender fragments ('y','p','g')
        mergedLines = []
        for line in rawLines:
            if not mergedLines:
                mergedLines.append(line)
            else:
                prevS,prevE = mergedLines[-1]
                currS,currE = line
                if (currS-prevE)<18:
                    mergedLines[-1] = (prevS,currE)
                else:
                    mergedLines.append(line)

        mergedLines = MergeUndersizedFragments(mergedLines)
        mergedLines = MergeOrSplitToExpectedLineCount(mergedLines,rowInkCount,expectedLineCount)

    drawCanvas = rawImage.convert('RGB')
    drawObj = ImageDraw.Draw(drawCanvas)
    drawObj.line([(0,topY),(w,topY)],fill=(0,0,255),width=2)
    drawObj.line([(0,handwritingStartY),(w,handwritingStartY)],fill=(0,200,255),width=1)
    drawObj.line([(0,botY),(w,botY)],fill=(0,0,255),width=2)

    ExtractedLineSamples = []
    lineWidths = []
    lineWordCounts = []

    for idx,(s,e) in enumerate(mergedLines):
        #offset from handwritingStartY (not topY), since hwRegion/s/e start there
        absY1 = handwritingStartY+max(0,s-5)
        absY2 = handwritingStartY+min(regionH,e+5)
        cropH = absY2-absY1

        lineBinary = hwRegion[max(0,s-3):min(regionH,e+3),:]
        colSum = np.sum(lineBinary>0,axis=0)
        inkCols = np.where(colSum>0)[0]
        if len(inkCols)==0: continue

        absX1 = max(0,inkCols[0]-8)
        absX2 = min(w,inkCols[-1]+8)
        cropW = absX2-absX1
        lineWidths.append(cropW)
        lineWordCounts.append(CountWordsFromInkColumns(inkCols,cropH))

        #numbered red box on the preview, matching this line's returned index
        drawObj.rectangle([absX1,absY1,absX2,absY2],outline=(255,0,0),width=2)
        drawObj.text((absX1,max(0,absY1-14)),str(idx),fill=(0,128,0))

        rawLineCrop = RawGrayscaleArray[absY1:absY2,absX1:absX2]
        scaleRatio = targetHeight/float(cropH)
        newWidth = min(maxWidth,max(1,int(cropW*scaleRatio)))

        cropPIL = Image.fromarray(rawLineCrop)
        resizedPIL = cropPIL.resize((newWidth,targetHeight),Image.Resampling.LANCZOS)

        paddedCanvas = Image.new('L',(maxWidth,targetHeight),color=255)
        paddedCanvas.paste(resizedPIL,(0,0))

        ExtractedLineSamples.append({
            'line_idx': idx,
            'raw_crop': rawLineCrop,
            'processed_patch': paddedCanvas,
            'label_text': labelLines[idx] if labelLines is not None and idx<len(labelLines) else ""
        })

    formattedTextLines = labelLines if labelLines is not None and len(labelLines)==len(ExtractedLineSamples) else SplitWordsIntoLines(printedWords,lineWordCounts)

    return ExtractedLineSamples,formattedTextLines,drawCanvas

#standalone script: extracts every page in author 150's folder, saving
#numbered preview images + line crops (+ labels where available)
if __name__=="__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    #SCRIPT_DIR's own parent-parent is the repo root (same depth here as the
    #original Software/CNN location -- CleanPipeline is a sibling of CNN)
    AUTHOR_150_DIR = os.path.abspath(os.path.join(
        SCRIPT_DIR,"..","..","Data","Datasets","IAMpages10","150"
    ))

    outputDir = os.path.join(SCRIPT_DIR,"handwriting_line_extraction_output")
    previewsDir = os.path.join(outputDir,"previews")
    cropsBaseDir = os.path.join(outputDir,"line_crops")

    os.makedirs(previewsDir,exist_ok=True)
    os.makedirs(cropsBaseDir,exist_ok=True)

    if os.path.exists(AUTHOR_150_DIR):

        imagePaths = sorted(glob.glob(os.path.join(AUTHOR_150_DIR,"*.png")))
        print(f"Found {len(imagePaths)} images in '{AUTHOR_150_DIR}' for processing...\n")

        passedCount = 0

        for imgPath in imagePaths:
            baseName = os.path.splitext(os.path.basename(imgPath))[0]
            labelLines = ReadLabelLines(imgPath)
            expectedLineCount = len(labelLines) if labelLines else None
            print(f"-> Processing '{baseName}.png'...")

            samples,textLines,previewImg = ExtractLinePatches(
                imgPath,
                expectedLineCount=expectedLineCount,
                labelLines=labelLines if labelLines else None
            )

            if previewImg is not None and len(samples)>0:

                previewPath = os.path.join(previewsDir,f"{baseName}_preview.png")
                previewImg.save(previewPath)

                pageCropsDir = os.path.join(cropsBaseDir,baseName)
                os.makedirs(pageCropsDir,exist_ok=True)

                for s in samples:
                    lineFilename = f"line_{s['line_idx']:02d}.png"
                    s['processed_patch'].save(os.path.join(pageCropsDir,lineFilename))

                    if s.get('label_text'):
                        labelFilename = f"line_{s['line_idx']:02d}.txt"
                        with open(os.path.join(pageCropsDir,labelFilename),"w",encoding="utf-8") as f:
                            f.write(s['label_text']+"\n")

                if expectedLineCount is not None and len(samples)==expectedLineCount:
                    passedCount+=1
                    countStatus = "PASS"
                elif expectedLineCount is not None:
                    countStatus = "FAIL"
                else:
                    countStatus = "NO LABEL"

                expectedText = expectedLineCount if expectedLineCount is not None else "?"
                print(f"   [{countStatus}] Extracted {len(samples)} lines | Expected {expectedText} | Preview saved: {os.path.basename(previewPath)}")
            else:
                print(f"   [Warning] No lines detected in '{baseName}.png'.")

        print(f"\n[Finished!] Processed all {len(imagePaths)} images.")
        print(f" Label-count matches: {passedCount}/{len(imagePaths)}")
        print(f" All Bounding Box Previews saved to:\n   {previewsDir}")
        print(f" All Extracted Line Crops saved to:\n   {cropsBaseDir}")
    else:
        print(f"Error: Could not find directory at '{AUTHOR_150_DIR}'")
