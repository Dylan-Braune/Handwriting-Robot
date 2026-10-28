"""Where does the boundary between 't' and 'h' actually land, and who put it there?

Nine of abhinav's twelve stored 'h' glyphs carry a t-crossbar: the cut that
should separate 't' from 'h' is landing to the LEFT of the 't', so the whole
pair ends up in the 'h' cell.  Two things decide that boundary and only one of
them can be at fault:

    the alignment   CtcForcedAlign gives each character a column span.  if the
                    span it reports for 'h' already covers the 't', the reader
                    is wrong about where the letters are and no amount of
                    snapping will save it
    the snapping    _Snap then moves the boundary up to half an x-height to the
                    cheapest column nearby, where cheap means thin ink low down.
                    if the span is right and the snap walks past the 't', the
                    cut rule is at fault

This measures both, for every 'th' in the writer's pages, and prints which.
It changes nothing: it is a probe, run before choosing a fix.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import numpy as np
import paths, recognise as REC, segment as SEG, style as ST, train as TR

WRITER = sys.argv[1] if len(sys.argv) > 1 else "abhinav"
PAIR = sys.argv[2] if len(sys.argv) > 2 else "th"

model = REC.PaperCRNNNumpy(checkpointPath=ST.PERSONAL_WEIGHTS)
print("reader: %s" % ST.PERSONAL_WEIGHTS.name)


#----------------------------------------------------------------------
#Boundaries: ExtractLineGlyphs' own cut arithmetic, reported instead of used
#the code below is copied from ExtractLineGlyphs deliberately rather than
#refactored out of it: a probe that shares code with what it measures can
#be made to agree with it by accident
#----------------------------------------------------------------------
def Boundaries(gray, text):
    t = REC.TensorFromResized(REC.ResizeLineImageFixed(gray))
    lp = model.forward(t)[:, 0, :]
    res = REC.CtcForcedAlign(lp, text)
    if res is None:
        return None
    align, conf = res
    T = lp.shape[0]
    rawW = gray.shape[1]
    ink = SEG.BinarizeLine(gray)
    band = SEG.CoreBand(ink)
    if band is None:
        return None
    top, base = band
    xh = float(base-top)

    colProf = ink.sum(axis=0).astype(np.float64)
    k = max(3, int(round(xh*0.12))) | 1
    colSm = np.convolve(colProf, np.ones(k)/k, mode="same")
    rowsIdx = np.arange(ink.shape[0])[:, None]
    hiY = np.where(ink, rowsIdx, ink.shape[0]).min(axis=0).astype(np.float64)
    topAbove = np.clip((base-hiY)/max(1.0, xh), 0.0, 3.0)
    thin = colSm/max(1e-6, float(np.percentile(colSm[colSm > 0], 60))
                     if (colSm > 0).any() else 1.0)
    cutCost = thin+1.15*topAbove
    cutCost[colProf == 0] = 0.0
    snapR = max(3, int(round(xh*0.5)))

    def Snap(x):
        a = int(np.clip(round(x)-snapR, 0, rawW-1))
        b = int(np.clip(round(x)+snapR+1, 1, rawW))
        if b-a < 2:
            return float(x)
        off = np.abs(np.arange(a, b)-x)/max(1.0, snapR)
        return float(a+int(np.argmin(cutCost[a:b]+0.6*off)))

    origH = gray.shape[0]
    charXs = []
    for ch, sp in align:
        if sp is None:
            charXs.append((ch, None, None))
            continue
        charXs.append((ch,
                       REC.FrameXToPixel(sp[0]/T*REC.INPUT_WIDTH, rawW, origH),
                       REC.FrameXToPixel(sp[1]/T*REC.INPUT_WIDTH, rawW, origH)))
    valid = [(i, c) for i, c in enumerate(charXs) if c[1] is not None]
    if len(valid) < 2:
        return None
    centres = [0.5*(c[1]+c[2]) for _, c in valid]
    nV = len(valid)
    minCell = max(1.0, 0.08*xh)
    bnd = [0.0]*(nV+1)
    bnd[0] = max(0.0, centres[0]-(0.5*(centres[1]-centres[0]) if nV > 1 else xh))
    bnd[nV] = min(float(rawW), centres[-1]+(0.5*(centres[-1]-centres[-2]) if nV > 1 else xh))
    mids = [None]*(nV+1)
    for k2 in range(1, nV):
        mids[k2] = 0.5*(centres[k2-1]+centres[k2])
        s = Snap(mids[k2])
        lo = bnd[k2-1]+minCell
        hi = max(lo, bnd[nV]-(nV-k2)*minCell)
        bnd[k2] = min(max(s, lo), hi)
    return dict(valid=valid, bnd=bnd, mids=mids, xh=xh, conf=conf, snapR=snapR)


rows = []
for imgPath, labelPath in paths.PersonalWriterPages(WRITER):
    gray = TR.ReadImageFile(imgPath)
    lines, _ov, _info = SEG.SegmentLines(gray)
    texts = [g for g in TR.ReadLabelLines(str(imgPath), str(labelPath))
             if g.strip() != "MESS"]
    n = min(len(lines), len(texts))
    for i in range(n):
        b = Boundaries(np.asarray(lines[i]["image"], dtype=np.uint8), texts[i])
        if b is None:
            continue
        valid, bnd, mids, xh = b["valid"], b["bnd"], b["mids"], b["xh"]
        for k2 in range(1, len(valid)):
            chL = valid[k2-1][1][0]
            chR = valid[k2][1][0]
            if chL+chR != PAIR:
                continue
            spanL = (valid[k2-1][1][1], valid[k2-1][1][2])
            spanR = (valid[k2][1][1], valid[k2][1][2])
            cut = bnd[k2]
            #the span the reader gave the right-hand letter is the test: a cut
            #left of its start means the left letter's ink falls in its cell
            rows.append(dict(
                page=imgPath.stem, line=i, xh=xh,
                cut=cut, mid=mids[k2], snapMove=(cut-mids[k2])/xh,
                cutVsRightSpan=(cut-spanR[0])/xh,
                cutVsLeftSpanEnd=(cut-spanL[1])/xh,
                midVsRightSpan=(mids[k2]-spanR[0])/xh,
                rightCellW=(bnd[k2+1]-cut)/xh,
                leftCellW=(cut-bnd[k2-1])/xh))

if not rows:
    raise SystemExit("no %r pairs found in %s's pages" % (PAIR, WRITER))

A = lambda key: np.array([r[key] for r in rows], dtype=float)
nLeftOfSpan = int((A("cutVsRightSpan") < -0.15).sum())
nMidLeftOfSpan = int((A("midVsRightSpan") < -0.15).sum())

report = []
report.append("%s, %r pairs found: %d" % (WRITER, PAIR, len(rows)))
report.append("")
report.append("all distances in x-heights, negative means left of the reference")
report.append("")
report.append("  cut minus the right letter's span start   median %+.2f  mean %+.2f" %
              (np.median(A("cutVsRightSpan")), A("cutVsRightSpan").mean()))
report.append("  alignment midpoint minus that same start  median %+.2f  mean %+.2f" %
              (np.median(A("midVsRightSpan")), A("midVsRightSpan").mean()))
report.append("  how far the snap moved the boundary       median %+.2f  mean %+.2f  worst %+.2f" %
              (np.median(A("snapMove")), A("snapMove").mean(),
               A("snapMove")[np.argmax(np.abs(A("snapMove")))]))
report.append("  cut minus the left letter's span END      median %+.2f   (>0 means it ate into it)" %
              np.median(A("cutVsLeftSpanEnd")))
report.append("")
report.append("  cells: left %.2f xh wide, right %.2f xh wide (median)" %
              (np.median(A("leftCellW")), np.median(A("rightCellW"))))
report.append("")
report.append("  cuts landing left of the right letter's span:        %d of %d (%.0f%%)" %
              (nLeftOfSpan, len(rows), 100.0*nLeftOfSpan/len(rows)))
report.append("  of those, already wrong BEFORE snapping:             %d (%.0f%%)" %
              (nMidLeftOfSpan, 100.0*nMidLeftOfSpan/max(1, len(rows))))
report.append("")
if nLeftOfSpan == 0:
    report.append("VERDICT: the boundary is not the problem for this pair.")
elif nMidLeftOfSpan >= 0.6*nLeftOfSpan:
    report.append("VERDICT: the ALIGNMENT is at fault.  the reader's own span for the "
                  "right letter already starts right of the boundary, so clamping the "
                  "snap cannot fix it.")
else:
    report.append("VERDICT: the SNAPPING is at fault.  the alignment puts the boundary "
                  "in the right place and the search for a cheap column walks past the "
                  "left letter.  clamping the snap to the alignment's spans should fix it.")

report.append("")
report.append("worst ten, by how far the cut sits left of the right letter's span")
report.append("%-14s %5s %8s %8s %8s %9s %9s" %
              ("page", "line", "xh px", "cut-span", "snap move", "left cell", "right cell"))
for r in sorted(rows, key=lambda r: r["cutVsRightSpan"])[:10]:
    report.append("%-14s %5d %8.1f %+8.2f %+9.2f %9.2f %9.2f" %
                  (r["page"], r["line"], r["xh"], r["cutVsRightSpan"],
                   r["snapMove"], r["leftCellW"], r["rightCellW"]))

text = "\n".join(report)
print(text)
out = HERE/("diagnose_%s_%s.txt" % (WRITER, PAIR))
out.write_text(text+"\n", encoding="utf-8")
print("\nwrote %s" % out)
