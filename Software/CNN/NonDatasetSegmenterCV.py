"""
NonDatasetSegmenterCV.py  (library / OpenCV version)

Label-free page -> line segmentation for personal (non-IAM) handwriting photos.

Design (identical algorithm to NonDatasetSegmenterFP.py, which reimplements
every image operation from first principles in numpy):

  1.  Load + EXIF + scale-normalize to TARGET_LONG_SIDE.
  2.  PAGE DETECTION: the paper is the largest bright region of the photo.
      Everything outside it (desk, keyboards, screens, hands) is discarded.
  3.  Illumination correction (divide by heavy blur) on the page region.
  4.  Ink binarization (adaptive threshold), red-ink masking (teacher marks,
      margin lines), speckle removal.
  5.  DESKEW: search rotation maximizing row-projection "peakiness".
  6.  RULE-LINE REMOVAL: printed/drawn ruled lines and long vertical margin
      lines are detected and subtracted.
  7.  COMPONENT ANALYSIS: connected components; per-component MESS scoring
      (diagrams/sketches) using size, fill ratio, enclosed-hole area.
  8.  LINE GROUPING: glyph components chain left-to-right into lines with
      local baseline curves.  Line masks are NON-RECTANGULAR: a line owns
      exactly its own components, so overlapping ascenders/descenders of
      neighbouring lines do not bleed into each other's crops.
  9.  Ordered output: text lines and MESS blocks interleaved by y position.
      Each text line is rendered onto a white canvas from its own component
      mask only.

Output interface (both versions):
    ProcessPage(imgPath) -> (results, previewPIL, meta)
      results: ordered list of dicts:
        {order, tag: 'TEXT'|'MESS', bbox:(x1,y1,x2,y2), raw_crop: np.uint8 HxW
         grayscale (white background), n_components}
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageOps

TARGET_LONG_SIDE = 2400


# ---------------------------------------------------------------------------
# Stage 1: load + scale normalize
# ---------------------------------------------------------------------------
def LoadImage(imgPath, targetLongSide=TARGET_LONG_SIDE):
    img = Image.open(imgPath)
    img = ImageOps.exif_transpose(img).convert('RGB')
    w, h = img.size
    scale = targetLongSide / float(max(w, h))
    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                     Image.Resampling.LANCZOS)
    return np.array(img)  # HxWx3 uint8


# ---------------------------------------------------------------------------
# Stage 2: page detection -- largest bright region
# ---------------------------------------------------------------------------
def DetectPageMask(rgb):
    """The paper is the largest bright, LOW-SATURATION region containing the
    image centre area.  Colour-cluttered background (wood, keyboards, screens)
    is either darker or more saturated than paper."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    blur = cv2.GaussianBlur(gray, (0, 0), 4)
    otsuVal, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    paperish = (blur >= otsuVal) & (sat < 90)
    # light open to break thin bridges to bright background objects
    paperish = cv2.morphologyEx(paperish.astype(np.uint8), cv2.MORPH_OPEN,
                                np.ones((9, 9), np.uint8)).astype(bool)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        paperish.astype(np.uint8), connectivity=4)
    if n <= 1:
        return np.ones(gray.shape, bool)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == best)

    # PASS 2: re-threshold against the page's own brightness, so mid-gray
    # background (fabric, wood) that slipped past a low global Otsu is cut.
    paperMed = float(np.median(blur[mask]))
    paperish2 = (blur >= paperMed - 50) & (sat < 90)
    paperish2 = cv2.morphologyEx(paperish2.astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((9, 9), np.uint8)).astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        paperish2.astype(np.uint8), connectivity=4)
    if n > 1:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == best)
    mask = _FillHoles(mask)
    # close small nicks, then fill again
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((15, 15), np.uint8)).astype(bool)
    mask = _FillHoles(mask)

    # trim ragged bleed: keep the largest contiguous row block whose mask
    # width is a substantial fraction of the page's typical width; same for
    # columns.  Cuts background (fabric/desk) that grazed the thresholds.
    mask = _TrimRagged(mask)

    # brightness trim: background (fabric, cork, desk) inside the mask is
    # darker than paper -- drop rows/cols whose masked median brightness is
    # far below the page's, then keep the largest contiguous block.
    mask = _TrimDarkEdges(mask, blur)

    # erode so the page's own dark edge/shadow border is excluded
    er = max(4, int(min(mask.shape) * 0.018))
    mask = cv2.erode(mask.astype(np.uint8), np.ones((er, er), np.uint8)).astype(bool)

    if mask.mean() < 0.15:
        return np.ones(gray.shape, bool)
    return mask


def _TrimDarkEdges(mask, blur):
    if not mask.any():
        return mask
    pageMed = float(np.median(blur[mask]))
    cut = pageMed - 45

    for axis in (0, 1):
        med = np.full(mask.shape[axis], pageMed)
        idx = np.nonzero(mask.sum(axis=1 - axis) > 0)[0]
        for i in idx:
            sel = mask[i] if axis == 0 else mask[:, i]
            vals = (blur[i] if axis == 0 else blur[:, i])[sel]
            med[i] = np.median(vals)
        good = med >= cut
        # largest contiguous good block
        best, bestSpan, i = 0, (0, len(good)), 0
        while i < len(good):
            if good[i]:
                s = i
                while i < len(good) and good[i]:
                    i += 1
                if i - s > best:
                    best, bestSpan = i - s, (s, i)
            else:
                i += 1
        keep = np.zeros(len(good), bool)
        keep[bestSpan[0]:bestSpan[1]] = True
        mask = mask & (keep[:, None] if axis == 0 else keep[None, :])
    return mask


def _TrimRagged(mask):
    for axis in (1, 0):
        prof = mask.sum(axis=axis).astype(np.float64)
        if prof.max() <= 0:
            return mask
        good = prof >= 0.55 * np.median(prof[prof > prof.max() * 0.2])
        # largest contiguous run of good rows/cols
        best, bestSpan = 0, (0, len(good))
        i = 0
        while i < len(good):
            if good[i]:
                s = i
                while i < len(good) and good[i]:
                    i += 1
                if i - s > best:
                    best, bestSpan = i - s, (s, i)
            else:
                i += 1
        keep = np.zeros(len(good), bool)
        keep[bestSpan[0]:bestSpan[1]] = True
        if axis == 1:
            mask = mask & keep[:, None]
        else:
            mask = mask & keep[None, :]
    return mask


def _FillHoles(mask):
    h, w = mask.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    inv = (~mask).astype(np.uint8)
    cv2.floodFill(inv, ff, (0, 0), 2)
    return mask | (inv == 1)


# ---------------------------------------------------------------------------
# Stage 3+4: illumination correction, binarization, red masking
# ---------------------------------------------------------------------------
def CorrectIllumination(gray):
    bg = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), gray.shape[1] / 30.0)
    corr = gray.astype(np.float32) / (bg + 1e-3) * 255.0
    return np.clip(corr, 0, 255).astype(np.uint8)


def RedInkMask(rgb):
    """Red / dark-red annotation ink (teacher marks, margin rules, stamps).
    Catches saturated red AND darker maroon strokes where red only moderately
    dominates but green/blue are clearly suppressed."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    bright = (r > g * 1.30) & (r > b * 1.30) & ((r - np.minimum(g, b)) > 22)
    dark = (r > g * 1.18) & (r > b * 1.18) & ((r - np.minimum(g, b)) > 14) & (r < 190)
    return bright | dark


def BinarizeInk(illum, pageMask):
    blockSize = 2 * (illum.shape[1] // 60) + 1
    binv = cv2.adaptiveThreshold(illum, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, max(15, blockSize), 12)
    ink = (binv > 0) & pageMask
    return ink


def RemoveEdgeComponents(ink, pageMask, textH):
    """Drop narrow components hugging the page mask's left/right edge: the
    sliver of an adjacent page, spine shadows, and rule stubs running off the
    page edge.  Genuine text starts inside the margin."""
    cols = np.nonzero(pageMask.any(axis=0))[0]
    if len(cols) == 0:
        return ink
    mx1, mx2 = int(cols.min()), int(cols.max())
    w = ink.shape[1]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8)
    keep = np.ones(n, bool)
    for i in range(1, n):
        cw = stats[i, cv2.CC_STAT_WIDTH]
        cx = stats[i, cv2.CC_STAT_LEFT]
        if cw >= w * 0.15:
            continue
        if cx <= mx1 + 8 or cx + cw >= mx2 - 8:
            keep[i] = False
    keep[0] = False
    return keep[labels] & ink


def BreakRuleNetworks(ink, textH):
    """Safety net: a single component spanning most of the page in BOTH
    directions with very low fill is a residual rules/margin network that
    chained text together.  Strip its moderately-long thin runs so the text
    riding on it separates.  A real drawing never spans the whole page at
    ~2% fill."""
    h, w = ink.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8)
    kill = np.zeros_like(ink)
    for i in range(1, n):
        cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        big = (ch > h * 0.5 and cw > w * 0.5) or (ch > h * 0.25 and cw > w * 0.6)
        if not big:
            continue
        if area / float(ch * cw) > 0.055:
            continue
        kill |= (labels == i)
    if not kill.any():
        return ink
    hRun = _HorizontalRunLengths(kill)
    vRun = _VerticalRunLengths(kill)
    thin = ((hRun >= textH) & (vRun <= 6)) | ((vRun >= textH) & (hRun <= 6))
    return ink & ~(kill & thin)


def RemoveSpeckles(ink, minSize=10):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minSize
    return keep[labels]


# ---------------------------------------------------------------------------
# Stage 5: deskew
# ---------------------------------------------------------------------------
def EstimateSkew(ink, searchRange=5.0):
    small = cv2.resize(ink.astype(np.uint8) * 255, None, fx=0.25, fy=0.25,
                       interpolation=cv2.INTER_AREA)
    h, w = small.shape
    center = (w / 2.0, h / 2.0)

    def score(angle):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(small, M, (w, h), flags=cv2.INTER_NEAREST)
        prof = (rot > 0).sum(axis=1).astype(np.float64)
        return float(np.var(prof))

    best, bestS = 0.0, -1.0
    for a in np.arange(-searchRange, searchRange + 1e-9, 0.5):
        s = score(a)
        if s > bestS:
            bestS, best = s, float(a)
    for a in np.arange(best - 0.5, best + 0.5 + 1e-9, 0.1):
        s = score(a)
        if s > bestS:
            bestS, best = s, float(a)
    return best


def Rotate(arr, angle, isMask=False, fill=0):
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
    flags = cv2.INTER_NEAREST if isMask else cv2.INTER_LINEAR
    return cv2.warpAffine(arr, M, (w, h), flags=flags, borderValue=fill)


# ---------------------------------------------------------------------------
# Stage 6: rule-line removal (horizontal ruled lines + vertical margin lines)
# ---------------------------------------------------------------------------
def _HorizontalRunLengths(ink):
    """Per-pixel length of the horizontal ink run the pixel belongs to."""
    h, w = ink.shape
    a = ink.astype(np.int32)
    starts = np.zeros_like(a)
    starts[:, 0] = a[:, 0]
    starts[:, 1:] = (a[:, 1:] == 1) & (a[:, :-1] == 0)
    runId = np.cumsum(starts.reshape(-1)).reshape(h, w)
    runId[a == 0] = 0
    if runId.max() == 0:
        return np.zeros_like(a)
    counts = np.bincount(runId.reshape(-1))
    counts[0] = 0
    return counts[runId]


def _VerticalRunLengths(ink):
    return _HorizontalRunLengths(ink.T).T


def UnderlineLike(c, textH, labels=None):
    """A single wide stroke: an underline / squiggle divider, never a diagram.
    Test: at most columns, the ink occupies ONE thin vertical band.  A
    diagram of the same width (e.g. two parallel long edges of a drawn box)
    has a large per-column vertical spread instead."""
    if c['w'] <= 4.0 * textH or c['area'] / max(1, c['w']) > 6.5:
        return False
    if labels is None:
        return True
    sub = labels[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']] == c['id']
    cols = np.nonzero(sub.any(axis=0))[0]
    if len(cols) == 0:
        return True
    ys = np.arange(sub.shape[0])[:, None]
    inkY = np.where(sub, ys, -1)
    ymax = inkY.max(axis=0)
    inkY2 = np.where(sub, ys, 10 ** 9)
    ymin = inkY2.min(axis=0)
    spread = (ymax - ymin)[cols]
    return float(np.median(spread)) <= max(8.0, textH * 0.4)


def RemoveRuleLines(ink, textH, illum=None):
    """Rule lines = long thin horizontal runs; margin lines = long thin
    vertical runs.  Thin candidate stubs are CHAINED across the page (rules
    get broken into stubs wherever handwriting crosses them); a chain is a
    rule only if it spans most of the page AND reaches both margins -- a
    diagram's long strokes are long but local, so they survive.  Only thin
    pixels are ever removed, so letter strokes crossing a rule keep their
    vertical parts."""
    hRun = _HorizontalRunLengths(ink)
    vRun = _VerticalRunLengths(ink)

    h, w = ink.shape

    minRuleLen = max(50, int(textH * 3.5))
    longH = (hRun >= minRuleLen)
    ruleThk = float(np.median(vRun[longH])) if longH.any() else 2.0
    thkCut = max(4, int(ruleThk * 1.8))

    candH = (hRun >= max(20, int(textH * 0.8))) & (vRun <= thkCut)
    ruleH = _ChainLongStructures(candH, minSpan=w * 0.6, axis=0, textH=textH,
                                 thk=thkCut, lo=w * 0.15, hi=w * 0.85, illum=illum)

    longV = vRun >= max(60, int(textH * 4.0))
    vThk = max(4, int((float(np.median(hRun[longV])) if longV.any() else 2.0) * 1.8))
    candV = (vRun >= max(20, int(textH * 0.8))) & (hRun <= vThk)
    ruleV = _ChainLongStructures(candV, minSpan=h * 0.45, axis=1, textH=textH,
                                 thk=vThk, lo=h * 0.2, hi=h * 0.8, illum=illum)

    ruleMask = ruleH | ruleV
    ruleMask = cv2.dilate(ruleMask.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
    ruleMask &= ((vRun <= thkCut + 2) | (hRun <= vThk + 2))
    return ink & ~ruleMask, ruleMask


def _ChainLongStructures(cand, minSpan, axis, textH, thk, lo=None, hi=None,
                         illum=None):
    """Group thin candidate stubs into chains along `axis` (0=horizontal).
    A chain whose total extent exceeds minSpan is a rule/margin line: all its
    stubs are removed.  Handles rules broken by handwriting crossing them,
    plus sloped/curved rules that defeat single-run length tests."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(
        cand.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(cand)
    comps = []
    for i in range(1, n):
        x, y, wc, hc, area = stats[i]
        d = dict(id=i, x=int(x), y=int(y), w=int(wc), h=int(hc),
                 cx=float(cents[i][0]), cy=float(cents[i][1]))
        if illum is not None:
            sub = labels[y:y + hc, x:x + wc] == i
            d['dark'] = 255.0 - float(np.median(illum[y:y + hc, x:x + wc][sub]))
        comps.append(d)
    if axis == 1:  # work in transposed coords for vertical chains
        for c in comps:
            c['x'], c['y'], c['w'], c['h'] = c['y'], c['x'], c['h'], c['w']
            c['cx'], c['cy'] = c['cy'], c['cx']

    comps.sort(key=lambda c: c['x'])
    uf = _UnionFind(len(comps))
    for i, a in enumerate(comps):
        for j in range(i + 1, len(comps)):
            b = comps[j]
            gap = b['x'] - (a['x'] + a['w'])
            if gap > textH * 2.5:
                break
            dy = abs(b['cy'] - a['cy'])
            slopeAllow = thk * 2 + 0.08 * max(0, gap) + 0.04 * (a['w'] + b['w'])
            if dy > slopeAllow:
                continue
            # darkness gate: a dark pen stroke (diagram edge, underline) must
            # not chain with much lighter printed rule stubs
            if 'dark' in a and abs(a['dark'] - b['dark']) > 55:
                continue
            uf.union(i, j)

    groups = {}
    for i in range(len(comps)):
        groups.setdefault(uf.find(i), []).append(comps[i])

    removeIds = set()
    for g in groups.values():
        x1 = min(c['x'] for c in g)
        x2 = max(c['x'] + c['w'] for c in g)
        if (x2 - x1) < minSpan:
            continue
        if lo is not None and x1 > lo:
            continue
        if hi is not None and x2 < hi:
            continue
        removeIds.update(c['id'] for c in g)

    keep = np.zeros(n, bool)
    for i in removeIds:
        keep[i] = True
    return keep[labels]


def FilterFaintComponents(ink, illum):
    """Printed notebook rules, bleed-through and shadows are much FAINTER than
    pen ink.  Split component darkness contrasts with a 1D Otsu and keep only
    the dark cluster -- but only if the two clusters are well separated
    (a clean page with uniformly dark ink must not lose its lightest word)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8)
    if n <= 2:
        return ink
    darkness = np.zeros(n, np.float64)
    for i in range(1, n):
        x, y, w, h = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
        sub = labels[y:y + h, x:x + w] == i
        vals = illum[y:y + h, x:x + w][sub]
        darkness[i] = 255.0 - float(np.percentile(vals, 25))  # contrast

    d = darkness[1:]
    lo, hi = np.percentile(d, 10), np.percentile(d, 90)
    if hi - lo < 60:          # uniformly-inked page: nothing to filter
        return ink
    thr = _Otsu1D(d)
    if thr <= lo or thr >= hi:
        return ink
    keep = np.zeros(n, bool)
    keep[1:] = darkness[1:] >= thr
    return keep[labels]


def _Otsu1D(values, bins=64):
    hist, edges = np.histogram(values, bins=bins)
    total = values.size
    sumAll = np.dot(np.arange(bins), hist)
    sumB = wB = maxVar = 0.0
    threshBin = 0
    for i in range(bins):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB, mF = sumB / wB, (sumAll - sumB) / wF
        var = wB * wF * (mB - mF) ** 2
        if var > maxVar:
            maxVar, threshBin = var, i
    return float(edges[threshBin + 1])


# ---------------------------------------------------------------------------
# Stage 7: components + MESS scoring
# ---------------------------------------------------------------------------
def ComponentStats(ink):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        comps.append(dict(id=i, x=int(x), y=int(y), w=int(w), h=int(h),
                          area=int(area), cx=float(cents[i][0]), cy=float(cents[i][1])))
    return labels, comps


def EstimateTextHeight(comps):
    """Robust typical glyph-cluster height: median of mid-sized comp heights.
    Excludes extreme-aspect comps (rule/margin line fragments)."""
    hs = np.array([c['h'] for c in comps
                   if c['area'] >= 30 and c['w'] < max(1, c['h']) * 12],
                  dtype=np.float64)
    if len(hs) == 0:
        return 30.0
    med = float(np.median(hs))
    # refine: median of heights within [0.3, 3]x of first estimate
    sel = hs[(hs > med * 0.3) & (hs < med * 3.0)]
    return float(np.median(sel)) if len(sel) else med


def ScoreComponentMess(comp, labels, textH):
    """Per-component diagram-ness score in [0,1].

    Driven by the LARGEST single enclosed hole, relative to the page's glyph
    scale: a hand-drawn box/circle traps ONE hole many times larger than a
    letter counter or an underline pocket -- total hole ratio can't tell a
    heading with an underline from a diagram, but largest-hole-size can.
    Secondary signal: very wide + very sparse strokes (open line-art like a
    long thin rod that never closes a loop)."""
    x, y, w, h = comp['x'], comp['y'], comp['w'], comp['h']
    sub = (labels[y:y + h, x:x + w] == comp['id'])
    area = comp['area']

    padded = np.zeros((h + 2, w + 2), bool)
    padded[1:-1, 1:-1] = sub
    filled = _FillHoles(padded)
    holesMask = filled & ~padded
    maxHole = 0.0
    if holesMask.any():
        nh, hl = cv2.connectedComponents(holesMask.astype(np.uint8), connectivity=4)
        if nh > 1:
            maxHole = float(np.bincount(hl.ravel())[1:].max())
    holeUnits = maxHole / max(1.0, textH * textH)
    holes = max(0.0, min(1.0, (holeUnits - 0.7) / 0.8))

    fill = area / max(1.0, w * h)
    bigAndSparse = (w > textH * 6.0 and h > textH * 1.8 and fill < 0.05)
    sparse = 0.7 if bigAndSparse else 0.0

    return float(max(holes, sparse))


# ---------------------------------------------------------------------------
# Stage 8: line grouping
# ---------------------------------------------------------------------------
def EstimateLinePitch(proj, textH):
    """Dominant line spacing via autocorrelation of the row projection.
    Falls back to textH * 1.7 when no clear periodicity exists."""
    p = proj - proj.mean()
    ac = np.correlate(p, p, mode='full')[len(p) - 1:]
    lo, hi = max(6, int(textH * 0.8)), min(len(ac) - 1, int(textH * 4.0))
    if hi <= lo + 2:
        return textH * 1.7
    seg = ac[lo:hi]
    if seg.max() <= 0:
        return textH * 1.7
    return float(lo + int(np.argmax(seg)))


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _CompMask(labels, c):
    if 'pixmask' in c:
        return c['pixmask']
    return labels[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']] == c['id']


def _LineCurve(comps):
    """Piecewise-linear y(x) through area-weighted centroids, x-sorted."""
    pts = sorted(((c['cx'], c['cy'], c['area']) for c in comps))
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    ws = np.array([p[2] for p in pts], dtype=np.float64)
    sm = np.empty_like(ys)
    for i in range(len(ys)):
        lo, hi = max(0, i - 2), min(len(ys), i + 3)
        sm[i] = np.average(ys[lo:hi], weights=ws[lo:hi])
    return xs, sm


def _CurveY(curve, x):
    xs, ys = curve
    if len(xs) == 1:
        return float(ys[0])
    return float(np.interp(x, xs, ys))


def GroupLines(ink, labels, comps, textH, imgH):
    """Chain-based line building (robust to skew/curvature/perspective):

    1. MESS comps found by hole/sparsity scoring, grouped into blocks;
       comps inside a block's bbox are absorbed into it.
    2. CORE text comps (ordinary glyph height) are chained left-to-right:
       link two comps when horizontally adjacent and vertically aligned.
       Chains follow the actual slope of each written line.
    3. Chains that overlap in x at the same local y merge; each chain gets a
       local baseline CURVE y(x).
    4. Leftover comps: small marks join the nearest curve; interline-touching
       tall comps are split pixel-wise between the curves they span.
    """
    # ---- 1. mess
    for c in comps:
        c['mess'] = ScoreComponentMess(c, labels, textH)
    messCand = [c for c in comps if c['mess'] >= 0.5
                and c['area'] > textH * textH * 0.8 and c['h'] > textH * 1.2]

    # NEIGHBOUR VETO: a candidate embedded in a row of ordinary glyphs is a
    # heading letter-cluster (e.g. big underlined title chars whose underline
    # pockets mimic diagram holes), not a diagram.
    def _NeighborVeto(c):
        neigh = []
        for o in comps:
            if o is c or o['mess'] >= 0.5:
                continue
            if o['h'] < textH * 0.35 or o['h'] > textH * 2.4:
                continue
            if o['area'] < textH * textH * 0.2:
                continue
            if c['x'] <= o['cx'] <= c['x'] + c['w']:
                continue
            oy = max(0, min(c['y'] + c['h'], o['y'] + o['h']) - max(c['y'], o['y']))
            if oy < 0.6 * o['h']:
                continue
            gap = max(o['x'] - (c['x'] + c['w']), c['x'] - (o['x'] + o['w']))
            if gap < textH * 6.0:
                neigh.append(o)
        if len(neigh) < 3:
            return False
        # 75th percentile: capitals/ascenders set a line's true glyph scale;
        # the median gets dragged down by lowercase x-height comps
        refH = float(np.percentile([o['h'] for o in neigh], 75))
        return c['h'] <= 1.9 * refH

    # a decisive closed-shape score (a clean drawn box) overrides the veto --
    # small drawing parts often have other drawing fragments beside them
    messComps = [c for c in messCand if c['mess'] >= 0.85 or not _NeighborVeto(c)]

    messBlocks = []
    for c in sorted(messComps, key=lambda c: -c['area']):
        placed = False
        for b in messBlocks:
            gapY = max(0, max(c['y'], b['y1']) - min(c['y'] + c['h'], b['y2']))
            gapX = max(0, max(c['x'], b['x1']) - min(c['x'] + c['w'], b['x2']))
            if gapY < textH * 0.8 and gapX < textH * 2.0:
                b['comps'].append(c)
                b['y1'], b['y2'] = min(b['y1'], c['y']), max(b['y2'], c['y'] + c['h'])
                b['x1'], b['x2'] = min(b['x1'], c['x']), max(b['x2'], c['x'] + c['w'])
                placed = True
                break
        if not placed:
            messBlocks.append(dict(comps=[c], y1=c['y'], y2=c['y'] + c['h'],
                                   x1=c['x'], x2=c['x'] + c['w']))

    messIds = {c['id'] for c in messComps}
    textComps = [c for c in comps if c['id'] not in messIds]

    # CORE bbox per block: bbox of the block's TALL COLUMNS -- columns where
    # the ink's vertical spread exceeds a glyph height, i.e. actual drawn
    # shapes.  A thin stroke (underline / squiggle / lone rail) running from
    # under a heading into the drawing must not let the block's bbox swallow
    # the heading text beside it.
    for b in messBlocks:
        cx1 = cy1 = 10 ** 9
        cx2 = cy2 = -1
        for c in b['comps']:
            sub = labels[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']] == c['id']
            cols = sub.any(axis=0)
            ys = np.arange(sub.shape[0])[:, None]
            ymax = np.where(sub, ys, -1).max(axis=0)
            ymin = np.where(sub, ys, 10 ** 9).min(axis=0)
            tall = cols & ((ymax - ymin) > textH)
            if not tall.any():
                continue
            tx = np.nonzero(tall)[0]
            cx1 = min(cx1, c['x'] + int(tx.min()))
            cx2 = max(cx2, c['x'] + int(tx.max()) + 1)
            cy1 = min(cy1, c['y'] + int(ymin[tall].min()))
            cy2 = max(cy2, c['y'] + int(ymax[tall].max()) + 1)
        b['hasCore'] = cx2 >= 0
        if b['hasCore']:
            b['cx1'], b['cx2'], b['cy1'], b['cy2'] = cx1, cx2, cy1, cy2
        else:
            b['cx1'], b['cx2'], b['cy1'], b['cy2'] = b['x1'], b['x2'], b['y1'], b['y2']

    def blockFor(c, frac=0.5):
        for b in messBlocks:
            ox = max(0, min(c['x'] + c['w'], b['cx2']) - max(c['x'], b['cx1']))
            oy = max(0, min(c['y'] + c['h'], b['cy2']) - max(c['y'], b['cy1']))
            if ox * oy > frac * c['w'] * c['h']:
                return b
        return None

    # ---- 2. chain core comps.  Wide stroke-thin comps (underlines) never
    # seed or join chains directly -- they attach to the line above later.
    core = [c for c in textComps
            if 0.3 * textH <= c['h'] <= 1.8 * textH and c['area'] >= textH * 1.2
            and not UnderlineLike(c, textH, labels)]
    core.sort(key=lambda c: c['cx'])
    uf = _UnionFind(len(core))
    for i, a in enumerate(core):
        bestJ, bestCost = -1, None
        for j in range(i + 1, len(core)):
            b = core[j]
            gap = b['x'] - (a['x'] + a['w'])
            if gap > textH * 6.0:
                break
            dy = abs(b['cy'] - a['cy'])
            if dy > textH * 0.75:
                continue
            if gap < -a['w'] * 0.6:   # mostly overlapping in x: same word bits
                gap = 0
            cost = max(0, gap) + 3.0 * dy
            if bestCost is None or cost < bestCost:
                bestCost, bestJ = cost, j
        if bestJ >= 0:
            uf.union(i, bestJ)

    chains = {}
    for i, c in enumerate(core):
        chains.setdefault(uf.find(i), []).append(c)
    chainList = [dict(comps=cs) for cs in chains.values()]

    # ---- 3. merge chains sharing the same local y
    def chainSpan(ch):
        return (min(c['x'] for c in ch['comps']),
                max(c['x'] + c['w'] for c in ch['comps']))

    merged = True
    while merged:
        merged = False
        for ch in chainList:
            ch['curve'] = _LineCurve(ch['comps'])
        chainList.sort(key=lambda ch: np.mean(ch['curve'][1]))
        for i in range(len(chainList)):
            for j in range(i + 1, len(chainList)):
                a, b = chainList[i], chainList[j]
                ax1, ax2 = chainSpan(a)
                bx1, bx2 = chainSpan(b)
                o1, o2 = max(ax1, bx1), min(ax2, bx2)
                if o2 <= o1:
                    # disjoint in x: allow merge if the nearer curve ends line up
                    xq = (o1 + o2) / 2.0
                    dy = abs(_CurveY(a['curve'], xq) - _CurveY(b['curve'], xq))
                    if dy < textH * 0.55:
                        a['comps'] += b['comps']
                        del chainList[j]
                        merged = True
                        break
                    continue
                xs = np.linspace(o1, o2, 7)
                dys = [abs(_CurveY(a['curve'], x) - _CurveY(b['curve'], x)) for x in xs]
                if np.mean(dys) < textH * 0.6:
                    a['comps'] += b['comps']
                    del chainList[j]
                    merged = True
                    break
            if merged:
                break

    for ch in chainList:
        ch['curve'] = _LineCurve(ch['comps'])

    # ---- 3b. pitch-aware second merge: fragments split off a line (a word
    # written higher/lower than its neighbours) sit closer than a whole line
    # pitch, so use the page's own measured pitch as the yardstick.
    centers = sorted(float(np.mean(ch['curve'][1])) for ch in chainList)
    gaps = [b - a for a, b in zip(centers, centers[1:]) if b - a > textH * 0.5]
    pitch = float(np.median(gaps)) if gaps else textH * 1.8

    merged = True
    while merged:
        merged = False
        for ch in chainList:
            ch['curve'] = _LineCurve(ch['comps'])
        for i in range(len(chainList)):
            for j in range(len(chainList)):
                if i == j:
                    continue
                a, b = chainList[i], chainList[j]
                ax1, ax2 = chainSpan(a)
                bx1, bx2 = chainSpan(b)
                o1, o2 = max(ax1, bx1), min(ax2, bx2)
                if o2 > o1:
                    xs = np.linspace(o1, o2, 7)
                    dys = [abs(_CurveY(a['curve'], x) - _CurveY(b['curve'], x)) for x in xs]
                    ok = np.mean(dys) < pitch * 0.45
                else:
                    xq = (o1 + o2) / 2.0
                    dy = abs(_CurveY(a['curve'], xq) - _CurveY(b['curve'], xq))
                    ok = dy < pitch * 0.4
                if ok:
                    a['comps'] += b['comps']
                    del chainList[j]
                    merged = True
                    break
            if merged:
                break

    for ch in chainList:
        ch['curve'] = _LineCurve(ch['comps'])

    # ---- 3c. demote weak chains (scribble fragments, stray marks): they must
    # either look like a real line or be clearly isolated on their own band.
    survivors, demoted = [], []
    for ch in chainList:
        cs = ch['comps']
        width = max(c['x'] + c['w'] for c in cs) - min(c['x'] for c in cs)
        strong = len(cs) >= 3 or (width >= 3.0 * textH and
                                  sum(c['area'] for c in cs) >= 2.0 * textH * textH)
        if not strong:
            myY = float(np.mean(ch['curve'][1]))
            nearest = min((abs(float(np.mean(o['curve'][1])) - myY)
                           for o in chainList if o is not ch), default=1e9)
            strong = nearest > pitch * 0.8   # isolated band: keep (e.g. page numbers)
        (survivors if strong else demoted).append(ch)
    chainList = survivors

    # ---- 4. leftovers
    def nearestChain(x, y, maxDy):
        best, bestDy = None, maxDy
        for ch in chainList:
            x1, x2 = chainSpan(ch)
            pen = 0.0
            if x < x1:
                pen = (x1 - x) * 0.15
            elif x > x2:
                pen = (x - x2) * 0.15
            dy = abs(_CurveY(ch['curve'], x) - y) + pen
            if dy < bestDy:
                bestDy, best = dy, ch
        return best

    assignedIds = {id(c) for ch in chainList for c in ch['comps']}
    for c in textComps:
        if id(c) in assignedIds:
            continue
        if UnderlineLike(c, textH, labels):
            # an underline belongs to the words ABOVE it
            best, bestDy = None, textH * 1.5
            for ch in chainList:
                cy = _CurveY(ch['curve'], c['cx'])
                dy = c['cy'] - cy      # positive: underline below the line
                if -0.2 * textH < dy < bestDy:
                    bestDy, best = dy, ch
            if best is not None:
                best['comps'].append(c)
            else:
                b = blockFor(c)
                if b is not None:
                    b['comps'].append(c)
            continue
        if c['h'] <= 1.8 * textH:
            ch = nearestChain(c['cx'], c['cy'], textH * 1.3)
            b = blockFor(c)
            if b is not None and (ch is None or blockFor(c, frac=0.75) is not None):
                b['comps'].append(c)
            elif ch is not None:
                ch['comps'].append(c)
            continue
        # tall interline-touching comp: split pixel-wise between nearest curves
        sub = _CompMask(labels, c)
        ys, xs = np.nonzero(sub)
        gx, gy = xs + c['x'], ys + c['y']
        targets = {}
        for k in range(len(gx)):
            ch = nearestChain(float(gx[k]), float(gy[k]), textH * 2.5)
            targets.setdefault(id(ch) if ch else None, []).append(k)
        for key, idxs in targets.items():
            if key is None:
                continue
            ch = next(cc for cc in chainList if id(cc) == key)
            sel = np.zeros_like(sub)
            sel[ys[idxs], xs[idxs]] = True
            syy, sxx = np.nonzero(sel)
            part = dict(id=c['id'],
                        x=c['x'] + int(sxx.min()), y=c['y'] + int(syy.min()),
                        w=int(sxx.max() - sxx.min() + 1), h=int(syy.max() - syy.min() + 1),
                        area=int(sel.sum()),
                        cx=c['x'] + float(sxx.mean()), cy=c['y'] + float(syy.mean()),
                        pixmask=sel[syy.min():syy.max() + 1, sxx.min():sxx.max() + 1])
            if part['area'] >= 8:
                ch['comps'].append(part)

    # ---- 5. finalize lines
    textLines = []
    for ch in chainList:
        cs = ch['comps']
        totalArea = sum(c['area'] for c in cs)
        maxH = max(c['h'] for c in cs)
        if totalArea < textH * textH * 0.35 or maxH < textH * 0.35:
            continue
        # solid-blob mark filter: stray stamps/stars = 1-3 compact high-fill
        # blobs and nothing glyph-like
        fills = [c['area'] / max(1.0, c['w'] * c['h']) for c in cs]
        width = max(c['x'] + c['w'] for c in cs) - min(c['x'] for c in cs)
        if len(cs) <= 3 and min(fills) > 0.45 and width < textH * 3.5:
            continue
        textLines.append(dict(comps=cs,
                              yc=float(np.average([c['cy'] for c in cs],
                                                  weights=[c['area'] for c in cs]))))

    # ---- 6. merge mess blocks that belong to ONE drawing: overlapping in x
    # with no text line running between them vertically.
    lineCenters = sorted(l['yc'] for l in textLines)
    mergedBlocks = True
    while mergedBlocks:
        mergedBlocks = False
        for i in range(len(messBlocks)):
            for j in range(i + 1, len(messBlocks)):
                a, b = messBlocks[i], messBlocks[j]
                ox = min(a['x2'], b['x2']) - max(a['x1'], b['x1'])
                if ox <= 0:
                    continue
                gy1, gy2 = min(a['y2'], b['y2']), max(a['y1'], b['y1'])
                if gy2 - gy1 > 0 and any(gy1 < yc < gy2 for yc in lineCenters):
                    continue
                if gy2 - gy1 > (max(a['y2'], b['y2']) - min(a['y1'], b['y1'])):
                    continue
                a['comps'] += b['comps']
                a['y1'], a['y2'] = min(a['y1'], b['y1']), max(a['y2'], b['y2'])
                a['x1'], a['x2'] = min(a['x1'], b['x1']), max(a['x2'], b['x2'])
                # core bbox: only genuine drawing comps define it -- an
                # underline/squiggle-only block must not extend it over
                # neighbouring heading text
                if a['hasCore'] and b['hasCore']:
                    a['cy1'], a['cy2'] = min(a['cy1'], b['cy1']), max(a['cy2'], b['cy2'])
                    a['cx1'], a['cx2'] = min(a['cx1'], b['cx1']), max(a['cx2'], b['cx2'])
                elif b['hasCore']:
                    a['cy1'], a['cy2'], a['cx1'], a['cx2'] = b['cy1'], b['cy2'], b['cx1'], b['cx2']
                    a['hasCore'] = True
                del messBlocks[j]
                mergedBlocks = True
                break
            if mergedBlocks:
                break

    # ---- 5b. a "text line" living mostly INSIDE a mess block's bbox is part
    # of the drawing (hatching, axis labels), not a line of prose
    keptLines = []
    for l in textLines:
        cs = l['comps']
        lx1 = min(c['x'] for c in cs); lx2 = max(c['x'] + c['w'] for c in cs)
        ly1 = min(c['y'] for c in cs); ly2 = max(c['y'] + c['h'] for c in cs)
        absorbed = False
        for b in messBlocks:
            if not b['hasCore']:
                continue
            ox = max(0, min(lx2, b['cx2']) - max(lx1, b['cx1']))
            oy = max(0, min(ly2, b['cy2']) - max(ly1, b['cy1']))
            if ox * oy > 0.7 * max(1, (lx2 - lx1) * (ly2 - ly1)):
                b['comps'] += cs
                absorbed = True
                break
        if not absorbed:
            keptLines.append(l)
    textLines = keptLines

    for b in messBlocks:
        b['yc'] = 0.5 * (b['y1'] + b['y2'])

    return textLines, messBlocks


# ---------------------------------------------------------------------------
# Stage 9: rendering + ordering
# ---------------------------------------------------------------------------
def RenderLine(gray, labels, comps, pad=6, inkRaw=None, deskew=False):
    """Render one line onto a white canvas from its own ink.

    * inkRaw: raw binarized ink (before rule removal / faint filtering) with
      thin-horizontal runs stripped.  Pixels of it adjacent to the line's own
      strokes are re-included, so stroke bottoms / descender tips shaved off
      by earlier cleanup come back, while rule slivers stay out (the recovery
      kernel is tall and narrow).
    * deskew: rotate the crop by the line's own fitted slope so a line that
      still rises/falls after the global page deskew comes out level.
    """
    x1 = min(c['x'] for c in comps) - pad
    y1 = min(c['y'] for c in comps) - pad
    x2 = max(c['x'] + c['w'] for c in comps) + pad
    y2 = max(c['y'] + c['h'] for c in comps) + pad
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)

    mask = np.zeros((y2 - y1, x2 - x1), bool)
    for c in comps:
        sy, sx = slice(c['y'] - y1, c['y'] - y1 + c['h']), slice(c['x'] - x1, c['x'] - x1 + c['w'])
        if 'pixmask' in c:
            mask[sy, sx] |= c['pixmask']
        else:
            mask[sy, sx] |= (labels[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']] == c['id'])

    mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if inkRaw is not None:
        near = cv2.dilate(mask.astype(np.uint8), np.ones((9, 3), np.uint8)).astype(bool)
        mask |= (near & inkRaw[y1:y2, x1:x2])

    crop = np.full(mask.shape, 255, np.uint8)
    crop[mask] = gray[y1:y2, x1:x2][mask]

    if deskew and len(comps) >= 3 and crop.shape[1] > 2 * crop.shape[0]:
        cxs = np.array([c['cx'] for c in comps])
        cys = np.array([c['cy'] for c in comps])
        ws = np.sqrt([c['area'] for c in comps])
        slope = np.polyfit(cxs, cys, 1, w=ws)[0]
        angle = float(np.degrees(np.arctan(slope)))
        if 0.4 < abs(angle) < 12.0:
            h, w = crop.shape
            margin = int(abs(np.tan(np.radians(angle))) * w / 2) + 4
            padded = np.full((h + 2 * margin, w), 255, np.uint8)
            padded[margin:margin + h] = crop
            M = cv2.getRotationMatrix2D((w / 2.0, padded.shape[0] / 2.0), angle, 1.0)
            rot = cv2.warpAffine(padded, M, (w, padded.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderValue=255)
            ys, xs = np.nonzero(rot < 250)
            if len(ys):
                a = max(0, ys.min() - pad)
                b = min(rot.shape[0], ys.max() + 1 + pad)
                crop = rot[a:b]

    return crop, (x1, y1, x2, y2)


def ProcessPage(imgPath):
    rgb = LoadImage(imgPath)
    pageMask = DetectPageMask(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    illum = CorrectIllumination(gray)

    ink0 = BinarizeInk(illum, pageMask) & ~RedInkMask(rgb)
    ink0 = RemoveSpeckles(ink0)

    angle = EstimateSkew(ink0)
    if abs(angle) >= 0.15:
        rgb = Rotate(rgb, angle, fill=(255, 255, 255))
        pageMask = Rotate(pageMask.astype(np.uint8), angle, isMask=True).astype(bool)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        illum = CorrectIllumination(gray)
        ink0 = BinarizeInk(illum, pageMask) & ~RedInkMask(rgb)
        ink0 = RemoveSpeckles(ink0)

    # crop-time recovery source: raw ink (before the faint filter and rule
    # removal) minus thin-horizontal runs, so shaved stroke bottoms come back
    # into the crops without re-admitting rule slivers
    hR, vR = _HorizontalRunLengths(ink0), _VerticalRunLengths(ink0)
    inkRaw = ink0 & ~((hR >= 10) & (vR <= 4))

    ink0 = FilterFaintComponents(ink0, illum)
    ink0 = RemoveEdgeComponents(ink0, pageMask, 30)

    _, roughComps = ComponentStats(ink0)
    roughH = EstimateTextHeight(roughComps)

    ink, ruleMask = RemoveRuleLines(ink0, roughH, illum=illum)
    # second pass: with the bulk of the rules gone, run lengths recompute and
    # previously-shielded stubs (thick spots, crossings) become removable
    ink, _rm2 = RemoveRuleLines(ink, roughH, illum=illum)
    ink = RemoveSpeckles(ink, minSize=12)
    ink = BreakRuleNetworks(ink, roughH)
    ink = RemoveSpeckles(ink, minSize=12)

    labels, comps = ComponentStats(ink)
    textH = EstimateTextHeight(comps)
    textLines, messBlocks = GroupLines(ink, labels, comps, textH, ink.shape[0])

    items = [dict(tag='TEXT', yc=l['yc'], comps=l['comps']) for l in textLines]
    items += [dict(tag='MESS', yc=b['yc'], comps=b['comps'],
                   y1=b['y1'], y2=b['y2']) for b in messBlocks]
    items.sort(key=lambda it: it['yc'])
    # reading order: a heading written at the TOP of its diagram's y-range is
    # read before the diagram, even if the diagram's centre sits barely higher
    for k in range(len(items) - 1):
        a, b = items[k], items[k + 1]
        if a['tag'] == 'MESS' and b['tag'] == 'TEXT' and a['y1'] <= b['yc'] <= a['y2']:
            items[k], items[k + 1] = b, a

    results = []
    preview = Image.fromarray(rgb.copy())
    from PIL import ImageDraw
    drawObj = ImageDraw.Draw(preview)
    for order, it in enumerate(items):
        crop, bbox = RenderLine(gray, labels, it['comps'], inkRaw=inkRaw,
                                deskew=(it['tag'] == 'TEXT'))
        color = (255, 140, 0) if it['tag'] == 'MESS' else (0, 190, 0)
        drawObj.rectangle(list(bbox), outline=color, width=3)
        drawObj.text((bbox[0], max(0, bbox[1] - 14)), f"{order}:{it['tag']}", fill=color)
        results.append(dict(order=order, tag=it['tag'], bbox=bbox,
                            raw_crop=crop, n_components=len(it['comps'])))

    meta = dict(skew=angle, textH=textH,
                nText=sum(1 for r in results if r['tag'] == 'TEXT'),
                nMess=sum(1 for r in results if r['tag'] == 'MESS'))
    return results, preview, meta
