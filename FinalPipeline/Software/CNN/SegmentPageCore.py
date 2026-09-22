"""
SegmentPageCore.py -- first-principles (numpy + PIL only, no OpenCV/scipy)
page -> line segmenter for personal (non-IAM) handwriting photos, plus its
own test harness. Run this file directly to test it against the sample
pages in NOGIT/NonDatasetImages/.

Pipeline: load/normalize -> page-mask detection -> illumination correction +
binarization -> deskew -> rule-line removal -> connected components + MESS
(diagram) scoring -> chain-based line grouping -> per-line rendering.

ProcessPage(imgPath) -> (results, previewPIL, meta)
  results: ordered list of {order, tag: 'TEXT'|'MESS', bbox, raw_crop, n_components}
"""

import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import RawImageOps as F

TARGET_LONG_SIDE = 2400
INPUT_H, INPUT_W = 64, 640  # keep in sync with TrainText.py


# ---------------------------------------------------------------------------
# Load + normalize
# ---------------------------------------------------------------------------
def LoadImage(imgPath, targetLongSide=TARGET_LONG_SIDE):
    img = Image.open(imgPath)
    img = ImageOps.exif_transpose(img).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]
    scale = targetLongSide / float(max(w, h))
    newH, newW = max(1, int(h * scale)), max(1, int(w * scale))
    out = F.ResizeBilinear(arr, newH, newW)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Page detection: largest bright, low-saturation region
# ---------------------------------------------------------------------------
def DetectPageMask(rgb):
    gray = F.RgbToGray(rgb)
    sat = F.Saturation(rgb)

    blur = np.clip(F.GaussianBlur(gray, 4), 0, 255)
    otsuVal = F.OtsuThresholdValue(blur.astype(np.uint8))

    paperish = (blur >= otsuVal) & (sat < 90)
    paperish = F.Open(paperish, 9, 9)

    labels, n = F.LabelComponents(paperish, connectivity=4)
    if n == 0:
        return np.ones(gray.shape, bool)
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    best = int(np.argmax(areas))
    mask = labels == best

    # re-threshold against the page's own brightness
    paperMed = float(np.median(blur[mask]))
    paperish2 = (blur >= paperMed - 50) & (sat < 90)
    paperish2 = F.Open(paperish2, 9, 9)
    labels, n = F.LabelComponents(paperish2, connectivity=4)
    if n > 0:
        areas = np.bincount(labels.ravel())
        areas[0] = 0
        best = int(np.argmax(areas))
        mask = labels == best
    mask = F.FillHoles(mask)
    mask = F.Close(mask, 15, 15)
    mask = F.FillHoles(mask)

    mask = _TrimRagged(mask)
    mask = _TrimDarkEdges(mask, blur)

    # small erosion so the page's own shadow/edge doesn't get treated as ink
    er = max(4, int(min(mask.shape) * 0.006))
    kk = 2 * (er // 2) + 1
    mask = F.Erode(mask, kk, kk)

    if mask.mean() < 0.15:
        return np.ones(gray.shape, bool)
    return mask


def _LargestGoodBlock(good):
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
    return keep


def _TrimRagged(mask):
    for axis in (1, 0):
        prof = mask.sum(axis=axis).astype(np.float64)
        if prof.max() <= 0:
            return mask
        good = prof >= 0.55 * np.median(prof[prof > prof.max() * 0.2])
        keep = _LargestGoodBlock(good)
        mask = mask & (keep[:, None] if axis == 1 else keep[None, :])
    return mask


def _TrimDarkEdges(mask, blur):
    if not mask.any():
        return mask
    pageMed = float(np.median(blur[mask]))
    cut = pageMed - 45

    for axis in (0, 1):
        n = mask.shape[axis]
        med = np.full(n, pageMed)
        counts = mask.sum(axis=1 - axis)
        idx = np.nonzero(counts > 0)[0]
        for i in idx:
            sel = mask[i] if axis == 0 else mask[:, i]
            vals = (blur[i] if axis == 0 else blur[:, i])[sel]
            med[i] = np.median(vals)
        keep = _LargestGoodBlock(med >= cut)
        mask = mask & (keep[:, None] if axis == 0 else keep[None, :])
    return mask


# ---------------------------------------------------------------------------
# Illumination, binarization, red-ink mask, speckles
# ---------------------------------------------------------------------------
def CorrectIllumination(gray):
    bg = F.GaussianBlur(gray, gray.shape[1] / 30.0)
    corr = gray.astype(np.float64) / (bg + 1e-3) * 255.0
    return np.clip(corr, 0, 255).astype(np.uint8)


def RedInkMask(rgb):
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    bright = (r > g * 1.30) & (r > b * 1.30) & ((r - np.minimum(g, b)) > 22)
    dark = (r > g * 1.18) & (r > b * 1.18) & ((r - np.minimum(g, b)) > 14) & (r < 190)
    return bright | dark


def BinarizeInk(illum, pageMask):
    blockSize = max(15, 2 * (illum.shape[1] // 60) + 1)
    return F.AdaptiveThresholdInv(illum, blockSize, 12) & pageMask


def HysteresisRecoverInk(illum, pageMask, strong, weakC=3):
    """Crop-rendering-only recovery of faint stroke fade (never used for the
    real ink used in segmentation/classification -- see BinarizeInk). Weak
    threshold pixels are kept if within a small proximity of strong ink
    (not strict adjacency: a faded glyph can anti-alias into a weak-only
    blob that never quite touches its own strong stroke)."""
    blockSize = max(15, 2 * (illum.shape[1] // 60) + 1)
    weak = F.AdaptiveThresholdInv(illum, blockSize, weakC) & pageMask
    labels, n = F.LabelComponents(weak, connectivity=8)
    if n == 0:
        return strong
    touchesStrong = np.zeros(n + 1, bool)
    touched = labels[F.Dilate(strong, 9, 25)]
    touched = touched[touched > 0]
    touchesStrong[touched] = True
    return strong | (touchesStrong[labels] & weak)


def RemoveSpeckles(ink, minSize=10):
    labels, n = F.LabelComponents(ink, connectivity=8)
    if n == 0:
        return ink
    areas = np.bincount(labels.ravel())
    keep = areas >= minSize
    keep[0] = False
    return keep[labels]


# ---------------------------------------------------------------------------
# Deskew
# ---------------------------------------------------------------------------
def EstimateSkew(ink, searchRange=5.0):
    small = F.DownsampleArea(ink, 4) > 0

    def score(angle):
        rot = F.Rotate(small.astype(np.uint8), angle, nearest=True, fill=0)
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
    return F.Rotate(arr, -angle, nearest=isMask, fill=fill)


# ---------------------------------------------------------------------------
# Rule-line removal
# ---------------------------------------------------------------------------
def _HorizontalRunLengths(ink):
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
    if c['w'] <= 4.0 * textH or c['area'] / max(1, c['w']) > 6.5:
        return False
    if labels is None:
        return True
    sub = labels[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']] == c['id']
    cols = np.nonzero(sub.any(axis=0))[0]
    if len(cols) == 0:
        return True
    ys = np.arange(sub.shape[0])[:, None]
    ymax = np.where(sub, ys, -1).max(axis=0)
    ymin = np.where(sub, ys, 10 ** 9).min(axis=0)
    spread = (ymax - ymin)[cols]
    return float(np.median(spread)) <= max(8.0, textH * 0.4)


def RemoveRuleLines(ink, textH, illum=None):
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

    candV = (vRun >= max(20, int(textH * 0.8))) & (hRun <= thkCut)
    ruleV = _ChainLongStructures(candV, minSpan=h * 0.45, axis=1, textH=textH,
                                 thk=thkCut, lo=h * 0.2, hi=h * 0.8, illum=illum)

    ruleMask = ruleH | ruleV
    ruleMask = F.Dilate(ruleMask, 2, 2)
    ruleMask &= ((vRun <= thkCut + 2) | (hRun <= thkCut + 2))
    return ink & ~ruleMask, ruleMask


def _ChainLongStructures(cand, minSpan, axis, textH, thk, lo=None, hi=None, illum=None):
    labels, n = F.LabelComponents(cand, connectivity=8)
    if n == 0:
        return np.zeros_like(cand)
    stats = F.ComponentStatsFromLabels(labels, n)
    comps = []
    for i, st in enumerate(stats, start=1):
        if st is None:
            continue
        d = dict(id=i, x=st['x'], y=st['y'], w=st['w'], h=st['h'], cx=st['cx'], cy=st['cy'])
        if illum is not None:
            sub = labels[st['y']:st['y'] + st['h'], st['x']:st['x'] + st['w']] == i
            d['dark'] = 255.0 - float(np.median(
                illum[st['y']:st['y'] + st['h'], st['x']:st['x'] + st['w']][sub]))
        comps.append(d)
    if axis == 1:
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

    keep = np.zeros(n + 1, bool)
    for i in removeIds:
        keep[i] = True
    return keep[labels]


def RemoveEdgeComponents(ink, pageMask, textH):
    """Drop narrow components hugging the page mask's edge (adjacent-page
    slivers, spine shadows, rule stubs) -- real text starts inside the margin."""
    cols = np.nonzero(pageMask.any(axis=0))[0]
    if len(cols) == 0:
        return ink
    mx1, mx2 = int(cols.min()), int(cols.max())
    w = ink.shape[1]
    labels, n = F.LabelComponents(ink, connectivity=8)
    if n == 0:
        return ink
    stats = F.ComponentStatsFromLabels(labels, n)
    keep = np.ones(n + 1, bool)
    for i, st in enumerate(stats, start=1):
        if st is None:
            continue
        if st['w'] >= w * 0.15:
            continue
        touchL = st['x'] <= mx1 + 8
        touchR = st['x'] + st['w'] >= mx2 - 8
        if touchL or touchR:
            keep[i] = False
    keep[0] = False
    return keep[labels] & ink


def BreakRuleNetworks(ink, textH):
    """A component spanning most of the page in both directions at very low
    fill is a residual rule/margin network chaining text together -- strip
    its thin runs so the text separates."""
    h, w = ink.shape
    labels, n = F.LabelComponents(ink, connectivity=8)
    if n == 0:
        return ink
    stats = F.ComponentStatsFromLabels(labels, n)
    kill = np.zeros_like(ink)
    for i, st in enumerate(stats, start=1):
        if st is None:
            continue
        big = (st['h'] > h * 0.5 and st['w'] > w * 0.5) or \
              (st['h'] > h * 0.25 and st['w'] > w * 0.6)
        if not big:
            continue
        if st['area'] / float(st['h'] * st['w']) > 0.055:
            continue
        kill |= (labels == i)
    if not kill.any():
        return ink
    hRun = _HorizontalRunLengths(kill)
    vRun = _VerticalRunLengths(kill)
    thin = ((hRun >= textH) & (vRun <= 6)) | ((vRun >= textH) & (hRun <= 6))
    return ink & ~(kill & thin)


def FilterFaintComponents(ink, illum):
    labels, n = F.LabelComponents(ink, connectivity=8)
    if n <= 1:
        return ink
    stats = F.ComponentStatsFromLabels(labels, n)
    darkness = np.zeros(n + 1, np.float64)
    for i, st in enumerate(stats, start=1):
        if st is None:
            continue
        sub = labels[st['y']:st['y'] + st['h'], st['x']:st['x'] + st['w']] == i
        vals = illum[st['y']:st['y'] + st['h'], st['x']:st['x'] + st['w']][sub]
        darkness[i] = 255.0 - float(np.percentile(vals, 25))

    d = darkness[1:]
    lo, hi = np.percentile(d, 10), np.percentile(d, 90)
    if hi - lo < 60:
        return ink
    thr = _Otsu1D(d)
    if thr <= lo or thr >= hi:
        return ink
    keep = darkness >= thr
    keep[0] = False
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
# Components + MESS scoring
# ---------------------------------------------------------------------------
def ComponentStats(ink):
    labels, n = F.LabelComponents(ink, connectivity=8)
    stats = F.ComponentStatsFromLabels(labels, n)
    comps = []
    for i, st in enumerate(stats, start=1):
        if st is None:
            continue
        comps.append(dict(id=i, x=st['x'], y=st['y'], w=st['w'], h=st['h'],
                          area=st['area'], cx=st['cx'], cy=st['cy']))
    return labels, comps


def EstimateTextHeight(comps):
    hs = np.array([c['h'] for c in comps
                   if c['area'] >= 30 and c['w'] < max(1, c['h']) * 12],
                  dtype=np.float64)
    if len(hs) == 0:
        return 30.0
    med = float(np.median(hs))
    sel = hs[(hs > med * 0.3) & (hs < med * 3.0)]
    return float(np.median(sel)) if len(sel) else med


def ScoreComponentMess(comp, labels, textH):
    """0..1 diagram-ness, driven by largest single enclosed hole relative to
    glyph scale, plus a wide-and-sparse (open line-art) fallback signal."""
    x, y, w, h = comp['x'], comp['y'], comp['w'], comp['h']
    sub = (labels[y:y + h, x:x + w] == comp['id'])
    area = comp['area']

    padded = np.zeros((h + 2, w + 2), bool)
    padded[1:-1, 1:-1] = sub
    filled = F.FillHoles(padded)
    holesMask = filled & ~padded
    maxHole = 0.0
    if holesMask.any():
        hl, nh = F.LabelComponents(holesMask, connectivity=4)
        if nh > 0:
            maxHole = float(np.bincount(hl.ravel())[1:].max())
    holeUnits = maxHole / max(1.0, textH * textH)
    holes = max(0.0, min(1.0, (holeUnits - 0.7) / 0.8))

    fill = area / max(1.0, w * h)
    bigAndSparse = (w > textH * 6.0 and h > textH * 1.8 and fill < 0.05)
    sparse = 0.7 if bigAndSparse else 0.0

    return float(max(holes, sparse))


# ---------------------------------------------------------------------------
# Line grouping
# ---------------------------------------------------------------------------
def EstimateLinePitch(proj, textH):
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
    # 1. MESS candidates + neighbour veto (rules out big heading letters)
    for c in comps:
        c['mess'] = ScoreComponentMess(c, labels, textH)
    messCand = [c for c in comps if c['mess'] >= 0.5
                and c['area'] > textH * textH * 0.8 and c['h'] > textH * 1.2]

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
        refH = float(np.percentile([o['h'] for o in neigh], 75))
        return c['h'] <= 1.9 * refH

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

    # 2. chain core comps left-to-right
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
            if gap < -a['w'] * 0.6:
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

    # 3. merge chains sharing the same local y (baseline curve)
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

    # 3b. pitch-aware second merge
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

    # 3c. demote weak chains
    def _ChainStats(ch):
        cs = ch['comps']
        width = max(c['x'] + c['w'] for c in cs) - min(c['x'] for c in cs)
        return len(cs), width, sum(c['area'] for c in cs)

    survivors, demoted = [], []
    for ch in chainList:
        nc, width, area = _ChainStats(ch)
        strong = nc >= 3 or (width >= 3.0 * textH and area >= 2.0 * textH * textH)
        if not strong:
            myY = float(np.mean(ch['curve'][1]))
            nearest = min((abs(float(np.mean(o['curve'][1])) - myY)
                           for o in chainList if o is not ch), default=1e9)
            strong = nearest > pitch * 0.8
        (survivors if strong else demoted).append(ch)
    chainList = survivors

    # 4. leftovers: underlines attach above; small marks join nearest curve;
    # tall interline comps split pixel-wise between curves they span
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
            best, bestDy = None, textH * 1.5
            for ch in chainList:
                cy = _CurveY(ch['curve'], c['cx'])
                dy = c['cy'] - cy
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

    # 5. finalize lines
    textLines = []
    for ch in chainList:
        cs = ch['comps']
        totalArea = sum(c['area'] for c in cs)
        maxH = max(c['h'] for c in cs)
        if totalArea < textH * textH * 0.35 or maxH < textH * 0.35:
            continue
        fills = [c['area'] / max(1.0, c['w'] * c['h']) for c in cs]
        width = max(c['x'] + c['w'] for c in cs) - min(c['x'] for c in cs)
        if len(cs) <= 3 and min(fills) > 0.45 and width < textH * 3.5:
            continue
        textLines.append(dict(comps=cs,
                              yc=float(np.average([c['cy'] for c in cs],
                                                  weights=[c['area'] for c in cs]))))

    # 6. merge mess blocks belonging to one drawing
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

    # 5b. absorb "text lines" living inside a drawing's core bbox
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
# Rendering + ordering
# ---------------------------------------------------------------------------
def RenderLine(gray, labels, comps, pad=6, inkRaw=None, inkRawLabels=None, gapPx=6, deskew=False):
    """Render one line from its own ink onto a white canvas. inkRaw/inkRawLabels
    (hysteresis-recovered ink, see HysteresisRecoverInk -- never affects
    segmentation, only what gets painted here) recover faded stroke pixels
    that strong-only binarization missed. The search window is padded
    generously (3x gapPx) so a badly faded trailing glyph is reachable at
    all, then trimmed back to a tight box around whatever ink actually
    ended up in the mask."""
    searchPad = max(pad, gapPx * 3) if inkRawLabels is not None else pad
    x1 = max(0, min(c['x'] for c in comps) - searchPad)
    y1 = max(0, min(c['y'] for c in comps) - pad)
    x2 = min(gray.shape[1], max(c['x'] + c['w'] for c in comps) + searchPad)
    y2 = min(gray.shape[0], max(c['y'] + c['h'] for c in comps) + pad)

    mask = np.zeros((y2 - y1, x2 - x1), bool)
    for c in comps:
        sy = slice(c['y'] - y1, c['y'] - y1 + c['h'])
        sx = slice(c['x'] - x1, c['x'] - x1 + c['w'])
        if 'pixmask' in c:
            mask[sy, sx] |= c['pixmask']
        else:
            mask[sy, sx] |= (labels[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']] == c['id'])

    mask = F.Dilate(mask, 3, 3)
    if inkRaw is not None:
        near = F.Dilate(mask, 9, 3)
        mask |= (near & inkRaw[y1:y2, x1:x2])
    if inkRawLabels is not None:
        # anisotropic reach: wide horizontally (bridge a whole missed
        # glyph), narrow vertically (never bridge the gap between lines)
        window = inkRawLabels[y1:y2, x1:x2] > 0
        reachPx = gapPx * 3
        near = F.Dilate(mask, 4, 2 * reachPx + 1)
        mask |= (near & window)

        ys_, xs_ = np.nonzero(mask)
        if len(xs_):
            tx1 = max(0, xs_.min() - pad)
            tx2 = min(mask.shape[1], xs_.max() + 1 + pad)
            ty1 = max(0, ys_.min() - pad)
            ty2 = min(mask.shape[0], ys_.max() + 1 + pad)
            mask = mask[ty1:ty2, tx1:tx2]
            x1, x2 = x1 + tx1, x1 + tx2
            y1, y2 = y1 + ty1, y1 + ty2

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
            rot = F.Rotate(padded, angle, fill=255)
            ys, xs = np.nonzero(rot < 250)
            if len(ys):
                a = max(0, ys.min() - pad)
                b = min(rot.shape[0], ys.max() + 1 + pad)
                crop = rot[a:b]

    return crop, (x1, y1, x2, y2)


def ProcessPage(imgPath):
    rgb = LoadImage(imgPath)
    pageMask = DetectPageMask(rgb)
    gray = F.RgbToGray(rgb)
    illum = CorrectIllumination(gray)

    ink0 = BinarizeInk(illum, pageMask) & ~RedInkMask(rgb)
    ink0 = RemoveSpeckles(ink0)

    angle = EstimateSkew(ink0)
    if abs(angle) >= 0.15:
        rgb = Rotate(rgb, angle, fill=255)
        pageMask = Rotate(pageMask.astype(np.uint8), angle, isMask=True).astype(bool)
        gray = F.RgbToGray(rgb)
        illum = CorrectIllumination(gray)
        ink0 = BinarizeInk(illum, pageMask) & ~RedInkMask(rgb)
        ink0 = RemoveSpeckles(ink0)

    # crop-time-only recovery source (see RenderLine); minus thin horizontal
    # runs so rule-line stubs never come back either way
    inkRecovered = HysteresisRecoverInk(illum, pageMask, ink0)
    hR, vR = _HorizontalRunLengths(inkRecovered), _VerticalRunLengths(inkRecovered)
    inkRaw = inkRecovered & ~((hR >= 10) & (vR <= 4))
    inkRawLabels, _ = F.LabelComponents(inkRaw, connectivity=8)

    ink0 = FilterFaintComponents(ink0, illum)
    ink0 = RemoveEdgeComponents(ink0, pageMask, 30)

    _, roughComps = ComponentStats(ink0)
    roughH = EstimateTextHeight(roughComps)

    ink, ruleMask = RemoveRuleLines(ink0, roughH, illum=illum)
    ink, ruleMask2 = RemoveRuleLines(ink, roughH, illum=illum)  # 2nd pass: newly-unshielded stubs
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
    for k in range(len(items) - 1):
        a, b = items[k], items[k + 1]
        if a['tag'] == 'MESS' and b['tag'] == 'TEXT' and a['y1'] <= b['yc'] <= a['y2']:
            items[k], items[k + 1] = b, a

    results = []
    preview = Image.fromarray(rgb.copy())
    drawObj = ImageDraw.Draw(preview)
    for order, it in enumerate(items):
        crop, bbox = RenderLine(gray, labels, it['comps'], inkRaw=inkRaw,
                                inkRawLabels=inkRawLabels, gapPx=max(8, int(textH * 0.55)),
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


# ---------------------------------------------------------------------------
# Self-test harness (was NonDatasetSegmenterTest.py + NonDatasetPreprocessing.py
# -- folded in here so the FP engine is fully self-contained in one file)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, 'NOGIT', 'NonDatasetImages')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'NOGIT', 'NonDatasetTestOutput', 'fp')


def ReadLabels(imgPath):
    labelPath = os.path.splitext(imgPath)[0] + '_labels.txt'
    if not os.path.exists(labelPath):
        return None
    rows = []
    with open(labelPath, encoding='utf-8') as f:
        for ln in f:
            ln = ln.rstrip('\n')
            if not ln.strip():
                continue
            parts = ln.split('\t', 1)
            rows.append(parts[1] if len(parts) == 2 else ln)
    return rows


def Score(detected, labelRows):
    expTags = ['MESS' if r.strip() == 'MESS' else 'TEXT' for r in labelRows]
    detTags = [r['tag'] for r in detected]
    n = max(len(expTags), len(detTags))
    match = sum(1 for i in range(min(len(expTags), len(detTags))) if expTags[i] == detTags[i])
    return (match / n if n else 1.0), expTags, detTags


def _RunOnImage(imgPath):
    name = os.path.splitext(os.path.basename(imgPath))[0]
    results, preview, meta = ProcessPage(imgPath)

    previewDir = os.path.join(OUTPUT_DIR, 'previews')
    cropDir = os.path.join(OUTPUT_DIR, 'crops', name)
    modelDir = os.path.join(OUTPUT_DIR, 'crops_model', name)
    for d in (previewDir, cropDir, modelDir):
        os.makedirs(d, exist_ok=True)
    preview.save(os.path.join(previewDir, name + '_preview.png'))

    for r in results:
        fname = f"line_{r['order']:02d}_{r['tag']}.png"
        img = Image.fromarray(r['raw_crop'])
        img.save(os.path.join(cropDir, fname))
        g = img.convert('L')
        s = min(INPUT_W / g.width, INPUT_H / g.height)
        g = g.resize((max(1, int(g.width * s)), max(1, int(g.height * s))), Image.Resampling.BILINEAR)
        canvas = Image.new('L', (INPUT_W, INPUT_H), 255)
        canvas.paste(g, (0, (INPUT_H - g.height) // 2))
        canvas.save(os.path.join(modelDir, fname))

    report = dict(page=name, detected=len(results),
                  detected_mess=sum(1 for r in results if r['tag'] == 'MESS'),
                  skew_deg=round(meta.get('skew', 0.0), 2),
                  text_height_px=round(meta.get('textH', 0.0), 1))

    labels = ReadLabels(imgPath)
    if labels is None:
        print(f"{name}: no label file -- segmented {len(results)} boxes (no accuracy score)")
        return report

    acc, expTags, detTags = Score(results, labels)
    report.update(expected=len(expTags), expected_mess=expTags.count('MESS'), accuracy=round(acc, 4))
    print(f"{name}: acc={acc*100:.1f}%  expected {len(expTags)} rows ({expTags.count('MESS')} MESS) | "
          f"detected {len(detTags)} ({detTags.count('MESS')} MESS) | skew={report['skew_deg']}deg")
    if acc < 1.0:
        for i in range(max(len(expTags), len(detTags))):
            e = expTags[i] if i < len(expTags) else '--'
            d = detTags[i] if i < len(detTags) else '--'
            if e != d:
                lbl = labels[i][:50] if i < len(labels) else ''
                print(f"   {i:2d}  exp={e:4s} det={d:4s}  {lbl}   <<< MISMATCH")
    return report


def RunAll():
    imagePaths = sorted(p for ext in ('*.png', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG')
                        for p in glob.glob(os.path.join(IMAGES_DIR, ext)))
    if not imagePaths:
        print(f'No images found in {IMAGES_DIR}')
        return []
    reports = [_RunOnImage(p) for p in imagePaths]
    scored = [r['accuracy'] for r in reports if 'accuracy' in r]
    if scored:
        print(f'== OVERALL: {100*sum(scored)/len(scored):.1f}% ==')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(reports, f, indent=2)
    print(f"\nPreviews:    {os.path.join(OUTPUT_DIR, 'previews')}")
    print(f"Crops:       {os.path.join(OUTPUT_DIR, 'crops')}")
    print(f"Model crops: {os.path.join(OUTPUT_DIR, 'crops_model')}")
    return reports


if __name__ == '__main__':
    RunAll()
