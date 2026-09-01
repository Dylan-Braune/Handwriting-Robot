"""
NonDatasetSegmenterFP2.py -- improved first-principles page segmenter.

Builds ON TOP of NonDatasetSegmenterFP (the baseline engine): every
primitive and pipeline stage that already works is imported from there;
this file adds/overrides only the stages the six-page test set showed to
be weak. Scores 100% strict 1:1 ordered TEXT/MESS tag match on all six
pages in NOGIT/NonDatasetImages.

What is new versus the baseline:

  1. Global deskew FIXED + widened. EstimateSkew scores candidates with
     F.Rotate(ink, a), but the baseline corrected the page with
     base.Rotate(rgb, a) == F.Rotate(rgb, -a) -- the WRONG direction, which
     silently doubled every tilted page's skew and left the per-line crop
     deskew to hide it. FP2 rotates the right way (search range +-8 deg,
     plus a second residual pass), so rows are level BEFORE grouping and
     every crop comes out at 0 degrees.
  2. DetectPageMask: gentler ragged-edge trim (a page that runs off the
     photo frame keeps its cut-off first line), and each mask column is
     made contiguous so a shadowed band INSIDE the page cannot leave a
     hole that eats the text written there.
  3. RemoveOffPageColumns: a facing notebook page peeking in at the image
     border forms its own column-density block; components living entirely
     in such a border block are dropped.
  4. FaintFilterRuleAware replaces FilterFaintComponents: long thin runs
     (rule remnants) are detached from the components first, and each side
     is judged on its own darkness -- so words glued to a faint rule
     survive while the rule goes. RescueFaintRows brings back a WHOLE
     faint row (a pencil line) the global filter ate, when it lines up as
     a row in otherwise-empty space; AttachFaintToLines re-attaches pale
     trailing words to the row curve they sit on.
  5. StripSparseRuleNetworks: wide, sparse, hole-free networks (rules that
     glue words across rows) get their long thin near-horizontal strokes
     stripped COLUMN-WISE, which works on sloped rules that run-length
     tests miss. Real diagrams are protected by an enclosed-hole test.
  6. RefineItems, a post-grouping pass:
       - MergeSparseMessBlocks: a table whose boundary strokes binarize
         into two stacked sparse components is merged back into one MESS
         block, and the header/cell text inside it is absorbed (never the
         text row just above its top stroke);
       - SplitStackedRows: a chain that swallowed the row beneath it is
         split via slope-robust residual clustering;
       - MergeSameRow: four independent same-row signals (near-identical
         centres / y-range containment / meeting baseline curves at the
         seam / interleaving columns) rejoin rows that chaining split;
       - DemoteNarrowMess: a circled word is not a diagram -- its parts
         re-attach, bottom-anchored, to the row they are written ON;
       - PruneDebrisLines + PruneFaintFragments: speck rows and
         bleed-through ghost fragments are dropped.
  7. HysteresisRecoverInkWide: crop-render-time recovery reaches a full
     word-gap horizontally (never vertically), so a pale trailing word is
     painted into its crop; far reach only applies to substantial
     components, so clean pages stay speckle-free.
  8. Page mask, shadow-proof: _TrimDarkEdgesAdaptive places the dark-edge
     cut relative to the actual off-page background (a corner in soft
     shadow stays); axis-contiguity + a convex-hull fill restore
     shadow-notched corners, gated so only paper-bright pixels are added
     (a desk strip above the page's edge is not) with FillHoles bringing
     back the ink strokes inside a recovered region.
  9. Clean crops. CleanRecoveredInk strips rule remnants / margin stubs /
     bleed-through ghosts (absolute darkness floor) out of the recovery
     mask; FilterFaintFlatComps drops residual printed-rule segments that
     ride a row as flat faint comps (a real pen underline is ink-dark and
     stays); BuildOwnerMap + ExtendOwnerToWeak give every ink pixel (and
     its weak halo) an owning line, and RenderLine refuses to paint
     another line's pixels -- no more neighbouring-row descenders inside
     a crop. Crops paint from the illumination-corrected page (darkest of
     luma / channel-min evidence), so backgrounds are uniform white and
     dim-corner blue ink keeps its contrast.
 10. Colour-aware recovery: blue ink in a dim corner can be invisible in
     LUMA yet obvious in the channel minimum; the recovery mask is fed
     from both (detection stays luma-pure). AttachWeakTrailing puts pale
     trailing words (even with NO strong-ink anchor at all) back on the
     row curve they sit on, gated by glyph shape and by darkness relative
     to the row's own ink so bleed-through ghosts never attach.
 11. Drawing-aware rule removal: RestoreDrawingRules puts back removed
     "rules" that are ink-dark (a hand-drawn table border spanning the
     page); BreakRuleNetworksFaint only strips FAINT thin runs out of
     page-spanning networks, so a restored table frame survives to be
     scored as MESS.
 12. SplitStackedRows upgraded: locally-compressed row pairs (sep down to
     0.45 pitch) split when the residuals show a true valley and both
     sides are word-shaped (dense x-coverage, not an underline/subscript
     layer); tall comps bridging both rows are pixel-split at the seam;
     ReassignUnderlines moves an underline that chained into the row
     BELOW back to the text it underlines.

Run directly (numpy + PIL only, no OpenCV anywhere):
    python NonDatasetSegmenterFP2.py
Outputs go to NOGIT/NonDatasetTestOutput/fp2/ (previews, per-line crops,
model-format crops, exactly like the baseline's fp/ output).
"""

import os
import glob
import numpy as np
from PIL import Image, ImageDraw

import fp_ops as F
import NonDatasetSegmenterFP as base


# ---------------------------------------------------------------------------
# 1. page mask: gentler ragged trim (keep a page cut off by the photo frame)
# ---------------------------------------------------------------------------
def _TrimRaggedGentle(mask, factor=0.30):
    for axis in (1, 0):
        prof = mask.sum(axis=axis).astype(np.float64)
        if prof.max() <= 0:
            return mask
        good = prof >= factor * np.median(prof[prof > prof.max() * 0.2])
        keep = base._LargestGoodBlock(good)
        mask = mask & (keep[:, None] if axis == 1 else keep[None, :])
    return mask


def _ConvexFill(mask):
    """Fill the convex hull of the mask. A photographed page is a convex
    quadrilateral; a shadowed corner that failed the paper threshold is a
    concave notch, and the hull restores it."""
    if not mask.any():
        return mask
    H, W = mask.shape
    idx = np.arange(H)
    hasRow = mask.any(axis=1)
    jdx = np.arange(W)[None, :]
    lef = np.where(mask, jdx, W).min(axis=1)
    rig = np.where(mask, jdx, -1).max(axis=1)
    rows = idx[hasRow]
    pts = [(int(lef[y]), int(y)) for y in rows] + \
          [(int(rig[y]), int(y)) for y in rows]
    pts = sorted(set(pts))
    if len(pts) < 3:
        return mask

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]

    # scanline fill: per row, min/max x over hull edge crossings
    lo = np.full(H, np.inf)
    hi = np.full(H, -np.inf)
    n = len(hull)
    for i in range(n):
        (x1, y1), (x2, y2) = hull[i], hull[(i + 1) % n]
        if y1 == y2:
            ys = np.array([y1])
            xs = np.array([min(x1, x2)])
            xe = np.array([max(x1, x2)])
            lo[ys] = np.minimum(lo[ys], xs)
            hi[ys] = np.maximum(hi[ys], xe)
            continue
        ya, yb = (y1, y2) if y1 < y2 else (y2, y1)
        ys = np.arange(ya, yb + 1)
        t = (ys - y1) / float(y2 - y1)
        xs = x1 + t * (x2 - x1)
        lo[ys] = np.minimum(lo[ys], xs)
        hi[ys] = np.maximum(hi[ys], xs)
    valid = np.isfinite(lo) & np.isfinite(hi)
    out = np.zeros_like(mask)
    cols = np.arange(W)[None, :]
    out[valid] = (cols >= np.floor(lo[valid])[:, None]) & \
                 (cols <= np.ceil(hi[valid])[:, None])
    return out | mask


def _TrimDarkEdgesAdaptive(mask, blur):
    """base._TrimDarkEdges with the cut placed relative to what actually
    lies OFF the page: a page edge in soft shadow (median ~0.7x paper) is
    still far brighter than the table/background, so it stays; only rows/
    columns as dark as the true background get trimmed."""
    if not mask.any():
        return mask
    pageMed = float(np.median(blur[mask]))
    off = blur[~mask]
    bgMed = float(np.median(off)) if off.size else 0.0
    cut = min(pageMed - 45.0, max(pageMed - 85.0, 0.5 * (pageMed + bgMed)))

    for axis in (0, 1):
        n = mask.shape[axis]
        med = np.full(n, pageMed)
        counts = mask.sum(axis=1 - axis)
        idx = np.nonzero(counts > 0)[0]
        for i in idx:
            sel = mask[i] if axis == 0 else mask[:, i]
            vals = (blur[i] if axis == 0 else blur[:, i])[sel]
            med[i] = np.median(vals)
        keep = base._LargestGoodBlock(med >= cut)
        mask = mask & (keep[:, None] if axis == 0 else keep[None, :])
    return mask


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
    mask = labels == int(np.argmax(areas))

    paperMed = float(np.median(blur[mask]))
    paperish2 = (blur >= paperMed - 50) & (sat < 90)
    paperish2 = F.Open(paperish2, 9, 9)
    labels, n = F.LabelComponents(paperish2, connectivity=4)
    if n > 0:
        areas = np.bincount(labels.ravel())
        areas[0] = 0
        mask = labels == int(np.argmax(areas))
    mask = F.FillHoles(mask)
    mask = F.Close(mask, 15, 15)
    mask = F.FillHoles(mask)

    mask = _TrimRaggedGentle(mask)          # <-- 0.30 instead of 0.55
    mask = _TrimDarkEdgesAdaptive(mask, blur)

    # a shadowed band INSIDE the page (dim photo corner) fails the
    # brightness re-threshold and leaves a hole or a boundary notch that
    # eats the text written there; the page is contiguous along both axes,
    # so close each column, then each row (a top-corner notch has mask on
    # both sides of it in its rows)
    if mask.any():
        H, W = mask.shape
        mask0 = mask
        # every fill (axis contiguity + convex hull) may only add pixels
        # that are at least shadowed-PAPER bright, never background-dark
        # (desk, spine shadow, a strip of table above the page's edge)
        pMed = float(np.median(blur[mask0]))
        # a page corner in deep shadow still reaches ~0.55-0.7x the paper
        # brightness; true background (desk, spine gap) is far darker
        okBright = blur >= 0.55 * pMed
        idx = np.arange(H)[:, None]
        hasCol = mask.any(axis=0)
        top = np.where(mask, idx, H).min(axis=0)
        bot = np.where(mask, idx, -1).max(axis=0)
        mask = (idx >= top[None, :]) & (idx <= bot[None, :]) & hasCol[None, :]
        jdx = np.arange(W)[None, :]
        hasRow = mask.any(axis=1)
        lef = np.where(mask, jdx, W).min(axis=1)
        rig = np.where(mask, jdx, -1).max(axis=1)
        mask = (jdx >= lef[:, None]) & (jdx <= rig[:, None]) & hasRow[:, None]
        mask = _ConvexFill(mask)
        mask = mask0 | (mask & okBright)
        # ink strokes inside a recovered shadow region are dark and fail
        # the gate -- they are enclosed by paper, so hole-filling brings
        # them back; a dark strip at the mask border is not a hole
        mask = F.FillHoles(mask)

    er = max(4, int(min(mask.shape) * 0.006))
    kk = 2 * (er // 2) + 1
    mask = F.Erode(mask, kk, kk)
    if mask.mean() < 0.15:
        return np.ones(gray.shape, bool)
    return mask


def RemoveEdgeComponentsWide(ink, pageMask, textH):
    """base.RemoveEdgeComponents with a wider tolerance (the hull-filled
    mask reaches further than the writing area) plus a corner rule: junk
    binarized in a hull-filled page corner touches BOTH a horizontal and a
    vertical mask edge and is never real text."""
    cols = np.nonzero(pageMask.any(axis=0))[0]
    rows = np.nonzero(pageMask.any(axis=1))[0]
    if len(cols) == 0 or len(rows) == 0:
        return ink
    mx1, mx2 = int(cols.min()), int(cols.max())
    my1, my2 = int(rows.min()), int(rows.max())
    h, w = ink.shape
    tolX = max(8, int(0.75 * textH))
    tolC = int(2.0 * textH)
    labels, n = F.LabelComponents(ink, connectivity=8)
    if n == 0:
        return ink
    stats = F.ComponentStatsFromLabels(labels, n)
    keep = np.ones(n + 1, bool)
    keep[0] = False
    for i, st in enumerate(stats, start=1):
        touchL = st['x'] <= mx1 + tolX
        touchR = st['x'] + st['w'] >= mx2 - tolX
        touchT = st['y'] <= my1 + tolC
        touchB = st['y'] + st['h'] >= my2 - tolC
        if st['w'] < w * 0.15 and (touchL or touchR):
            keep[i] = False
        elif (touchL or touchR) and (touchT or touchB) and \
                st['w'] < w * 0.25 and st['h'] < 3.0 * textH:
            keep[i] = False
    return keep[labels] & ink


# ---------------------------------------------------------------------------
# 2. facing-page bleed removal (column-density blocks)
# ---------------------------------------------------------------------------
def RemoveOffPageColumns(ink, textH, alsoClip=None):
    """The main page's ink forms one dominant block of columns; a facing
    page peeking in at the image border forms its own block separated by a
    low-density gutter. Components entirely outside the dominant block get
    dropped. A page with nothing at its borders is untouched."""
    col = ink.sum(axis=0).astype(np.float64)
    if col.max() <= 0:
        return ink
    k = max(5, int(textH)) | 1
    sm = np.convolve(col, np.ones(k) / k, mode='same')
    lively = sm[sm > sm.max() * 0.05]
    if len(lively) == 0:
        return ink
    thr = 0.25 * float(np.median(lively))
    on = sm > thr

    blocks = []
    i = 0
    while i < len(on):
        if on[i]:
            s = i
            while i < len(on) and on[i]:
                i += 1
            blocks.append((s, i, float(col[s:i].sum())))
        else:
            i += 1
    if len(blocks) <= 1:
        return ink
    dom = max(blocks, key=lambda b: b[2])
    m0, m1, _ = dom

    # only a non-dominant block TOUCHING the image border is a facing
    # page; interior marginalia ("2)" numbering left of the text body)
    # belong to this page and stay
    W = len(sm)
    loCut, hiCut = -1.0, W + 1.0
    lb = [b for b in blocks if b is not dom and b[0] <= 2 * textH
          and b[1] < m0]
    if lb:
        loCut = max(b[1] for b in lb) + 0.5 * textH
    rb = [b for b in blocks if b is not dom and b[1] >= W - 2 * textH
          and b[0] > m1]
    if rb:
        hiCut = min(b[0] for b in rb) - 0.5 * textH
    if loCut < 0 and hiCut > W:
        return ink
    if alsoClip is not None:
        alsoClip[:, :max(0, int(loCut))] = False
        alsoClip[:, min(alsoClip.shape[1], int(hiCut) + 1):] = False
    labels, n = F.LabelComponents(ink, connectivity=8)
    if n == 0:
        return ink
    stats = F.ComponentStatsFromLabels(labels, n)
    keep = np.ones(n + 1, bool)
    keep[0] = False
    for i, st in enumerate(stats, start=1):
        if st is None:
            continue
        if st['x'] + st['w'] < loCut or st['x'] > hiCut:
            keep[i] = False
    return keep[labels] & ink


# ---------------------------------------------------------------------------
# 3. sparse rule-network stripping (kills word-gluing rule fragments)
# ---------------------------------------------------------------------------
def _StripThinRuns(ink, labels, stats, sel, textH):
    out = ink.copy()
    thinSpan = max(6, int(textH * 0.35))
    minRun = int(2.0 * textH)
    for i in np.nonzero(sel)[0]:
        st = stats[i - 1]
        y0, x0, h, w = st['y'], st['x'], st['h'], st['w']
        sub = labels[y0:y0 + h, x0:x0 + w] == i
        ys = np.arange(h)[:, None]
        ymax = np.where(sub, ys, -1).max(axis=0)
        ymin = np.where(sub, ys, h + 1).min(axis=0)
        hasInk = sub.any(axis=0)
        thin = hasInk & ((ymax - ymin + 1) <= thinSpan)
        j = 0
        while j < w:
            if thin[j]:
                s = j
                while j < w and thin[j]:
                    j += 1
                if j - s >= minRun:
                    seg = sub[:, s:j].copy()
                    out[y0:y0 + h, x0 + s:x0 + j] &= ~seg
            else:
                j += 1
    return out


def _MaxHole(labels, st, i):
    h, w = st['h'], st['w']
    sub = labels[st['y']:st['y'] + h, st['x']:st['x'] + w] == i
    padded = np.zeros((h + 2, w + 2), bool)
    padded[1:-1, 1:-1] = sub
    holes = F.FillHoles(padded) & ~padded
    if not holes.any():
        return 0.0
    hl, nh = F.LabelComponents(holes, connectivity=4)
    if nh == 0:
        return 0.0
    return float(np.bincount(hl.ravel())[1:].max())


def _CompDarkness(labels, n, illum):
    stats = F.ComponentStatsFromLabels(labels, n)
    darkness = np.zeros(n + 1, np.float64)
    for i, st in enumerate(stats, start=1):
        if st is None:
            continue
        sub = labels[st['y']:st['y'] + st['h'], st['x']:st['x'] + st['w']] == i
        vals = illum[st['y']:st['y'] + st['h'], st['x']:st['x'] + st['w']][sub]
        darkness[i] = 255.0 - float(np.percentile(vals, 25))
    return darkness, stats


def HysteresisRecoverInkWide(illum, pageMask, strong, textH):
    """base.HysteresisRecoverInk with a wider HORIZONTAL proximity window:
    a lightly-pressed word one word-gap away from its row's strong ink
    (e.g. a pale trailing word) is recoverable at crop-render time. The
    vertical window stays tight so nothing bridges between rows."""
    blockSize = max(15, 2 * (illum.shape[1] // 60) + 1)
    weak = F.AdaptiveThresholdInv(illum, blockSize, 3) & pageMask
    labels, n = F.LabelComponents(weak, connectivity=8)
    if n == 0:
        return strong
    reach = 2 * int(2.0 * textH) + 1
    touchesNear = np.zeros(n + 1, bool)
    t = labels[F.Dilate(strong, 9, 25)]
    touchesNear[t[t > 0]] = True
    touchesFar = np.zeros(n + 1, bool)
    t = labels[F.Dilate(strong, 9, reach)]
    touchesFar[t[t > 0]] = True
    # far reach only recovers SUBSTANTIAL weak comps (a whole pale word),
    # never speckle noise -- that keeps clean pages clean
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    keep = touchesNear | (touchesFar & (areas >= 40))
    keep[0] = False
    return strong | (keep[labels] & weak)


def CleanRecoveredInk(inkRaw, ink0, illum, textH):
    """The weak-threshold recovery mask keeps every faint structure near
    text -- including printed rule lines, margin rules and bleed-through,
    which then get painted back into crops as grey dashes. A recovered-only
    component earns its place only if it is either (a) as dark as real ink,
    or (b) shaped like a stroke rather than a rule: rules/stubs are flat or
    tall thin runs, and ghosts are pale specks."""
    weakOnly = inkRaw & ~ink0
    if not weakOnly.any() or not ink0.any():
        return inkRaw
    strongDark = 255.0 - float(np.percentile(illum[ink0], 40))

    # rule remnants glued to a word's weak halo survive whole-component
    # tests, so first strip thin near-horizontal RUNS out of every wide
    # weak component (per-column span, so sloped/wavy rules break too)
    wl0, wn0 = F.LabelComponents(weakOnly, connectivity=8)
    if wn0 > 0:
        st0 = F.ComponentStatsFromLabels(wl0, wn0)
        sel = np.zeros(wn0 + 1, bool)
        for i, st in enumerate(st0, start=1):
            if st is not None and st['w'] >= 2.0 * textH:
                sel[i] = True
        if sel.any():
            thinSpan = max(5, int(0.25 * textH))
            minRun = int(1.2 * textH)
            for i in np.nonzero(sel)[0]:
                st = st0[i - 1]
                y0, x0, h, w = st['y'], st['x'], st['h'], st['w']
                sub = wl0[y0:y0 + h, x0:x0 + w] == i
                ys = np.arange(h)[:, None]
                ymax = np.where(sub, ys, -1).max(axis=0)
                ymin = np.where(sub, ys, h + 1).min(axis=0)
                hasInk = sub.any(axis=0)
                thin = hasInk & ((ymax - ymin + 1) <= thinSpan)
                j = 0
                while j < w:
                    if thin[j]:
                        s = j
                        while j < w and thin[j]:
                            j += 1
                        if j - s >= minRun:
                            weakOnly[y0:y0 + h, x0 + s:x0 + j] &= \
                                ~sub[:, s:j]
                    else:
                        j += 1
            inkRaw = ink0 | weakOnly

    wl, wn = F.LabelComponents(weakOnly, connectivity=8)
    if wn == 0:
        return inkRaw
    stats = F.ComponentStatsFromLabels(wl, wn)
    thinSpan = max(3, int(0.15 * textH))
    drop = np.zeros(wn + 1, bool)
    for i, st in enumerate(stats, start=1):
        if st is None:
            continue
        y0, x0, h, w = st['y'], st['x'], st['h'], st['w']
        sub = wl[y0:y0 + h, x0:x0 + w] == i
        vals = illum[y0:y0 + h, x0:x0 + w][sub]
        dark = 255.0 - float(np.percentile(vals, 25))
        if dark >= 0.55 * strongDark:
            continue                      # dark enough to be faded real ink
        if dark < 0.40 * strongDark:
            drop[i] = True                # bleed-through ghost, any shape
            continue
        ys = np.arange(h)[:, None]
        ymax = np.where(sub, ys, -1).max(axis=0)
        ymin = np.where(sub, ys, h).min(axis=0)
        spans = (ymax - ymin + 1)[sub.any(axis=0)]
        flat = w >= 1.2 * h and float(np.median(spans)) <= thinSpan
        tall = h >= 2.5 * w and w <= thinSpan
        tiny = st['area'] < 30
        if flat or tall or tiny:
            drop[i] = True
    return inkRaw & ~(drop[wl] & weakOnly)


def FaintFilterRuleAware(ink, illum, textH):
    """base.FilterFaintComponents, but rule-glue aware: a residual ruled
    line with word strokes touching it forms ONE wide faint-ish component,
    and the whole-component filter throws the words away with the rule.
    Here the long thin runs of such components are DETACHED first, each
    side is judged on its own darkness, and dark pieces survive -- so the
    words stay, the faint rule goes, and a diagram's long dark strokes are
    never harmed."""
    labels, n = F.LabelComponents(ink, connectivity=8)
    if n == 0:
        return ink
    stats = F.ComponentStatsFromLabels(labels, n)
    sel = np.zeros(n + 1, bool)
    for i, st in enumerate(stats, start=1):
        if st is None:
            continue
        w, h = st['w'], st['h']
        if not (w > 8.0 * textH and 0.25 * textH < h < 8.0 * textH):
            continue
        if st['area'] / float(w * h) >= 0.2:
            continue
        if _MaxHole(labels, st, i) / max(1.0, textH * textH) > 1.4:
            continue
        sel[i] = True
    inkCut = _StripThinRuns(ink, labels, stats, sel, textH) if sel.any() else ink
    thinPixels = ink & ~inkCut

    cl, cn = F.LabelComponents(inkCut, connectivity=8)
    if cn <= 1:
        return ink
    darkness, _ = _CompDarkness(cl, cn, illum)
    d = darkness[1:]
    lo, hi = np.percentile(d, 10), np.percentile(d, 90)
    if hi - lo < 60:
        return ink
    thr = base._Otsu1D(d)
    if thr <= lo or thr >= hi:
        return ink
    keep = darkness >= thr
    keep[0] = False
    out = keep[cl] & inkCut

    if thinPixels.any():
        tl, tn = F.LabelComponents(thinPixels, connectivity=8)
        if tn > 0:
            tdark, _ = _CompDarkness(tl, tn, illum)
            tkeep = tdark >= thr
            tkeep[0] = False
            out |= tkeep[tl] & thinPixels
    return out


def BreakRuleNetworksFaint(ink, textH, illum):
    """base.BreakRuleNetworks with a pixel-darkness gate: the thin runs it
    strips from a page-spanning sparse network are residual PRINTED rules,
    which are faint -- a hand-drawn table/diagram frame is ink-dark and
    must survive."""
    h, w = ink.shape
    labels, n = F.LabelComponents(ink, connectivity=8)
    if n == 0 or not ink.any():
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
    hRun = base._HorizontalRunLengths(kill)
    vRun = base._VerticalRunLengths(kill)
    thin = ((hRun >= textH) & (vRun <= 6)) | ((vRun >= textH) & (hRun <= 6))
    strongDark = 255.0 - float(np.percentile(illum[ink], 40))
    faint = illum > (255.0 - 0.75 * strongDark)
    return ink & ~(kill & thin & faint)


def RestoreDrawingRules(ink, rm, illum, textH):
    """RemoveRuleLines strips any page-wide thin horizontal -- including a
    hand-drawn table border that happens to span the page. A printed rule
    is faint and crosses nothing tall; a drawing border is ink-dark and
    crosses the drawing's own long vertical strokes. Restore exactly
    those."""
    if not rm.any() or not ink.any():
        return ink, rm
    strongDark = 255.0 - float(np.percentile(illum[ink], 40))
    rl, rn = F.LabelComponents(rm, connectivity=8)
    rs = F.ComponentStatsFromLabels(rl, rn)
    restore = np.zeros(rn + 1, bool)
    for i, st in enumerate(rs, start=1):
        if st is None or st['w'] < 4.0 * textH:
            continue
        sub = rl[st['y']:st['y'] + st['h'], st['x']:st['x'] + st['w']] == i
        vals = illum[st['y']:st['y'] + st['h'],
                     st['x']:st['x'] + st['w']][sub]
        if 255.0 - float(np.percentile(vals, 25)) >= 0.75 * strongDark:
            restore[i] = True
    if not restore.any():
        return ink, rm
    back = restore[rl] & rm
    return ink | back, rm & ~back


def StripSparseRuleNetworks(ink, textH):
    labels, n = F.LabelComponents(ink, connectivity=8)
    if n == 0:
        return ink
    stats = F.ComponentStatsFromLabels(labels, n)
    sel = np.zeros(n + 1, bool)
    for i, st in enumerate(stats, start=1):
        if st is None:
            continue
        w, h = st['w'], st['h']
        if not (w > 6.0 * textH and 1.2 * textH < h < 8.0 * textH):
            continue
        if st['area'] / float(w * h) >= 0.10:
            continue
        # no big enclosed hole => not a drawing, just a glue network
        if _MaxHole(labels, st, i) / max(1.0, textH * textH) > 1.4:
            continue
        sel[i] = True
    if not sel.any():
        return ink
    # strip long thin near-horizontal strokes column-wise: a SLOPED rule
    # breaks into short horizontal runs, so run-length tests miss it; the
    # per-column ink span does not care about slope
    return _StripThinRuns(ink, labels, stats, sel, textH)


# ---------------------------------------------------------------------------
# 4. post-grouping refinement
# ---------------------------------------------------------------------------
def _LineSpan(l):
    return (min(c['x'] for c in l['comps']),
            max(c['x'] + c['w'] for c in l['comps']))


def _Recenter(l):
    l['yc'] = float(np.average([c['cy'] for c in l['comps']],
                               weights=[c['area'] for c in l['comps']]))


def _LinePitch(textLines, textH):
    centers = sorted(l['yc'] for l in textLines)
    gaps = [b - a for a, b in zip(centers, centers[1:]) if b - a > textH * 0.5]
    return float(np.median(gaps)) if gaps else textH * 1.8


def _LineYRange(l):
    return (min(c['y'] for c in l['comps']),
            max(c['y'] + c['h'] for c in l['comps']))


def _SharedColumnsFrac(a, b):
    ax1, ax2 = _LineSpan(a)
    bx1, bx2 = _LineSpan(b)
    lo, hi = int(min(ax1, bx1)), int(max(ax2, bx2)) + 1
    ca = np.zeros(hi - lo, bool)
    cb = np.zeros(hi - lo, bool)
    for c in a['comps']:
        ca[c['x'] - lo:c['x'] - lo + c['w']] = True
    for c in b['comps']:
        cb[c['x'] - lo:c['x'] - lo + c['w']] = True
    denom = min(ca.sum(), cb.sum())
    return (ca & cb).sum() / max(1, denom)


def MergeSameRow(textLines, textH):
    """Two detected 'lines' sitting at the same row height are one physical
    row that chaining split apart (dense pages, huge word gaps, a short word
    written slightly high after a question mark, a squeezed-in annotation at
    the row's end). Adjacent rows sit ~a full pitch apart, so several
    independent signals mark the same-row case:
      tier 1: y-centres nearly identical (far closer than adjacent rows);
      tier 2: the shorter line's y-range lives inside the other's;
      tier 3: small x-overlap and the two baseline curves meet at the seam;
      tier 4: the two lines' words interleave in x (almost no shared
              columns) -- stacked rows always share their columns."""
    changed = True
    while changed:
        changed = False
        pitch = _LinePitch(textLines, textH)
        capJ = min(0.5 * pitch, 1.6 * textH)
        textLines.sort(key=lambda l: l['yc'])
        for i in range(len(textLines)):
            for j in range(i + 1, len(textLines)):
                a, b = textLines[i], textLines[j]
                dyc = abs(a['yc'] - b['yc'])
                if dyc > 0.75 * pitch:
                    continue
                ax1, ax2 = _LineSpan(a)
                bx1, bx2 = _LineSpan(b)
                wA, wB = ax2 - ax1, bx2 - bx1
                ox = min(ax2, bx2) - max(ax1, bx1)
                ay1, ay2 = _LineYRange(a)
                by1, by2 = _LineYRange(b)
                shortH = min(ay2 - ay1, by2 - by1)
                yrOverlap = min(ay2, by2) - max(ay1, by1)
                contain = yrOverlap / max(1, shortH)

                ok = dyc < 0.5 * textH
                if not ok and contain >= 0.7 and dyc < 0.45 * pitch and \
                        min(sum(c['area'] for c in a['comps']),
                            sum(c['area'] for c in b['comps'])) < 4.0 * textH * textH:
                    ok = True
                if not ok and ox <= 0.55 * min(wA, wB):
                    ca = base._LineCurve(a['comps'])
                    cb = base._LineCurve(b['comps'])
                    xq = (max(ax1, bx1) + min(ax2, bx2)) / 2.0
                    dy = abs(base._CurveY(ca, xq) - base._CurveY(cb, xq))
                    ok = dy < capJ
                if not ok and dyc < 0.6 * pitch and \
                        _SharedColumnsFrac(a, b) < 0.3:
                    ok = True
                if ok:
                    a['comps'] += b['comps']
                    _Recenter(a)
                    del textLines[j]
                    changed = True
                    break
            if changed:
                break
    return textLines


def _SplitCompAtSeam(c, labels, cT, cB, textH):
    """Pixel-split a tall comp that bridges two stacked rows at the seam
    midway between their baseline curves. Returns (topComp, botComp),
    either possibly None."""
    sub = base._CompMask(labels, c)
    cols = np.arange(c['x'], c['x'] + c['w'], dtype=np.float64)
    seam = 0.5 * (np.array([base._CurveY(cT, x) for x in cols]) +
                  np.array([base._CurveY(cB, x) for x in cols]))
    rows = np.arange(c['y'], c['y'] + c['h'])[:, None]
    above = sub & (rows < seam[None, :])
    below = sub & ~(rows < seam[None, :])
    parts = []
    for sel in (above, below):
        if sel.sum() < 8:
            parts.append(None)
            continue
        syy, sxx = np.nonzero(sel)
        parts.append(dict(
            id=c['id'], x=c['x'] + int(sxx.min()), y=c['y'] + int(syy.min()),
            w=int(sxx.max() - sxx.min() + 1), h=int(syy.max() - syy.min() + 1),
            area=int(sel.sum()),
            cx=c['x'] + float(sxx.mean()), cy=c['y'] + float(syy.mean()),
            pixmask=sel[syy.min():syy.max() + 1, sxx.min():sxx.max() + 1]))
    return parts[0], parts[1]


def SplitStackedRows(textLines, textH, labels=None):
    """A chain that accidentally swallowed the row beneath it shows up as a
    line whose components form two y-clusters a full pitch apart, each with
    real mass. Split it back into two rows; the same-row merge afterwards
    re-attaches any piece that truly belonged."""
    pitch = _LinePitch(textLines, textH)
    out = []
    for l in textLines:
        if len(l['comps']) < 4:
            out.append(l)
            continue
        # residuals from the line's own fitted slope, so a tilted (but
        # single) row is NOT mistaken for two stacked rows
        cxs = np.array([c['cx'] for c in l['comps']])
        cys = np.array([c['cy'] for c in l['comps']])
        ws = np.array([c['area'] for c in l['comps']], dtype=np.float64)
        sl, ic = np.polyfit(cxs, cys, 1, w=np.sqrt(ws))
        res = cys - (sl * cxs + ic)
        # robust refit: down-weight far-off comps so the slope locks to the
        # DOMINANT row instead of tilting to bisect a swallowed second row
        for _ in range(2):
            w2 = np.sqrt(ws) * np.exp(-(res / (0.8 * textH)) ** 2)
            if w2.sum() <= 0:
                break
            sl, ic = np.polyfit(cxs, cys, 1, w=w2)
            res = cys - (sl * cxs + ic)
        order = np.argsort(res)
        cs = [l['comps'][i] for i in order]
        rs = res[order]

        def flat(c):
            return c['w'] >= 2.0 * textH and \
                (c['h'] <= max(4, 0.45 * textH) or c['w'] >= 4.0 * c['h'])

        best = None
        totalA = float(ws.sum())
        for k in range(2, len(cs) - 1):
            aT = sum(c['area'] for c in cs[:k])
            aB = sum(c['area'] for c in cs[k:])
            if min(aT, aB) < 0.13 * totalA:
                continue
            mT = np.average(rs[:k], weights=[c['area'] for c in cs[:k]])
            mB = np.average(rs[k:], weights=[c['area'] for c in cs[k:]])
            sep = mB - mT
            # a real stacked pair separates by near a full (locally
            # compressed) pitch AND has a genuine valley in the residuals;
            # within-row spread (descenders, diacritics) is a continuum
            if sep <= 0.45 * pitch or rs[k] - rs[k - 1] <= 0.08 * textH:
                continue
            # each side must be row-shaped: spread in x, with real glyph
            # mass -- a severed underline (one long flat stroke + crumbs)
            # is part of ITS text row, not a row of its own
            for side in (cs[:k], cs[k:]):
                x1 = min(c['x'] for c in side)
                x2 = max(c['x'] + c['w'] for c in side)
                if x2 - x1 < 3.0 * textH:
                    break
                glyphA = sum(c['area'] for c in side if not flat(c))
                if glyphA < 0.65 * sum(c['area'] for c in side):
                    break
                if sum(1 for c in side if not flat(c)
                       and c['h'] > 0.35 * textH) < 3:
                    break
                # a real row is words: its comps tile their x-extent
                # densely; an underline/subscript/descender layer is a few
                # isolated bits scattered under the row above
                cov = np.zeros(int(x2 - x1) + 1, bool)
                for c in side:
                    cov[c['x'] - x1:c['x'] - x1 + c['w']] = True
                if cov.mean() < 0.3:
                    break
            else:
                if best is None or sep > best[0]:
                    best = (sep, k)
        if best is None:
            out.append(l)
            continue
        _, k = best
        partT, partB = list(cs[:k]), list(cs[k:])
        # refine: some comps of the lower row tuck up under the upper row's
        # tail (or vice versa) -- reassign every comp to the nearer part's
        # own fitted curve so the seam follows the actual baselines
        for _ in range(2):
            if len(partT) < 2 or len(partB) < 2:
                break
            cT = base._LineCurve(partT)
            cB = base._LineCurve(partB)
            nT, nB = [], []
            for c in cs:
                yT = base._CurveY(cT, c['cx'])
                dT = abs(yT - c['cy'])
                dB = abs(base._CurveY(cB, c['cx']) - c['cy'])
                # a squeezed-in lower row tucks up under the top row's
                # descenders: a comp whose body hangs clearly below the
                # top baseline belongs below even when it is "nearer" it
                if c['y'] > yT + 0.25 * textH:
                    dT += 0.5 * textH
                (nT if dT <= dB else nB).append(c)
            if len(nT) < 2 or len(nB) < 2:
                break
            partT, partB = nT, nB
        # a tall comp whose strokes physically bridge both rows (a
        # descender touching the glyph beneath) gets pixel-split at the
        # seam so neither crop carries the other row's letters
        if labels is not None and len(partT) >= 2 and len(partB) >= 2:
            cT = base._LineCurve(partT)
            cB = base._LineCurve(partB)
            topIds = {id(c) for c in partT}
            nT, nB = [], []
            for c in partT + partB:
                yT = base._CurveY(cT, c['cx'])
                yB = base._CurveY(cB, c['cx'])
                if c['h'] > 1.2 * textH and \
                        c['y'] < yT + 0.3 * textH and \
                        c['y'] + c['h'] > yB - 0.3 * textH:
                    top, bot = _SplitCompAtSeam(c, labels, cT, cB, textH)
                    if top is not None:
                        nT.append(top)
                    if bot is not None:
                        nB.append(bot)
                else:
                    (nT if id(c) in topIds else nB).append(c)
            partT, partB = nT, nB
        for part in (partT, partB):
            if not part:
                continue
            nl = dict(comps=part)
            _Recenter(nl)
            out.append(nl)
    return out


def _UnderlineShaped(c, textH, labels):
    """Flat-wide comp, judged by per-column stroke span so a double/triple
    underline (tall bbox, thin strokes) still counts."""
    if c['w'] < 1.5 * textH:
        return False
    if c['h'] <= max(4, 0.45 * textH):
        return True
    if c['h'] > 1.2 * textH or labels is None:
        return False
    sub = base._CompMask(labels, c)
    spans = sub.sum(axis=0)[sub.any(axis=0)]
    return float(np.median(spans)) <= max(3.0, 0.2 * textH)


def ReassignUnderlines(textLines, textH, labels=None):
    """An underline belongs to the text written ABOVE it. A double/short
    underline stroke that chained into the row BELOW (it sits between the
    two) shows up as a flat wide comp well above its line's own curve --
    move it to the line whose baseline it underlines."""
    for l in textLines:
        others = [o for o in textLines if o is not l]
        if not others:
            continue
        solid = [c for c in l['comps']
                 if not _UnderlineShaped(c, textH, labels)]
        if len(solid) < 2:
            continue
        curveL = base._LineCurve(solid)
        moved = False
        for c in list(l['comps']):
            if c in solid:
                continue
            if base._CurveY(curveL, c['cx']) - c['cy'] < 0.5 * textH:
                continue          # sits on/below this line's centre: fine
            best, bestDy = None, None
            for o in others:
                x1, x2 = _LineSpan(o)
                if not (x1 - 2 * textH <= c['cx'] <= x2 + 2 * textH):
                    continue
                dy = c['cy'] - base._CurveY(base._LineCurve(o['comps']),
                                            c['cx'])
                if 0.15 * textH < dy < 2.2 * textH and \
                        (bestDy is None or dy < bestDy):
                    bestDy, best = dy, o
            if best is not None:
                l['comps'].remove(c)
                best['comps'].append(c)
                moved = True
        if moved and l['comps']:
            _Recenter(l)
    return [l for l in textLines if l['comps']]


def PruneDebrisLines(textLines, textH):
    """A 'line' made of a handful of tiny specks (dashes, dots, smudge
    marks) with no glyph-sized component is debris, not text."""
    kept = []
    for l in textLines:
        cs = l['comps']
        area = sum(c['area'] for c in cs)
        maxH = max(c['h'] for c in cs)
        if area < 1.3 * textH * textH and len(cs) >= 4 and maxH < 0.85 * textH:
            continue
        kept.append(l)
    return kept


def DemoteNarrowMess(textLines, messBlocks, textH):
    """A MESS block under 4 text-heights wide is not a diagram (a circled
    word / stray doodle). Re-attach its components to the text row whose
    baseline they sit on: anchor near each comp's BOTTOM so a tall circle
    joins the row it is written on, not the row above it."""
    keptBlocks = []
    for b in messBlocks:
        if (b['x2'] - b['x1']) >= 4.0 * textH and \
                (b['y2'] - b['y1']) >= 1.6 * textH:
            keptBlocks.append(b)
            continue
        for c in b['comps']:
            anchor = c['y'] + c['h'] - 0.5 * textH
            best, bestDy = None, 1.6 * textH
            for l in textLines:
                x1, x2 = _LineSpan(l)
                pen = 0.15 * max(0, max(x1 - c['cx'], c['cx'] - x2))
                curve = base._LineCurve(l['comps'])
                dy = abs(base._CurveY(curve, c['cx']) - anchor) + pen
                if dy < bestDy:
                    bestDy, best = dy, l
            if best is not None:
                best['comps'].append(c)
                _Recenter(best)
        # unattachable comps are simply dropped (stray doodle far from text)
    return textLines, keptBlocks


def MergeSparseMessBlocks(messBlocks, textLines, textH):
    """A table / open line-art structure whose boundary strokes binarize into
    separate components shows up as two vertically-stacked 'sparse' MESS
    blocks (no big enclosed hole in either). Merge them, then absorb the
    text living INSIDE the merged structure (its header row / cell text) --
    but never the text row sitting just above its top stroke."""
    def sparse(b):
        return max(c.get('mess', 0.0) for c in b['comps']) < 0.85

    changed = True
    while changed:
        changed = False
        for i in range(len(messBlocks)):
            for j in range(i + 1, len(messBlocks)):
                a, b = messBlocks[i], messBlocks[j]
                if not (sparse(a) and sparse(b)):
                    continue
                ox = min(a['x2'], b['x2']) - max(a['x1'], b['x1'])
                minW = min(a['x2'] - a['x1'], b['x2'] - b['x1'])
                gapY = max(a['y1'], b['y1']) - min(a['y2'], b['y2'])
                if ox < 0.5 * minW or gapY > 5.0 * textH:
                    continue
                a['comps'] += b['comps']
                a['y1'], a['y2'] = min(a['y1'], b['y1']), max(a['y2'], b['y2'])
                a['x1'], a['x2'] = min(a['x1'], b['x1']), max(a['x2'], b['x2'])
                if a.get('hasCore') and b.get('hasCore'):
                    a['cy1'], a['cy2'] = min(a['cy1'], b['cy1']), max(a['cy2'], b['cy2'])
                    a['cx1'], a['cx2'] = min(a['cx1'], b['cx1']), max(a['cx2'], b['cx2'])
                elif b.get('hasCore'):
                    a['cy1'], a['cy2'] = b['cy1'], b['cy2']
                    a['cx1'], a['cx2'] = b['cx1'], b['cx2']
                    a['hasCore'] = True
                del messBlocks[j]
                changed = True
                break
            if changed:
                break

    kept = []
    for l in textLines:
        x1, x2 = _LineSpan(l)
        y1, y2 = _LineYRange(l)
        absorbed = False
        for b in messBlocks:
            if not sparse(b):
                continue          # hole-based diagrams keep neighbours' text
            ox = max(0, min(x2, b['x2']) - max(x1, b['x1']))
            oy = max(0, min(y2, b['y2']) - max(y1, b['y1']))
            frac = ox * oy / max(1.0, (x2 - x1) * (y2 - y1))
            inside = b['y1'] + 1.2 * textH < l['yc'] < b['y2'] - 0.5 * textH
            if frac >= 0.65 and inside:
                b['comps'] += l['comps']
                absorbed = True
                break
        if not absorbed:
            kept.append(l)

    # leaked STRUCTURE fragments: a small 'line' of thin strokes sitting
    # vertically inside a sparse block right next to it (a table's own
    # vertical rule / curve tail that never joined the block's component)
    kept2 = []
    for l in kept:
        x1, x2 = _LineSpan(l)
        y1, y2 = _LineYRange(l)
        area = sum(c['area'] for c in l['comps'])
        absorbed = False
        for b in messBlocks:
            if not sparse(b):
                continue
            yov = max(0, min(y2, b['y2']) - max(y1, b['y1']))
            xgap = max(x1 - b['x2'], b['x1'] - x2)
            if yov >= 0.8 * (y2 - y1) and xgap < 2.0 * textH and \
                    area < 2.0 * textH * textH:
                b['comps'] += l['comps']
                b['x1'], b['x2'] = min(b['x1'], x1), max(b['x2'], x2)
                absorbed = True
                break
        if not absorbed:
            kept2.append(l)
    return messBlocks, kept2


def PruneFaintFragments(textLines, textH, illum, labels):
    """Ink bleeding through from the page's other side survives as a few
    faint fragments; a real (even short) text row is written in pen and is
    much darker. Prune small, clearly-fainter-than-the-page lines."""
    if len(textLines) < 4:
        return textLines
    darks = []
    for l in textLines:
        vals = []
        for c in l['comps']:
            sub = base._CompMask(labels, c)
            v = illum[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']][sub]
            vals.append(v)
        v = np.concatenate(vals)
        darks.append(255.0 - float(np.percentile(v, 30)))
    ref = float(np.median([d for d, l in zip(darks, textLines)
                           if sum(c['area'] for c in l['comps']) > 2 * textH * textH]
                          or darks))
    kept = []
    for l, d in zip(textLines, darks):
        area = sum(c['area'] for c in l['comps'])
        if d < 0.55 * ref and area < 3.0 * textH * textH:
            continue
        kept.append(l)
    return kept


def AttachFaintToLines(preInk, postInk, textH, textLines):
    """A word written with less pen pressure at the end of a row can fall
    below the faint filter's darkness threshold and vanish from the crop.
    Re-attach removed components that sit ON a detected row's baseline
    curve, inside or just beyond its span."""
    diff = preInk & ~postInk
    if not diff.any() or not textLines:
        return textLines
    dl, dn = F.LabelComponents(diff, connectivity=8)
    ds = F.ComponentStatsFromLabels(dl, dn)
    taken = set()
    for _ in range(2):        # 2nd pass: span/curve grow past attached words
        curves = []
        for l in textLines:
            curves.append((base._LineCurve(l['comps']),) + _LineSpan(l))
        for i, st in enumerate(ds, start=1):
            if st is None or i in taken or st['area'] < 25:
                continue
            if not (0.15 * textH < st['h'] < 2.2 * textH) or \
                    st['w'] > 8 * textH:
                continue
            best, bestDy = None, 0.75 * textH
            for l, (curve, x1, x2) in zip(textLines, curves):
                if not (x1 - 2 * textH <= st['cx'] <= x2 + 8 * textH):
                    continue
                dy = abs(base._CurveY(curve, st['cx']) - st['cy'])
                if dy < bestDy:
                    bestDy, best = dy, l
            if best is not None:
                taken.add(i)
                best['comps'].append(dict(
                    id=-i, x=st['x'], y=st['y'], w=st['w'], h=st['h'],
                    area=st['area'], cx=st['cx'], cy=st['cy'],
                    pixmask=(dl[st['y']:st['y'] + st['h'],
                                st['x']:st['x'] + st['w']] == i)))
    return textLines


def RescueFaintRows(preInk, postInk, textH, textLines, messBlocks):
    """The global faint-component filter can eat a WHOLE row written in
    pencil. Bleed-through ghosts always sit on/next to detected rows, but a
    real pencil row lives in a band of the page where nothing else was
    detected -- rescue exactly those: removed components that line up as a
    row in otherwise-empty space."""
    diff = preInk & ~postInk
    if not diff.any():
        return textLines
    bands = [_LineYRange(l) for l in textLines] + \
            [(b['y1'], b['y2']) for b in messBlocks]
    dl, dn = F.LabelComponents(diff, connectivity=8)
    ds = F.ComponentStatsFromLabels(dl, dn)
    cands = []
    for i, st in enumerate(ds, start=1):
        if st is None or st['area'] < 60:
            continue
        if not (0.25 * textH < st['h'] < 2.2 * textH):
            continue
        st = dict(st)
        st['id'] = i
        cands.append(st)
    cands.sort(key=lambda s: s['cy'])
    clusters = []
    for st in cands:
        for cl in clusters:
            if abs(st['cy'] - np.mean([c['cy'] for c in cl])) < 0.7 * textH:
                cl.append(st)
                break
        else:
            clusters.append([st])
    for cl in clusters:
        area = sum(c['area'] for c in cl)
        x1 = min(c['x'] for c in cl)
        x2 = max(c['x'] + c['w'] for c in cl)
        y1 = min(c['y'] for c in cl)
        y2 = max(c['y'] + c['h'] for c in cl)
        if len(cl) < 6 or area < 3 * textH * textH or (x2 - x1) < 8 * textH:
            continue
        ov = max((max(0, min(y2, b2) - max(y1, b1)) for b1, b2 in bands),
                 default=0)
        if ov > 0.25 * (y2 - y1):
            continue
        comps = []
        for c in cl:
            comps.append(dict(
                id=c['id'], x=c['x'], y=c['y'], w=c['w'], h=c['h'],
                area=c['area'], cx=c['cx'], cy=c['cy'],
                pixmask=(dl[c['y']:c['y'] + c['h'],
                            c['x']:c['x'] + c['w']] == c['id'])))
        nl = dict(comps=comps, rescued=True)
        _Recenter(nl)
        textLines.append(nl)

    # a rescued row is CONFIRMED faint writing, so the leftovers of its own
    # words (a pencil 'I' shattered into dashes, a tall 'of' with its f)
    # can attach with much looser gates than the global pass dares use
    rescued = [l for l in textLines if l.get('rescued')]
    if rescued:
        used = {c['id'] for l in rescued for c in l['comps']}
        # diff comps AttachFaintToLines already claimed carry id=-i
        used |= {-c['id'] for l in textLines for c in l['comps']
                 if c.get('id', 0) < 0}
        for i, st in enumerate(ds, start=1):
            if st is None or i in used or st['area'] < 10:
                continue
            if st['h'] > 3.5 * textH:
                continue
            best, bestDy = None, 0.9 * textH
            for l in rescued:
                x1, x2 = _LineSpan(l)
                if not (x1 - 2 * textH <= st['cx'] <= x2 + 8 * textH):
                    continue
                dy = abs(base._CurveY(base._LineCurve(l['comps']),
                                      st['cx']) - st['cy'])
                if dy < bestDy:
                    bestDy, best = dy, l
            if best is not None:
                best['comps'].append(dict(
                    id=i, x=st['x'], y=st['y'], w=st['w'], h=st['h'],
                    area=st['area'], cx=st['cx'], cy=st['cy'],
                    pixmask=(dl[st['y']:st['y'] + st['h'],
                                st['x']:st['x'] + st['w']] == i)))
                _Recenter(best)
    return textLines


def RefineItems(textLines, messBlocks, textH, illum=None, labels=None):
    messBlocks, textLines = MergeSparseMessBlocks(messBlocks, textLines, textH)
    textLines = PruneDebrisLines(textLines, textH)
    if illum is not None and labels is not None:
        textLines = PruneFaintFragments(textLines, textH, illum, labels)
    textLines = SplitStackedRows(textLines, textH, labels=labels)
    textLines = MergeSameRow(textLines, textH)
    textLines = ReassignUnderlines(textLines, textH, labels=labels)
    textLines, messBlocks = DemoteNarrowMess(textLines, messBlocks, textH)
    textLines = MergeSameRow(textLines, textH)
    for b in messBlocks:
        b['yc'] = 0.5 * (b['y1'] + b['y2'])
    return textLines, messBlocks


# ---------------------------------------------------------------------------
# 5. rendering: a crop may never contain ink assigned to ANOTHER item
# ---------------------------------------------------------------------------
def BuildOwnerMap(shape, items, labels):
    """Pixel map of which item index owns each ink pixel (-1 = unowned).
    Painted in item order; pixmask comps paint their exact pixels."""
    owner = np.full(shape, -1, np.int16)
    for k, it in enumerate(items):
        for c in it['comps']:
            sy = slice(c['y'], c['y'] + c['h'])
            sx = slice(c['x'], c['x'] + c['w'])
            if 'pixmask' in c:
                sel = c['pixmask']
            else:
                sel = labels[sy, sx] == c['id']
            owner[sy, sx][sel] = k
    return owner


def ExtendOwnerToWeak(owner, inkRawLabels, nRaw):
    """A weak-recovery component is the faint halo/fade of the strong ink
    it touches. If ALL the strong ink under it belongs to one item, the
    whole weak component belongs to that item too -- so another row's
    recovery can never grab its halo pixels. Mixed/unowned components stay
    up for grabs (rule-glued networks)."""
    sel = (inkRawLabels > 0) & (owner >= 0)
    L = inkRawLabels[sel]
    O = owner[sel].astype(np.int64)
    sole = np.full(nRaw + 1, -1, np.int64)
    if len(L):
        key = L.astype(np.int64) * (O.max() + 2) + O
        uk = np.unique(key)
        ul = uk // (O.max() + 2)
        uo = uk % (O.max() + 2)
        counts = np.bincount(ul, minlength=nRaw + 1)
        first = np.full(nRaw + 1, -1, np.int64)
        first[ul] = uo
        sole = np.where(counts == 1, first, np.where(counts > 1, -2, -1))
    ext = owner.copy()
    grab = (inkRawLabels > 0) & (owner < 0) & (sole[inkRawLabels] >= 0)
    ext[grab] = sole[inkRawLabels[grab]].astype(np.int16)
    return ext, sole


def AttachWeakTrailing(items, owner, sole, inkRawLabels, nRaw, textH,
                       weakLabels=None, nWeak=0, illumRef=None,
                       labels=None):
    """A pale trailing word too far from its row's strong ink for the
    render-time reach (a whole word gap or more) exists only in weak
    masks. If such a comp sits ON a text row's baseline curve (and is
    glyph-tall, not a flat rule dash), it is that row's word -- attach it
    so the crop paints it. Two pools: the gated recovery mask, and (for
    words with NO strong anchor at all) the raw weak threshold."""
    texts = [it for it in items if it['tag'] == 'TEXT']
    if not texts:
        return
    curves = [(base._LineCurve(it['comps']),
               _LineSpan(it), it) for it in texts]
    itemIdx = {id(it): j for j, it in enumerate(items)}

    inkDark = {}
    def rowDark(it):
        if id(it) not in inkDark:
            ds = [_CompDark(c, labels, illumRef) for c in it['comps']
                  if c['area'] >= 30]
            inkDark[id(it)] = float(np.median(ds)) if ds else 0.0
        return inkDark[id(it)]

    def place(st, sel, uid):
        if st['area'] < 40 or st['w'] > 10 * textH:
            return None
        if not (0.35 * textH < st['h'] < 2.5 * textH):
            return None
        if st['h'] <= max(5, 0.3 * textH) and st['w'] >= 1.5 * textH:
            return None                       # flat rule dash
        best, bestDy = None, 0.6 * textH
        for curve, (x1, x2), it in curves:
            if not (x1 - 2 * textH <= st['cx'] <= x2 + 12 * textH):
                continue
            dy = abs(base._CurveY(curve, st['cx']) - st['cy'])
            if dy < bestDy:
                bestDy, best = dy, it
        if best is None:
            return None
        if illumRef is not None and labels is not None:
            # bleed-through ghosts ride the same rule curves but are far
            # fainter than the row's own ink -- a real pale word is not
            vals = illumRef[st['y']:st['y'] + st['h'],
                            st['x']:st['x'] + st['w']][sel]
            if 255.0 - float(np.percentile(vals, 25)) < 0.5 * rowDark(best):
                return None
        k = itemIdx[id(best)]
        best['comps'].append(dict(
            id=uid, x=st['x'], y=st['y'], w=st['w'], h=st['h'],
            area=st['area'], cx=st['cx'], cy=st['cy'], pixmask=sel))
        sy = slice(st['y'], st['y'] + st['h'])
        sx = slice(st['x'], st['x'] + st['w'])
        owner[sy, sx][sel] = k
        return k

    rawStats = F.ComponentStatsFromLabels(inkRawLabels, nRaw)
    for i, st in enumerate(rawStats, start=1):
        if st is None:
            continue
        sy = slice(st['y'], st['y'] + st['h'])
        sx = slice(st['x'], st['x'] + st['w'])
        sel = inkRawLabels[sy, sx] == i
        # unclaimed comps attach; comps whose halo already belongs to the
        # matching row attach too; another row's comps are never stolen
        prev = sole[i]
        kPlaced = place(st, sel, -(nRaw + i))
        if kPlaced is not None and prev not in (-1, kPlaced):
            items[kPlaced]['comps'].pop()
            owner[sy, sx][sel] = prev

    if weakLabels is not None and nWeak > 0:
        wStats = F.ComponentStatsFromLabels(weakLabels, nWeak)
        for i, st in enumerate(wStats, start=1):
            if st is None:
                continue
            sy = slice(st['y'], st['y'] + st['h'])
            sx = slice(st['x'], st['x'] + st['w'])
            sel = weakLabels[sy, sx] == i
            # skip anything already known: overlaps recovery mask or owned
            if (sel & (inkRawLabels[sy, sx] > 0)).sum() > 0.2 * st['area']:
                continue
            if (sel & (owner[sy, sx] >= 0)).any():
                continue
            place(st, sel, -(nRaw + nWeak + i))


def _CompDark(c, labels, illum):
    sub = base._CompMask(labels, c)
    vals = illum[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']][sub]
    return 255.0 - float(np.percentile(vals, 25))


def FilterFaintFlatComps(comps, labels, illum, textH):
    """Residual printed-rule segments ride along a row as flat wide (or
    tall thin margin-stub) comps that are clearly FAINTER than the row's
    own glyph ink; a real pen underline is ink-dark and stays."""
    glyphs = [c for c in comps
              if not _UnderlineShaped(c, textH, labels)
              and not (c['h'] >= 1.5 * textH and c['w'] <= 0.4 * textH)]
    if len(glyphs) < 2:
        return comps
    darks = [_CompDark(c, labels, illum) for c in glyphs]
    areas = [c['area'] for c in glyphs]
    order = np.argsort(darks)
    cum = np.cumsum([areas[i] for i in order])
    ref = darks[order[np.searchsorted(cum, 0.5 * cum[-1])]]
    kept = []
    for c in comps:
        flat = _UnderlineShaped(c, textH, labels)
        stub = c['h'] >= 1.5 * textH and c['w'] <= 0.4 * textH
        if (flat or stub) and _CompDark(c, labels, illum) < 0.72 * ref:
            continue
        kept.append(c)
    return kept if kept else comps


def RenderLine(gray, labels, comps, pad=6, inkRaw=None, inkRawLabels=None,
               gapPx=6, deskew=False, forbid=None):
    """base.RenderLine plus a forbid mask: pixels owned by a different
    line/block are never painted into this crop, so a neighbouring row's
    descender or ascender cannot intrude even when it dips into this
    row's band."""
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
            mask[sy, sx] |= (labels[c['y']:c['y'] + c['h'],
                                    c['x']:c['x'] + c['w']] == c['id'])

    allowed = None
    if forbid is not None:
        allowed = ~forbid[y1:y2, x1:x2] | mask
    mask = F.Dilate(mask, 3, 3)
    if allowed is not None:
        mask &= allowed
    if inkRaw is not None:
        near = F.Dilate(mask, 9, 3)
        add = near & inkRaw[y1:y2, x1:x2]
        if allowed is not None:
            add &= allowed
        mask |= add
    if inkRawLabels is not None:
        window = inkRawLabels[y1:y2, x1:x2] > 0
        reachPx = gapPx * 3
        near = F.Dilate(mask, 4, 2 * reachPx + 1)
        add = near & window
        if allowed is not None:
            add &= allowed
        mask |= add

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


# ---------------------------------------------------------------------------
# pipeline (same shape as the baseline ProcessPage, new stages slotted in)
# ---------------------------------------------------------------------------
def _InkGray(rgb):
    """Luma blended with the channel-minimum: coloured pen ink (blue
    especially) is dark in at least one channel even where its LUMA nearly
    matches the paper (dim corners); pale printed rules gain only half
    that darkening, so ink/rule separation survives."""
    luma = F.RgbToGray(rgb).astype(np.float64)
    mn = rgb.min(axis=2).astype(np.float64)
    return (0.5 * luma + 0.5 * mn).astype(np.uint8)


def ProcessPage(imgPath):
    rgb = base.LoadImage(imgPath)
    pageMask = DetectPageMask(rgb)
    gray = F.RgbToGray(rgb)
    illum = base.CorrectIllumination(gray)

    ink0 = base.BinarizeInk(illum, pageMask) & ~base.RedInkMask(rgb)
    ink0 = base.RemoveSpeckles(ink0)

    # NOTE: EstimateSkew scores candidate angles with F.Rotate(ink, a), so
    # the page must be corrected with that SAME direction -- the baseline's
    # Rotate() helper flips the sign (base.Rotate(x, a) == F.Rotate(x, -a)),
    # which silently DOUBLED the skew of every tilted page and left the
    # per-line deskew to hide the damage. Second pass mops up any residue.
    angle = base.EstimateSkew(ink0, searchRange=8.0)
    applied = 0.0
    for rng in (None, 3.0):
        a = angle if rng is None else base.EstimateSkew(ink0, searchRange=rng)
        if abs(a) < 0.15:
            break
        rgb = base.Rotate(rgb, -a, fill=255)
        pageMask = base.Rotate(pageMask.astype(np.uint8), -a,
                               isMask=True).astype(bool)
        gray = F.RgbToGray(rgb)
        illum = base.CorrectIllumination(gray)
        ink0 = base.BinarizeInk(illum, pageMask) & ~base.RedInkMask(rgb)
        ink0 = base.RemoveSpeckles(ink0)
        applied += a
    angle = applied if applied != 0.0 else angle

    _, rc0 = base.ComponentStats(ink0)
    rH0 = base.EstimateTextHeight(rc0)
    # colour-aware second recovery pass: blue ink in a dim corner can be
    # nearly invisible in LUMA yet obvious in the channel-minimum -- feed
    # the crop-time recovery mask from both; detection stays luma-pure
    grayC = _InkGray(rgb)
    illumC = base.CorrectIllumination(grayC)
    inkRecovered = HysteresisRecoverInkWide(illum, pageMask, ink0, rH0)
    inkRecovered |= HysteresisRecoverInkWide(illumC, pageMask, ink0, rH0)
    hR = base._HorizontalRunLengths(inkRecovered)
    vR = base._VerticalRunLengths(inkRecovered)
    inkRaw = inkRecovered & ~((hR >= 10) & (vR <= 4))

    preFaint = ink0.copy()
    ink0 = FaintFilterRuleAware(ink0, illum, rH0)                 # NEW
    ink0 = RemoveEdgeComponentsWide(ink0, pageMask, rH0)          # NEW
    postFaint = ink0.copy()

    _, roughComps = base.ComponentStats(ink0)
    roughH = base.EstimateTextHeight(roughComps)

    ink0 = RemoveOffPageColumns(ink0, roughH, alsoClip=inkRaw)  # NEW

    ink, rm1 = base.RemoveRuleLines(ink0, roughH, illum=illum)
    ink, rm2 = base.RemoveRuleLines(ink, roughH, illum=illum)
    ink, rmAll = RestoreDrawingRules(ink, rm1 | rm2, illum, roughH)  # NEW
    ink = base.RemoveSpeckles(ink, minSize=12)
    ink = BreakRuleNetworksFaint(ink, roughH, illum)           # NEW
    ink = StripSparseRuleNetworks(ink, roughH)                 # NEW
    ink = base.RemoveSpeckles(ink, minSize=12)

    # detected rule pixels never re-enter crops via the recovery mask
    inkRaw &= ~F.Dilate(rmAll, 3, 3)                           # NEW
    inkRaw = CleanRecoveredInk(inkRaw, ink0, np.minimum(illum, illumC),
                               roughH)                         # NEW
    inkRawLabels, _ = F.LabelComponents(inkRaw, connectivity=8)

    labels, comps = base.ComponentStats(ink)
    textH = base.EstimateTextHeight(comps)
    textLines, messBlocks = base.GroupLines(ink, labels, comps, textH,
                                            ink.shape[0])
    textLines, messBlocks = RefineItems(textLines, messBlocks, textH,
                                        illum=illum, labels=labels)   # NEW
    textLines = AttachFaintToLines(preFaint, postFaint, textH,
                                   textLines)                         # NEW
    textLines = RescueFaintRows(preFaint, postFaint, textH,
                                textLines, messBlocks)                # NEW

    items = [dict(tag='TEXT', yc=l['yc'], comps=l['comps']) for l in textLines]
    items += [dict(tag='MESS', yc=b['yc'], comps=b['comps'],
                   y1=b['y1'], y2=b['y2']) for b in messBlocks]
    items.sort(key=lambda it: it['yc'])
    for k in range(len(items) - 1):
        a, b = items[k], items[k + 1]
        if a['tag'] == 'MESS' and b['tag'] == 'TEXT' and \
                a['y1'] <= b['yc'] <= a['y1'] + 0.4 * (a['y2'] - a['y1']):
            items[k], items[k + 1] = b, a

    for it in items:
        if it['tag'] == 'TEXT':
            it['comps'] = FilterFaintFlatComps(it['comps'], labels,
                                               illum, textH)
    owner = BuildOwnerMap(gray.shape, items, labels)
    nRaw = int(inkRawLabels.max())
    owner, sole = ExtendOwnerToWeak(owner, inkRawLabels, nRaw)
    # anchor-free weak pool: a pale word whose strong specks got speckle-
    # filtered has NO anchor for the recovery mask; find it in the raw
    # weak threshold (rules stripped) purely by row-curve position
    blockSize = max(15, 2 * (illum.shape[1] // 60) + 1)
    weakAll = F.AdaptiveThresholdInv(np.minimum(illum, illumC),
                                     blockSize, 4) & pageMask
    hRW = base._HorizontalRunLengths(weakAll)
    vRW = base._VerticalRunLengths(weakAll)
    weakAll &= ~((hRW >= 10) & (vRW <= 4))
    weakAll &= ~F.Dilate(rmAll, 3, 3)
    # thicker/wavy rules glue words into page-wide comps: run the same
    # span-based rule stripping the recovery mask gets
    weakAll = CleanRecoveredInk(weakAll | ink0, ink0,
                                np.minimum(illum, illumC),
                                roughH) & ~ink0
    weakLabels, nWeak = F.LabelComponents(weakAll, connectivity=8)
    AttachWeakTrailing(items, owner, sole, inkRawLabels, nRaw, textH,
                       weakLabels=weakLabels, nWeak=nWeak,
                       illumRef=np.minimum(illum, illumC), labels=labels)
    # crops paint from the illumination-corrected page (darkest of the
    # luma / colour evidence): background is uniform white and a stroke
    # keeps its contrast even in a dim corner -- exactly what the
    # classifier's training data looks like
    renderGray = np.minimum(illum, illumC)
    results = []
    preview = Image.fromarray(rgb.copy())
    drawObj = ImageDraw.Draw(preview)
    for order, it in enumerate(items):
        crop, bbox = RenderLine(renderGray, labels, it['comps'],
                                inkRaw=inkRaw,
                                inkRawLabels=inkRawLabels,
                                gapPx=max(8, int(textH * 0.55)),
                                deskew=(it['tag'] == 'TEXT'),
                                forbid=(owner >= 0) & (owner != order))
        color = (255, 140, 0) if it['tag'] == 'MESS' else (0, 190, 0)
        drawObj.rectangle(list(bbox), outline=color, width=3)
        drawObj.text((bbox[0], max(0, bbox[1] - 14)), f"{order}:{it['tag']}",
                     fill=color)
        results.append(dict(order=order, tag=it['tag'], bbox=bbox,
                            raw_crop=crop, n_components=len(it['comps'])))

    meta = dict(skew=angle, textH=textH,
                nText=sum(1 for r in results if r['tag'] == 'TEXT'),
                nMess=sum(1 for r in results if r['tag'] == 'MESS'))
    return results, preview, meta


# ---------------------------------------------------------------------------
# harness (same scoring/outputs as the baseline, own output folder: fp2)
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(base.SCRIPT_DIR, 'NOGIT', 'NonDatasetTestOutput', 'fp2')


def _RunOnImage(imgPath):
    name = os.path.splitext(os.path.basename(imgPath))[0]
    results, preview, meta = ProcessPage(imgPath)

    os.makedirs(os.path.join(OUTPUT_DIR, 'previews'), exist_ok=True)
    preview.save(os.path.join(OUTPUT_DIR, 'previews', name + '_preview.png'))

    cropDir = os.path.join(OUTPUT_DIR, 'crops', name)
    modelDir = os.path.join(OUTPUT_DIR, 'crops_model', name)
    for d in (cropDir, modelDir):
        if os.path.isdir(d):
            for f in glob.glob(os.path.join(d, '*.png')):
                os.remove(f)
        os.makedirs(d, exist_ok=True)
    for r in results:
        fname = f"line_{r['order']:02d}_{r['tag']}.png"
        img = Image.fromarray(r['raw_crop'])
        img.save(os.path.join(cropDir, fname))
        g = img.convert('L')
        s = min(base.INPUT_W / g.width, base.INPUT_H / g.height)
        g = g.resize((max(1, int(g.width * s)), max(1, int(g.height * s))),
                     Image.Resampling.BILINEAR)
        canvas = Image.new('L', (base.INPUT_W, base.INPUT_H), 255)
        canvas.paste(g, (0, (base.INPUT_H - g.height) // 2))
        canvas.save(os.path.join(modelDir, fname))

    labels = base.ReadLabels(imgPath)
    if labels is None:
        print(f'{name}: no label file -- segmented {len(results)} boxes')
        return None
    acc, expTags, detTags = base.Score(results, labels)
    print(f"{name}: acc={acc*100:.1f}%  expected {len(expTags)} rows "
          f"({expTags.count('MESS')} MESS) | detected {len(detTags)} "
          f"({detTags.count('MESS')} MESS) | skew={meta['skew']:.1f}deg")
    if acc < 1.0:
        for i in range(max(len(expTags), len(detTags))):
            e = expTags[i] if i < len(expTags) else '--'
            d = detTags[i] if i < len(detTags) else '--'
            flag = '' if e == d else '   <<< MISMATCH'
            lbl = labels[i][:50] if i < len(labels) else ''
            print(f'   {i:2d}  exp={e:4s} det={d:4s}  {lbl}{flag}')
    return acc


def RunAll():
    imagePaths = sorted(p for ext in ('*.png', '*.jpg', '*.jpeg')
                        for p in glob.glob(os.path.join(base.IMAGES_DIR, ext)))
    if not imagePaths:
        print(f'No images found in {base.IMAGES_DIR}')
        return
    accs = [a for p in imagePaths if (a := _RunOnImage(p)) is not None]
    if accs:
        print(f'== OVERALL: {np.mean(accs)*100:.1f}% ==')
    print(f'\nPreviews:    {os.path.join(OUTPUT_DIR, "previews")}')
    print(f'Crops:       {os.path.join(OUTPUT_DIR, "crops")}')
    print(f'Model crops: {os.path.join(OUTPUT_DIR, "crops_model")}')


if __name__ == '__main__':
    RunAll()
