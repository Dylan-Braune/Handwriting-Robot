"""
SegmentPage.py -- first-principles page segmenter (numpy + PIL only, no
OpenCV/scipy). Used by the server to turn one photographed handwriting page
into ordered TEXT/MESS line crops, for both classification and training.

Single self-contained module: every primitive this pipeline needs --
low-level numpy image maths, baseline segmentation primitives, and the
improved/override stages real photos needed -- lives in this one file, in
three clearly marked sections below. Every stage is still live and
measurably changes real output (confirmed by ablation against real camera
photos); this is server-pipeline code only, not a standalone test tool.

What the override stages add versus the baseline primitives:

  1. Global deskew FIXED + widened. EstimateSkew scores candidates with
     Rotate(ink, a) [raw primitive], but the baseline corrected the page
     with the OLD wrapper direction Rotate(rgb, a) == RotateRaw(rgb, -a) --
     the WRONG direction, which silently doubled every tilted page's skew
     and left the per-line crop deskew to hide it. This rotates the right
     way (search range +-8 deg, plus a second residual pass), so rows are
     level BEFORE grouping and every crop comes out at 0 degrees.
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
     survive while the rule goes. AttachFaintToLines re-attaches pale
     trailing words to the row curve they sit on.
  5. StripSparseRuleNetworks: wide, sparse, hole-free networks (rules that
     glue words across rows) get their long thin near-horizontal strokes
     stripped COLUMN-WISE, which works on sloped rules that run-length
     tests miss. Real diagrams are protected by an enclosed-hole test.
  6. RefineItems, a post-grouping pass: MergeSameRow (four independent
     same-row signals -- near-identical centres / y-range containment /
     meeting baseline curves at the seam / interleaving columns -- rejoin
     rows that chaining split) then ReassignUnderlines (an underline that
     chained into the row BELOW is moved back to the text it underlines).
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

Entry point: ProcessPage(imgPath) -> (results, preview, meta).
"""

import numpy as np
from PIL import Image, ImageDraw, ImageOps

TARGET_LONG_SIDE = 2400
INPUT_H, INPUT_W = 64, 640  # keep in sync with TrainText.py


# ===========================================================================
# === from RawImageOps.py: low-level numpy image primitives ===
#
# Every operation the segmentation pipeline needs, implemented from scratch:
# no OpenCV, no scipy, no PIL processing (PIL is used elsewhere ONLY to
# decode and encode image files). This is the maths layer for the baseline
# primitives section below.
#
# Implementations chosen for clarity + vectorized numpy speed:
#   * Box sums via 2D cumulative-sum tables -> O(1) per pixel for any window.
#   * Gaussian blur approximated by 3 successive box blurs (central limit
#     theorem; error vs a true Gaussian is far below the noise floor of a
#     phone photo).
#   * Binary erosion/dilation with rectangular kernels via the same box sums.
#   * Connected components with a run-based two-pass union-find (rows are
#     encoded as ink runs; runs touching between adjacent rows are unioned).
#   * Hole filling via background labelling: a background region is a hole
#     iff it does not touch the image border.
#   * Rotation / resize via inverse-mapped bilinear sampling.
# ===========================================================================

# ---------------------------------------------------------------------------
# Box sums / blurs
# ---------------------------------------------------------------------------
def _Integral(img):
    """Summed-area table with a zero row/col on top/left."""
    ii = np.zeros((img.shape[0] + 1, img.shape[1] + 1), np.float64)
    np.cumsum(np.cumsum(img, axis=0), axis=1, out=ii[1:, 1:])
    return ii


def BoxSum(img, ry, rx):
    """Sum over a (2*ry+1) x (2*rx+1) window centred per pixel, with edge
    clamping (window truncated at borders)."""
    h, w = img.shape
    ii = _Integral(img)
    y = np.arange(h)
    x = np.arange(w)
    y1 = np.clip(y - ry, 0, h)[:, None]
    y2 = np.clip(y + ry + 1, 0, h)[:, None]
    x1 = np.clip(x - rx, 0, w)[None, :]
    x2 = np.clip(x + rx + 1, 0, w)[None, :]
    return ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]


def BoxCount(ry, rx, h, w):
    """Pixel count of the clamped window at each position."""
    y = np.arange(h)
    x = np.arange(w)
    cy = (np.clip(y + ry + 1, 0, h) - np.clip(y - ry, 0, h))[:, None]
    cx = (np.clip(x + rx + 1, 0, w) - np.clip(x - rx, 0, w))[None, :]
    return cy * cx


def BoxMean(img, ry, rx):
    return BoxSum(img.astype(np.float64), ry, rx) / BoxCount(ry, rx, *img.shape)


def GaussianBlur(img, sigma):
    """3x iterated box blur ~= Gaussian of the requested sigma."""
    if sigma <= 0:
        return img.astype(np.float64)
    # box radius so that 3 passes give variance ~= sigma^2:
    # var(box of full width W) = (W^2 - 1)/12 ; 3 passes -> (W^2-1)/4
    r = max(1, int(round(np.sqrt(4.0 * sigma * sigma / 3.0 + 1.0) / 2.0)))
    out = img.astype(np.float64)
    for _ in range(3):
        out = BoxMean(out, r, r)
    return out


def GaussianBlur1D(arr, sigma):
    r = max(1, int(round(np.sqrt(4.0 * sigma * sigma / 3.0 + 1.0) / 2.0)))
    out = arr.astype(np.float64)
    kernel = np.ones(2 * r + 1) / (2 * r + 1)
    for _ in range(3):
        out = np.convolve(np.pad(out, r, mode='edge'), kernel, mode='same')[r:-r]
    return out


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
def RgbToGray(rgb):
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)
    return np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)


def Saturation(rgb):
    """HSV S channel scaled to 0..255 (matches cv2 convention)."""
    f = rgb.astype(np.float64)
    mx = f.max(axis=2)
    mn = f.min(axis=2)
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9) * 255.0, 0.0)
    return s.astype(np.uint8)


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------
def OtsuThresholdValue(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    sumAll = np.dot(np.arange(256), hist)
    sumB = wB = 0.0
    maxVar, thresh = 0.0, 0
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sumAll - sumB) / wF
        var = wB * wF * (mB - mF) ** 2
        if var > maxVar:
            maxVar, thresh = var, t
    return thresh


def AdaptiveThresholdInv(gray, blockSize, C):
    """ink = pixel darker than (local gaussian mean - C).  Mirrors cv2's
    ADAPTIVE_THRESH_GAUSSIAN_C: the local mean is gaussian-weighted with
    cv2's derived sigma for the given block size."""
    sigma = 0.3 * ((blockSize - 1) * 0.5 - 1) + 0.8
    localMean = GaussianBlur(gray.astype(np.float64), sigma)
    return gray.astype(np.float64) < (localMean - C)


# ---------------------------------------------------------------------------
# Binary morphology (rectangular kernels) via box sums
# ---------------------------------------------------------------------------
def Dilate(mask, ky, kx):
    """Kernel of size ky x kx (odd sizes; centre anchored)."""
    ry, rx = ky // 2, kx // 2
    s = BoxSum(mask.astype(np.float64), ry, rx)
    # asymmetric even kernels: cv2 anchors differently, but the pipeline only
    # uses small symmetric-ish kernels where this is equivalent enough
    return s > 0.5


def Erode(mask, ky, kx):
    ry, rx = ky // 2, kx // 2
    s = BoxSum(mask.astype(np.float64), ry, rx)
    cnt = BoxCount(ry, rx, *mask.shape)
    return s >= cnt - 0.5


def Open(mask, ky, kx):
    return Dilate(Erode(mask, ky, kx), ky, kx)


def Close(mask, ky, kx):
    return Erode(Dilate(mask, ky, kx), ky, kx)


# ---------------------------------------------------------------------------
# Connected components: run-based two-pass union-find
# ---------------------------------------------------------------------------
class _RunUF:
    __slots__ = ('parent',)

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        p = self.parent
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _RowRuns(mask):
    """Per row: arrays of (start, end) column indices of ink runs."""
    h, w = mask.shape
    padded = np.zeros((h, w + 2), bool)
    padded[:, 1:-1] = mask
    d = np.diff(padded.astype(np.int8), axis=1)
    runs = []
    for r in range(h):
        starts = np.nonzero(d[r] == 1)[0]
        ends = np.nonzero(d[r] == -1)[0]
        runs.append((starts, ends))
    return runs


def LabelComponents(mask, connectivity=8):
    """Returns (labels int32 array with 0 = background, count)."""
    h, w = mask.shape
    rowRuns = _RowRuns(mask)
    # global run ids
    runIdStart = []
    total = 0
    for r in range(h):
        runIdStart.append(total)
        total += len(rowRuns[r][0])
    if total == 0:
        return np.zeros((h, w), np.int32), 0

    uf = _RunUF(total)
    pad = 1 if connectivity == 8 else 0
    for r in range(1, h):
        s1, e1 = rowRuns[r - 1]
        s2, e2 = rowRuns[r]
        if len(s1) == 0 or len(s2) == 0:
            continue
        i = j = 0
        while i < len(s1) and j < len(s2):
            # runs [s1[i], e1[i]) and [s2[j], e2[j]) touch?
            if s1[i] < e2[j] + pad and s2[j] < e1[i] + pad:
                uf.union(runIdStart[r - 1] + i, runIdStart[r] + j)
            if e1[i] < e2[j]:
                i += 1
            else:
                j += 1

    # resolve roots -> compact labels
    rootLabel = {}
    runLabel = np.zeros(total, np.int32)
    nextLabel = 1
    for k in range(total):
        root = uf.find(k)
        lab = rootLabel.get(root)
        if lab is None:
            lab = nextLabel
            rootLabel[root] = lab
            nextLabel += 1
        runLabel[k] = lab

    labels = np.zeros((h, w), np.int32)
    for r in range(h):
        starts, ends = rowRuns[r]
        base = runIdStart[r]
        for i in range(len(starts)):
            labels[r, starts[i]:ends[i]] = runLabel[base + i]
    return labels, nextLabel - 1


def ComponentStatsFromLabels(labels, count):
    """Per label 1..count: x, y, w, h, area, cx, cy (like cv2 stats)."""
    if count == 0:
        return []
    ys, xs = np.nonzero(labels)
    ls = labels[ys, xs]
    order = np.argsort(ls, kind='stable')
    ys, xs, ls = ys[order], xs[order], ls[order]
    bounds = np.searchsorted(ls, np.arange(1, count + 2))
    stats = []
    for i in range(count):
        a, b = bounds[i], bounds[i + 1]
        if a == b:
            stats.append(None)
            continue
        yy, xx = ys[a:b], xs[a:b]
        stats.append(dict(x=int(xx.min()), y=int(yy.min()),
                          w=int(xx.max() - xx.min() + 1),
                          h=int(yy.max() - yy.min() + 1),
                          area=int(b - a),
                          cx=float(xx.mean()), cy=float(yy.mean())))
    return stats


def FillHoles(mask):
    """Fill background regions not connected to the border (4-connectivity
    on the background, matching cv2 floodFill from a corner)."""
    bg = ~mask
    labels, count = LabelComponents(bg, connectivity=4)
    if count == 0:
        return mask
    border = np.zeros(count + 1, bool)
    border[labels[0, :]] = True
    border[labels[-1, :]] = True
    border[labels[:, 0]] = True
    border[labels[:, -1]] = True
    border[0] = True
    hole = ~border[labels] & bg
    return mask | hole


# ---------------------------------------------------------------------------
# Geometry: resize + rotate (inverse-mapped bilinear)
# ---------------------------------------------------------------------------
def ResizeBilinear(img, newH, newW):
    h, w = img.shape[:2]
    ys = (np.arange(newH) + 0.5) * h / newH - 0.5
    xs = (np.arange(newW) + 0.5) * w / newW - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    fy = np.clip(ys - y0, 0, 1)[:, None]
    fx = np.clip(xs - x0, 0, 1)[None, :]
    if img.ndim == 2:
        a = img[y0][:, x0].astype(np.float64)
        b = img[y0][:, x1].astype(np.float64)
        c = img[y1][:, x0].astype(np.float64)
        d = img[y1][:, x1].astype(np.float64)
        out = a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx + c * fy * (1 - fx) + d * fy * fx
        return out
    chans = [ResizeBilinear(img[:, :, k], newH, newW) for k in range(img.shape[2])]
    return np.stack(chans, axis=2)


def DownsampleArea(mask, factor):
    """Block-mean downsample of a binary/float mask by integer factor."""
    h, w = mask.shape
    hh, ww = h // factor, w // factor
    m = mask[:hh * factor, :ww * factor].astype(np.float64)
    return m.reshape(hh, factor, ww, factor).mean(axis=(1, 3))


def RotateRaw(arr, angleDeg, nearest=False, fill=0):
    """Rotate about the image centre, output same size (like warpAffine).
    This is the raw numpy primitive; Rotate() below is the pipeline-facing
    wrapper that flips the angle sign for the baseline's convention -- the
    two are DIFFERENT functions that happened to share a name in the
    pre-merge files (RawImageOps.Rotate vs SegmentPageCore.Rotate), so this
    one was renamed on merge to avoid a collision. Callers that previously
    said F.Rotate(...) now say RotateRaw(...); callers that previously said
    base.Rotate(...) now say the bare Rotate(...) below, unchanged."""
    theta = np.deg2rad(angleDeg)
    cosT, sinT = np.cos(theta), np.sin(theta)
    if arr.ndim == 3:
        chans = [RotateRaw(arr[:, :, k], angleDeg, nearest=nearest,
                        fill=(fill[k] if hasattr(fill, '__len__') else fill))
                 for k in range(arr.shape[2])]
        return np.stack(chans, axis=2)

    h, w = arr.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    # inverse map: rotate output coords by -angle
    dx = xx - cx
    dy = yy - cy
    srcX = cosT * dx - sinT * dy + cx
    srcY = sinT * dx + cosT * dy + cy

    if nearest:
        sx = np.rint(srcX).astype(int)
        sy = np.rint(srcY).astype(int)
        valid = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        out = np.full(arr.shape, fill, dtype=arr.dtype)
        out[valid] = arr[sy[valid], sx[valid]]
        return out

    x0 = np.floor(srcX).astype(int)
    y0 = np.floor(srcY).astype(int)
    fx = srcX - x0
    fy = srcY - y0
    valid = (x0 >= 0) & (x0 < w - 1) & (y0 >= 0) & (y0 < h - 1)
    x0c = np.clip(x0, 0, w - 2)
    y0c = np.clip(y0, 0, h - 2)
    a = arr[y0c, x0c].astype(np.float64)
    b = arr[y0c, x0c + 1].astype(np.float64)
    c = arr[y0c + 1, x0c].astype(np.float64)
    d = arr[y0c + 1, x0c + 1].astype(np.float64)
    interp = a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx + c * fy * (1 - fx) + d * fy * fx
    out = np.full(arr.shape, float(fill), np.float64)
    out[valid] = interp[valid]
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(np.rint(out), 0, 255).astype(arr.dtype)
    return out.astype(arr.dtype)


# ===========================================================================
# === from SegmentPageCore.py: baseline segmentation primitives ===
#
# Image loading/normalizing, illumination correction, binarization, deskew,
# rule-line removal, connected-component/MESS scoring, and chain-based line
# grouping. The override section below (SegmentPage.py's own stages) builds
# on top of these: it overrides page-mask detection, ink recovery, and
# rendering with improved versions -- only the primitives that section
# still actually calls live here.
# ===========================================================================

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
    out = ResizeBilinear(arr, newH, newW)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Page detection: largest bright, low-saturation region
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Illumination, binarization, red-ink mask, speckles
# ---------------------------------------------------------------------------
def CorrectIllumination(gray):
    bg = GaussianBlur(gray, gray.shape[1] / 30.0)
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
    return AdaptiveThresholdInv(illum, blockSize, 12) & pageMask


def RemoveSpeckles(ink, minSize=10):
    labels, n = LabelComponents(ink, connectivity=8)
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
    small = DownsampleArea(ink, 4) > 0

    def score(angle):
        rot = RotateRaw(small.astype(np.uint8), angle, nearest=True, fill=0)
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
    return RotateRaw(arr, -angle, nearest=isMask, fill=fill)


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
    ruleMask = Dilate(ruleMask, 2, 2)
    ruleMask &= ((vRun <= thkCut + 2) | (hRun <= thkCut + 2))
    return ink & ~ruleMask, ruleMask


def _ChainLongStructures(cand, minSpan, axis, textH, thk, lo=None, hi=None, illum=None):
    labels, n = LabelComponents(cand, connectivity=8)
    if n == 0:
        return np.zeros_like(cand)
    stats = ComponentStatsFromLabels(labels, n)
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
    labels, n = LabelComponents(ink, connectivity=8)
    stats = ComponentStatsFromLabels(labels, n)
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
    filled = FillHoles(padded)
    holesMask = filled & ~padded
    maxHole = 0.0
    if holesMask.any():
        hl, nh = LabelComponents(holesMask, connectivity=4)
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


# ===========================================================================
# === SegmentPage.py's own improved/override stages ===
#
# Every stage below is still live and measurably changes real output
# (confirmed by ablation against real camera photos) -- functions that
# turned out to have no effect, or that existed only for a separate
# baseline-scoring/test harness, have been removed.
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. page mask: gentler ragged trim (keep a page cut off by the photo frame)
# ---------------------------------------------------------------------------
def _TrimRaggedGentle(mask, factor=0.30):
    for axis in (1, 0):
        prof = mask.sum(axis=axis).astype(np.float64)
        if prof.max() <= 0:
            return mask
        good = prof >= factor * np.median(prof[prof > prof.max() * 0.2])
        keep = _LargestGoodBlock(good)
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
    """_TrimDarkEdges with the cut placed relative to what actually lies OFF
    the page: a page edge in soft shadow (median ~0.7x paper) is still far
    brighter than the table/background, so it stays; only rows/columns as
    dark as the true background get trimmed."""
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
        keep = _LargestGoodBlock(med >= cut)
        mask = mask & (keep[:, None] if axis == 0 else keep[None, :])
    return mask


def DetectPageMask(rgb):
    gray = RgbToGray(rgb)
    sat = Saturation(rgb)
    blur = np.clip(GaussianBlur(gray, 4), 0, 255)
    otsuVal = OtsuThresholdValue(blur.astype(np.uint8))

    paperish = (blur >= otsuVal) & (sat < 90)
    paperish = Open(paperish, 9, 9)
    labels, n = LabelComponents(paperish, connectivity=4)
    if n == 0:
        return np.ones(gray.shape, bool)
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    mask = labels == int(np.argmax(areas))

    paperMed = float(np.median(blur[mask]))
    paperish2 = (blur >= paperMed - 50) & (sat < 90)
    paperish2 = Open(paperish2, 9, 9)
    labels, n = LabelComponents(paperish2, connectivity=4)
    if n > 0:
        areas = np.bincount(labels.ravel())
        areas[0] = 0
        mask = labels == int(np.argmax(areas))
    mask = FillHoles(mask)
    mask = Close(mask, 15, 15)
    mask = FillHoles(mask)

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
        mask = FillHoles(mask)

    er = max(4, int(min(mask.shape) * 0.006))
    kk = 2 * (er // 2) + 1
    mask = Erode(mask, kk, kk)
    if mask.mean() < 0.15:
        return np.ones(gray.shape, bool)
    return mask


def RemoveEdgeComponentsWide(ink, pageMask, textH):
    """RemoveEdgeComponents with a wider tolerance (the hull-filled mask
    reaches further than the writing area) plus a corner rule: junk
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
    labels, n = LabelComponents(ink, connectivity=8)
    if n == 0:
        return ink
    stats = ComponentStatsFromLabels(labels, n)
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
    labels, n = LabelComponents(ink, connectivity=8)
    if n == 0:
        return ink
    stats = ComponentStatsFromLabels(labels, n)
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
    holes = FillHoles(padded) & ~padded
    if not holes.any():
        return 0.0
    hl, nh = LabelComponents(holes, connectivity=4)
    if nh == 0:
        return 0.0
    return float(np.bincount(hl.ravel())[1:].max())


def _CompDarkness(labels, n, illum):
    stats = ComponentStatsFromLabels(labels, n)
    darkness = np.zeros(n + 1, np.float64)
    for i, st in enumerate(stats, start=1):
        if st is None:
            continue
        sub = labels[st['y']:st['y'] + st['h'], st['x']:st['x'] + st['w']] == i
        vals = illum[st['y']:st['y'] + st['h'], st['x']:st['x'] + st['w']][sub]
        darkness[i] = 255.0 - float(np.percentile(vals, 25))
    return darkness, stats


def HysteresisRecoverInkWide(illum, pageMask, strong, textH):
    """HysteresisRecoverInk with a wider HORIZONTAL proximity window: a
    lightly-pressed word one word-gap away from its row's strong ink (e.g.
    a pale trailing word) is recoverable at crop-render time. The vertical
    window stays tight so nothing bridges between rows."""
    blockSize = max(15, 2 * (illum.shape[1] // 60) + 1)
    weak = AdaptiveThresholdInv(illum, blockSize, 3) & pageMask
    labels, n = LabelComponents(weak, connectivity=8)
    if n == 0:
        return strong
    reach = 2 * int(2.0 * textH) + 1
    touchesNear = np.zeros(n + 1, bool)
    t = labels[Dilate(strong, 9, 25)]
    touchesNear[t[t > 0]] = True
    touchesFar = np.zeros(n + 1, bool)
    t = labels[Dilate(strong, 9, reach)]
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
    wl0, wn0 = LabelComponents(weakOnly, connectivity=8)
    if wn0 > 0:
        st0 = ComponentStatsFromLabels(wl0, wn0)
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

    wl, wn = LabelComponents(weakOnly, connectivity=8)
    if wn == 0:
        return inkRaw
    stats = ComponentStatsFromLabels(wl, wn)
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
    """FilterFaintComponents, but rule-glue aware: a residual ruled line
    with word strokes touching it forms ONE wide faint-ish component, and
    the whole-component filter throws the words away with the rule. Here
    the long thin runs of such components are DETACHED first, each side is
    judged on its own darkness, and dark pieces survive -- so the words
    stay, the faint rule goes, and a diagram's long dark strokes are never
    harmed."""
    labels, n = LabelComponents(ink, connectivity=8)
    if n == 0:
        return ink
    stats = ComponentStatsFromLabels(labels, n)
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

    cl, cn = LabelComponents(inkCut, connectivity=8)
    if cn <= 1:
        return ink
    darkness, _ = _CompDarkness(cl, cn, illum)
    d = darkness[1:]
    lo, hi = np.percentile(d, 10), np.percentile(d, 90)
    if hi - lo < 60:
        return ink
    thr = _Otsu1D(d)
    if thr <= lo or thr >= hi:
        return ink
    keep = darkness >= thr
    keep[0] = False
    out = keep[cl] & inkCut

    if thinPixels.any():
        tl, tn = LabelComponents(thinPixels, connectivity=8)
        if tn > 0:
            tdark, _ = _CompDarkness(tl, tn, illum)
            tkeep = tdark >= thr
            tkeep[0] = False
            out |= tkeep[tl] & thinPixels
    return out


def StripSparseRuleNetworks(ink, textH):
    labels, n = LabelComponents(ink, connectivity=8)
    if n == 0:
        return ink
    stats = ComponentStatsFromLabels(labels, n)
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
                    ca = _LineCurve(a['comps'])
                    cb = _LineCurve(b['comps'])
                    xq = (max(ax1, bx1) + min(ax2, bx2)) / 2.0
                    dy = abs(_CurveY(ca, xq) - _CurveY(cb, xq))
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


def _UnderlineShaped(c, textH, labels):
    """Flat-wide comp, judged by per-column stroke span so a double/triple
    underline (tall bbox, thin strokes) still counts."""
    if c['w'] < 1.5 * textH:
        return False
    if c['h'] <= max(4, 0.45 * textH):
        return True
    if c['h'] > 1.2 * textH or labels is None:
        return False
    sub = _CompMask(labels, c)
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
        curveL = _LineCurve(solid)
        moved = False
        for c in list(l['comps']):
            if c in solid:
                continue
            if _CurveY(curveL, c['cx']) - c['cy'] < 0.5 * textH:
                continue          # sits on/below this line's centre: fine
            best, bestDy = None, None
            for o in others:
                x1, x2 = _LineSpan(o)
                if not (x1 - 2 * textH <= c['cx'] <= x2 + 2 * textH):
                    continue
                dy = c['cy'] - _CurveY(_LineCurve(o['comps']),
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


def AttachFaintToLines(preInk, postInk, textH, textLines):
    """A word written with less pen pressure at the end of a row can fall
    below the faint filter's darkness threshold and vanish from the crop.
    Re-attach removed components that sit ON a detected row's baseline
    curve, inside or just beyond its span."""
    diff = preInk & ~postInk
    if not diff.any() or not textLines:
        return textLines
    dl, dn = LabelComponents(diff, connectivity=8)
    ds = ComponentStatsFromLabels(dl, dn)
    taken = set()
    for _ in range(2):        # 2nd pass: span/curve grow past attached words
        curves = []
        for l in textLines:
            curves.append((_LineCurve(l['comps']),) + _LineSpan(l))
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
                dy = abs(_CurveY(curve, st['cx']) - st['cy'])
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


def RefineItems(textLines, messBlocks, textH, labels=None):
    textLines = MergeSameRow(textLines, textH)
    textLines = ReassignUnderlines(textLines, textH, labels=labels)
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
    curves = [(_LineCurve(it['comps']),
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
            dy = abs(_CurveY(curve, st['cx']) - st['cy'])
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

    rawStats = ComponentStatsFromLabels(inkRawLabels, nRaw)
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
        wStats = ComponentStatsFromLabels(weakLabels, nWeak)
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
    sub = _CompMask(labels, c)
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
    """Baseline line-rendering plus a forbid mask: pixels owned by a
    different line/block are never painted into this crop, so a
    neighbouring row's descender or ascender cannot intrude even when it
    dips into this row's band."""
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
    mask = Dilate(mask, 3, 3)
    if allowed is not None:
        mask &= allowed
    if inkRaw is not None:
        near = Dilate(mask, 9, 3)
        add = near & inkRaw[y1:y2, x1:x2]
        if allowed is not None:
            add &= allowed
        mask |= add
    if inkRawLabels is not None:
        window = inkRawLabels[y1:y2, x1:x2] > 0
        reachPx = gapPx * 3
        near = Dilate(mask, 4, 2 * reachPx + 1)
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
            rot = RotateRaw(padded, angle, fill=255)
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
    luma = RgbToGray(rgb).astype(np.float64)
    mn = rgb.min(axis=2).astype(np.float64)
    return (0.5 * luma + 0.5 * mn).astype(np.uint8)


def ProcessPage(imgPath):
    rgb = LoadImage(imgPath)
    pageMask = DetectPageMask(rgb)
    gray = RgbToGray(rgb)
    illum = CorrectIllumination(gray)

    ink0 = BinarizeInk(illum, pageMask) & ~RedInkMask(rgb)
    ink0 = RemoveSpeckles(ink0)

    # NOTE: EstimateSkew scores candidate angles with RotateRaw(ink, a), so
    # the page must be corrected with that SAME direction -- the baseline's
    # Rotate() wrapper flips the sign (Rotate(x, a) == RotateRaw(x, -a)),
    # which silently DOUBLED the skew of every tilted page and left the
    # per-line deskew to hide the damage. Second pass mops up any residue.
    angle = EstimateSkew(ink0, searchRange=8.0)
    applied = 0.0
    for rng in (None, 3.0):
        a = angle if rng is None else EstimateSkew(ink0, searchRange=rng)
        if abs(a) < 0.15:
            break
        rgb = Rotate(rgb, -a, fill=255)
        pageMask = Rotate(pageMask.astype(np.uint8), -a,
                          isMask=True).astype(bool)
        gray = RgbToGray(rgb)
        illum = CorrectIllumination(gray)
        ink0 = BinarizeInk(illum, pageMask) & ~RedInkMask(rgb)
        ink0 = RemoveSpeckles(ink0)
        applied += a
    angle = applied if applied != 0.0 else angle

    _, rc0 = ComponentStats(ink0)
    rH0 = EstimateTextHeight(rc0)
    # colour-aware second recovery pass: blue ink in a dim corner can be
    # nearly invisible in LUMA yet obvious in the channel-minimum -- feed
    # the crop-time recovery mask from both; detection stays luma-pure
    grayC = _InkGray(rgb)
    illumC = CorrectIllumination(grayC)
    inkRecovered = HysteresisRecoverInkWide(illum, pageMask, ink0, rH0)
    inkRecovered |= HysteresisRecoverInkWide(illumC, pageMask, ink0, rH0)
    hR = _HorizontalRunLengths(inkRecovered)
    vR = _VerticalRunLengths(inkRecovered)
    inkRaw = inkRecovered & ~((hR >= 10) & (vR <= 4))

    preFaint = ink0.copy()
    ink0 = FaintFilterRuleAware(ink0, illum, rH0)                 # NEW
    ink0 = RemoveEdgeComponentsWide(ink0, pageMask, rH0)          # NEW
    postFaint = ink0.copy()

    _, roughComps = ComponentStats(ink0)
    roughH = EstimateTextHeight(roughComps)

    ink0 = RemoveOffPageColumns(ink0, roughH, alsoClip=inkRaw)  # NEW

    ink, rm1 = RemoveRuleLines(ink0, roughH, illum=illum)
    ink, rm2 = RemoveRuleLines(ink, roughH, illum=illum)
    rmAll = rm1 | rm2
    ink = RemoveSpeckles(ink, minSize=12)
    ink = StripSparseRuleNetworks(ink, roughH)                 # NEW
    ink = RemoveSpeckles(ink, minSize=12)

    # detected rule pixels never re-enter crops via the recovery mask
    inkRaw &= ~Dilate(rmAll, 3, 3)                           # NEW
    inkRaw = CleanRecoveredInk(inkRaw, ink0, np.minimum(illum, illumC),
                               roughH)                         # NEW
    inkRawLabels, _ = LabelComponents(inkRaw, connectivity=8)

    labels, comps = ComponentStats(ink)
    textH = EstimateTextHeight(comps)
    textLines, messBlocks = GroupLines(ink, labels, comps, textH,
                                       ink.shape[0])
    textLines, messBlocks = RefineItems(textLines, messBlocks, textH,
                                        labels=labels)   # NEW
    textLines = AttachFaintToLines(preFaint, postFaint, textH,
                                   textLines)                         # NEW

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
    weakAll = AdaptiveThresholdInv(np.minimum(illum, illumC),
                                   blockSize, 4) & pageMask
    hRW = _HorizontalRunLengths(weakAll)
    vRW = _VerticalRunLengths(weakAll)
    weakAll &= ~((hRW >= 10) & (vRW <= 4))
    weakAll &= ~Dilate(rmAll, 3, 3)
    # thicker/wavy rules glue words into page-wide comps: run the same
    # span-based rule stripping the recovery mask gets
    weakAll = CleanRecoveredInk(weakAll | ink0, ink0,
                                np.minimum(illum, illumC),
                                roughH) & ~ink0
    weakLabels, nWeak = LabelComponents(weakAll, connectivity=8)
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
