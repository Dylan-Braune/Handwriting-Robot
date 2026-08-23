"""
fp_ops.py -- first-principles image operations in pure numpy.

Every operation the segmentation pipeline needs, implemented from scratch:
no OpenCV, no scipy, no PIL processing (PIL is used elsewhere ONLY to decode
and encode image files).  This is the maths layer for
NonDatasetSegmenterFP.py.

Implementations chosen for clarity + vectorized numpy speed:
  * Box sums via 2D cumulative-sum tables -> O(1) per pixel for any window.
  * Gaussian blur approximated by 3 successive box blurs (central limit
    theorem; error vs a true Gaussian is far below the noise floor of a
    phone photo).
  * Binary erosion/dilation with rectangular kernels via the same box sums.
  * Connected components with a run-based two-pass union-find (rows are
    encoded as ink runs; runs touching between adjacent rows are unioned).
  * Hole filling via background labelling: a background region is a hole iff
    it does not touch the image border.
  * Rotation / resize via inverse-mapped bilinear sampling.
"""

import numpy as np


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


def Rotate(arr, angleDeg, nearest=False, fill=0):
    """Rotate about the image centre, output same size (like warpAffine)."""
    theta = np.deg2rad(angleDeg)
    cosT, sinT = np.cos(theta), np.sin(theta)
    if arr.ndim == 3:
        chans = [Rotate(arr[:, :, k], angleDeg, nearest=nearest,
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
