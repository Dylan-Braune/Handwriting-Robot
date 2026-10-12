"""The handful of things the LIVE server actually needs from the style
profile pipeline: loading a saved profile, forced-aligning a recognizer's
output against known text, and the binarize/skeletonize/band primitives
`np_inference/author_model.py` reuses for stroke normalization.

Split out of BuildStyleProfile.py (which also contains ~1000 lines of
OFFLINE glyph-library-building code -- BuildAll, ExtractLineGlyphs, etc.
-- only ever run by hand to rebuild a profile, never touched by a live
request) so importing this module doesn't drag that in, and doesn't need
TrainText.py (and the torch/torchvision/scipy/pytesseract it imports)
just for one constant now duplicated in np_inference.text_model.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_inference.text_model import CHAR_TO_IDX

SCRIPT_DIR = Path(__file__).resolve().parent
PROFILE_DIR = SCRIPT_DIR.parent / "NOGIT" / "StyleProfiles10"


# Image primitives copied verbatim from SegmentPage.py rather than
# imported from it: importing that module for these dragged its whole
# 1,659 lines of page-layout analysis into every live request's import
# closure, just for Otsu, a closing and a connected-component labeller.
# Same reasoning that split this module out of BuildStyleProfile.py.
# Verified identical to the originals; keep them so if SegmentPage changes.

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



def LoadProfile(authorId):
    p = PROFILE_DIR / f"{authorId}.json"
    with open(p, encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CTC forced alignment (Viterbi over the blank-interleaved label sequence)
# ---------------------------------------------------------------------------
def CtcForcedAlign(logProbs, text):
    """logProbs: (T, C) numpy log-probs. Returns per-char (t0, t1) spans
    (inclusive-exclusive) of the timesteps Viterbi assigns to each char of
    `text`, or None if the text can't be aligned."""
    labels = [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]
    if not labels:
        return None
    ext = [0]
    for l in labels:
        ext += [l, 0]
    S, T = len(ext), logProbs.shape[0]
    if T < len(labels):
        return None
    NEG = -1e30
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), np.int32)
    dp[0, 0] = logProbs[0, ext[0]]
    if S > 1:
        dp[0, 1] = logProbs[0, ext[1]]
    for t in range(1, T):
        emit = logProbs[t, ext]
        stay = dp[t - 1]
        prev1 = np.concatenate(([NEG], dp[t - 1, :-1]))
        prev2 = np.concatenate(([NEG, NEG], dp[t - 1, :-2]))
        # skip-transition only allowed onto a non-blank that differs from
        # the non-blank two back (standard CTC topology)
        for s in range(S):
            if s >= 2 and (ext[s] == 0 or ext[s] == ext[s - 2]):
                prev2[s] = NEG
        cand = np.stack([stay, prev1, prev2])
        best = np.argmax(cand, axis=0)
        dp[t] = cand[best, np.arange(S)] + emit
        bp[t] = best
    endS = S - 1 if S == 1 else (S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2)
    path = np.zeros(T, np.int32)
    s = endS
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= bp[t, s]
    spans = []
    for ci in range(len(labels)):
        sIdx = 1 + 2 * ci
        ts = np.nonzero(path == sIdx)[0]
        if len(ts) == 0:
            spans.append(None)
        else:
            spans.append((int(ts[0]), int(ts[-1]) + 1))
    kept = [c for c in text if c in CHAR_TO_IDX]
    # mean per-frame log-prob along the chosen path: low means the model
    # never really recognised this line, so its char boundaries (and any
    # glyphs cut from them) are not trustworthy
    conf = float(np.mean([logProbs[t, ext[path[t]]] for t in range(T)]))
    return list(zip(kept, spans)), conf


# ---------------------------------------------------------------------------
# Skeletonization (Zhang-Suen thinning, vectorized numpy)
# ---------------------------------------------------------------------------
def Skeletonize(mask):
    img = mask.astype(np.uint8).copy()

    def neighbours(P):
        p = np.pad(P, 1)
        return (p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:],
                p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2])

    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P2, P3, P4, P5, P6, P7, P8, P9 = neighbours(img)
            ring = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            B = sum(ring[:8])
            A = sum(((ring[k] == 0) & (ring[k + 1] == 1)).astype(np.uint8)
                    for k in range(8))
            if step == 0:
                c1 = (P2 * P4 * P6) == 0
                c2 = (P4 * P6 * P8) == 0
            else:
                c1 = (P2 * P4 * P8) == 0
                c2 = (P2 * P6 * P8) == 0
            cond = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c1 & c2
            if cond.any():
                img[cond] = 0
                changed = True
    return img.astype(bool)


def _HRun(mask):
    """Horizontal run length at every pixel (same idea as the segmenter's
    helper, kept local so this module stays self-contained)."""
    h, w = mask.shape
    a = mask.astype(np.int32)
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


def BinarizeLine(gray, stripRules=True, closeGaps=True):
    """gray: uint8 numpy (ink dark). Returns bool ink mask.

    Underlines and ruled-paper lines are removed: a stroke that runs
    horizontally for far longer than a letter is wide while staying only a
    pen-width thick is a rule, not handwriting. Leaving them in poisons
    everything downstream -- they glue every letter together (so the hand
    measures as fully cursive), drag the slant estimate toward horizontal,
    and paste a bar across every extracted glyph."""
    thr = OtsuThresholdValue(gray)
    ink = gray < thr
    if stripRules and ink.any():
        h = ink.shape[0]
        hRun = _HRun(ink)
        vRun = _HRun(ink.T).T
        rule = (hRun >= max(24, int(1.1 * h))) & (vRun <= max(3, int(0.12 * h)))
        if rule.any():
            # keep the crossing points of real strokes: only drop rule
            # pixels that no vertical stroke passes through
            ink = ink & ~(rule & (vRun <= max(3, int(0.12 * h))))
    if closeGaps and ink.any():
        # A light, fast hand lays down thin strokes that the scan breaks
        # into pieces; skeletonizing those gives letter fragments instead
        # of letters (worst on tall thin strokes: h, l, t). A small closing
        # rejoins a hairline break without merging neighbouring letters --
        # the kernel is a fraction of the stroke pitch, not a fixed size.
        h = ink.shape[0]
        k = int(np.clip(round(0.045 * h), 2, 4))
        ink = Close(ink, k, k)
    labels, n = LabelComponents(ink, connectivity=8)
    if n:
        areas = np.bincount(labels.ravel())
        keep = areas >= 6
        keep[0] = False
        ink = keep[labels]
    return ink


def CoreBand(ink):
    """Baseline / topline of the x-height band from the horizontal ink
    profile (rows with >= 45% of peak density)."""
    prof = ink.sum(axis=1).astype(np.float64)
    if prof.max() <= 0:
        return None
    rows = np.nonzero(prof >= 0.45 * prof.max())[0]
    top, base = int(rows[0]), int(rows[-1])
    if base - top < 3:
        return None
    return top, base


def EstimateSlantDeg(ink):
    """Shear-search: the slant is the shear that makes vertical strokes
    vertical, i.e. maximizes the sharpness of the column projection.

    Sign convention (verified against a synthetically sheared test image):
    POSITIVE = the normal forward/rightward lean, so de-slanting a glyph is
    `x -= tan(slant) * y` with y measured up from the baseline."""
    ys, xs = np.nonzero(ink)
    if len(xs) < 50:
        return 0.0
    yc = ys.mean()
    best, bestScore = 0.0, -1.0
    for deg in np.arange(-45, 45.5, 1.5):
        sh = np.tan(np.radians(deg))
        xsh = (xs + (ys - yc) * -sh).astype(np.int64)
        prof = np.bincount(xsh - xsh.min())
        score = float((prof.astype(np.float64) ** 2).sum())
        if score > bestScore:
            bestScore, best = score, float(deg)
    return -best
