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

import SegmentPage as F
from np_inference.text_model import CHAR_TO_IDX

SCRIPT_DIR = Path(__file__).resolve().parent
PROFILE_DIR = SCRIPT_DIR.parent / "NOGIT" / "StyleProfiles10"


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
    thr = F.OtsuThresholdValue(gray)
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
        ink = F.Close(ink, k, k)
    labels, n = F.LabelComponents(ink, connectivity=8)
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
