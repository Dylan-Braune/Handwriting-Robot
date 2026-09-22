"""
BuildStyleProfile.py -- per-author handwriting style extraction for the
Handwriting-Robot's WRITING side.

From each of the 10 target authors' pages (Data/Datasets/IAMpages10) this
builds a reusable style profile:

  (a) a stroke-level GLYPH LIBRARY: every line crop is force-aligned
      against its transcript with the frozen text recognizer (CTC Viterbi
      alignment -- the model is only run for inference, never modified),
      which gives an x-span per character even in fully cursive writing.
      The ink in each span is binarized, skeletonized to 1px centerlines
      (Zhang-Suen), traced into ordered polylines, normalized into a
      slant-removed baseline/x-height frame, and stored as one glyph
      VARIANT (several variants per character are kept, so an author's
      natural variation survives).

  (b) global STYLE PARAMETERS measured over all lines: slant angle,
      x-height and ascender/descender ratios, letter pitch and word
      spacing, baseline drift, stroke width, and how connected the
      writing is (fraction of adjacent in-word letters whose ink is one
      component -- the cursive-ness that synthesis must reproduce).

Profiles are fitted ONLY on non-holdout pages (same holdout rule as the
trainers), so synthesis can be evaluated on text it never saw, and are
saved to NOGIT/StyleProfiles10/<author>.json for reuse.

Run directly to build all 10 profiles:
    python BuildStyleProfile.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# the shared modules (RawImageOps, TrainText, SegmentPage, ExtractIAMLines)
# live one level up in Software/CNN
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import RawImageOps as F
from TrainText import (
    CHAR_TO_IDX,
    CHARSET,
    IAMLineDatasetRaw,
    PaperCRNN,
    _decode_png,
    resize_line_image_fixed,
    frame_x_to_pixel,
    INPUT_WIDTH,
    tensor_from_resized,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[2] / "Data" / "Datasets" / "IAMpages10"
CACHE_DIR = SCRIPT_DIR.parent / "NOGIT" / "line_cache_authors10"
TEXT_WEIGHTS = SCRIPT_DIR.parent / "NOGIT" / "weights" / "paper_cnn_bilstm_ctc_best.pt"
# Prefer the best available recogniser, in order:
#   1) joint (Teklia + personal, warm-started, trained together every
#      epoch so nothing gets forgotten) -- 94.03%/91.82% char-acc on
#      clean Teklia val/test AND 89.64% on held-out personal lines,
#      i.e. strictly better than either of the below on its own domain.
#   2) HF-trained (6480 clean Teklia/IAM-line images, general handwriting
#      only) -- measured 77.2%->83.7% mean char accuracy over 145 holdout
#      pages versus the original below, and much better on cursive hands
#      specifically (writer 151: 82.1% -> 95.1%).
#   3) the original, page-segmented-only checkpoint (fallback).
# This choice feeds BOTH the CTC alignment cuts and the legibility judge.
for _name in ("paper_cnn_bilstm_ctc_joint_best.pt", "paper_cnn_bilstm_ctc_hf_best.pt"):
    _candidate = SCRIPT_DIR.parent / "NOGIT" / "weights" / _name
    if _candidate.exists():
        TEXT_WEIGHTS = _candidate
        break
PROFILE_DIR = SCRIPT_DIR.parent / "NOGIT" / "StyleProfiles10"

MAX_VARIANTS_PER_CHAR = 12

# Drop lines whose CTC forced alignment scored below this percentile of the
# author's own lines -- their character boundaries (and the glyphs cut at
# them) are not trustworthy. Higher = fewer but cleaner glyphs.
ALIGN_CONF_PCT = 25

# variant fragment gates (0 disables)
MIN_PEN_LEN = 1.05
MIN_LONGEST_STROKE = 0.55


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


# ---------------------------------------------------------------------------
# Skeleton -> ordered polylines
# ---------------------------------------------------------------------------
_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def TracePolylines(skel):
    """Walks the skeleton graph: each returned polyline is a maximal chain
    between endpoints/junctions (or a loop), as [(x, y), ...]. Chains that
    meet at a junction and continue in nearly the same direction are then
    joined, so one pen stroke stays one polyline."""
    pts = set((int(y), int(x)) for y, x in zip(*np.nonzero(skel)))
    if not pts:
        return []

    def nbrs(p):
        y, x = p
        out = []
        for dy, dx in _OFFS:
            q = (y + dy, x + dx)
            if q not in pts:
                continue
            # a diagonal step is redundant when an orthogonal 2-step path
            # exists; keeping it creates fake degree-3 "staircase" nodes
            if dy and dx and ((y + dy, x) in pts or (y, x + dx) in pts):
                continue
            out.append(q)
        return out

    deg = {p: len(nbrs(p)) for p in pts}
    usedEdges = set()
    polylines = []

    def walk(start, first):
        chain = [start, first]
        usedEdges.add((start, first))
        usedEdges.add((first, start))
        cur, prev = first, start
        while deg[cur] == 2:
            nxt = [n for n in nbrs(cur) if n != prev]
            if not nxt:
                break
            n = nxt[0]
            if (cur, n) in usedEdges:
                break
            usedEdges.add((cur, n))
            usedEdges.add((n, cur))
            chain.append(n)
            prev, cur = cur, n
        return chain

    seeds = sorted([p for p in pts if deg[p] != 2]) or sorted(pts)[:1]
    for p in seeds:
        for n in nbrs(p):
            if (p, n) not in usedEdges:
                polylines.append(walk(p, n))
    # leftover pure loops
    covered = {q for ch in polylines for q in ch}
    for p in sorted(pts - covered):
        if deg.get(p, 0) == 2:
            ns = nbrs(p)
            if ns and (p, ns[0]) not in usedEdges:
                polylines.append(walk(p, ns[0]))

    chains = [[(float(x), float(y)) for (y, x) in ch] for ch in polylines
              if len(ch) >= 2]
    chains = _PruneSpurs(chains)
    return _JoinAtJunctions(chains)


def _Dir(pts, fromEnd, n=4):
    """Unit direction of a chain's end, pointing AWAY from the chain."""
    p = np.asarray(pts, np.float64)
    if fromEnd:
        a, b = p[max(0, len(p) - 1 - n)], p[-1]
    else:
        a, b = p[min(len(p) - 1, n)], p[0]
    v = b - a
    L = np.hypot(*v)
    return v / L if L > 1e-6 else np.array([1.0, 0.0])


def _PruneSpurs(chains, minLen=2.5):
    """Drop very short branches that dangle off a junction -- thinning
    artefacts, not pen strokes."""
    if len(chains) <= 1:
        return chains
    ends = {}
    for c in chains:
        for e in (tuple(c[0]), tuple(c[-1])):
            ends[e] = ends.get(e, 0) + 1
    out = []
    for c in chains:
        L = float(np.hypot(*np.diff(np.asarray(c), axis=0).T).sum())
        dangling = ends[tuple(c[0])] == 1 or ends[tuple(c[-1])] == 1
        if L < minLen and dangling and len(chains) > 1:
            continue
        out.append(c)
    return out or chains


def _JoinAtJunctions(chains, maxAngleDeg=75.0):
    """A junction where two chains continue in nearly the same direction is
    one pen stroke crossing itself (loops of e/l/o, t crossbars); join the
    best-aligned pair repeatedly."""
    chains = [list(c) for c in chains]
    changed = True
    while changed and len(chains) > 1:
        changed = False
        best = None
        for i in range(len(chains)):
            for j in range(len(chains)):
                if i == j:
                    continue
                for iEnd in (0, 1):
                    for jEnd in (0, 1):
                        pi = chains[i][-1] if iEnd else chains[i][0]
                        pj = chains[j][-1] if jEnd else chains[j][0]
                        if abs(pi[0] - pj[0]) > 1.5 or abs(pi[1] - pj[1]) > 1.5:
                            continue
                        di = _Dir(chains[i], iEnd)
                        dj = _Dir(chains[j], jEnd)
                        # continuing straight through means di ~ -dj
                        cosang = float(-np.dot(di, dj))
                        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                        if ang < maxAngleDeg and (best is None or ang < best[0]):
                            best = (ang, i, j, iEnd, jEnd)
        if best is not None:
            _, i, j, iEnd, jEnd = best
            a = chains[i] if iEnd else chains[i][::-1]
            b = chains[j] if not jEnd else chains[j][::-1]
            merged = a + b[1:] if a[-1] == b[0] else a + b
            for k in sorted((i, j), reverse=True):
                del chains[k]
            chains.append(merged)
            changed = True
    return chains


def SimplifyPolyline(poly, eps=0.6):
    """Douglas-Peucker."""
    if len(poly) < 3:
        return poly
    p = np.asarray(poly, np.float64)
    a, b = p[0], p[-1]
    ab = b - a
    L = np.hypot(*ab) + 1e-9
    rel = p - a
    d = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / L
    i = int(np.argmax(d))
    if d[i] > eps:
        return SimplifyPolyline(poly[:i + 1], eps)[:-1] + \
               SimplifyPolyline(poly[i:], eps)
    return [poly[0], poly[-1]]


# ---------------------------------------------------------------------------
# Line-level measurements
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Glyph extraction from one aligned line
# ---------------------------------------------------------------------------
def _TrimLigatureTails(strokes, connL=True, connR=True, lowY=0.42, minKeep=0.18,
                       climbY=1.6, climbSlope=2.2):
    """A cursive cut lands mid-ligature, so the glyph arrives with a
    carrier stroke hanging off each side, wherever the pen was CONFIRMED
    (via `connL`/`connR`, the same stroke-crossing test used to measure
    connectedness) to have been continuously joined to the previous/next
    letter. Trim the leading/trailing run of points that keeps moving
    monotonically outward while staying below a height ceiling -- that is
    the connector, not the letter.

    Two ceilings are tried, in order: the low, near-horizontal `lowY`
    case (a connector between two ordinary x-height letters), and a
    taller, steeper `climbY`/`climbSlope` case for a connector climbing
    into a TALL letter (e.g. "T" into "h") -- a low letter joining a tall
    one has to climb quickly, which the original single low/gentle
    threshold never matched, leaving an unmistakable stray diagonal
    baked into every affected glyph. Only attempted on a side confirmed
    connected by `connL`/`connR`: a genuine pen-lift before this letter
    means everything here is this letter's own ink, and guessing at a
    phantom connector from shape alone risks cutting a real stroke.

    Returns (trimmedStrokes, entryPoint, exitPoint)."""
    if not strokes:
        return strokes, (0.0, 0.0), (0.0, 0.0)
    allPts = [p for s in strokes for p in s]
    entry, exitPt = strokes[0][0], strokes[-1][-1]
    xs = [p[0] for p in allPts]
    span = max(xs) - min(xs)
    if span <= 1e-6:
        return strokes, entry, exitPt

    def runLen(s, connected, maxHeight, maxSlope):
        """How many leading points form a low(ish), outward-running
        carrier. A ligature keeps moving forward and never climbs faster
        than `maxSlope`; a letter's own leg does too at first, which is
        why this is only tried on a CONFIRMED join, not guessed from
        shape alone."""
        if not connected:
            return 0
        i = 0
        while i + 1 < len(s):
            dx = s[i + 1][0] - s[i][0]
            dy = s[i + 1][1] - s[i][1]
            if s[i][1] >= maxHeight or dx < 0 or abs(dy) > maxSlope * max(dx, 1e-6):
                break
            i += 1
        return i

    out = []
    for k, s in enumerate(strokes):
        t = list(s)
        if k == 0:
            i = runLen(t, connL, lowY, 0.7) or runLen(t, connL, climbY, climbSlope)
            if i and (t[-1][0] - t[i][0]) > minKeep * span:
                t = t[i:]
        if k == len(strokes) - 1 and len(t) >= 2:
            # the exit connector leaves to the right, so on the REVERSED
            # walk its x decreases -- mirror x to reuse the same test
            rev = [(-x, y) for (x, y) in t[::-1]]
            r = runLen(rev, connR, lowY, 0.7) or runLen(rev, connR, climbY, climbSlope)
            if r and (t[len(t) - 1 - r][0] - t[0][0]) > minKeep * span:
                t = t[:len(t) - r]
        if len(t) >= 2:
            out.append(t)
    if not out:
        return strokes, entry, exitPt
    return out, out[0][0], out[-1][-1]


def _SpanInk(ink, x0, x1):
    sub = np.zeros_like(ink)
    sub[:, max(0, x0):min(ink.shape[1], x1)] = ink[:, max(0, x0):min(ink.shape[1], x1)]
    return sub


def ExtractLineGlyphs(gray, text, model, device):
    """Returns (glyphInstances, lineStats) or (None, None).
    Each glyph instance: dict(char, strokes(normalized), advance, entryY,
    exitY, connL, connR, hasInk)."""
    rawW = gray.shape[1]
    pil = Image.fromarray(gray)
    stretched = resize_line_image_fixed(pil)
    t = tensor_from_resized(stretched).unsqueeze(0).to(device)
    with torch.no_grad():
        lp = model(t)[:, 0, :].cpu().numpy()      # (T, C)
    res = CtcForcedAlign(lp, text)
    if res is None:
        return None, None
    align, conf = res
    T = lp.shape[0]

    ink = BinarizeLine(gray)
    band = CoreBand(ink)
    if band is None:
        return None, None
    top, base = band
    xh = float(base - top)
    slant = EstimateSlantDeg(ink)
    shear = np.tan(np.radians(slant))


    # Territory boundaries from span CENTRES, not span edges: a CTC peak is
    # typically 1-2 frames wide and lags the ink, so raw spans clip letters
    # to slivers. Each char owns the strip halfway to each neighbour's
    # centre; the cut is then snapped to the nearest ink-profile minimum
    # (in cursive that is the thinnest point of the connecting ligature).
    # Where may a cut fall? In cursive the letters are joined, so the only
    # honest boundaries are the ligature crossings: columns where the ink
    # is THIN (one stroke thick) and sits LOW (near the baseline). Score
    # every column on those two cues plus distance from the predicted
    # boundary, and snap to the cheapest.
    # NOTE: cutting along the slant was tried (boundaries and membership
    # both computed on de-slanted x, with the recognizer's boundaries
    # converted into that frame). It is geometrically the more correct
    # thing to do and it visibly improved the tall letters of the steeply
    # slanted hands -- but it measured WORSE end to end (writer-ID 81.8%
    # -> 68.6% uncorrected / 66.6% corrected), because glyphs cut on
    # vertical boundaries carry their slant context and retile more
    # naturally at synthesis time. Kept vertical on the evidence.
    colProf = ink.sum(axis=0).astype(np.float64)
    k = max(3, int(round(xh * 0.12))) | 1
    colSm = np.convolve(colProf, np.ones(k) / k, mode='same')
    rowsIdx = np.arange(ink.shape[0])[:, None]
    hiY = np.where(ink, rowsIdx, ink.shape[0]).min(axis=0).astype(np.float64)
    # height of the highest ink in each column, measured up from baseline
    topAbove = np.clip((base - hiY) / max(1.0, xh), 0.0, 3.0)
    thin = colSm / max(1e-6, float(np.percentile(colSm[colSm > 0], 60))
                       if (colSm > 0).any() else 1.0)
    cutCost = thin + 1.15 * topAbove
    cutCost[colProf == 0] = 0.0          # a true gap is always a fine cut
    snapR = max(3, int(round(xh * 0.5)))

    def _Snap(x):
        a = int(np.clip(round(x) - snapR, 0, rawW - 1))
        b = int(np.clip(round(x) + snapR + 1, 1, rawW))
        if b - a < 2:
            return float(x)
        win = cutCost[a:b].copy()
        # mild preference for staying near the recognizer's boundary
        off = np.abs(np.arange(a, b) - x) / max(1.0, snapR)
        return float(a + int(np.argmin(win + 0.6 * off)))

    # frame index -> canvas x (t/T*INPUT_WIDTH) -> ORIGINAL pixel x. The
    # second step used to be a plain "* rawW / T", which was only correct
    # because resize_line_image_fixed used to stretch the whole line to
    # fill the canvas exactly. Now that it fits-and-pads instead (see that
    # function's docstring), the same canvas x can correspond to a
    # different original x depending on how much of the canvas is real
    # content vs padding -- frame_x_to_pixel is the actual inverse.
    origH = gray.shape[0]
    charXs = []
    for ch, sp in align:
        if sp is None:
            charXs.append((ch, None, None))
            continue
        x0 = frame_x_to_pixel(sp[0] / T * INPUT_WIDTH, rawW, origH)
        x1 = frame_x_to_pixel(sp[1] / T * INPUT_WIDTH, rawW, origH)
        charXs.append((ch, x0, x1))
    valid = [(i, c) for i, c in enumerate(charXs) if c[1] is not None]
    centres = [0.5 * (c[1] + c[2]) for _, c in valid]
    cuts = []
    for k2 in range(len(valid)):
        i, (ch, x0, x1) = valid[k2]
        c = centres[k2]
        left = (c - 0.5 * (centres[1] - centres[0]) if len(centres) > 1 else c - xh) \
            if k2 == 0 else 0.5 * (centres[k2 - 1] + c)
        right = (c + 0.5 * (centres[-1] - centres[-2]) if len(centres) > 1 else c + xh) \
            if k2 == len(valid) - 1 else 0.5 * (c + centres[k2 + 1])
        left = max(0.0, _Snap(left) if k2 > 0 else left)
        right = min(float(rawW), _Snap(right) if k2 < len(valid) - 1 else right)
        cuts.append((i, ch, left, right, x0, x1))

    glyphs = [None] * len(charXs)
    inkArea = float(ink.sum())

    # Skeletonize the WHOLE line once and cut the traced centerlines at the
    # territory boundaries. Thinning each column slice separately (the
    # obvious approach) is wrong: a stroke truncated at the slice edge
    # thins into a spurious spine along the cut, which then gets stored as
    # a horizontal bar across the glyph.
    skelAll = Skeletonize(ink)
    polysAll = TracePolylines(skelAll)
    skelLenTotal = sum(
        float(np.hypot(*(np.diff(np.asarray(p), axis=0).T)).sum())
        for p in polysAll) or 1.0

    bounds = [(int(round(c[2])), int(round(c[3]))) for c in cuts]
    segsFor = {i: [] for i in range(len(charXs))}
    crossings = set()
    for p in polysAll:
        cur, curT = [], None
        for pt in p:
            t = None
            for k2, (L, R) in enumerate(bounds):
                if L <= pt[0] < R:
                    t = cuts[k2][0]
                    break
            if t != curT:
                if curT is not None and len(cur) >= 2:
                    segsFor[curT].append(cur)
                if curT is not None and t is not None:
                    crossings.add((min(curT, t), max(curT, t)))
                cur = [pt] if t is not None else []
                curT = t
            elif t is not None:
                cur.append(pt)
        if curT is not None and len(cur) >= 2:
            segsFor[curT].append(cur)

    for (i, ch, left, right, ax0, ax1) in cuts:
        if ch == ' ':
            glyphs[i] = dict(char=' ', width=(right - left) / xh)
            continue
        polys = [s for s in segsFor[i] if len(s) >= 2]
        if not polys:
            glyphs[i] = None
            continue
        # a centerline that runs across a boundary means the pen never
        # lifted between those letters -- the honest definition of
        # "connected" for measuring how cursive a hand is
        connL = any(i in pr for pr in crossings if i - 1 in pr)
        connR = any(i in pr for pr in crossings if i + 1 in pr)

        # normalize strokes: origin at the left cut, baseline, y up, /xh,
        # then remove the author slant
        norm = []
        for p in polys:
            p = SimplifyPolyline(p, eps=0.6)
            q = []
            for (x, y) in p:
                yn = (base - y) / xh
                xn = (x - left) / xh        # local to the cut window
                xn = xn - shear * yn        # de-slant
                q.append((round(xn, 3), round(yn, 3)))
            norm.append(q)
        # sort strokes left-to-right by start, each oriented left-to-right
        norm = [(s if s[0][0] <= s[-1][0] else s[::-1]) for s in norm]
        norm.sort(key=lambda s: min(pt[0] for pt in s))
        # the cut runs through the middle of a ligature, so a cursive glyph
        # arrives with half a connector stuck on each side. Trim those to
        # the letter BODY and remember where the pen entered/left, so the
        # synthesizer draws exactly one connector of its own choosing.
        norm, entry, exitPt = _TrimLigatureTails(norm, connL=connL, connR=connR)
        if not norm:
            glyphs[i] = None
            continue
        # Align in the frame the glyph will be DRAWN in. The stored shape is
        # de-slanted (upright) so jitter/scaling behave sensibly, but the
        # synthesizer re-applies the shear -- so the origin must be chosen
        # from the RE-SLANTED x, otherwise a tall slanted letter lands up to
        # (shear x ascender) x-heights off its cell and letters collide.
        xr = [px + shear * py for s in norm for (px, py) in s]
        x0r, x1r = min(xr), max(xr)
        norm = [[(round(px - x0r, 3), py) for (px, py) in s] for s in norm]
        ysAll = [pt[1] for s in norm for pt in s]
        glyphs[i] = dict(
            char=ch, strokes=norm,
            width=round(x1r - x0r, 3),
            advance=round((right - left) / xh, 3),
            # x0r is the gap from the cell's left edge to where the ink
            # actually starts: keeping it preserves the author's own letter
            # gaps instead of butting every glyph against the previous one
            lead=round(float(np.clip(x0r, 0.0, 1.5)), 3),
            top=round(max(ysAll), 3), bot=round(min(ysAll), 3),
            entryX=round(entry[0] - x0r, 3), entryY=round(entry[1], 3),
            exitX=round(exitPt[0] - x0r, 3), exitY=round(exitPt[1], 3),
            connL=connL, connR=connR)

    stats = dict(slant=slant, xh=xh, alignConf=conf,
                 strokeW=inkArea / max(1.0, skelLenTotal),
                 # the territory each character was actually cut from, kept
                 # so the cuts can be drawn back onto the page and checked
                 cuts=[(int(i), ch, float(left), float(right))
                       for (i, ch, left, right, _ax0, _ax1) in cuts])
    return glyphs, stats


# ---------------------------------------------------------------------------
# Variant quality gates
# ---------------------------------------------------------------------------
# Typographic height class of each letter, used as a PRIOR (not tied to any
# author): a variant filed under 'l' whose ink is x-height-only is an
# alignment slip that captured the neighbouring letter, not an 'l'.
_ASCEND = set('bdfhklt') | set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
_DESCEND = set('gjpqy')
_XONLY = set('acemnorsuvwxz')


def _ClassOk(ch, top, bot):
    if ch in _XONLY:
        return top <= 1.42 and bot >= -0.42
    if ch in _ASCEND:
        return top >= 1.25
    if ch in _DESCEND:
        return bot <= -0.20
    return True


def _ResampleN(poly, n):
    p = np.asarray(poly, np.float64)
    if len(p) < 2:
        return np.zeros((n, 2))
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-9:
        return np.repeat(p[:1], n, axis=0)
    t = np.linspace(0.0, cum[-1], n)
    return np.stack([np.interp(t, cum, p[:, 0]), np.interp(t, cum, p[:, 1])], 1)


def _AlignStroke(s, N):
    """resample one stroke to N points, oriented left-to-right, loops rolled
    to start at the leftmost point."""
    s = np.asarray(s, float)
    if len(s) < 2:
        return None
    if float(np.hypot(*(s[0] - s[-1]))) < 0.30 and len(s) >= 4:
        i0 = int(np.argmin(s[:, 0]))
        s = np.concatenate([s[i0:], s[:i0 + 1]])
    elif s[0, 0] > s[-1, 0]:
        s = s[::-1]
    return _ResampleN(s, N)


def _DenoisedPrototype(variants, minN=3, N=44):
    """Robust median of an author's variants of one letter, per stroke.

    Groups variants by stroke count, takes the modal group, orders each
    variant's strokes left-to-right, and takes the trimmed median of every
    stroke across variants. Loops are rolled to a common start. Returns
    list-of-strokes (each a list of [x, y]) or None if too few consistent
    variants. Cut noise is roughly random, so this recovers the author's
    true letterform without leaving their style."""
    by_nc = {}
    for g in variants:
        if all(len(s) >= 2 for s in g['strokes']):
            by_nc.setdefault(len(g['strokes']), []).append(g)
    if not by_nc:
        return None
    nc = max(by_nc, key=lambda k: len(by_nc[k]))
    group = by_nc[nc]
    if len(group) < minN or nc > 5:
        return None
    stacks = [[] for _ in range(nc)]
    for g in group:
        ss = sorted(g['strokes'], key=lambda s: min(p[0] for p in s))
        ok = True
        for i, s in enumerate(ss):
            a = _AlignStroke(s, N)
            if a is None:
                ok = False
                break
            stacks[i].append(a)
        if not ok:
            for st in stacks:
                if st:
                    st.pop()
    proto = []
    for st in stacks:
        if len(st) < minN:
            return None
        A = np.stack(st)
        # two iterations of "align each variant to the running mean by a
        # small x-shift, re-mean" -- for a heavy cursive hand the cut lands
        # at a different point in the letter each time, so a plain median
        # smears; shifting each curve to best-match the mean first sharpens
        # the prototype for exactly the authors that need it most
        med = np.median(A, axis=0)
        for _ in range(2):
            shifted = []
            for c in A:
                best, bestd = c, 1e9
                for dx in np.linspace(-0.22, 0.22, 9):
                    cc = c.copy()
                    cc[:, 0] += dx
                    d = float(np.hypot(*(cc - med).T).mean())
                    if d < bestd:
                        bestd, best = d, cc
                shifted.append(best)
            A = np.stack(shifted)
            med = np.median(A, axis=0)
        dev = np.sqrt(((A - med) ** 2).sum(-1)).mean(1)
        A = A[dev <= np.percentile(dev, 80)]
        proto.append(np.median(A, axis=0))
    x0 = min(float(s[:, 0].min()) for s in proto)
    return [[[round(float(x - x0), 3), round(float(y), 3)] for x, y in s]
            for s in proto]


def SmoothStroke(pts, step=0.055, passes=2):
    """Take the pixel-grid staircase out of a traced centreline.

    Thinning snaps the skeleton to integer pixels, so at a 33px x-height a
    diagonal stroke is stored as a run of 0.6px zigzags (measured: median
    turn angle between segments 8.9 deg, 90th percentile 59.5 deg). That is
    an artefact of the raster, not something the writer did. Resample to an
    even spacing, then average each point with its neighbours, holding the
    endpoints fixed so the letter keeps its extent and its join points."""
    p = np.asarray(pts, np.float64)
    if len(p) < 3:
        return [[round(float(x), 3), round(float(y), 3)] for x, y in p]
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-6:
        return [[round(float(x), 3), round(float(y), 3)] for x, y in p]
    n = max(3, int(round(total / step)) + 1)
    t = np.linspace(0.0, total, n)
    q = np.stack([np.interp(t, cum, p[:, 0]), np.interp(t, cum, p[:, 1])], 1)
    for _ in range(passes):
        inner = 0.25 * q[:-2] + 0.5 * q[1:-1] + 0.25 * q[2:]
        q = np.vstack([q[:1], inner, q[-1:]])
    return [[round(float(x), 3), round(float(y), 3)] for x, y in q]


def _PenLength(g):
    """Total centerline length of a glyph, in x-heights."""
    tot = 0.0
    for s in g['strokes']:
        p = np.asarray(s, np.float64)
        if len(p) >= 2:
            tot += float(np.hypot(*np.diff(p, axis=0).T).sum())
    return tot


def _LongestStroke(g):
    best = 0.0
    for s in g['strokes']:
        p = np.asarray(s, np.float64)
        if len(p) >= 2:
            best = max(best, float(np.hypot(*np.diff(p, axis=0).T).sum()))
    return best


def _ShapeGrid(g, n=12):
    """Soft occupancy grid over the glyph's own box -- a cheap descriptor
    for comparing variants of the same character.

    Points are interpolated ALONG each stroke, not just at the polyline
    vertices: after simplification a stroke may be only a handful of
    points, and gridding those alone describes the corners rather than the
    letter, which leaves the descriptor unable to tell a clean letter from
    a fragment. The grid is then blurred so a small positional difference
    reads as similar rather than as a total mismatch, and L2-normalized so
    similarity can be measured as a cosine."""
    xs, ys = [], []
    for st in g['strokes']:
        for (x0, y0), (x1, y1) in zip(st, st[1:]):
            d = max(abs(x1 - x0), abs(y1 - y0))
            k = max(2, int(d * 24))
            for t in np.linspace(0.0, 1.0, k):
                xs.append(x0 + t * (x1 - x0))
                ys.append(y0 + t * (y1 - y0))
        if len(st) == 1:
            xs.append(st[0][0])
            ys.append(st[0][1])
    if not xs:
        return None
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    w = max(1e-3, xs.max() - xs.min())
    h = max(1e-3, ys.max() - ys.min())
    gx = np.clip(((xs - xs.min()) / w * (n - 1)).astype(int), 0, n - 1)
    gy = np.clip(((ys - ys.min()) / h * (n - 1)).astype(int), 0, n - 1)
    grid = np.zeros((n, n), np.float32)
    np.add.at(grid, (gy, gx), 1.0)
    grid = np.sqrt(grid)                       # damp long dwells
    # 3x3 blur: tolerate small shifts between two writings of one letter
    pad = np.pad(grid, 1)
    grid = sum(pad[a:a + n, b:b + n] * (0.25 if (a == 1 and b == 1) else 0.09375)
               for a in range(3) for b in range(3))
    v = grid.ravel()
    nrm = float(np.linalg.norm(v))
    return v / nrm if nrm > 1e-9 else None


# Cross-author prototype of each character, learned from every author's
# variants together. With only ~10 pages per author a single author does
# not provide enough examples to tell a well-formed letter from a fragment
# or a cut that swallowed its neighbour -- but ten authors do: whatever all
# of them share IS the letter. Used only to REJECT an author's malformed
# variants, never to replace them, so their own letterforms survive.
_LETTER_PRIOR = {}


def BuildLetterPrior(libsByAuthor, minSamples=12):
    """libsByAuthor: {author: {char: [variants]}} -> {char: prototype grid}"""
    byChar = {}
    for lib in libsByAuthor.values():
        for ch, vs in lib.items():
            byChar.setdefault(ch, []).extend(vs)
    prior = {}
    for ch, vs in byChar.items():
        grids = [g for g in (_ShapeGrid(v) for v in vs) if g is not None]
        if len(grids) < minSamples:
            continue
        G = np.vstack(grids)
        proto = np.median(G, axis=0)          # robust to fragments in the pool
        nrm = float(np.linalg.norm(proto))
        if nrm > 1e-9:
            prior[ch] = proto / nrm
    return prior


def _PriorScore(g, ch, prior):
    """How far this variant sits from the cross-author idea of the letter."""
    proto = prior.get(ch)
    grid = _ShapeGrid(g)
    if proto is None or grid is None:
        return 0.0
    # both unit-norm, so this is a cosine distance in [0, 1]
    return float(1.0 - np.dot(grid, proto))


def PriorFilter(variants, ch, prior, keepFrac=0.7, minKeep=3):
    """Drop the variants least like the letter, keeping the author's own
    spread of well-formed ones."""
    if not prior or ch not in prior or len(variants) <= minKeep:
        return variants
    scored = sorted(variants, key=lambda g: _PriorScore(g, ch, prior))
    k = max(minKeep, int(round(keepFrac * len(scored))))
    return scored[:k]


def _ConsensusFilter(variants, keepFrac=0.75):
    """Drop variants whose shape disagrees with the character's medoid --
    this is what removes glyphs contaminated by a neighbouring letter."""
    if len(variants) <= 3:
        return variants
    grids = [_ShapeGrid(g) for g in variants]
    ok = [(g, d) for g, d in zip(variants, grids) if d is not None]
    if len(ok) <= 3:
        return [g for g, _ in ok] or variants
    G = np.stack([d for _, d in ok])
    D = np.abs(G[:, None, :] - G[None, :, :]).sum(-1)
    medoid = int(np.argmin(D.sum(1)))
    dist = D[medoid]
    order = np.argsort(dist)
    n = max(3, int(round(len(ok) * keepFrac)))
    return [ok[i][0] for i in order[:n]]


# ---------------------------------------------------------------------------
# Author profile aggregation
# ---------------------------------------------------------------------------
def RenderRefStats(gray):
    """How the author's REAL line looks once put through the classifier's
    own preprocessing (aspect-squashed to INPUT_H x INPUT_W). Stroke weight
    and ink density are part of a hand's style, and the writer-ID model is
    sensitive to them, so synthesis has to reproduce these -- not just the
    letter shapes."""
    from TrainText import resize_line_image_fixed
    pil = Image.fromarray(gray).convert('L')
    a = np.array(resize_line_image_fixed(pil), dtype=np.float32)
    # Paper is not white and ink is not black on a real scan: the crops the
    # writer-ID model was trained on sit on light-grey paper with soft,
    # anti-aliased stroke edges. Rendering pure black on pure white is a
    # domain shift the model reacts to, so these levels get reproduced.
    bright = a[a >= 200]
    dark = a[a < 128]
    out = dict(inkFrac=float((a < 128).mean()),
               aspect=float(pil.width) / max(1.0, pil.height),
               slantMeas=np.nan, ascMeas=np.nan, descMeas=np.nan,
               paperLevel=float(np.median(bright)) if bright.size else 255.0,
               inkLevel=float(np.median(dark)) if dark.size else 40.0,
               midFrac=float(((a >= 128) & (a < 200)).mean()),
               wordGap=np.nan, heightXh=np.nan)
    # Word spacing measured from the real ink gaps rather than from the
    # recognizer's territory for the space character: CTC gives a space a
    # narrow slice (there is nothing to see there), which badly
    # underestimates how far apart the author sets their words.
    ink = BinarizeLine(gray)
    band = CoreBand(ink)
    if band is not None and ink.any():
        top, base = band
        xh = float(base - top)
        if xh >= 4:
            col = ink.any(axis=0)
            gaps, run = [], 0
            for v in col:
                if v:
                    if run:
                        gaps.append(run)
                    run = 0
                else:
                    run += 1
            wide = [g / xh for g in gaps if g / xh > 0.5]
            if wide:
                out['wordGap'] = float(np.median(wide))
            ys = np.nonzero(ink.any(axis=1))[0]
            out['heightXh'] = float(ys.max() - ys.min() + 1) / xh
            # line-level slant and ascender/descender reach, measured the
            # same way they will be measured on the synthesized render so
            # the two are directly comparable and can be calibrated
            out['slantMeas'] = float(EstimateSlantDeg(ink))
            out['ascMeas'] = float(base - ys.min()) / xh
            out['descMeas'] = float(base - ys.max()) / xh
    return out


def ExtractAuthorRaw(authorId, lineItems, model, device):
    """Runs the expensive per-line extraction once and returns the raw
    result, so profile FILTERING can be re-tuned without re-skeletonizing
    every page. Cached on disk by BuildAll."""
    parsed = []
    for gray, text in lineItems:
        glyphs, stats = ExtractLineGlyphs(gray, text, model, device)
        if glyphs is not None:
            stats = dict(stats)
            stats['ref'] = RenderRefStats(gray)
            parsed.append((glyphs, stats))
    return parsed


def _MedianWordGap(refs, spaceAdvs):
    vals = [r['wordGap'] for r in (refs or [])
            if r.get('wordGap') == r.get('wordGap')]     # drop NaN
    if vals:
        return float(np.median(vals))
    return float(np.median(spaceAdvs)) if spaceAdvs else 1.2


def BuildAuthorProfile(authorId, parsed, refs=None, prior=None,
                       verbose=True):
    """parsed: output of ExtractAuthorRaw. refs: optional list of
    RenderRefStats dicts for this author's real lines."""
    lib = {}
    slants, xhs, strokeWs = [], [], []
    ascRatios, descRatios = [], []
    advances = {}
    spaceAdvs = []
    connPairs, adjPairs = 0, 0

    if not parsed:
        return None
    confs = [st['alignConf'] for _, st in parsed]
    # only trust char boundaries from lines the recognizer actually read:
    # a badly-aligned line yields glyphs cut at the wrong places
    confCut = (float(np.percentile(confs, ALIGN_CONF_PCT))
               if len(confs) >= 8 else -1e9)

    for glyphs, stats in parsed:
        if stats['alignConf'] < confCut:
            continue
        slants.append(stats['slant'])
        xhs.append(stats['xh'])
        strokeWs.append(stats['strokeW'] / max(1.0, stats['xh']))
        for k, g in enumerate(glyphs):
            if g is None:
                continue
            ch = g['char']
            if ch == ' ':
                spaceAdvs.append(g['width'])
                continue
            if ch.islower():
                if g['top'] > 1.35:
                    ascRatios.append(g['top'])
                if g['bot'] < -0.25:
                    descRatios.append(g['bot'])
            advances.setdefault(ch, []).append(g['advance'])
            lib.setdefault(ch, []).append(g)
            # connectedness between adjacent in-word letters
            if k + 1 < len(glyphs) and glyphs[k + 1] is not None and \
                    glyphs[k + 1].get('char', ' ') != ' ' and 'connR' in g:
                adjPairs += 1
                if g['connR']:
                    connPairs += 1

    if not xhs:
        return None

    # prune each char's variants: drop degenerate/outlier instances, keep
    # the most typical MAX_VARIANTS (nearest to the median width/height)
    lib2 = {}
    for ch, insts in lib.items():
        good = [g for g in insts
                if 0.05 < g['width'] < 6.0 and (g['top'] - g['bot']) > 0.25
                and len(g['strokes']) <= 6
                and _ClassOk(ch, g['top'], g['bot'])
                # ink much wider than the cell the recognizer gave it means
                # the cut swallowed part of a neighbouring letter
                and g['lead'] + g['width'] <= 1.45 * g['advance'] + 0.25
                # a letter needs a letter's worth of pen path: a light hand
                # whose thin strokes broke up in the scan leaves fragments
                # that pass every shape test but are not the letter
                and _PenLength(g) >= MIN_PEN_LEN
                and _LongestStroke(g) >= MIN_LONGEST_STROKE]
        if not good:
            continue
        # anchor on a LOW percentile: if many variants are doubled letters
        # the median is inflated and would bless them
        wRef = float(np.percentile([g['width'] for g in good], 40))
        hMed = float(np.median([g['top'] - g['bot'] for g in good]))
        tight = [g for g in good
                 if 0.6 * wRef <= g['width'] <= 1.5 * wRef
                 and 0.55 * hMed <= (g['top'] - g['bot']) <= 1.7 * hMed]
        good = tight if len(tight) >= 3 else good
        # reject this author's malformed variants using what all ten
        # authors agree the letter looks like, THEN fall back to their
        # own internal consensus
        good = PriorFilter(good, ch, prior)
        for g in good:
            # how far this prototype sits from the cross-author idea of the
            # letter; synthesis uses it to decide whether the author's own
            # shape is trustworthy or the cursive baseline should stand in
            g['priorD'] = round(_PriorScore(g, ch, prior), 4)
        good = _ConsensusFilter(good)
        good.sort(key=lambda g: abs(g['width'] - wRef) / max(0.2, wRef) +
                  abs((g['top'] - g['bot']) - hMed) / max(0.2, hMed))
        good = good[:MAX_VARIANTS_PER_CHAR]
        # the same de-staircasing on the variants themselves -- at a low
        # anchor level these ARE the output, so the raster zigzag would be
        # drawn as if the writer's hand shook
        for g in good:
            g['strokes'] = [SmoothStroke(s) for s in g['strokes']]
        lib2[ch] = good

    # denoised prototype: the robust per-stroke median of an author's own
    # aligned variants of a letter cancels the ~random cut noise and
    # recovers their true letterform. Prepended (so synthesis prefers it)
    # only where it measurably beats the raw variants on priorD.
    # The author's own anchor, built two ways so synthesis can choose:
    #   idealMedoid -- the single REAL variant most typical of their own set.
    #                  An actual thing they wrote, so a multi-stroke letter
    #                  keeps its structure intact.
    #   idealGlyphs -- the per-stroke median of their variants. Smoother,
    #                  but averaging can blur a letter whose variants differ
    #                  in how the strokes are arranged.
    medoid = {}
    for ch, vs in lib2.items():
        grids = [(_ShapeGrid(g), g) for g in vs]
        grids = [(v, g) for v, g in grids if v is not None]
        if len(grids) < 2:
            continue
        G = np.stack([v for v, _ in grids])
        D = np.abs(G[:, None, :] - G[None, :, :]).sum(-1)
        med = grids[int(np.argmin(D.sum(1)))][1]
        medoid[ch] = dict(
            strokes=[SmoothStroke(s) for s in med['strokes']],
            priorD=round(float(med.get('priorD', 0.5)), 4),
            advance=round(float(med.get('advance', 0.9)), 3),
            lead=round(float(med.get('lead', 0.0)), 3),
            width=round(float(med.get('width', 0.9)), 3),
            top=round(float(med.get('top', 1.0)), 3),
            bot=round(float(med.get('bot', 0.0)), 3),
            entryY=round(float(med.get('entryY', 0.3)), 3),
            exitY=round(float(med.get('exitY', 0.3)), 3),
            nPool=len(vs))

    ideal = {}
    for ch, vs in lib2.items():
        proto = _DenoisedPrototype(vs)
        if proto is None:
            continue
        pd = _PriorScore({'strokes': proto}, ch, prior)
        rawPd = float(np.median([g.get('priorD', 0.5) for g in vs]))
        pts = [p for s in proto for p in s]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # THE AUTHOR'S OWN IDEAL LETTERFORM. The same robust median, but kept
        # as this writer's personal anchor rather than as one more variant:
        # synthesis can then clean a letter up toward how THEY form it
        # instead of toward a generic alphabet, which is what stops every
        # author's substituted letters coming out as the same shape.
        # `priorD` is recorded so synthesis can tell whether this ideal is
        # actually a recognisable letter -- averaging twelve malformed cuts
        # gives a tidy shape that is still the wrong letter, and those have
        # to fall through to the generic anchor.
        ideal[ch] = dict(
            strokes=[SmoothStroke(s) for s in proto], priorD=round(pd, 4),
            advance=round(float(np.median([g['advance'] for g in vs])), 3),
            lead=round(float(np.median([g.get('lead', 0.0) for g in vs])), 3),
            width=round(max(xs) - min(xs), 3),
            top=round(max(ys), 3), bot=round(min(ys), 3),
            entryY=round(float(np.median([g.get('entryY', 0.3) for g in vs])), 3),
            exitY=round(float(np.median([g.get('exitY', 0.3) for g in vs])), 3),
            nPool=len(vs))
        if pd < rawPd - 0.02 and pd < 0.32:
            g0 = dict(vs[0])
            g0.update(strokes=proto, priorD=round(pd, 4), denoised=True,
                      width=round(max(xs) - min(xs), 3),
                      top=round(max(ys), 3), bot=round(min(ys), 3))
            lib2[ch] = [g0] + vs

    refs = refs or [st['ref'] for _, st in parsed if 'ref' in st]
    prof = dict(
        authorId=authorId,
        # the author's own ink density / line proportions as the classifier
        # sees them -- synthesis calibrates its stroke weight to match
        inkFracRef=round(float(np.median([r['inkFrac'] for r in refs])), 5)
        if refs else None,
        aspectRef=round(float(np.median([r['aspect'] for r in refs])), 2)
        if refs else None,
        paperLevel=round(float(np.median([r['paperLevel'] for r in refs])), 1)
        if refs else 255.0,
        inkLevel=round(float(np.median([r['inkLevel'] for r in refs])), 1)
        if refs else 40.0,
        midFracRef=round(float(np.median([r['midFrac'] for r in refs])), 4)
        if refs else 0.0,
        slantMeasRef=round(float(np.nanmedian([r['slantMeas'] for r in refs])), 3)
        if refs else None,
        ascMeasRef=round(float(np.nanmedian([r['ascMeas'] for r in refs])), 3)
        if refs else None,
        descMeasRef=round(float(np.nanmedian([r['descMeas'] for r in refs])), 3)
        if refs else None,

        slantDeg=round(float(np.median(slants)), 2),
        xHeightPx=round(float(np.median(xhs)), 1),
        strokeWidthXh=round(float(np.median(strokeWs)), 3),
        ascender=round(float(np.median(ascRatios)) if ascRatios else 1.7, 2),
        descender=round(float(np.median(descRatios)) if descRatios else -0.6, 2),
        wordSpaceXh=round(_MedianWordGap(refs, spaceAdvs), 2),
        connectedness=round(connPairs / max(1, adjPairs), 3),
        letterAdvance={ch: round(float(np.median(v)), 3)
                       for ch, v in advances.items()},
        nLines=len(xhs),
        glyphs=lib2,
        idealGlyphs=ideal,
        idealMedoid=medoid,
    )
    if verbose:
        nVar = sum(len(v) for v in lib2.values())
        print(f"  {authorId}: {len(xhs)} lines, {len(lib2)} chars, "
              f"{nVar} variants | slant {prof['slantDeg']:+.1f} deg, "
              f"conn {prof['connectedness']:.2f}, "
              f"asc {prof['ascender']:.2f}, desc {prof['descender']:.2f}")
    return prof


# ---------------------------------------------------------------------------
# Driver: build all 10 profiles from non-holdout pages
# ---------------------------------------------------------------------------
def LoadProfile(authorId):
    p = PROFILE_DIR / f"{authorId}.json"
    with open(p, encoding='utf-8') as f:
        return json.load(f)


RAW_DIR = SCRIPT_DIR.parent / "NOGIT" / "GlyphCache10"
LEGIBLE_CORE_PATH = SCRIPT_DIR.parent / "NOGIT" / "LegibleCore10.json"


def BuildLegibleCore(rawLibs, prior, keepFrac=0.35, minPool=8):
    """DISCARDED APPROACH, kept for the record. The medoid of every author's
    best-formed real variants per character reads back at only ~76% char /
    ~39% word through the frozen recognizer -- the consensus of a messy
    cursive letter is still messy. SynthesizeHandwriting uses the hand-drawn
    single-stroke print font `_FB` as its legibility anchor instead (~96% /
    ~84%). Not called by BuildAll; run by hand if you want to revisit it.

    One clean, style-neutral letterform per character: the medoid of the
    best-formed variants pooled across ALL ten authors.

    This is what synthesis blends an author's own (sometimes malformed)
    letterform toward when legibility must be guaranteed. It is a REAL
    handwritten shape -- an actual variant, not an average, so multi-stroke
    letters stay intact -- chosen to be the one most typical of what every
    hand agrees the letter looks like. Stored in the same normalized frame
    as a glyph variant (origin at the left of the ink, baseline y=0, y up,
    1.0 = x-height), so `SynthesizeHandwriting._BlendGlyph` can interpolate
    toward it directly."""
    byChar = {}
    for lib in rawLibs.values():
        for ch, vs in lib.items():
            byChar.setdefault(ch, []).extend(vs)

    core = {}
    for ch, vs in byChar.items():
        good = [g for g in vs
                if 'strokes' in g and 0.05 < g.get('width', 0) < 3.0
                and (g['top'] - g['bot']) > 0.25 and len(g['strokes']) <= 5
                and _ClassOk(ch, g['top'], g['bot'])
                and _PenLength(g) >= MIN_PEN_LEN
                and _LongestStroke(g) >= MIN_LONGEST_STROKE]
        if len(good) < minPool:
            continue
        good.sort(key=lambda g: _PriorScore(g, ch, prior))
        pool = good[:max(minPool, int(round(keepFrac * len(good))))]
        grids = [(_ShapeGrid(g), g) for g in pool]
        grids = [(v, g) for v, g in grids if v is not None]
        if len(grids) < 3:
            continue
        G = np.stack([v for v, _ in grids])
        D = np.abs(G[:, None, :] - G[None, :, :]).sum(-1)
        med = grids[int(np.argmin(D.sum(1)))][1]
        adv = float(np.median([g['advance'] for g in pool]))
        core[ch] = dict(
            strokes=[[[round(x, 3), round(y, 3)] for (x, y) in s]
                     for s in med['strokes']],
            advance=round(adv, 3),
            lead=round(float(med.get('lead', 0.0)), 3),
            width=round(float(med.get('width', adv)), 3),
            entryY=round(float(med.get('entryY', 0.3)), 3),
            exitY=round(float(med.get('exitY', 0.3)), 3),
            nPool=len(pool))
    LEGIBLE_CORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEGIBLE_CORE_PATH, 'w', encoding='utf-8') as f:
        json.dump(core, f)
    print("  legible core: %d characters (%s)"
          % (len(core), ''.join(sorted(core))))
    return core


def BuildAll(maxLinesPerAuthor=None, useCache=True):
    """Extraction is expensive and filtering is cheap, so the raw per-line
    extraction is cached: re-running only re-filters unless the cache is
    missing or `useCache` is off."""
    import pickle

    base = IAMLineDatasetRaw(root_dir=str(DATA_DIR), cache_dir=str(CACHE_DIR))
    byAuthor = {}
    for s in base.samples:
        if s['is_holdout']:
            continue                      # profiles never see holdout text
        a = s['page_key'].split('/')[0]
        byAuthor.setdefault(a, []).append(s)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    model = device = None
    allParsed = {}

    for a in sorted(byAuthor):
        rawPath = RAW_DIR / f"{a}.pkl"
        parsed = None
        if useCache and rawPath.exists():
            with open(rawPath, 'rb') as f:
                parsed = pickle.load(f)
        if parsed is None:
            if model is None:
                device = torch.device('cuda' if torch.cuda.is_available()
                                      else 'cpu')
                model = PaperCRNN(num_classes=len(CHARSET) + 1).to(device)
                sd = torch.load(TEXT_WEIGHTS, map_location=device,
                                weights_only=False)
                if 'model_state_dict' in sd:
                    sd = sd['model_state_dict']
                model.load_state_dict(sd)
                model.eval()
            items = []
            for s in byAuthor[a][:maxLinesPerAuthor]:
                gray = np.array(_decode_png(s['image_png']).convert('L'))
                items.append((gray, s['text']))
            parsed = ExtractAuthorRaw(a, items, model, device)
            with open(rawPath, 'wb') as f:
                pickle.dump(parsed, f)
        # reference render stats come straight from the real line images,
        # so they cost nothing and never depend on the extraction cache
        refs = [RenderRefStats(np.array(_decode_png(s['image_png'])
                                        .convert('L')))
                for s in byAuthor[a][:maxLinesPerAuthor]]
        allParsed[a] = (parsed, refs)

    # Pass 1 gathered every author's raw variants. Pool them to learn what
    # each LETTER looks like across all ten hands: one author's ~10 pages
    # cannot separate a clean letter from a fragment or a bad cut, but ten
    # authors together can. The prior only REJECTS an author's own worst
    # variants -- it never substitutes another hand's letterform.
    rawLibs = {}
    for a2, (parsed2, _r) in allParsed.items():
        lib = {}
        for glyphs, _st in parsed2:
            for g in glyphs:
                if g and g.get('char', ' ') != ' ' and 'strokes' in g:
                    lib.setdefault(g['char'], []).append(g)
        rawLibs[a2] = lib
    prior = BuildLetterPrior(rawLibs)
    print('  letter prior learned for %d characters' % len(prior))

    for a2 in sorted(allParsed):
        parsed2, refs2 = allParsed[a2]
        prof = BuildAuthorProfile(a2, parsed2, refs=refs2, prior=prior)
        if prof is None:
            print('  %s: FAILED (no usable lines)' % a2)
            continue
        with open(PROFILE_DIR / ('%s.json' % a2), 'w', encoding='utf-8') as f:
            json.dump(prof, f)
    print(f"\nProfiles saved to {PROFILE_DIR}")


if __name__ == '__main__':
    BuildAll()
