"""Boundary-technique comparison on one page: same cleaning and line centres as SegmentLean.py, several
ways of finding the rough boundaries between lines. One overlay per technique; every piece of ink that a
boundary cuts through is painted magenta.

Usage: python SegmentVariants.py page.jpg outdir
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _box(a, r):
    """Mean over a (2r+1)x(2r+1) window using an integral image."""
    p = np.pad(a, r, mode="edge").astype(np.float64)
    ii = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    k = 2 * r + 1
    return (ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]) / (k * k)


def _any(m, r, axis):
    """True where any pixel within r along `axis` is set."""
    a = np.moveaxis(m.astype(np.int32), axis, 0)
    n = a.shape[0]
    c = np.cumsum(np.pad(a, [(r + 1, r)] + [(0, 0)] * (a.ndim - 1)), 0)
    return np.moveaxis((c[2 * r + 1:2 * r + 1 + n] - c[:n]) > 0, 0, axis)


def _long_runs(m, n, axis):
    """Pixels lying on a run of >= n consecutive set pixels along `axis`."""
    a = np.moveaxis(m.astype(np.int32), axis, 0)
    L = a.shape[0]
    if L < n:
        return np.zeros(m.shape, bool)
    c = np.cumsum(np.pad(a, [(1, 0)] + [(0, 0)] * (a.ndim - 1)), 0)
    hit = ((c[n:] - c[:-n]) == n).astype(np.int32)                    # run starts at i
    ch = np.cumsum(np.pad(hit, [(1, 0)] + [(0, 0)] * (a.ndim - 1)), 0)
    j = np.arange(L)
    lo, hi = np.maximum(0, j - n + 1), np.minimum(j, L - n)
    cov = ((ch[hi + 1] - ch[lo]) > 0) & (hi >= lo)[:, None] if a.ndim == 2 else None
    return np.moveaxis(cov, 0, axis)


def _otsu(v):
    h, _ = np.histogram(np.clip(v, 0, 1.2), bins=256, range=(0, 1.2))
    p = h / h.sum()
    w, m = np.cumsum(p), np.cumsum(p * np.arange(256))
    s = (m[-1] * w - m) ** 2 / np.maximum(w * (1 - w), 1e-12)
    return (int(np.argmax(s)) + 0.5) * 1.2 / 256


def _ink(gray):
    ratio = gray / np.maximum(_box(gray, 40), 1.0)
    mask = ratio < _otsu(ratio)
    solid = _box(mask.astype(np.float64), 4) > 0.999                 # 9x9 solid ink: binder interior, shadow
    return mask & ~(_box(solid.astype(np.float64), 3) > 0), ratio


def _pitch(mask):
    """Line spacing P from the autocorrelation of the row-ink profile; also the smoothed profile."""
    W = mask.shape[1]
    prof = mask[:, int(0.05 * W):int(0.97 * W)].sum(1).astype(float)
    k = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2)
    sm = np.convolve(prof, k / k.sum(), "same")
    a = sm - sm.mean()
    ac = np.correlate(a, a, "full")[len(a) - 1:]
    seg = ac[15:min(200, len(ac) // 3)]
    P = 15 + next(i for i in range(1, len(seg) - 1)
                  if seg[i] >= 0.7 * seg.max() and seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1])
    return P, sm


def _seam(cost, step=3):
    """Left-to-right minimum-cost path (row index per column), up to `step` rows per column."""
    h, W = cost.shape
    acc = cost.copy()
    back = np.zeros((h, W), np.int8)
    idx = np.arange(h)
    shifts = range(-step, step + 1)
    for x in range(1, W):
        p = acc[:, x - 1]
        cand = np.full((len(shifts), h), np.inf)
        for i, d in enumerate(shifts):
            if d < 0:
                cand[i, -d:] = p[:d] + 0.02 * abs(d)
            elif d > 0:
                cand[i, :-d] = p[d:] + 0.02 * d
            else:
                cand[i] = p
        j = cand.argmin(0)
        acc[:, x] += cand[j, idx]
        back[:, x] = j - step
    y = np.empty(W, int)
    y[-1] = acc[:, -1].argmin()
    for x in range(W - 1, 0, -1):
        y[x - 1] = y[x] + back[y[x], x]
    return y


def _label(mask):
    """8-connected component labelling (run based union-find) -> (labels, count)."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    parent = [0]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    prev = []
    for y in range(H):
        d = np.diff(np.r_[0, mask[y].view(np.int8), 0])
        cur, k = [], 0
        for a, b in zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()):
            while k < len(prev) and prev[k][1] < a:
                k += 1
            l, kk = 0, k
            while kk < len(prev) and prev[kk][0] <= b:
                r = find(prev[kk][2])
                if l == 0:
                    l = r
                elif r != l:
                    parent[max(r, l)] = min(r, l)
                    l = min(r, l)
                kk += 1
            if l == 0:
                parent.append(len(parent))
                l = len(parent) - 1
            cur.append((a, b, l))
            lab[y, a:b] = l
        prev = cur
    roots = np.array([find(i) for i in range(len(parent))])
    _, comp = np.unique(roots, return_inverse=True)
    return comp[lab].astype(np.int32), int(comp.max())


def _paper(g, P):
    """Paper region of a photo: the largest bright area after the handwriting is blurred away, shrunk a little
    from its edge. Drops the clamp, the table, the page edge and its shadow. A photo with no dark surround
    (nothing clearly darker than the paper) keeps everything."""
    H, W = g.shape
    s = 4
    c = _box(g[:H // s * s, :W // s * s].reshape(H // s, s, W // s, s).mean((1, 3)), max(1, int(0.15 * P / s)))
    dark = c < 0.45 * np.median(c)                       # clearly darker than the typical paper
    if dark.mean() < 0.002:
        return np.ones((H, W), bool)
    lab, n = _label(~dark)
    big = np.kron(lab == np.bincount(lab.ravel())[1:].argmax() + 1, np.ones((s, s), bool))
    full = np.ones((H, W), bool)
    full[:big.shape[0], :big.shape[1]] = big
    r = int(0.4 * P)
    return ~_any(_any(~full, r, 0), r, 1)


def SegmentVariants(path, out):
    t0 = time.time()
    full = Image.open(path).convert("L")
    half = full.resize((full.width // 2, full.height // 2), Image.BOX)

    # 1. deskew by the sharpest row profile
    mask, _ = _ink(np.asarray(half, dtype=np.float64))
    q = Image.fromarray(mask.astype(np.uint8) * 255).resize((half.width // 2, half.height // 2))
    best = max(np.arange(-4, 4.01, 0.5),
               key=lambda a: float((np.asarray(q.rotate(a)).sum(1).astype(float) ** 2).sum()))
    full = full.rotate(best, Image.BILINEAR, fillcolor=255)
    half = half.rotate(best, Image.BILINEAR, fillcolor=255)
    g = np.asarray(half, dtype=np.float64)
    mask, ratio = _ink(g)
    H, W = mask.shape

    # 2. long thin straight strokes are not handwriting: underlines, ruled-off lines, rules, binder edge
    P, _ = _pitch(mask)
    paper = _paper(g, P)
    paperRemoved = int((mask & ~paper).sum())
    mask &= paper
    flat = _any(_long_runs(_any(mask, 1, 0), int(1.25 * P), 1), 1, 0) & mask          # tolerant of a slight tilt
    upright = _any(_long_runs(_any(mask, 1, 1), int(1.2 * P), 0), 1, 1) & mask
    removed = int((flat | upright).sum())
    mask = mask & ~flat & ~upright

    # 2b. page-boundary artifacts: pieces on the left border, thin slivers on the right border or tall and thin, far left of the text
    rc = max(1, int(0.1 * P))
    dil = _any(_any(mask, rc, 0), rc, 1)
    closed = ~_any(_any(~dil, rc, 0), rc, 1)             # tiny pen-lift gaps joined
    lab, nc = _label(closed)
    ys, xs = np.nonzero(mask)
    pl = lab[ys, xs]
    area = np.bincount(pl, minlength=nc + 1).astype(float)
    top, bot = np.full(nc + 1, H), np.zeros(nc + 1, int)
    xmin, xmax = np.full(nc + 1, W), np.zeros(nc + 1, int)
    np.minimum.at(top, pl, ys)
    np.maximum.at(bot, pl, ys)
    np.minimum.at(xmin, pl, xs)
    np.maximum.at(xmax, pl, xs)
    tall = (bot - top + 1) > 2.2 * P
    thin = (xmax - xmin + 1) <= 0.5 * P
    touchL, touchR = xmin <= 2, xmax >= W - 3
    letters = (area >= 0.065 * P * P) & ~tall
    xL = np.percentile(xmin[letters], 5)
    vbar = ((xmax - xmin + 1) <= 0.12 * P) & ((bot - top + 1) >= 1.2 * P)        # thin and taller than any letter
    drop = touchL | (touchR & thin) | (tall & thin) | vbar | (xmax < xL - 1.2 * P)     # edge, rule and fold slivers
    drop[0] = False
    nArtifacts = int(drop.sum())
    mask &= ~drop[lab]

    # 3. line spacing and centres
    P, sm = _pitch(mask)
    peaks = [i for i in range(1, H - 1) if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1] and sm[i] > 0.06 * sm.max()]
    centres = []
    for c in sorted(peaks, key=lambda i: -sm[i]):
        if all(abs(c - o) >= 0.6 * P for o in centres):
            centres.append(c)
    centres.sort()


    # --- boundary techniques -----------------------------------------------------------------------------
    plain = _box(mask.astype(np.float64), 1) > 0

    def smear(mk, down, up):                                   # extend ink downward (dots) / upward (tails)
        r = mk.copy()
        for d in range(1, down + 1):
            r[d:] |= mk[:-d]
        for d in range(1, up + 1):
            r[:-d] |= mk[d:]
        return _box(r.astype(np.float64), 1) > 0

    down, up = int(0.25 * P), int(0.2 * P)
    variants = {
        "v0_baseline": (plain, 4.0, 3),
        "v1_dots_down": (smear(mask, down, 0), 4.0, 3),
        "v2_tails_up": (smear(mask, 0, up), 4.0, 3),
        "v3_both": (smear(mask, down, up), 4.0, 3),
        "v4_strong_ink": (plain, 12.0, 4),
        "v5_both_strong": (smear(mask, down, up), 12.0, 4),
    }
    ys, xs = np.nonzero(mask)
    pl = lab[ys, xs]
    Y = np.arange(H)[:, None]
    base = Image.fromarray(np.clip(ratio * 255, 0, 255).astype(np.uint8)).convert("RGB")
    out.mkdir(parents=True, exist_ok=True)
    print("%-16s %8s %12s %14s %8s" % ("variant", "seconds", "ink on path", "cut pieces", "cut px"))
    for name, (cost_ink, weight, step) in variants.items():
        t1 = time.time()
        cuts = [np.full(W, max(0, centres[0] - int(0.75 * P)))]
        for c0, c1 in zip(centres, centres[1:]):
            yy = np.arange(c0, c1 + 1)[:, None]
            mid, half_w = (c0 + c1) / 2.0, max(1.0, (c1 - c0) / 2.0)
            cuts.append(c0 + _seam(weight * cost_ink[c0:c1 + 1].astype(np.float64) + 0.05 * ((yy - mid) / half_w) ** 2, step))
        cuts.append(np.full(W, min(H - 1, centres[-1] + int(0.75 * P))))
        L = len(cuts) - 1
        region = sum((Y >= cuts[n][None, :]).astype(np.int16) for n in range(1, L))
        rp = region[ys, xs]
        cnt = np.bincount(pl * L + rp, minlength=(lab.max() + 1) * L).reshape(-1, L)
        srt = np.sort(cnt, 1)
        cutPiece = srt[:, -2] >= 8                              # a piece with >= 8 px on a second line's side
        onPath = sum(int(mask[np.clip(c, 0, H - 1), np.arange(W)].sum()) for c in cuts[1:-1])
        img = np.asarray(base).copy()
        bad = cutPiece[pl]
        img[ys[bad], xs[bad]] = (255, 0, 255)                   # ink of pieces that the boundaries cut through
        ov = Image.fromarray(img)
        d = ImageDraw.Draw(ov)
        for n, cu in enumerate(cuts):
            d.line(list(zip(range(W), cu.tolist())), fill=(255, 0, 0) if n % 2 else (0, 140, 255), width=1)
        ov.save(out / (name + "_overlay.png"))
        print("%-16s %8.1f %12d %14d %8d" % (name, time.time() - t1, onPath, int(cutPiece.sum()), int((srt[:, -2][cutPiece]).sum())))


if __name__ == "__main__":
    SegmentVariants(sys.argv[1], Path(sys.argv[2]))
