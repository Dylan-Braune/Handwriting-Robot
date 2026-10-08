"""Simple, fast text-line separation for a photographed handwritten page.

1. normalise lighting (divide by a blurred background) and threshold ink (Otsu);
   drop big solid blobs (binder, shadows) and full-height bars
2. deskew by maximising the sharpness of the row-ink profile
3. find line centres as peaks of the row-ink profile (pitch from autocorrelation)
4. between neighbouring centres, find the path that crosses the least ink
   (dynamic programming, left to right): a rough boundary between two lines
5. label connected pieces of ink (strokes, dots, tails) and give each piece, whole,
   to the line that holds most of it; a small dot goes to the letter just below it
6. each line = its own ink pieces: crop it, white out everything else

Usage: python SegmentSimple.py page.jpg outdir
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


def _otsu(v):
    h, _ = np.histogram(np.clip(v, 0, 1.2), bins=256, range=(0, 1.2))
    p = h / h.sum()
    w, m = np.cumsum(p), np.cumsum(p * np.arange(256))
    s = (m[-1] * w - m) ** 2 / np.maximum(w * (1 - w), 1e-12)
    return (int(np.argmax(s)) + 0.5) * 1.2 / 256


def _ink(gray):
    """gray float image -> (ink mask, lighting-normalised image in 0..1)."""
    ratio = gray / np.maximum(_box(gray, 40), 1.0)
    mask = ratio < _otsu(ratio)
    solid = _box(mask.astype(np.float64), 4) > 0.999          # 9x9 solid ink: binder, shadow, patch
    mask &= ~(_box(solid.astype(np.float64), 3) > 0)
    bars = _box(mask.mean(0, keepdims=True).repeat(3, 0), 3)[0] > 0.4    # full-height bars
    mask[:, bars] = False
    return mask, ratio


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
        for i, d in enumerate(shifts):                 # cand[i, y] = p[y + d] + penalty
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
        runs = list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))
        cur, k = [], 0
        for a, b in runs:
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


def SegmentLines(path):
    t0 = time.time()
    full = Image.open(path).convert("L")
    half = full.resize((full.width // 2, full.height // 2), Image.BOX)

    # deskew: pick the angle whose row profile is sharpest
    mask, _ = _ink(np.asarray(half, dtype=np.float64))
    q = Image.fromarray(mask.astype(np.uint8) * 255).resize((half.width // 2, half.height // 2))
    best = max(np.arange(-4, 4.01, 0.5),
               key=lambda a: float((np.asarray(q.rotate(a)).sum(1).astype(float) ** 2).sum()))
    full = full.rotate(best, Image.BILINEAR, fillcolor=255)
    half = half.rotate(best, Image.BILINEAR, fillcolor=255)
    g = np.asarray(half, dtype=np.float64)
    mask, ratio = _ink(g)
    H, W = mask.shape

    # line centres from the row-ink profile
    prof = mask[:, int(0.05 * W):int(0.97 * W)].sum(1).astype(float)
    k = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2)
    sm = np.convolve(prof, k / k.sum(), "same")
    a = sm - sm.mean()
    ac = np.correlate(a, a, "full")[len(a) - 1:]
    seg = ac[15:min(200, len(ac) // 3)]
    P = 15 + next(i for i in range(1, len(seg) - 1)
                  if seg[i] >= 0.7 * seg.max() and seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1])
    peaks = [i for i in range(1, H - 1) if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1] and sm[i] > 0.06 * sm.max()]
    centres = []
    for c in sorted(peaks, key=lambda i: -sm[i]):
        if all(abs(c - o) >= 0.6 * P for o in centres):
            centres.append(c)
    centres.sort()

    # rough boundaries: least-ink path between each pair of neighbouring centres
    cuts = [np.full(W, max(0, centres[0] - int(0.75 * P)))]
    m = (_box(mask.astype(np.float64), 1) > 0).astype(np.float64)
    for c0, c1 in zip(centres, centres[1:]):
        yy = np.arange(c0, c1 + 1)[:, None]
        mid, half_w = (c0 + c1) / 2.0, max(1.0, (c1 - c0) / 2.0)
        cuts.append(c0 + _seam(4.0 * m[c0:c1 + 1] + 0.05 * ((yy - mid) / half_w) ** 2))
    cuts.append(np.full(W, min(H - 1, centres[-1] + int(0.75 * P))))
    L = len(cuts) - 1

    # give every connected piece of ink, whole, to the line holding most of it
    lab, nc = _label(mask)
    Y = np.arange(H)[:, None]
    region = sum((Y >= cuts[n][None, :]).astype(np.int16) for n in range(1, L)) if L > 1 else np.zeros((H, W), np.int16)
    valid = mask & (Y >= cuts[0][None, :]) & (Y < cuts[-1][None, :])
    cnt = np.bincount(lab[valid] * L + region[valid], minlength=(nc + 1) * L).reshape(nc + 1, L)
    area = cnt.sum(1)
    ys, xs = np.nonzero(mask)
    pl = lab[ys, xs]
    cutsArr = np.array(cuts)
    rp = region[ys, xs]
    T, B = cutsArr[rp, xs], cutsArr[rp + 1, xs]
    fr = (ys - T) / np.maximum(B - T, 1)
    inCore = (fr >= 0.25) & (fr < 0.75) & valid[ys, xs]
    coreCnt = np.bincount(pl[inCore] * L + rp[inCore], minlength=(nc + 1) * L).reshape(nc + 1, L)
    owner = np.where(coreCnt.sum(1) > 0, coreCnt.argmax(1), cnt.argmax(1))   # head of a g/y decides, not its tail
    owner[area == 0] = -1
    top, bot = np.full(nc + 1, H), np.zeros(nc + 1, int)
    np.minimum.at(top, pl, ys)
    np.maximum.at(bot, pl, ys)
    owner[(bot - top) > 2.2 * P] = -1                                    # binder outline, margin rule
    # loose small pieces (dots, tails, ticks) belong to the nearest big piece directly above or below
    # them; every threshold is relative to the page's own line spacing P
    xmin, xmax = np.full(nc + 1, W), np.zeros(nc + 1, int)
    np.minimum.at(xmin, pl, xs)
    np.maximum.at(xmax, pl, xs)
    big = area >= 0.065 * P * P
    small = (area > 0) & (area < 0.02 * P * P)
    narrow = (area >= 0.02 * P * P) & (area < 0.17 * P * P) & ((xmax - xmin) <= 0.65 * P) & ((bot - top) >= 0.4 * P)

    def nearest(l, win):
        xa, xb = max(0, xmin[l] - 2), xmax[l] + 3
        best = None
        for gap in range(1, win + 1):
            for side, yy in (("up", top[l] - gap), ("down", bot[l] + gap)):
                if 0 <= yy < H:
                    row = lab[yy, xa:xb]
                    hit = [v for v in np.unique(row[row > 0]) if v != l and big[v] and owner[v] >= 0]
                    if hit:
                        return owner[hit[0]]
        return best

    for l in np.nonzero(small | narrow)[0]:
        o = nearest(l, int(0.35 * P) if small[l] else max(2, int(0.12 * P)))
        if o is not None:
            owner[l] = o

    # crops from the full-resolution, lighting-normalised page
    bg_full = np.asarray(Image.fromarray(_box(g, 40).astype(np.float32)).resize(full.size, Image.BILINEAR))
    norm = np.clip(np.asarray(full, dtype=np.float32) / np.maximum(bg_full, 1.0), 0, 1) * 255
    f = full.width / W
    own = owner[lab]
    ink_per_line = np.bincount(own[mask & (own >= 0)], minlength=L)
    minInk = max(60, 0.12 * np.median(ink_per_line[ink_per_line > 0]))
    lines = []
    for n in range(L):
        ink = mask & (own == n)
        ys, xs = np.nonzero(ink)
        if len(ys) < minInk or ys.max() - ys.min() < 0.45 * P:   # nothing, or too flat to be a text line
            continue
        pad = int(0.15 * P)
        y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + 1 + pad)
        x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + 1 + pad)
        Y0, Y1, X0, X1 = int(y0 * f), int(y1 * f), int(x0 * f), int(x1 * f)
        near = _box(ink[y0:y1, x0:x1].astype(np.float64), 3) > 0
        near = np.asarray(Image.fromarray(near.astype(np.uint8) * 255).resize((X1 - X0, Y1 - Y0))) > 127
        crop = np.where(near, norm[Y0:Y1, X0:X1], 255).astype(np.uint8)
        lines.append(dict(order=len(lines), image=Image.fromarray(crop), bbox=(X0, Y0, X1, Y1)))

    # overlay for a quick visual check: rough boundaries
    ov = Image.fromarray(np.clip(ratio * 255, 0, 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(ov)
    for n, cu in enumerate(cuts):
        d.line(list(zip(range(W), cu.tolist())), fill=(255, 0, 0) if n % 2 else (0, 140, 255), width=2)
    return lines, ov, dict(angle=float(best), pitch=int(P), nLines=len(lines), seconds=time.time() - t0)


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    lines, ov, info = SegmentLines(src)
    for ln in lines:
        ln["image"].save(out / f"{ln['order']:02d}.png")
    ov.save(out / "_cuts_overlay.jpg", quality=88)
    print(info)
