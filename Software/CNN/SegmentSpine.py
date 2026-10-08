"""Text-line separation by line spines + neighbour-aware piece assignment.

1. lighting normalisation, ink mask, deskew, line spacing P and rough line centres
   (shared helpers from SegmentSimple.py)
2. density map = ink blurred wide and short; each text line's SPINE is the ridge of that
   map (dynamic-programming path inside the band between neighbouring line centres)
3. connected pieces of ink (strokes, dots, tails, quote marks) each choose a spine by
       energy = distance to the spine (weaker for small pieces)
              + penalty for taking a different line than the pieces next to them
   solved by a few rounds of iterated conditional modes
4. a piece that really spans two lines (tall) is split pixel by pixel by nearest spine
5. each line = its own ink: crop, white out the rest

Everything is relative to the page's own line spacing P.
Usage: python SegmentSpine.py page.jpg outdir
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from SegmentSimple import _box, _ink, _label, _seam


def _smooth(a, r, axis):
    """Box mean of radius r along one axis (integral image)."""
    a = np.moveaxis(a.astype(np.float64), axis, 0)
    n = a.shape[0]
    c = np.cumsum(np.pad(a, [(r + 1, r)] + [(0, 0)] * (a.ndim - 1), mode="edge"), 0)
    return np.moveaxis((c[2 * r + 1:2 * r + 1 + n] - c[:n]) / (2 * r + 1), 0, axis)


def SegmentLines(path, lam=0.5):
    t0 = time.time()
    full = Image.open(path).convert("L")
    half = full.resize((full.width // 2, full.height // 2), Image.BOX)

    # deskew
    mask, _ = _ink(np.asarray(half, dtype=np.float64))
    q = Image.fromarray(mask.astype(np.uint8) * 255).resize((half.width // 2, half.height // 2))
    best = max(np.arange(-4, 4.01, 0.5),
               key=lambda a: float((np.asarray(q.rotate(a)).sum(1).astype(float) ** 2).sum()))
    full = full.rotate(best, Image.BILINEAR, fillcolor=255)
    half = half.rotate(best, Image.BILINEAR, fillcolor=255)
    g = np.asarray(half, dtype=np.float64)
    mask, ratio = _ink(g)
    H, W = mask.shape

    # line spacing P and rough line centres from the row-ink profile
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
    L = len(centres)

    # spines: ridge of the wide-and-short density map inside the band around each centre
    D = mask.astype(np.float64)
    for _ in range(2):
        D = _smooth(_smooth(D, max(1, int(0.2 * P)), 0), int(1.5 * P), 1)
    D /= D.max()
    edges = [max(0, centres[0] - int(0.6 * P))] + [(c0 + c1) // 2 for c0, c1 in zip(centres, centres[1:])] \
        + [min(H, centres[-1] + int(0.6 * P))]
    spines = np.array([edges[n] + _seam(-D[edges[n]:edges[n + 1]], step=2) for n in range(L)])    # (L, W)
    Y = np.arange(H)[:, None]
    support = []
    for n in range(L):
        cols = np.nonzero((mask & (np.abs(Y - spines[n][None, :]) < 0.5 * P)).any(0))[0]
        support.append((cols.min(), cols.max()) if len(cols) else (0, -1))

    # connected pieces, and their distance to every spine
    lab, nc = _label(mask)
    ys, xs = np.nonzero(mask)
    pl = lab[ys, xs]
    area = np.bincount(pl, minlength=nc + 1).astype(float)
    cx = np.bincount(pl, xs, nc + 1) / np.maximum(area, 1)
    cy = np.bincount(pl, ys, nc + 1) / np.maximum(area, 1)
    top, bot = np.full(nc + 1, H), np.zeros(nc + 1, int)
    np.minimum.at(top, pl, ys)
    np.maximum.at(bot, pl, ys)
    ids = np.nonzero((area > 0) & ((bot - top) <= 2.2 * P))[0]          # tall = binder outline / margin rule
    ci = np.minimum(cx[ids].astype(int), W - 1)
    dist = np.empty((len(ids), L))
    for n in range(L):
        x1, x2 = support[n]
        out = np.maximum(0, np.maximum(x1 - cx[ids], cx[ids] - x2))
        dist[:, n] = (np.abs(cy[ids] - spines[n][ci]) + 0.15 * out) / P
    conf = np.clip(area[ids] / (0.1 * P * P), 0.15, 1.0)               # small pieces trust their neighbours

    # neighbour graph and iterated conditional modes
    pts = np.stack([cx[ids], cy[ids]], 1)
    dd = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    nbr = np.where((dd < 1.0 * P) & (dd > 0), np.exp(-dd / (0.35 * P)), 0.0) * conf[None, :]
    label = dist.argmin(1)
    for _ in range(8):
        changed = 0
        for i in np.argsort(area[ids]):
            cand = np.nonzero(dist[i] < dist[i].min() + 1.5)[0]
            e = [conf[i] * dist[i, c] + lam * nbr[i][label != c].sum() for c in cand]
            new = cand[int(np.argmin(e))]
            changed += new != label[i]
            label[i] = new
        if not changed:
            break
    owner = np.full(nc + 1, -1)
    owner[ids] = label

    # a tall piece touching two lines: split pixel by pixel by nearest spine
    own = owner[lab]
    for l in ids[(bot[ids] - top[ids]) > 1.5 * P]:
        sel = lab == l
        py, px = np.nonzero(sel)
        near = np.argmin(np.abs(py[:, None] - spines[:, px].T), axis=1)
        own[py, px] = near

    # crops from the full-resolution, lighting-normalised page
    bg_full = np.asarray(Image.fromarray(_box(g, 40).astype(np.float32)).resize(full.size, Image.BILINEAR))
    norm = np.clip(np.asarray(full, dtype=np.float32) / np.maximum(bg_full, 1.0), 0, 1) * 255
    f = full.width / W
    perLine = np.bincount(own[mask & (own >= 0)], minlength=L)
    minInk = max(60, 0.12 * np.median(perLine[perLine > 0]))
    lines = []
    for n in range(L):
        ink = mask & (own == n)
        yy, xx = np.nonzero(ink)
        if len(yy) < minInk or yy.max() - yy.min() < 0.45 * P:
            continue
        pad = int(0.15 * P)
        y0, y1 = max(0, yy.min() - pad), min(H, yy.max() + 1 + pad)
        x0, x1 = max(0, xx.min() - pad), min(W, xx.max() + 1 + pad)
        Y0, Y1, X0, X1 = int(y0 * f), int(y1 * f), int(x0 * f), int(x1 * f)
        nz = _box(ink[y0:y1, x0:x1].astype(np.float64), 3) > 0
        nz = np.asarray(Image.fromarray(nz.astype(np.uint8) * 255).resize((X1 - X0, Y1 - Y0))) > 127
        crop = np.where(nz, norm[Y0:Y1, X0:X1], 255).astype(np.uint8)
        lines.append(dict(order=len(lines), image=Image.fromarray(crop), bbox=(X0, Y0, X1, Y1)))

    ov = Image.fromarray(np.clip(ratio * 255, 0, 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(ov)
    for n in range(L):
        d.line(list(zip(range(W), spines[n].tolist())), fill=(255, 0, 0) if n % 2 else (0, 140, 255), width=2)
    return lines, ov, dict(angle=float(best), pitch=int(P), nLines=len(lines), seconds=time.time() - t0)


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    lines, ov, info = SegmentLines(src)
    for ln in lines:
        ln["image"].save(out / f"{ln['order']:02d}.png")
    ov.save(out / "_spines_overlay.jpg", quality=88)
    print(info)
