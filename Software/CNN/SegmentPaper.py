"""Text-line segmentation following Barakat et al., "Learning-Free Text Line Segmentation for
Historical Handwritten Documents", Applied Sciences 10(22):8276 (2020).

1. character height range = (mean +- std of connected-component heights) halved
2. blob lines: scale-normalised Laplacian of anisotropic Gaussians (sigma_x = e * sigma_y), strongest
   response over a set of scales covering that height range
3. blob-line binarisation: component tree over thresholds of the response map; a component is a
   valid blob line if a k-knot piecewise-linear least-squares spline fits it with mean error (FS1)
   below the maximum character height, otherwise its higher-threshold children are examined
4. broken blob lines are merged when the connecting direction lies between their directions
   and their vertical distance is below the maximum character height
5. each connected piece of ink is labelled with a blob line by energy minimisation:
   data cost (distance to the blob line) + smoothness cost (nearby pieces prefer the same label)
   + label cost (blob lines with little ink are spurious)
6. pieces overlapping two blob lines are split pixel by pixel by nearest blob line

Differences from the paper (its numeric settings were not available to me): elongation e = 5,
10 spline knots, 20 threshold levels; the energy is minimised by iterated conditional modes with a
label-removal pass, not graph cuts. Binder/shadow removal is reused from SegmentSimple.
Usage: python SegmentPaper.py page.jpg outdir
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from SegmentSimple import _box, _ink, _label

ELONG, KNOTS, LEVELS, SCALES = 5.0, 10, 20, 5


def _conv(a, k, axis):
    r = len(k) // 2
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    p = np.pad(a, [(r, r)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    out = np.zeros(a.shape, np.float64)
    for i, w in enumerate(k):
        out += w * p[i:i + n]
    return np.moveaxis(out, 0, axis)


def _gauss(sig):
    r = max(1, int(3 * sig))
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-x * x / (2 * sig * sig))
    g /= g.sum()
    return g, g * (x * x / sig ** 4 - 1 / sig ** 2)            # kernel, its second derivative


def _fit(xs, ys, knots_n):
    """k-knot piecewise-linear least-squares spline -> (knot x, knot y, mean abs error)."""
    x0, x1 = xs.min(), xs.max()
    k = int(min(knots_n, max(2, (x1 - x0) // 3)))
    kx = np.linspace(x0, x1, k)
    B = np.stack([np.interp(xs, kx, np.eye(k)[j]) for j in range(k)], 1)
    w = np.linalg.lstsq(B, ys.astype(np.float64), rcond=None)[0]
    return kx, w, float(np.abs(ys - B @ w).mean())


def SegmentLines(path):
    t0 = time.time()
    full = Image.open(path).convert("L")
    half = full.resize((full.width // 2, full.height // 2), Image.BOX)
    g = np.asarray(half, dtype=np.float64)
    mask, ratio = _ink(g)
    H, W = mask.shape

    # 1. character height range from connected-component heights (outer 10% of the page excluded)
    lab, nc = _label(mask)
    ys, xs = np.nonzero(mask)
    pl = lab[ys, xs]
    area = np.bincount(pl, minlength=nc + 1).astype(float)
    top, bot = np.full(nc + 1, H), np.zeros(nc + 1, int)
    np.minimum.at(top, pl, ys)
    np.maximum.at(bot, pl, ys)
    cx = np.bincount(pl, xs, nc + 1) / np.maximum(area, 1)
    cy = np.bincount(pl, ys, nc + 1) / np.maximum(area, 1)
    hts = (bot - top + 1).astype(float)
    ok = (area > 0) & (hts >= 6) & (hts <= 60) & (cx > 0.1 * W) & (cx < 0.9 * W) & (cy > 0.1 * H) & (cy < 0.9 * H)
    mu, sd = hts[ok].mean(), hts[ok].std()
    maxH = mu + sd                                            # maximum character height (half-res px)
    lo, hi = max(2.0, (mu - sd) / 2), (mu + sd) / 2           # the paper's "half of the range"

    # 2. blob lines at quarter resolution
    q = mask[:H // 2 * 2, :W // 2 * 2].reshape(H // 2, 2, W // 2, 2).mean((1, 3))
    Hq, Wq = q.shape
    resp = np.zeros((Hq, Wq))
    for h in np.linspace(lo, hi, SCALES):
        sy = max(0.8, h / 2.8 / 2)
        sx = ELONG * sy
        gx, gx2 = _gauss(sx)
        gy, gy2 = _gauss(sy)
        lap = _conv(_conv(q, gx, 1), gy2, 0) + _conv(_conv(q, gx2, 1), gy, 0)
        resp = np.maximum(resp, -sy * sy * lap)               # scale-normalised, bright blobs positive
    levels = np.linspace(0, resp.max(), LEVELS + 1)[1:-1]
    maxHq = maxH / 2

    # 3. component tree: accept a component when a spline fits it, else look at its children
    blobs = []

    def visit(sub, y0, x0, lvl):
        h, w = sub.shape
        lab_, n = _label(sub & (resp[y0:y0 + h, x0:x0 + w] > levels[lvl]))
        for cid in range(1, n + 1):
            py, px = np.nonzero(lab_ == cid)
            if len(py) < 20:
                continue
            kx, ky, fs1 = _fit(px + x0, py + y0, KNOTS)
            if fs1 < maxHq:
                blobs.append((kx * 2, ky * 2))                # to half-res coordinates
            elif lvl + 1 < len(levels):
                ya, xa = py.min(), px.min()
                m2 = np.zeros((py.max() - ya + 1, px.max() - xa + 1), bool)
                m2[py - ya, px - xa] = True
                visit(m2, y0 + ya, x0 + xa, lvl + 1)

    visit(np.ones((Hq, Wq), bool), 0, 0, 0)

    # 4. merge broken blob lines: extend each along its own slope across the gap; join if both reach the other
    def slope(kx, ky, end):
        m = max(2, len(kx) // 3)
        xs_, ys_ = (kx[-m:], ky[-m:]) if end else (kx[:m], ky[:m])
        return float(np.polyfit(xs_, ys_, 1)[0]) if np.ptp(xs_) > 1 else 0.0

    merged = True
    while merged:
        merged = False
        blobs.sort(key=lambda b: b[0][0])
        for i, a in enumerate(blobs):
            for j, b in enumerate(blobs):
                if i == j or a[0][-1] >= b[0][0]:
                    continue
                gap = b[0][0] - a[0][-1]
                ya = a[1][-1] + np.clip(slope(a[0], a[1], True), -0.1, 0.1) * gap
                yb = b[1][0] - np.clip(slope(b[0], b[1], False), -0.1, 0.1) * gap
                if abs(ya - b[1][0]) < 0.6 * maxH and abs(yb - a[1][-1]) < 0.6 * maxH:
                    blobs[i] = (np.r_[a[0], b[0]], np.r_[a[1], b[1]])
                    del blobs[j]
                    merged = True
                    break
            if merged:
                break
    L = len(blobs)
    curves = []                                               # (x sample, y sample) per blob line
    for kx, ky in blobs:
        xx = np.arange(int(kx[0]), int(kx[-1]) + 1)
        curves.append((xx, np.interp(xx, kx, ky)))

    # 5. energy minimisation over pieces
    ids = np.nonzero((area > 0) & (hts <= 2.2 * hi * 4))[0]
    pts = np.stack([cx[ids], cy[ids]], 1)
    n = len(ids)
    D = np.empty((n, L))
    for j, (xx, yy) in enumerate(curves):
        D[:, j] = np.sqrt((pts[:, None, 0] - xx[None]) ** 2 + (pts[:, None, 1] - yy[None]) ** 2).min(1) / maxH
    dd = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    np.fill_diagonal(dd, np.inf)
    nn = np.argsort(dd, 1)[:, :3]
    pair = np.zeros((n, n))
    for i in range(n):
        pair[i, nn[i]] = 1
    pair = np.maximum(pair, pair.T)
    d2 = dd[pair > 0] ** 2
    W_ = pair * np.exp(-dd ** 2 / (2 * d2.mean()))
    dens = []
    for xx, yy in curves:                                      # fraction of ink near the blob line (label cost)
        cnt = 0
        for x_, y_ in zip(xx[::4], yy[::4]):
            y1, y2 = int(max(0, y_ - 0.5 * maxH)), int(min(H, y_ + 0.5 * maxH))
            cnt += mask[y1:y2, int(x_)].sum()
        dens.append(cnt / max(1.0, len(xx[::4]) * maxH))
    labelCost = 4.0 * (1 - np.minimum(1, np.array(dens) / 0.12))
    lab_i = D.argmin(1)
    for _ in range(10):
        changed = 0
        for i in np.argsort(area[ids]):
            cand = np.argsort(D[i])[:4]
            e = [D[i, c] + 1.0 * W_[i][lab_i != c].sum() for c in cand]
            new = cand[int(np.argmin(e))]
            changed += new != lab_i[i]
            lab_i[i] = new
        for j in np.unique(lab_i):                             # try deleting each label (label cost)
            mem = np.nonzero(lab_i == j)[0]
            alt = D[mem].copy()
            alt[:, j] = np.inf
            used = np.unique(lab_i)
            keep = [u for u in used if u != j]
            if not keep:
                continue
            sub = alt[:, keep]
            newl = np.array(keep)[sub.argmin(1)]
            if (sub.min(1) - D[mem, j]).sum() < labelCost[j]:
                lab_i[mem] = newl
                changed += len(mem)
        if not changed:
            break
    owner = np.full(nc + 1, -1)
    owner[ids] = lab_i
    own = owner[lab]

    # 6. split pieces that overlap two blob lines
    spine = np.full((L, W), np.nan)
    for j, (xx, yy) in enumerate(curves):
        spine[j, np.clip(xx, 0, W - 1)] = yy
    near = lambda py, px: np.nanargmin(np.abs(py[:, None] - spine[:, px].T), axis=1)
    for l in ids[(bot[ids] - top[ids]) > 1.0 * maxH]:
        py, px = np.nonzero(lab == l)
        d = np.abs(py[:, None] - np.where(np.isnan(spine[:, px].T), 1e9, spine[:, px].T))
        close = (d < 0.6 * maxH).sum(0)
        if (close > 0.15 * len(py)).sum() >= 2:
            own[py, px] = d.argmin(1)

    # crops
    bg_full = np.asarray(Image.fromarray(_box(g, 40).astype(np.float32)).resize(full.size, Image.BILINEAR))
    norm = np.clip(np.asarray(full, dtype=np.float32) / np.maximum(bg_full, 1.0), 0, 1) * 255
    f = full.width / W
    inkPer = np.bincount(own[mask & (own >= 0)], minlength=L)
    minInk = max(60, 0.12 * np.median(inkPer[inkPer > 0]))
    order = np.argsort([np.nanmean(c[1]) for c in curves])
    lines = []
    for j in order:
        ink = mask & (own == j)
        yy, xx = np.nonzero(ink)
        if len(yy) < minInk or yy.max() - yy.min() < 0.45 * maxH:
            continue
        pad = int(0.3 * maxH)
        y0, y1 = max(0, yy.min() - pad), min(H, yy.max() + 1 + pad)
        x0, x1 = max(0, xx.min() - pad), min(W, xx.max() + 1 + pad)
        Y0, Y1, X0, X1 = int(y0 * f), int(y1 * f), int(x0 * f), int(x1 * f)
        nz = _box(ink[y0:y1, x0:x1].astype(np.float64), 3) > 0
        nz = np.asarray(Image.fromarray(nz.astype(np.uint8) * 255).resize((X1 - X0, Y1 - Y0))) > 127
        crop = np.where(nz, norm[Y0:Y1, X0:X1], 255).astype(np.uint8)
        lines.append(dict(order=len(lines), image=Image.fromarray(crop), bbox=(X0, Y0, X1, Y1)))

    ov = Image.fromarray(np.clip(ratio * 255, 0, 255).astype(np.uint8)).convert("RGB")
    d = ImageDraw.Draw(ov)
    for j, (xx, yy) in enumerate(curves):
        d.line(list(zip(xx.tolist(), yy.tolist())), fill=(255, 0, 0) if j % 2 else (0, 140, 255), width=2)
    return lines, ov, dict(charHeight=round(float(mu), 1), blobLines=L, nLines=len(lines), seconds=time.time() - t0)


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    lines, ov, info = SegmentLines(src)
    for ln in lines:
        ln["image"].save(out / f"{ln['order']:02d}.png")
    ov.save(out / "_blobs_overlay.jpg", quality=88)
    print(info)
