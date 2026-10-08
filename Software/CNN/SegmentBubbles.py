"""Stage 1 diagnostic: bubble every ink pixel and see what counts as a line.

1. lighting-normalise, threshold ink (Otsu), drop big solid blobs (binder, shadow patches)
2. letter height h = median height of the small connected pieces of ink
3. bubble = ink smeared sideways by 1.5 h and up/down by 0.3 h, so a line becomes one blob
4. bounding box of every bubble, coloured by shape (all thresholds relative to h):
     green  = line-like   (wide enough, not too tall)
     red    = too small   (dot, mark, noise)
     orange = too tall    (merged lines, binder, margin rule)

Usage: python SegmentBubbles.py page.jpg out.jpg
"""
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from SegmentSimple import _box, _ink, _label

RX, RY = 1.5, 0.3           # bubble radius as a multiple of the letter height h
MIN_W, MAX_H = 2.5, 3.5     # line-like: width >= 2.5 h and height <= 3.5 h


def _any(mask, r, axis):
    """True where any pixel within r along `axis` is set."""
    a = np.moveaxis(mask.astype(np.float64), axis, 0)
    n = a.shape[0]
    c = np.cumsum(np.pad(a, [(r + 1, r)] + [(0, 0)] * (a.ndim - 1)), 0)
    return np.moveaxis((c[2 * r + 1:2 * r + 1 + n] - c[:n]) > 0, 0, axis)


def Bubbles(path):
    t0 = time.time()
    full = Image.open(path).convert("L")
    half = full.resize((full.width // 2, full.height // 2), Image.BOX)
    mask, ratio = _ink(np.asarray(half, dtype=np.float64))
    H, W = mask.shape

    lab, n = _label(mask)
    ys, xs = np.nonzero(mask)
    pl = lab[ys, xs]
    top, bot = np.full(n + 1, H), np.zeros(n + 1, int)
    np.minimum.at(top, pl, ys)
    np.maximum.at(bot, pl, ys)
    hts = (bot - top + 1)[1:]
    h = float(np.median(hts[(hts >= 5) & (hts <= 80)]))

    bubble = _any(_any(mask, int(RX * h), 1), max(1, int(RY * h)), 0)
    blab, bn = _label(bubble)
    ys, xs = np.nonzero(bubble)
    pb = blab[ys, xs]
    y0, y1 = np.full(bn + 1, H), np.zeros(bn + 1, int)
    x0, x1 = np.full(bn + 1, W), np.zeros(bn + 1, int)
    np.minimum.at(y0, pb, ys)
    np.maximum.at(y1, pb, ys)
    np.minimum.at(x0, pb, xs)
    np.maximum.at(x1, pb, xs)

    ov = Image.fromarray(np.clip(ratio * 255, 0, 255).astype(np.uint8)).convert("RGB")
    tint = Image.new("RGB", ov.size)
    td = ImageDraw.Draw(tint)
    d = ImageDraw.Draw(ov)
    counts = {"line-like": 0, "too small": 0, "too tall": 0}
    for b in range(1, bn + 1):
        w, hh = x1[b] - x0[b] + 1, y1[b] - y0[b] + 1
        if hh > MAX_H * h:
            kind, col = "too tall", (255, 150, 0)
        elif w < MIN_W * h:
            kind, col = "too small", (230, 0, 0)
        else:
            kind, col = "line-like", (0, 170, 0)
        counts[kind] += 1
        td.rectangle([x0[b], y0[b], x1[b], y1[b]], fill=tuple(c // 4 for c in col))
        d.rectangle([x0[b], y0[b], x1[b], y1[b]], outline=col, width=2)
    ov = Image.fromarray(np.clip(np.asarray(ov, np.int16) + np.asarray(tint, np.int16), 0, 255).astype(np.uint8))
    return ov, dict(letterHeight=round(h, 1), bubbleBoxes=bn, **counts, seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    ov, info = Bubbles(sys.argv[1])
    ov.save(sys.argv[2], quality=90)
    print(info)
