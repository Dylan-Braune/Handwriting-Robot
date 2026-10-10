"""Trace the dot detector: every small blob in the band above the x-height, and
which cell (if any) it was matched to.

    python dot_trace.py yeukita --page 0 --line 0
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: F401

import numpy as np

import BuildStyleProfile as BSP
import SegmentLean as SL
from authors_config import personal_author_pages
from ExtractIAMLines import ReadLabelLines
from np_inference.text_model import PaperCRNNNumpy
from SegmentPage import LabelComponents

ap = argparse.ArgumentParser()
ap.add_argument("author")
ap.add_argument("--page", type=int, default=0)
ap.add_argument("--line", type=int, default=0)
ap.add_argument("--reach", type=float, default=0.6)
args = ap.parse_args()

img_path, label_path = list(personal_author_pages(args.author))[args.page]
lean, _ov, _info = SL.SegmentLines(str(img_path))
gt = [g for g in ReadLabelLines(str(img_path), str(label_path))
      if g.strip() != "MESS"]
gray = np.asarray(lean[args.line]["image"])
text = gt[args.line]
model = PaperCRNNNumpy(checkpoint_path=BSP.PERSONAL_WEIGHTS)

# reproduce the extraction up to the cut list
glyphs, stats = BSP.ExtractLineGlyphs(gray, text, model)
ink = BSP.BinarizeLine(gray)
top, base = BSP.CoreBand(ink)
xh = float(base - top)
cuts = stats["cuts"]
print(f"text {text!r}")
print(f"xh={xh:.1f} top={top} base={base}")
print(f"dotted cells: "
      + ", ".join(f"{ch}({int(L)}-{int(R)})" for (_i, ch, L, R) in cuts
                  if ch in "ij"))

bounds = [(int(round(L)), int(round(R))) for (_i, _c, L, R) in cuts]
topRow = int(max(0, base - 1.35 * xh))
botRow = int(min(ink.shape[0], base - 0.70 * xh))
print(f"band rows {topRow}..{botRow}  (y {0.70:.2f}..{1.35:.2f} xh)")
band = ink[topRow:botRow, :]
lab, n = LabelComponents(band, connectivity=8)
print(f"{n} blobs in band")
for li in range(1, n + 1):
    ys, xs = np.nonzero(lab == li)
    w = xs.max() - xs.min() + 1
    h = ys.max() - ys.min() + 1
    cx = 0.5 * (xs.min() + xs.max())
    cy = topRow + 0.5 * (ys.min() + ys.max())
    small = w <= 0.40 * xh and h <= 0.40 * xh and xs.size <= 0.15 * xh * xh
    round_ = abs(w - h) <= max(2, 0.6 * max(w, h))
    tag = "DOT" if (small and round_) else "   "
    # nearest dotted cell
    best, bestd = None, 1e9
    for k2, (i, ch, L, R) in enumerate(cuts):
        if ch not in "ij":
            continue
        sub = ink[:, max(0, bounds[k2][0]):max(0, bounds[k2][1])]
        yy = np.nonzero(sub.any(axis=1))[0]
        inkTop = float(base - yy.min()) if yy.size else -1e9
        if inkTop >= cy:
            continue
        d = 0.0 if L <= cx <= R else min(abs(cx - L), abs(cx - R))
        if d <= args.reach * xh and d < bestd:
            best, bestd = (k2, ch), d
    print(f"  {tag} {xs.size:>4}px w={w:>3} h={h:>3} y={(base-cy)/xh:>5.2f}xh "
          f"x {xs.min():>4}..{xs.max():<4} cx={cx:>6.1f}  -> "
          + (f"cell {best[1]!r} d={bestd:.1f}px" if best else "NO CELL"))
