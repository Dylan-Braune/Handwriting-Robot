"""Measure the ink inside a letter's cell: connected blobs, their size and the
height above the baseline. This is how the dot's real position gets pinned down.

    python dot_probe.py yeukita i --page 0 --line 0
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
ap.add_argument("letter")
ap.add_argument("--page", type=int, default=0)
ap.add_argument("--line", type=int, default=0)
args = ap.parse_args()

img_path, label_path = list(personal_author_pages(args.author))[args.page]
lean, _ov, _info = SL.SegmentLines(str(img_path))
gt = [g for g in ReadLabelLines(str(img_path), str(label_path))
      if g.strip() != "MESS"]
gray = np.asarray(lean[args.line]["image"])
text = gt[args.line]
model = PaperCRNNNumpy(checkpoint_path=BSP.PERSONAL_WEIGHTS)
glyphs, stats = BSP.ExtractLineGlyphs(gray, text, model)
ink = BSP.BinarizeLine(gray)
band = BSP.CoreBand(ink)
top, base = band
xh = float(base - top)
print(f"{args.author} {img_path.name} line {args.line}")
print(f"  text {text!r}")
print(f"  xh={xh:.1f}px  top={top} base={base}  ink height rows "
      f"{np.nonzero(ink.any(axis=1))[0].min()}..{np.nonzero(ink.any(axis=1))[0].max()}")

for (i, ch, L, R) in stats["cuts"]:
    if ch != args.letter:
        continue
    L, R = int(round(L)), int(round(R))
    # FULL column height, not just the top band: the dot may sit above x-height
    for label, x0, x1 in (("cell", L, R), ("wide", L - 6, R + 6)):
        x0, x1 = max(0, x0), max(0, x1)
        sub = ink[:, x0:x1]
        lab, n = LabelComponents(sub, connectivity=8)
        print(f"\n  {label} {x0}-{x1} ({(x1-x0)/xh:.2f} xh wide)")
        for li in range(1, n + 1):
            ys, xs = np.nonzero(lab == li)
            if xs.size == 0:
                continue
            w = xs.max() - xs.min() + 1
            h = ys.max() - ys.min() + 1
            yTop = (base - ys.min()) / xh
            yBot = (base - ys.max()) / xh
            print(f"    blob: {xs.size:>4} px  w={w:>3} h={h:>3}  "
                  f"y {yBot:>5.2f}..{yTop:<5.2f} xh   "
                  f"x {x0+xs.min():>4}..{x0+xs.max():<4}")
