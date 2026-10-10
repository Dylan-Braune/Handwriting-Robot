"""Crop a line and mark where the cutter put each letter's cell, so the ink of
one letter can be inspected directly.

    python show_cell.py yeukita i --page 0 --line 0
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: F401

import numpy as np
from PIL import Image, ImageDraw

import BuildStyleProfile as BSP
import SegmentLean as SL
from authors_config import personal_author_pages
from ExtractIAMLines import ReadLabelLines
from np_inference.text_model import PaperCRNNNumpy

ap = argparse.ArgumentParser()
ap.add_argument("author")
ap.add_argument("letter")
ap.add_argument("--page", type=int, default=0)
ap.add_argument("--line", type=int, default=0)
ap.add_argument("--scale", type=float, default=3.0)
args = ap.parse_args()

img_path, label_path = list(personal_author_pages(args.author))[args.page]
lean, _ov, _info = SL.SegmentLines(str(img_path))
gt = [g for g in ReadLabelLines(str(img_path), str(label_path))
      if g.strip() != "MESS"]
n = min(len(lean), len(gt))
gray = np.asarray(lean[args.line]["image"])
text = gt[args.line]
print(f"{img_path.name} line {args.line}: {text!r}")

model = PaperCRNNNumpy(checkpoint_path=BSP.PERSONAL_WEIGHTS)
glyphs, stats = BSP.ExtractLineGlyphs(gray, text, model)
if glyphs is None:
    raise SystemExit("extraction failed")
cuts = stats["cuts"]

im = Image.fromarray(gray).convert("RGB")
a = np.array(im)
a[a[:, :, 0] < 170] = [20, 20, 60]
im = Image.fromarray(a)
S = args.scale
im = im.resize((int(im.width * S), int(im.height * S)), Image.Resampling.NEAREST)
d = ImageDraw.Draw(im)
band = BSP.CoreBand(BSP.BinarizeLine(gray))
top, base = band
d.line([(0, base * S), (im.width, base * S)], fill=(90, 190, 90))
d.line([(0, top * S), (im.width, top * S)], fill=(205, 205, 90))
hits = []
for (i, ch, L, R) in cuts:
    col = (220, 25, 25) if ch == args.letter else (170, 170, 200)
    d.line([(L * S, 0), (L * S, im.height)], fill=col)
    d.text((L * S + 2, 2), ch, fill=(180, 0, 0))
    if ch == args.letter:
        hits.append((L, R))
out = HERE / "out" / f"cell_{args.author}_{args.letter}_{args.line}.png"
im.save(out)
print(f"cut cells for {args.letter!r}: "
      + ", ".join(f"{L:.0f}-{R:.0f} ({R-L:.0f}px = {(R-L)/stats['xh']:.2f} xh)"
                  for L, R in hits))
print(f"xh = {stats['xh']:.1f}px, baseline row = {base}")
print("wrote", out)
