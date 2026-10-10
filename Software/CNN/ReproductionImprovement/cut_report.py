"""Compare a page's letter cells before/after a cutter change, using the
criterion that a cell should be roughly the width of the letter in it.

    python cut_report.py yeukita --lines 10
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

XONLY = set("acemnorsuvwxz")
ASCEND = set("bdfhklt")
DESCEND = set("gjpqy")

ap = argparse.ArgumentParser()
ap.add_argument("author")
ap.add_argument("--lines", type=int, default=10)
ap.add_argument("--page", type=int, default=0)
args = ap.parse_args()

img_path, label_path = list(personal_author_pages(args.author))[args.page]
lean, _ov, _info = SL.SegmentLines(str(img_path))
gt = [g for g in ReadLabelLines(str(img_path), str(label_path))
      if g.strip() != "MESS"]
n = min(len(lean), len(gt))
items = list(zip([np.asarray(l["image"]) for l in lean[:n]], gt[:n]))[:args.lines]

model = PaperCRNNNumpy(checkpoint_path=BSP.PERSONAL_WEIGHTS)
rows = []
for gray, text in items:
    glyphs, _st = BSP.ExtractLineGlyphs(gray, text, model)
    if glyphs is None:
        continue
    for g in glyphs:
        if not g or g.get("char", " ") == " ":
            continue
        ch = g["char"]
        rows.append((ch, g["width"], g["top"] - g["bot"], len(g["strokes"])))

def ok_vert(ch, h, ):
    return True

w = np.array([r[1] for r in rows]); h = np.array([r[2] for r in rows])
ns = np.array([r[3] for r in rows])
print(f"{args.author}/{img_path.name}: {len(items)} lines, {len(rows)} letters")
print(f"  cell width  (xh): median {np.median(w):.2f}  p90 {np.percentile(w,90):.2f}"
      f"  max {w.max():.2f}   over 1.6: {(w>1.6).mean()*100:.1f}%")
print(f"  cell height (xh): median {np.median(h):.2f}  p90 {np.percentile(h,90):.2f}"
      f"  max {h.max():.2f}   over 2.0: {(h>2.0).mean()*100:.1f}%")
print(f"  strokes/letter : median {np.median(ns):.0f}  max {ns.max():.0f}"
      f"   over 4: {(ns>4).mean()*100:.1f}%")
print("\n  per letter (width / height / strokes):")
seen = {}
for ch, ww, hh, nn in rows:
    seen.setdefault(ch, []).append((ww, hh, nn))
for ch in sorted(seen):
    a = np.array(seen[ch])
    print(f"    {ch!r:>4} n={len(a):>3}  w {np.median(a[:,0]):.2f}  "
          f"h {np.median(a[:,1]):.2f}  strokes {np.median(a[:,2]):.0f}")
