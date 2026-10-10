"""Extract a few of an author's lines with the CURRENT BuildStyleProfile and
print the stroke geometry of one letter, so a cut fix can be judged quickly
without re-extracting the whole author.

    python test_cut.py yeukita i --lines 6
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
from authors_config import personal_author_pages, VAL_FRACTION
from ExtractIAMLines import ReadLabelLines
from np_inference.text_model import PaperCRNNNumpy

ap = argparse.ArgumentParser()
ap.add_argument("author")
ap.add_argument("letter")
ap.add_argument("--lines", type=int, default=6)
ap.add_argument("--page", type=int, default=0)
args = ap.parse_args()

pages = list(personal_author_pages(args.author))
img_path, label_path = pages[args.page]
lean, _ov, info = SL.SegmentLines(str(img_path))
gt = [g for g in ReadLabelLines(str(img_path), str(label_path))
      if g.strip() != "MESS"]
n = min(len(lean), len(gt))
items = list(zip([np.asarray(l["image"]) for l in lean[:n]], gt[:n]))
print(f"{img_path.name}: {n} lines, pitch={info['pitch']}")

model = PaperCRNNNumpy(checkpoint_path=BSP.PERSONAL_WEIGHTS)
found = 0
if args.letter == "*":                       # dump every letter on a few lines
    for idx, (gray, text) in enumerate(items):
        glyphs, stats = BSP.ExtractLineGlyphs(gray, text, model)
        if glyphs is None:
            continue
        print(f"\nline {idx}: {text[:56]!r}")
        for g in glyphs:
            if not g or g.get("char", " ") == " ":
                continue
            print(f"   {g['char']!r:>4} adv {g['advance']:.2f}  w {g['width']:.2f}  "
                  f"h {g['top']-g['bot']:.2f}  strokes {len(g['strokes'])}  "
                  f"conn {'L' if g.get('connL') else '-'}"
                  f"{'R' if g.get('connR') else '-'}")
        found += 1
        if found >= args.lines:
            break
    raise SystemExit(0)

for idx, (gray, text) in enumerate(items):
    if args.letter not in text:
        continue
    glyphs, stats = BSP.ExtractLineGlyphs(gray, text, model)
    if glyphs is None:
        continue
    for g in glyphs:
        if not g or g.get("char") != args.letter:
            continue
        found += 1
        print(f"\nline {idx}: {text[:56]!r}")
        print(f"  FINISHED glyph: {len(g['strokes'])} strokes, "
              f"w {g['width']:.2f} h {g['top']-g['bot']:.2f}")
        for si, s in enumerate(g["strokes"]):
            xs = [p[0] for p in s]
            ys = [p[1] for p in s]
            print(f"    stroke {si}: x {min(xs):>6.2f}..{max(xs):<6.2f} "
                  f"y {min(ys):>6.2f}..{max(ys):<6.2f}  ({len(s)} pts)")
        if found >= args.lines:
            break
    if found >= args.lines:
        break
print(f"\nfound {found} {args.letter!r} instances")
