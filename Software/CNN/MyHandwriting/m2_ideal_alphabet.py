"""Render the per-author IDEAL alphabet -- the anchor built from that
author's own writing -- so it can be judged letter by letter.

Each cell is the robust median of that author's own variants of the letter,
drawn in the normalized frame with the baseline and x-height marked. The
label under each shows priorD (distance from the cross-author idea of that
letter, lower = more obviously that letter) and how many variants it was
built from. Letters with no ideal at all are shown as an empty cell.
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import _env
from _env import OUT_DIR

import SynthesizeHandwriting as SY

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _cell(g, px, w, h, font, ch, gate):
    im = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(im)
    top = 16
    if g is None:
        d.text((4, 2), ch, fill=140, font=font)
        d.text((4, h - 16), "none", fill=170, font=font)
        return im
    strokes = g["strokes"]
    xs = [p[0] for s in strokes for p in s]
    ys = [p[1] for s in strokes for p in s]
    x0, y1 = min(xs), max(ys)
    # baseline (y=0) and x-height (y=1) guides
    for yv, shade in ((0.0, 165), (1.0, 215)):
        yy = top + (y1 - yv) * px
        if top - 2 < yy < h - 14:
            d.line([(0, yy), (w, yy)], fill=shade)
    for s in strokes:
        pts = [(6 + (x - x0) * px, top + (y1 - y) * px) for (x, y) in s]
        if len(pts) >= 2:
            d.line(pts, fill=0, width=3, joint="curve")
    pd = float(g.get("priorD", 1.0))
    mark = "OK " if pd <= gate else "gen"
    d.text((4, 2), ch, fill=90, font=font)
    d.text((4, h - 16), f"{mark} {pd:.2f} n{g.get('nPool', 0)}", fill=110, font=font)
    return im


def main(author, px=30, cols=7, gate=0.45, source="medoid"):
    prof = SY.LoadAllProfiles()[author]
    ideal = prof.get("idealMedoid" if source == "medoid" else "idealGlyphs") or {}
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    cw, chh = int(px * 2.6), int(px * 3.6)
    rows = (len(LETTERS) + cols - 1) // cols
    sheet = Image.new("L", (cols * cw + 12, rows * chh + 34), 255)
    d = ImageDraw.Draw(sheet)
    n_ok = sum(1 for c in LETTERS
               if c in ideal and float(ideal[c].get("priorD", 1)) <= gate)
    d.text((6, 6), f"author {author} -- {source} anchor alphabet   "
                   f"({len(ideal)} built, {n_ok}/26 pass gate {gate})",
           fill=0, font=font)
    for i, ch in enumerate(LETTERS):
        r, c = divmod(i, cols)
        sheet.paste(_cell(ideal.get(ch), px, cw - 6, chh - 6, font, ch, gate),
                    (6 + c * cw, 28 + r * chh))
    p = OUT_DIR / f"{source}_{author}.png"
    sheet.save(p)
    print(f"-> {p}")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("authors")
    ap.add_argument("--gate", type=float, default=0.45)
    ap.add_argument("--source", default="medoid", help="medoid|median")
    args = ap.parse_args()
    for a in args.authors.split(","):
        main(a, gate=args.gate, source=args.source)
