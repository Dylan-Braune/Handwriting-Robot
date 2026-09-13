"""Render the stored variants of chosen characters for one author, so the
glyph library can be inspected by eye.

Each variant is drawn in its own cell with the baseline and the x-height
line marked, and the strokes numbered in drawing order, so a spurious
leading stroke (a ligature tail that survived trimming) is visible as an
extra mark before the letter.
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import _env
from _env import OUT_DIR

import SynthesizeHandwriting as SY


def render_variant(g, px=110, pad=34):
    """One glyph, normalized frame (x right, y UP, 1.0 = x-height)."""
    strokes = g["strokes"]
    xs = [p[0] for s in strokes for p in s]
    ys = [p[1] for s in strokes for p in s]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    W = int((x1 - x0) * px) + 2 * pad
    H = int((y1 - y0) * px) + 2 * pad
    im = Image.new("L", (max(W, 60), max(H, 60)), 255)
    d = ImageDraw.Draw(im)

    def T(p):
        return (pad + (p[0] - x0) * px, pad + (y1 - p[1]) * px)

    # baseline (y=0) and x-height line (y=1) for reference
    for yv, shade in ((0.0, 170), (1.0, 215)):
        yy = pad + (y1 - yv) * px
        if -5 < yy < im.height + 5:
            d.line([(0, yy), (im.width, yy)], fill=shade)
    for i, s in enumerate(strokes):
        pts = [T(p) for p in s]
        if len(pts) >= 2:
            d.line(pts, fill=0, width=3, joint="curve")
        else:
            d.ellipse([pts[0][0] - 2, pts[0][1] - 2, pts[0][0] + 2, pts[0][1] + 2],
                      fill=0)
        d.text((pts[0][0] - 4, pts[0][1] - 14), str(i + 1), fill=110)
    return im


def main(author, chars, maxV=6):
    prof = SY.LoadAllProfiles()[author]
    lib = prof["glyphs"]
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    rows = []
    for ch in chars:
        vs = lib.get(ch, [])[:maxV]
        if not vs:
            print(f"  {ch!r}: no variants")
            continue
        cells = [render_variant(g) for g in vs]
        rows.append((ch, vs, cells))
        print(f"  {ch!r}: {len(lib.get(ch, []))} variants stored, showing {len(vs)}")

    if not rows:
        return
    cellH = max(c.height for _ch, _vs, cs in rows for c in cs)
    W = 90 + max(sum(c.width + 10 for c in cs) for _ch, _vs, cs in rows)
    H = sum(cellH + 34 for _ in rows) + 10
    sheet = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(sheet)
    y = 6
    for ch, vs, cells in rows:
        d.text((6, y + cellH // 2), f"'{ch}'", fill=0, font=font)
        x = 90
        for g, c in zip(vs, cells):
            sheet.paste(c, (x, y))
            d.text((x, y + cellH + 2),
                   f"pD{g.get('priorD', 0):.2f} w{g.get('width', 0):.2f} "
                   f"n{len(g['strokes'])}", fill=90, font=font)
            x += c.width + 10
        y += cellH + 34
    p = OUT_DIR / f"glyphs_{author}.png"
    sheet.save(p)
    print(f"-> {p}  {sheet.size}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("author")
    ap.add_argument("chars")
    ap.add_argument("--max", type=int, default=6)
    args = ap.parse_args()
    main(args.author, list(args.chars), maxV=args.max)
