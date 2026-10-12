"""Render an author's stored glyph library as a contact sheet -- every
variant synthesis can actually draw from, after the profile's own filtering.
The quickest way to see whether extraction produced real letterforms.

    python SegmentationBest/show_glyphs.py              # every author
    python SegmentationBest/show_glyphs.py yeukita owen # just these
    python SegmentationBest/show_glyphs.py --chars aeiou
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

THIS_DIR = Path(__file__).resolve().parent
CNN_DIR = THIS_DIR.parent
PROFILE_DIR = CNN_DIR / "NOGIT" / "StyleProfiles10"
OUT_DIR = THIS_DIR / "GlyphSheets"

PX = 34            # pixels per x-height
CELL_W, CELL_H = 78, 112
BASE_Y = 76        # baseline within a cell
LBL = 30           # left gutter for the character label


def draw_variant(d, glyph, ox, oy):
    d.line([ox, oy + BASE_Y, ox + CELL_W - 3, oy + BASE_Y], fill=(255, 205, 205))
    d.line([ox, oy + BASE_Y - PX, ox + CELL_W - 3, oy + BASE_Y - PX], fill=(212, 228, 255))
    for stroke in glyph["strokes"]:
        pts = [(ox + 8 + px * PX, oy + BASE_Y - py * PX) for (px, py) in stroke]
        if len(pts) >= 2:
            d.line(pts, fill=(0, 0, 0), width=2, joint="curve")


def sheet(author, chars=None, max_cols=10):
    path = PROFILE_DIR / f"{author}.json"
    if not path.exists():
        print(f"[skip] {author}: no profile at {path}")
        return None
    glyphs = json.load(open(path, encoding="utf-8"))["glyphs"]
    if chars:
        glyphs = {c: v for c, v in glyphs.items() if c in chars}
    if not glyphs:
        print(f"[skip] {author}: no matching characters")
        return None
    # letters first (lower, then upper), then digits and punctuation
    order = sorted(glyphs, key=lambda c: (not c.isalpha(), not c.islower(), c))
    ncol = min(max_cols, max(len(v) for v in glyphs.values()))
    img = Image.new("RGB", (LBL + CELL_W * ncol, CELL_H * len(order)), "white")
    d = ImageDraw.Draw(img)
    for r, ch in enumerate(order):
        oy = r * CELL_H
        d.text((5, oy + BASE_Y - 24), ch, fill=(200, 0, 0))
        d.text((5, oy + BASE_Y - 8), str(len(glyphs[ch])), fill=(160, 160, 160))
        for c, glyph in enumerate(glyphs[ch][:ncol]):
            draw_variant(d, glyph, LBL + c * CELL_W, oy)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{author}.png"
    img.save(out)
    print(f"{author}: {len(order)} chars, {sum(len(v) for v in glyphs.values())} variants -> {out}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("authors", nargs="*", help="author ids (default: all with a profile)")
    p.add_argument("--chars", default=None, help="only these characters, e.g. aeiou")
    args = p.parse_args()
    authors = args.authors or [p.stem for p in sorted(PROFILE_DIR.glob("*.json"))]
    for a in authors:
        sheet(a, set(args.chars) if args.chars else None)


if __name__ == "__main__":
    main()
