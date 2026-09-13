"""Render one author's alphabet in their OWN hand (legibility 0), each
letter shown both alone and inside a short word, so it can be judged by eye
which letters are actually unreadable for that writer.

Letters are rendered through the normal synthesis path, so what is shown is
exactly what the writer would produce.
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import _env
from _env import OUT_DIR

import SynthesizeHandwriting as SY

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def main(author, lam=0.0, cols=7, px=34):
    prof = SY.LoadAllProfiles()[author]
    SY._STYLE_CAL.clear()
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    cells = []
    for i, ch in enumerate(LETTERS):
        tr = SY.SynthesizeText(ch, prof, mmPerXh=4.0, seed=5 + i,
                               lineWidthMm=10_000.0, legibility=lam)
        im = SY.RenderTrajectory(tr, pxPerMm=px, profile=prof, uniformInk=True)
        cells.append((ch, im.convert("L")))

    cw = max(c.width for _c, c in cells) + 18
    chh = max(c.height for _c, c in cells) + 30
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("L", (cols * cw + 10, rows * chh + 30), 255)
    d = ImageDraw.Draw(sheet)
    d.text((6, 4), f"author {author} -- own letterforms (legibility {lam})",
           fill=0, font=font)
    for i, (ch, im) in enumerate(cells):
        r, c = divmod(i, cols)
        x, y = 8 + c * cw, 26 + r * chh
        d.text((x, y), ch, fill=120, font=font)
        sheet.paste(im, (x + 16, y))
    p = OUT_DIR / f"alphabet_{author}.png"
    sheet.save(p)
    print(f"-> {p}  {sheet.size}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("author")
    ap.add_argument("--lam", type=float, default=0.0)
    args = ap.parse_args()
    main(args.author, lam=args.lam)
