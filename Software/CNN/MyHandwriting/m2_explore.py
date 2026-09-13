"""Vary one thing at a time from the baseline and render them all together.

Every row is the SAME sentence, author, and seed -- only the named knob
differs -- so any visible change is caused by that knob alone. The author's
real handwriting sits at the top of each sheet for the identity comparison.
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import _env
from _env import OUT_DIR

import SynthesizeHandwriting as SY
import BuildStyleProfile as SP
from TrainText import IAMLineDatasetRaw, _decode_png

TEXT = "the quick brown fox jumps over the lazy dog"

# name, legibility, jitter, {module attribute overrides}
# The round-3 winner is now the DEFAULT in SynthesizeHandwriting, so these
# rows only sweep the one setting left to choose per author: how much of the
# legible anchor to mix in.
# At a fixed anchor level, how far to trust the author's OWN ideal letterform
# as that anchor before falling back to the generic alphabet.
VARIANTS = [
    ("GENERIC anchor (old)",  0.90, 0.15, dict(IDEAL_ANCHOR_MAX_PRIORD=0.0)),
    ("REAL pick, gate .45",   0.90, 0.15, dict(IDEAL_SOURCE='medoid',
                                               IDEAL_ANCHOR_MAX_PRIORD=0.45)),
    ("REAL pick, always",     0.90, 0.15, dict(IDEAL_SOURCE='medoid',
                                               IDEAL_ANCHOR_MAX_PRIORD=9.0)),
    ("AVERAGED, always",      0.90, 0.15, dict(IDEAL_SOURCE='median',
                                               IDEAL_ANCHOR_MAX_PRIORD=9.0)),
    ("REAL pick + blend",     0.90, 0.15, dict(IDEAL_SOURCE='medoid',
                                               IDEAL_ANCHOR_MAX_PRIORD=9.0,
                                               BLEND_TOWARD_ANCHOR=True)),
    ("REAL+blend, anchor 50%", 0.50, 0.15, dict(IDEAL_SOURCE='medoid',
                                                IDEAL_ANCHOR_MAX_PRIORD=9.0,
                                                BLEND_TOWARD_ANCHOR=True)),
]

_DEFAULTS = {}


def _snapshot(keys):
    for k in keys:
        if k not in _DEFAULTS:
            _DEFAULTS[k] = getattr(SY, k)


def _apply(settings):
    allkeys = set()
    for _n, _l, _j, s in VARIANTS:
        allkeys |= set(s)
    _snapshot(allkeys)
    for k in allkeys:                      # reset everything each time
        setattr(SY, k, _DEFAULTS[k])
    for k, v in settings.items():
        setattr(SY, k, v)


def main(author, text=TEXT, seed=0, px=19, out=None):
    profs = SY.LoadAllProfiles()
    prof = profs[author]
    base = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    realIm = None
    for s in base.samples:
        if s["page_key"].split("/")[0] == author:
            realIm = _decode_png(s["image_png"]).convert("L")
            break

    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 17)
    except OSError:
        font = ImageFont.load_default()

    rows = []
    if realIm is not None:
        rows.append(("REAL ink", realIm))
    for name, lam, jit, settings in VARIANTS:
        _apply(settings)
        SY._STYLE_CAL.clear()
        tr = SY.SynthesizeText(text, prof, mmPerXh=4.0, seed=seed,
                               lineWidthMm=10_000.0, legibility=lam, jitter=jit)
        im = SY.RenderTrajectory(tr, pxPerMm=px, profile=prof, uniformInk=True)
        rows.append((name, im.convert("L")))
        print(f"  {name}")
    _apply({})

    rowH, labelW, pad = 76, 168, 8
    sc = []
    for name, im in rows:
        s = rowH / max(1, im.height)
        sc.append((name, im.resize((max(1, int(im.width * s)), rowH), Image.LANCZOS)))
    W = labelW + max(im.width for _n, im in sc) + 20
    H = len(sc) * (rowH + 8) + pad + 24
    sheet = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(sheet)
    d.text((6, 4), f"author {author} -- one knob changed per row", fill=0, font=font)
    y = pad + 20
    for name, im in sc:
        d.text((6, y + rowH // 2 - 8), name, fill=0, font=font)
        sheet.paste(im, (labelW, y))
        y += rowH + 8
    p = OUT_DIR / (out or f"explore_{author}.png")
    sheet.save(p)
    print(f"-> {p}  {sheet.size}")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("author")
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main(args.author, text=args.text, out=args.out)
