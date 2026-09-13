"""At anchor 0, knock out ONE measured style feature at a time.

Nothing generic is mixed in at legibility 0, so every row here is built
entirely from the author's own extracted strokes. Each row replaces exactly
one measured value with a neutral one, so whatever changes between that row
and the baseline is what that feature was contributing.
"""
import argparse
import copy

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import _env
from _env import OUT_DIR

import SynthesizeHandwriting as SY
import BuildStyleProfile as SP
from TrainText import IAMLineDatasetRaw, _decode_png

TEXT = "the quick brown fox jumps over the lazy dog"


def _uniform_advance(prof):
    p = copy.deepcopy(prof)
    med = float(np.median([v for v in p.get("letterAdvance", {}).values()])
                or 0.9)
    for ch, vs in p["glyphs"].items():
        for g in vs:
            g["advance"] = med
            g["lead"] = 0.0
    return p


def _flat_reach(prof):
    p = copy.deepcopy(prof)
    p["ascender"], p["descender"] = 1.7, -0.6
    p["ascMeasRef"], p["descMeasRef"] = 1.7, -0.6
    return p


# name -> function returning a modified copy of the profile
ABLATIONS = [
    ("baseline (all theirs)", lambda p: p),
    ("slant -> 0",            lambda p: {**copy.deepcopy(p), "slantDeg": 0.0,
                                         "slantMeasRef": 0.0}),
    ("no joins",              lambda p: {**copy.deepcopy(p), "connectedness": 0.0}),
    ("always join",           lambda p: {**copy.deepcopy(p), "connectedness": 1.0}),
    ("generic asc/desc",      _flat_reach),
    ("uniform letter width",  _uniform_advance),
    ("generic word gap",      lambda p: {**copy.deepcopy(p), "wordSpaceXh": 1.2}),
]


def main(author, text=TEXT, seed=0, px=19, out=None):
    prof = SY.LoadAllProfiles()[author]
    base = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    realIm = None
    for s in base.samples:
        if s["page_key"].split("/")[0] == author:
            realIm = _decode_png(s["image_png"]).convert("L")
            break
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    rows = []
    if realIm is not None:
        rows.append(("REAL ink", realIm))
    for name, fn in ABLATIONS:
        p = fn(prof)
        SY._STYLE_CAL.clear()
        tr = SY.SynthesizeText(text, p, mmPerXh=4.0, seed=seed,
                               lineWidthMm=10_000.0, legibility=0.0)
        rows.append((name, SY.RenderTrajectory(tr, pxPerMm=px, profile=p,
                                               uniformInk=True).convert("L")))
        print(f"  {name}")

    rowH, labelW, pad = 74, 176, 8
    sc = [(n, im.resize((max(1, int(im.width * rowH / max(1, im.height))), rowH),
                        Image.LANCZOS)) for n, im in rows]
    W = labelW + max(im.width for _n, im in sc) + 18
    H = len(sc) * (rowH + 8) + pad + 22
    sheet = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(sheet)
    d.text((6, 4), f"author {author} -- anchor 0%, one measured feature removed per row",
           fill=0, font=font)
    y = pad + 20
    for n, im in sc:
        d.text((6, y + rowH // 2 - 8), n, fill=0, font=font)
        sheet.paste(im, (labelW, y))
        y += rowH + 8
    p = OUT_DIR / (out or f"ablate_{author}.png")
    sheet.save(p)
    print(f"-> {p}")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("author")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main(args.author, out=args.out)
