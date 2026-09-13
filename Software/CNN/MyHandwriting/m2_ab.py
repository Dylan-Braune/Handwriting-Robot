"""Hot/cold comparison harness.

For each author it renders the SAME sentence under several named variants and
stacks them under a strip of that author's REAL handwriting, so the two
things that matter can be judged together by eye:

  * is it still recognisably this person's hand?  (compare to the real strip)
  * can the words be read?                        (read the line)

No classifier is involved. Variants are described declaratively below so a
round is one edit and one run.
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


def _apply(settings):
    """settings: dict of SynthesizeHandwriting module attributes to set."""
    for k, v in settings.items():
        setattr(SY, k, v)


def main(authors, variants, text=TEXT, seed=0, px=19, out="m2_ab.png"):
    profs = SY.LoadAllProfiles()
    base = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    real = {}
    for s in base.samples:
        real.setdefault(s["page_key"].split("/")[0], s)

    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 17)
    except OSError:
        font = ImageFont.load_default()

    blocks = []
    for a in authors:
        prof = profs[a]
        rows = []
        if a in real:
            rows.append(("REAL ink", _decode_png(real[a]["image_png"]).convert("L")))
        for name, lam, settings in variants:
            _apply(settings)
            SY._STYLE_CAL.clear()
            tr = SY.SynthesizeText(text, prof, mmPerXh=4.0, seed=seed,
                                   lineWidthMm=10_000.0, legibility=lam)
            rows.append((name, SY.RenderTrajectory(tr, pxPerMm=px, profile=prof,
                                                   uniformInk=True).convert("L")))
        blocks.append((a, rows))

    rowH, labelW, pad, gap = 78, 150, 8, 26
    scaled = []
    for a, rows in blocks:
        sr = []
        for name, im in rows:
            s = rowH / max(1, im.height)
            sr.append((name, im.resize((max(1, int(im.width * s)), rowH),
                                       Image.LANCZOS)))
        scaled.append((a, sr))
    W = labelW + max(im.width for _a, sr in scaled for _n, im in sr) + 20
    H = sum(len(sr) * (rowH + 6) + gap for _a, sr in scaled) + pad
    sheet = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(sheet)
    y = pad
    for a, sr in scaled:
        for name, im in sr:
            d.text((6, y + rowH // 2 - 8), f"{a} {name}", fill=0, font=font)
            sheet.paste(im, (labelW, y))
            y += rowH + 6
        y += gap
        d.line([(0, y - gap // 2), (W, y - gap // 2)], fill=190)
    p = OUT_DIR / out
    sheet.save(p)
    print(f"-> {p}  {sheet.size}")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors", default="150,152,153")
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--out", default="m2_ab.png")
    args = ap.parse_args()

    # ---- round 1 -------------------------------------------------------
    # A: give up legibility for identity, on the existing dial.
    # B: buy legibility WITHOUT touching letterforms, by not drawing the
    #    inherited join tail at a word start.
    VARIANTS = [
        ("A more-identity  (lam .45)", 0.45, dict(CLEAN_WORD_START=False)),
        ("BASELINE         (lam .65)", 0.65, dict(CLEAN_WORD_START=False)),
        ("B cleaner-starts (lam .65)", 0.65, dict(CLEAN_WORD_START=True)),
    ]
    main(args.authors.split(","), VARIANTS, text=args.text, out=args.out)
