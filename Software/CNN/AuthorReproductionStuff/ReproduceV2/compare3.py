"""
compare3.py -- one sheet, three rows per author:
  real held-out line  |  legacy_synth (glyph library)  |  one_dm

    python compare3.py 153 155 150 384
"""
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw

import _env
from _env import OUT_DIR, DATA_DIR
from TrainText import IAMLineDatasetRaw, _decode_png
import BuildStyleProfile as SP
from htg_backend import LegacySynthBackend, OneDMBackend
from reproduce_v2 import load_text_model, load_shape_model, reproduce


def real_line(author_id):
    base = IAMLineDatasetRaw(root_dir=str(DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    for s in base.samples:
        if s["page_key"].split("/")[0] == author_id and s.get("is_holdout") \
                and 18 <= len(s["text"]) <= 44:
            return _decode_png(s["image_png"]).convert("L"), s["text"]
    return None, None


def fit(im, h=60, maxw=1400):
    s = h / im.height
    im = im.resize((max(1, int(im.width * s)), h), Image.Resampling.LANCZOS)
    return im.crop((0, 0, min(maxw, im.width), h))


def main():
    authors = sys.argv[1:] or ["153", "155", "150", "384"]
    device = torch.device("cpu")
    tm = load_text_model(device)
    sm, mp = load_shape_model(device)
    models = (tm, sm, mp)
    legacy = LegacySynthBackend()
    onedm = OneDMBackend(steps=25)

    rows = []
    for a in authors:
        real, text = real_line(a)
        if real is None:
            print("no held-out line for", a); continue
        print(f"{a}: {text!r}")
        rl = reproduce(text, a, backend=legacy, k=6, device=device,
                       _models=models, verbose=False)
        od = reproduce(text, a, backend=onedm, k=3, device=device,
                       _models=models, verbose=False)
        print(f"   legacy CER {rl['cer']:.0%} | one_dm CER {od['cer']:.0%}")
        rows.append((a, text, real, rl["image"], od["image"]))

    h = 60
    cols = [fit(r[2]) for r in rows] + [fit(r[3]) for r in rows] + [fit(r[4]) for r in rows]
    W = 150 + max(c.width for c in cols) + 16
    H = sum(3 * h + 46 for _ in rows) + 16
    sh = Image.new("L", (W, H), 245)
    d = ImageDraw.Draw(sh)
    y = 8
    for a, text, real, lg, od in rows:
        d.text((6, y + 24), str(a), fill=0)
        d.text((150, y), text[:80], fill=110)
        for i, (lbl, im) in enumerate([("real", real), ("glyph", lg), ("one-DM", od)]):
            d.text((6, y + 14 + i * h + h // 2), lbl, fill=90)
            sh.paste(fit(im), (150, y + 14 + i * h))
        d.line([(0, y + 3 * h + 22), (W, y + 3 * h + 22)], fill=200)
        y += 3 * h + 46
    p = OUT_DIR / "compare3.png"
    sh.save(p)
    print("saved", p)


if __name__ == "__main__":
    main()
