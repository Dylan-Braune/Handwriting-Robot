"""
novel_test.py -- DiffBrush on sentences that appear NOWHERE in IAM
(present-day phrasing, so the model has never seen this text in any hand).

    python novel_test.py            # default authors + sentences
    python novel_test.py 150 151 384 551
"""
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw

import _env
from _env import OUT_DIR
from htg_backend import DiffBrushBackend
from reproduce_v2 import (load_text_model, load_shape_model, reproduce,
                          holdout_refs, ctc_read, levenshtein)

NOVEL = [
    "The wifi password is taped under the router",
    "She uploaded the photos before boarding the flight",
    "My phone battery died during the video call",
    "Please scan the QR code at the checkout counter",
    "We debugged the app until three in the morning",
    "The drone filmed the coastline at sunset",
]


def fit(im, h=58, maxw=1500):
    s = h / im.height
    im = im.resize((max(1, int(im.width * s)), h), Image.Resampling.LANCZOS)
    return im.crop((0, 0, min(maxw, im.width), h))


def main():
    authors = sys.argv[1:] or ["150", "151", "153", "551"]
    sents = NOVEL[:3]
    device = torch.device("cpu")
    tm = load_text_model(device)
    sm, mp = load_shape_model(device)
    models = (tm, sm, mp)
    be = DiffBrushBackend(steps=26)

    rows = []          # (label, image)
    for a in authors:
        refs = holdout_refs(a, 3)
        strip = sorted(refs, key=lambda i: -i.width)[0] if refs else None
        rows.append((f"{a}  REAL (for style ref)", strip))
        for s in sents:
            r = reproduce(s, a, backend=be, k=2, device=device,
                          _models=models, verbose=False)
            cer = r["cer"]
            print(f"{a} | CER {cer:5.1%} | saw: {r['pred']!r}")
            rows.append((f"{a}  “{s}”", r["image"]))

    W = 540 + max(fit(im).width for _, im in rows if im is not None) + 20
    H = sum(70 for _ in rows) + 16
    sh = Image.new("L", (W, H), 248)
    d = ImageDraw.Draw(sh)
    y = 8
    for lbl, im in rows:
        d.text((8, y + 4), lbl[:90], fill=60)
        if im is not None:
            sh.paste(fit(im), (540, y))
        y += 70
    p = OUT_DIR / "novel_test.png"
    sh.save(p)
    print("\nsaved", p)


if __name__ == "__main__":
    main()
