"""
compare_exemplar.py -- first-principles fragment-collage vs DiffBrush.

Per author: real held-out line  |  exemplar (best-of-k)  |  DiffBrush
with each row's CER against the target text (frozen PaperCRNN reader).

    python compare_exemplar.py 150 153 384
"""
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw

import _env
from _env import OUT_DIR, DATA_DIR
from TrainText import IAMLineDatasetRaw, _decode_png, levenshtein
import BuildStyleProfile as SP
from exemplar import library_for, synthesize_best, load_text_model, _read_line
from htg_backend import DiffBrushBackend


def real_line(aid):
    ds = IAMLineDatasetRaw(root_dir=str(DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    for s in ds.samples:
        if s["page_key"].split("/")[0] == aid and s["is_holdout"] and \
                18 <= len(s["text"]) <= 46:
            return _decode_png(s["image_png"]).convert("L"), s["text"]
    return None, None


def fit(im, h=60):
    s = h / im.height
    return im.resize((max(1, int(im.width * s)), h), Image.Resampling.LANCZOS)


def main():
    authors = sys.argv[1:] or ["150", "153", "384"]
    dev = torch.device("cpu")
    tm = load_text_model(dev)
    db = DiffBrushBackend(steps=26)

    rows = []
    for a in authors:
        real, text = real_line(a)
        if real is None:
            print("no line for", a); continue
        print(f"\n=== {a}: {text!r}")
        lib = library_for(a, tm, dev)
        ex_im, ex_pred, ex_cer = synthesize_best(text, lib, tm, dev, k=8)
        strip = sorted([real], key=lambda i: -i.width)[0]
        db_im = db._model.generate_line(text, strip, seed=0)
        db_cer = levenshtein(_read_line(tm, dev, db_im).lower(), text.lower()) / max(1, len(text))
        real_cer = levenshtein(_read_line(tm, dev, real).lower(), text.lower()) / max(1, len(text))
        print(f"  real CER {real_cer:.0%} | exemplar CER {ex_cer:.0%} "
              f"({len(lib['frags'])} frags) | diffbrush CER {db_cer:.0%}")
        rows.append((a, text, [("real", real, real_cer), ("exemplar", ex_im, ex_cer),
                               ("DiffBrush", db_im, db_cer)]))

    h = 60
    W = 260 + max(fit(im).width for _, _, cells in rows for _, im, _ in cells) + 20
    H = sum(3 * (h + 6) + 30 for _ in rows) + 12
    sh = Image.new("L", (W, H), 248)
    d = ImageDraw.Draw(sh)
    y = 8
    for a, text, cells in rows:
        d.text((6, y), f"{a}   “{text}”", fill=40)
        y += 16
        for lbl, im, cer in cells:
            d.text((6, y + h // 2 - 4), f"{lbl}  {cer:.0%}", fill=90)
            sh.paste(fit(im), (250, y))
            y += h + 6
        y += 14
    p = OUT_DIR / "compare_exemplar.png"
    sh.save(p)
    print("\nsaved", p)


if __name__ == "__main__":
    main()
