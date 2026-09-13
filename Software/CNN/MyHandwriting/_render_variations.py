"""One-off: render the same sentence several times, sampling among the top
real instances of each repeated fragment (see method1_exemplar._tile pool=3),
and stack the results into a single labelled comparison sheet."""
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import _env
from _env import OUT_DIR, LIB_DIR
from method1_exemplar import synthesize

TEXT = "the quick brown fox jumps over the lazy dog"
N = 6

lib = pickle.load(open(LIB_DIR / "method1_exemplar.pkl", "rb"))

try:
    label_font = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 16)
except OSError:
    label_font = ImageFont.load_default()

rows = []
for seed in range(N):
    im, stats = synthesize(TEXT, lib=lib, seed=seed, vary=True)
    rows.append((seed, im, stats))
    print(f"  seed={seed}: {im.size}  anchor={stats['n_anchor']} skip={stats['n_skip']}")

pad = 14
label_w = 70
maxW = max(im.width for _, im, _ in rows)
H = sum(im.height + pad for _, im, _ in rows) + pad
sheet = Image.new("L", (label_w + maxW + pad, H), 255)
draw = ImageDraw.Draw(sheet)
y = pad
for seed, im, stats in rows:
    draw.text((6, y + im.height // 2 - 8), f"#{seed}", fill=0, font=label_font)
    sheet.paste(im, (label_w, y))
    y += im.height + pad

out = OUT_DIR / "method1_variations.png"
sheet.save(out)
print(f"\n-> {out}  {sheet.size}")
