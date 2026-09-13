"""
mystyle_test.py -- DiffBrush on the NON-DATASET handwriting in the holdout
pages. Takes a photographed personal page, segments + cleans one strip of
its writing, and asks DiffBrush to write a fresh sentence in that hand.

    python mystyle_test.py
"""
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw

import _env
from _env import NOGIT_DIR, OUT_DIR
import SegmentPage as SEG
from diffbrush_infer import DiffBrush

PAGES = {
    "baseline_model (round print)":   NOGIT_DIR / "holdout_test_pages" / "baseline_model.png",
    "gantry_planning (scratchy slant)": NOGIT_DIR / "holdout_test_pages" / "gantry_planning.png",
    "repeatability_notes (loose round)": NOGIT_DIR / "holdout_test_pages" / "repeatability_notes.jpg",
}
TARGET = "The quiet harbour filled slowly with morning light"


def clean_crop(arr):
    """raw line crop (grayscale, illumination-corrected) -> clean 64px strip,
    white paper / dark ink, matching IAM64 style-crop convention."""
    g = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    g = g.astype(np.uint8)
    # Otsu; ink should end up dark on white
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() < 127:                      # ink is the majority -> invert
        bw = 255 - bw
    # light open to kill speckle, then tighten to the ink
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ys, xs = np.where(bw < 128)
    if len(xs) < 20:
        return None
    bw = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h = 64
    s = h / bw.shape[0]
    bw = cv2.resize(bw, (max(1, int(bw.shape[1] * s)), h), interpolation=cv2.INTER_AREA)
    return Image.fromarray(bw).convert("L")


def build_strip(page_path, want_w=760):
    results, _, _ = SEG.ProcessPage(str(page_path))
    texts = [r for r in results if r["tag"] == "TEXT"]
    texts.sort(key=lambda r: -(r["raw_crop"].shape[1]))     # widest first
    parts, w = [], 0
    for r in texts[:4]:
        c = clean_crop(r["raw_crop"])
        if c is None:
            continue
        parts.append(c)
        w += c.width + 24
        if w >= want_w:
            break
    if not parts:
        return None
    W = sum(p.width for p in parts) + 24 * (len(parts) - 1)
    strip = Image.new("L", (min(W, 1024), 64), 255)
    x = 0
    for p in parts:
        if x >= strip.width:
            break
        strip.paste(p, (x, 0))
        x += p.width + 24
    return strip


def fit(im, h=58, maxw=1500):
    s = h / im.height
    im = im.resize((max(1, int(im.width * s)), h), Image.Resampling.LANCZOS)
    return im.crop((0, 0, min(maxw, im.width), h))


def main():
    device = torch.device("cpu")
    model = DiffBrush(device="cpu", steps=30)

    rows = []
    for name, path in PAGES.items():
        print(f"[{name}] segmenting {path.name} ...")
        strip = build_strip(path)
        if strip is None:
            print("  no usable strip"); continue
        strip.save(OUT_DIR / f"_mystyle_strip_{path.stem}.png")
        outs = []
        for seed in (0, 1):
            outs.append(model.generate_line(TARGET, strip, seed=seed))
        rows.append((name, strip, outs))
        print(f"  done ({strip.size})")

    H = sum(3 * 66 + 20 for _ in rows) + 20
    sheet = Image.new("L", (1650, H), 248)
    d = ImageDraw.Draw(sheet)
    y = 10
    for name, strip, outs in rows:
        d.text((8, y + 6), name, fill=40)
        d.text((8, y + 24), f"“{TARGET}”", fill=110)
        d.text((150, y + 4), "real strip (cleaned)", fill=110)
        sheet.paste(fit(strip), (330, y))
        for i, o in enumerate(outs):
            d.text((150, y + 66 + i * 66 + 4), f"DiffBrush #{i+1}", fill=110)
            sheet.paste(fit(o), (330, y + 66 + i * 66))
        d.line([(0, y + 3 * 66 + 12), (1650, y + 3 * 66 + 12)], fill=210)
        y += 3 * 66 + 20
    p = OUT_DIR / "mystyle_test.png"
    sheet.save(p)
    print("\nsaved", p)


if __name__ == "__main__":
    main()
