"""
METHOD 3 -- library route (existence proof).

Uses a pretrained styled-text diffusion model (DiffBrush, ICCV 2025) as a
black box, conditioned on ONE clean strip of the writer's real handwriting.
It generates a whole line in the writer's style in one pass. Zero training.

Point: show that reproducing arbitrary text in *your own* hand is achievable
in principle -- not only for the IAM authors the in-house models were built
on. This is NOT a from-first-principles method and does not run on the SBC;
it is the upper-bound reference the other two methods are compared against.

    python method3_library.py "some new sentence"

Requires NOGIT/_vendor/ (DiffBrush checkout + checkpoint + SD-1.5 VAE) --
see AuthorReproductionStuff/REVISED_METHOD.md section 7.
"""
import argparse
import sys

import numpy as np
import cv2
from PIL import Image

import _env
from _env import OUT_DIR, CNN_DIR, PAGES_DIR

sys.path.insert(0, str(CNN_DIR / "AuthorReproductionStuff" / "ReproduceV2"))
import SegmentPage as SEG


def clean_strip(want_w=760):
    """Widest clean line(s) from the writer's pages -> one 64px grey strip,
    white paper / dark ink (the convention DiffBrush's style encoder wants)."""
    parts, w = [], 0
    for page in sorted(PAGES_DIR.glob("page*.jpg")):
        results, _, _ = SEG.ProcessPage(str(page))
        crops = sorted((r["raw_crop"] for r in results if r["tag"] == "TEXT"),
                       key=lambda c: -c.shape[1])
        for c in crops:
            g = c if c.ndim == 2 else cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
            _, bw = cv2.threshold(g.astype(np.uint8), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if bw.mean() < 127:
                bw = 255 - bw
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            ys, xs = np.where(bw < 128)
            if len(xs) < 40:
                continue
            bw = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            s = 64 / bw.shape[0]
            bw = cv2.resize(bw, (max(1, int(bw.shape[1] * s)), 64),
                            interpolation=cv2.INTER_AREA)
            parts.append(Image.fromarray(bw).convert("L"))
            w += bw.shape[1] + 24
            if w >= want_w:
                break
        if w >= want_w:
            break
    if not parts:
        return None
    W = min(1024, sum(p.width for p in parts) + 24 * (len(parts) - 1))
    strip = Image.new("L", (W, 64), 255)
    x = 0
    for p in parts:
        if x >= W:
            break
        strip.paste(p, (x, 0))
        x += p.width + 24
    return strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?",
                    default="the quick brown fox jumps over the lazy dog")
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    strip = clean_strip()
    if strip is None:
        raise SystemExit("no usable strip from pages/")
    strip.save(OUT_DIR / "method3_stylestrip.png")

    from diffbrush_infer import DiffBrush
    model = DiffBrush(device="cpu", steps=args.steps)
    im = model.generate_line(args.text, strip, seed=0)
    out = OUT_DIR / "method3_sample.png"
    im.save(out)
    print(f"\n  '{args.text}'\n  -> {out}  {im.size}")


if __name__ == "__main__":
    main()
