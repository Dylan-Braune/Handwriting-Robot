"""Render the same sentence from the PRE-CHANGE profiles and the current
ones, stacked per author, so the two can be compared by eye.

The pre-change profiles carry neither `selfD` nor `ligSag`, so the current
synthesis code falls back to exactly the original behaviour when it loads
them (rank by the cross-author prior, generic ligature curve).
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import _env
from _env import OUT_DIR, NOGIT_DIR

import SynthesizeHandwriting as SY

TEXT = "the quick brown fox jumps over the lazy dog"


def render_all(profileDir, text, seed, pxPerMm):
    SY.PROFILE_DIR = Path(profileDir)
    SY._STYLE_CAL.clear()
    profiles = SY.LoadAllProfiles()
    out = {}
    for a in sorted(profiles):
        prof = profiles[a]
        traj = SY.SynthesizeText(text, prof, mmPerXh=4.0, seed=seed,
                                 lineWidthMm=10_000.0,
                                 legibility=prof.get("legibilityLambda", 0.0))
        out[a] = SY.RenderTrajectory(traj, pxPerMm=pxPerMm, profile=prof,
                                     uniformInk=True)
    return out


def main(text=TEXT, seed=0, pxPerMm=20.0, authors=None):
    before = render_all(NOGIT_DIR / "StyleProfiles10_prebackup", text, seed, pxPerMm)
    after = render_all(NOGIT_DIR / "StyleProfiles10", text, seed, pxPerMm)
    keys = sorted(set(before) & set(after))
    if authors:
        keys = [a for a in keys if a in authors]

    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    rowH, labelW, pad = 74, 96, 6
    rows = []
    for a in keys:
        for tag, im in (("BEFORE", before[a]), ("AFTER ", after[a])):
            s = rowH / max(1, im.height)
            rows.append((a, tag, im.convert("L").resize(
                (max(1, int(im.width * s)), rowH), Image.LANCZOS)))
    W = labelW + max(r[2].width for r in rows) + 16
    H = sum(rowH + 6 for _ in rows) + len(keys) * 10 + pad
    sheet = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(sheet)
    y = pad
    for i, (a, tag, im) in enumerate(rows):
        d.text((4, y + rowH // 2 - 8), f"{a} {tag}", fill=0, font=font)
        sheet.paste(im, (labelW, y))
        y += rowH + 6
        if tag == "AFTER ":
            y += 10
            d.line([(0, y - 6), (W, y - 6)], fill=200)
    p = OUT_DIR / "m2_before_after.png"
    sheet.save(p)
    print(f"-> {p}  {sheet.size}")

    for a in keys:
        for tag, src in (("before", before), ("after", after)):
            src[a].save(OUT_DIR / f"ba_{a}_{tag}.png")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--authors", default=None)
    args = ap.parse_args()
    main(text=args.text, seed=args.seed,
         authors=args.authors.split(",") if args.authors else None)
