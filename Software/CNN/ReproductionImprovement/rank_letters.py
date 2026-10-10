"""Rank an author's letters by how far their stored glyphs sit from the
cross-author prototype of that letter (the `priorD` the profile already stores),
and show the worst variant of each worst letter, so the letters whose STORED
SHAPE is not actually the letter can be picked out fast.

priorD is a cosine distance in [0,1]: low = this looks like what ten authors
agree the letter looks like. The score was computed at build time against every
author's glyphs pooled, so a high value means the shape is unusual -- either the
author's own idiosyncrasy, or a cut that is not really that letter.

    python rank_letters.py yeukita [--top 12]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: F401

import numpy as np
from PIL import Image, ImageDraw
import SynthesizeHandwriting as SY

ap = argparse.ArgumentParser()
ap.add_argument("author")
ap.add_argument("--top", type=int, default=12)
ap.add_argument("--compare", default="",
                help="rank against this other author's profile as a control")
args = ap.parse_args()


def load(a):
    return json.loads((SY.PROFILE_DIR / f"{a}.json").read_text(encoding="utf-8"))


def rank(prof):
    rows = []
    for ch, vs in prof["glyphs"].items():
        if ch.isspace() or not vs:
            continue
        pd = np.array([g.get("priorD", 0.0) for g in vs], dtype=float)
        rows.append((float(np.median(pd)), float(pd.max()), ch, len(vs), pd))
    rows.sort(reverse=True)
    return rows


def sheet(prof, rows, top, path):
    CW, CH, SC = 100, 140, 54
    img = Image.new("RGB", (CW * 4, (CH + 20) * top), (252, 252, 250))
    d = ImageDraw.Draw(img)
    y = 0
    for med, mx, ch, n, pd in rows[:top]:
        vs = prof["glyphs"][ch]
        order = np.argsort(-pd)          # worst first
        d.text((4, y + 3), f"{ch!r}  median priorD {med:.2f}  worst {mx:.2f}",
               fill=(0, 0, 0))
        for i in range(min(4, n)):
            g = vs[int(order[i])]
            cx = i * CW
            base = y + 20 + 0.74 * CH
            xs = [p[0] for s in g["strokes"] for p in s]
            if not xs:
                continue
            ox = cx + 8 - min(xs) * SC
            d.rectangle([cx + 1, y + 19, cx + CW - 2, y + 20 + CH - 2],
                        outline=(230, 230, 230))
            d.line([cx + 2, base, cx + CW - 3, base], fill=(170, 190, 235))
            d.line([cx + 2, base - SC, cx + CW - 3, base - SC], fill=(235, 235, 235))
            for si, s in enumerate(g["strokes"]):
                if len(s) < 2:
                    continue
                q = [(ox + x * SC, base - y2 * SC) for (x, y2) in s]
                d.line(q, fill=(0, 0, 0) if si == 0 else (200, 60, 30), width=2)
            d.text((cx + 4, y + 22), f"{pd[int(order[i])]:.2f}", fill=(150, 0, 0))
        y += CH + 20
    img.save(path)
    return path


prof = load(args.author)
rows = rank(prof)
print(f"{args.author}: {len(rows)} letters, ranked by how far the stored shape "
      f"sits from the cross-author prototype of that letter\n")
print(f"{'ch':>4} {'variants':>9} {'medianD':>8} {'worstD':>7}")
for med, mx, ch, n, pd in rows[:args.top]:
    print(f"{ch!r:>4} {n:>9} {med:>8.3f} {mx:>7.3f}")
p = sheet(prof, rows, args.top, HERE / "out" / f"ranked_{args.author}.png")
print(f"\nwrote {p}  (4 worst variants of each, number = priorD)")

if args.compare:
    other = load(args.compare)
    rows2 = rank(other)
    print(f"\n--- control: {args.compare} (top {args.top}) ---")
    for med, mx, ch, n, pd in rows2[:args.top]:
        print(f"{ch!r:>4} {n:>9} {med:>8.3f} {mx:>7.3f}")
    sheet(other, rows2, args.top, HERE / "out" / f"ranked_{args.compare}.png")
