"""Inspect the RAW glyphs of a re-extracted cache, before the profile's
quality filters, and compare against what the filtered profile keeps.

    python raw_check.py yeukita i,l,a,o
"""
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: F401

import numpy as np
from PIL import Image, ImageDraw

author = sys.argv[1] if len(sys.argv) > 1 else "yeukita"
letters = (sys.argv[2] if len(sys.argv) > 2 else "i,l,a,o").split(",")
N = int(sys.argv[3]) if len(sys.argv) > 3 else 12

pkl = HERE / "out" / "reextract" / author / f"{author}.pkl"
with open(pkl, "rb") as f:
    parsed = pickle.load(f)

raw = {}
for glyphs, _st in parsed:
    for g in glyphs:
        if g and g.get("char", " ") != " ":
            raw.setdefault(g["char"], []).append(g)

CW, CH, SC = 108, 145, 56
for ch in letters:
    vs = raw.get(ch, [])[:N]
    if not vs:
        print(f"{ch!r}: none")
        continue
    # structural check: how many of the first N are a single piece
    img = Image.new("RGB", (CW * min(N, 6), CH * ((len(vs) + 5) // 6) + 20),
                    (252, 252, 250))
    d = ImageDraw.Draw(img)
    d.text((4, 3), f"{ch!r}  raw: {len(raw.get(ch, []))} variants "
                   f"(showing {len(vs)})", fill=(0, 0, 0))
    for i, g in enumerate(vs):
        cx = (i % 6) * CW
        cy = 20 + (i // 6) * CH
        base = cy + 0.74 * CH
        xs = [p[0] for s in g["strokes"] for p in s]
        ys = [p[1] for s in g["strokes"] for p in s]
        if not xs:
            continue
        ox = cx + 8 - min(xs) * SC
        d.rectangle([cx + 1, cy + 1, cx + CW - 2, cy + CH - 2], outline=(230, 230, 230))
        d.line([cx + 2, base, cx + CW - 3, base], fill=(170, 190, 235))
        d.line([cx + 2, base - SC, cx + CW - 3, base - SC], fill=(236, 236, 236))
        for si, s in enumerate(g["strokes"]):
            if len(s) < 2:
                continue
            q = [(ox + x * SC, base - y2 * SC) for (x, y2) in s]
            d.line(q, fill=(0, 0, 0) if si == 0 else (200, 60, 30), width=2)
        d.text((cx + 4, cy + 3), f"{len(g['strokes'])}s "
                                 f"h{g['top']-g['bot']:.1f} w{g['width']:.1f}",
               fill=(130, 130, 130))
    out = HERE / "out" / f"raw_{author}_{ch}.png"
    img.save(out)
    print(f"{ch!r}: {len(raw.get(ch, []))} raw variants -> wrote {out}")
