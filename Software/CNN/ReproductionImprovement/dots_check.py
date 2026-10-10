"""Where do the dots go? For i/j, report how often the stored glyph still has
its dot (a separate small stroke above the x-height band) versus a bare stem.

    python dots_check.py yeukita out/reextract/yeukita/yeukita.pkl
    python dots_check.py yeukita            # compares the live profile
"""
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: F401

import numpy as np
import SynthesizeHandwriting as SY

author = sys.argv[1]
src = sys.argv[2] if len(sys.argv) > 2 else None

if src:
    with open(HERE / src if not Path(src).is_absolute() else src, "rb") as f:
        parsed = pickle.load(f)
    lib = {}
    for glyphs, _st in parsed:
        for g in glyphs:
            if g and g.get("char", " ") != " " and "strokes" in g:
                lib.setdefault(g["char"], []).append(g)
    label = f"raw cache ({src})"
else:
    prof = json.loads((SY.PROFILE_DIR / f"{author}.json").read_text(encoding="utf-8"))
    lib = prof["glyphs"]
    label = "live profile"

print(f"{author} -- {label}")
print(f"{'ch':>3} {'n':>5} {'hasDot':>7} {'frac':>7}  {'median strokes':>14}")
for ch in ("i", "j"):
    vs = lib.get(ch, [])
    if not vs:
        print(f"{ch!r:>3}  none")
        continue
    hasdot = 0
    ns = []
    for g in vs:
        ns.append(len(g["strokes"]))
        # a dot = a stroke whose whole extent sits above the x-height and is tiny
        dot = False
        for s in g["strokes"]:
            if len(s) < 1:
                continue
            ys = [p[1] for p in s]
            xs = [p[0] for p in s]
            if min(ys) > 0.85 and (max(ys) - min(ys)) < 0.35 \
                    and (max(xs) - min(xs)) < 0.35:
                dot = True
        hasdot += dot
    print(f"{ch!r:>3} {len(vs):>5} {hasdot:>7} {hasdot/len(vs):>6.0%}  "
          f"{np.median(ns):>14.1f}")
