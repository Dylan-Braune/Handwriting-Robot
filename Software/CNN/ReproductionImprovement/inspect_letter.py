"""Print the stroke-level detail of an author's stored i (and any other letter),
so it is clear whether the dot survived extraction.

    python inspect_letter.py yeukita i             # live profile
    python inspect_letter.py yeukita i --pkl out/reextract/yeukita/yeukita.pkl
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: F401

import SynthesizeHandwriting as SY

ap = argparse.ArgumentParser()
ap.add_argument("author")
ap.add_argument("letter")
ap.add_argument("--pkl", default="")
ap.add_argument("--max", type=int, default=10)
args = ap.parse_args()

if args.pkl:
    with open(HERE / args.pkl, "rb") as f:
        parsed = pickle.load(f)
    lib = {}
    for glyphs, _st in parsed:
        for g in glyphs:
            if g and g.get("char", " ") != " " and "strokes" in g:
                lib.setdefault(g["char"], []).append(g)
    src = args.pkl
else:
    prof = json.loads((SY.PROFILE_DIR / f"{args.author}.json").read_text(encoding="utf-8"))
    lib = prof["glyphs"]
    src = "live profile"

vs = lib.get(args.letter, [])
print(f"{args.author} {args.letter!r} from {src}: {len(vs)} variants")
print(f"{'#':>3} {'nstr':>5} {'glyph top':>9} {'bot':>6} {'w':>6}   strokes "
      f"(x0..x1, y0..y1)")
for i, g in enumerate(vs[:args.max]):
    print(f"{i:>3} {len(g['strokes']):>5} {g['top']:>9.2f} {g['bot']:>6.2f} "
          f"{g['width']:>6.2f}")
    for si, s in enumerate(g["strokes"]):
        xs = [p[0] for p in s]
        ys = [p[1] for p in s]
        print(f"      stroke {si}: {len(s):>3} pts  x {min(xs):>6.2f}..{max(xs):<6.2f} "
              f"y {min(ys):>6.2f}..{max(ys):<6.2f}")
