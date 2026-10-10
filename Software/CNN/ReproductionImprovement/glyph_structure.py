"""Is a stored glyph actually the letter it is filed under?

A stored glyph is synthesised from skeleton polylines, so it should be ONE
connected shape (or two for a dotted i/j or a crossed t). Anything else is
either a fragment of a letter (a cut that missed) or a cut that swallowed part
of a neighbour. This reports, per letter:

    parts   connected components of the stored strokes
    h, w    height / width in x-heights
    broken  parts > expected, or nothing connected at all

Expected parts: 1 for every letter, 2 for i j t x = (dot / crossbar), 3 for a
letter that is genuinely drawn in three pieces.

    python glyph_structure.py yeukita
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: F401

import numpy as np
from PIL import Image, ImageDraw
import SynthesizeHandwriting as SY

TWO_PART = set("ijtx=")
author = sys.argv[1]
prof = json.loads((SY.PROFILE_DIR / f"{author}.json").read_text(encoding="utf-8"))
lib = prof["glyphs"]
S = 24.0


def parts_of(g):
    """Count connected components of the glyph's own strokes."""
    xs = [p[0] for s in g["strokes"] for p in s]
    ys = [p[1] for s in g["strokes"] for p in s]
    if not xs:
        return 0, 0.0, 0.0
    pad = 3
    W = int((max(xs) - min(xs)) * S) + 2 * pad + 2
    H = int((max(ys) - min(ys)) * S) + 2 * pad + 2
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for s in g["strokes"]:
        if len(s) < 2:
            continue
        q = [(pad + (x - min(xs)) * S, H - pad - (y - min(ys)) * S) for (x, y) in s]
        d.line(q, fill=255, width=3)
    m = np.asarray(im) > 0
    seen = np.zeros_like(m, bool)
    n = 0
    for y0 in range(m.shape[0]):
        for x0 in np.nonzero(m[y0] & ~seen[y0])[0]:
            if seen[y0, x0]:
                continue
            n += 1
            st = [(y0, int(x0))]
            seen[y0, x0] = True
            while st:
                y, x = st.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1] \
                                and m[yy, xx] and not seen[yy, xx]:
                            seen[yy, xx] = True
                            st.append((yy, xx))
    return n, float(max(ys) - min(ys)), float(max(xs) - min(xs))


print(f"author {author}: {len(lib)} letters, "
      f"{sum(len(v) for v in lib.values())} variants")
print(f"{'ch':>3} {'n':>3} {'parts(median)':>13} {'maxParts':>9} {'medH':>6} "
      f"{'medW':>6} {'#broken':>8}")
bad_total = 0
for ch in sorted(lib):
    if ch.isspace() or not lib[ch]:
        continue
    exp = 2 if ch in TWO_PART else 1
    res = [parts_of(g) for g in lib[ch]]
    nparts = np.array([r[0] for r in res])
    hs = np.array([r[1] for r in res])
    ws = np.array([r[2] for r in res])
    broken = int((nparts > exp).sum() + (nparts == 0).sum())
    bad_total += broken
    flag = f"   <-- {broken} bad" if broken else ""
    print(f"{ch!r:>3} {len(lib[ch]):>3} {int(np.median(nparts)):>13} "
          f"{int(nparts.max()):>9} {np.median(hs):>6.2f} {np.median(ws):>6.2f} "
          f"{broken:>8}{flag}")
print(f"\ntotal variants whose shape is not one piece: {bad_total}")
