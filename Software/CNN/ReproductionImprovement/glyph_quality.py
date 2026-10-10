"""Quantify how clean the stored glyphs are, per author.

Ground-truth-free but objective, and it is what a cleaner cut has to improve:

  height   a letter's stored ink height in x-heights. An 'e' or an 'n' is
           x-height only, so a big value means the cut swallowed a neighbouring
           ascender/descender or a piece of a tall letter.
  width    stored glyph width in x-heights. Print letters run ~0.6-1.2; two or
           three x-heights means the cell covers more than one letter.
  multi    fraction of the glyph's columns carrying 2+ separate ink runs, i.e.
           ink from more than one stroke. High on a fragment or a wide cut.

Usage: python glyph_quality.py <author> [--compare-stamp <stamp>]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths

import numpy as np
from PIL import Image

XONLY = set("acemnorsuvwxz")
ASCEND = set("bdfhklt")
DESCEND = set("gjpqy")


def rasterise(g, pad=4):
    """Rasterise one glyph's strokes into a small boolean mask (x right, y up
    from the baseline), so its real ink geometry can be measured."""
    xs = [p[0] for s in g["strokes"] for p in s]
    ys = [p[1] for s in g["strokes"] for p in s]
    if not xs:
        return None
    S = 20.0                                   # px per x-height
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    W = int((x1 - x0) * S) + 2 * pad + 2
    H = int((y1 - y0) * S) + 2 * pad + 2
    im = Image.new("L", (W, H), 0)
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    for s in g["strokes"]:
        if len(s) < 2:
            continue
        q = [(pad + (x - x0) * S, H - pad - (y - y0) * S) for (x, y) in s]
        d.line(q, fill=255, width=2)
    return np.asarray(im) > 0


def multi_run_fraction(mask):
    """Fraction of columns whose ink has 2+ separate vertical runs."""
    if mask is None or not mask.any():
        return float("nan")
    inc = mask.astype(np.int8)
    starts = np.zeros_like(inc)
    starts[0] = inc[0]
    starts[1:] = inc[1:] & (1 - inc[:-1])
    runs = starts.sum(axis=0)
    cols = mask.any(axis=0)
    if not cols.any():
        return float("nan")
    return float((runs[cols] > 1).mean())


def ok_vertical(ch, top, bot):
    if ch in XONLY:
        return top <= 1.6 and bot >= -0.6
    if ch in ASCEND:
        return bot >= -0.6
    if ch in DESCEND:
        return top <= 1.7
    return True


def measure(lib):
    rows = []
    for ch, vs in lib.items():
        if ch.isspace() or not vs:
            continue
        for g in vs:
            h = g["top"] - g["bot"]
            w = g["width"]
            m = rasterise(g)
            rows.append(dict(ch=ch, h=h, w=w, nstroke=len(g["strokes"]),
                             multi=multi_run_fraction(m),
                             bad=not ok_vertical(ch, g["top"], g["bot"]),
                             priorD=g.get("priorD", float("nan"))))
    return rows


def report(name, lib, quiet=False):
    rows = measure(lib)
    if not rows:
        print(f"{name}: no glyphs")
        return None
    h = np.array([r["h"] for r in rows])
    w = np.array([r["w"] for r in rows])
    mu = np.array([r["multi"] for r in rows])
    bad = np.array([r["bad"] for r in rows])
    pd = np.array([r["priorD"] for r in rows])
    s = dict(n=len(rows), letters=len([c for c in lib if not c.isspace()]),
             medH=float(np.median(h)), p90H=float(np.percentile(h, 90)),
             maxH=float(h.max()), medW=float(np.median(w)),
             p90W=float(np.percentile(w, 90)), badFrac=float(bad.mean()),
             medMulti=float(np.nanmedian(mu)), medPriorD=float(np.nanmedian(pd)),
             fracH_gt2=float((h > 2.0).mean()),
             fracW_gt1_8=float((w > 1.8).mean()))
    if not quiet:
        print(f"{name}: {s['n']} variants / {s['letters']} letters")
        print(f"   height (xh)     median {s['medH']:.2f}  p90 {s['p90H']:.2f}"
              f"  max {s['maxH']:.2f}   over 2.0: {s['fracH_gt2']*100:.1f}%")
        print(f"   width  (xh)     median {s['medW']:.2f}  p90 {s['p90W']:.2f}"
              f"                          over 1.8: {s['fracW_gt1_8']*100:.1f}%")
        print(f"   vert-impossible {s['badFrac']*100:.1f}%   "
              f"multi-run cols {s['medMulti']*100:.1f}%   "
              f"priorD {s['medPriorD']:.3f}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("author")
    ap.add_argument("--compare-stamp", default=None,
                    help="compare against out/backup/<author>_<stamp>/<author>.json")
    args = ap.parse_args()

    after = paths.CNN / "NOGIT" / "StyleProfiles10" / f"{args.author}.json"
    lib_a = json.loads(after.read_text(encoding="utf-8"))["glyphs"]
    print("=" * 62)
    a = report("AFTER ", lib_a)

    if args.compare_stamp:
        bdir = HERE / "out" / "backup" / f"{args.author}_{args.compare_stamp}"
        before = bdir / f"{args.author}.json"
        if before.exists():
            lib_b = json.loads(before.read_text(encoding="utf-8"))["glyphs"]
            print("-" * 62)
            b = report("BEFORE", lib_b)
            print("-" * 62)
            print(f"{'metric':>18} {'before':>9} {'after':>9}  {'change':>9}")
            for k in ("n", "letters", "medH", "p90H", "maxH", "medW", "p90W",
                      "badFrac", "medMulti", "medPriorD"):
                bv, av = b[k], a[k]
                print(f"{k:>18} {bv:>9.3f} {av:>9.3f}  {av-bv:>+9.3f}")


if __name__ == "__main__":
    main()
