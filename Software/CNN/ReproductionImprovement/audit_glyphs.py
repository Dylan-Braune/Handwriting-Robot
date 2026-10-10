"""Which of an author's stored glyphs does the recognizer actually read as the
letter they are filed under?

A glyph rendered alone on a blank line reads as nothing even when it is a
perfect letter (the recognizer was trained on lines), so each variant is placed
INTO A CARRIER WORD: "non" for 'o', "nan" for 'a', and so on. The carrier is
scored with CTC forced alignment against the exact string, and the reported
number is the probability the recognizer assigns to the character slot holding
the variant.

    good   -> the glyph reads as its letter in context
    bad    -> it does not (a fragment, a cut that swallowed a neighbour, or a
              letterform the recognizer genuinely cannot read)

    python audit_glyphs.py yeukita
    python audit_glyphs.py yeukita --carrier n
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths  # noqa: F401

import numpy as np
import SynthesizeHandwriting as SY
from ProfileIO import CtcForcedAlign
from np_inference.text_model import (PaperCRNNNumpy, resize_line_image_fixed,
                                     tensor_from_resized, CHAR_TO_IDX)

VOWELS = set("aeiouy")
CARRIER_VOWEL = "n"      # "non"
CARRIER_CONS = "o"       # "ovo"

ap = argparse.ArgumentParser()
ap.add_argument("author")
ap.add_argument("--carrier", default=None, help="force one carrier letter")
ap.add_argument("--json", default=None, help="write per-variant scores here")
args = ap.parse_args()

prof = json.loads((SY.PROFILE_DIR / f"{args.author}.json").read_text(encoding="utf-8"))
lib = prof["glyphs"]
reader = PaperCRNNNumpy()
mmPerXh, pxPerMm = 4.0, 18.0


def carrier_for(ch):
    if args.carrier:
        return args.carrier
    return CARRIER_VOWEL if ch.lower() in VOWELS else CARRIER_CONS


def slot_prob(var, ch, carrier):
    """Render `carrier + variant + carrier` and return P(ch) on the middle slot."""
    text = carrier + ch + carrier
    # a one-word mini profile: the carrier glyph comes from the author, the
    # middle letter is a single synthetic 'own' glyph
    mock = dict(prof)
    mock["glyphs"] = dict(lib)
    mock["glyphs"][carrier] = lib.get(carrier, [var])
    mock["glyphs"][ch] = [var]
    mock["legibilityLambda"] = 0.0          # keep the author's own shapes
    mock["connectedness"] = 0.0             # no joins: score the shape alone
    try:
        traj = SY.SynthesizeText(text, mock, mmPerXh=mmPerXh, seed=1,
                                 legibility=0.0)
        img = SY.RenderTrajectory(traj, pxPerMm=pxPerMm, profile=mock)
    except Exception as e:                       # pragma: no cover
        return float("nan"), f"render failed: {e}"
    lp = reader.forward(tensor_from_resized(resize_line_image_fixed(img)))[:, 0, :]
    res = CtcForcedAlign(lp, text)
    if res is None:
        return float("nan"), "no alignment"
    align, _conf = res
    if len(align) != 3:
        return float("nan"), f"aligned {len(align)} of 3"
    _c, span = align[1]
    if span is None:
        return 0.0, "empty slot"
    idx = CHAR_TO_IDX[ch]
    return float(np.exp(lp[span[0]:span[1], idx]).mean()), ""


print(f"author {args.author}: {len(lib)} letters, "
      f"{sum(len(v) for v in lib.values())} variants")
print(f"{'ch':>3} {'n':>3} {'medianP':>8} {'maxP':>7} {'good/kept':>10}  per-variant P")
report = {}
for ch in sorted(lib):
    if ch.isspace() or not lib[ch]:
        continue
    carrier = carrier_for(ch)
    if carrier == ch:
        carrier = "l" if ch != "l" else "n"
    ps = []
    for var in lib[ch]:
        p, _note = slot_prob(var, ch, carrier)
        ps.append(p)
    ps = np.array(ps, dtype=float)
    good = int((ps >= 0.5).sum())
    report[ch] = dict(n=len(ps), median=float(np.nanmedian(ps)),
                      max=float(np.nanmax(ps)), good=good,
                      ps=[None if np.isnan(p) else round(float(p), 3) for p in ps])
    flag = "" if good >= max(3, len(ps) // 2) else "   <-- MOSTLY BAD"
    print(f"{ch!r:>3} {len(ps):>3} {np.nanmedian(ps):>8.3f} {np.nanmax(ps):>7.3f} "
          f"{good:>4}/{len(ps):<5}  {[round(float(p),2) for p in ps]}{flag}")
    sys.stdout.flush()

if args.json:
    Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", args.json)
