"""Audit stored glyphs by putting each variant into a REAL WORD and asking the
recognizer to read that word back.

The first version of this embedded each variant in a carrier like "oho"/"mnm",
which is an unnatural string and made good letters score badly. This version
uses a real word per letter ("the" for 'h', "man" for 'm', ...) and reports the
probability of the whole word being read correctly, with the variant in the
word's own position. A glyph that is really the letter lets the word read; a
fragment does not.

    python audit_glyphs2.py yeukita
    python audit_glyphs2.py yeukita --letters i,h,m
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

# letter -> (word, index of that letter in the word). Words chosen so the
# letter appears once, mostly as a lowercase x-height letter.
WORDS = {
    "a": ("man", 1), "b": ("bad", 0), "c": ("cat", 0), "d": ("dog", 0),
    "e": ("set", 1), "f": ("for", 0), "g": ("got", 0), "h": ("the", 1),
    "i": ("win", 1), "j": ("jam", 0), "k": ("key", 0), "l": ("let", 0),
    "m": ("man", 0), "n": ("not", 0), "o": ("for", 1), "p": ("put", 0),
    "q": ("que", 0), "r": ("red", 0), "s": ("set", 0), "t": ("the", 0),
    "u": ("put", 1), "v": ("van", 0), "w": ("was", 0), "x": ("fox", 1),
    "y": ("you", 0), "z": ("zoo", 0),
}

ap = argparse.ArgumentParser()
ap.add_argument("author")
ap.add_argument("--letters", default="")
ap.add_argument("--json", default="")
args = ap.parse_args()
only = set(args.letters.replace(",", "")) if args.letters else None

prof = json.loads((SY.PROFILE_DIR / f"{args.author}.json").read_text(encoding="utf-8"))
lib = prof["glyphs"]
reader = PaperCRNNNumpy()


def word_prob(var, ch):
    word, pos = WORDS[ch]
    mock = dict(prof)
    mock["glyphs"] = dict(lib)
    mock["glyphs"][ch] = [var]
    mock["legibilityLambda"] = 0.0
    mock["connectedness"] = 0.0
    try:
        traj = SY.SynthesizeText(word, mock, mmPerXh=4.0, seed=1, legibility=0.0)
        img = SY.RenderTrajectory(traj, pxPerMm=18.0, profile=mock)
    except Exception:
        return float("nan")
    lp = reader.forward(tensor_from_resized(resize_line_image_fixed(img)))[:, 0, :]
    res = CtcForcedAlign(lp, word)
    if res is None:
        return float("nan")
    align, _c = res
    if len(align) != len(word):
        return float("nan")
    span = align[pos][1]
    if span is None:
        return 0.0
    return float(np.exp(lp[span[0]:span[1], CHAR_TO_IDX[ch]]).mean())


print(f"author {args.author}  ({len(lib)} letters, "
      f"{sum(len(v) for v in lib.values())} variants)")
print(f"{'ch':>3} {'word':>6} {'n':>3} {'median':>7} {'good/n':>8}   per-variant P")
report = {}
for ch in sorted(lib):
    if ch.isspace() or not lib[ch] or ch not in WORDS:
        continue
    if only and ch not in only:
        continue
    word = WORDS[ch][0]
    ps = np.array([word_prob(v, ch) for v in lib[ch]], dtype=float)
    good = int(np.nansum(ps >= 0.5))
    report[ch] = dict(word=word, n=len(ps), median=float(np.nanmedian(ps)),
                      good=good,
                      ps=[None if np.isnan(p) else round(float(p), 3) for p in ps])
    flag = "" if good >= max(3, round(0.6 * len(ps))) else "   <-- BAD GLYPHS"
    print(f"{ch!r:>3} {word:>6} {len(ps):>3} {np.nanmedian(ps):>7.3f} "
          f"{good:>3}/{len(ps):<4}   {[round(float(p),2) for p in ps]}{flag}")
    sys.stdout.flush()

if args.json:
    Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", args.json)
