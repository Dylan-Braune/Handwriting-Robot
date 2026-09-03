"""
SynthesizeHandwriting.py -- text + author ID -> pen trajectory in the
author's handwriting style.

Consumes the profiles built by BuildStyleProfile.py (glyph prototype library +
measured style parameters) and emits an ordered list of polylines with
explicit pen-up/pen-down structure, in millimetres, ready for WriteGCode.py.

How a word is built:
  * every character is realised from one of the author's stored VARIANTS of
    that glyph (chosen per instance, so repeated letters differ the way a
    real hand varies), falling back to a shape borrowed from a
    case/similar-letter neighbour and finally to a built-in skeleton font
    for characters the author never wrote;
  * letters are placed by the author's own measured per-character advance,
    scaled by x-height;
  * for a CURSIVE author (measured `connectedness`), consecutive letters in
    a word are joined by a ligature stroke from the previous glyph's exit
    point to the next glyph's entry point, and the two glyphs plus the
    ligature are emitted as ONE pen-down polyline -- the pen only lifts
    where that author actually lifts it;
  * per-instance jitter (sub-x-height position/rotation/scale noise, plus a
    slow baseline drift and a small random slant wobble) keeps repeated
    text from looking stamped;
  * the author's slant is applied as a shear at the end, so glyphs stay
    stored upright and shared spacing maths is slant-independent.

Public API:
    SynthesizeText(text, profile, mmPerXh=..., seed=...) -> Trajectory
    Trajectory.strokes -> list of [(xmm, ymm), ...] pen-down polylines
    RenderTrajectory(traj, ...) -> PIL image (same look as training crops)
"""

import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from BuildStyleProfile import LoadProfile, PROFILE_DIR   # noqa: F401

SCRIPT_DIR = Path(__file__).resolve().parent

# Fallback skeleton font: single-stroke letter shapes in the same
# normalized frame the profiles use (x right, y up from baseline, 1.0 =
# x-height). Only used for characters an author never wrote.
_FB = {
    'a': [[(0.75, 0.75), (0.45, 0.95), (0.15, 0.75), (0.1, 0.4), (0.35, 0.05),
           (0.65, 0.1), (0.78, 0.35), (0.78, 0.0)], [(0.78, 0.3), (0.95, 0.0)]],
    'b': [[(0.1, 1.8), (0.1, 0.0)], [(0.1, 0.55), (0.35, 0.85), (0.7, 0.7),
           (0.75, 0.3), (0.5, 0.0), (0.15, 0.1)]],
    'c': [[(0.8, 0.75), (0.5, 0.95), (0.18, 0.7), (0.15, 0.3), (0.45, 0.03),
           (0.82, 0.2)]],
    'd': [[(0.8, 1.8), (0.8, 0.0)], [(0.8, 0.7), (0.5, 0.95), (0.15, 0.7),
           (0.15, 0.3), (0.45, 0.03), (0.8, 0.25)]],
    'e': [[(0.15, 0.45), (0.8, 0.5), (0.65, 0.85), (0.3, 0.9), (0.12, 0.55),
           (0.2, 0.15), (0.55, 0.0), (0.85, 0.15)]],
    'f': [[(0.75, 1.75), (0.5, 1.85), (0.35, 1.55), (0.35, 0.0)],
          [(0.1, 0.95), (0.7, 0.95)]],
    'g': [[(0.8, 0.9), (0.8, -0.45), (0.5, -0.62), (0.2, -0.5)],
          [(0.8, 0.7), (0.5, 0.95), (0.15, 0.7), (0.18, 0.3), (0.5, 0.05),
           (0.8, 0.25)]],
    'h': [[(0.1, 1.8), (0.1, 0.0)], [(0.1, 0.6), (0.4, 0.9), (0.72, 0.7),
           (0.75, 0.0)]],
    'i': [[(0.2, 0.9), (0.2, 0.0)], [(0.2, 1.25), (0.2, 1.3)]],
    'j': [[(0.35, 0.9), (0.35, -0.45), (0.1, -0.6)], [(0.35, 1.25), (0.35, 1.3)]],
    'k': [[(0.1, 1.8), (0.1, 0.0)], [(0.72, 0.9), (0.1, 0.35)],
          [(0.32, 0.5), (0.75, 0.0)]],
    'l': [[(0.15, 1.8), (0.15, 0.12), (0.4, 0.0)]],
    'm': [[(0.05, 0.9), (0.05, 0.0)], [(0.05, 0.65), (0.28, 0.9), (0.5, 0.65),
           (0.5, 0.0)], [(0.5, 0.65), (0.72, 0.9), (0.95, 0.65), (0.95, 0.0)]],
    'n': [[(0.1, 0.9), (0.1, 0.0)], [(0.1, 0.62), (0.4, 0.92), (0.75, 0.68),
           (0.75, 0.0)]],
    'o': [[(0.45, 0.95), (0.15, 0.7), (0.15, 0.28), (0.45, 0.02),
           (0.8, 0.28), (0.8, 0.7), (0.45, 0.95)]],
    'p': [[(0.1, 0.9), (0.1, -0.6)], [(0.1, 0.6), (0.42, 0.9), (0.78, 0.66),
           (0.75, 0.25), (0.42, 0.0), (0.12, 0.15)]],
    'q': [[(0.82, 0.9), (0.82, -0.6)], [(0.82, 0.65), (0.5, 0.92),
           (0.15, 0.66), (0.2, 0.25), (0.5, 0.0), (0.82, 0.2)]],
    'r': [[(0.12, 0.9), (0.12, 0.0)], [(0.12, 0.6), (0.4, 0.9), (0.7, 0.85)]],
    's': [[(0.78, 0.82), (0.42, 0.95), (0.18, 0.75), (0.45, 0.5),
           (0.72, 0.38), (0.6, 0.05), (0.2, 0.1)]],
    't': [[(0.35, 1.45), (0.35, 0.15), (0.6, 0.0)], [(0.1, 0.95), (0.65, 0.95)]],
    'u': [[(0.1, 0.9), (0.1, 0.25), (0.4, 0.0), (0.72, 0.25), (0.72, 0.9)],
          [(0.72, 0.35), (0.72, 0.0)]],
    'v': [[(0.08, 0.9), (0.42, 0.0), (0.78, 0.9)]],
    'w': [[(0.05, 0.9), (0.25, 0.0), (0.48, 0.7), (0.7, 0.0), (0.92, 0.9)]],
    'x': [[(0.1, 0.9), (0.8, 0.0)], [(0.8, 0.9), (0.1, 0.0)]],
    'y': [[(0.1, 0.9), (0.1, 0.3), (0.42, 0.05), (0.75, 0.3), (0.75, 0.9)],
          [(0.75, 0.4), (0.75, -0.45), (0.45, -0.62), (0.18, -0.5)]],
    'z': [[(0.1, 0.9), (0.8, 0.9), (0.12, 0.05), (0.85, 0.05)]],
    '.': [[(0.15, 0.06), (0.2, 0.0)]],
    ',': [[(0.2, 0.1), (0.12, -0.22)]],
    "'": [[(0.15, 1.75), (0.1, 1.35)]],
    '"': [[(0.1, 1.75), (0.06, 1.4)], [(0.32, 1.75), (0.28, 1.4)]],
    '-': [[(0.05, 0.45), (0.6, 0.45)]],
    '!': [[(0.15, 1.75), (0.15, 0.3)], [(0.15, 0.06), (0.18, 0.0)]],
    '?': [[(0.08, 1.5), (0.3, 1.78), (0.6, 1.6), (0.55, 1.25), (0.32, 1.05),
           (0.32, 0.75)], [(0.3, 0.06), (0.33, 0.0)]],
    ':': [[(0.15, 0.86), (0.18, 0.8)], [(0.15, 0.06), (0.18, 0.0)]],
    ';': [[(0.15, 0.86), (0.18, 0.8)], [(0.2, 0.1), (0.12, -0.22)]],
    '(': [[(0.5, 1.8), (0.2, 1.0), (0.5, -0.3)]],
    ')': [[(0.15, 1.8), (0.45, 1.0), (0.15, -0.3)]],
    '0': [[(0.45, 1.7), (0.12, 1.3), (0.12, 0.4), (0.45, 0.0), (0.78, 0.4),
           (0.78, 1.3), (0.45, 1.7)]],
    '1': [[(0.15, 1.4), (0.42, 1.7), (0.42, 0.0)]],
    '2': [[(0.1, 1.4), (0.35, 1.72), (0.72, 1.55), (0.66, 1.1), (0.1, 0.0),
           (0.8, 0.0)]],
    '3': [[(0.12, 1.6), (0.5, 1.72), (0.72, 1.4), (0.4, 0.95), (0.75, 0.6),
           (0.55, 0.05), (0.12, 0.15)]],
    '4': [[(0.62, 1.7), (0.08, 0.55), (0.85, 0.55)], [(0.62, 1.1), (0.62, 0.0)]],
    '5': [[(0.75, 1.7), (0.2, 1.7), (0.15, 1.0), (0.5, 1.1), (0.75, 0.75),
           (0.6, 0.15), (0.15, 0.1)]],
    '6': [[(0.72, 1.65), (0.3, 1.35), (0.15, 0.6), (0.42, 0.02), (0.75, 0.3),
           (0.6, 0.72), (0.2, 0.72)]],
    '7': [[(0.08, 1.7), (0.82, 1.7), (0.35, 0.0)]],
    '8': [[(0.45, 0.85), (0.18, 1.2), (0.42, 1.7), (0.7, 1.25), (0.45, 0.85),
           (0.15, 0.45), (0.42, 0.02), (0.75, 0.42), (0.45, 0.85)]],
    '9': [[(0.75, 1.05), (0.35, 1.02), (0.2, 1.42), (0.52, 1.7), (0.75, 1.35),
           (0.7, 0.4), (0.3, 0.05)]],
}

# ---------------------------------------------------------------------------
# Cursive baseline alphabet.
#
# Standard joined-script letterforms, drawn as ONE stroke each, entering at
# the baseline-ish connection height on the left and leaving at the same
# height on the right, which is what makes cursive join legibly. These are
# generic shapes, not any author's -- they exist so that a letter whose own
# extracted prototype is malformed (a fragment, or a cut that swallowed its
# neighbour) still comes out as a readable letter.
#
# With only ~10 pages per author some letters never get a clean sample, so
# style has to come from RULES that generalise rather than from stored
# examples: whatever is used here is re-shaped by the author's own measured
# slant, x-height, ascender/descender reach, width and spacing, so it is
# written in their proportions even though the skeleton is generic.
# Coordinates: x right, y up from baseline, 1.0 = x-height.
_CX_IN, _CX_OUT = 0.0, 0.30      # entry / exit height, in x-heights
_CURSIVE = {
    'a': [[(0.0, 0.30), (0.20, 0.62), (0.46, 0.72), (0.62, 0.52), (0.60, 0.18),
           (0.42, 0.02), (0.20, 0.10), (0.16, 0.34), (0.30, 0.52), (0.52, 0.46),
           (0.62, 0.20), (0.64, 0.02), (0.86, 0.30)]],
    'b': [[(0.0, 0.30), (0.14, 1.10), (0.24, 1.66), (0.30, 1.20), (0.26, 0.44),
           (0.34, 0.10), (0.56, 0.04), (0.70, 0.24), (0.62, 0.48), (0.42, 0.52),
           (0.60, 0.34), (0.84, 0.30)]],
    'c': [[(0.0, 0.30), (0.22, 0.58), (0.50, 0.70), (0.62, 0.54), (0.52, 0.36),
           (0.30, 0.16), (0.34, 0.04), (0.58, 0.10), (0.80, 0.30)]],
    'd': [[(0.0, 0.30), (0.22, 0.58), (0.46, 0.70), (0.58, 0.50), (0.56, 0.20),
           (0.40, 0.04), (0.22, 0.14), (0.20, 0.40), (0.36, 0.58), (0.60, 0.60),
           (0.72, 1.62), (0.66, 0.60), (0.68, 0.16), (0.90, 0.30)]],
    'e': [[(0.0, 0.30), (0.24, 0.34), (0.46, 0.44), (0.34, 0.62), (0.16, 0.50),
           (0.18, 0.24), (0.38, 0.06), (0.62, 0.14), (0.78, 0.30)]],
    'f': [[(0.0, 0.30), (0.18, 0.90), (0.34, 1.62), (0.44, 1.72), (0.46, 1.20),
           (0.36, 0.30), (0.26, -0.48), (0.14, -0.62), (0.06, -0.40),
           (0.30, -0.20), (0.60, 0.14), (0.82, 0.30)]],
    'g': [[(0.0, 0.30), (0.22, 0.58), (0.48, 0.70), (0.60, 0.50), (0.56, 0.20),
           (0.40, 0.04), (0.22, 0.16), (0.24, 0.42), (0.42, 0.56), (0.62, 0.44),
           (0.64, 0.04), (0.58, -0.44), (0.40, -0.62), (0.20, -0.48),
           (0.34, -0.28), (0.66, 0.06), (0.86, 0.30)]],
    'h': [[(0.0, 0.30), (0.14, 1.06), (0.24, 1.66), (0.30, 1.10), (0.26, 0.34),
           (0.30, 0.06), (0.44, 0.42), (0.58, 0.62), (0.72, 0.48), (0.70, 0.14),
           (0.88, 0.30)]],
    'i': [[(0.0, 0.30), (0.20, 0.60), (0.32, 0.72), (0.34, 0.28), (0.40, 0.06),
           (0.58, 0.24), (0.66, 0.30)], [(0.34, 1.06), (0.36, 1.14)]],
    'j': [[(0.0, 0.30), (0.22, 0.62), (0.36, 0.74), (0.36, 0.10),
           (0.30, -0.44), (0.16, -0.62), (0.04, -0.44), (0.24, -0.26),
           (0.52, 0.10), (0.70, 0.30)], [(0.38, 1.06), (0.40, 1.14)]],
    'k': [[(0.0, 0.30), (0.14, 1.06), (0.24, 1.66), (0.30, 1.00), (0.26, 0.30),
           (0.30, 0.04), (0.36, 0.36), (0.62, 0.60), (0.44, 0.34), (0.52, 0.14),
           (0.74, 0.06), (0.88, 0.30)]],
    'l': [[(0.0, 0.30), (0.16, 1.10), (0.28, 1.68), (0.34, 1.10), (0.30, 0.30),
           (0.36, 0.06), (0.58, 0.16), (0.74, 0.30)]],
    'm': [[(0.0, 0.30), (0.10, 0.60), (0.18, 0.72), (0.22, 0.20), (0.30, 0.56),
           (0.44, 0.70), (0.54, 0.52), (0.52, 0.16), (0.60, 0.54), (0.74, 0.70),
           (0.86, 0.52), (0.84, 0.14), (1.02, 0.30)]],
    'n': [[(0.0, 0.30), (0.12, 0.60), (0.20, 0.72), (0.24, 0.18), (0.34, 0.56),
           (0.50, 0.70), (0.62, 0.52), (0.60, 0.14), (0.78, 0.30)]],
    'o': [[(0.0, 0.30), (0.20, 0.58), (0.44, 0.70), (0.60, 0.52), (0.58, 0.22),
           (0.40, 0.04), (0.20, 0.16), (0.22, 0.44), (0.44, 0.58), (0.62, 0.50),
           (0.66, 0.36), (0.84, 0.34)]],
    'p': [[(0.0, 0.30), (0.14, 0.62), (0.22, 0.74), (0.20, 0.10),
           (0.14, -0.50), (0.24, -0.20), (0.30, 0.40), (0.46, 0.66),
           (0.66, 0.56), (0.64, 0.26), (0.44, 0.08), (0.66, 0.14),
           (0.84, 0.30)]],
    'q': [[(0.0, 0.30), (0.22, 0.58), (0.46, 0.70), (0.60, 0.50), (0.56, 0.18),
           (0.38, 0.04), (0.20, 0.18), (0.24, 0.44), (0.46, 0.58), (0.64, 0.42),
           (0.66, 0.02), (0.62, -0.44), (0.76, -0.26), (0.90, 0.30)]],
    'r': [[(0.0, 0.30), (0.14, 0.58), (0.24, 0.72), (0.26, 0.44), (0.36, 0.64),
           (0.52, 0.66), (0.44, 0.48), (0.44, 0.14), (0.64, 0.28)]],
    's': [[(0.0, 0.30), (0.20, 0.56), (0.40, 0.68), (0.44, 0.50), (0.24, 0.34),
           (0.20, 0.14), (0.40, 0.04), (0.60, 0.14), (0.72, 0.30)]],
    't': [[(0.0, 0.30), (0.20, 0.90), (0.30, 1.44), (0.34, 0.90), (0.30, 0.20),
           (0.38, 0.04), (0.60, 0.16), (0.72, 0.30)],
          [(0.14, 0.86), (0.52, 0.90)]],
    'u': [[(0.0, 0.30), (0.14, 0.62), (0.20, 0.72), (0.20, 0.20), (0.32, 0.04),
           (0.48, 0.16), (0.52, 0.72), (0.54, 0.24), (0.60, 0.08),
           (0.80, 0.30)]],
    'v': [[(0.0, 0.30), (0.14, 0.62), (0.22, 0.72), (0.34, 0.14), (0.50, 0.62),
           (0.56, 0.44), (0.52, 0.28), (0.72, 0.34)]],
    'w': [[(0.0, 0.30), (0.12, 0.62), (0.20, 0.72), (0.30, 0.12), (0.44, 0.66),
           (0.56, 0.14), (0.70, 0.66), (0.74, 0.42), (0.70, 0.28),
           (0.90, 0.34)]],
    'x': [[(0.0, 0.30), (0.16, 0.60), (0.28, 0.68), (0.56, 0.08),
           (0.70, 0.26)], [(0.24, 0.10), (0.62, 0.66)]],
    'y': [[(0.0, 0.30), (0.14, 0.62), (0.20, 0.72), (0.22, 0.22), (0.34, 0.06),
           (0.50, 0.20), (0.54, 0.72), (0.52, 0.10), (0.44, -0.46),
           (0.28, -0.62), (0.14, -0.44), (0.34, -0.26), (0.62, 0.06),
           (0.80, 0.30)]],
    'z': [[(0.0, 0.30), (0.18, 0.60), (0.44, 0.62), (0.24, 0.24), (0.16, 0.06),
           (0.36, 0.02), (0.44, -0.34), (0.30, -0.50), (0.20, -0.34),
           (0.46, -0.10), (0.66, 0.20), (0.78, 0.30)]],
}

_FB_ADV = 1.0

# ---------------------------------------------------------------------------
# Uniform pen. The gantry writes every author with ONE physical pen at a
# constant stroke width, so ink weight / density is not a style channel it
# can reproduce. When UNIFORM_INK is on, every author is rendered at the
# same width (expressed in x-heights so it is resolution-independent, and
# matched to TrainAuthorShape.PEN_WIDTH_XH so the shape-only writer-ID model
# sees the same normalization it was trained on) and the per-author
# inkFracRef / strokeWidthXh / inkCoreLevel fields are ignored.
# ---------------------------------------------------------------------------
UNIFORM_INK = True
UNIFORM_PEN_WIDTH_XH = 0.11         # bold enough for a real pen, and a hair
                                  # more legible than 0.14 (probe: +2pt char)
UNIFORM_SOFT_FRAC = 0.30            # blur radius as a fraction of strokePx

# Legibility guards (tested in phase 3; no-ops at these defaults so the
# pre-legibility baseline is unchanged).
JOIN_PROB_CAP = 1.0                # cap the per-pair cursive-join probability
CORE_SLANT_CAP_DEG = 22.0          # print-anchor glyphs are never sheared past
                                  # this -- a 40deg shear makes even a clean
                                  # letterform collide and ramp illegibly
CORE_JITTER_FRAC = 0.15           # print-anchor glyphs barely wobble
COLLISION_MIN_GAP_XH = 0.05        # force this much clear space (x-h) between
                                  # a glyph's ink and the previous glyph's

# Legibility anchor.
#
# Tested (EvaluateLegibility sweep + font probe): a data-driven "legible
# core" -- the cross-author medoid of every real extracted variant per
# character -- reads back at only ~76% char / ~39% word, no better than the
# raw author hands, because the consensus of a messy cursive letter is still
# a messy cursive letter. The hand-drawn single-stroke PRINT font `_FB`
# reads at ~96% char / ~84% word. So `_FB` (reshaped to each author's
# ascender/descender/slant by `_BaselineGlyph`) is the legibility anchor,
# and `legibility` in [0, 1] is the probability -- biased by how malformed
# the author's own prototype is -- that a glyph is drawn from it instead of
# the author's hand. lam 0 = pure author (pre-legibility behaviour),
# lam 1 = everything printed.

# A stored prototype further than this (cosine distance) from the
# cross-author letter prototype is treated as unusable -- a fragment, or a
# cut that swallowed a neighbour. Set from the measured distribution of
# those distances (p50 0.25, p90 0.41, max 0.92), so roughly the worst
# tenth of prototypes are replaced by the generic skeleton.
_PRIOR_REJECT = 0.25


def _BaselineGlyph(ch, profile, printOnly=False):
    """Generic legible letterform, re-proportioned into the author's own
    style (scaled to their ascender/descender reach; the author's slant and
    jitter are applied later by SynthesizeText).

    The single-stroke PRINT font `_FB` reads back at ~96% char / ~84% word
    through the frozen recognizer -- far clearer than any extracted glyph --
    so it is the legibility anchor. `printOnly=True` forces it; otherwise a
    cursive author keeps the joined `_CURSIVE` skeleton (89% / 62%) for a
    letter that merely never got a clean sample."""
    conn = float(profile.get('connectedness', 0.0))
    src = _CURSIVE if (conn >= 0.5 and not printOnly) else _FB
    key = ch if ch in src else ch.lower()
    if key not in src:
        return None
    asc = float(profile.get('ascender', 1.7))
    desc = float(profile.get('descender', -0.6))
    strokes = []
    for st in src[key]:
        q = []
        for (x, y) in st:
            if y > 1.0:
                y = 1.0 + (y - 1.0) * max(0.35, (asc - 1.0) / 0.75)
            elif y < 0.0:
                y = y * max(0.35, abs(desc) / 0.55)
            q.append((x, y))
        strokes.append(q)
    xs = [p[0] for st in strokes for p in st]
    adv = (max(xs) - min(xs)) + 0.12
    return strokes, adv

# how many of a character's stored variants are eligible per instance
_ELIGIBLE_VARIANTS = 2


class Trajectory:
    """Pen path in millimetres. y is UP; the caller's writer flips if needed."""

    def __init__(self, strokes, meta):
        self.strokes = strokes        # list of [(x, y), ...] pen-DOWN paths
        self.meta = meta

    def Bounds(self):
        pts = [p for s in self.strokes for p in s]
        if not pts:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def PenTravelMm(self):
        down = sum(float(np.hypot(*np.diff(np.asarray(s), axis=0).T).sum())
                   for s in self.strokes if len(s) > 1)
        up = 0.0
        for a, b in zip(self.strokes, self.strokes[1:]):
            up += float(math.dist(a[-1], b[0]))
        return down, up


# ---------------------------------------------------------------------------
# Glyph selection
# ---------------------------------------------------------------------------
_SIMILAR = {'I': 'l', 'O': '0', 'l': 'I', '0': 'O', 'o': '0', ';': ':',
            '"': "'", '!': 'l'}


# Tested: giving connected hands the JOINED cursive skeleton as their anchor
# (to preserve the "connected look") costs ~15pt char / ~30pt word
# legibility on the delivery path and does NOT recover style (152/153 stay
# at 0/5 shape-ID). Legibility is the priority, so every anchor glyph is now
# the upright PRINT form; the connected look, where it survives, comes from
# the author's own kept glyphs + ligatures, not the anchor. Set > 1.0 to
# disable the cursive anchor entirely.
CORE_CURSIVE_CONN = 2.0


def _CoreGlyph(ch, profile):
    """The legible anchor letterform for `ch`, reshaped to this author.

    A connected hand (measured connectedness >= CORE_CURSIVE_CONN) gets the
    JOINED cursive skeleton -- entry/exit at connection height so the joins
    read -- so its output still looks like joined script, not print. A
    disconnected hand gets the upright print skeleton. Either way it is
    scaled to the author's ascender/descender reach; slant, spacing and the
    joins themselves are applied later.

    Returns (strokes, advance, lead, isCursive)."""
    cursive = float(profile.get('connectedness', 0.0)) >= CORE_CURSIVE_CONN
    b = _BaselineGlyph(ch, profile, printOnly=not cursive)
    return None if b is None else (b[0], b[1], 0.0, cursive)


# priorD is cosine distance from the cross-author consensus letter. The
# per-author distributions (measured) sit at p50 ~= 0.22-0.33, p90 ~= 0.32-0.47
# -- i.e. most of an author's stored letterforms are "atypical" simply
# because that IS their hand, not because they are broken. So the glyph
# swap is now rare: keep the author's own letter unless it is genuinely
# unresolvable, and fix legibility through LAYOUT (slant cap, join control,
# spacing) and the empirical repair pass instead. This keeps each hand
# recognisable.
_GLYPH_CLEAN = 0.12        # priorD at/below which the author's letter is a
                          # clean example -- keep it
_GLYPH_GARBAGE = 0.55      # at/above this it is replaced regardless of lam


def _SwapProb(lam, priorD):
    """P(replace this glyph with the style-aware anchor letterform).

    Legibility-first: the user needs the text readable, and measurement
    shows an author's *own* letterform is only reliably readable when it is
    a genuinely clean extraction (priorD <= _GLYPH_CLEAN -- typically a
    denoised prototype). Those are kept; everything else swaps to the anchor
    readily, scaled by `lam`. The anchor still carries the author's slant,
    size, spacing and joins, so the hand is not erased."""
    d = float(priorD)
    if d >= _GLYPH_GARBAGE:
        return 1.0
    if lam <= 1e-3:
        return 0.0
    if d <= _GLYPH_CLEAN:
        return 0.06 * lam
    bad = float(np.clip((d - _GLYPH_CLEAN) / (_GLYPH_GARBAGE - _GLYPH_CLEAN),
                        0.0, 1.0))
    return float(np.clip(lam * (0.55 + 0.9 * bad), 0.0, 1.0))


def _GlyphSource(profile, ch, rng, prevExitY=None, lam=0.0, forceCore=False,
                 forcePrint=False):
    """Returns (strokes, advance, lead, source). Uses one of the author's
    own variants, unless the prototype is unresolvable, `forceCore` is set
    (repair pass -> style-aware anchor), `forcePrint` is set (repair pass on
    a letter still unreadable in the cursive anchor -> upright print), or --
    rarely -- `_SwapProb` fires."""
    lib = profile['glyphs']
    adv = profile.get('letterAdvance', {})
    if forcePrint:
        b = _BaselineGlyph(ch, profile, printOnly=True)
        if b is not None:
            return b[0], b[1], 0.0, 'core'

    def pack(g, k=1.0):
        strokes = [[(x * k, y * k) for (x, y) in s] for s in g['strokes']]
        # the variant's OWN advance, never the character median: a wide
        # variant advanced by a narrow median would collide with the next
        # letter (and vice versa, leaving a hole)
        a = max(float(g.get('advance', 0.9)),
                float(g.get('lead', 0.0)) + float(g.get('width', 0.9)) + 0.04)
        return strokes, a * k, float(g.get('lead', 0.0)) * k

    def pick(cands):
        """Favour the more typical variants (the list is sorted by
        typicality) while still varying -- uniform choice over a noisy tail
        is what makes synthesized cursive look scrambled.

        When joining onto a previous letter, also favour the variant whose
        own ENTRY height matches where the pen actually is: in cursive a
        letter's shape depends on what it is joined to, so mixing a
        high-entry variant onto a low exit is what produces the tangled
        joins."""
        # Only the few most TYPICAL variants are eligible. A writer forms a
        # given letter consistently; sampling uniformly over a dozen
        # variants averages their signature away (measured: 75.5% writer-ID
        # over 12 variants vs 80.5% over the top 2). Keeping more than one
        # still beats a single frozen template, which reads as stamped.
        cands = cands[:_ELIGIBLE_VARIANTS]
        w = []
        for i, g in enumerate(cands):
            wi = 1.0 / (1.0 + 1.4 * i)
            if prevExitY is not None:
                dy = abs(float(g.get('entryY', 0.0)) - prevExitY)
                wi *= math.exp(-(dy / 0.45) ** 2)
            w.append(wi)
        tot = sum(w)
        if tot <= 1e-9:
            return cands[0]
        r = rng.random() * tot
        acc = 0.0
        for i, wi in enumerate(w):
            acc += wi
            if r <= acc:
                return cands[i]
        return cands[-1]

    if ch in lib and lib[ch]:
        cands = lib[ch]
        # rank by priorD, take the most well-formed the author actually wrote
        ranked = sorted(cands, key=lambda g: g.get('priorD', 0.5))
        g = pick(ranked)
        d = float(g.get('priorD', 0.5))
        core = _CoreGlyph(ch, profile)
        if core is not None and (forceCore
                                 or rng.random() < _SwapProb(lam, d)):
            return core[0], core[1], core[2], 'core'
        s, a, lead = pack(g)
        return s, a, lead, 'own'
    for alt in (ch.swapcase(), _SIMILAR.get(ch, '')):
        if alt and alt in lib and lib[alt]:
            core = _CoreGlyph(ch, profile)
            if core is not None and (forceCore
                                     or rng.random() < _SwapProb(lam, 0.4)):
                return core[0], core[1], core[2], 'core'
            g = pick(sorted(lib[alt], key=lambda x: x.get('priorD', 0.5)))
            if alt == ch.swapcase() and ch.isupper():
                # borrow the lowercase shape, grown to capital height
                k = profile.get('ascender', 1.7) / max(0.6, g['top'] or 1.0)
                k = float(np.clip(k, 1.0, 2.0))
                s, a, lead = pack(g, k)
                return s, a, lead, 'case'
            s, a, lead = pack(g)
            return s, a, lead, 'similar'
    core = _CoreGlyph(ch, profile)
    if core is not None:
        return core[0], core[1], core[2], 'core'
    key = ch.lower() if ch.lower() in _FB else ch
    if key in _FB:
        strokes = [list(map(tuple, s)) for s in _FB[key]]
        return strokes, adv.get(ch, _FB_ADV), 0.0, 'fallback'
    return [], adv.get(ch, 0.6), 0.0, 'blank'


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
def _Jitter(strokes, rng, amp, rotAmp, scaleAmp):
    dx = rng.gauss(0.0, amp)
    dy = rng.gauss(0.0, amp * 0.6)
    th = math.radians(rng.gauss(0.0, rotAmp))
    sx = 1.0 + rng.gauss(0.0, scaleAmp)
    sy = 1.0 + rng.gauss(0.0, scaleAmp)
    c, s = math.cos(th), math.sin(th)
    out = []
    for stk in strokes:
        q = []
        for (x, y) in stk:
            x2, y2 = x * sx, y * sy
            q.append((x2 * c - y2 * s + dx, x2 * s + y2 * c + dy))
        out.append(q)
    return out


def _Resample(poly, step):
    """Even arc-length resampling: the plotter wants smooth motion, and it
    also makes the rendered stroke width uniform."""
    if len(poly) < 2:
        return poly
    p = np.asarray(poly, np.float64)
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-9:
        return [tuple(p[0])]
    n = max(2, int(round(total / step)) + 1)
    t = np.linspace(0.0, total, n)
    x = np.interp(t, cum, p[:, 0])
    y = np.interp(t, cum, p[:, 1])
    return list(zip(x.tolist(), y.tolist()))


def _ResampleN(poly, n):
    """Arc-length resample to exactly `n` points -- so two shapes can be
    blended vertex-for-vertex."""
    p = np.asarray(poly, np.float64)
    if len(p) == 0:
        return []
    if len(p) == 1:
        return [tuple(p[0])] * n
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-9:
        return [tuple(p[0])] * n
    t = np.linspace(0.0, cum[-1], n)
    return list(zip(np.interp(t, cum, p[:, 0]).tolist(),
                    np.interp(t, cum, p[:, 1]).tolist()))


def _BlendGlyph(aStrokes, cStrokes, lam):
    """Interpolate the author's letterform toward the legible-core letterform
    in the upright x-height frame. lam 0 = pure author, 1 = pure core.

    Strokes are matched by index (both libraries store strokes sorted
    left-to-right). If the stroke counts differ the shapes are not
    vertex-comparable, so the core is taken whole once it is the majority
    (lam >= 0.5) and the author kept otherwise."""
    if lam <= 1e-3 or not cStrokes:
        return aStrokes
    cS = [list(map(tuple, s)) for s in cStrokes]
    if lam >= 1.0 - 1e-3 or len(aStrokes) != len(cS):
        return cS if lam >= 0.5 else aStrokes
    out = []
    for sa, sc in zip(aStrokes, cS):
        m = max(len(sa), len(sc), 2)
        ra, rc = _ResampleN(sa, m), _ResampleN(sc, m)
        out.append([((1 - lam) * xa + lam * xc, (1 - lam) * ya + lam * yc)
                    for (xa, ya), (xc, yc) in zip(ra, rc)])
    return out


def SynthesizeText(text, profile, mmPerXh=4.0, seed=None, lineWidthMm=180.0,
                   jitter=0.5, legibility=0.0, perCharLam=None, _cal=None):
    """text -> Trajectory (mm, y up, origin at first baseline).

    mmPerXh: physical size of one x-height. lineWidthMm: wrap width."""
    rng = random.Random(seed)
    xh = float(mmPerXh)
    cal = _cal if _cal is not None else (
        StyleCalibration(profile, mmPerXh) if USE_STYLE_CALIBRATION
        else dict(shearDelta=0.0, ascK=1.0, descK=1.0))
    slantDeg = profile.get('slantDeg', 0.0)
    # legibility caps the lean of the AUTHOR's own glyphs on a soft curve
    # (45deg at lam 0 -> ~26deg at lam 1); the print anchor is capped harder
    # (CORE_SLANT_CAP_DEG) further down. A 40deg author hand stays visibly
    # slanted, just not to the point where ascenders ramp off their cell.
    authorSlantCap = 45.0 - 19.0 * float(np.clip(legibility, 0.0, 1.0))
    slantDeg = float(np.clip(slantDeg, -authorSlantCap, authorSlantCap))
    slant = math.tan(math.radians(slantDeg)) + cal['shearDelta']
    conn = float(profile.get('connectedness', 0.0))
    # legibility thins out the cursive joins (tangled joins are the main
    # thing that makes a connected hand unreadable) without removing them --
    # a connected author still writes visibly connected at lam 0.65.
    connEff = conn * (1.0 - 0.6 * float(np.clip(legibility, 0.0, 1.0)))
    wordGap = float(profile.get('wordSpaceXh', 1.2))
    ascender = float(profile.get('ascender', 1.7))
    descender = float(profile.get('descender', -0.6))
    lineStep = (ascender - descender + 0.85) * xh

    ampP = 0.022 * jitter          # position jitter, x-heights
    ampR = 2.2 * jitter            # rotation jitter, degrees
    ampS = 0.05 * jitter           # scale jitter
    driftAmp = 0.05 * jitter       # slow baseline drift, x-heights

    strokes = []
    penX, penY = 0.0, 0.0
    lineIdx = 0
    drift = 0.0
    driftV = 0.0
    usage = {}
    vpos = 0                         # index into text.replace('\n',' '), for perCharLam

    def flush(chain):
        if len(chain) >= 2:
            strokes.append(_Resample(chain, step=0.35))
        elif len(chain) == 1:
            strokes.append(chain + [(chain[0][0] + 0.05, chain[0][1])])

    for word in _SplitWords(text):
        if word == '\n':
            lineIdx += 1
            penX = 0.0
            penY = -lineIdx * lineStep
            vpos += 1
            continue
        wWidth = _WordWidth(word, profile) * xh
        if penX > 1e-6 and penX + wWidth > lineWidthMm:
            lineIdx += 1
            penX = 0.0
            penY = -lineIdx * lineStep
        chain = []                  # current pen-down polyline
        prevExit = None
        prevBodyMaxX = None
        prevSrc = None
        prevCoreCursive = False
        for ci, ch in enumerate(word):
            lam = legibility
            forceCore = forcePrint = False
            if perCharLam:
                pc = float(perCharLam.get(vpos, 0.0))
                lam = max(lam, min(pc, 1.0))
                forceCore = pc >= 0.999
                forcePrint = pc >= 1.9
            vpos += 1
            wantEntryY = None
            if connEff >= 0.2 and ci > 0 and prevExit is not None and \
                    ch.isalpha() and word[ci - 1].isalpha():
                wantEntryY = (prevExit[1] - penY) / xh - drift
            gStrokes, adv, lead, src = _GlyphSource(profile, ch, rng,
                                                    prevExitY=wantEntryY,
                                                    lam=lam, forceCore=forceCore,
                                                    forcePrint=forcePrint)
            usage[src] = usage.get(src, 0) + 1
            if not gStrokes:
                penX += adv * xh
                continue
            isCore = (src == 'core')
            coreCursive = isCore and conn >= CORE_CURSIVE_CONN
            # the anchor is a designed-legible shape: don't re-stretch its
            # proportions or wobble it as hard as an extracted glyph -- that
            # re-introduces the ambiguity it is there to remove. It DOES keep
            # the author's lean (a joined hand is meant to slant), just
            # capped so ascenders don't ramp off the cell.
            if not isCore:
                gStrokes = _ApplyCal(gStrokes, cal)
            jf = CORE_JITTER_FRAC if isCore else 1.0
            gStrokes = _Jitter(gStrokes, rng, ampP * jf, ampR * jf, ampS * jf)
            glyphSlant = slant
            glyphDrift = drift
            if isCore:
                capDeg = 30.0 if coreCursive else CORE_SLANT_CAP_DEG
                cap = math.tan(math.radians(capDeg))
                glyphSlant = max(-cap, min(cap, slant))
                glyphDrift = (0.5 if coreCursive else 0.25) * drift
            # slow drift of the baseline within a line (a real hand wanders)
            driftV = 0.85 * driftV + rng.gauss(0.0, driftAmp * 0.35)
            drift = float(np.clip(drift + driftV, -0.25, 0.25))

            def place(px):
                out = []
                for stk in gStrokes:
                    out.append([(px + (lead + x + glyphSlant * y) * xh,
                                 penY + (y + glyphDrift) * xh) for (x, y) in stk])
                return out

            placed = place(penX)
            # collision guard: if this glyph's ink would land on top of the
            # previous glyph's body, push it (and the pen) right until there
            # is a clear gap.
            if COLLISION_MIN_GAP_XH is not None and prevBodyMaxX is not None:
                curMinX = min(p[0] for s in placed for p in s)
                need = prevBodyMaxX + COLLISION_MIN_GAP_XH * xh - curMinX
                if need > 0:
                    penX += need
                    placed = place(penX)

            # the BODY stroke (widest in x) is what carries the cursive
            # connection; dots, crossbars and accents are separate pen-downs
            # and must not become the exit point for the next letter
            bodyI = int(np.argmax([max(p[0] for p in s) - min(p[0] for p in s)
                                   for s in placed]))
            body = placed[bodyI]
            # `connectedness` is the measured FRACTION of adjacent in-word
            # letter pairs this author actually joins, so it is used as the
            # PROBABILITY of joining each pair -- not as an on/off switch.
            # As a threshold it joined every pair for anyone above it, which
            # fused whole words into single wide blobs: letter pitch and
            # component width came out far too large and the number of
            # separate ink pieces far too low, which is exactly the geometry
            # a nearest-author matcher keys on.
            # Join through a pair when: both glyphs are the author's own
            # extracted forms, OR both are the cursive anchor (a connected
            # hand keeps writing connected even through the cleaned-up
            # letters). A PRINT anchor glyph is always read as a separate
            # letter -- never join through it.
            bothOwn = not isCore and prevSrc != 'core'
            bothCursiveCore = coreCursive and prevCoreCursive
            joinable = (ci > 0 and prevExit is not None
                        and ch.isalpha() and word[ci - 1].isalpha()
                        and (bothOwn or bothCursiveCore)
                        and rng.random() < min(connEff, JOIN_PROB_CAP))
            if joinable:
                chain += _Ligature(prevExit, body[0], xh, rng)
                chain += body
            else:
                flush(chain)
                chain = list(body)
            prevExit = chain[-1]
            for k2, extra in enumerate(placed):
                if k2 != bodyI:
                    flush(extra)
            prevBodyMaxX = max(p[0] for p in body)
            prevSrc = src
            prevCoreCursive = coreCursive
            penX += adv * xh
        flush(chain)
        penX += wordGap * xh * (0.85 + 0.3 * rng.random())
        vpos += 1               # the space between words

    meta = dict(author=profile.get('authorId'), mmPerXh=mmPerXh,
                nStrokes=len(strokes), glyphSources=usage,
                slantDeg=profile.get('slantDeg', 0.0),
                connectedness=conn, lines=lineIdx + 1,
                ascender=ascender, descender=descender)
    return Trajectory(strokes, meta)


# Line-level slant/ascender calibration. Synthesis drifts on the two
# geometric statistics that a person actually looks at: author 150's
# ascenders render ~7% tall, author 551's slant ~9 deg steep. Correcting
# them measured WORSE against the original ink-sensitive writer-ID model
# (86.0% -> 84.5%) but BETTER against the shape-only model that ignores pen
# weight (59.2% -> 63.3%, and 58.3% -> 66.7% through G-code).
#
# The shape-only judge is the one that predicts the physical robot, which
# writes every author with the same pen, so this is ON.
USE_STYLE_CALIBRATION = True

_STYLE_CAL = {}


def StyleCalibration(profile, mmPerXh=4.0, pxPerMm=18.0):
    """Correct the two line-level style statistics that synthesis drifts on.

    Glyphs are stored per-instance, so a line assembled from them does not
    automatically reproduce the author's line-level slant and ascender
    reach: measured on a probe render, ascenders come out taller and steep
    hands steeper than the author's own pages. That drift is what pushed
    author 150's synthesis into author 384's territory (both neat print
    hands, distinguished mainly by ascender height and lean).

    So: render a probe, measure it the same way the real pages were
    measured, and return the shear delta and ascender/descender scaling
    that bring the two into agreement. Cached per author."""
    key = (profile.get('authorId'), round(mmPerXh, 3), round(pxPerMm, 3))
    if key in _STYLE_CAL:
        return _STYLE_CAL[key]
    import BuildStyleProfile as _SP
    ref = (profile.get('slantMeasRef'), profile.get('ascMeasRef'),
           profile.get('descMeasRef'))
    cal = dict(shearDelta=0.0, ascK=1.0, descK=1.0)
    if None in ref or any(r != r for r in ref):
        _STYLE_CAL[key] = cal
        return cal
    slantRef, ascRef, descRef = ref
    sample = ('the quick brown fox jumps over lazy dogs and by half past '
              'eight it kept flying quietly')
    for _ in range(3):
        traj = SynthesizeText(sample, profile, mmPerXh=mmPerXh, seed=17,
                              lineWidthMm=10_000.0, jitter=0.0, _cal=cal)
        img = RenderTrajectory(traj, pxPerMm=pxPerMm, profile=profile)
        ink = _SP.BinarizeLine(np.array(img))
        band = _SP.CoreBand(ink)
        if band is None or not ink.any():
            break
        top, base = band
        xh = float(base - top)
        if xh < 4:
            break
        ys = np.nonzero(ink.any(axis=1))[0]
        gotSlant = _SP.EstimateSlantDeg(ink)
        gotAsc = float(base - ys.min()) / xh
        gotDesc = float(base - ys.max()) / xh
        cal = dict(
            shearDelta=cal['shearDelta'] +
            math.tan(math.radians(slantRef)) - math.tan(math.radians(gotSlant)),
            ascK=float(np.clip(cal['ascK'] * (ascRef - 1.0) /
                               max(1e-3, gotAsc - 1.0), 0.5, 1.6)),
            descK=float(np.clip(cal['descK'] * descRef / min(-1e-3, gotDesc),
                                0.5, 1.6)))
    _STYLE_CAL[key] = cal
    return cal


def _ApplyCal(strokes, cal):
    """Compress/stretch only what lies OUTSIDE the x-height band, so the
    ascender and descender reach change without disturbing x-height."""
    ascK, descK = cal['ascK'], cal['descK']
    if abs(ascK - 1.0) < 1e-3 and abs(descK - 1.0) < 1e-3:
        return strokes
    out = []
    for s in strokes:
        q = []
        for (x, y) in s:
            if y > 1.0:
                y = 1.0 + (y - 1.0) * ascK
            elif y < 0.0:
                y = y * descK
            q.append((x, y))
        out.append(q)
    return out


def _Ligature(p0, p1, xh, rng):
    """The carry-across stroke between two joined letters. It sags only in
    proportion to how far it has to travel -- a fixed dip turns a short
    join into a loop, which is what makes synthesized cursive look
    tangled."""
    dx = p1[0] - p0[0]
    if abs(dx) < 0.06 * xh:
        return [p0, p1]
    sag = min(0.11 * xh, 0.16 * abs(dx)) + rng.gauss(0.0, 0.015 * xh)
    mid = ((p0[0] + p1[0]) * 0.5, min(p0[1], p1[1]) - sag)
    return _Resample([p0, mid, p1], step=0.35)


def _SplitWords(text):
    out = []
    for chunk in text.split('\n'):
        out += [w for w in chunk.split(' ') if w]
        out.append('\n')
    return out[:-1] if out and out[-1] == '\n' else out


def _WordWidth(word, profile):
    adv = profile.get('letterAdvance', {})
    return sum(adv.get(c, 0.9) for c in word)


# ---------------------------------------------------------------------------
# Rendering (must look like the training crops: black ink on white)
# ---------------------------------------------------------------------------
def _Draw(traj, pxPerMm, padMm, strokePx, paper=255, ink=0, softPx=0.0):
    x0, y0, x1, y1 = traj.Bounds()
    W = max(8, int((x1 - x0 + 2 * padMm) * pxPerMm))
    H = max(8, int((y1 - y0 + 2 * padMm) * pxPerMm))
    img = Image.new('L', (W, H), int(round(paper)))
    d = ImageDraw.Draw(img)
    fill = int(round(ink))
    for s in traj.strokes:
        pts = [((x - x0 + padMm) * pxPerMm, (y1 - y + padMm) * pxPerMm)
               for (x, y) in s]
        if len(pts) >= 2:
            d.line(pts, fill=fill, width=strokePx, joint='curve')
            # round the stroke ends: a pen leaves no square corners
            r = strokePx / 2.0
            if r >= 1.0:
                for (px, py) in (pts[0], pts[-1]):
                    d.ellipse([px - r, py - r, px + r, py + r], fill=fill)
        else:
            d.point(pts, fill=fill)
    if softPx > 0.05:
        img = img.filter(ImageFilter.GaussianBlur(radius=softPx))
    return img


def RenderTrajectory(traj, pxPerMm=8.0, padMm=None, strokePx=None,
                     profile=None, size=None, matchInk=True, uniformInk=None):
    """Render as a training-style crop (black ink, white paper).

    With `uniformInk` (default `UNIFORM_INK`) every author is drawn at one
    constant stroke width -- what the single-pen gantry actually produces,
    and the domain the shape-only writer-ID model is trained on. The
    per-author ink fields are ignored.

    With `uniformInk=False` the legacy behaviour is used: stroke WEIGHT is
    treated as style and, when the profile carries the author's measured ink
    density (`inkFracRef`), the width is searched to reproduce it."""
    prof = profile or {}
    mm = traj.meta.get('mmPerXh', 4.0)
    if uniformInk is None:
        uniformInk = UNIFORM_INK
    if padMm is None:
        # padding proportional to the writing, not a fixed slab: a fixed
        # pad inflates the crop height and skews the line's aspect ratio
        padMm = 0.18 * mm

    if uniformInk:
        sp = strokePx or max(1, int(round(UNIFORM_PEN_WIDTH_XH * mm * pxPerMm)))
        img = _Draw(traj, pxPerMm, padMm, sp, paper=255, ink=0,
                    softPx=UNIFORM_SOFT_FRAC * max(1.0, sp))
        if size is not None:
            img = img.resize(size, Image.Resampling.BILINEAR)
        return img

    if strokePx is None:
        sw = prof.get('strokeWidthXh', 0.12)
        strokePx = max(1, int(round(sw * mm * pxPerMm)))

    paper = prof.get('paperLevel', 255.0)
    # NOTE: `inkLevel` in the profile is the median of the real crop's
    # already-dark pixels, i.e. a whole-stroke average that includes the
    # soft edges -- NOT the colour of the stroke core. Painting entire
    # strokes at it puts them right at the classifier's ink threshold and
    # wrecks the density calibration (measured: 76% -> 46% writer-ID), so
    # the core is drawn dark and the soft edge comes from the blur instead.
    inkLv = prof.get('inkCoreLevel', 0.0)
    soft = prof.get('softScale', 0.30) * max(1.0, strokePx)
    target = prof.get('inkFracRef') if matchInk else None
    if target:
        img, strokePx = _MatchInkDensity(traj, pxPerMm, padMm, strokePx,
                                         target, mm, paper, inkLv, soft)
    else:
        img = _Draw(traj, pxPerMm, padMm, strokePx, paper, inkLv, soft)
    if size is not None:
        img = img.resize(size, Image.Resampling.BILINEAR)
    return img


def _MatchInkDensity(traj, pxPerMm, padMm, startPx, target, mm,
                     paper=255, ink=0, soft=0.0):
    """Choose the stroke width whose rendered ink density matches the
    author's real pages, by searching on THIS line.

    It has to be a search, not a formula: after the classifier's
    downsampling a thin stroke anti-aliases to light grey and drops out of
    the <128 ink test entirely, so density is a step-like function of
    width. And it has to be on this line rather than a fixed probe,
    because density also depends on how much text the line holds -- a
    short line stretched to the fixed input width has fatter letters."""
    from TrainText import resize_line_image_fixed

    hi = max(3, int(0.55 * mm * pxPerMm))
    cache = {}

    def probe(px):
        px = int(max(1, min(hi, px)))
        if px not in cache:
            im = _Draw(traj, pxPerMm, padMm, px, paper, ink,
                       0.30 * max(1.0, px))
            got = float((np.array(resize_line_image_fixed(im)) < 128).mean())
            cache[px] = (im, got, abs(got - target))
        return cache[px]

    cur = int(max(1, min(hi, startPx)))
    im, got, err = probe(cur)
    step = -1 if got > target else 1
    while 1 <= cur + step <= hi:
        im2, got2, err2 = probe(cur + step)
        if err2 >= err:
            break
        cur, im, got, err = cur + step, im2, got2, err2
    return im, cur


def LoadAllProfiles():
    profs = {}
    for p in sorted(PROFILE_DIR.glob('*.json')):
        with open(p, encoding='utf-8') as f:
            profs[p.stem] = json.load(f)
    return profs


if __name__ == '__main__':
    text = sys.argv[1] if len(sys.argv) > 1 else \
        'The quick brown fox jumps over the lazy dog'
    profs = LoadAllProfiles()
    outDir = SCRIPT_DIR / 'NOGIT' / 'SynthPreview'
    outDir.mkdir(parents=True, exist_ok=True)
    for a, prof in profs.items():
        traj = SynthesizeText(text, prof, seed=1)
        RenderTrajectory(traj, profile=prof).save(outDir / f'{a}.png')
        print(f"{a}: {traj.meta['nStrokes']} strokes {traj.meta['glyphSources']}")
    print('wrote', outDir)


# ---------------------------------------------------------------------------
# Legibility-first synthesis
# ---------------------------------------------------------------------------
REPAIR_ROUNDS = 3          # targeted re-draw passes after best-of-N
REPAIR_WORST_K = 3         # characters pinned to the legible core per round

# How hard to lean on the print anchor when a profile carries no explicit
# legibilityLambda. Measured sweet spot: ~0.65 gives ~86% char / ~48% word
# aggregate on novel text through the full delivery path, vs ~77% / ~37% at
# the previous per-author scheme. Lower = more of the author's own hand
# (less legible); higher = closer to plain print.
DEFAULT_LEGIBILITY = 0.65


def _LineLogProbs(reader, img, device):
    """(T, C) log-probs from the frozen recognizer for one line image."""
    import torch
    from TrainText import resize_line_image_fixed, tensor_from_resized
    t = tensor_from_resized(resize_line_image_fixed(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        return reader(t)[:, 0, :].cpu().numpy()


def _WeakChars(reader, img, device, visible):
    """Visible-text positions the recognizer is least sure about, worst
    first. Uses CTC forced alignment against the intended text and scores
    each character by its own class log-prob over its assigned frames."""
    try:
        from BuildStyleProfile import CtcForcedAlign
        from TrainText import CHAR_TO_IDX
    except Exception:
        return []
    lp = _LineLogProbs(reader, img, device)
    res = CtcForcedAlign(lp, visible)
    if res is None:
        return []
    align, _conf = res
    keptPos = [j for j, c in enumerate(visible) if c in CHAR_TO_IDX]
    scored = []
    for i, (c, sp) in enumerate(align):
        if i >= len(keptPos) or not c.strip():
            continue
        if sp is None:
            scored.append((-1e9, keptPos[i]))
        else:
            scored.append((float(lp[sp[0]:sp[1], CHAR_TO_IDX[c]].mean()),
                           keptPos[i]))
    scored.sort()
    return [pos for _s, pos in scored]


def SynthesizeLegible(text, profile, nTries=6, mmPerXh=4.0, lineWidthMm=180.0,
                      jitter=0.5, seed=0, reader=None, device=None,
                      pxPerMm=18.0, legibility=None, repair=True):
    """Best-of-N in the author's style, then a targeted repair pass.

    1. draw the line `nTries` times (same library + measured parameters,
       different variant/jitter draws) at the author's blend level
       `legibility` (defaults to profile['legibilityLambda']), keep the one
       the frozen recognizer reads best;
    2. repair: find the characters the recognizer still cannot read, pin
       just those to the legible core (lam = 1) and re-draw, keep if it
       reads better. Repeat up to REPAIR_ROUNDS times.

    Every candidate is a real rendering in this hand -- step 2 only cleans
    up the few letters that were unreadable, it does not restyle the line.
    Falls back to a single plain synthesis if the recognizer is missing.
    """
    if legibility is None:
        legibility = float(profile.get('legibilityLambda', DEFAULT_LEGIBILITY))
    if reader is None:
        try:
            import torch
            import VerifyRewrite as _V
            device = device or torch.device('cpu')
            reader = _V.LoadTextModel(device)
        except Exception:
            return SynthesizeText(text, profile, mmPerXh=mmPerXh, seed=seed,
                                  lineWidthMm=lineWidthMm, jitter=jitter,
                                  legibility=legibility)
    import VerifyRewrite as _V
    visible = text.replace('\n', ' ')

    def score(traj):
        img = RenderTrajectory(traj, pxPerMm=pxPerMm, profile=profile,
                               uniformInk=True)
        got = _V.ReadText(reader, img, device)
        return _V.CharAcc(got, visible), img

    base = 0 if seed is None else int(seed)
    best, bestTraj, bestImg = -1.0, None, None
    for k in range(max(1, nTries)):
        traj = SynthesizeText(text, profile, mmPerXh=mmPerXh,
                              seed=base + 977 * k, lineWidthMm=lineWidthMm,
                              jitter=jitter, legibility=legibility)
        sc, img = score(traj)
        if sc > best:
            best, bestTraj, bestImg = sc, traj, img
        if best >= 0.99:
            break

    perChar = {}
    rounds = 0
    if repair:
        for rounds in range(1, REPAIR_ROUNDS + 1):
            if best >= 0.995:
                break
            weak = _WeakChars(reader, bestImg, device, visible)
            # only the SINGLE worst still-pinned character escalates to
            # upright print (last resort); new weak characters get the
            # style-aware anchor
            escalate = [p for p in weak[:1] if perChar.get(p, 0.0) == 1.0]
            fresh = [p for p in weak if p not in perChar][:REPAIR_WORST_K]
            if not escalate and not fresh:
                break
            for p in escalate:
                perChar[p] = 2.0
            for p in fresh:
                perChar[p] = 1.0
            cand = SynthesizeText(text, profile, mmPerXh=mmPerXh, seed=base,
                                  lineWidthMm=lineWidthMm, jitter=jitter,
                                  legibility=legibility, perCharLam=perChar)
            sc, img = score(cand)
            if sc > best:
                best, bestTraj, bestImg = sc, cand, img

    bestTraj.meta['legibilityScore'] = round(float(best), 3)
    bestTraj.meta['legibilityTries'] = k + 1
    bestTraj.meta['legibilityLambda'] = round(float(legibility), 3)
    bestTraj.meta['repairRounds'] = rounds
    bestTraj.meta['repairedChars'] = len(perChar)
    return bestTraj
