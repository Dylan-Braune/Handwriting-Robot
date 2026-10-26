#text+authorId -> pen trajectory in that author's handwriting style, and
#trajectory -> G-code/step-schedule/plotter-preview for the gantry.
#
#covers both halves of "text+author -> trajectory -> G-code": trajectory
#synthesis (originally this file) and G-code generation (merged in from
#the former WriteGCode.py).
#
#consumes the profiles built by BuildStyleProfile.py (glyph prototype
#library + measured style parameters) and emits an ordered list of
#polylines with explicit pen-up/pen-down structure, in millimetres.
#
#how a word is built:
#  * every character comes from one of the author's stored VARIANTS of
#    that glyph (chosen per instance so repeated letters differ like a
#    real hand), falling back to a case/similar-letter neighbour and
#    finally to a built-in skeleton font for characters never written;
#  * letters are placed by the author's own measured per-character
#    advance, scaled by x-height;
#  * for a CURSIVE author (measured connectedness), consecutive letters
#    in a word are joined by a ligature stroke, and the two glyphs plus
#    the ligature are emitted as ONE pen-down polyline;
#  * per-instance jitter (position/rotation/scale noise, baseline drift,
#    slant wobble) keeps repeated text from looking stamped;
#  * the author's slant is applied as a shear at the end, so glyphs stay
#    stored upright and shared spacing maths is slant-independent.
#
#public API (trajectory synthesis): SynthesizeText, SynthesizeLegible,
#SynthesizeJointBestOf, LoadAllProfiles, RenderTrajectory.
#
#=== HARDWARE CONFIG -- see GantryConfig below for the one-block machine
#calibration (motor steps/mm, pen pulse timing, work-area bounds etc).
#the pen is a one-way gear motor with no driver: every 90 degrees of
#rotation TOGGLES the pen, so it is never commanded to a state, only
#pulsed, and this module tracks pen state in software.
#
#public API (G-code generation): WriteGcode, RenderGcodePreview.
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import MyImage
from PIL import Image,ImageDraw,ImageFilter

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ProfileIO import PROFILE_DIR

SCRIPT_DIR = Path(__file__).resolve().parent
#this file lives in Software/CleanPipeline, a sibling of Software/CNN where
#the NOGIT demo-output folder used by the __main__ block below still lives
CNN_DATA_DIR = Path(__file__).resolve().parent.parent/"CNN"

#fallback skeleton font: single-stroke letter shapes in the normalized
#frame the profiles use (x right, y up from baseline, 1.0 = x-height).
#only used for characters an author never wrote
_FB = {
    'a': [[(0.75,0.75),(0.45,0.95),(0.15,0.75),(0.1,0.4),(0.35,0.05),
           (0.65,0.1),(0.78,0.35),(0.78,0.0)],[(0.78,0.3),(0.95,0.0)]],
    'b': [[(0.1,1.8),(0.1,0.0)],[(0.1,0.55),(0.35,0.85),(0.7,0.7),
           (0.75,0.3),(0.5,0.0),(0.15,0.1)]],
    'c': [[(0.8,0.75),(0.5,0.95),(0.18,0.7),(0.15,0.3),(0.45,0.03),
           (0.82,0.2)]],
    'd': [[(0.8,1.8),(0.8,0.0)],[(0.8,0.7),(0.5,0.95),(0.15,0.7),
           (0.15,0.3),(0.45,0.03),(0.8,0.25)]],
    'e': [[(0.15,0.45),(0.8,0.5),(0.65,0.85),(0.3,0.9),(0.12,0.55),
           (0.2,0.15),(0.55,0.0),(0.85,0.15)]],
    'f': [[(0.75,1.75),(0.5,1.85),(0.35,1.55),(0.35,0.0)],
          [(0.1,0.95),(0.7,0.95)]],
    'g': [[(0.8,0.9),(0.8,-0.45),(0.5,-0.62),(0.2,-0.5)],
          [(0.8,0.7),(0.5,0.95),(0.15,0.7),(0.18,0.3),(0.5,0.05),
           (0.8,0.25)]],
    'h': [[(0.1,1.8),(0.1,0.0)],[(0.1,0.6),(0.4,0.9),(0.72,0.7),
           (0.75,0.0)]],
    'i': [[(0.2,0.9),(0.2,0.0)],[(0.2,1.25),(0.2,1.3)]],
    'j': [[(0.35,0.9),(0.35,-0.45),(0.1,-0.6)],[(0.35,1.25),(0.35,1.3)]],
    'k': [[(0.1,1.8),(0.1,0.0)],[(0.72,0.9),(0.1,0.35)],
          [(0.32,0.5),(0.75,0.0)]],
    'l': [[(0.15,1.8),(0.15,0.12),(0.4,0.0)]],
    'm': [[(0.05,0.9),(0.05,0.0)],[(0.05,0.65),(0.28,0.9),(0.5,0.65),
           (0.5,0.0)],[(0.5,0.65),(0.72,0.9),(0.95,0.65),(0.95,0.0)]],
    'n': [[(0.1,0.9),(0.1,0.0)],[(0.1,0.62),(0.4,0.92),(0.75,0.68),
           (0.75,0.0)]],
    'o': [[(0.45,0.95),(0.15,0.7),(0.15,0.28),(0.45,0.02),
           (0.8,0.28),(0.8,0.7),(0.45,0.95)]],
    'p': [[(0.1,0.9),(0.1,-0.6)],[(0.1,0.6),(0.42,0.9),(0.78,0.66),
           (0.75,0.25),(0.42,0.0),(0.12,0.15)]],
    'q': [[(0.82,0.9),(0.82,-0.6)],[(0.82,0.65),(0.5,0.92),
           (0.15,0.66),(0.2,0.25),(0.5,0.0),(0.82,0.2)]],
    'r': [[(0.12,0.9),(0.12,0.0)],[(0.12,0.6),(0.4,0.9),(0.7,0.85)]],
    's': [[(0.78,0.82),(0.42,0.95),(0.18,0.75),(0.45,0.5),
           (0.72,0.38),(0.6,0.05),(0.2,0.1)]],
    't': [[(0.35,1.45),(0.35,0.15),(0.6,0.0)],[(0.1,0.95),(0.65,0.95)]],
    'u': [[(0.1,0.9),(0.1,0.25),(0.4,0.0),(0.72,0.25),(0.72,0.9)],
          [(0.72,0.35),(0.72,0.0)]],
    'v': [[(0.08,0.9),(0.42,0.0),(0.78,0.9)]],
    'w': [[(0.05,0.9),(0.25,0.0),(0.48,0.7),(0.7,0.0),(0.92,0.9)]],
    'x': [[(0.1,0.9),(0.8,0.0)],[(0.8,0.9),(0.1,0.0)]],
    'y': [[(0.1,0.9),(0.1,0.3),(0.42,0.05),(0.75,0.3),(0.75,0.9)],
          [(0.75,0.4),(0.75,-0.45),(0.45,-0.62),(0.18,-0.5)]],
    'z': [[(0.1,0.9),(0.8,0.9),(0.12,0.05),(0.85,0.05)]],
    '.': [[(0.15,0.06),(0.2,0.0)]],
    ',': [[(0.2,0.1),(0.12,-0.22)]],
    "'": [[(0.15,1.75),(0.1,1.35)]],
    '"': [[(0.1,1.75),(0.06,1.4)],[(0.32,1.75),(0.28,1.4)]],
    '-': [[(0.05,0.45),(0.6,0.45)]],
    '!': [[(0.15,1.75),(0.15,0.3)],[(0.15,0.06),(0.18,0.0)]],
    '?': [[(0.08,1.5),(0.3,1.78),(0.6,1.6),(0.55,1.25),(0.32,1.05),
           (0.32,0.75)],[(0.3,0.06),(0.33,0.0)]],
    ':': [[(0.15,0.86),(0.18,0.8)],[(0.15,0.06),(0.18,0.0)]],
    ';': [[(0.15,0.86),(0.18,0.8)],[(0.2,0.1),(0.12,-0.22)]],
    '(': [[(0.5,1.8),(0.2,1.0),(0.5,-0.3)]],
    ')': [[(0.15,1.8),(0.45,1.0),(0.15,-0.3)]],
    '0': [[(0.45,1.7),(0.12,1.3),(0.12,0.4),(0.45,0.0),(0.78,0.4),
           (0.78,1.3),(0.45,1.7)]],
    '1': [[(0.15,1.4),(0.42,1.7),(0.42,0.0)]],
    '2': [[(0.1,1.4),(0.35,1.72),(0.72,1.55),(0.66,1.1),(0.1,0.0),
           (0.8,0.0)]],
    '3': [[(0.12,1.6),(0.5,1.72),(0.72,1.4),(0.4,0.95),(0.75,0.6),
           (0.55,0.05),(0.12,0.15)]],
    '4': [[(0.62,1.7),(0.08,0.55),(0.85,0.55)],[(0.62,1.1),(0.62,0.0)]],
    '5': [[(0.75,1.7),(0.2,1.7),(0.15,1.0),(0.5,1.1),(0.75,0.75),
           (0.6,0.15),(0.15,0.1)]],
    '6': [[(0.72,1.65),(0.3,1.35),(0.15,0.6),(0.42,0.02),(0.75,0.3),
           (0.6,0.72),(0.2,0.72)]],
    '7': [[(0.08,1.7),(0.82,1.7),(0.35,0.0)]],
    '8': [[(0.45,0.85),(0.18,1.2),(0.42,1.7),(0.7,1.25),(0.45,0.85),
           (0.15,0.45),(0.42,0.02),(0.75,0.42),(0.45,0.85)]],
    '9': [[(0.75,1.05),(0.35,1.02),(0.2,1.42),(0.52,1.7),(0.75,1.35),
           (0.7,0.4),(0.3,0.05)]],
}

#cursive baseline alphabet: standard joined-script letterforms, drawn as
#ONE stroke each, entering/exiting at connection height so they join
#legibly. Generic shapes, not any author's -- used when an author's own
#extracted prototype is malformed. x right, y up from baseline, 1.0=x-height
_CX_IN,_CX_OUT = 0.0,0.30
_CURSIVE = {
    'a': [[(0.0,0.30),(0.20,0.62),(0.46,0.72),(0.62,0.52),(0.60,0.18),
           (0.42,0.02),(0.20,0.10),(0.16,0.34),(0.30,0.52),(0.52,0.46),
           (0.62,0.20),(0.64,0.02),(0.86,0.30)]],
    'b': [[(0.0,0.30),(0.14,1.10),(0.24,1.66),(0.30,1.20),(0.26,0.44),
           (0.34,0.10),(0.56,0.04),(0.70,0.24),(0.62,0.48),(0.42,0.52),
           (0.60,0.34),(0.84,0.30)]],
    'c': [[(0.0,0.30),(0.22,0.58),(0.50,0.70),(0.62,0.54),(0.52,0.36),
           (0.30,0.16),(0.34,0.04),(0.58,0.10),(0.80,0.30)]],
    'd': [[(0.0,0.30),(0.22,0.58),(0.46,0.70),(0.58,0.50),(0.56,0.20),
           (0.40,0.04),(0.22,0.14),(0.20,0.40),(0.36,0.58),(0.60,0.60),
           (0.72,1.62),(0.66,0.60),(0.68,0.16),(0.90,0.30)]],
    'e': [[(0.0,0.30),(0.24,0.34),(0.46,0.44),(0.34,0.62),(0.16,0.50),
           (0.18,0.24),(0.38,0.06),(0.62,0.14),(0.78,0.30)]],
    'f': [[(0.0,0.30),(0.18,0.90),(0.34,1.62),(0.44,1.72),(0.46,1.20),
           (0.36,0.30),(0.26,-0.48),(0.14,-0.62),(0.06,-0.40),
           (0.30,-0.20),(0.60,0.14),(0.82,0.30)]],
    'g': [[(0.0,0.30),(0.22,0.58),(0.48,0.70),(0.60,0.50),(0.56,0.20),
           (0.40,0.04),(0.22,0.16),(0.24,0.42),(0.42,0.56),(0.62,0.44),
           (0.64,0.04),(0.58,-0.44),(0.40,-0.62),(0.20,-0.48),
           (0.34,-0.28),(0.66,0.06),(0.86,0.30)]],
    'h': [[(0.0,0.30),(0.14,1.06),(0.24,1.66),(0.30,1.10),(0.26,0.34),
           (0.30,0.06),(0.44,0.42),(0.58,0.62),(0.72,0.48),(0.70,0.14),
           (0.88,0.30)]],
    'i': [[(0.0,0.30),(0.20,0.60),(0.32,0.72),(0.34,0.28),(0.40,0.06),
           (0.58,0.24),(0.66,0.30)],[(0.34,1.06),(0.36,1.14)]],
    'j': [[(0.0,0.30),(0.22,0.62),(0.36,0.74),(0.36,0.10),
           (0.30,-0.44),(0.16,-0.62),(0.04,-0.44),(0.24,-0.26),
           (0.52,0.10),(0.70,0.30)],[(0.38,1.06),(0.40,1.14)]],
    'k': [[(0.0,0.30),(0.14,1.06),(0.24,1.66),(0.30,1.00),(0.26,0.30),
           (0.30,0.04),(0.36,0.36),(0.62,0.60),(0.44,0.34),(0.52,0.14),
           (0.74,0.06),(0.88,0.30)]],
    'l': [[(0.0,0.30),(0.16,1.10),(0.28,1.68),(0.34,1.10),(0.30,0.30),
           (0.36,0.06),(0.58,0.16),(0.74,0.30)]],
    'm': [[(0.0,0.30),(0.10,0.60),(0.18,0.72),(0.22,0.20),(0.30,0.56),
           (0.44,0.70),(0.54,0.52),(0.52,0.16),(0.60,0.54),(0.74,0.70),
           (0.86,0.52),(0.84,0.14),(1.02,0.30)]],
    'n': [[(0.0,0.30),(0.12,0.60),(0.20,0.72),(0.24,0.18),(0.34,0.56),
           (0.50,0.70),(0.62,0.52),(0.60,0.14),(0.78,0.30)]],
    'o': [[(0.0,0.30),(0.20,0.58),(0.44,0.70),(0.60,0.52),(0.58,0.22),
           (0.40,0.04),(0.20,0.16),(0.22,0.44),(0.44,0.58),(0.62,0.50),
           (0.66,0.36),(0.84,0.34)]],
    'p': [[(0.0,0.30),(0.14,0.62),(0.22,0.74),(0.20,0.10),
           (0.14,-0.50),(0.24,-0.20),(0.30,0.40),(0.46,0.66),
           (0.66,0.56),(0.64,0.26),(0.44,0.08),(0.66,0.14),
           (0.84,0.30)]],
    'q': [[(0.0,0.30),(0.22,0.58),(0.46,0.70),(0.60,0.50),(0.56,0.18),
           (0.38,0.04),(0.20,0.18),(0.24,0.44),(0.46,0.58),(0.64,0.42),
           (0.66,0.02),(0.62,-0.44),(0.76,-0.26),(0.90,0.30)]],
    'r': [[(0.0,0.30),(0.14,0.58),(0.24,0.72),(0.26,0.44),(0.36,0.64),
           (0.52,0.66),(0.44,0.48),(0.44,0.14),(0.64,0.28)]],
    's': [[(0.0,0.30),(0.20,0.56),(0.40,0.68),(0.44,0.50),(0.24,0.34),
           (0.20,0.14),(0.40,0.04),(0.60,0.14),(0.72,0.30)]],
    't': [[(0.0,0.30),(0.20,0.90),(0.30,1.44),(0.34,0.90),(0.30,0.20),
           (0.38,0.04),(0.60,0.16),(0.72,0.30)],
          [(0.14,0.86),(0.52,0.90)]],
    'u': [[(0.0,0.30),(0.14,0.62),(0.20,0.72),(0.20,0.20),(0.32,0.04),
           (0.48,0.16),(0.52,0.72),(0.54,0.24),(0.60,0.08),
           (0.80,0.30)]],
    'v': [[(0.0,0.30),(0.14,0.62),(0.22,0.72),(0.34,0.14),(0.50,0.62),
           (0.56,0.44),(0.52,0.28),(0.72,0.34)]],
    'w': [[(0.0,0.30),(0.12,0.62),(0.20,0.72),(0.30,0.12),(0.44,0.66),
           (0.56,0.14),(0.70,0.66),(0.74,0.42),(0.70,0.28),
           (0.90,0.34)]],
    'x': [[(0.0,0.30),(0.16,0.60),(0.28,0.68),(0.56,0.08),
           (0.70,0.26)],[(0.24,0.10),(0.62,0.66)]],
    'y': [[(0.0,0.30),(0.14,0.62),(0.20,0.72),(0.22,0.22),(0.34,0.06),
           (0.50,0.20),(0.54,0.72),(0.52,0.10),(0.44,-0.46),
           (0.28,-0.62),(0.14,-0.44),(0.34,-0.26),(0.62,0.06),
           (0.80,0.30)]],
    'z': [[(0.0,0.30),(0.18,0.60),(0.44,0.62),(0.24,0.24),(0.16,0.06),
           (0.36,0.02),(0.44,-0.34),(0.30,-0.50),(0.20,-0.34),
           (0.46,-0.10),(0.66,0.20),(0.78,0.30)]],
}

_FB_ADV = 1.0

#the gantry writes every author with ONE physical pen at a constant stroke
#width, so ink weight is not a style channel it can reproduce -- when on,
#every author renders at the same width (in x-heights, matched to the
#shape-only writer-ID model's training normalization)
UNIFORM_INK = True
UNIFORM_PEN_WIDTH_XH = 0.085   #thinner than old 0.11 -- a heavy stroke fills small loops

UNIFORM_SOFT_FRAC = 0.30   #blur radius as a fraction of strokePx

#legibility guards (tested in phase 3; no-ops at these defaults)
JOIN_PROB_CAP = 1.0   #cap the per-pair cursive-join probability
CORE_SLANT_CAP_DEG = 22.0   #print-anchor glyphs never shear past this

CORE_JITTER_FRAC = 0.15   #print-anchor glyphs barely wobble
COLLISION_MIN_GAP_XH = 0.05   #forced clear space (x-h) between a glyph and the previous one

#legibility anchor: the hand-drawn single-stroke PRINT font _FB reads at
#~96% char/~84% word through the frozen recognizer, far clearer than any
#data-driven consensus glyph -- so _FB (reshaped to each author's own
#ascender/descender/slant) is the legibility anchor, and `legibility` in
#[0,1] is the probability a glyph is drawn from it instead of the author's
#own hand. lam 0 = pure author, lam 1 = everything printed.

#a stored prototype further than this (cosine distance) from the
#cross-author letter prototype is unusable -- set from the measured
#distribution (p50 0.25, p90 0.41, max 0.92)
_PRIOR_REJECT = 0.25


#generic legible letterform, reproportioned into the author's own style
#(ascender/descender reach; slant+jitter applied later by SynthesizeText).
#printOnly forces the upright print font; otherwise a cursive author keeps
#the joined _CURSIVE skeleton for a letter that never got a clean sample
def _BaselineGlyph(ch,profile,printOnly=False):
    conn = float(profile.get('connectedness',0.0))
    src = _CURSIVE if (conn>=0.5 and not printOnly) else _FB
    key = ch if ch in src else ch.lower()
    if key not in src:
        return None
    asc = float(profile.get('ascender',1.7))
    desc = float(profile.get('descender',-0.6))
    strokes = []
    for st in src[key]:
        q = []
        for (x,y) in st:
            if y>1.0:
                y = 1.0+(y-1.0)*max(0.35,(asc-1.0)/0.75)
            elif y<0.0:
                y = y*max(0.35,abs(desc)/0.55)
            q.append((x,y))
        strokes.append(q)
    xs = [p[0] for st in strokes for p in st]
    adv = (max(xs)-min(xs))+0.12
    return strokes,adv

#how many of a character's stored variants are eligible per instance.
#widened from 2: with only the top two, one unlucky prototype dominated a
#letter everywhere it appeared; 5 gives room to avoid a bad one while
#still favouring the typical forms
_ELIGIBLE_VARIANTS = 5


#pen path in millimetres. y is UP; the caller's writer flips if needed
class Trajectory:

    def __init__(self,strokes,meta):
        self.strokes = strokes   #list of [(x,y),...] pen-DOWN paths
        self.meta = meta

    def Bounds(self):
        pts = [p for s in self.strokes for p in s]
        if not pts:
            return (0.0,0.0,0.0,0.0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs),min(ys),max(xs),max(ys))

    def PenTravelMm(self):
        down = sum(float(np.hypot(*np.diff(np.asarray(s),axis=0).T).sum())
                   for s in self.strokes if len(s)>1)
        up = 0.0
        for a,b in zip(self.strokes,self.strokes[1:]):
            up+=float(math.dist(a[-1],b[0]))
        return down,up

#glyph selection
_SIMILAR = {'I': 'l','O': '0','l': 'I','0': 'O','o': '0',';': ':',
            '"': "'",'!': 'l'}

#a connected hand (measured connectedness >= this) gets the joined cursive
#skeleton as its anchor, so a substituted letter can still connect to its
#neighbours. Tried lowering 0.5->0.3 (2026-09-19): fixed author 150's
#confusion with "dylan" but regressed writer-ID/text/word on the official
#NOVEL_SENTENCES pool -- i.e. it overfit to the specific validation pools.
#reverted to 0.5; see ab_test_log.md change #5. set > 1.0 to disable entirely
CORE_CURSIVE_CONN = 0.5

#how much `legibility` thins out the author's own join rate. ZERO: a
#writer's join rate is a measured property of their hand, not a quality
#knob -- coupling it to legibility silently converted cursive hands into
#print ones, and the joins are most of what makes those hands recognisable
JOIN_LEGIBILITY_COUPLING = 0.0


#the legible anchor letterform for `ch`, reshaped to this author. A
#connected hand (connectedness >= CORE_CURSIVE_CONN) gets the JOINED
#cursive skeleton; a disconnected hand gets upright print. Either way it
#is scaled to the author's ascender/descender reach; slant/spacing/joins
#applied later. Returns (strokes, advance, lead, isCursive).
#preference order: (1) the author's own robust median/medoid for this
#letter (idealGlyphs/idealMedoid, built by BuildStyleProfile), only when
#recognisable (priorD gates it); (2) the generic cursive/print skeleton.
def _CoreGlyph(ch,profile):
    cursive = float(profile.get('connectedness',0.0))>=CORE_CURSIVE_CONN
    src = (profile.get('idealMedoid') if IDEAL_SOURCE=='medoid'
           else profile.get('idealGlyphs')) or {}
    own = src.get(ch)
    if own and float(own.get('priorD',1.0))<=IDEAL_ANCHOR_MAX_PRIORD:
        strokes = [[tuple(p) for p in s] for s in own['strokes']]
        adv = max(float(own.get('advance',0.9)),
                  float(own.get('lead',0.0))+float(own.get('width',0.9))+0.04)
        return strokes,adv,float(own.get('lead',0.0)),cursive
    b = _BaselineGlyph(ch,profile,printOnly=not cursive)
    return None if b is None else (b[0],b[1],0.0,cursive)

#how far the author's OWN ideal letterform may sit from the cross-author
#idea of that letter and still be used as their anchor. per-author
#distributions sit at p50~0.22-0.33, p90~0.32-0.47 -- most "atypical"
#letterforms are just that author's hand, not broken, so the swap is rare
IDEAL_ANCHOR_MAX_PRIORD = 0.30

#'medoid' = the single most typical REAL variant (intact); 'median' = the
#per-stroke median (smoother, but can blur a letter with inconsistent strokes)
IDEAL_SOURCE = 'medoid'

_GLYPH_CLEAN = 0.12   #priorD at/below which the author's letter is clean -- keep it
_GLYPH_GARBAGE = 0.55   #at/above this it is replaced regardless of lam


#P(replace this glyph with the style-aware anchor letterform).
#legibility-first: an author's OWN letterform is only reliably readable
#when it's a genuinely clean extraction (priorD<=_GLYPH_CLEAN); everything
#else swaps to the anchor readily, scaled by `lam`. The anchor still
#carries the author's slant/size/spacing/joins, so the hand isn't erased.
#above this lam the author's library is committed to the anchor ENTIRELY,
#rather than tapering. Measured on the most joined hand: mixing her own
#fragments into anchor strokes mid-line scored WORSE on writer-ID than
#either pure extreme (p=63% mixed vs 76% own-only vs 94% all-anchor), so
#there is no safe middle ground to taper through -- a line that keeps a
#consistent letterform throughout reads as one hand, a line that alternates
#between two sources reads as neither.
_COMMIT_LAM = 0.85


def _SwapProb(lam,priorD):
    d = float(priorD)
    if d>=_GLYPH_GARBAGE:
        return 1.0
    if lam>=_COMMIT_LAM:
        return 1.0
    if lam<=1e-3:
        return 0.0
    if d<=_GLYPH_CLEAN:
        return 0.06*lam
    bad = float(np.clip((d-_GLYPH_CLEAN)/(_GLYPH_GARBAGE-_GLYPH_CLEAN),
                        0.0,1.0))
    return float(np.clip(lam*(0.55+0.9*bad),0.0,1.0))


#how good an author's OWN extracted glyph library is, and therefore how far
#to trust it. The median priorD of all their variants is the measurement:
#a well separated hand cuts cleanly into one glyph per character and sits
#near 0.18-0.22, a heavily joined hand has no gap to cut at, so its
#"variants" are arbitrary fragments of one continuous curve and the median
#climbs past 0.30 (measured over the 13 stored writers: owen 0.191,
#abhinav 0.180, yeukita 0.224, writer 153 0.242, writer 151 0.307,
#thiya 0.318). The ramp below therefore leaves the clean hands on their own
#letterforms and sends the broken ones to the style-aware anchor, instead of
#applying one hand-set legibility to writers whose libraries are not
#comparably trustworthy.
_AUTO_LAM_GOOD = 0.24   #median priorD at/below which the library is trusted
_AUTO_LAM_BAD = 0.32   #at/above this the library is fragments, not letters


def AutoLegibility(profile):
    ds = [float(g.get('priorD',0.5))
          for v in profile.get('glyphs',{}).values() for g in v]
    if not ds:
        return 1.0
    med = float(np.median(ds))
    return float(np.clip((med-_AUTO_LAM_GOOD)/(_AUTO_LAM_BAD-_AUTO_LAM_GOOD),
                         0.0,1.0))


#returns (strokes, advance, lead, source). Uses one of the author's own
#variants, unless the prototype is unresolvable, forceCore (repair pass ->
#style-aware anchor), forcePrint (repair pass, still unreadable -> upright
#print), or -- rarely -- _SwapProb fires.
def _GlyphSource(profile,ch,rng,prevExitY=None,lam=0.0,forceCore=False,
                 forcePrint=False):
    lib = profile['glyphs']
    adv = profile.get('letterAdvance',{})
    if forcePrint:
        b = _BaselineGlyph(ch,profile,printOnly=True)
        if b is not None:
            return b[0],b[1],0.0,'core'

    def pack(g,k=1.0):
        strokes = [[(x*k,y*k) for (x,y) in s] for s in g['strokes']]
        #the variant's OWN advance, never the character median -- a wide
        #variant advanced by a narrow median would collide with the next letter
        a = max(float(g.get('advance',0.9)),
                float(g.get('lead',0.0))+float(g.get('width',0.9))+0.04)
        return strokes,a*k,float(g.get('lead',0.0))*k

    #favour the more typical variants (sorted by typicality) while still
    #varying -- uniform choice over a noisy tail makes synthesized cursive
    #look scrambled. When joining onto a previous letter, also favour the
    #variant whose own entry height matches where the pen actually is.
    def pick(cands):
        #only the top few are eligible: a writer forms a given letter
        #consistently, so sampling uniformly over a dozen variants
        #averages their signature away
        cands = cands[:_ELIGIBLE_VARIANTS]
        w = []
        for i,g in enumerate(cands):
            wi = 1.0/(1.0+1.4*i)
            if prevExitY is not None:
                dy = abs(float(g.get('entryY',0.0))-prevExitY)
                wi*=math.exp(-(dy/0.45)**2)
            w.append(wi)
        tot = sum(w)
        if tot<=1e-9:
            return cands[0]
        r = rng.random()*tot
        acc = 0.0
        for i,wi in enumerate(w):
            acc+=wi
            if r<=acc:
                return cands[i]
        return cands[-1]

    if ch in lib and lib[ch]:
        cands = lib[ch]
        #rank by priorD, take the most well-formed the author actually wrote
        ranked = sorted(cands,key=lambda g: g.get('priorD',0.5))
        g = pick(ranked)
        d = float(g.get('priorD',0.5))
        core = _CoreGlyph(ch,profile)
        if core is not None and (forceCore
                                 or rng.random()<_SwapProb(lam,d)):
            return core[0],core[1],core[2],'core'
        s,a,lead = pack(g)
        return s,a,lead,'own'
    for alt in (ch.swapcase(),_SIMILAR.get(ch,'')):
        if alt and alt in lib and lib[alt]:
            core = _CoreGlyph(ch,profile)
            if core is not None and (forceCore
                                     or rng.random()<_SwapProb(lam,0.4)):
                return core[0],core[1],core[2],'core'
            g = pick(sorted(lib[alt],key=lambda x: x.get('priorD',0.5)))
            if alt==ch.swapcase() and ch.isupper():
                #borrow the lowercase shape, grown to capital height
                k = profile.get('ascender',1.7)/max(0.6,g['top'] or 1.0)
                k = float(np.clip(k,1.0,2.0))
                s,a,lead = pack(g,k)
                return s,a,lead,'case'
            s,a,lead = pack(g)
            return s,a,lead,'similar'
    core = _CoreGlyph(ch,profile)
    if core is not None:
        return core[0],core[1],core[2],'core'
    key = ch.lower() if ch.lower() in _FB else ch
    if key in _FB:
        strokes = [list(map(tuple,s)) for s in _FB[key]]
        return strokes,adv.get(ch,_FB_ADV),0.0,'fallback'
    return [],adv.get(ch,0.6),0.0,'blank'


#synthesis
def _Jitter(strokes,rng,amp,rotAmp,scaleAmp):
    dx = rng.gauss(0.0,amp)
    dy = rng.gauss(0.0,amp*0.6)
    th = math.radians(rng.gauss(0.0,rotAmp))
    sx = 1.0+rng.gauss(0.0,scaleAmp)
    sy = 1.0+rng.gauss(0.0,scaleAmp)
    c,s = math.cos(th),math.sin(th)
    out = []
    for stk in strokes:
        q = []
        for (x,y) in stk:
            x2,y2 = x*sx,y*sy
            q.append((x2*c-y2*s+dx,x2*s+y2*c+dy))
        out.append(q)
    return out


#even arc-length resampling: smooth motion for the plotter, and it also
#makes the rendered stroke width uniform
def _Resample(poly,step):
    if len(poly)<2:
        return poly
    p = np.asarray(poly,np.float64)
    seg = np.hypot(*np.diff(p,axis=0).T)
    cum = np.concatenate([[0.0],np.cumsum(seg)])
    total = float(cum[-1])
    if total<1e-9:
        return [tuple(p[0])]
    n = max(2,int(round(total/step))+1)
    t = np.linspace(0.0,total,n)
    x = np.interp(t,cum,p[:,0])
    y = np.interp(t,cum,p[:,1])
    return list(zip(x.tolist(),y.tolist()))


#text -> Trajectory (mm, y up, origin at first baseline). mmPerXh: physical
#size of one x-height. lineWidthMm: wrap width. legibility=None (default)
#reads the author's own tuned profile['legibilityLambda'], falling back to
#AutoLegibility(profile), which measures how trustworthy that author's own
#extracted glyph library actually is -- pass an explicit float to override
#for one call, e.g. an A/B test.
def SynthesizeText(text,profile,mmPerXh=4.0,seed=None,lineWidthMm=180.0,
                   jitter=0.15,legibility=None,perCharLam=None,_cal=None):
    if legibility is None:
        legibility = float(profile.get('legibilityLambda',
                                       AutoLegibility(profile)))
    rng = random.Random(seed)
    xh = float(mmPerXh)
    cal = _cal if _cal is not None else (
        StyleCalibration(profile,mmPerXh) if USE_STYLE_CALIBRATION
        else dict(shearDelta=0.0,ascK=1.0,descK=1.0))
    slantDeg = profile.get('slantDeg',0.0)
    #legibility caps the lean of the AUTHOR's own glyphs on a soft curve
    #(45deg at lam 0 -> ~26deg at lam 1); the print anchor is capped harder
    #(CORE_SLANT_CAP_DEG) further down
    authorSlantCap = 45.0-19.0*float(np.clip(legibility,0.0,1.0))
    slantDeg = float(np.clip(slantDeg,-authorSlantCap,authorSlantCap))
    slant = math.tan(math.radians(slantDeg))+cal['shearDelta']
    conn = float(profile.get('connectedness',0.0))
    #legibility thins out the cursive joins (tangled joins are the main
    #thing that makes a connected hand unreadable) without removing them
    connEff = conn*(1.0-JOIN_LEGIBILITY_COUPLING
                      *float(np.clip(legibility,0.0,1.0)))
    wordGap = float(profile.get('wordSpaceXh',1.2))
    ascender = float(profile.get('ascender',1.7))
    descender = float(profile.get('descender',-0.6))
    lineStep = (ascender-descender+0.85)*xh

    ampP = 0.022*jitter   #position jitter, x-heights
    ampR = 2.2*jitter   #rotation jitter, degrees
    ampS = 0.05*jitter   #scale jitter
    driftAmp = 0.05*jitter   #slow baseline drift, x-heights

    strokes = []
    penX,penY = 0.0,0.0
    lineIdx = 0
    drift = 0.0
    driftV = 0.0
    usage = {}
    vpos = 0   #index into text.replace('\n',' '), for perCharLam

    def flush(chain):
        if len(chain)>=2:
            strokes.append(_Resample(chain,step=0.35))
        elif len(chain)==1:
            strokes.append(chain+[(chain[0][0]+0.05,chain[0][1])])

    for word in _SplitWords(text):
        if word=='\n':
            lineIdx+=1
            penX = 0.0
            penY = -lineIdx*lineStep
            vpos+=1
            continue
        wWidth = _WordWidth(word,profile)*xh
        if penX>1e-6 and penX+wWidth>lineWidthMm:
            lineIdx+=1
            penX = 0.0
            penY = -lineIdx*lineStep
        chain = []   #current pen-down polyline
        prevExit = None
        prevBodyMaxX = None
        prevSrc = None
        prevCoreCursive = False
        for ci,ch in enumerate(word):
            lam = legibility
            forceCore = forcePrint = False
            if perCharLam:
                pc = float(perCharLam.get(vpos,0.0))
                lam = max(lam,min(pc,1.0))
                forceCore = pc>=0.999
                forcePrint = pc>=1.9
            vpos+=1
            wantEntryY = None
            if connEff>=0.2 and ci>0 and prevExit is not None and \
                    ch.isalpha() and word[ci-1].isalpha():
                wantEntryY = (prevExit[1]-penY)/xh-drift
            gStrokes,adv,lead,src = _GlyphSource(profile,ch,rng,
                                                    prevExitY=wantEntryY,
                                                    lam=lam,forceCore=forceCore,
                                                    forcePrint=forcePrint)
            usage[src] = usage.get(src,0)+1
            if not gStrokes:
                penX+=adv*xh
                continue
            isCore = (src=='core')
            coreCursive = isCore and conn>=CORE_CURSIVE_CONN
            #the anchor is a designed-legible shape: don't re-stretch its
            #proportions or wobble it as hard as an extracted glyph. It DOES
            #keep the author's lean, just capped so ascenders don't ramp off
            if not isCore:
                gStrokes = _ApplyCal(gStrokes,cal)
            jf = CORE_JITTER_FRAC if isCore else 1.0
            gStrokes = _Jitter(gStrokes,rng,ampP*jf,ampR*jf,ampS*jf)
            glyphSlant = slant
            glyphDrift = drift
            if isCore:
                capDeg = 30.0 if coreCursive else CORE_SLANT_CAP_DEG
                cap = math.tan(math.radians(capDeg))
                glyphSlant = max(-cap,min(cap,slant))
                glyphDrift = (0.5 if coreCursive else 0.25)*drift
            #slow drift of the baseline within a line (a real hand wanders)
            driftV = 0.85*driftV+rng.gauss(0.0,driftAmp*0.35)
            drift = float(np.clip(drift+driftV,-0.25,0.25))

            def place(px):
                out = []
                for stk in gStrokes:
                    out.append([(px+(lead+x+glyphSlant*y)*xh,
                                 penY+(y+glyphDrift)*xh) for (x,y) in stk])
                return out

            placed = place(penX)
            #collision guard: if this glyph's ink would land on top of the
            #previous glyph's body, push it (and the pen) right for clearance
            if COLLISION_MIN_GAP_XH is not None and prevBodyMaxX is not None:
                curMinX = min(p[0] for s in placed for p in s)
                need = prevBodyMaxX+COLLISION_MIN_GAP_XH*xh-curMinX
                if need>0:
                    penX+=need
                    placed = place(penX)

            #the BODY stroke (widest in x) carries the cursive connection;
            #dots/crossbars/accents are separate pen-downs and must not
            #become the exit point for the next letter
            bodyI = int(np.argmax([max(p[0] for p in s)-min(p[0] for p in s)
                                   for s in placed]))
            body = placed[bodyI]
            #connectedness is the measured FRACTION of adjacent letter pairs
            #this author actually joins, used as a PROBABILITY per pair, not
            #an on/off threshold (a threshold fused whole words into blobs).
            #join through a pair when both glyphs are the author's own
            #extracted forms, OR both are the cursive anchor -- a PRINT
            #anchor glyph is always read as a separate letter, never joined
            bothOwn = not isCore and prevSrc!='core'
            bothCursiveCore = coreCursive and prevCoreCursive
            joinable = (ci>0 and prevExit is not None
                        and ch.isalpha() and word[ci-1].isalpha()
                        and (bothOwn or bothCursiveCore)
                        and rng.random()<min(connEff,JOIN_PROB_CAP))
            if joinable:
                chain+=_Ligature(prevExit,body[0],xh,rng)
                chain+=body
            else:
                flush(chain)
                chain = list(body)
            prevExit = chain[-1]
            for k2,extra in enumerate(placed):
                if k2!=bodyI:
                    flush(extra)
            prevBodyMaxX = max(p[0] for p in body)
            prevSrc = src
            prevCoreCursive = coreCursive
            penX+=adv*xh
        flush(chain)
        penX+=wordGap*xh*(0.85+0.3*rng.random())
        vpos+=1   #the space between words

    meta = dict(author=profile.get('authorId'),mmPerXh=mmPerXh,
                nStrokes=len(strokes),glyphSources=usage,
                slantDeg=profile.get('slantDeg',0.0),
                connectedness=conn,lines=lineIdx+1,
                ascender=ascender,descender=descender)
    return Trajectory(strokes,meta)

#line-level slant/ascender calibration: correcting measured WORSE against
#the ink-sensitive writer-ID model (86.0%->84.5%) but BETTER against the
#shape-only model that ignores pen weight (59.2%->63.3%, 58.3%->66.7%
#through G-code). The shape-only judge is the one that predicts the
#physical robot (same pen for every author), so this stays ON.
USE_STYLE_CALIBRATION = True

_STYLE_CAL = {}


#correct the two line-level style statistics synthesis drifts on: glyphs
#are stored per-instance, so a line assembled from them doesn't
#automatically reproduce the author's line-level slant/ascender reach.
#renders a probe, measures it the same way real pages were measured, and
#returns the shear delta + ascender/descender scaling that correct the
#drift. Cached per author.
def StyleCalibration(profile,mmPerXh=4.0,pxPerMm=18.0):
    key = (profile.get('authorId'),round(mmPerXh,3),round(pxPerMm,3))
    if key in _STYLE_CAL:
        return _STYLE_CAL[key]
    import ProfileIO as _SP
    ref = (profile.get('slantMeasRef'),profile.get('ascMeasRef'),
           profile.get('descMeasRef'))
    cal = dict(shearDelta=0.0,ascK=1.0,descK=1.0)
    if None in ref or any(r!=r for r in ref):
        _STYLE_CAL[key] = cal
        return cal
    slantRef,ascRef,descRef = ref
    sample = ('the quick brown fox jumps over lazy dogs and by half past '
              'eight it kept flying quietly')
    for _ in range(3):
        traj = SynthesizeText(sample,profile,mmPerXh=mmPerXh,seed=17,
                              lineWidthMm=10_000.0,jitter=0.0,_cal=cal)
        img = RenderTrajectory(traj,pxPerMm=pxPerMm,profile=profile)
        ink = _SP.BinarizeLine(np.array(img))
        band = _SP.CoreBand(ink)
        if band is None or not ink.any():
            break
        top,base = band
        xh = float(base-top)
        if xh<4:
            break
        ys = np.nonzero(ink.any(axis=1))[0]
        gotSlant = _SP.EstimateSlantDeg(ink)
        gotAsc = float(base-ys.min())/xh
        gotDesc = float(base-ys.max())/xh
        cal = dict(
            shearDelta=cal['shearDelta']+
            math.tan(math.radians(slantRef))-math.tan(math.radians(gotSlant)),
            ascK=float(np.clip(cal['ascK']*(ascRef-1.0)/
                               max(1e-3,gotAsc-1.0),0.5,1.6)),
            descK=float(np.clip(cal['descK']*descRef/min(-1e-3,gotDesc),
                                0.5,1.6)))
    _STYLE_CAL[key] = cal
    return cal


#compress/stretch only what lies OUTSIDE the x-height band, so ascender
#and descender reach change without disturbing x-height
def _ApplyCal(strokes,cal):
    ascK,descK = cal['ascK'],cal['descK']
    if abs(ascK-1.0)<1e-3 and abs(descK-1.0)<1e-3:
        return strokes
    out = []
    for s in strokes:
        q = []
        for (x,y) in s:
            if y>1.0:
                y = 1.0+(y-1.0)*ascK
            elif y<0.0:
                y = y*descK
            q.append((x,y))
        out.append(q)
    return out


#the carry-across stroke between two joined letters. It sags only in
#proportion to how far it has to travel -- a fixed dip turns a short join
#into a loop, which is what makes synthesized cursive look tangled
def _Ligature(p0,p1,xh,rng):
    dx = p1[0]-p0[0]
    if abs(dx)<0.06*xh:
        return [p0,p1]
    sag = min(0.11*xh,0.16*abs(dx))+rng.gauss(0.0,0.015*xh)
    mid = ((p0[0]+p1[0])*0.5,min(p0[1],p1[1])-sag)
    return _Resample([p0,mid,p1],step=0.35)


def _SplitWords(text):
    out = []
    for chunk in text.split('\n'):
        out+=[w for w in chunk.split(' ') if w]
        out.append('\n')
    return out[:-1] if out and out[-1]=='\n' else out


def _WordWidth(word,profile):
    adv = profile.get('letterAdvance',{})
    return sum(adv.get(c,0.9) for c in word)


#rendering (must look like the training crops: black ink on white)
def _Draw(traj,pxPerMm,padMm,strokePx,paper=255,ink=0,softPx=0.0):
    x0,y0,x1,y1 = traj.Bounds()
    W = max(8,int((x1-x0+2*padMm)*pxPerMm))
    H = max(8,int((y1-y0+2*padMm)*pxPerMm))
    #drawn with my own rasteriser, not a library.  this image is what both
    #judges score in the best-of-N search, so how the ink is laid down is
    #part of the design and not a detail
    arr = np.full((H,W),np.uint8(np.clip(round(float(paper)),0,255)),np.uint8)
    fill = int(round(ink))
    for s in traj.strokes:
        #trajectory is in mm with y UP, the image is in pixels with y DOWN,
        #so y flips here: (y1-y) measures down from the top of the bounds
        pts = [((x-x0+padMm)*pxPerMm,(y1-y+padMm)*pxPerMm)
               for (x,y) in s]
        #DrawPolyline measures distance to the stroke so the round pen ends
        #and round joins come out on their own, no separate end caps needed
        MyImage.DrawPolyline(arr,pts,fill,strokePx)
    if softPx>0.05:
        #a real pen edge is not a hard step, the blur softens it.  sigma is
        #a fraction of the pen width so a thick pen gets a wider soft edge
        arr = MyImage.Blur(arr,softPx)
    return Image.fromarray(arr)


#render as a training-style crop (black ink, white paper). With
#`uniformInk` (default UNIFORM_INK) every author is drawn at one constant
#stroke width -- what the single-pen gantry actually produces, and the
#domain the shape-only writer-ID model is trained on; per-author ink
#fields are ignored. With uniformInk=False the legacy behaviour searches
#the stroke width to reproduce the author's measured ink density.
def RenderTrajectory(traj,pxPerMm=8.0,padMm=None,strokePx=None,
                     profile=None,size=None,matchInk=True,uniformInk=None):
    prof = profile or {}
    mm = traj.meta.get('mmPerXh',4.0)
    if uniformInk is None:
        uniformInk = UNIFORM_INK
    if padMm is None:
        #padding proportional to the writing, not a fixed slab -- a fixed
        #pad would inflate the crop height and skew the line's aspect ratio
        padMm = 0.18*mm

    if uniformInk:
        sp = strokePx or max(1,int(round(UNIFORM_PEN_WIDTH_XH*mm*pxPerMm)))
        img = _Draw(traj,pxPerMm,padMm,sp,paper=255,ink=0,
                    softPx=UNIFORM_SOFT_FRAC*max(1.0,sp))
        if size is not None:
            #my own resampler, see MyImage.Resize
            img = Image.fromarray(MyImage.Resize(np.asarray(img,dtype=np.uint8),size[0],size[1],'linear'))
        return img

    if strokePx is None:
        sw = prof.get('strokeWidthXh',0.12)
        strokePx = max(1,int(round(sw*mm*pxPerMm)))

    paper = prof.get('paperLevel',255.0)
    #NOTE: profile's inkLevel is the median of the real crop's already-dark
    #pixels (a whole-stroke average including soft edges), NOT the stroke
    #core colour -- painting whole strokes at it wrecks density calibration
    #(measured 76%->46% writer-ID), so the core is dark and soft edge comes
    #from the blur instead
    inkLv = prof.get('inkCoreLevel',0.0)
    soft = prof.get('softScale',0.30)*max(1.0,strokePx)
    target = prof.get('inkFracRef') if matchInk else None
    if target:
        img,strokePx = _MatchInkDensity(traj,pxPerMm,padMm,strokePx,
                                         target,mm,paper,inkLv,soft)
    else:
        img = _Draw(traj,pxPerMm,padMm,strokePx,paper,inkLv,soft)
    if size is not None:
        #my own resampler, see MyImage.Resize
        img = Image.fromarray(MyImage.Resize(np.asarray(img,dtype=np.uint8),size[0],size[1],'linear'))
    return img


#choose the stroke width whose rendered ink density matches the author's
#real pages, by searching on THIS line. Has to be a search, not a formula:
#after classifier downsampling a thin stroke anti-aliases to light grey
#and drops out of the <128 ink test, so density is a step-like function
#of width -- and it has to be on this line (not a fixed probe) since
#density also depends on how much text the line holds
def _MatchInkDensity(traj,pxPerMm,padMm,startPx,target,mm,
                     paper=255,ink=0,soft=0.0):
    from Classify import ResizeLineImageFixed

    hi = max(3,int(0.55*mm*pxPerMm))
    cache = {}

    def probe(px):
        px = int(max(1,min(hi,px)))
        if px not in cache:
            im = _Draw(traj,pxPerMm,padMm,px,paper,ink,
                       0.30*max(1.0,px))
            got = float((np.array(ResizeLineImageFixed(im))<128).mean())
            cache[px] = (im,got,abs(got-target))
        return cache[px]

    cur = int(max(1,min(hi,startPx)))
    im,got,err = probe(cur)
    step = -1 if got>target else 1
    while 1<=cur+step<=hi:
        im2,got2,err2 = probe(cur+step)
        if err2>=err:
            break
        cur,im,got,err = cur+step,im2,got2,err2
    return im,cur


def LoadAllProfiles():
    profs = {}
    for p in sorted(PROFILE_DIR.glob('*.json')):
        with open(p,encoding='utf-8') as f:
            profs[p.stem] = json.load(f)
    return profs

if __name__=='__main__':
    text = sys.argv[1] if len(sys.argv)>1 else \
        'The quick brown fox jumps over the lazy dog'
    profs = LoadAllProfiles()
    outDir = CNN_DATA_DIR/'NOGIT'/'SynthPreview'
    outDir.mkdir(parents=True,exist_ok=True)
    for a,prof in profs.items():
        traj = SynthesizeText(text,prof,seed=1)
        RenderTrajectory(traj,profile=prof).save(outDir/f'{a}.png')
        print(f"{a}: {traj.meta['nStrokes']} strokes {traj.meta['glyphSources']}")
    print('wrote',outDir)

#legibility-first synthesis
REPAIR_ROUNDS = 3   #targeted re-draw passes after best-of-N
REPAIR_WORST_K = 3   #characters pinned to the legible core per round

#how hard to lean on the print anchor when a profile carries no explicit
#legibilityLambda. Measured sweet spot ~0.65 gives ~86% char/~48% word
#aggregate on novel text, vs ~77%/~37% at the previous per-author scheme
DEFAULT_LEGIBILITY = 0.65


#(T, C) log-probs from the frozen recognizer for one line image
def _LineLogProbs(reader,img):
    from Classify import ResizeLineImageFixed,TensorFromResized
    t = TensorFromResized(ResizeLineImageFixed(img))
    return reader.forward(t)[:,0,:]


#visible-text positions the recognizer is least sure about, worst first.
#uses CTC forced alignment against the intended text and scores each
#character by its own class log-prob over its assigned frames
def _WeakChars(reader,img,visible):
    try:
        from ProfileIO import CtcForcedAlign
        from Charset import CHAR_TO_IDX
    except Exception:
        return []
    lp = _LineLogProbs(reader,img)
    res = CtcForcedAlign(lp,visible)
    if res is None:
        return []
    align,_conf = res
    keptPos = [j for j,c in enumerate(visible) if c in CHAR_TO_IDX]
    scored = []
    for i,(c,sp) in enumerate(align):
        if i>=len(keptPos) or not c.strip():
            continue
        if sp is None:
            scored.append((-1e9,keptPos[i]))
        else:
            scored.append((float(lp[sp[0]:sp[1],CHAR_TO_IDX[c]].mean()),
                           keptPos[i]))
    scored.sort()
    return [pos for _s,pos in scored]


#best-of-N in the author's style, then a targeted repair pass:
#1. draw the line `nTries` times (same library + measured params,
#   different variant/jitter draws) at blend level `legibility`, keep the
#   one the frozen recognizer reads best;
#2. repair: find characters the recognizer still can't read, pin just
#   those to the legible core (lam=1) and re-draw, keep if it reads
#   better. Repeat up to REPAIR_ROUNDS times.
#every candidate is a real rendering in this hand -- repair only cleans up
#unreadable letters, it does not restyle the line. Falls back to a single
#plain synthesis if the recognizer is missing.
def SynthesizeLegible(text,profile,nTries=6,mmPerXh=4.0,lineWidthMm=180.0,
                      jitter=0.5,seed=0,reader=None,
                      pxPerMm=18.0,legibility=None,repair=True):
    if legibility is None:
        legibility = float(profile.get('legibilityLambda',DEFAULT_LEGIBILITY))
    if reader is None:
        try:
            from Classify import PaperCRNNNumpy
            reader = PaperCRNNNumpy()
        except Exception:
            return SynthesizeText(text,profile,mmPerXh=mmPerXh,seed=seed,
                                  lineWidthMm=lineWidthMm,jitter=jitter,
                                  legibility=legibility)
    from Classify import ReadText,CharAcc
    visible = text.replace('\n',' ')

    def score(traj):
        img = RenderTrajectory(traj,pxPerMm=pxPerMm,profile=profile,
                               uniformInk=True)
        got = ReadText(img,reader)
        return CharAcc(got,visible),img

    base = 0 if seed is None else int(seed)
    best,bestTraj,bestImg = -1.0, None, None
    for k in range(max(1,nTries)):
        traj = SynthesizeText(text,profile,mmPerXh=mmPerXh,
                              seed=base+977*k,lineWidthMm=lineWidthMm,
                              jitter=jitter,legibility=legibility)
        sc,img = score(traj)
        if sc>best:
            best,bestTraj,bestImg = sc,traj,img
        if best>=0.99:
            break

    perChar = {}
    rounds = 0
    if repair:
        for rounds in range(1,REPAIR_ROUNDS+1):
            if best>=0.995:
                break
            weak = _WeakChars(reader,bestImg,visible)
            #only the SINGLE worst still-pinned character escalates to
            #upright print (last resort); new weak characters get the
            #style-aware anchor
            escalate = [p for p in weak[:1] if perChar.get(p,0.0)==1.0]
            fresh = [p for p in weak if p not in perChar][:REPAIR_WORST_K]
            if not escalate and not fresh:
                break
            for p in escalate:
                perChar[p] = 2.0
            for p in fresh:
                perChar[p] = 1.0
            cand = SynthesizeText(text,profile,mmPerXh=mmPerXh,seed=base,
                                  lineWidthMm=lineWidthMm,jitter=jitter,
                                  legibility=legibility,perCharLam=perChar)
            sc,img = score(cand)
            if sc>best:
                best,bestTraj,bestImg = sc,cand,img

    bestTraj.meta['legibilityScore'] = round(float(best),3)
    bestTraj.meta['legibilityTries'] = k+1
    bestTraj.meta['legibilityLambda'] = round(float(legibility),3)
    bestTraj.meta['repairRounds'] = rounds
    bestTraj.meta['repairedChars'] = len(perChar)
    return bestTraj


#best-of-N, like SynthesizeLegible, but scored by the HARMONIC MEAN of
#text-recognizer accuracy and writer-ID confidence for `author`, instead
#of text accuracy alone.
#
#WHY HARMONIC MEAN, specifically: plain best-of-N (score by legibility
#only) raises text accuracy but can *cost* writer-ID, because the single
#most-legible draw out of N random jitter/variant draws has no reason to
#also be the most distinctive one -- for some authors the two axes pull in
#different directions across the N candidates, and a score that only looks
#at text always takes the legibility side of that coin flip. The harmonic
#mean of (textAcc, widConf) is the right combinator for this because it is
#dominated by whichever of the two is SMALLER: a candidate that is
#brilliant on one axis and poor on the other still gets a LOW combined
#score (unlike the arithmetic mean, which would let a 0.95/0.30 candidate
#look as good as a balanced 0.65/0.65 one). So best-of-N under the harmonic
#mean is forced to pick a draw that is a genuine compromise on BOTH axes,
#not a draw that merely wins on one of them. Measured (10-author check) to
#raise text accuracy on every author with NO writer-ID regression, and
#raised writer-ID for several authors already at 33-67% up to 100%.
#
#the repair pass that follows is JOINT-AWARE for the same reason: unlike
#SynthesizeLegible's plain repair (which pins weak characters to a
#print/anchor shape purely by text score, and was measured to cost
#writer-ID for authors whose profile doesn't lean on the anchor), a
#proposed repair here is only ACCEPTED if the candidate's joint score
#improves -- a repair that fixes a letter but makes the line look like
#someone else is rejected instead of silently taking the writer-ID hit.
def SynthesizeJointBestOf(author,text,profile,nTries=6,mmPerXh=4.0,
                          lineWidthMm=180.0,jitter=0.5,seed=0,
                          reader=None,authorModel=None,authorMapping=None,
                          pxPerMm=18.0,legibility=None,
                          repair=True,repairRounds=3,repairWorstK=2):
    if legibility is None:
        legibility = float(profile.get('legibilityLambda',
                                       AutoLegibility(profile)))
    if reader is None:
        from Classify import PaperCRNNNumpy
        reader = PaperCRNNNumpy()
    if authorModel is None:
        from Classify import AuthorClassifierCNNNumpy
        authorModel = AuthorClassifierCNNNumpy()
    if authorMapping is None:
        authorMapping = authorModel.author_mapping
    from Classify import ReadText,CharAcc
    from Classify import ClassifyImage
    visible = text.replace('\n',' ')
    authorIdx = authorMapping[author]

    #the harmonic-mean combinator described above: 2*a*b/(a+b) rather than
    #(a+b)/2, specifically so a candidate weak on EITHER axis is penalized,
    #not averaged away by strength on the other axis
    def jointScore(img):
        textAcc = CharAcc(ReadText(img,reader),visible)
        _predIdx,probs = ClassifyImage(img,authorModel)
        widConf = float(probs[authorIdx])
        combined = (2*textAcc*widConf/(textAcc+widConf)
                   if (textAcc+widConf)>1e-9 else 0.0)
        return combined,textAcc,widConf

    base = 0 if seed is None else int(seed)
    best,bestTraj,bestImg = -1.0, None, None
    for k in range(max(1,nTries)):
        traj = SynthesizeText(text,profile,mmPerXh=mmPerXh,
                              seed=base+977*k,lineWidthMm=lineWidthMm,
                              jitter=jitter,legibility=legibility)
        img = RenderTrajectory(traj,pxPerMm=pxPerMm,profile=profile,
                               uniformInk=True)
        combined,_textAcc,_widConf = jointScore(img)
        if combined>best:
            best,bestTraj,bestImg = combined,traj,img
        if best>=0.99:
            break

    perChar = {}
    if repair:
        from ProfileIO import CtcForcedAlign
        from Charset import CHAR_TO_IDX
        for _round in range(repairRounds):
            if best>=0.995:
                break
            weak = _WeakChars(reader,bestImg,visible)
            fresh = [p for p in weak if p not in perChar][:repairWorstK]
            if not fresh:
                break
            trial = dict(perChar)
            for p in fresh:
                trial[p] = 1.0
            cand = SynthesizeText(text,profile,mmPerXh=mmPerXh,seed=base,
                                  lineWidthMm=lineWidthMm,jitter=jitter,
                                  legibility=legibility,perCharLam=trial)
            img = RenderTrajectory(cand,pxPerMm=pxPerMm,profile=profile,
                                   uniformInk=True)
            combined,_textAcc,_widConf = jointScore(img)
            if combined>best:
                best,bestTraj,bestImg = combined,cand,img
                perChar = trial
            #accepted or not, don't retry the same characters again

    bestTraj.meta['jointScore'] = round(float(best),3)
    bestTraj.meta['jointTries'] = k+1
    bestTraj.meta['jointRepairedChars'] = len(perChar)
    bestTraj.meta['legibilityLambda'] = round(float(legibility),3)
    return bestTraj

#===========================================================================
#MACHINE CONFIGURATION
#(merged in from the former G-code module -- see the module comment block's
#hardware-config section above)
#===========================================================================
@dataclass
class GantryConfig:
    #--- motion transmission -------------------------------------------
    fullStepsPerRev: int = 200   #NEMA 17
    microstepping: int = 16   #A4988 MS1/MS2/MS3 = 1/16
    mmPerRevX: float = 40.0   #GT2 2mm pitch x 20 teeth
    mmPerRevY: float = 40.0
    #--- work area ------------------------------------------------------
    boundsMinXmm: float = 0.0
    boundsMinYmm: float = 0.0
    boundsMaxXmm: float = 200.0
    boundsMaxYmm: float = 200.0
    originXmm: float = 10.0   #where text starts on the bed
    originYmm: float = 180.0
    #--- speeds ---------------------------------------------------------
    drawFeedMmMin: float = 900.0   #pen-down feed
    travelFeedMmMin: float = 2400.0   #pen-up rapid
    accelMmS2: float = 400.0   #trapezoidal accel limit
    minSegmentMm: float = 0.12   #collapse shorter moves
    #--- pen actuator (one-way gear motor, 90 deg per toggle) -----------
    penPulseMs: float = 260.0   #time for one 90-degree rotation
    penSettleMs: float = 90.0   #dwell after a toggle
    penUpCode: str = 'M5'   #firmware maps these to the pulse
    penDownCode: str = 'M3'
    penStartsUp: bool = True   #assumed state at power-on

    @property
    def microstepsPerRev(self):
        return self.fullStepsPerRev*self.microstepping

    @property
    def stepsPerMmX(self):
        return self.microstepsPerRev/self.mmPerRevX

    @property
    def stepsPerMmY(self):
        return self.microstepsPerRev/self.mmPerRevY


#===========================================================================
#Trajectory -> machine coordinates
#===========================================================================
#places the trajectory on the bed at (originX, originY) with y DOWN the
#page, and (optionally) scales it down if it would leave the work area.
#returns (strokesXY, scale)
def ToMachine(traj,cfg,fitToBounds=True,optimizeOrder=True):
    x0,y0,x1,y1 = traj.Bounds()
    w,h = max(1e-6,x1-x0),max(1e-6,y1-y0)
    scale = 1.0
    if fitToBounds:
        availW = cfg.boundsMaxXmm-cfg.originXmm
        availH = cfg.originYmm-cfg.boundsMinYmm
        scale = min(1.0,availW/w,availH/h)
    out = []
    for s in traj.strokes:
        #both frames have y UP (standard plotter convention): the text's
        #top row lands at originY and later rows descend from there
        q = [(cfg.originXmm+(x-x0)*scale,
              cfg.originYmm-(y1-y)*scale)
             for (x,y) in s]
        out.append(_Decimate(q,cfg.minSegmentMm))
    if optimizeOrder:
        out = OptimizeStrokeOrder(out)
    return out,scale


def _Decimate(poly,minSeg):
    if len(poly)<2:
        return poly
    out = [poly[0]]
    for p in poly[1:-1]:
        if math.dist(p,out[-1])>=minSeg:
            out.append(p)
    out.append(poly[-1])
    return out


#greedy nearest-neighbour ordering of the pen-down strokes, so the pen
#stops criss-crossing the page between them. Only the ORDER changes -- the
#same strokes are drawn, so the result is pixel-identical, but pen-up
#travel and job time drop substantially. Strokes are grouped into text
#rows first and each row ordered left-to-right-ish within itself, so the
#machine still writes the page in reading order.
def OptimizeStrokeOrder(strokes,rowTolMm=None):
    if len(strokes)<3:
        return strokes
    ys = [sum(p[1] for p in s)/len(s) for s in strokes]
    if rowTolMm is None:
        heights = [max(p[1] for p in s)-min(p[1] for p in s)
                   for s in strokes]
        rowTolMm = max(1.0,1.2*float(np.median(heights)))
    order = sorted(range(len(strokes)),key=lambda i: ys[i])
    rows,cur = [],[order[0]]
    for i in order[1:]:
        if abs(ys[i]-ys[cur[-1]])<=rowTolMm:
            cur.append(i)
        else:
            rows.append(cur)
            cur = [i]
    rows.append(cur)

    out = []
    pen = None
    for row in rows:
        remaining = list(row)
        while remaining:
            if pen is None:
                k = min(remaining,key=lambda i: strokes[i][0][0])
            else:
                k = min(remaining,
                        key=lambda i: min(math.dist(pen,strokes[i][0]),
                                          math.dist(pen,strokes[i][-1])))
            remaining.remove(k)
            s = strokes[k]
            #entering from whichever end is closer is free for a plotter
            if pen is not None and \
                    math.dist(pen,s[-1])<math.dist(pen,s[0]):
                s = s[::-1]
            out.append(s)
            pen = s[-1]
    return out


def _Clamp(strokes,cfg):
    lo = (cfg.boundsMinXmm,cfg.boundsMinYmm)
    hi = (cfg.boundsMaxXmm,cfg.boundsMaxYmm)
    clipped = 0
    out = []
    for s in strokes:
        q = []
        for (x,y) in s:
            cx = min(max(x,lo[0]),hi[0])
            cy = min(max(y,lo[1]),hi[1])
            if (cx,cy)!=(x,y):
                clipped+=1
            q.append((cx,cy))
        out.append(q)
    return out,clipped


#===========================================================================
#pen state machine: one-way motor, 90 deg per toggle
#===========================================================================
#the gear motor cannot reverse and has no position feedback, so pen state
#is tracked here: each state CHANGE emits exactly one 90-degree pulse.
#emitting a pulse when no change is needed would invert the pen for the
#rest of the job, so every transition goes through this class.
class PenController:

    def __init__(self,cfg):
        self.cfg = cfg
        self.isUp = cfg.penStartsUp
        self.pulses = 0

    def SetPen(self,wantUp):
        if wantUp==self.isUp:
            return []
        self.isUp = wantUp
        self.pulses+=1
        code = self.cfg.penUpCode if wantUp else self.cfg.penDownCode
        dwell = (self.cfg.penPulseMs+self.cfg.penSettleMs)/1000.0
        return [f"{code}   ; pen {'UP' if wantUp else 'DOWN'} "
                f"(one 90deg pulse of the one-way gear motor)",
                f"G4 P{dwell:.3f}   ; wait for the 90deg rotation to finish"]


#===========================================================================
#G-code
#===========================================================================
def WriteGcode(traj,cfg,path,title=''):
    strokes,scale = ToMachine(traj,cfg)
    strokes,clipped = _Clamp(strokes,cfg)
    pen = PenController(cfg)
    L = []
    L.append(f"; Handwriting-Robot G-code{(' -- '+title) if title else ''}")
    L.append(f"; author={traj.meta.get('author')} strokes={len(strokes)} "
             f"scale={scale:.3f}")
    L.append(f"; steps/mm  X={cfg.stepsPerMmX:.3f}  Y={cfg.stepsPerMmY:.3f} "
             f"({cfg.microstepsPerRev} microsteps/rev)")
    L.append(f"; pen: one-way gear motor, {cfg.penUpCode}=up "
             f"{cfg.penDownCode}=down, each = one 90deg pulse")
    if clipped:
        L.append(f"; WARNING: {clipped} point(s) clamped to work area")
    L.append('G21   ; millimetres')
    L.append('G90   ; absolute')
    L+=pen.SetPen(True)
    L.append(f"G0 F{cfg.travelFeedMmMin:.0f}")
    for s in strokes:
        if len(s)<2:
            continue
        L+=pen.SetPen(True)
        L.append(f"G0 X{s[0][0]:.3f} Y{s[0][1]:.3f}")
        L+=pen.SetPen(False)
        L.append(f"G1 F{cfg.drawFeedMmMin:.0f}")
        for (x,y) in s[1:]:
            L.append(f"G1 X{x:.3f} Y{y:.3f}")
    L+=pen.SetPen(True)
    L.append(f"G0 X{cfg.originXmm:.3f} Y{cfg.originYmm:.3f}   ; park")
    L.append('M2')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        f.write('\n'.join(L)+'\n')
    return dict(lines=len(L),strokes=len(strokes),penPulses=pen.pulses,
                scale=scale,clipped=clipped,path=str(path))


#===========================================================================
#preview: rasterize the EMITTED G-code (round-trip check)
#===========================================================================
#returns pen-down polylines in machine mm, by replaying the file
def ParseGcode(path):
    strokes = []
    cur = []
    penDown = False
    x = y = 0.0
    with open(path,encoding='utf-8') as f:
        for raw in f:
            line = raw.split(';')[0].strip()
            if not line:
                continue
            up = line.upper()
            code = up.split()[0]
            if code in ('M5',):
                if penDown and len(cur)>=2:
                    strokes.append(cur)
                cur,penDown = [], False
                continue
            if code in ('M3',):
                penDown = True
                cur = [(x,y)]
                continue
            if code in ('G0','G1'):
                nx = re.search(r'X(-?\d+\.?\d*)',up)
                ny = re.search(r'Y(-?\d+\.?\d*)',up)
                if nx:
                    x = float(nx.group(1))
                if ny:
                    y = float(ny.group(1))
                if nx or ny:
                    if penDown:
                        cur.append((x,y))
    if penDown and len(cur)>=2:
        strokes.append(cur)
    return strokes


def RenderGcodePreview(gcodePath,cfg,pxPerMm=6.0,strokePx=2,
                       showTravel=True,path=None):
    strokes = ParseGcode(gcodePath)
    W = int((cfg.boundsMaxXmm-cfg.boundsMinXmm)*pxPerMm)
    H = int((cfg.boundsMaxYmm-cfg.boundsMinYmm)*pxPerMm)
    img = Image.new('RGB',(max(8,W),max(8,H)),(255,255,255))
    d = ImageDraw.Draw(img)
    d.rectangle([0,0,W-1,H-1],outline=(210,210,210))

    def P(pt):
        #machine Y is up, image Y is down
        return ((pt[0]-cfg.boundsMinXmm)*pxPerMm,
                (cfg.boundsMaxYmm-pt[1])*pxPerMm)

    if showTravel:
        for a,b in zip(strokes,strokes[1:]):
            d.line([P(a[-1]),P(b[0])],fill=(235,170,170),width=1)
    for s in strokes:
        if len(s)>=2:
            d.line([P(p) for p in s],fill=(0,0,0),width=strokePx,
                   joint='curve')
    if path:
        img.save(path)
    return img,strokes

