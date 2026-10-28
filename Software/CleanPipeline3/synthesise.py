#u23539195 Dylan Braune.  writing new text in a writer's hand.  FU 6, R5
#
#a style profile and a sentence in, a pen path in millimetres out.  for every
#letter it picks one of that writer's stored shapes, or their clean fallback
#letterform when the stored ones are too rough to read, places it, joins it to
#the last one if that hand joins, and adds a little wobble so two goes are not
#identical
#
#the output is a Trajectory: lists of points the pen draws with the pen down.
#reproduce.py turns that into g-code, this file never touches the machine
import json
import math
import random

import numpy as np
from pathlib import Path

import paths
from recognise import (ResizeLineImageFixed,TensorFromResized,CtcForcedAlign,
                       CHAR_TO_IDX,PaperCRNNNumpy,AuthorClassifierCNNNumpy,
                       ReadText,CharAcc,ClassifyImage)
from segment import (BinarizeLine,Blur,CoreBand,DrawPolyline,EstimateSlantDeg,PolylineLength,Resize)


#======================================================================
#WHERE THINGS LIVE
#
#======================================================================
#---SCRIPT_DIR: this file's folder---
SCRIPT_DIR = Path(__file__).resolve().parent

#---CNN_DATA_DIR: the data folder beside the code---
CNN_DATA_DIR = Path(__file__).resolve().parent.parent/"CNN"



#======================================================================
#THE FALLBACK ALPHABETS
#letterforms to use when the writer's own stored shape is unusable.  these
#are shapes I drew by hand as point lists, not shapes from any writer
#======================================================================
#----------------------------------------------------------------------
#_FB: a plain printed alphabet, drawn as point lists
#x goes right, y goes up from the baseline, and 1.0 is one x-height, so
#the same shape works at any size.  used for a separated hand
#----------------------------------------------------------------------
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

#---_CX_IN and _CX_OUT: the height a cursive letter is entered and left at---
_CX_IN,_CX_OUT = 0.0,0.30

#----------------------------------------------------------------------
#_CURSIVE: a joined alphabet, one stroke each
#every letter starts and ends at the same connection height, so any two
#can be joined without a jump.  used for a joined hand
#----------------------------------------------------------------------
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

#---_FB_ADV: how far the pen moves on after each fallback letter---
_FB_ADV = 1.0

#---_SIMILAR: letters that can stand in for each other, like l for I---
_SIMILAR = {'I': 'l','O': '0','l': 'I','0': 'O','o': '0',';': ':',
            '"': "'",'!': 'l'}

#----------------------------------------------------------------------
#_BaselineGlyph: fetch a fallback letter, scaled to this writer
#takes the printed or joined shape and stretches it to the writer's own
#reach above and below the line, so a substituted letter is still the
#right size for the hand around it
#----------------------------------------------------------------------
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



#======================================================================
#HOW THE INK IS LAID DOWN
#the gantry has one pen at one width, so ink is uniform by design
#======================================================================
#---UNIFORM_INK: draw every writer at one pen width, which is what the machine does---
UNIFORM_INK = True

#---UNIFORM_PEN_WIDTH_XH: that width, in x-heights---
UNIFORM_PEN_WIDTH_XH = 0.085   #thinner than old 0.11 -- a heavy stroke fills small loops

#---UNIFORM_SOFT_FRAC: how soft the drawn edge is, as a fraction of pen width---
UNIFORM_SOFT_FRAC = 0.30   #blur radius as a fraction of strokePx



#======================================================================
#THE LIMITS LAYOUT WORKS TO
#every tunable number in one place, each with what it decides
#======================================================================
#---JOIN_PROB_CAP: never join more often than this, even for a fully joined hand---
JOIN_PROB_CAP = 1.0   #cap the per-pair cursive-join probability

#----------------------------------------------------------------------
#LETTER SPACING
#a glyph's stored `advance` is the width of the CELL it was cut from, not
#of its ink.  where the cut took in part of a neighbour the cell is far
#too wide, and the letter that follows is pushed away from it: measured
#over the five photographed writers the median trailing gap is 0.04 to
#0.13 x-heights, but the tail runs to 1.38 -- wider than a letter.  that
#is what draws a long flat connector across empty paper.
#
#the cure is to keep each writer's own spacing but refuse the tail of
#their own distribution, so a gap is clamped between their 25th and 90th
#percentile.  the lower end matters too: the old floor was ink+0.04,
#which is nothing, and letters ran into each other
#----------------------------------------------------------------------
#only the TOP of the range is taken from the writer.  raising the floor as
#well was tried and made things worse: it pushed apart letters that were
#sitting correctly, turning "number" into "num ber" and "the" into "t he",
#while fixing nothing the cap had not already fixed
GAP_HI_PCT = 90

#---MAX_LETTER_GAP_XH: and never more than this, whatever the writer---
#binds only on the writers with a wide tail -- thiya sits at 0.41 and
#dylan at 0.30, where owen and abhinav are already inside it
MAX_LETTER_GAP_XH = 0.30

#---MIN_LETTER_GAP_XH: the old floor, kept so letters cannot touch---
MIN_LETTER_GAP_XH = 0.03

#----------------------------------------------------------------------
#MAX_JOIN_GAP_XH: how far apart two letters may be and still be joined
#the join decision used to ask only what the letters were and how cursive
#the hand is, never how far apart they had ended up, so a connector was
#drawn across any distance at all.  a hand lifts the pen instead
#----------------------------------------------------------------------
MAX_JOIN_GAP_XH = 0.80

#----------------------------------------------------------------------
#WHERE A JOIN ATTACHES TO A LETTER
#the connector used to run between the two ENDS of whichever stroke was
#widest in x, in the order that stroke's points happened to be stored.
#neither half of that is anatomy:
#
#  * widest in x picks a t or f CROSSBAR over its own stem, because a bar
#    is wider than the stroke it crosses.  measured over all 25 profiles
#    the bar wins 269 times, so the pen leaves the letter from the tip of
#    the bar and the stem is flushed as a leftover.
#  * strokes are oriented left-to-right at cut time (style._TrimLigature-
#    Tails), which for a near-vertical letter is a coin flip on pixel
#    noise: 9 per cent of stored glyphs have a body x-span under a quarter
#    of their own height, and l leaves from its TOP in 23 of 215 cases.
#
#together those put a third of all exit points (33%) and getting on for
#half of all entry points (42%) above 0.7 x-heights, up in the letter
#rather than down on the baseline where a join belongs
#----------------------------------------------------------------------
#---JOIN_ANATOMY: choose the join point from the letter's shape---
JOIN_ANATOMY = True

#---JOIN_FLAT_MAX_ASPECT: below this height/width a stroke is a bar, not a body---
#a crossbar is by definition much wider than it is tall.  such a stroke
#may not carry the connection, whatever its x-span
JOIN_FLAT_MAX_ASPECT = 0.35

#---JOIN_VERT_MIN_XSPAN_XH: under this x-span, left-to-right ordering is noise---
#set at the measured degeneracy threshold, so the rule touches only the
#9 per cent of glyphs whose ordering was never meaningful.  for those the
#pen leaves at the LOW end, which is where the next letter is reached from
JOIN_VERT_MIN_XSPAN_XH = 0.25

#---JOIN_NO_ENTRY_ABOVE_XH: a letter walked upright may not be joined INTO up here---
#a single-stroke upright letter has one low end and one high one, so once
#the pen is made to LEAVE at the bottom it necessarily ARRIVES at the top.
#there is no honest join into that end -- a real hand lifts and starts the
#downstroke -- so the connector is refused rather than drawn as a climb
JOIN_NO_ENTRY_ABOVE_XH = 0.70

#----------------------------------------------------------------------
#JOIN_MAX_ATTACH_XH: no connector attaches above the x-height band
#a cursive connector runs along the baseline and arrives from below, so it
#cannot honestly meet a letter above x-height.  the stored glyphs separate
#on exactly that line, because a skeleton trace of an ascender starts at
#the top of the ascender while a trace of a short letter cannot:
#
#  entry height, median over all 25 profiles
#    b 1.42   l 1.47   t 1.30   k 0.90   i 0.88      <- ascender tops
#    v 0.84   w 0.75   o 0.70   u 0.81   e 0.64      <- never above 1.08 at p90
#
#so 1.15 sits in the gap.  v, w, o, u and c keep every join they had --
#they genuinely finish and start at x-height -- while the climb up to an
#ascender tip is refused.  it costs 16.8 per cent of entry attachments and
#7.7 per cent of exit attachments, and those are the implausible ones
#----------------------------------------------------------------------
JOIN_MAX_ATTACH_XH = 1.15

#---CORE_SLANT_CAP_DEG: how far a fallback letter is allowed to lean---
CORE_SLANT_CAP_DEG = 22.0   #print-anchor glyphs never shear past this

#---CORE_JITTER_FRAC: fallback letters get less wobble, they are meant to be readable---
CORE_JITTER_FRAC = 0.15   #print-anchor glyphs barely wobble

#---COLLISION_MIN_GAP_XH: the clearance kept between one letter and the next---
COLLISION_MIN_GAP_XH = 0.05   #forced clear space (x-h) between a glyph and the previous one

#----------------------------------------------------------------------
#_ELIGIBLE_VARIANTS: how many stored shapes of a letter may be picked
#a writer forms a letter fairly consistently, so picking evenly from a
#dozen variants averages their handwriting away.  only the most typical
#few are eligible
#----------------------------------------------------------------------
_ELIGIBLE_VARIANTS = 5

#---CORE_CURSIVE_CONN: how joined a hand must be before its fallback is the joined alphabet---
CORE_CURSIVE_CONN = 0.5

#---JOIN_LEGIBILITY_COUPLING: how much legibility thins out joins, zero on purpose---
JOIN_LEGIBILITY_COUPLING = 0.0

#---IDEAL_ANCHOR_MAX_PRIORD: how rough the writer's own best letterform may be and still be used---
IDEAL_ANCHOR_MAX_PRIORD = 0.30

#---IDEAL_SOURCE: use the most typical REAL variant, not an average of several---
IDEAL_SOURCE = 'medoid'

#---_PRIOR_REJECT: beyond this distance a stored shape is not treated as a letter---
_PRIOR_REJECT = 0.25   #prior distance above which a stored variant is not offered



#======================================================================
#THE PEN PATH
#
#======================================================================
#----------------------------------------------------------------------
#Trajectory: the pen path, in millimetres
#a list of strokes, each a list of points drawn with the pen down.  y is
#up here; whoever writes the file flips it if their machine needs it.
#also reports its own size and how far the pen travels up and down
#----------------------------------------------------------------------
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
        down = sum(PolylineLength(s)
                   for s in self.strokes if len(s)>1)
        up = 0.0
        for a,b in zip(self.strokes,self.strokes[1:]):
            up+=float(math.dist(a[-1],b[0]))
        return down,up



#======================================================================
#CHOOSING WHICH SHAPE TO DRAW FOR A LETTER
#the writer's own shape when it is good enough, the fallback when it is not
#======================================================================
#---_GLYPH_CLEAN: at or below this distance the shape is clean, keep it---
_GLYPH_CLEAN = 0.12   #prior distance at or below which the shape is clean, keep it

#---_GLYPH_GARBAGE: at or above this it is replaced whatever else is set---
_GLYPH_GARBAGE = 0.55   #at or above this it is replaced whatever lam says

#---_COMMIT_LAM: above this legibility setting the library is abandoned entirely---
_COMMIT_LAM = 0.85   #at or above this lam the library goes entirely to the anchor

#----------------------------------------------------------------------
#_SwapProb: the chance this shape gets swapped for the fallback
#clean shapes are kept, hopeless ones always go, and in between the
#chance rises with both the legibility setting and how rough the shape is
#----------------------------------------------------------------------
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

#---_AUTO_LAM_GOOD: a library this typical is trusted completely---
_AUTO_LAM_GOOD = 0.24   #median prior distance at or below which the library is trusted

#---_AUTO_LAM_BAD: this far off and the variants are fragments, not letters---
_AUTO_LAM_BAD = 0.32   #at or above this the variants are fragments, not letters

#----------------------------------------------------------------------
#AutoLegibility: work out how far to trust this writer's library
#uses the middle distance-from-typical of all their stored shapes.  a
#very joined hand has no honest place to cut between letters, so its
#library scores badly and the fallback letterforms get used instead
#----------------------------------------------------------------------
def AutoLegibility(profile):
    ds = [float(g.get('priorD',0.5))
          for v in profile.get('glyphs',{}).values() for g in v]
    if not ds:
        return 1.0
    med = float(np.median(ds))
    return float(np.clip((med-_AUTO_LAM_GOOD)/(_AUTO_LAM_BAD-_AUTO_LAM_GOOD),
                         0.0,1.0))

#----------------------------------------------------------------------
#_CoreGlyph: the fallback letterform for one character
#prefers the writer's own most typical version of that letter, and only
#drops to the generic alphabet if even that is too rough.  a joined hand
#gets the joined version so a substitute can still connect
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#_GapRange: the spacing this writer leaves after a letter, as a range
#taken from their own stored glyphs rather than from a fixed number, so a
#wide hand stays wide and a tight one stays tight.  cached on the profile
#because it is asked for once per letter drawn
#----------------------------------------------------------------------
def _GapRange(profile):
    r = profile.get('_gapRange')
    if r is None:
        gaps = []
        for ch,vs in (profile.get('glyphs') or {}).items():
            if len(ch)!=1 or not ch.isalpha():
                continue
            for g in vs:
                gaps.append(float(g.get('advance',0.9))
                            -float(g.get('lead',0.0))
                            -float(g.get('width',0.9)))
        hi = float(np.percentile(gaps,GAP_HI_PCT)) if gaps else 0.25
        lo = MIN_LETTER_GAP_XH
        hi = min(MAX_LETTER_GAP_XH,max(lo+0.02,hi))
        r = (lo,hi)
        profile['_gapRange'] = r
    return r


#---_Advance: how far to move on after a glyph, with the gap reined in---
def _Advance(g,profile):
    lead = float(g.get('lead',0.0))
    wid = float(g.get('width',0.9))
    lo,hi = _GapRange(profile)
    gap = float(g.get('advance',0.9))-lead-wid
    return lead+wid+min(max(gap,lo),hi)


def _CoreGlyph(ch,profile):
    cursive = float(profile.get('connectedness',0.0))>=CORE_CURSIVE_CONN
    src = (profile.get('idealMedoid') if IDEAL_SOURCE=='medoid'
           else profile.get('idealGlyphs')) or {}
    own = src.get(ch)
    if own and float(own.get('priorD',1.0))<=IDEAL_ANCHOR_MAX_PRIORD:
        strokes = [[tuple(p) for p in s] for s in own['strokes']]
        adv = _Advance(own,profile)
        return strokes,adv,float(own.get('lead',0.0)),cursive
    b = _BaselineGlyph(ch,profile,printOnly=not cursive)
    return None if b is None else (b[0],b[1],0.0,cursive)

#----------------------------------------------------------------------
#_GlyphSource: pick the shape to draw, and say where it came from
#ranks the writer's variants by how typical they are, picks one favouring
#the typical (and, when joining, one whose entry height matches where the
#pen already is), then maybe swaps it for the fallback.  returns the
#strokes, how far to advance, and which source was used
#----------------------------------------------------------------------
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
        #the writer's own advance, with the gap after the ink held inside
        #their own usual range: the stored figure is the width of the cut
        #cell, and a cell that swallowed part of a neighbour is far too wide
        a = _Advance(g,profile)
        return strokes,a*k,float(g.get('lead',0.0))*k

        #---Pick: favour typical variants but keep varying, since a uniform choice
    #---over the noisy tail looks scrambled.  when joining, also favour the
    #---variant whose entry height matches where the pen already is
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



#======================================================================
#MAKING IT LOOK HANDWRITTEN, NOT PRINTED
#nobody writes the same letter twice identically, so a little variation
#======================================================================
#----------------------------------------------------------------------
#_Jitter: nudge, turn and scale a letter very slightly
#a small random shift, rotation and size change per letter.  without it
#every 'e' on the page is pixel identical and it reads as printed
#----------------------------------------------------------------------
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

#---_Resample: respace a path so the points are evenly spread---
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

#---USE_STYLE_CALIBRATION: whether to correct the measured style back towards the writer's real pages---
USE_STYLE_CALIBRATION = True

#---_STYLE_CAL: the cached correction per writer---
_STYLE_CAL = {}

#----------------------------------------------------------------------
#StyleCalibration: how far the output drifts from the real pages
#compares what gets measured off synthetic output against what was
#measured off the writer's real lines, giving a correction
#----------------------------------------------------------------------
def StyleCalibration(profile,mmPerXh=4.0,pxPerMm=18.0):
    key = (profile.get('authorId'),round(mmPerXh,3),round(pxPerMm,3))
    if key in _STYLE_CAL:
        return _STYLE_CAL[key]
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
        ink = BinarizeLine(np.array(img))
        band = CoreBand(ink)
        if band is None or not ink.any():
            break
        top,base = band
        xh = float(base-top)
        if xh<4:
            break
        ys = np.nonzero(ink.any(axis=1))[0]
        gotSlant = EstimateSlantDeg(ink)
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

#---_ApplyCal: apply that correction to a letter's strokes---
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



#======================================================================
#LAYING THE LETTERS OUT INTO A LINE
#advance, join, wobble the baseline, start a new line at the margin
#======================================================================
#----------------------------------------------------------------------
#_Ligature: the joining stroke between two letters
#a curve from where the pen left one letter to where it enters the next.
#it only dips in proportion to how far it has to travel, because a fixed
#dip turns a short join into a loop, which is what makes synthetic
#joined writing look tangled
#----------------------------------------------------------------------
#---USE_TANGENT_LIGATURE: shape the join from the pen's heading---
#measured against the midpoint-sag join on five lightly joined writers over
#three sentences: character accuracy 83.36 -> 84.24 per cent, writer
#confidence 78.62 -> 74.29.  the whole of that writer-ID loss is writer 588
#alone, which fell 83.0 -> 61.4 while 152 gained on both counts, so it is off
#by default until a full sentence set says whether 588's drop is real
USE_TANGENT_LIGATURE = False

#---TANGENT_PULL: how far along each tangent the control point sits---
#as a fraction of the gap being bridged.  larger sweeps more widely out of
#the previous letter before turning towards the next
TANGENT_PULL = 0.42


#---_Tangent: the direction a path is travelling at one of its ends---
def _Tangent(pts,atStart,n=4):
    p = np.asarray(pts,dtype=np.float64)
    if len(p)<2:
        return None
    seg = p[:min(n,len(p))] if atStart else p[-min(n,len(p)):]
    v = seg[-1]-seg[0]
    nrm = float(np.linalg.norm(v))
    return (v/nrm) if nrm>1e-9 else None


#----------------------------------------------------------------------
#_LigatureTangent: a join that leaves and arrives along the pen's heading
#the earlier join knew only two points, so it left the previous letter at
#whatever angle its sag produced rather than along the stroke it was
#continuing from, which is what made a join read as grafted on.  this one
#places the two control points of a cubic along the outgoing and incoming
#directions, so the curve is tangent to both letters: it leaves exactly
#the way the pen was already going and arrives exactly the way the next
#letter starts, the way a french curve is laid against two fixed slopes
#----------------------------------------------------------------------
def _LigatureTangent(p0,p1,dOut,dIn,xh,rng):
    gap = math.hypot(p1[0]-p0[0],p1[1]-p0[1])
    if gap<0.06*xh:
        return [p0,p1]
    k = TANGENT_PULL*gap
    jitter = rng.gauss(0.0,0.02*xh)
    c1 = (p0[0]+dOut[0]*k, p0[1]+dOut[1]*k+jitter)
    c2 = (p1[0]-dIn[0]*k,  p1[1]-dIn[1]*k)
    out = []
    for i in range(11):
        t = i/10.0
        u = 1.0-t
        out.append((u*u*u*p0[0]+3*u*u*t*c1[0]+3*u*t*t*c2[0]+t*t*t*p1[0],
                    u*u*u*p0[1]+3*u*u*t*c1[1]+3*u*t*t*c2[1]+t*t*t*p1[1]))
    return _Resample(out,step=0.35)


def _Ligature(p0,p1,xh,rng):
    dx = p1[0]-p0[0]
    if abs(dx)<0.06*xh:
        return [p0,p1]
    sag = min(0.11*xh,0.16*abs(dx))+rng.gauss(0.0,0.015*xh)
    #control height follows BOTH ends, so a join that has to climb does not
    #first dive under the lower letter and hook back up
    ctrlY = min(p0[1],p1[1])+0.35*abs(p1[1]-p0[1])-sag
    ctrl = ((p0[0]+p1[0])*0.5,ctrlY)
    #quadratic through that control point, so the curve leaves and arrives
    #along the pen direction instead of turning a corner at the midpoint
    out = []
    for i in range(9):
        t = i/8.0
        u = 1.0-t
        out.append((u*u*p0[0]+2*u*t*ctrl[0]+t*t*p1[0],
                    u*u*p0[1]+2*u*t*ctrl[1]+t*t*p1[1]))
    return _Resample(out,step=0.35)

#----------------------------------------------------------------------
#_BodyStroke: which stroke carries the join, and which way it is walked
#two questions, both answered from the shape in front of it rather than
#from the order the skeleton tracer happened to emit:
#  which stroke -- the widest one that is not a bar.  a dot never wins on
#    width, so dots were already safe; a crossbar did win, and is now ruled
#    out by its own flatness
#  which way round -- left-to-right while that means something, and the low
#    end last once the letter is too upright for x to decide
#the pen leaves at the end of the returned stroke, so reversing it is what
#moves a join off the top of an l and back down to the baseline
#----------------------------------------------------------------------
def _BodyStroke(placed,xh,baseY=0.0):
    spans = [max(p[0] for p in s)-min(p[0] for p in s) for s in placed]
    highs = [max(p[1] for p in s)-min(p[1] for p in s) for s in placed]
    if not JOIN_ANATOMY:
        bi = int(np.argmax(spans))
        return bi,list(placed[bi]),True,True
    #a bar is far wider than it is tall: it crosses a letter, it is not one
    cand = [i for i in range(len(placed))
            if highs[i]>=JOIN_FLAT_MAX_ASPECT*max(spans[i],1e-9)]
    if not cand:                       #every stroke is flat: nothing better to pick
        cand = list(range(len(placed)))
    bi = max(cand,key=lambda i: spans[i])
    body = list(placed[bi])
    entryOk = True
    #too upright for x to have ordered it honestly, so order it by height
    if spans[bi]<JOIN_VERT_MIN_XSPAN_XH*xh:
        if body[0][1]<body[-1][1]:
            body = body[::-1]          #y is up, so the lower end goes last
        #and the end the pen now arrives at may be too high to join into
        if (body[0][1]-baseY)>JOIN_NO_ENTRY_ABOVE_XH*xh:
            entryOk = False
    #neither end may be joined above the x-height band, whatever the letter
    if (body[0][1]-baseY)>JOIN_MAX_ATTACH_XH*xh:
        entryOk = False
    exitOk = (body[-1][1]-baseY)<=JOIN_MAX_ATTACH_XH*xh
    return bi,body,entryOk,exitOk

#---_SplitWords: text into words and line breaks---
def _SplitWords(text):
    out = []
    for chunk in text.split('\n'):
        out+=[w for w in chunk.split(' ') if w]
        out.append('\n')
    return out[:-1] if out and out[-1]=='\n' else out

#---_WordWidth: roughly how wide a word will be, for wrapping---
def _WordWidth(word,profile):
    adv = profile.get('letterAdvance',{})
    return sum(adv.get(c,0.9) for c in word)

#----------------------------------------------------------------------
#SynthesizeText: the main job, text in and a pen path out
#walks the text letter by letter: pick a shape, calibrate it, wobble it,
#place it, push it right if it would collide, join it to the previous
#letter if this hand joins, and let the baseline wander slightly as a
#real hand does
#----------------------------------------------------------------------
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
        prevExitOk = True
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

            #the BODY stroke carries the cursive connection; dots, crossbars
            #and accents are separate pen-downs and must not become the exit
            #point for the next letter
            bodyI,body,entryOk,exitOk = _BodyStroke(placed,xh,penY+glyphDrift*xh)
            bothOwn = not isCore and prevSrc!='core'   #both are the writer's own shapes
            bothCursiveCore = coreCursive and prevCoreCursive   #or both the cursive anchor
            #how far the pen would have to reach to make this join.  a hand
            #that far apart lifts instead, and drawing it leaves a long flat
            #dash across the paper
            joinGap = ((body[0][0]-prevExit[0])/xh) if prevExit is not None else 9e9
            joinable = (ci>0 and prevExit is not None
                        and ch.isalpha() and word[ci-1].isalpha()
                        and (bothOwn or bothCursiveCore)   #a print anchor never joins
                        and joinGap<=MAX_JOIN_GAP_XH
                        and entryOk and prevExitOk   #neither end up in the ascenders
                        and rng.random()<min(connEff,JOIN_PROB_CAP))   #draw per pair, not a threshold
            if joinable:
                #the tangents come from the real geometry either side: the tail
                #of what has been drawn, and the head of the letter being joined
                dOut = _Tangent(chain,False) if USE_TANGENT_LIGATURE else None
                dIn = _Tangent(body,True) if USE_TANGENT_LIGATURE else None
                if dOut is not None and dIn is not None:
                    chain+=_LigatureTangent(prevExit,body[0],dOut,dIn,xh,rng)
                else:
                    chain+=_Ligature(prevExit,body[0],xh,rng)
                chain+=body
            else:
                flush(chain)
                chain = list(body)
            prevExit = chain[-1]
            prevExitOk = exitOk
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



#======================================================================
#TURNING THE PEN PATH INTO AN IMAGE
#needed because the two judging networks score pictures, not paths
#======================================================================
#----------------------------------------------------------------------
#_Draw: rasterise the pen path, black ink on white paper
#draws each stroke with the round pen from segment.py, then softens the
#edge slightly, because a real pen edge is not a hard step
#----------------------------------------------------------------------
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
        DrawPolyline(arr,pts,fill,strokePx)
    if softPx>0.05:
        #a real pen edge is not a hard step, the blur softens it.  sigma is
        #a fraction of the pen width so a thick pen gets a wider soft edge
        arr = Blur(arr,softPx)
    return arr

#----------------------------------------------------------------------
#RenderTrajectory: pen path to a crop like the training data
#same size and look as the crops the networks were trained on, so their
#scores mean something.  one pen width for everyone by default, which
#is what the machine actually draws
#----------------------------------------------------------------------
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
            #my own resampler, see Resize
            img = Resize(np.asarray(img,dtype=np.uint8),size[0],size[1],'linear')
        return img

    if strokePx is None:
        sw = prof.get('strokeWidthXh',0.12)
        strokePx = max(1,int(round(sw*mm*pxPerMm)))

    paper = prof.get('paperLevel',255.0)
    #inkLevel in the profile is the median of the already dark pixels, ie a whole
    #stroke average including the soft edges, NOT the colour of the stroke core.
    #so the core is painted dark and the soft edge comes from the blur instead
    inkLv = prof.get('inkCoreLevel',0.0)
    soft = prof.get('softScale',0.30)*max(1.0,strokePx)
    target = prof.get('inkFracRef') if matchInk else None
    if target:
        img,strokePx = _MatchInkDensity(traj,pxPerMm,padMm,strokePx,
                                         target,mm,paper,inkLv,soft)
    else:
        img = _Draw(traj,pxPerMm,padMm,strokePx,paper,inkLv,soft)
    if size is not None:
        #my own resampler, see Resize
        img = Resize(np.asarray(img,dtype=np.uint8),size[0],size[1],'linear')
    return img

#---_MatchInkDensity: search for the pen width that matches the writer's real ink---
def _MatchInkDensity(traj,pxPerMm,padMm,startPx,target,mm,
                     paper=255,ink=0,soft=0.0):

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



#======================================================================
#CHECKING THE RESULT, AND TRYING AGAIN
#the reader and the namer score a draw, and the worst letters get redrawn
#======================================================================
#---_LineLogProbs: the reader's column scores for a rendered line---
def _LineLogProbs(reader,img):
    t = TensorFromResized(ResizeLineImageFixed(img))
    return reader.forward(t)[:,0,:]

#----------------------------------------------------------------------
#_WeakChars: which letters the reader is least sure about
#aligns the known text against the scores and reports the characters
#with the worst scores, so only those get replaced on the next try
#----------------------------------------------------------------------
def _WeakChars(reader,img,visible):
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

#---REPAIR_ROUNDS: how many times to redraw the weak letters---
REPAIR_ROUNDS = 3   #targeted redraw passes after the best of N search

#---REPAIR_WORST_K: how many letters to fix per round---
REPAIR_WORST_K = 3   #characters pinned to the anchor per repair round

#---DEFAULT_LEGIBILITY: starting legibility when the caller does not say---
DEFAULT_LEGIBILITY = 0.65   #lam used only by the older legibility-only search

#----------------------------------------------------------------------
#SynthesizeLegible: draw, check, fix the worst letters, repeat
#renders, asks the reader which characters are weak, and redraws just
#those with their fallback letterform, a few rounds
#----------------------------------------------------------------------
def SynthesizeLegible(text,profile,nTries=6,mmPerXh=4.0,lineWidthMm=180.0,
                      jitter=0.5,seed=0,reader=None,
                      pxPerMm=18.0,legibility=None,repair=True):
    if legibility is None:
        legibility = float(profile.get('legibilityLambda',DEFAULT_LEGIBILITY))
    if reader is None:
        try:
            reader = PaperCRNNNumpy()
        except Exception:
            return SynthesizeText(text,profile,mmPerXh=mmPerXh,seed=seed,
                                  lineWidthMm=lineWidthMm,jitter=jitter,
                                  legibility=legibility)
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

#----------------------------------------------------------------------
#SynthesizeJointBestOf: draw it several times and keep the best
#scores every attempt with BOTH judges at once, the reader for whether
#it says the right words and the namer for whether it still looks like
#this writer, combined so one cannot be traded away for the other.
#this is what the results in the report were produced with
#----------------------------------------------------------------------
def SynthesizeJointBestOf(author,text,profile,nTries=6,mmPerXh=4.0,
                          lineWidthMm=180.0,jitter=0.5,seed=0,
                          reader=None,authorModel=None,authorMapping=None,
                          pxPerMm=18.0,legibility=None,
                          repair=True,repairRounds=3,repairWorstK=2):
    if legibility is None:
        legibility = float(profile.get('legibilityLambda',
                                       AutoLegibility(profile)))
    if reader is None:
        reader = PaperCRNNNumpy()
    if authorModel is None:
        authorModel = AuthorClassifierCNNNumpy()
    if authorMapping is None:
        authorMapping = authorModel.authorMapping
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
