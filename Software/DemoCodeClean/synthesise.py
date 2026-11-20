#u23539195 D.R. Braune.  writing new text in a writer's hand
#a style profile and a sentence in, a pen path in millimetres out
#for each letter one of the writer's stored shapes is placed, or their clean
#fallback letterform when the stored ones are too rough, then joined to the
#letter before it if that hand joins
import math
import random

import numpy as np

from recognise import (ResizeLineImageFixed,TensorFromResized,CtcForcedAlign,
                       CHAR_TO_IDX,PaperCRNNNumpy,AuthorClassifierCNNNumpy,
                       ReadText,CharAcc,ClassifyImage)
from segment import (BinarizeLine,Blur,CoreBand,DrawPolyline,EstimateSlantDeg,PolylineLength,Resize)

#a printed alphabet drawn as point lists, x runs right, y runs up from the
#baseline, and 1.0 is one x-height, so a shape works at any size
_FB = {
    'a': [[(0.75,0.75),(0.45,0.95),(0.15,0.75),(0.1,0.4),(0.35,0.05),   #y runs up from the baseline and 1.0 is one x-height
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

#a joined alphabet, one stroke each, every letter starting and ending at
#the same connection height so any two can be joined without a jump
_CURSIVE = {
    'a': [[(0.0,0.30),(0.20,0.62),(0.46,0.72),(0.62,0.52),(0.60,0.18),   #every cursive letter starts and ends at the same height of 0.30
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

#how far the pen moves on after a fallback letter
_FB_ADV = 1.0   # one x-height of advance after a fallback letter

#letters that can stand in for each other
_SIMILAR = {'I': 'l','O': '0','l': 'I','0': 'O','o': '0',';': ':',   #shapes close enough to stand in for one another
            '"': "'",'!': 'l'}

#----------------------------------------------------------------------
#_BaselineGlyph: fetches a fallback letter, stretched to this writer's own reach above and
#below the line
#----------------------------------------------------------------------
def _BaselineGlyph(ch,profile,printOnly=False):
    conn = float(profile.get('connectedness',0.0))
    src = _CURSIVE if (conn>=0.5 and not printOnly) else _FB  # a joined hand draws from the cursive alphabet
    key = ch if ch in src else ch.lower()
    if key not in src:
        return None
    asc = float(profile.get('ascender',1.7))
    desc = float(profile.get('descender',-0.6))
    strokes = []
    for st in src[key]:
        q = []
        for (x,y) in st:
            if y>1.0:   #the part above the band stretches to this writer ascender
                y = 1.0+(y-1.0)*max(0.35,(asc-1.0)/0.75)   #scaled by their reach over the 0.75 the alphabet was drawn at
            elif y<0.0:   # and the part below stretches to their descender
                y = y*max(0.35,abs(desc)/0.55)
            q.append((x,y))
        strokes.append(q)
    xs = [p[0] for st in strokes for p in st]
    adv = (max(xs)-min(xs))+0.12   #the ink width plus a small gap is the advance
    return strokes,adv

#the pen width every writer is drawn at, in x-heights, which is what the
#machine's single pen does
UNIFORM_PEN_WIDTH_XH = 0.085   #every writer is drawn at the pen width the machine has

#how soft the drawn edge is, as a fraction of the pen width
UNIFORM_SOFT_FRAC = 0.30

#the percentile of a writer's own letter gaps taken as their upper limit
GAP_HI_PCT = 90   # the 90th percentile of this writer own letter gaps

#and the absolute limits on a letter gap, in x-heights
MAX_LETTER_GAP_XH = 0.30
MIN_LETTER_GAP_XH = 0.03

#how far apart two letters may be and still be joined, in x-heights
MAX_JOIN_GAP_XH = 0.80   #a gap over eight tenths of an x-height is left unjoined

#below this height over width a stroke is a crossbar, which may not carry
#the join
JOIN_FLAT_MAX_ASPECT = 0.35   #height under 35 percent of width makes a stroke a crossbar

#under this x-span, left-to-right stroke ordering is noise, so the join
#point is chosen by height instead
JOIN_VERT_MIN_XSPAN_XH = 0.25

#how far a fallback letter may lean, in degrees
CORE_SLANT_CAP_DEG = 22.0   #the fallback alphabet leans no further than 22 degrees

#how much of the usual wobble a fallback letter gets
CORE_JITTER_FRAC = 0.15   #and takes 15 percent of the usual wobble

#the clearance kept between one letter's ink and the next, in x-heights
COLLISION_MIN_GAP_XH = 0.05    # ink is kept a twentieth of an x-height clear of the letter before

#how many of a letter's stored shapes may be picked from
_ELIGIBLE_VARIANTS = 5   #the draw runs over the five best ranked shapes

#how joined a hand must be before its fallback is the joined alphabet
CORE_CURSIVE_CONN = 0.5   #half the letter pairs joined makes the hand a cursive one

#below this joinedness the hand is treated as a print hand and
#draws no joins at all
JOIN_PRINT_CONN_FLOOR = 0.45   #under 0.45 joined the hand draws no joins at all

#how rough the writer's own best letterform may be and still be used
IDEAL_ANCHOR_MAX_PRIORD = 0.30

#----------------------------------------------------------------------
#Trajectory: the pen path in millimetres, a list of strokes, each a list of points
#drawn with the pen down, with y up
#----------------------------------------------------------------------
class Trajectory:

    def __init__(self,strokes,meta):
        self.strokes = strokes
        self.meta = meta

    def Bounds(self):
        pts = [p for s in self.strokes for p in s]   #every point of every stroke
        if not pts:
            return (0.0,0.0,0.0,0.0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs),min(ys),max(xs),max(ys))

    def PenTravelMm(self):
        down = sum(PolylineLength(s)   #distance walked with the pen on the paper
                   for s in self.strokes if len(s)>1)
        up = 0.0
        for a,b in zip(self.strokes,self.strokes[1:]):
            up+=float(math.dist(a[-1],b[0]))   #and the hops from one stroke end to the next start
        return down,up

#the prior distance at or below which a stored shape is kept
_GLYPH_CLEAN = 0.12   #a prior distance this low means a clean letterform

#at or above which it is always replaced
_GLYPH_GARBAGE = 0.55   #and this high means the cut is unusable

#the legibility setting above which the glyph storage is abandoned entirely
_COMMIT_LAM = 0.85

#----------------------------------------------------------------------
#_SwapProb: the chance a stored shape is swapped for the fallback, rising with both
#the legibility setting and how rough the shape is
#----------------------------------------------------------------------
def _SwapProb(lam,priorD):
    d = float(priorD)
    if d>=_GLYPH_GARBAGE:   # a shape past the garbage mark is always swapped
        return 1.0
    if lam>=_COMMIT_LAM:
        return 1.0
    if lam<=1e-3:
        return 0.0
    if d<=_GLYPH_CLEAN:
        return 0.06*lam   #a clean shape is swapped only rarely
    bad = float(np.clip((d-_GLYPH_CLEAN)/(_GLYPH_GARBAGE-_GLYPH_CLEAN),   # where the shape sits between clean and garbage
                        0.0,1.0))
    return float(np.clip(lam*(0.55+0.9*bad),0.0,1.0))   #the swap chance rises with both that and the setting

#the median prior distance at or below which the glyph storage is trusted
_AUTO_LAM_GOOD = 0.24  #a median distance here keeps the stored variants

#and at or above which the variants are fragments rather than letters
_AUTO_LAM_BAD = 0.32   # and here leans fully on the fallback letterforms

#----------------------------------------------------------------------
#AutoLegibility: how far to trust this writer's glyph storage, from the median distance from
#typical of all their stored shapes
#----------------------------------------------------------------------
def AutoLegibility(profile):
    ds = [float(g.get('priorD',0.5))
          for v in profile.get('glyphs',{}).values() for g in v]
    if not ds:
        return 1.0
    med = float(np.median(ds))   #the median distance across every stored shape
    return float(np.clip((med-_AUTO_LAM_GOOD)/(_AUTO_LAM_BAD-_AUTO_LAM_GOOD),   # mapped onto 0 to 1 between those two marks
                         0.0,1.0))

#----------------------------------------------------------------------
#_GapRange: the letter gap range this writer leaves, taken from their own glyphs and
#held inside the absolute limits, cached on the profile
#----------------------------------------------------------------------
def _GapRange(profile):
    r = profile.get('_gapRange')
    if r is None:
        gaps = []
        for ch,vs in (profile.get('glyphs') or {}).items():
            if len(ch)!=1 or not ch.isalpha():
                continue
            for g in vs:
                gaps.append(float(g.get('advance',0.9))   #advance less lead less width is the gap after the letter
                            -float(g.get('lead',0.0))
                            -float(g.get('width',0.9)))
        hi = float(np.percentile(gaps,GAP_HI_PCT)) if gaps else 0.25   #the 90th percentile of those gaps
        lo = MIN_LETTER_GAP_XH
        hi = min(MAX_LETTER_GAP_XH,max(lo+0.02,hi))   # held inside the absolute limits
        r = (lo,hi)
        profile['_gapRange'] = r
    return r

#---_Advance: how far to move on after a glyph, with the gap held in that range---
def _Advance(g,profile):
    lead = float(g.get('lead',0.0))
    wid = float(g.get('width',0.9))
    lo,hi = _GapRange(profile)
    gap = float(g.get('advance',0.9))-lead-wid
    return lead+wid+min(max(gap,lo),hi)   #lead plus ink plus the gap clamped to that range

#----------------------------------------------------------------------
#_CoreGlyph: the fallback letterform, the writer's own most typical version of that
#letter, or the generic alphabet if even that is too rough
#----------------------------------------------------------------------
def _CoreGlyph(ch,profile):
    cursive = float(profile.get('connectedness',0.0))>=CORE_CURSIVE_CONN   #how joined the hand is picks which alphabet backs it
    src = profile.get('idealMedoid') or {}
    own = src.get(ch)
    if own and float(own.get('priorD',1.0))<=IDEAL_ANCHOR_MAX_PRIORD:  #the writer own medoid is used while it scores under 0.30
        strokes = [[tuple(p) for p in s] for s in own['strokes']]
        adv = _Advance(own,profile)
        return strokes,adv,float(own.get('lead',0.0)),cursive
    b = _BaselineGlyph(ch,profile,printOnly=not cursive)   #otherwise the generic alphabet stands in
    return None if b is None else (b[0],b[1],0.0,cursive)

#----------------------------------------------------------------------
#_GlyphSource: picks the shape to draw and says where it came from, ranks the writer's
#variants by how typical they are, picks one favouring the typical and,
#when joining, one whose entry height matches where the pen already is,
#then maybe swaps it for the fallback
#----------------------------------------------------------------------
def _GlyphSource(profile,ch,rng,prevExitY=None,lam=0.0,forceCore=False,
                 forcePrint=False,bestPrio=False):
    store = profile['glyphs']
    adv = profile.get('letterAdvance',{})
    if forcePrint:
        b = _BaselineGlyph(ch,profile,printOnly=True)
        if b is not None:
            return b[0],b[1],0.0,'core'

    def Pack(g,k=1.0):
        strokes = [[(x*k,y*k) for (x,y) in s] for s in g['strokes']]   # k scales the whole shape when a case is borrowed
        #the variant's own advance, never the character median
        a = _Advance(g,profile)   #the advance of this very variant
        return strokes,a*k,float(g.get('lead',0.0))*k

    def Pick(cands):
        #best-prior: no draw at all, take the best-formed shape this writer
        #produced.  the ranking is by priorD, so cands[0] is that shape; the
        #entry-height match below only breaks ties between eligible variants
        #and cannot outrank a cleaner letter when variation is switched off
        if bestPrio:
            return cands[0]
        cands = cands[:_ELIGIBLE_VARIANTS]   # only the five best ranked are eligible
        w = []
        for i,g in enumerate(cands):
            wi = 1.0/(1.0+1.4*i)    #weight falls off down the ranking so typical shapes come up more
            if prevExitY is not None:
                dy = abs(float(g.get('entryY',0.0))-prevExitY)    #how far this entry height sits from where the pen already is
                wi*=math.exp(-(dy/0.45)**2)   #a gaussian over that difference favours a matching entry
            w.append(wi)
        tot = sum(w)
        if tot<=1e-9:
            return cands[0]
        r = rng.random()*tot  #one draw across the weighted total
        acc = 0.0
        for i,wi in enumerate(w):
            acc+=wi
            if r<=acc:
                return cands[i]
        return cands[-1]

    if ch in store and store[ch]:
        cands = store[ch]
        ranked = sorted(cands,key=lambda g: g.get('priorD',0.5))   #most typical first by distance from the letter reference
        g = Pick(ranked)
        d = float(g.get('priorD',0.5))
        core = _CoreGlyph(ch,profile)
        if core is not None and (forceCore
                                 or rng.random()<_SwapProb(lam,d)):   #the swap is drawn per letter against that chance
            return core[0],core[1],core[2],'core'
        s,a,lead = Pack(g)
        return s,a,lead,'own'
    for alt in (ch.swapcase(),_SIMILAR.get(ch,'')):  #the other case first then a similar shape
        if alt and alt in store and store[alt]:
            core = _CoreGlyph(ch,profile)
            if core is not None and (forceCore
                                     or rng.random()<_SwapProb(lam,0.4)):
                return core[0],core[1],core[2],'core'
            g = Pick(sorted(store[alt],key=lambda x: x.get('priorD',0.5)))
            if alt==ch.swapcase() and ch.isupper():
                #borrow the lowercase shape, grown to capital height
                k = profile.get('ascender',1.7)/max(0.6,g['top'] or 1.0)   #scaled so the lower case shape reaches capital height
                k = float(np.clip(k,1.0,2.0))   # and never shrunk or more than doubled
                s,a,lead = Pack(g,k)
                return s,a,lead,'case'
            s,a,lead = Pack(g)
            return s,a,lead,'similar'
    core = _CoreGlyph(ch,profile)
    if core is not None:
        return core[0],core[1],core[2],'core'
    key = ch.lower() if ch.lower() in _FB else ch
    if key in _FB:
        strokes = [list(map(tuple,s)) for s in _FB[key]]
        return strokes,adv.get(ch,_FB_ADV),0.0,'fallback'   #the generic alphabet with no writer shape behind it
    return [],adv.get(ch,0.6),0.0,'blank'

#---_Jitter: applies a small random shift, rotation and size change to one letter---
def _Jitter(strokes,rng,amp,rotAmp,scaleAmp):
    dx = rng.gauss(0.0,amp)   # a gaussian shift sideways
    dy = rng.gauss(0.0,amp*0.6)   #and a smaller one up or down
    th = math.radians(rng.gauss(0.0,rotAmp))    #a small turn about the origin
    sx = 1.0+rng.gauss(0.0,scaleAmp)   #and a size change on each axis
    sy = 1.0+rng.gauss(0.0,scaleAmp)
    c,s = math.cos(th),math.sin(th)
    out = []
    for stk in strokes:
        q = []
        for (x,y) in stk:
            x2,y2 = x*sx,y*sy
            q.append((x2*c-y2*s+dx,x2*s+y2*c+dy))  #scaled then rotated then shifted
        out.append(q)
    return out

#---_Resample: respaces a path so its points are evenly spread---
def _Resample(poly,step):
    if len(poly)<2:
        return poly
    p = np.asarray(poly,np.float64)
    seg = np.hypot(*np.diff(p,axis=0).T)   # the step lengths along the path
    cum = np.concatenate([[0.0],np.cumsum(seg)])
    total = float(cum[-1])
    if total<1e-9:
        return [tuple(p[0])]
    n = max(2,int(round(total/step))+1)   #how many evenly spread points fit at this step
    t = np.linspace(0.0,total,n)
    x = np.interp(t,cum,p[:,0])   #x and y read off at those distances
    y = np.interp(t,cum,p[:,1])
    return list(zip(x.tolist(),y.tolist()))

#the cached style correction per writer
_STYLE_CAL = {}

#----------------------------------------------------------------------
#StyleCalibration: measures slant and reach off a plain synthesised line and compares them
#with the writer's real pages, giving a correction
#----------------------------------------------------------------------
def StyleCalibration(profile,mmPerXh=4.0,pxPerMm=18.0):
    key = (profile.get('authorId'),round(mmPerXh,3),round(pxPerMm,3))   #the correction is cached per writer and per size
    if key in _STYLE_CAL:
        return _STYLE_CAL[key]
    ref = (profile.get('slantMeasRef'),profile.get('ascMeasRef'),
           profile.get('descMeasRef'))
    cal = dict(shearDelta=0.0,ascK=1.0,descK=1.0)
    if None in ref or any(r!=r for r in ref):   #no measured reference leaves the correction at one
        _STYLE_CAL[key] = cal
        return cal
    slantRef,ascRef,descRef = ref
    sample = ('the quick brown fox jumps over lazy dogs and by half past '
              'eight it kept flying quietly')
    for _ in range(3):   #three rounds of measure and correct
        traj = SynthesizeText(sample,profile,mmPerXh=mmPerXh,seed=17,   # a plain sample line drawn with no wobble
                              lineWidthMm=10_000.0,jitter=0.0,_cal=cal)
        img = RenderTrajectory(traj,pxPerMm=pxPerMm)
        ink = BinarizeLine(np.array(img))   #the render measured the same way a real line is
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
            math.tan(math.radians(slantRef))-math.tan(math.radians(gotSlant)),    #the shear still missing between the real hand and the render
            ascK=float(np.clip(cal['ascK']*(ascRef-1.0)/   #reach above the band scaled towards the real one
                               max(1e-3,gotAsc-1.0),0.5,1.6)),
            descK=float(np.clip(cal['descK']*descRef/min(-1e-3,gotDesc),  # and the same for the reach below it
                                0.5,1.6)))
    _STYLE_CAL[key] = cal
    return cal

#---_ApplyCal: applies that correction to one letter's strokes---
def _ApplyCal(strokes,cal):
    ascK,descK = cal['ascK'],cal['descK']
    if abs(ascK-1.0)<1e-3 and abs(descK-1.0)<1e-3:
        return strokes
    out = []
    for s in strokes:
        q = []
        for (x,y) in s:
            if y>1.0:    #only the parts outside the x-height band are scaled
                y = 1.0+(y-1.0)*ascK
            elif y<0.0:
                y = y*descK
            q.append((x,y))
        out.append(q)
    return out

#----------------------------------------------------------------------
#_Ligature: the joining stroke between two letters, a quadratic whose control height
#follows both ends, so a join that has to climb does not dive first
#----------------------------------------------------------------------
def _Ligature(p0,p1,xh,rng):
    dx = p1[0]-p0[0]
    if abs(dx)<0.06*xh:    # two letters practically touching take a straight join
        return [p0,p1]
    sag = min(0.11*xh,0.16*abs(dx))+rng.gauss(0.0,0.015*xh)   #how far the join dips below the two ends
    ctrlY = min(p0[1],p1[1])+0.35*abs(p1[1]-p0[1])-sag   #the control height follows both ends so a climbing join keeps climbing
    ctrl = ((p0[0]+p1[0])*0.5,ctrlY)
    out = []
    for i in range(9):
        t = i/8.0   #nine points along the curve
        u = 1.0-t
        out.append((u*u*p0[0]+2*u*t*ctrl[0]+t*t*p1[0],   #the quadratic bezier of the two ends and that control point
                    u*u*p0[1]+2*u*t*ctrl[1]+t*t*p1[1]))
    return _Resample(out,step=0.35)

#----------------------------------------------------------------------
#_BodyStroke: which stroke carries the join and which way it is walked, the widest
#stroke that is not a crossbar, left to right while that means something
#and low end last once the letter is too upright for x to decide
#----------------------------------------------------------------------
def _BodyStroke(placed,xh):
    spans = [max(p[0] for p in s)-min(p[0] for p in s) for s in placed]   #the x span of each stroke
    highs = [max(p[1] for p in s)-min(p[1] for p in s) for s in placed]   # and the y span
    cand = [i for i in range(len(placed))   #strokes tall enough against their width stay in
            if highs[i]>=JOIN_FLAT_MAX_ASPECT*max(spans[i],1e-9)]
    if not cand:
        cand = list(range(len(placed)))
    bi = max(cand,key=lambda i: spans[i])   #the widest of those carries the join
    body = list(placed[bi])
    if spans[bi]<JOIN_VERT_MIN_XSPAN_XH*xh:  #too upright for x order to mean anything
        if body[0][1]<body[-1][1]:
            body = body[::-1]   #y is up, so the lower end goes last
    return bi,body

#---_SplitWords: text into words and line breaks---
def _SplitWords(text):
    out = []
    for chunk in text.split('\n'):
        out+=[w for w in chunk.split(' ') if w]  # words of one line
        out.append('\n')   #with a marker where the line breaks
    return out[:-1] if out and out[-1]=='\n' else out

#---_WordWidth: roughly how wide a word will be, for wrapping---
def _WordWidth(word,profile):
    adv = profile.get('letterAdvance',{})
    return sum(adv.get(c,0.9) for c in word)   #the advances of its letters added up

#----------------------------------------------------------------------
#SynthesizeText: text in, a pen path out, for each letter pick a shape, calibrate it,
#wobble it, place it, push it right if it would collide, join it to the
#previous letter if this hand joins, and let the baseline wander slightly
#----------------------------------------------------------------------
def SynthesizeText(text,profile,mmPerXh=4.0,seed=None,lineWidthMm=180.0,
                   jitter=0.15,legibility=None,perCharLam=None,_cal=None,
                   bestPrio=False):
    if legibility is None:
        legibility = float(profile.get('legibilityLambda',
                                       AutoLegibility(profile)))
    rng = random.Random(seed)
    xh = float(mmPerXh)
    cal = _cal if _cal is not None else StyleCalibration(profile,mmPerXh)
    slantDeg = profile.get('slantDeg',0.0)
    #legibility caps the lean of the writer's own glyphs on a soft curve
    authorSlantCap = 45.0-19.0*float(np.clip(legibility,0.0,1.0))   #the lean cap tightens from 45 to 26 degrees as legibility rises
    slantDeg = float(np.clip(slantDeg,-authorSlantCap,authorSlantCap))
    slant = math.tan(math.radians(slantDeg))+cal['shearDelta']   #the slant as a shear plus the measured correction
    conn = float(profile.get('connectedness',0.0))
    #a hand below the floor draws no joins at all
    connEff = 0.0 if conn<JOIN_PRINT_CONN_FLOOR else conn   #a hand under the floor joins nothing
    wordGap = float(profile.get('wordSpaceXh',1.2))
    ascender = float(profile.get('ascender',1.7))
    descender = float(profile.get('descender',-0.6))
    lineStep = (ascender-descender+0.85)*xh   #line pitch covers the full reach plus a gap of 0.85

    ampP = 0.022*jitter   #position jitter, in x-heights
    ampR = 2.2*jitter   #rotation jitter, in degrees
    ampS = 0.05*jitter   #scale jitter
    driftAmp = 0.05*jitter   #slow baseline drift, in x-heights

    strokes = []
    penX,penY = 0.0,0.0
    lineIdx = 0
    drift = 0.0
    driftV = 0.0
    usage = {}
    vpos = 0   #index into the text with newlines as spaces, for perCharLam

    def Flush(chain):
        if len(chain)>=2:
            strokes.append(_Resample(chain,step=0.35))    #every finished pen down path respaced evenly
        elif len(chain)==1:
            strokes.append(chain+[(chain[0][0]+0.05,chain[0][1])])   #a single point is drawn as a very short dash

    for word in _SplitWords(text):
        if word=='\n':
            lineIdx+=1
            penX = 0.0
            penY = -lineIdx*lineStep   #each new line drops by one line pitch
            vpos+=1
            continue
        wWidth = _WordWidth(word,profile)*xh
        if penX>1e-6 and penX+wWidth>lineWidthMm:   #a word past the right edge wraps to the next line
            lineIdx+=1
            penX = 0.0
            penY = -lineIdx*lineStep
        chain = []   #the current pen-down polyline
        prevExit = None
        prevBodyMaxX = None
        prevSrc = None
        prevCoreCursive = False
        for ci,ch in enumerate(word):
            lam = legibility
            forceCore = forcePrint = False
            if perCharLam:
                pc = float(perCharLam.get(vpos,0.0))   # the repair setting for this character position
                lam = max(lam,min(pc,1.0))
                forceCore = pc>=0.999   #a value of 1 forces the fallback letterform
                forcePrint = pc>=1.9    #and 2 forces the printed one
            vpos+=1
            wantEntryY = None
            if connEff>=0.2 and ci>0 and prevExit is not None and \
                    ch.isalpha() and word[ci-1].isalpha():
                wantEntryY = (prevExit[1]-penY)/xh-drift   # the height the pen left the last letter at in x-heights
            gStrokes,adv,lead,src = _GlyphSource(profile,ch,rng,
                                                    prevExitY=wantEntryY,
                                                    lam=lam,forceCore=forceCore,
                                                    forcePrint=forcePrint,
                                                    bestPrio=bestPrio)
            usage[src] = usage.get(src,0)+1
            if not gStrokes:
                penX+=adv*xh
                continue
            isCore = (src=='core')
            coreCursive = isCore and conn>=CORE_CURSIVE_CONN   #a fallback drawn from the joined alphabet
            #the fallback is a designed-legible shape: it keeps the writer's
            #lean but is not re-stretched or wobbled as hard
            if not isCore:
                gStrokes = _ApplyCal(gStrokes,cal)
            jf = CORE_JITTER_FRAC if isCore else 1.0    # a fallback takes a fraction of the wobble
            gStrokes = _Jitter(gStrokes,rng,ampP*jf,ampR*jf,ampS*jf)
            glyphSlant = slant
            glyphDrift = drift
            if isCore:
                capDeg = 30.0 if coreCursive else CORE_SLANT_CAP_DEG
                cap = math.tan(math.radians(capDeg))   #the lean cap as a shear
                glyphSlant = max(-cap,min(cap,slant))
                glyphDrift = (0.5 if coreCursive else 0.25)*drift   # and it follows the baseline drift only part way
            driftV = 0.85*driftV+rng.gauss(0.0,driftAmp*0.35)   #the drift speed carries 85 percent forward plus a nudge
            drift = float(np.clip(drift+driftV,-0.25,0.25))   #and the drift itself stays inside a quarter x-height

            def Place(px):
                out = []
                for stk in gStrokes:
                    out.append([(px+(lead+x+glyphSlant*y)*xh,   #x takes the lead the shape and the shear times height
                                 penY+(y+glyphDrift)*xh) for (x,y) in stk])   # and y takes the height plus the drift
                return out

            placed = Place(penX)
            #if this glyph's ink would land on the previous glyph's body,
            #push it and the pen right for clearance
            if prevBodyMaxX is not None:
                curMinX = min(p[0] for s in placed for p in s)   #the leftmost ink of this letter
                need = prevBodyMaxX+COLLISION_MIN_GAP_XH*xh-curMinX   # how far it has to move right to clear the one before
                if need>0:
                    penX+=need
                    placed = Place(penX)

            #the body stroke carries the join, dots and crossbars are
            #separate pen-downs and must not become the next exit point
            bodyI,body = _BodyStroke(placed,xh)   #the stroke the join attaches to
            bothOwn = not isCore and prevSrc!='core'   # two stored shapes side by side
            bothCursiveCore = coreCursive and prevCoreCursive
            joinGap = ((body[0][0]-prevExit[0])/xh) if prevExit is not None else 9e9   #the gap to bridge in x-heights
            joinable = (ci>0 and prevExit is not None   #both letters inside one word
                        and ch.isalpha() and word[ci-1].isalpha()
                        and (bothOwn or bothCursiveCore)   #a print fallback never joins
                        and joinGap<=MAX_JOIN_GAP_XH
                        and rng.random()<connEff)   #drawn per pair, not by a threshold
            if joinable:
                chain+=_Ligature(prevExit,body[0],xh,rng)   #the join curve then the letter keeps the pen down
                chain+=body
            else:
                Flush(chain)   # otherwise the pen lifts and a new path starts
                chain = list(body)
            prevExit = chain[-1]
            for k2,extra in enumerate(placed):   #dots and crossbars go down as their own strokes
                if k2!=bodyI:
                    Flush(extra)
            prevBodyMaxX = max(p[0] for p in body)
            prevSrc = src
            prevCoreCursive = coreCursive
            penX+=adv*xh
        Flush(chain)
        penX+=wordGap*xh*(0.85+0.3*rng.random())   #the word gap varies between 0.85 and 1.15 of the stored one
        vpos+=1   #the space between words

    meta = dict(author=profile.get('authorId'),mmPerXh=mmPerXh,
                nStrokes=len(strokes),glyphSources=usage,
                slantDeg=profile.get('slantDeg',0.0),
                connectedness=conn,lines=lineIdx+1,
                ascender=ascender,descender=descender,
                #the pitch the text was laid out at, which the G-code writer
                #rescales against
                lineStepMm=lineStep)
    return Trajectory(strokes,meta)

#----------------------------------------------------------------------
#_Draw: rasterises the pen path as black ink on white paper with the round pen,
#then softens the edge slightly
#----------------------------------------------------------------------
def _Draw(traj,pxPerMm,padMm,strokePx,paper=255,ink=0,softPx=0.0):
    x0,y0,x1,y1 = traj.Bounds()
    W = max(8,int((x1-x0+2*padMm)*pxPerMm))  #the image sized from the path bounds plus padding
    H = max(8,int((y1-y0+2*padMm)*pxPerMm))
    arr = np.full((H,W),np.uint8(np.clip(round(float(paper)),0,255)),np.uint8)
    fill = int(round(ink))
    for s in traj.strokes:
        #the trajectory is in mm with y up, the image in pixels with y down
        pts = [((x-x0+padMm)*pxPerMm,(y1-y+padMm)*pxPerMm)   # y is flipped because the image counts rows downwards
               for (x,y) in s]
        DrawPolyline(arr,pts,fill,strokePx)
    if softPx>0.05:
        arr = Blur(arr,softPx)   #sigma is a fraction of the pen width
    return arr

#----------------------------------------------------------------------
#RenderTrajectory: pen path to a crop the same size and look as the training data, so the
#networks' scores on it mean something
#----------------------------------------------------------------------
def RenderTrajectory(traj,pxPerMm=8.0,padMm=None,strokePx=None,size=None):
    mm = traj.meta.get('mmPerXh',4.0)
    if padMm is None:
        padMm = 0.18*mm   #padding proportional to the writing
    sp = strokePx or max(1,int(round(UNIFORM_PEN_WIDTH_XH*mm*pxPerMm)))   #pen width in x-heights turned into pixels
    img = _Draw(traj,pxPerMm,padMm,sp,paper=255,ink=0,
                softPx=UNIFORM_SOFT_FRAC*max(1.0,sp))   #the edge blurred by 30 percent of that width
    if size is not None:
        img = Resize(np.asarray(img,dtype=np.uint8),size[0],size[1],'linear')
    return img

#---_LineLogProbs: the reader's column scores for a rendered line---
def _LineLogProbs(reader,img):
    t = TensorFromResized(ResizeLineImageFixed(img))
    return reader.Forward(t)[:,0,:]   #the column scores for this rendered line

#---_WeakChars: the positions of the characters the reader is least sure about, worst first---
def _WeakChars(reader,img,visible):
    lp = _LineLogProbs(reader,img)
    res = CtcForcedAlign(lp,visible)   #the known text aligned against those scores
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
            scored.append((float(lp[sp[0]:sp[1],CHAR_TO_IDX[c]].mean()),   #the mean score the reader gave this character over its columns
                           keptPos[i]))
    scored.sort()  # the least certain characters come first
    return [pos for _s,pos in scored]

#----------------------------------------------------------------------
#SynthesizeJointBestOf: draws the line several times and keeps the best, scoring every attempt
#with both judges at once: the reader for whether it says the right words
#and the namer for whether it still looks like this writer, weak
#characters are then redrawn with their fallback letterform
#----------------------------------------------------------------------
def SynthesizeJointBestOf(author,text,profile,nTries=6,mmPerXh=4.0,
                          lineWidthMm=180.0,jitter=0.5,seed=0,
                          reader=None,authorModel=None,authorMapping=None,
                          pxPerMm=18.0,legibility=None,
                          repair=True,repairRounds=3,repairWorstK=2,
                          bestPrio=False):
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
    authorIdx = authorMapping[author]   #the class index this writer holds in the namer

    #the harmonic mean, so a candidate weak on either axis is penalised
    #rather than carried by its strength on the other
    def Score(img):
        textAcc = CharAcc(ReadText(img,reader),visible)   #how much of the text reads back correctly
        _predIdx,probs = ClassifyImage(img,authorModel)
        widConf = float(probs[authorIdx])   # and how strongly the namer still calls it this writer
        combined = (2*textAcc*widConf/(textAcc+widConf)   #their harmonic mean so a weak axis drags the score down
                   if (textAcc+widConf)>1e-9 else 0.0)
        return combined,textAcc,widConf

    base = 0 if seed is None else int(seed)
    best,bestTraj,bestImg = -1.0, None, None
    for k in range(max(1,nTries)):
        traj = SynthesizeText(text,profile,mmPerXh=mmPerXh,
                              seed=base+977*k,lineWidthMm=lineWidthMm,   #a different seed for each attempt
                              jitter=jitter,legibility=legibility,
                              bestPrio=bestPrio)
        img = RenderTrajectory(traj,pxPerMm=pxPerMm)
        combined,_textAcc,_widConf = Score(img)
        if combined>best:
            best,bestTraj,bestImg = combined,traj,img
        if best>=0.99:   # a near perfect candidate ends the search
            break

    perChar = {}
    if repair:
        for _round in range(repairRounds):
            if best>=0.995:
                break
            weak = _WeakChars(reader,bestImg,visible)   # the characters the reader struggled with
            fresh = [p for p in weak if p not in perChar][:repairWorstK]    #the worst two not already repaired
            if not fresh:
                break
            trial = dict(perChar)
            for p in fresh:
                trial[p] = 1.0   #those positions forced onto the fallback letterform
            cand = SynthesizeText(text,profile,mmPerXh=mmPerXh,seed=base,
                                  lineWidthMm=lineWidthMm,jitter=jitter,
                                  legibility=legibility,perCharLam=trial,
                                  bestPrio=bestPrio)
            img = RenderTrajectory(cand,pxPerMm=pxPerMm)
            combined,_textAcc,_widConf = Score(img)
            if combined>best:   #the repair is kept only while the score rises
                best,bestTraj,bestImg = combined,cand,img
                perChar = trial

    bestTraj.meta['jointScore'] = round(float(best),3)
    bestTraj.meta['jointTries'] = k+1
    bestTraj.meta['jointRepairedChars'] = len(perChar)
    bestTraj.meta['legibilityLambda'] = round(float(legibility),3)
    bestTraj.meta['bestPrio'] = bool(bestPrio)
    return bestTraj
