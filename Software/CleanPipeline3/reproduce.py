#u23539195 Dylan Braune.  drawing the writing.  FU 7, FU 8, R1, R5
#
#two halves, and they are independent:
#  top     a pen path in millimetres to a g-code file, pure arithmetic, runs
#          anywhere, and this is the half the results were measured on
#  bottom  driving the real gantry: steppers, endstops, homing and the pen
#          lift.  needs the pi and the wiring, so it is guarded
#
#nothing here knows about handwriting.  it takes a list of strokes and draws
#them, whether they came from synthesis or from tracing a real page
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import paths


#======================================================================
#THE MACHINE'S OWN NUMBERS
#gearing, work area, speeds and the pen actuator, all in one place
#======================================================================
#----------------------------------------------------------------------
#GantryConfig: every physical number about the machine
#steps per revolution, belt pitch, work area, feed rates and how the pen
#lift behaves.  it also works out steps per millimetre from the gearing
#rather than having it typed in, so changing a pulley changes one line
#----------------------------------------------------------------------
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
    #zero means "use what the switches measured", which __post_init__ fills
    #in below.  a number here is still trimmed to the measured travel, so
    #nothing can promise more bed than the carriage can reach
    boundsMaxXmm: float = 0.0
    boundsMaxYmm: float = 0.0
    #originXmm is a bed offset, NOT a margin: pageMarginXmm is the margin.
    #it used to be 10, which was added on top of the margin and put every
    #line a centimetre right of where the margin says it should be
    originXmm: float = 0.0
    #the top of the sheet, where line 1 sits.  zero means the top of the
    #calibrated area.  it used to be a hardcoded 180, which on a bed that
    #measures 250.3 mm put the first line 94 mm down the page
    originYmm: float = 0.0
    #--- page margins ---------------------------------------------------
    #drawn output is photographed and read back, and the segmenter needs a
    #page to look like a page: paper around the writing, not writing to the
    #edge.  these match the writers' own sheets, measured off the
    #photographs at about 13 to 15 mm at the left and 24 mm at the top
    pageMarginXmm: float = 15.0
    #1 cm from the calibrated top of the bed to the first line, which is
    #where the ruling on the paper starts.  it was 24, measured off the
    #writers' photographed sheets, which is the right number for a scanned
    #page and the wrong one for placing the pen
    pageMarginYmm: float = 10.0
    #--- which ruled line to write on -----------------------------------
    #without this every job starts at the same height and each new line is
    #drawn over the last one.  the offset moves the whole trajectory down
    #the page in whole ruled lines, so a sheet can be filled one line at a
    #time instead of being re-fed for every word
    lineHeightMm: float = 8.4   #the ruling on the paper being written on
    lineOffsetMm: float = 0.0   #how far down from the first line to start
    #--- sizing the writing to the ruling -------------------------------
    #the synthesiser lays text out at a pitch of its own, (ascender -
    #descender + 0.85) x-heights, which at the default 4 mm x-height comes
    #to between 12 and 18 mm depending on how long the writer's ascenders
    #are.  nothing then related that to the paper, so on an 8.4 mm ruling
    #the writing came out too big and every line after the first landed
    #most of a line low.
    #
    #what gets sized to the ruling is the X-HEIGHT, not the whole letter.
    #ruled paper is not written to by making the ascenders and descenders
    #fit between two rules -- they cross into the neighbouring bands, which
    #is why a hand with long ascenders still sits on an ordinary ruling.
    #measured on the five photographed writers' own pages, x-height over
    #line pitch is
    #
    #    dylan 0.392   owen 0.403   yeukita 0.414   abhinav 0.483
    #    thiya 0.596                              mean 0.458
    #
    #so 0.46 of the ruling is what these hands actually do, and on 8.4 mm
    #that is a 3.8 mm x-height.  forcing the full ascender-to-descender
    #extent inside the band instead gave 1.6 to 2.3 mm, about half size
    fitToRuling: bool = True
    rulingXhFrac: float = 0.46   #x-height as a share of the ruling
    #a cap, not the target: synthesis stretches ascenders and descenders
    #(StyleCalibration) past what the writer's own pages show, and a few
    #outlier glyphs reach further still.  this stops one freak glyph
    #driving the whole job down in size, while leaving normal overshoot
    #into the lines above and below alone
    rulingMaxExtent: float = 1.25   #tallest line, as a multiple of the ruling
    #--- speeds ---------------------------------------------------------
    drawFeedMmMin: float = 900.0   #pen-down feed
    travelFeedMmMin: float = 2400.0   #pen-up rapid
    accelMmS2: float = 400.0   #trapezoidal accel limit
    minSegmentMm: float = 0.12   #collapse shorter moves
    #--- pen actuator (one-way gear motor, 90 deg per toggle) -----------
    #the two dwells are separate because they are not the same job.  lifting
    #only has to clear the paper, so it can be cut hard; dropping has to let
    #the nib seat before the first millimetre is drawn, and cutting that one
    #too far shows up as faint stroke starts rather than as an obvious fault
    penUpDwellMs: float = 350.0   #wait after a lift pulse
    penDownDwellMs: float = 350.0   #wait after a drop pulse, before drawing
    penUpCode: str = 'M5'   #firmware maps these to the pulse
    penDownCode: str = 'M3'
    penStartsUp: bool = True   #assumed state at power-on
    #--- keeping the pen down -------------------------------------------
    #if the next stroke starts this close to where the last one ended, the
    #pen draws straight into it instead of lifting, travelling a fraction of
    #a millimetre and dropping again.  a hand does the same thing, and each
    #avoided lift saves two 90 degree pulses of the one-way actuator
    joinStrokesMm: float = 0.6
    #a stroke whose own two ends are this close is finished off, so a letter
    #drawn as a ring meets itself rather than stopping just short
    closeRingMm: float = 0.6

    #------------------------------------------------------------------
    #the work area is described twice: here, which is what the g-code is
    #planned against, and by Calibrate(), which is what the carriage can
    #actually reach between the switches.  when this one promised more,
    #the extra was not an error -- it was clamped onto the boundary at
    #run time, so the end of every long line was drawn flat against the
    #right edge.  10.8 mm were being lost that way, which is a word
    #------------------------------------------------------------------
    def __post_init__(self):
        usableW = X_SWITCH_TRAVEL_MM-2*EDGE_TOLERANCE_MM
        usableH = Y_SWITCH_TRAVEL_MM-2*EDGE_TOLERANCE_MM
        self.boundsMaxXmm = (usableW if self.boundsMaxXmm <= 0
                             else min(self.boundsMaxXmm, usableW))
        self.boundsMaxYmm = (usableH if self.boundsMaxYmm <= 0
                             else min(self.boundsMaxYmm, usableH))
        #the sheet starts at the top of what the machine can reach, so the
        #first line is one top margin down from there and every line after
        #it is one ruling lower
        if self.originYmm <= 0:
            self.originYmm = self.boundsMaxYmm

    #---usableWidthMm: what is left for writing once both margins are kept---
    @property
    def usableWidthMm(self):
        return self.boundsMaxXmm-self.originXmm-2*self.pageMarginXmm

    @property
    def microstepsPerRev(self):
        return self.fullStepsPerRev*self.microstepping

    @property
    def stepsPerMmX(self):
        return self.microstepsPerRev/self.mmPerRevX

    @property
    def stepsPerMmY(self):
        return self.microstepsPerRev/self.mmPerRevY



#======================================================================
#PEN PATH TO BED COORDINATES
#placing the writing on the bed, in the order that wastes least travel
#======================================================================
#----------------------------------------------------------------------
#ToMachine: put the drawing on the bed at the right size and place
#moves the path to where text starts on the bed, and shrinks it only if
#it would otherwise run off the work area.  never enlarges, so asking
#for small writing gets small writing
#----------------------------------------------------------------------
#----------------------------------------------------------------------
#XhForRuling: the x-height to lay text out at so it fits the ruling
#ToMachine can always shrink an oversized job onto the ruling afterwards,
#but shrinking it afterwards is not enough on its own: the synthesiser has
#already decided where to break lines, and it broke them for the larger
#size, so the shrunk text uses about half the width of the paper and takes
#twice as many lines to say the same thing.  choosing the x-height BEFORE
#laying out puts the line breaks in the right places.
#
#the tallest glyph the text will actually use sets the figure, taken from
#the stored glyphs rather than the writer's nominal ascender, which runs
#short.  ToMachine stays as the safety net if this estimate is optimistic
#----------------------------------------------------------------------
def XhForRuling(profile,cfg,text=None,default=4.0,probe=True):
    ruling = float(getattr(cfg,'lineHeightMm',8.4))
    #the writer's own x-height, if their pages were measured when the
    #profile was built; otherwise what the five photographed hands average
    frac = float(profile.get('xhOverPitch',0.0) or 0.0)
    if not 0.15<=frac<=0.90:
        frac = float(getattr(cfg,'rulingXhFrac',0.46))
    xh = ruling*frac
    if not probe or not text:
        return xh
    #the cap is checked against what synthesis really draws, not against the
    #stored glyph heights: StyleCalibration stretches ascenders and
    #descenders past the figures on the profile, turning a predicted 7.6 mm
    #line into 12.4 mm on owen's hand.  one plain layout is drawn and
    #measured, and the x-height is brought down only if the tallest line
    #breaks the cap.  it is never raised -- the x-height fraction is the
    #target, and the cap exists to stop a freak glyph, not to fill the band
    try:
        import synthesise as SYN
        t = SYN.SynthesizeText(text,profile,mmPerXh=xh,seed=0,
                               lineWidthMm=getattr(cfg,'usableWidthMm',180.0))
        got = _TallestLine(t)
        cap = ruling*float(getattr(cfg,'rulingMaxExtent',1.60))
        if got>cap>0:
            xh *= cap/got
    except Exception:
        pass   #a probe that will not run must not stop a job being written
    return xh


#----------------------------------------------------------------------
#_TallestLine: how much vertical room one line of this text really needs
#groups the strokes onto their own text line by the pitch the synthesiser
#laid them out at, then measures how far the ink reaches either side of
#that line's baseline.
#
#the reach is taken at the 98th and 2nd percentile of the strokes, not at
#the outright highest and lowest.  a handful of strokes come out far taller
#than the writer ever wrote -- on owen the lowest descender reaches 1.19
#x-heights where the 2nd percentile is 0.54 -- and letting one of those set
#the size drove the whole job down by a third.  the robust figure matches
#the writer's own measured proportions almost exactly: owen 2.22 x-heights
#against the 2.19 his profile records for ascender plus descender
#----------------------------------------------------------------------
_EXTENT_HI_PCT = 98.0
_EXTENT_LO_PCT = 2.0

def _TallestLine(traj):
    pitch = float(traj.meta.get('lineStepMm',0.0) or 0.0)
    rows = {}
    for s in traj.strokes:
        if not s:
            continue
        ys = [p[1] for p in s]
        if pitch>1e-6:
            #the stroke belongs to the line its own ink sits nearest to
            k = round(-(sum(ys)/len(ys))/pitch)
            base = -k*pitch
        else:
            k,base = 0,0.0
        rows.setdefault(k,[]).append((max(ys)-base,min(ys)-base))
    if not rows:
        return 0.0
    worst = 0.0
    for v in rows.values():
        tops = np.array([t for t,_b in v],dtype=np.float64)
        bots = np.array([b for _t,b in v],dtype=np.float64)
        if tops.size<8:   #too few strokes for a percentile to mean anything
            worst = max(worst,float(tops.max()-bots.min()))
        else:
            worst = max(worst,float(np.percentile(tops,_EXTENT_HI_PCT)
                                   -np.percentile(bots,_EXTENT_LO_PCT)))
    return worst


def ToMachine(traj,cfg,fitToBounds=True,optimizeOrder=True,rowTolMm=None):
    x0,y0,x1,y1 = traj.Bounds()
    w,h = max(1e-6,x1-x0),max(1e-6,y1-y0)
    scale = 1.0
    mx,my = cfg.pageMarginXmm,cfg.pageMarginYmm
    #the line offset eats into the height still available below it, so a job
    #placed near the foot of the page is scaled to what is left rather than
    #being drawn off the bottom of the sheet
    off = max(0.0,float(getattr(cfg,'lineOffsetMm',0.0)))
    if fitToBounds:
        #the margin is subtracted TWICE: once for the left one the text
        #starts after, once for a right one of the same width.  taking it
        #off only once let a line run to the very edge of the travel, which
        #is where it met the run-time clamp and lost its last word
        availW = cfg.boundsMaxXmm-cfg.originXmm-2*mx
        availH = (cfg.originYmm-my-off)-cfg.boundsMinYmm
        scale = min(1.0,availW/w,availH/h)
        #size the writing to the ruling rather than to whatever x-height the
        #synthesiser happened to use.  this one may scale UP as well as down,
        #so it replaces the natural size instead of capping it -- but it is
        #still held inside the sheet by the width and height limits.
        #
        #this does NOT squeeze the letters between two rules.  the job is
        #laid out at an x-height taken from the ruling (XhForRuling), and
        #ascenders and descenders are left to cross into the bands above and
        #below exactly as they do in the writer's own pages.  all that
        #happens here is the cap: if the tallest line runs past
        #rulingMaxExtent the whole job comes down proportionally
        if getattr(cfg,'fitToRuling',False):
            tall = _TallestLine(traj)
            if tall>1e-6:
                cap = (float(getattr(cfg,'lineHeightMm',8.4))
                       *float(getattr(cfg,'rulingMaxExtent',1.60)))
                scale = min(1.0,cap/tall,availW/w,availH/h)
    #scaling the letters to fit the ruling does NOT by itself put the lines
    #on the ruling: the gap between them comes out as the synthesiser's pitch
    #times that scale, which lands anywhere between 7.0 and 8.7 mm depending
    #on the writer.  the two have to be set separately -- the scale sizes the
    #letters, and each line is then moved onto its own ruled line.
    #
    #each line is placed by its MEASURED centre, not by the pitch it was laid
    #out at.  the synthesiser wanders the baseline as a real hand does
    #(driftV in SynthesizeText) and never resets that wander at a line break,
    #so correcting the nominal pitch alone still left the lines 8.45 to 8.51
    #apart instead of 8.40.  measuring each line and placing it takes the
    #drift out from between the lines while leaving it inside them
    pitch = float(traj.meta.get('lineStepMm',0.0) or 0.0)
    ruledPitch = float(getattr(cfg,'lineHeightMm',8.4))
    respace = (getattr(cfg,'fitToRuling',False) and pitch>1e-6
               and int(traj.meta.get('lines',1) or 1)>1)
    centres = {}
    if respace:
        rows = {}
        for s in traj.strokes:
            if not s:
                continue
            k = round(-(sum(p[1] for p in s)/len(s))/pitch)
            ys = [p[1] for p in s]
            lo,hi = rows.get(k,(1e18,-1e18))
            rows[k] = (min(lo,min(ys)),max(hi,max(ys)))
        if rows:
            k0 = min(rows)   #the topmost line stays where it is
            c0 = (rows[k0][0]+rows[k0][1])*0.5
            for k,(lo,hi) in rows.items():
                ck = (lo+hi)*0.5
                #put this line's centre exactly (k-k0) rulings below line k0
                centres[k] = (c0-ck)*scale-(k-k0)*ruledPitch
    out = []
    for s in traj.strokes:
        #the line this stroke belongs to, from where its own ink sits
        shift = 0.0
        if respace and s:
            k = round(-(sum(p[1] for p in s)/len(s))/pitch)
            shift = centres.get(k,0.0)
        #both frames have y UP (standard plotter convention): the text's
        #top row lands at originY and later rows descend from there
        q = [(cfg.originXmm+mx+(x-x0)*scale,
              cfg.originYmm-my-off-(y1-y)*scale+shift)
             for (x,y) in s]
        out.append(_Decimate(q,cfg.minSegmentMm))
    if optimizeOrder:
        out = OptimizeStrokeOrder(out,rowTolMm=rowTolMm)
    return out,scale

#----------------------------------------------------------------------
#_Decimate: drop points closer together than the machine can resolve
#a path from a traced stroke can have a point per pixel.  moves shorter
#than the minimum segment are merged, which shortens the file a lot and
#changes nothing you could see
#----------------------------------------------------------------------
def _Decimate(poly,minSeg):
    if len(poly)<2:
        return poly
    out = [poly[0]]
    for p in poly[1:-1]:
        if math.dist(p,out[-1])>=minSeg:
            out.append(p)
    out.append(poly[-1])
    return out

#----------------------------------------------------------------------
#OptimizeStrokeOrder: draw the strokes in a sensible order
#works roughly in bands down the page and, inside a band, takes the
#nearest unvisited stroke next.  cuts the time spent moving with the
#pen up, which is wasted time
#----------------------------------------------------------------------
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

#---_Clamp: keep every point inside the work area, as a last safety net---
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



#======================================================================
#WRITING THE G-CODE FILE
#the pen toggle is one way, so its state has to be tracked, not commanded
#======================================================================
#---NewPenState: start tracking where the pen is, up or down---
def NewPenState(cfg):
    return [cfg.penStartsUp,0]

#----------------------------------------------------------------------
#SetPenLines: the lines needed to get the pen to a wanted state
#the actuator turns 90 degrees per pulse and cannot be told an absolute
#position, so this tracks what state it should be in and only emits a
#toggle when the state actually needs to change
#----------------------------------------------------------------------
def SetPenLines(state,cfg,wantUp):
    if wantUp==state[0]:
        return []
    state[0] = wantUp
    state[1]+= 1
    code = cfg.penUpCode if wantUp else cfg.penDownCode
    #dwell = one 90 degree turn of the cam plus settling, in seconds
    dwell = (cfg.penUpDwellMs if wantUp else cfg.penDownDwellMs)/1000.0
    out = [f"{code}   ; pen {'UP' if wantUp else 'DOWN'} "
           f"(one 90deg pulse of the one-way gear motor)"]
    if dwell>0:   #a zero dwell is a valid setting: emit no wait at all
        out.append(f"G4 P{dwell:.3f}   ; wait for the 90deg rotation to finish")
    return out

#----------------------------------------------------------------------
#WriteGcode: a pen path in, a g-code file out
#places and orders the strokes, then writes a move to the start of each
#stroke with the pen up, a pen down, the stroke itself at drawing feed,
#and a pen up at the end
#----------------------------------------------------------------------
def WriteGcode(traj,cfg,path,title='',rowTolMm=None):
    strokes,scale = ToMachine(traj,cfg,rowTolMm=rowTolMm)
    strokes,clipped = _Clamp(strokes,cfg)
    pen = NewPenState(cfg)
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
    L+=SetPenLines(pen,cfg,True)
    L.append(f"G0 F{cfg.travelFeedMmMin:.0f}")
    here = None        #where the pen actually is, in millimetres
    joined = rings = 0
    joinedMm = 0.0
    #distance is split by pen state so the page can report what a feed change
    #actually buys, instead of estimating off one fixed test word
    drawMm = travelMm = 0.0
    for s in strokes:
        if len(s)<2:
            continue
        #a ring is finished off so the shape meets itself
        if (cfg.closeRingMm>0 and len(s)>3
                and math.dist(s[0],s[-1])<=cfg.closeRingMm
                and s[0]!=s[-1]):
            s = list(s)+[s[0]]
            rings += 1
        if here is not None and math.dist(here,s[0])<=cfg.joinStrokesMm:
            #already there: draw into this stroke rather than lift and land.
            #the gap is drawn as ink, so it is counted: a join wider than a
            #fraction of the x-height would show as a line that is not writing
            L.append(f"G1 F{cfg.drawFeedMmMin:.0f}")
            joined += 1
            joinedMm += math.dist(here,s[0])
            drawMm += math.dist(here,s[0])
        else:
            L+=SetPenLines(pen,cfg,True)
            #F is modal and shared between G0 and G1, so the travel feed set
            #in the header is overwritten by the first G1 drawing feed and
            #never comes back.  every travel after stroke one was crossing
            #the page at the DRAWING speed until this was restated here
            L.append(f"G0 F{cfg.travelFeedMmMin:.0f} "
                     f"X{s[0][0]:.3f} Y{s[0][1]:.3f}")
            if here is not None:
                travelMm += math.dist(here,s[0])
            L+=SetPenLines(pen,cfg,False)
            L.append(f"G1 F{cfg.drawFeedMmMin:.0f}")
        prev = s[0]
        for (x,y) in s[1:]:
            L.append(f"G1 X{x:.3f} Y{y:.3f}")
            drawMm += math.dist(prev,(x,y))
            prev = (x,y)
        here = s[-1]
    L+=SetPenLines(pen,cfg,True)
    L.append(f"G0 F{cfg.travelFeedMmMin:.0f} "
             f"X{cfg.originXmm:.3f} Y{cfg.originYmm:.3f}   ; park")
    if here is not None:
        travelMm += math.dist(here,(cfg.originXmm,cfg.originYmm))
    L.append('M2')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        f.write('\n'.join(L)+'\n')
    #where it actually landed, so the page can be drawn rather than guessed
    #at.  these are machine millimetres with y measured UP from the bed, the
    #same frame the g-code is in
    ys = [p[1] for s in strokes for p in s] or [0.0]
    xs = [p[0] for s in strokes for p in s] or [0.0]
    lineH = max(1e-6,float(getattr(cfg,'lineHeightMm',8.4)))
    off = float(getattr(cfg,'lineOffsetMm',0.0))
    firstLineY = cfg.originYmm-cfg.pageMarginYmm
    #a budget for the job, split into the three things a setting can change.
    #this ignores acceleration, so it is a floor on the real time, not a
    #prediction of it -- short segments never reach their commanded feed
    drawS = 60.0*drawMm/max(1.0,cfg.drawFeedMmMin)
    travelS = 60.0*travelMm/max(1.0,cfg.travelFeedMmMin)
    #pen[1] counts toggles but not which way, so the dwell is counted as the
    #mean of the two: lifts and drops alternate, so over a page they are equal
    dwellS = pen[1]*(cfg.penUpDwellMs+cfg.penDownDwellMs)/2000.0
    return dict(lines=len(L),strokes=len(strokes),penPulses=pen[1],
                scale=scale,clipped=clipped,path=str(path),
                joinedStrokes=joined,closedRings=rings,
                joinedInkMm=round(joinedMm,2),
                drawMm=round(drawMm,1),travelMm=round(travelMm,1),
                drawSeconds=round(drawS,1),travelSeconds=round(travelS,1),
                dwellSeconds=round(dwellS,1),
                estimatedSeconds=round(drawS+travelS+dwellS,1),
                placement=dict(topYmm=round(max(ys),2),bottomYmm=round(min(ys),2),
                               leftXmm=round(min(xs),2),rightXmm=round(max(xs),2),
                               lineOffsetMm=round(off,2),lineHeightMm=round(lineH,3),
                               lineIndex=int(round(off/lineH)),
                               #ruled lines this job itself occupies.  the page
                               #has to advance by this, not by one, or a job
                               #that wrapped is written over by the next one
                               textLines=max(1,int(traj.meta.get('lines',1) or 1)),
                               firstLineYmm=round(firstLineY,2),
                               #how many whole ruled lines fit below the first
                               linesOnPage=int((firstLineY-cfg.boundsMinYmm)//lineH)+1,
                               bedTopYmm=round(cfg.originYmm,2),
                               bedBottomYmm=round(cfg.boundsMinYmm,2),
                               bedLeftXmm=round(cfg.originXmm,2),
                               bedRightXmm=round(cfg.boundsMaxXmm,2)))

#----------------------------------------------------------------------
#ParseGcode: read a g-code file back into strokes
#written deliberately as a separate reader, not sharing code with the
#writer, so comparing the two actually tests the file rather than testing
#one shared assumption twice
#----------------------------------------------------------------------
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



#======================================================================
#TALKING TO THE HARDWARE
#from here down needs the real machine.  the imports are guarded so the
#top half of this file still works on a laptop
#======================================================================
#the gpiod v2 python api: request_lines and LineSettings, with the enums
#in gpiod.line.  on anything that is not the board this fails, and the
#names are bound to None instead so that Connect() can report it as a
#missing gantry rather than raising a NameError on the check itself
try:
    import gpiod
    from gpiod.line import Direction,Value
    _GPIOD_IMPORT_ERROR = None
except Exception as _e:
    gpiod = None
    Direction = None
    Value = None
    _GPIOD_IMPORT_ERROR = _e

#---GantryNotConnectedError: raised when something needs hardware that is not there---
class GantryNotConnectedError(RuntimeError):
    #no gpiod / no gpio chip available -- not the odroid, or the device
    #couldn't be opened. callers should report this cleanly instead of a
    #raw exception.
    pass

#---NotCalibratedError: raised when a move is asked for before homing---
class NotCalibratedError(RuntimeError):
    #Calibrate() hasn't succeeded yet this session. never persisted to disk:
    #a mechanical bump between sessions would silently invalidate a saved
    #value, so it must be redone every restart.
    pass

#---CHIP_PATH: which gpio chip the pins are on---
CHIP_PATH = "/dev/gpiochip0"

#---the gpio library import, guarded so this file still loads without it---
DIR1_PIN,STEP1_PIN = 64,68   #x motor -- physical pins 7, 11

#------
DIR2_PIN,STEP2_PIN = 81,69   #y motor -- physical pins 12, 13

#------
X_MIN_PIN,X_MAX_PIN = 72,65  #physical pins 15, 16

#------
Y_MIN_PIN,Y_MAX_PIN = 66,67  #physical pins 18, 22



#======================================================================
#THE WIRING AND THE STEPPER NUMBERS
#which pin is what, and how many steps make a millimetre
#======================================================================
#---PEN_SWITCH_PIN: the switch that reads whether the pen is up---
PEN_SWITCH_PIN = 70          #physical pin 33 (input, feedback)

#---PEN_MOTOR_PIN: the pen lift motor---
PEN_MOTOR_PIN = 71           #physical pin 35 (output)

#---ENDSTOP_PINS: the four limit switches, by name---
ENDSTOP_PINS = {"X_MIN": X_MIN_PIN,"X_MAX": X_MAX_PIN,
                "Y_MIN": Y_MIN_PIN,"Y_MAX": Y_MAX_PIN}

#---INVERT_X: whether x counts up the other way---
INVERT_X = True

#---INVERT_Y: and y---
INVERT_Y = True

#---MICROSTEPS: microstepping the driver is set to---
MICROSTEPS = 16

#---STEPS_PER_REV: full steps in one motor revolution---
STEPS_PER_REV = 200*MICROSTEPS

#---STEPS_PER_MM: worked out from the gearing above---
STEPS_PER_MM = 80.0   #only used before Calibrate() has run

#---CIRCLE_STEP_DELAY_S: pause between steps, which sets the speed---
CIRCLE_STEP_DELAY_S = 0.000002

#---DEFAULT_CIRCLE_RADIUS_MM: radius for the test circle---
DEFAULT_CIRCLE_RADIUS_MM = 8.0

#----------------------------------------------------------------------
#X_SWITCH_TRAVEL_MM / Y_SWITCH_TRAVEL_MM: how far the carriage moves
#between the two switches on each axis.  Calibrate() divides the steps it
#counted by these, so they set steps per millimetre for everything after:
#a travel figure that is too big makes steps/mm too small, and every
#drawn length comes out short by the same proportion
#
#the 100 x 100 mm test square measured 97.6 wide and 99.4 tall, which puts
#the true travels at 204.0 x 97.6/100 = 199.2 and 262.0 x 99.4/100 = 260.3.
#the old figures were nominal rail lengths, not measured ones
#
#these are worked back from ONE drawn square and should be confirmed by
#drawing another and measuring it.  a square that still comes out wrong by
#the same proportion means the travel is still off; one that is right in
#length but not square is the separate mechanical fault below
#----------------------------------------------------------------------
X_SWITCH_TRAVEL_MM = 199.2   #was 204.0, nominal

#---Y_SWITCH_TRAVEL_MM: the same for y---
Y_SWITCH_TRAVEL_MM = 260.3   #was 262.0, nominal

#----------------------------------------------------------------------
#SQUARENESS: the axes are not quite at right angles to each other
#the same test square was out by about 1 degree, which is 1.75 mm of skew
#over 100 mm.  that is mechanical -- the x rail is not perpendicular to
#the y rail -- and no number here can correct it, because a skew is not a
#scale.  it is recorded so the next person measuring a square knows that
#about 1.75 mm of the error is expected and is not a calibration fault
#----------------------------------------------------------------------
SQUARENESS_ERROR_DEG = 1.0

#---EDGE_TOLERANCE_MM: margin kept off the limits---
EDGE_TOLERANCE_MM = 5.0        #kept clear of both switches on both axes

#---HOMING_STEP_DELAY_S: homing is run slowly on purpose---
HOMING_STEP_DELAY_S = 0.0008   #per-step pulse time while homing (both axes)

#---HOMING_MAX_STEPS: give up homing after this many steps---
HOMING_MAX_STEPS = 200_000     #a switch that never triggers is a fault

#---ENDSTOP_DEBOUNCE_S: ignore switch chatter for this long---
ENDSTOP_DEBOUNCE_S = 0.006     #gap between confirmation samples below

#---ENDSTOP_CONFIRM_SAMPLES: a switch must read triggered this many times to count---
ENDSTOP_CONFIRM_SAMPLES = 3    #reads in a row before a switch is trusted,

#---RETRACT_MM: how far to back off a switch once it trips---
RETRACT_MM = 5.0               #0.5cm pulled back off a switch on emergency

#---PEN_UP_OVERRUN_S: how long the pen pulse is allowed to run---
PEN_UP_OVERRUN_S = 0.04   #extra run time after the switch first trips, so

#---GCODE_MIN_FEED_MM_S: slowest feed the runner will accept---
GCODE_MIN_FEED_MM_S = 0.1   #floor so a stray f0 doesn't divide by zero

#---_GCODE_WORD_RE: splits a g-code line into letter and number---
_GCODE_WORD_RE = re.compile(r'([A-Za-z])\s*(-?[0-9]*\.?[0-9]+)')

#---_RETREAT_SIGN: which way to move to come off each switch---
_RETREAT_SIGN = {"X_MIN":+1,"X_MAX":-1,"Y_MIN":+1,"Y_MAX":-1}



#======================================================================
#WHERE THE MACHINE CURRENTLY IS
#module level state, because there is only ever one machine
#======================================================================
#---Calibration: the work area found by homing, or None---
Calibration = {
    "done": False,
    "steps_per_mm_x": STEPS_PER_MM,
    "steps_per_mm_y": STEPS_PER_MM,
    "usable_width_mm": X_SWITCH_TRAVEL_MM-2*EDGE_TOLERANCE_MM,
    "usable_height_mm": Y_SWITCH_TRAVEL_MM-2*EDGE_TOLERANCE_MM,
}

#---req: the gpio request handle---
req = None                #live gpiod line request once Connect() succeeds

#---CurrentMode: what the machine is doing now---
CurrentMode = "RPM"        #RPM | CIRCLE | GCODE | CALIBRATING

#---targetRpm: wanted speed for the test spin---
targetRpm = 0.0            #must start at 0: nonzero here would move the

#---stepDelayS: the pause that produces that speed---
stepDelayS = 0.0           #gantry the instant the main loop starts running

#---running: whether the worker thread should keep going---
running = True

#---CurrentXSteps: where x is, counted in steps from home---
CurrentXSteps = 0

#---CurrentYSteps: and y---
CurrentYSteps = 0

#---theta: angle around the test circle---
theta = 0.0

#---LastCircleTime: when the last circle step happened---
LastCircleTime = time.perf_counter()

#---CircleIntervalS: gap between circle steps---
CircleIntervalS = 0.0015

#---CircleRadiusMm: the test circle's radius---
CircleRadiusMm = DEFAULT_CIRCLE_RADIUS_MM

#---CircleRadiusSteps: that radius in steps---
CircleRadiusSteps = CircleRadiusMm*STEPS_PER_MM

#---estopped: true once an emergency stop has fired---
estopped = False

#---TrippedSwitch: which switch caused it---
TrippedSwitch = None



#======================================================================
#CONNECTING, AND THE SAFETY CHECKS
#nothing moves until the pins are claimed and no switch is already pressed
#======================================================================
#---Connect: claim the gpio pins and set them to a safe state---
def Connect():
    global req
    if req is not None:
        return
    if gpiod is None:
        raise GantryNotConnectedError(f"gpiod not available: {_GPIOD_IMPORT_ERROR}")
    try:
        req = gpiod.request_lines(
            CHIP_PATH,
            consumer="stepper-control",
            config={
                DIR1_PIN: gpiod.LineSettings(direction=Direction.OUTPUT,output_value=Value.ACTIVE),
                STEP1_PIN: gpiod.LineSettings(direction=Direction.OUTPUT,output_value=Value.INACTIVE),
                DIR2_PIN: gpiod.LineSettings(direction=Direction.OUTPUT,output_value=Value.ACTIVE),
                STEP2_PIN: gpiod.LineSettings(direction=Direction.OUTPUT,output_value=Value.INACTIVE),
                X_MIN_PIN: gpiod.LineSettings(direction=Direction.INPUT),
                X_MAX_PIN: gpiod.LineSettings(direction=Direction.INPUT),
                Y_MIN_PIN: gpiod.LineSettings(direction=Direction.INPUT),
                Y_MAX_PIN: gpiod.LineSettings(direction=Direction.INPUT),
                PEN_SWITCH_PIN: gpiod.LineSettings(direction=Direction.INPUT),
                PEN_MOTOR_PIN: gpiod.LineSettings(direction=Direction.OUTPUT,output_value=Value.INACTIVE),
            },
        )
    except Exception as e:
        req = None
        raise GantryNotConnectedError(f"Could not open {CHIP_PATH}: {e}") from e
    CalculateDelay(targetRpm)

#---Disconnect: release the pins again---
def Disconnect():
    global req
    if req is None:
        return
    try:
        req.set_value(STEP1_PIN,Value.INACTIVE)
        req.set_value(STEP2_PIN,Value.INACTIVE)
        req.set_value(PEN_MOTOR_PIN,Value.INACTIVE)
    finally:
        req.release()
        req = None

#---IsConnected: whether the pins are currently claimed---
def IsConnected():
    return req is not None

#---CalculateDelay: turn a wanted rpm into a step pause---
def CalculateDelay(rpm):
    global stepDelayS
    if abs(rpm)<0.1:
        stepDelayS = 0
    else:
        stepDelayS = (60.0/(abs(rpm)*STEPS_PER_REV))/2.0

#----------------------------------------------------------------------
#TriggeredEndstop: is a limit switch pressed right now
#reads each switch several times and only believes it if it reads the
#same way every time, because a long wire picks up noise
#----------------------------------------------------------------------
def TriggeredEndstop(ignore=frozenset()):
    for name,pin in ENDSTOP_PINS.items():
        if name in ignore:
            continue
        if req.get_value(pin)!=Value.ACTIVE:
            continue
        for _ in range(ENDSTOP_CONFIRM_SAMPLES-1):   #confirm samples
            time.sleep(ENDSTOP_DEBOUNCE_S)
            if req.get_value(pin)!=Value.ACTIVE:
                break
        else:
            return name
    return None

#---_AllTriggeredEndstops: every switch currently reading pressed---
def _AllTriggeredEndstops():
    return [name for name in ENDSTOP_PINS if TriggeredEndstop(ignore=set(ENDSTOP_PINS)-{name})==name]

#---AxisDirValue: the direction pin level for a wanted direction---
def AxisDirValue(invert,positive):
    forward = positive!=invert   #xor
    return Value.ACTIVE if forward else Value.INACTIVE

#----------------------------------------------------------------------
#_RetractOffSwitch: back off a switch that is pressed
#a pressed switch means the axis is at its limit, so it is stepped the
#other way until the switch releases, before anything else is allowed
#----------------------------------------------------------------------
def _RetractOffSwitch(hit,ignore=frozenset()):
    global CurrentXSteps,CurrentYSteps
    isX = hit.startswith("X")
    sign = _RETREAT_SIGN[hit]
    dirPin,stepPin = (DIR1_PIN,STEP1_PIN) if isX else (DIR2_PIN,STEP2_PIN)
    invert = INVERT_X if isX else INVERT_Y
    stepsPerMm = Calibration["steps_per_mm_x"] if isX else Calibration["steps_per_mm_y"]
    TargetSteps = max(1,round(RETRACT_MM*stepsPerMm))

    req.set_value(dirPin,AxisDirValue(invert,sign>0))
    moved = 0
    for _ in range(TargetSteps):
        if TriggeredEndstop(ignore={hit}|set(ignore)) is not None:
            break   #something else tripped -- a real fault, stop pushing
        req.set_value(stepPin,Value.ACTIVE)
        time.sleep(HOMING_STEP_DELAY_S)
        req.set_value(stepPin,Value.INACTIVE)
        time.sleep(HOMING_STEP_DELAY_S)
        moved+=sign

    if isX:
        CurrentXSteps+=moved
    else:
        CurrentYSteps+=moved

#----------------------------------------------------------------------
#_TriggerEmergencyStop: stop everything, now
#cuts the drive, lifts the pen, marks the machine stopped and remembers
#which switch did it.  nothing moves again until it is cleared
#----------------------------------------------------------------------
def _TriggerEmergencyStop(hit):
    global estopped,TrippedSwitch
    if estopped:
        return
    estopped,TrippedSwitch = True,hit
    req.set_value(STEP1_PIN,Value.INACTIVE)
    req.set_value(STEP2_PIN,Value.INACTIVE)
    allHits = _AllTriggeredEndstops() or [hit]
    print(f"\n!!! EMERGENCY STOP -- end-stop(s) {', '.join(allHits)} triggered !!! "
          f"Retracting {RETRACT_MM}mm off each...")
    for h in allHits:
        _RetractOffSwitch(h,ignore=set(allHits)-{h})
    print("Retracted. Clear with 'r' once safe to resume.")



#======================================================================
#MOVING
#straight lines in steps, and the pen up and down
#======================================================================
#----------------------------------------------------------------------
#BresenhamMove: a straight line in whole steps
#keeps a running error term and steps whichever axis is furthest behind,
#so the two motors together stay on the straight line.  no floating
#point and no division per step
#----------------------------------------------------------------------
def BresenhamMove(dx,dy,stepDelayS=CIRCLE_STEP_DELAY_S):
    global CurrentXSteps,CurrentYSteps
    ax,ay = abs(dx),abs(dy)
    sx = 1 if dx>=0 else -1   #sign of each axis's motion
    sy = 1 if dy>=0 else -1
    req.set_value(DIR1_PIN,AxisDirValue(INVERT_X,sx>0))
    req.set_value(DIR2_PIN,AxisDirValue(INVERT_Y,sy>0))

    def pulse(stepPin):   #one step pulse: high, wait, low
        req.set_value(stepPin,Value.ACTIVE)
        time.sleep(stepDelayS)
        req.set_value(stepPin,Value.INACTIVE)

    if ax>=ay:   #x is the major axis
        err = ax//2
        for _ in range(ax):
            if TriggeredEndstop() is not None:
                return
            pulse(STEP1_PIN)
            CurrentXSteps+=sx
            err-=ay
            if err<0:   #minor axis (y) catches up by one step
                pulse(STEP2_PIN)
                CurrentYSteps+=sy
                err+=ax
    else:   #y is the major axis -- mirror image of the branch above
        err = ay//2
        for _ in range(ay):
            if TriggeredEndstop() is not None:
                return
            pulse(STEP2_PIN)
            CurrentYSteps+=sy
            err-=ax
            if err<0:
                pulse(STEP1_PIN)
                CurrentXSteps+=sx
                err+=ay

#---PenIsUp: read the pen switch---
def PenIsUp():
    return req.get_value(PEN_SWITCH_PIN)==Value.ACTIVE

#----------------------------------------------------------------------
#SetPen: raise or lower the pen and confirm it happened
#pulses the one way motor and watches the switch, giving up with an error
#if the pen never reports the state asked for
#----------------------------------------------------------------------
def SetPen(targetUp,timeoutS=3.0):
    if TriggeredEndstop() is not None:
        print(f"Refusing pen move -- end-stop {TrippedSwitch} still tripped.")
        return False
    if PenIsUp()==targetUp:
        return True   #already there -- the toggle motor must not run

    req.set_value(PEN_MOTOR_PIN,Value.ACTIVE)
    deadline = time.time()+timeoutS
    reached = False
    while time.time()<deadline:
        if TriggeredEndstop() is not None:
            break
        if PenIsUp()==targetUp:
            reached = True
            break
        time.sleep(0.002)
    if reached and targetUp:
        time.sleep(PEN_UP_OVERRUN_S)
    req.set_value(PEN_MOTOR_PIN,Value.INACTIVE)

    if not reached:
        print("Pen move did NOT confirm via switch -- check motor/wiring.")
    return reached

#---_StepAxis: step one axis until a stopping condition is met---
def _StepAxis(stepPin,dirPin,dirValue,stopWhen,ignore=frozenset()):
    req.set_value(dirPin,dirValue)
    steps = 0
    while True:
        hit = TriggeredEndstop(ignore=ignore)
        if hit is not None:
            if stopWhen(hit):
                return hit,steps
            raise RuntimeError(f"Unexpected end-stop {hit} triggered -- check wiring.")
        if steps>=HOMING_MAX_STEPS:
            raise RuntimeError("No expected end-stop triggered during homing -- check wiring.")
        req.set_value(stepPin,Value.ACTIVE)
        time.sleep(HOMING_STEP_DELAY_S)
        req.set_value(stepPin,Value.INACTIVE)
        time.sleep(HOMING_STEP_DELAY_S)
        steps+=1

#---_CalibrateAxis: drive one axis to both ends to learn its travel---
def _CalibrateAxis(stepPin,dirPin,minName,maxName,OtherAxisNames=frozenset()):
    print(f"[calibrate] driving until either {minName} or {maxName} is activated...")
    firstHit,_ = _StepAxis(stepPin,dirPin,Value.ACTIVE,
                               stopWhen=lambda h: h in (minName,maxName),
                               ignore=OtherAxisNames)
    other = maxName if firstHit==minName else minName
    print(f"[calibrate] {firstHit} activated. Reversing toward {other}...")
    #ignore firstHit too: it may still read triggered for the first few
    #steps of the reverse move (release lag), not a real fault.
    secondHit,TravelSteps = _StepAxis(stepPin,dirPin,Value.INACTIVE,
                                           stopWhen=lambda h: h==other,
                                           ignore={firstHit}|set(OtherAxisNames))
    print(f"[calibrate] {secondHit} activated. {TravelSteps} steps measured.")
    return firstHit,secondHit,TravelSteps

#----------------------------------------------------------------------
#Calibrate: home the machine and learn the work area
#drives each axis to both limits, allows for how far a switch moves
#before it trips, and stores the usable area.  until this has run, no
#absolute move is accepted
#----------------------------------------------------------------------
def Calibrate():
    global CurrentXSteps,CurrentYSteps

    if TriggeredEndstop() is not None:
        raise RuntimeError(f"Cannot calibrate -- end-stop {TrippedSwitch} already tripped.")

    print("[calibrate] === X axis ===")
    _,xSecond,xTravel = _CalibrateAxis(STEP1_PIN,DIR1_PIN,"X_MIN","X_MAX")
    CurrentXSteps = xTravel if xSecond=="X_MAX" else 0
    Calibration["steps_per_mm_x"] = xTravel/X_SWITCH_TRAVEL_MM
    _RetractOffSwitch(xSecond)
    print(f"[calibrate] retracted off {xSecond}.")

    print("[calibrate] === Y axis ===")
    _,ySecond,yTravel = _CalibrateAxis(
        STEP2_PIN,DIR2_PIN,"Y_MIN","Y_MAX",OtherAxisNames={"X_MIN","X_MAX"})
    CurrentYSteps = yTravel if ySecond=="Y_MAX" else 0
    Calibration["steps_per_mm_y"] = yTravel/Y_SWITCH_TRAVEL_MM
    _RetractOffSwitch(ySecond)
    print(f"[calibrate] retracted off {ySecond}.")

    Calibration["done"] = True
    Calibration["usable_width_mm"] = X_SWITCH_TRAVEL_MM-2*EDGE_TOLERANCE_MM
    Calibration["usable_height_mm"] = Y_SWITCH_TRAVEL_MM-2*EDGE_TOLERANCE_MM
    _GcodeLinearMove(0.0,0.0,feedMmMin=600.0)   #park at the safe start corner
    return dict(Calibration)



#======================================================================
#RUNNING A G-CODE FILE ON THE MACHINE
#the reader for the file, and the moves it turns into
#======================================================================
#---ParseGcodeLine: split one line into its letters and numbers---
def ParseGcodeLine(raw):
    line = raw.split(';',1)[0]
    line = re.sub(r'\([^)]*\)','',line).strip()
    if not line:
        return None
    cmd,params = None,{}
    for letter,num in _GCODE_WORD_RE.findall(line):
        letter = letter.upper()
        val = float(num)
        if cmd is None and letter in ('G','M'):
            cmd = f"{letter}{int(val)}"
        else:
            params[letter] = val
    return (cmd,params) if cmd else None

#---_ClampToUsableArea: keep a target inside the homed area---
def _ClampToUsableArea(xMm,yMm):
    w,h = Calibration["usable_width_mm"],Calibration["usable_height_mm"]
    cx,cy = min(max(xMm,0.0),w),min(max(yMm,0.0),h)
    if (cx,cy)!=(xMm,yMm):
        print(f"[gcode] clamped out-of-bounds target ({xMm:.1f}, {yMm:.1f}) -> ({cx:.1f}, {cy:.1f}) mm")
    return cx,cy

#---_GcodeLinearMove: carry out a straight move at a feed rate---
def _GcodeLinearMove(targetXMm,targetYMm,feedMmMin):
    targetXMm,targetYMm = _ClampToUsableArea(targetXMm,targetYMm)
    spmmX,spmmY = Calibration["steps_per_mm_x"],Calibration["steps_per_mm_y"]
    #mm=0 in this coordinate frame is EDGE_TOLERANCE_MM in from the
    #x_min/y_min switch, not the switch itself -- see Calibrate().
    x0Mm = CurrentXSteps/spmmX-EDGE_TOLERANCE_MM
    y0Mm = CurrentYSteps/spmmY-EDGE_TOLERANCE_MM
    distMm = math.hypot(targetXMm-x0Mm,targetYMm-y0Mm)
    TargetXSteps = round((targetXMm+EDGE_TOLERANCE_MM)*spmmX)
    TargetYSteps = round((targetYMm+EDGE_TOLERANCE_MM)*spmmY)
    dxSteps = TargetXSteps-CurrentXSteps
    dySteps = TargetYSteps-CurrentYSteps
    if dxSteps==0 and dySteps==0:
        return
    majorSteps = max(abs(dxSteps),abs(dySteps),1)
    feedMmS = max(feedMmMin/60.0,GCODE_MIN_FEED_MM_S)
    stepDelayS = max((distMm/feedMmS)/majorSteps,CIRCLE_STEP_DELAY_S)
    BresenhamMove(dxSteps,dySteps,stepDelayS=stepDelayS)

#----------------------------------------------------------------------
#_GcodeArcMove: carry out an arc move
#splits the arc into short straight moves, because the machine only
#knows how to go straight
#----------------------------------------------------------------------
def _GcodeArcMove(x0Mm,y0Mm,x1Mm,y1Mm,iMm,jMm,clockwise,feedMmMin):
    cx,cy = x0Mm+iMm,y0Mm+jMm
    radius = math.hypot(iMm,jMm)
    if radius<1e-6:
        _GcodeLinearMove(x1Mm,y1Mm,feedMmMin)
        return
    startAngle = math.atan2(y0Mm-cy,x0Mm-cx)
    endAngle = math.atan2(y1Mm-cy,x1Mm-cx)
    if clockwise:
        while endAngle>=startAngle:   #force a strictly decreasing sweep
            endAngle-=2.0*math.pi
    else:
        while endAngle<=startAngle:   #force a strictly increasing sweep
            endAngle+=2.0*math.pi
    angleSpan = endAngle-startAngle
    nSegments = max(4,int(abs(angleSpan)*radius/0.4))   #~0.4mm per chord
    for k in range(1,nSegments+1):
        if TriggeredEndstop() is not None:
            return
        a = startAngle+angleSpan*(k/nSegments)
        _GcodeLinearMove(cx+radius*math.cos(a),cy+radius*math.sin(a),feedMmMin)

#----------------------------------------------------------------------
#RunGcodeFile: read a file and draw it
#goes line by line, handling moves, arcs and the pen commands, and
#stopping immediately if a limit switch trips
#----------------------------------------------------------------------
def RunGcodeFile(path):
    if not Calibration["done"]:
        raise NotCalibratedError("Run calibration ('calibrate') before running a G-code file.")
    try:
        with open(path,'r') as f:
            lines = f.readlines()
    except OSError as e:
        print(f"Couldn't open '{path}': {e}")
        return

    absMode = True
    feedMmMin = 900.0
    curXMm = CurrentXSteps/STEPS_PER_MM
    curYMm = CurrentYSteps/STEPS_PER_MM
    print(f"Running G-code: {path} ({len(lines)} lines)")

    for lineno,raw in enumerate(lines,1):
        hit = TriggeredEndstop()
        if hit is not None:
            _TriggerEmergencyStop(hit)
            print(f"G-code job ABORTED at line {lineno}.")
            return
        if estopped:
            print(f"G-code job ABORTED at line {lineno} -- already emergency-stopped.")
            return

        parsed = ParseGcodeLine(raw)
        if parsed is None:
            continue
        cmd,p = parsed

        if cmd in ("G0","G1"):
            if "F" in p:
                feedMmMin = p["F"]
            targetX = p.get("X",curXMm) if absMode else curXMm+p.get("X",0.0)
            targetY = p.get("Y",curYMm) if absMode else curYMm+p.get("Y",0.0)
            _GcodeLinearMove(targetX,targetY,feedMmMin)
            curXMm,curYMm = targetX,targetY

        elif cmd in ("G2","G3"):
            if "F" in p:
                feedMmMin = p["F"]
            targetX = p.get("X",curXMm) if absMode else curXMm+p.get("X",0.0)
            targetY = p.get("Y",curYMm) if absMode else curYMm+p.get("Y",0.0)
            _GcodeArcMove(curXMm,curYMm,targetX,targetY,
                            p.get("I",0.0),p.get("J",0.0),cmd=="G2",feedMmMin)
            curXMm,curYMm = targetX,targetY

        elif cmd=="G4":
            deadline = time.time()+p.get("P",0.0)
            while time.time()<deadline:
                if TriggeredEndstop() is not None:
                    break
                time.sleep(0.005)

        elif cmd=="G90":
            absMode = True
        elif cmd=="G91":
            absMode = False
        elif cmd=="G20":
            print(f"[Line {lineno}] G20 (inches) -- values used as-is, NOT converted.")
        elif cmd=="G21":
            pass
        elif cmd=="M3":
            if not SetPen(False):
                print(f"[Line {lineno}] Pen-down did not confirm -- continuing anyway.")
        elif cmd=="M5":
            if not SetPen(True):
                print(f"[Line {lineno}] Pen-up did not confirm -- continuing anyway.")
        elif cmd in ("M2","M30"):
            print(f"[Line {lineno}] Program end ({cmd}).")
            return
        else:
            print(f"[Line {lineno}] Skipping unsupported command: {cmd}")

    print(f"G-code job '{path}' complete.")

#---WriteGcodeFile: save the test pattern as g-code---
def WriteGcodeFile(path):
    Connect()
    RunGcodeFile(path)
    return {"estopped": estopped,"tripped_switch": TrippedSwitch}

#---InputListener: the keyboard control loop for manual testing---
def InputListener():
    global CurrentMode,targetRpm,running,theta,CurrentXSteps,CurrentYSteps
    global estopped,TrippedSwitch,CircleRadiusMm,CircleRadiusSteps

    while running:
        rawCmd = sys.stdin.readline().strip()
        if not rawCmd:
            continue
        cmd = rawCmd.lower()   #a 'g' filename keeps its original case below

        if cmd=="q":
            running = False
            break

        elif cmd=="calibrate":
            if estopped:
                print("Cannot calibrate while emergency-stopped -- clear with 'r' first.")
                continue
            #calibrating mode makes the main loop skip its own estop
            #reaction below -- Calibrate() drives into switches on purpose
            #and handles that itself.
            CurrentMode = "CALIBRATING"
            try:
                result = Calibrate()
                print(f"Calibrated. steps/mm: X={result['steps_per_mm_x']:.3f} "
                      f"Y={result['steps_per_mm_y']:.3f}  usable area: "
                      f"{result['usable_width_mm']:.1f} x {result['usable_height_mm']:.1f} mm")
            except RuntimeError as e:
                print(f"Calibration failed: {e}")
            finally:
                if not estopped:
                    CurrentMode,targetRpm = "RPM",0.0

        elif cmd=="r":
            hit = TriggeredEndstop()
            if hit is not None:
                print(f"Cannot clear -- end-stop {hit} is still triggered.")
            elif estopped:
                estopped,TrippedSwitch = False, None
                CurrentMode,targetRpm = "RPM",0.0
                print("Emergency stop cleared. Mode: Stopped.")
            else:
                print("Not stopped.")

        elif cmd=="u":
            print("Pen up..." if SetPen(True) else "Pen up FAILED.")
        elif cmd=="d":
            print("Pen down..." if SetPen(False) else "Pen down FAILED.")

        elif cmd=="c" or cmd.startswith("c "):
            if estopped:
                print("Cannot change mode while emergency-stopped -- clear with 'r' first.")
                continue
            parts = cmd.split()
            if len(parts)>1:
                try:
                    newR = float(parts[1])
                    if newR<=0:
                        print("Radius must be positive -- keeping previous radius.")
                    else:
                        CircleRadiusMm = newR
                except ValueError:
                    print(f"Couldn't parse '{parts[1]}' as a radius -- keeping previous radius.")
            CircleRadiusSteps = CircleRadiusMm*STEPS_PER_MM
            CurrentMode = "CIRCLE"
            theta,CurrentXSteps,CurrentYSteps = 0.0,0,0
            print(f"Mode: Circle (radius {CircleRadiusMm} mm)")

        elif cmd=="g" or cmd.startswith("g "):
            if estopped:
                print("Cannot start a G-code job while emergency-stopped -- clear with 'r' first.")
                continue
            path = rawCmd[1:].strip()   #original case preserved, unlike cmd
            if not path:
                print("Usage: g <path-to-gcode-file>")
                continue
            CurrentMode = "GCODE"   #main loop idles; this thread drives the moves
            RunGcodeFile(path)
            if not estopped:
                CurrentMode,targetRpm = "RPM",0.0

        else:
            try:
                targetRpm = float(cmd)
            except ValueError:
                continue
            if estopped:
                print("Cannot change mode while emergency-stopped -- clear with 'r' first.")
                continue
            CurrentMode = "RPM"
            if abs(targetRpm)<0.1:
                targetRpm = 0.0
                print("Mode: Stopped")
            else:
                req.set_value(DIR1_PIN,AxisDirValue(INVERT_X,targetRpm>0))
                req.set_value(DIR2_PIN,AxisDirValue(INVERT_Y,targetRpm>0))
                CalculateDelay(targetRpm)
                print(f"Mode: RPM ({targetRpm})")

#---run the listener when this file is run directly---
if __name__=="__main__":
    Connect()
    threading.Thread(target=InputListener,daemon=True).start()
    print("Ready. Commands: 'calibrate' (required before 'g'), 'c' (circle), 'g <file>' (run G-code), "
          "[number] (RPM/stop), 'u'/'d' (pen), 'r' (clear estop), 'q' (quit)")

    try:
        while running:
            #endstops always win, except while calibrating, which drives
            #into them on purpose.  reacting here would also fight
            #_StepAxis on the same pins from the other thread
            if CurrentMode=="CALIBRATING":
                time.sleep(0.01)
                continue

            hit = TriggeredEndstop()
            if hit is not None:
                _TriggerEmergencyStop(hit)
                targetRpm = 0.0
                time.sleep(0.005)
                continue
            elif estopped:
                #a switch clearing on its own never auto-resumes -- only
                #an explicit 'r' does.
                time.sleep(0.005)
                continue

            if CurrentMode=="RPM":
                if targetRpm!=0.0 and stepDelayS>0:
                    req.set_value(STEP1_PIN,Value.ACTIVE)
                    req.set_value(STEP2_PIN,Value.ACTIVE)
                    time.sleep(stepDelayS)
                    req.set_value(STEP1_PIN,Value.INACTIVE)
                    req.set_value(STEP2_PIN,Value.INACTIVE)
                    time.sleep(stepDelayS)
                else:
                    time.sleep(0.01)

            elif CurrentMode=="CIRCLE":
                now = time.perf_counter()
                if now-LastCircleTime>=CircleIntervalS:
                    LastCircleTime = now
                    theta = (theta+0.008)%(2.0*math.pi)
                    targetX = round(CircleRadiusSteps*math.sin(theta))
                    targetY = round(CircleRadiusSteps*(1.0-math.cos(theta)))
                    BresenhamMove(targetX-CurrentXSteps,targetY-CurrentYSteps)

            elif CurrentMode=="GCODE":
                time.sleep(0.01)   #InputListener thread drives this directly

    finally:
        Disconnect()
        print("Exited.")
