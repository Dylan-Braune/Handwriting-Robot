#u23539195 D.R. Braune.  drawing the writing
#two independent halves
#  top     a pen path in millimetres to a g-code file, pure arithmetic
#  bottom  driving the gantry, steppers, endstops, homing and the pen lift
#nothing here knows about handwriting, it takes strokes and draws them
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#every physical number about the machine, steps per millimetre is derived
#from the gearing rather than typed in
@dataclass
class GantryConfig:
    #motion transmission
    fullStepsPerRev: int = 200   #NEMA 17
    microstepping: int = 16   #A4988 MS1/MS2/MS3 set to 1/16
    mmPerRevX: float = 40.0   #GT2 2 mm pitch, 20 teeth
    mmPerRevY: float = 40.0
    #work area, zero means use the switch travel
    boundsMinXmm: float = 0.0
    boundsMinYmm: float = 0.0
    boundsMaxXmm: float = 0.0
    boundsMaxYmm: float = 0.0
    originXmm: float = 0.0
    originYmm: float = 0.0   #zero means the top of the calibrated area
    #page margins, in millimetres
    pageMarginXmm: float = 15.0
    pageMarginYmm: float = 10.0
    #the ruling on the paper, and how far down the page to start
    lineHeightMm: float = 8.4   #the spacing of the printed ruling on the paper
    lineOffsetMm: float = 0.0
    #whether to leave the lines at the pitch the synthesiser laid them out
    #at, or move each one onto a ruled line.  a page that will be
    #photographed and read back wants the laid-out pitch: on the ruling the
    #ascenders and descenders of neighbouring lines meet, and ink that
    #touches is cut as one line.  the respacing below still runs either
    #way, so the synthesiser's baseline wander is taken out from between
    #the lines and left inside them
    ignoreRuling: bool = True   # true keeps the pitch the synthesiser laid the lines out at
    #the x-height as a share of the ruling, and the cap on how tall one
    #line of writing may come out as a multiple of the ruling
    rulingXhFrac: float = 0.46   #x-height as a share of the ruling
    rulingMaxExtent: float = 1.25   # and one line may reach 1.25 rulings tall
    #speeds
    drawFeedMmMin: float = 900.0   #15 mm a second with the pen down
    travelFeedMmMin: float = 2400.0   #and 40 mm a second with it up
    minSegmentMm: float = 0.12   #shorter moves are collapsed
    #pen actuator: a one-way gear motor turning 90 degrees per pulse
    penUpDwellMs: float = 350.0
    penDownDwellMs: float = 350.0   #longer, to let the nib seat before drawing
    penUpCode: str = 'M5'
    penDownCode: str = 'M3'
    penStartsUp: bool = True
    #if the next stroke starts this close, the pen draws into it rather
    #than lifting, travelling a fraction of a millimetre and dropping
    joinStrokesMm: float = 0.6   #a hop under this is drawn as ink
    #a stroke whose own ends are this close is closed, so a ring meets itself
    closeRingMm: float = 0.6   #and a stroke with ends this close is closed into a ring

    #the work area promised here is trimmed to what the carriage can reach
    #between the switches
    def __post_init__(self):
        usableW = X_SWITCH_TRAVEL_MM-2*EDGE_TOLERANCE_MM   # the switch travel less a margin at each end
        usableH = Y_SWITCH_TRAVEL_MM-2*EDGE_TOLERANCE_MM
        self.boundsMaxXmm = (usableW if self.boundsMaxXmm <= 0   #a requested area is held inside that reach
                             else min(self.boundsMaxXmm, usableW))
        self.boundsMaxYmm = (usableH if self.boundsMaxYmm <= 0
                             else min(self.boundsMaxYmm, usableH))
        if self.originYmm <= 0:
            self.originYmm = self.boundsMaxYmm   #the origin sits at the top of the calibrated area

    #what is left for writing once both margins are kept
    @property
    def usableWidthMm(self):
        return self.boundsMaxXmm-self.originXmm-2*self.pageMarginXmm   # the margin comes off both the left and the right

    @property
    def microstepsPerRev(self):
        return self.fullStepsPerRev*self.microstepping   # 200 full steps at 1/16 gives 3200 microsteps a revolution

    @property
    def stepsPerMmX(self):
        return self.microstepsPerRev/self.mmPerRevX   # microsteps a revolution over mm a revolution gives steps per mm

    @property
    def stepsPerMmY(self):
        return self.microstepsPerRev/self.mmPerRevY

#----------------------------------------------------------------------
#XhForRuling: the x-height to lay text out at so it fits the ruling, chosen before
#layout because the line breaks depend on it, one plain layout is drawn
#and the x-height brought down if the tallest line breaks the cap
#----------------------------------------------------------------------
def XhForRuling(profile,cfg,text=None,default=4.0,probe=True):
    ruling = float(getattr(cfg,'lineHeightMm',8.4))
    frac = float(profile.get('xhOverPitch',0.0) or 0.0)   #the ratio measured from this writer own pages
    if not 0.15<=frac<=0.90:   #outside that range the configured fraction is used
        frac = float(getattr(cfg,'rulingXhFrac',0.46))
    xh = ruling*frac   # the ruling times that fraction is the x-height to lay out at
    if not probe or not text:
        return xh
    try:
        import synthesise as SYN
        t = SYN.SynthesizeText(text,profile,mmPerXh=xh,seed=0,
                               lineWidthMm=getattr(cfg,'usableWidthMm',180.0))
        got = _TallestLine(t)   #the tallest line the probe layout produced
        cap = ruling*float(getattr(cfg,'rulingMaxExtent',1.60))   # the cap the ruling allows
        if got>cap>0:
            xh *= cap/got  #the x-height scaled down in proportion to the overshoot
    except Exception:
        pass   #a probe that will not run must not stop a job being written
    return xh

#the percentiles the vertical reach of a line is taken at
_EXTENT_HI_PCT = 98.0   # the reach is read at the 98th and 2nd percentiles
_EXTENT_LO_PCT = 2.0

#----------------------------------------------------------------------
#_TallestLine: how much vertical room one line of this text needs, the strokes are
#grouped onto their text line by the pitch they were laid out at, and the
#reach is taken at those percentiles rather than at the outright extremes
#----------------------------------------------------------------------
def _TallestLine(traj):
    pitch = float(traj.meta.get('lineStepMm',0.0) or 0.0)
    rows = {}
    for s in traj.strokes:
        if not s:
            continue
        ys = [p[1] for p in s]
        if pitch>1e-6:
            k = round(-(sum(ys)/len(ys))/pitch)   # the mean height over the pitch says which text line a stroke is on
            base = -k*pitch
        else:
            k,base = 0,0.0
        rows.setdefault(k,[]).append((max(ys)-base,min(ys)-base))  # each stroke top and bottom measured from that line own baseline
    if not rows:
        return 0.0
    worst = 0.0
    for v in rows.values():
        tops = np.array([t for t,_b in v],dtype=np.float64)
        bots = np.array([b for _t,b in v],dtype=np.float64)
        if tops.size<8:   #too few strokes for a percentile to mean anything
            worst = max(worst,float(tops.max()-bots.min()))
        else:
            worst = max(worst,float(np.percentile(tops,_EXTENT_HI_PCT)   #the reach between those two percentiles
                                   -np.percentile(bots,_EXTENT_LO_PCT)))
    return worst

#----------------------------------------------------------------------
#ToMachine: places the drawing on the bed at the right size, shrinking it only if it
#would run off the work area, and sets each line's own centre one
#laid-out pitch below the one above
#----------------------------------------------------------------------
def ToMachine(traj,cfg,fitToBounds=True,optimizeOrder=True,rowTolMm=None):
    x0,y0,x1,y1 = traj.Bounds()
    w,h = max(1e-6,x1-x0),max(1e-6,y1-y0)   #the width and height of the whole drawing
    scale = 1.0
    mx,my = cfg.pageMarginXmm,cfg.pageMarginYmm
    off = max(0.0,float(getattr(cfg,'lineOffsetMm',0.0)))
    if fitToBounds:
        #the margin comes off twice, once each side
        availW = cfg.boundsMaxXmm-cfg.originXmm-2*mx   #the width left once both margins are kept
        availH = (cfg.originYmm-my-off)-cfg.boundsMinYmm   # and the height from the start line down to the bottom
        scale = min(1.0,availW/w,availH/h)   #the drawing shrinks only when it runs over
    #each line is placed by its own centre, not by the nominal pitch
    pitch = float(traj.meta.get('lineStepMm',0.0) or 0.0)
    ruledPitch = (pitch*scale if getattr(cfg,'ignoreRuling',True)   # the laid out pitch scaled or the paper ruling
                  else float(getattr(cfg,'lineHeightMm',8.4)))
    respace = pitch>1e-6 and int(traj.meta.get('lines',1) or 1)>1   #respacing only matters with more than one line
    centres = {}
    if respace:
        rows = {}
        for s in traj.strokes:
            if not s:
                continue
            k = round(-(sum(p[1] for p in s)/len(s))/pitch)   #which text line this stroke belongs to
            ys = [p[1] for p in s]
            lo,hi = rows.get(k,(1e18,-1e18))
            rows[k] = (min(lo,min(ys)),max(hi,max(ys)))    # the lowest and highest ink on that line
        if rows:
            k0 = min(rows)   #the topmost line stays where it is
            c0 = (rows[k0][0]+rows[k0][1])*0.5   #the vertical centre of the top line
            for k,(lo,hi) in rows.items():
                ck = (lo+hi)*0.5
                centres[k] = (c0-ck)*scale-(k-k0)*ruledPitch   #every other line set one pitch below the one above
    out = []
    for s in traj.strokes:
        shift = 0.0
        if respace and s:
            k = round(-(sum(p[1] for p in s)/len(s))/pitch)
            shift = centres.get(k,0.0)
        #both frames have y up, so the text's top row lands at originY
        q = [(cfg.originXmm+mx+(x-x0)*scale,    #x measured from the left margin
              cfg.originYmm-my-off-(y1-y)*scale+shift)   #and y down from the origin by the depth into the drawing
             for (x,y) in s]
        out.append(_Decimate(q,cfg.minSegmentMm))
    if optimizeOrder:
        out = OptimizeStrokeOrder(out,rowTolMm=rowTolMm)
    return out,scale

#---_Decimate: merges moves shorter than the machine can resolve---
def _Decimate(poly,minSeg):
    if len(poly)<2:
        return poly
    out = [poly[0]]
    for p in poly[1:-1]:
        if math.dist(p,out[-1])>=minSeg:   # points closer together than the machine resolves are dropped
            out.append(p)
    out.append(poly[-1])
    return out

#----------------------------------------------------------------------
#OptimizeStrokeOrder: orders the strokes in bands down the page and, inside a band, takes the
#nearest unvisited stroke next, which cuts the pen-up travel
#----------------------------------------------------------------------
def OptimizeStrokeOrder(strokes,rowTolMm=None):
    if len(strokes)<3:
        return strokes
    ys = [sum(p[1] for p in s)/len(s) for s in strokes]   # the mean height of each stroke
    if rowTolMm is None:
        heights = [max(p[1] for p in s)-min(p[1] for p in s)
                   for s in strokes]
        rowTolMm = max(1.0,1.2*float(np.median(heights)))  # the band tolerance is 1.2 median stroke heights
    order = sorted(range(len(strokes)),key=lambda i: ys[i])   #strokes sorted down the page
    rows,cur = [],[order[0]]
    for i in order[1:]:
        if abs(ys[i]-ys[cur[-1]])<=rowTolMm:   # strokes within that tolerance share a band
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
                k = min(remaining,key=lambda i: strokes[i][0][0])    #the first stroke of a band is the leftmost
            else:
                k = min(remaining,   #after that the nearest unvisited stroke end
                        key=lambda i: min(math.dist(pen,strokes[i][0]),
                                          math.dist(pen,strokes[i][-1])))
            remaining.remove(k)
            s = strokes[k]
            #entering from whichever end is closer costs a plotter nothing
            if pen is not None and \
                    math.dist(pen,s[-1])<math.dist(pen,s[0]):  #a stroke is reversed when its far end is the nearer one
                s = s[::-1]
            out.append(s)
            pen = s[-1]
    return out

#---_Clamp: holds every point inside the work area, and counts the ones moved---
def _Clamp(strokes,cfg):
    lo = (cfg.boundsMinXmm,cfg.boundsMinYmm)
    hi = (cfg.boundsMaxXmm,cfg.boundsMaxYmm)
    clipped = 0
    out = []
    for s in strokes:
        q = []
        for (x,y) in s:
            cx = min(max(x,lo[0]),hi[0])   #each coordinate held inside the work area
            cy = min(max(y,lo[1]),hi[1])
            if (cx,cy)!=(x,y):
                clipped+=1
            q.append((cx,cy))
        out.append(q)
    return out,clipped

#---NewPenState: starts tracking whether the pen is up, and how many pulses it has taken---
def NewPenState(cfg):
    return [cfg.penStartsUp,0]   #the tracked pen state and the pulse count

#----------------------------------------------------------------------
#SetPenLines: the lines needed to get the pen to a wanted state, the actuator turns 90
#degrees per pulse and cannot be told an absolute position, so a toggle is
#emitted only when the tracked state has to change
#----------------------------------------------------------------------
def SetPenLines(state,cfg,wantUp):
    if wantUp==state[0]:   #already in that state so no pulse is sent
        return []
    state[0] = wantUp
    state[1]+= 1   #one more 90 degree pulse counted
    code = cfg.penUpCode if wantUp else cfg.penDownCode
    dwell = (cfg.penUpDwellMs if wantUp else cfg.penDownDwellMs)/1000.0   # milliseconds turned into the seconds G4 wants
    out = [f"{code}   ; pen {'UP' if wantUp else 'DOWN'} "
           f"(one 90deg pulse of the one-way gear motor)"]
    if dwell>0:
        out.append(f"G4 P{dwell:.3f}   ; wait for the 90deg rotation to finish")
    return out

#----------------------------------------------------------------------
#WriteGcode: a pen path in, a G-code file out, a pen-up move to the start of each
#stroke, a pen down, the stroke at drawing feed, and a pen up at the end
#----------------------------------------------------------------------
def WriteGcode(traj,cfg,path,title='',rowTolMm=None):
    strokes,scale = ToMachine(traj,cfg,rowTolMm=rowTolMm)  # the path placed scaled and ordered on the bed
    strokes,clipped = _Clamp(strokes,cfg)   #and held inside the work area
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
    here = None        #where the pen is, in millimetres
    joined = rings = 0
    joinedMm = 0.0
    drawMm = travelMm = 0.0
    for s in strokes:
        if len(s)<2:
            continue
        if (cfg.closeRingMm>0 and len(s)>3
                and math.dist(s[0],s[-1])<=cfg.closeRingMm   #the two ends close enough to join
                and s[0]!=s[-1]):
            s = list(s)+[s[0]]   #the first point repeated at the end closes the ring
            rings += 1
        if here is not None and math.dist(here,s[0])<=cfg.joinStrokesMm:   #the next stroke starts near enough to draw into
            #already there, so draw into this stroke, the gap is drawn as
            #ink, so it is counted
            L.append(f"G1 F{cfg.drawFeedMmMin:.0f}")    # the feed drops to drawing speed with the pen still down
            joined += 1
            joinedMm += math.dist(here,s[0])
            drawMm += math.dist(here,s[0])
        else:
            L+=SetPenLines(pen,cfg,True)
            #f is modal and shared between G0 and G1, so the travel feed is
            #restated here rather than relying on the header
            L.append(f"G0 F{cfg.travelFeedMmMin:.0f} "   # a pen up move to the start of the stroke
                     f"X{s[0][0]:.3f} Y{s[0][1]:.3f}")
            if here is not None:
                travelMm += math.dist(here,s[0])
            L+=SetPenLines(pen,cfg,False)
            L.append(f"G1 F{cfg.drawFeedMmMin:.0f}")
        prev = s[0]
        for (x,y) in s[1:]:
            L.append(f"G1 X{x:.3f} Y{y:.3f}")    # one straight move a point with the pen down
            drawMm += math.dist(prev,(x,y))
            prev = (x,y)
        here = s[-1]
    L+=SetPenLines(pen,cfg,True)
    L.append(f"G0 F{cfg.travelFeedMmMin:.0f} "   #and the carriage parks back at the origin
             f"X{cfg.originXmm:.3f} Y{cfg.originYmm:.3f}   ; park")
    if here is not None:
        travelMm += math.dist(here,(cfg.originXmm,cfg.originYmm))
    L.append('M2')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        f.write('\n'.join(L)+'\n')
    #where the writing landed, in machine millimetres with y up
    ys = [p[1] for s in strokes for p in s] or [0.0]
    xs = [p[0] for s in strokes for p in s] or [0.0]
    lineH = max(1e-6,float(getattr(cfg,'lineHeightMm',8.4)))    # the ruling pitch in millimetres
    off = float(getattr(cfg,'lineOffsetMm',0.0))
    firstLineY = cfg.originYmm-cfg.pageMarginYmm
    #a time budget split three ways, it ignores acceleration, so it is a
    #floor on the real time rather than a prediction of it
    drawS = 60.0*drawMm/max(1.0,cfg.drawFeedMmMin)    #drawing distance over the draw feed in seconds
    travelS = 60.0*travelMm/max(1.0,cfg.travelFeedMmMin)   #and the same for the travel moves
    #pen[1] counts toggles but not direction, so the mean dwell is used
    dwellS = pen[1]*(cfg.penUpDwellMs+cfg.penDownDwellMs)/2000.0   # pulses times the mean of the two dwells
    return dict(lines=len(L),strokes=len(strokes),penPulses=pen[1],
                scale=scale,clipped=clipped,path=str(path),
                joinedStrokes=joined,closedRings=rings,
                joinedInkMm=round(joinedMm,2),
                drawMm=round(drawMm,1),travelMm=round(travelMm,1),
                drawSeconds=round(drawS,1),travelSeconds=round(travelS,1),
                dwellSeconds=round(dwellS,1),
                estimatedSeconds=round(drawS+travelS+dwellS,1),   # the three added together
                placement=dict(topYmm=round(max(ys),2),bottomYmm=round(min(ys),2),
                               leftXmm=round(min(xs),2),rightXmm=round(max(xs),2),
                               lineOffsetMm=round(off,2),lineHeightMm=round(lineH,3),
                               ignoredRuling=bool(getattr(cfg,'ignoreRuling',True)),
                               lineIndex=int(round(off/lineH)),    #which ruled line the job starts on
                               #the pitch the lines were really drawn at
                               drawnPitchMm=round(
                                   float(traj.meta.get('lineStepMm',0.0) or 0.0)*scale,2),
                               #ruled lines this job occupies, so the page
                               #advances by that and not by one
                               textLines=max(1,int(traj.meta.get('lines',1) or 1)),
                               firstLineYmm=round(firstLineY,2),
                               linesOnPage=int((firstLineY-cfg.boundsMinYmm)//lineH)+1,   #how many ruled lines fit between the start and the bottom
                               bedTopYmm=round(cfg.originYmm,2),
                               bedBottomYmm=round(cfg.boundsMinYmm,2),
                               bedLeftXmm=round(cfg.originXmm,2),
                               bedRightXmm=round(cfg.boundsMaxXmm,2)))

#the gpio line interface, off the board this import fails and the names are
#bound to None, which Connect reports as a missing gantry
try:
    import gpiod
    from gpiod.line import Direction,Value
    _GPIOD_IMPORT_ERROR = None
except Exception as _e:
    gpiod = None
    Direction = None
    Value = None
    _GPIOD_IMPORT_ERROR = _e

#---GantryNotConnectedError: raised when something needs hardware that is not present---
class GantryNotConnectedError(RuntimeError):
    pass

#---NotCalibratedError: raised when a move is asked for before homing has succeeded---
class NotCalibratedError(RuntimeError):
    pass

CHIP_PATH = "/dev/gpiochip0"

DIR1_PIN,STEP1_PIN = 64,68   #x motor, physical pins 7 and 11
DIR2_PIN,STEP2_PIN = 81,69   #y motor, physical pins 12 and 13
X_MIN_PIN,X_MAX_PIN = 72,65  #physical pins 15 and 16
Y_MIN_PIN,Y_MAX_PIN = 66,67  #physical pins 18 and 22
PEN_SWITCH_PIN = 70          #physical pin 33, input
PEN_MOTOR_PIN = 71           #physical pin 35, output

ENDSTOP_PINS = {"X_MIN": X_MIN_PIN,"X_MAX": X_MAX_PIN,
                "Y_MIN": Y_MIN_PIN,"Y_MAX": Y_MAX_PIN}

#whether each axis counts up the other way
INVERT_X = True   # the wiring makes a high direction line count downwards
INVERT_Y = True

MICROSTEPS = 16
STEPS_PER_REV = 200*MICROSTEPS   # 3200 microsteps a revolution

#used only before Calibrate has run
STEPS_PER_MM = 80.0   #3200 microsteps over 40 mm a revolution

#the shortest pause between step pulses
MIN_STEP_DELAY_S = 0.000002   #two microseconds between pulse edges

#how far the carriage moves between the two switches on each axis, which
#is what Calibrate divides its step count by to get steps per millimetre
X_SWITCH_TRAVEL_MM = 200.6   #the measured distance between the two x switches
Y_SWITCH_TRAVEL_MM = 261.3

#margin kept off the limit switches on both axes
EDGE_TOLERANCE_MM = 5.0

#homing is run slowly, with a step budget that fails a dead switch
HOMING_STEP_DELAY_S = 0.0008
HOMING_MAX_STEPS = 200_000   #a step budget that ends a homing move with a dead switch

#a switch must read triggered this many times in a row to be believed
ENDSTOP_DEBOUNCE_S = 0.006   #six milliseconds between confirming samples
ENDSTOP_CONFIRM_SAMPLES = 3

#how far to back off a switch once it trips
RETRACT_MM = 5.0

#extra run time after the pen switch first trips
PEN_UP_OVERRUN_S = 0.04

#the slowest feed the runner accepts, so a stray F0 cannot divide by zero
GCODE_MIN_FEED_MM_S = 0.1

#splits a G-code word into its letter and number
_GCODE_WORD_RE = re.compile(r'([A-Za-z])\s*(-?[0-9]*\.?[0-9]+)')   #one letter then a signed number

#which way to move to come off each switch
_RETREAT_SIGN = {"X_MIN":+1,"X_MAX":-1,"Y_MIN":+1,"Y_MAX":-1}   #a min switch retreats positive and a max switch negative

#the work area found by homing, held for this session only
Calibration = {
    "done": False,
    "steps_per_mm_x": STEPS_PER_MM,
    "steps_per_mm_y": STEPS_PER_MM,
    "usable_width_mm": X_SWITCH_TRAVEL_MM-2*EDGE_TOLERANCE_MM,    # the travel less the margin at both ends
    "usable_height_mm": Y_SWITCH_TRAVEL_MM-2*EDGE_TOLERANCE_MM,
}

#the live gpio line request, once Connect has succeeded
req = None

#where each axis is, counted in steps from home
CurrentXSteps = 0
CurrentYSteps = 0

#set once an emergency stop has fired, and which switch caused it
estopped = False
TrippedSwitch = None

#set when a stop is asked for from outside the stepping loops, which run
#on another thread and cannot be interrupted directly
abortEvent = threading.Event()

#set while a G-code run owns the motors, so a stop knows when it is safe
#to drive the parking move
_jobActive = threading.Event()

#whether the last G-code run was stopped early
lastJobAborted = False

#---Connect: claims the gpio pins and sets them to a safe state---
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

#---Disconnect: releases the pins again---
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

#----------------------------------------------------------------------
#TriggeredEndstop: the name of a limit switch reading triggered, confirmed over several
#samples because a long wire picks up noise
#----------------------------------------------------------------------
def TriggeredEndstop(ignore=frozenset()):
    for name,pin in ENDSTOP_PINS.items():
        if name in ignore:
            continue
        if req.get_value(pin)!=Value.ACTIVE:   #a line reading inactive settles it at once
            continue
        for _ in range(ENDSTOP_CONFIRM_SAMPLES-1):   #two more samples spaced by the debounce time
            time.sleep(ENDSTOP_DEBOUNCE_S)
            if req.get_value(pin)!=Value.ACTIVE:
                break
        else:   #every sample agreed so the switch is really triggered
            return name
    return None

#---_AllTriggeredEndstops: every switch currently reading triggered---
def _AllTriggeredEndstops():
    return [name for name in ENDSTOP_PINS if TriggeredEndstop(ignore=set(ENDSTOP_PINS)-{name})==name]   #each switch tested on its own with the others ignored

#---AxisDirValue: the direction pin level for a wanted direction---
def AxisDirValue(invert,positive):
    forward = positive!=invert    #the wanted direction exclusive or the axis inversion
    return Value.ACTIVE if forward else Value.INACTIVE

#---_RetractOffSwitch: steps an axis off a switch that is pressed, before anything else is allowed---
def _RetractOffSwitch(hit,ignore=frozenset()):
    global CurrentXSteps,CurrentYSteps
    isX = hit.startswith("X")
    sign = _RETREAT_SIGN[hit]
    dirPin,stepPin = (DIR1_PIN,STEP1_PIN) if isX else (DIR2_PIN,STEP2_PIN)
    invert = INVERT_X if isX else INVERT_Y
    stepsPerMm = Calibration["steps_per_mm_x"] if isX else Calibration["steps_per_mm_y"]
    TargetSteps = max(1,round(RETRACT_MM*stepsPerMm))   #five millimetres of retraction turned into steps

    req.set_value(dirPin,AxisDirValue(invert,sign>0))
    moved = 0
    for _ in range(TargetSteps):
        if TriggeredEndstop(ignore={hit}|set(ignore)) is not None:
            break   #something else tripped, so stop pushing
        req.set_value(stepPin,Value.ACTIVE)
        time.sleep(HOMING_STEP_DELAY_S)
        req.set_value(stepPin,Value.INACTIVE)
        time.sleep(HOMING_STEP_DELAY_S)
        moved+=sign  #the step counter follows the direction taken

    if isX:
        CurrentXSteps+=moved
    else:
        CurrentYSteps+=moved

#----------------------------------------------------------------------
#_TriggerEmergencyStop: cuts the drive, retracts off every tripped switch and marks the machine
#stopped until it is cleared
#----------------------------------------------------------------------
def _TriggerEmergencyStop(hit):
    global estopped,TrippedSwitch
    if estopped:
        return
    estopped,TrippedSwitch = True,hit
    req.set_value(STEP1_PIN,Value.INACTIVE)   #both step lines dropped low at once
    req.set_value(STEP2_PIN,Value.INACTIVE)
    allHits = _AllTriggeredEndstops() or [hit]
    print(f"\n!!! EMERGENCY STOP -- end-stop(s) {', '.join(allHits)} triggered !!! "
          f"Retracting {RETRACT_MM}mm off each...")
    for h in allHits:
        _RetractOffSwitch(h,ignore=set(allHits)-{h})
    print("Retracted. Clear with 'r' once safe to resume.")

#---RequestAbort: asks a running job to stop---
def RequestAbort():
    abortEvent.set()

#----------------------------------------------------------------------
#EmergencyStop: stops the running job, lifts the pen and parks at the corner, in that
#order, the stop latches only if the drawing thread fails to let go
#----------------------------------------------------------------------
def EmergencyStop(waitS=3.0,park=True):
    Connect()
    abortEvent.set()   # the running job is asked to stop first
    deadline = time.time()+waitS
    while _jobActive.is_set() and time.time()<deadline:   #waiting up to three seconds for the drawing thread to let go
        time.sleep(0.005)
    stillRunning = _jobActive.is_set()

    req.set_value(STEP1_PIN,Value.INACTIVE)
    req.set_value(STEP2_PIN,Value.INACTIVE)

    #forced, so the pen lifts even in a fault
    penUp = SetPen(True,force=True)
    parked,detail = False,""
    if stillRunning:
        detail = ("the drawing thread did not let go within %.1f s, so the "
                  "stop is still latched and the corner move was skipped"%waitS)
    else:
        abortEvent.clear()   #cleared before the park move
        if not park:
            detail = "parking not asked for"
        elif not Calibration["done"]:
            detail = ("not calibrated, so there is no known corner to drive "
                      "to -- the pen is up and the motors are idle")
        elif TriggeredEndstop() is not None:
            detail = ("end-stop %s is tripped, so the corner move was skipped "
                      "-- clear it, then re-home"%TriggeredEndstop())
        else:
            cfg = GantryConfig()
            _GcodeLinearMove(cfg.originXmm,cfg.originYmm,cfg.travelFeedMmMin)
            parked = True
    print("[stop] pen %s, %s%s"%("up" if penUp else "UP FAILED",
                                 "parked at the corner" if parked else "not parked",
                                 " -- "+detail if detail else ""))
    return {"stopped": True,"pen_up": bool(penUp),"parked": parked,
            "still_running": stillRunning,"latched": stillRunning,
            "detail": detail,"estopped": estopped,
            "tripped_switch": TrippedSwitch}

#----------------------------------------------------------------------
#BresenhamMove: a straight line in whole steps, a running error term steps whichever axis
#is furthest behind, so the two motors stay on the line
#----------------------------------------------------------------------
def BresenhamMove(dx,dy,stepDelayS=MIN_STEP_DELAY_S):
    global CurrentXSteps,CurrentYSteps
    ax,ay = abs(dx),abs(dy)   #how many steps each axis has to take
    sx = 1 if dx>=0 else -1   # and which way each one turns
    sy = 1 if dy>=0 else -1
    req.set_value(DIR1_PIN,AxisDirValue(INVERT_X,sx>0))
    req.set_value(DIR2_PIN,AxisDirValue(INVERT_Y,sy>0))

    def Pulse(stepPin):
        req.set_value(stepPin,Value.ACTIVE)
        time.sleep(stepDelayS)
        req.set_value(stepPin,Value.INACTIVE)

    if ax>=ay:   #x is the major axis
        err = ax//2   #the error term starts half a major step in
        for _ in range(ax):
            #checked per step, not per move, so a stop takes effect at once
            if abortEvent.is_set() or TriggeredEndstop() is not None:
                return
            Pulse(STEP1_PIN)
            CurrentXSteps+=sx
            err-=ay   #each major step takes the minor count off the error
            if err<0:   #the minor axis catches up by one step
                Pulse(STEP2_PIN)
                CurrentYSteps+=sy
                err+=ax   #and the error gains a full major count back
    else:   #y is the major axis
        err = ay//2    # the same with the roles of the two axes swapped
        for _ in range(ay):
            if abortEvent.is_set() or TriggeredEndstop() is not None:
                return
            Pulse(STEP2_PIN)
            CurrentYSteps+=sy
            err-=ax
            if err<0:
                Pulse(STEP1_PIN)
                CurrentXSteps+=sx
                err+=ay

#---PenIsUp: reads the pen switch---
def PenIsUp():
    return req.get_value(PEN_SWITCH_PIN)==Value.ACTIVE   #the switch reads active while the pen is lifted

#----------------------------------------------------------------------
#SetPen: pulses the one-way motor and watches the switch, giving up if the pen
#never reports the state asked for
#----------------------------------------------------------------------
def SetPen(targetUp,timeoutS=3.0,force=False):
    #a tripped switch blocks the pen too, unless force is passed
    hit = TriggeredEndstop()
    if hit is not None and not force:
        print(f"Refusing pen move -- end-stop {hit} still tripped.")
        return False
    if PenIsUp()==targetUp:   #the pen already sits where it is wanted
        return True

    req.set_value(PEN_MOTOR_PIN,Value.ACTIVE)   #the one way gear motor turns while this line is high
    deadline = time.time()+timeoutS
    reached = False
    while time.time()<deadline:   # polling the switch until it confirms or the time runs out
        if not force and TriggeredEndstop() is not None:
            break
        if PenIsUp()==targetUp:
            reached = True
            break
        time.sleep(0.002)
    if reached and targetUp:
        time.sleep(PEN_UP_OVERRUN_S)   # a little extra run so the lift completes past the switch
    req.set_value(PEN_MOTOR_PIN,Value.INACTIVE)

    if not reached:
        print("Pen move did NOT confirm via switch -- check motor/wiring.")
    return reached

#---_StepAxis: steps one axis until a stopping condition is met---
def _StepAxis(stepPin,dirPin,dirValue,stopWhen,ignore=frozenset()):
    req.set_value(dirPin,dirValue)
    steps = 0
    while True:
        hit = TriggeredEndstop(ignore=ignore)
        if hit is not None:
            if stopWhen(hit):
                return hit,steps
            raise RuntimeError(f"Unexpected end-stop {hit} triggered -- check wiring.")
        if steps>=HOMING_MAX_STEPS:   #the budget runs out with no switch reached
            raise RuntimeError("No expected end-stop triggered during homing -- check wiring.")
        req.set_value(stepPin,Value.ACTIVE)
        time.sleep(HOMING_STEP_DELAY_S)
        req.set_value(stepPin,Value.INACTIVE)
        time.sleep(HOMING_STEP_DELAY_S)
        steps+=1

#---_CalibrateAxis: drives one axis to both ends and counts the steps between them---
def _CalibrateAxis(stepPin,dirPin,minName,maxName,OtherAxisNames=frozenset()):
    print(f"[calibrate] driving until either {minName} or {maxName} is activated...")
    firstHit,_ = _StepAxis(stepPin,dirPin,Value.ACTIVE,   # driving one way until either switch of this axis trips
                               stopWhen=lambda h: h in (minName,maxName),
                               ignore=OtherAxisNames)
    other = maxName if firstHit==minName else minName
    print(f"[calibrate] {firstHit} activated. Reversing toward {other}...")
    #the first switch is ignored too: it may still read triggered for the
    #first few steps of the reverse move
    secondHit,TravelSteps = _StepAxis(stepPin,dirPin,Value.INACTIVE,   #then back the other way counting the steps between them
                                           stopWhen=lambda h: h==other,
                                           ignore={firstHit}|set(OtherAxisNames))
    print(f"[calibrate] {secondHit} activated. {TravelSteps} steps measured.")
    return firstHit,secondHit,TravelSteps

#----------------------------------------------------------------------
#Calibrate: homes both axes, sets steps per millimetre from the travel it counts, and
#stores the usable area, no absolute move is accepted before this runs
#----------------------------------------------------------------------
def Calibrate():
    global CurrentXSteps,CurrentYSteps

    if TriggeredEndstop() is not None:
        raise RuntimeError(f"Cannot calibrate -- end-stop {TrippedSwitch} already tripped.")

    print("[calibrate] === X axis ===")
    _,xSecond,xTravel = _CalibrateAxis(STEP1_PIN,DIR1_PIN,"X_MIN","X_MAX")
    CurrentXSteps = xTravel if xSecond=="X_MAX" else 0   #the position counter set from which end the axis finished at
    Calibration["steps_per_mm_x"] = xTravel/X_SWITCH_TRAVEL_MM   #steps counted over the measured travel gives steps per mm
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

#---ParseGcodeLine: splits one G-code line into its command and parameters---
def ParseGcodeLine(raw):
    line = raw.split(';',1)[0]    #anything after a semicolon is a comment
    line = re.sub(r'\([^)]*\)','',line).strip()   #and anything in round brackets too
    if not line:
        return None
    cmd,params = None,{}
    for letter,num in _GCODE_WORD_RE.findall(line):
        letter = letter.upper()
        val = float(num)
        if cmd is None and letter in ('G','M'):   #the first G or M word is the command
            cmd = f"{letter}{int(val)}"
        else:
            params[letter] = val   #and every other letter is a parameter
    return (cmd,params) if cmd else None

#---_ClampToUsableArea: holds a target inside the homed area---
def _ClampToUsableArea(xMm,yMm):
    w,h = Calibration["usable_width_mm"],Calibration["usable_height_mm"]
    cx,cy = min(max(xMm,0.0),w),min(max(yMm,0.0),h)   #the target held inside the homed area
    if (cx,cy)!=(xMm,yMm):
        print(f"[gcode] clamped out-of-bounds target ({xMm:.1f}, {yMm:.1f}) -> ({cx:.1f}, {cy:.1f}) mm")
    return cx,cy

#---_GcodeLinearMove: carries out a straight move at a feed rate---
def _GcodeLinearMove(targetXMm,targetYMm,feedMmMin):
    targetXMm,targetYMm = _ClampToUsableArea(targetXMm,targetYMm)
    spmmX,spmmY = Calibration["steps_per_mm_x"],Calibration["steps_per_mm_y"]
    #zero in this frame is EDGE_TOLERANCE_MM in from the min switch
    x0Mm = CurrentXSteps/spmmX-EDGE_TOLERANCE_MM   #the step counter turned back into millimetres
    y0Mm = CurrentYSteps/spmmY-EDGE_TOLERANCE_MM
    distMm = math.hypot(targetXMm-x0Mm,targetYMm-y0Mm)   #the straight line distance of the move
    TargetXSteps = round((targetXMm+EDGE_TOLERANCE_MM)*spmmX)    # the target in steps from the min switch
    TargetYSteps = round((targetYMm+EDGE_TOLERANCE_MM)*spmmY)
    dxSteps = TargetXSteps-CurrentXSteps   #how many steps each axis still has to take
    dySteps = TargetYSteps-CurrentYSteps
    if dxSteps==0 and dySteps==0:
        return
    majorSteps = max(abs(dxSteps),abs(dySteps),1)   #the axis with the most steps sets the timing
    feedMmS = max(feedMmMin/60.0,GCODE_MIN_FEED_MM_S)   #feed per minute turned into feed per second
    stepDelayS = max((distMm/feedMmS)/majorSteps,MIN_STEP_DELAY_S)   # move time spread over those steps gives the delay between pulses
    BresenhamMove(dxSteps,dySteps,stepDelayS=stepDelayS)

#----------------------------------------------------------------------
#_GcodeArcMove: splits an arc into short straight moves, the machine only
#goes straight
#----------------------------------------------------------------------
def _GcodeArcMove(x0Mm,y0Mm,x1Mm,y1Mm,iMm,jMm,clockwise,feedMmMin):
    cx,cy = x0Mm+iMm,y0Mm+jMm   # i and j are the centre offset from the start point
    radius = math.hypot(iMm,jMm)    #and their length is the radius
    if radius<1e-6:
        _GcodeLinearMove(x1Mm,y1Mm,feedMmMin)
        return
    startAngle = math.atan2(y0Mm-cy,x0Mm-cx)   #the angle of the start point about that centre
    endAngle = math.atan2(y1Mm-cy,x1Mm-cx)
    if clockwise:
        while endAngle>=startAngle:   #force a strictly decreasing sweep
            endAngle-=2.0*math.pi
    else:
        while endAngle<=startAngle:
            endAngle+=2.0*math.pi
    angleSpan = endAngle-startAngle
    nSegments = max(4,int(abs(angleSpan)*radius/0.4))   #about 0.4 mm per chord
    for k in range(1,nSegments+1):
        if abortEvent.is_set() or TriggeredEndstop() is not None:
            return
        a = startAngle+angleSpan*(k/nSegments)   #the angle at the end of this chord
        _GcodeLinearMove(cx+radius*math.cos(a),cy+radius*math.sin(a),feedMmMin)

#----------------------------------------------------------------------
#RunGcodeFile: reads a G-code file and draws it, holding _jobActive while the motors are
#being driven
#----------------------------------------------------------------------
def RunGcodeFile(path):
    global lastJobAborted
    lastJobAborted = False
    if not Calibration["done"]:
        raise NotCalibratedError("Run calibration before running a G-code file.")
    try:
        with open(path,'r') as f:
            lines = f.readlines()
    except OSError as e:
        print(f"Couldn't open '{path}': {e}")
        return

    absMode = True
    feedMmMin = 900.0
    curXMm = CurrentXSteps/STEPS_PER_MM   #the starting position in millimetres
    curYMm = CurrentYSteps/STEPS_PER_MM
    print(f"Running G-code: {path} ({len(lines)} lines)")

    _jobActive.set()
    try:
        _RunGcodeLines(path,lines,absMode,feedMmMin,curXMm,curYMm)
    finally:
        _jobActive.clear()

#---_RunGcodeLines: the body of that loop, split out so _jobActive clears however it ends---
def _RunGcodeLines(path,lines,absMode,feedMmMin,curXMm,curYMm):
    global lastJobAborted
    for lineno,raw in enumerate(lines,1):
        if abortEvent.is_set():   #a stop is checked for before every line
            lastJobAborted = True
            print(f"G-code job STOPPED at line {lineno} -- stop requested.")
            return
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
                feedMmMin = p["F"]   #the feed word is modal so it carries forward
            targetX = p.get("X",curXMm) if absMode else curXMm+p.get("X",0.0)   #absolute or added onto the current position
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
            deadline = time.time()+p.get("P",0.0)   #G4 waits the number of seconds in P
            while time.time()<deadline:
                if abortEvent.is_set() or TriggeredEndstop() is not None:
                    break
                time.sleep(0.005)

        elif cmd=="G90":
            absMode = True
        elif cmd=="G91":
            absMode = False
        elif cmd=="G21":
            pass
        elif cmd=="M3":
            if not SetPen(False):   #M3 puts the pen down
                print(f"[Line {lineno}] Pen-down did not confirm -- continuing anyway.")
        elif cmd=="M5":
            if not SetPen(True):   #and M5 lifts it
                print(f"[Line {lineno}] Pen-up did not confirm -- continuing anyway.")
        elif cmd in ("M2","M30"):
            print(f"[Line {lineno}] Program end ({cmd}).")
            return
        else:
            print(f"[Line {lineno}] Skipping unsupported command: {cmd}")

    print(f"G-code job '{path}' complete.")
