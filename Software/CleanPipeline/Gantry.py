#odroid n2+ direct-gpio gantry driver.
#bit-bangs the x/y stepper step/dir pins over gpiod, no microcontroller in
#between. end-stops are checked in the same loop as every step pulse, so
#"stop immediately on any trigger" is a real guarantee, not best-effort.
#
#wiring: all 5 switches are active-high (triggered = 3.3v), each with its own
#~10k pull-down resistor to gnd (done in hardware, not here).
#  - x_min, x_max, y_min, y_max: travel-limit switches.
#  - pen switch: feedback only. high = pen up, low = pen down.
#  - pen motor: a digital output driving a cam/eccentric that toggles pen
#    height each time it runs (not "move to position"). SetPen() runs it
#    only until the switch confirms the target was reached.
#
#gcode supported:
#  G0/G1 X.. Y.. F..        linear move (no separate rapid mode)
#  G2/G3 X.. Y.. I.. J.. F. arc, centre at (start + I, start + J)
#  G4 P..                   dwell P seconds
#  G90/G91                  absolute/relative positioning
#  G20/G21                  inches (values used as-is, NOT converted) / mm
#  M3 / M5                  pen down / pen up
#  M2 / M30                 program end
#  F sets feed rate (mm/min), persists until changed. an end-stop trip
#  retracts off the switch (see RETRACT_MM below), aborts the file, and
#  requires 'r' to clear before anything else runs.
#
#cli commands (via stdin):
#  calibrate   home both axes, measure real steps-per-mm, park at the safe
#              start corner. required once per session before 'g'.
#  <number>    rpm test mode (0 = stop)
#  c [radius]  circle demo
#  g <path>    run a g-code file
#  u / d       pen up / down
#  r           clear a tripped emergency stop (never auto-clears, and
#              refuses if a switch is still physically triggered)
#  q           quit
import math
import re
import sys
import threading
import time

#gpiod only exists on the odroid. importing this module elsewhere (a laptop,
#the web server) must not crash -- hardware access only happens inside
#Connect() and the functions that require it.
try:
    import gpiod
    from gpiod.line import Direction,Value
    _GPIOD_IMPORT_ERROR = None
except ImportError as _e:
    gpiod = None
    Direction = Value = None
    _GPIOD_IMPORT_ERROR = _e


class GantryNotConnectedError(RuntimeError):
    #no gpiod / no gpio chip available -- not the odroid, or the device
    #couldn't be opened. callers should report this cleanly instead of a
    #raw exception.
    pass


class NotCalibratedError(RuntimeError):
    #Calibrate() hasn't succeeded yet this session. never persisted to disk:
    #a mechanical bump between sessions would silently invalidate a saved
    #value, so it must be redone every restart.
    pass

#pin assignments (odroid n2+, gpiochip0 offsets)
CHIP_PATH = "/dev/gpiochip0"

DIR1_PIN,STEP1_PIN = 64,68   #x motor -- physical pins 7, 11
DIR2_PIN,STEP2_PIN = 81,69   #y motor -- physical pins 12, 13

X_MIN_PIN,X_MAX_PIN = 72,65  #physical pins 15, 16
Y_MIN_PIN,Y_MAX_PIN = 66,67  #physical pins 18, 22
PEN_SWITCH_PIN = 70          #physical pin 33 (input, feedback)
PEN_MOTOR_PIN = 71           #physical pin 35 (output)

ENDSTOP_PINS = {"X_MIN": X_MIN_PIN,"X_MAX": X_MAX_PIN,
                "Y_MIN": Y_MIN_PIN,"Y_MAX": Y_MAX_PIN}

#true/false per axis: flip if that axis moves the wrong physical way for a
#commanded positive/negative step delta. only ever matters for drawing
#direction -- Calibrate() below never needs these to be correct.
INVERT_X = True
INVERT_Y = True

#kinematics
MICROSTEPS = 16
STEPS_PER_REV = 200*MICROSTEPS
STEPS_PER_MM = 80.0   #only used before Calibrate() has run
CIRCLE_STEP_DELAY_S = 0.000002
DEFAULT_CIRCLE_RADIUS_MM = 8.0

#calibration: switch-to-switch travel, fully pressed to fully pressed (measured)
X_SWITCH_TRAVEL_MM = 204.0
Y_SWITCH_TRAVEL_MM = 262.0
EDGE_TOLERANCE_MM = 5.0        #kept clear of both switches on both axes
HOMING_STEP_DELAY_S = 0.0008   #per-step pulse time while homing (both axes)
HOMING_MAX_STEPS = 200_000     #a switch that never triggers is a fault
ENDSTOP_DEBOUNCE_S = 0.006     #gap between confirmation samples below
ENDSTOP_CONFIRM_SAMPLES = 3    #a pin must read active this many times in a
                               #row (spaced ENDSTOP_DEBOUNCE_S apart) before
                               #it's trusted -- a single re-check missed
                               #electrically-noisy motors (no flyback diode,
                               #shared ground return) that can hold a switch
                               #input high for several ms at a time, well
                               #past a one-shot debounce window
RETRACT_MM = 5.0               #0.5cm pulled back off a switch on emergency
                               #stop, so it isn't left latched high -- without
                               #this, clearing an estop with 'r' still leaves
                               #the gantry unable to move (the switch reads
                               #triggered again the instant it's cleared)

Calibration = {
    "done": False,
    "steps_per_mm_x": STEPS_PER_MM,
    "steps_per_mm_y": STEPS_PER_MM,
    "usable_width_mm": X_SWITCH_TRAVEL_MM-2*EDGE_TOLERANCE_MM,
    "usable_height_mm": Y_SWITCH_TRAVEL_MM-2*EDGE_TOLERANCE_MM,
}

#runtime state
req = None                #live gpiod line request once Connect() succeeds
CurrentMode = "RPM"        #RPM | CIRCLE | GCODE | CALIBRATING
targetRpm = 0.0            #must start at 0: nonzero here would move the
stepDelayS = 0.0           #gantry the instant the main loop starts running
running = True

CurrentXSteps = 0
CurrentYSteps = 0
theta = 0.0
LastCircleTime = time.perf_counter()
CircleIntervalS = 0.0015
CircleRadiusMm = DEFAULT_CIRCLE_RADIUS_MM
CircleRadiusSteps = CircleRadiusMm*STEPS_PER_MM

estopped = False
TrippedSwitch = None


#============================================================================
#connection
#============================================================================
#opens the gpio chip and claims every pin. safe to call repeatedly.
#raises GantryNotConnectedError for any failure reason (missing gpiod,
#chip busy, etc.) so callers only need to catch one type.
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


#releases the gpio lines (motors/pen off first). safe if never connected.
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


def IsConnected():
    return req is not None


#============================================================================
#low-level motion primitives
#============================================================================
def CalculateDelay(rpm):
    global stepDelayS
    if abs(rpm)<0.1:
        stepDelayS = 0
    else:
        stepDelayS = (60.0/(abs(rpm)*STEPS_PER_REV))/2.0


#name of the first end-stop currently high that isn't in `ignore`, or none.
#`ignore` skips those pins during the scan (not just filters the result), so
#a switch legitimately still triggered elsewhere (e.g. the other axis
#resting at its own limit) can never hide a genuinely different switch
#further down the scan.
#
#debounced: a pin must read active on ENDSTOP_CONFIRM_SAMPLES consecutive
#checks, ENDSTOP_DEBOUNCE_S apart, before it's trusted. a single re-check
#isn't enough against motor electrical noise, which can hold a pin high for
#several ms -- longer than one debounce gap. adds no delay to the common
#"nothing triggered" case, since the sampling only starts once a pin is
#seen high at all.
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


#every currently-triggered switch name, unlike TriggeredEndstop() which
#stops at the first one. more than one can be genuinely active at once --
#e.g. sitting in a corner, or right after homing, where both axes are
#still resting against the switch they just found.
def _AllTriggeredEndstops():
    return [name for name in ENDSTOP_PINS if TriggeredEndstop(ignore=set(ENDSTOP_PINS)-{name})==name]


#the Value to send an axis's dir pin to move positive (true) or negative
#(false), respecting that axis's invert flag. single source of truth:
#every direction decision in this file (drawing, homing, the rpm test
#command) goes through this, so they can never disagree with each other
#about which pin value means which physical direction.
def AxisDirValue(invert,positive):
    forward = positive!=invert   #xor
    return Value.ACTIVE if forward else Value.INACTIVE

#which way is "away" from each switch, in the same sign convention as
#CurrentXSteps/CurrentYSteps (positive = away from min, toward max).
_RETREAT_SIGN = {"X_MIN":+1,"X_MAX":-1,"Y_MIN":+1,"Y_MAX":-1}


#steps the tripped axis RETRACT_MM back off `hit`, updating the tracked
#step count as it goes. `ignore`: other switches that are expected to
#still read triggered right now (e.g. the other axis, mid-recovery from a
#corner) and must not be mistaken for a new fault. stops early (without
#raising) if some other, unexpected switch trips during the retract --
#that's a real mechanical problem, not something to push through.
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


#latches the emergency-stop state and immediately retracts off every switch
#currently triggered (not just `hit` -- e.g. a corner hit, or a g-code job
#aborting right after calibration, can leave both axes resting against a
#switch at once), so nothing stays physically latched and forces a manual
#power-cycle just to move again. runs at most once per trip -- safe to call
#from both the main loop and a g-code job, whichever notices the trigger
#first.
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


#steps both axes by (dx, dy) steps using Bresenham interpolation, so a
#diagonal move traces a straight line instead of one axis finishing early.
#stops early (without finishing) if an end-stop trips mid-move -- the
#caller's own loop handles the estop state.
#
#bresenham outline: pick the axis with the larger step count (ax/ay) as the
#"major" axis and step it every iteration; the minor axis only gets a step
#whenever its accumulated error crosses zero. `err` tracks (minor progress
#so far) - (major progress so far), scaled by the major count, so the minor
#axis ends up stepped in proportion to dx:dy without ever using floats.
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


#============================================================================
#pen
#============================================================================
PEN_UP_OVERRUN_S = 0.04   #extra run time after the switch first trips, so
                          #the pen mechanism finishes seating instead of
                          #stopping the instant the switch first makes contact


#high ("in") = up, low ("out") = down.
def PenIsUp():
    return req.get_value(PEN_SWITCH_PIN)==Value.ACTIVE


#runs the pen motor only until the switch confirms the target state, or
#timeoutS elapses (a real fault, reported not swallowed). refuses if an
#end-stop is already tripped.
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


#============================================================================
#calibration -- direction-agnostic: never needs to know in advance which way
#is "toward min". drives one way until either switch on an axis triggers,
#reverses to find the other one, and measures the steps between.
#============================================================================
#steps one axis at dirValue until stopWhen(triggered_name) is true. any
#other triggered switch not in `ignore` is a real fault (crossed wiring)
#and raises immediately. returns (name, steps).
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


#finds both switches on one axis without assuming which direction leads to
#which. OtherAxisNames: the other axis's switches, which may still read
#triggered if it was just calibrated (its own normal resting position, not
#a fault here).
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


#homes both axes, derives real steps-per-mm from the measured travel and
#the known switch-to-switch distance, then parks at the safe starting
#corner (EDGE_TOLERANCE_MM in from both switches). refuses if an end-stop
#is already tripped.
#
#each axis is retracted RETRACT_MM off the switch it ends homing on,
#immediately after that axis finishes -- otherwise it's left resting
#switch-down, which would trip the main loop's emergency stop the instant
#calibration mode ends (before the final park move even runs).
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


#============================================================================
#g-code
#============================================================================
GCODE_MIN_FEED_MM_S = 0.1   #floor so a stray f0 doesn't divide by zero
_GCODE_WORD_RE = re.compile(r'([A-Za-z])\s*(-?[0-9]*\.?[0-9]+)')


#one line -> (command, {letter: value}) or none for blank/comment lines.
#assumes one g/m word per line; every other letter is a param.
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


#clamps a target into the calibrated safe rectangle -- a second line of
#defence on top of the physical switches, so a bad/uploaded g-code
#coordinate gets pulled back in bounds instead of ever reaching one.
def _ClampToUsableArea(xMm,yMm):
    w,h = Calibration["usable_width_mm"],Calibration["usable_height_mm"]
    cx,cy = min(max(xMm,0.0),w),min(max(yMm,0.0),h)
    if (cx,cy)!=(xMm,yMm):
        print(f"[gcode] clamped out-of-bounds target ({xMm:.1f}, {yMm:.1f}) -> ({cx:.1f}, {cy:.1f}) mm")
    return cx,cy


#moves to an absolute mm position using the calibrated steps-per-mm per
#axis. step delta is computed from the current actual step count (not an
#accumulated float), so repeated small moves can't drift.
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


#g2/g3 arc, centre at (start + i, start + j). no native arc mode, so this
#interpolates into short linear segments (~0.4mm chords).
#
#arc outline: the centre is `start + (i, j)` per the gcode spec, so radius
#is just the length of that (i, j) offset. start/end angle come from
#atan2 of each endpoint relative to the centre; a clockwise (g2) arc must
#sweep with a strictly decreasing angle and a counter-clockwise (g3) arc
#with a strictly increasing one, so whichever one came out going the wrong
#way gets a full turn added/subtracted until it's consistent with
#start_angle. the arc is then chopped into enough segments (chord length
#~0.4mm) that each one is a straight line move, and each waypoint is sent
#through the same linear-move path as a regular G1.
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


#runs a g-code file to completion or until an end-stop aborts it. refuses
#to run at all until Calibrate() has succeeded this session -- checked
#here so it applies from both the web server and the cli.
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


#Server.py's entry point: connects if needed, runs the file, leaves the
#connection open afterward. raises GantryNotConnectedError or
#NotCalibratedError -- callers should catch both for a clean error instead
#of a 500.
def WriteGcodeFile(path):
    Connect()
    RunGcodeFile(path)
    return {"estopped": estopped,"tripped_switch": TrippedSwitch}


#============================================================================
#interactive cli
#============================================================================
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


if __name__=="__main__":
    Connect()
    threading.Thread(target=InputListener,daemon=True).start()
    print("Ready. Commands: 'calibrate' (required before 'g'), 'c' (circle), 'g <file>' (run G-code), "
          "[number] (RPM/stop), 'u'/'d' (pen), 'r' (clear estop), 'q' (quit)")

    try:
        while running:
            #end-stops always win -- except during calibrating, where
            #Calibrate() drives into switches on purpose and handles that
            #itself. reacting here too would force STEP1_PIN/STEP2_PIN off
            #from this thread while _StepAxis() (the other thread) is
            #mid-pulse on those same pins.
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
