"""
WriteGCode.py -- pen trajectory -> G-code, step/direction schedule, and a
plotter-preview raster, for the Handwriting-Robot gantry.

=== HARDWARE CONFIG -- EDIT THESE FOR YOUR MACHINE ===================
Everything machine-specific lives in the GantryConfig dataclass below, in
one block, so calibration is a one-line change:

  * X/Y: NEMA 17 (200 full steps/rev) on A4988 drivers at 1/16 microstep
    => MICROSTEPS_PER_REV = 200 * 16 = 3200.
    Travel per revolution depends on your transmission:
      - GT2 belt (2 mm pitch) on a 20-tooth pulley  -> 40 mm/rev  (default)
      - leadscrew                                    -> set mmPerRev = lead
    STEPS_PER_MM is then MICROSTEPS_PER_REV / mmPerRev  (= 80 steps/mm
    on the default belt setup). Set `mmPerRevX/Y` and nothing else.

  * PEN: a plain gear motor with NO driver that turns in ONE direction
    only. Every 90 degrees of rotation TOGGLES the pen (90 deg = up, the
    next 90 deg = down). So the pen is never commanded to a state, only
    pulsed; this module tracks the pen state in software (it cannot be
    read back or reversed) and emits exactly one 90-degree pulse whenever
    the state must change. `penPulseMs` is how long your motor needs to
    turn 90 degrees -- measure it once and set it here.

Outputs
  WriteGcode(traj, cfg, path)      -> standard G0/G1 with M3/M5 pen codes
  WriteStepSchedule(traj, cfg, p)  -> CSV of (t_us, axis, dir, nsteps) plus
                                      PEN pulse rows: what the firmware
                                      actually clocks out
  RenderGcodePreview(gcodePath)    -> rasterizes the EMITTED G-code (not the
                                      trajectory) so the drawing can be
                                      compared against the synthesized image
"""

import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT_DIR = Path(__file__).resolve().parent


# ===========================================================================
# MACHINE CONFIGURATION
# ===========================================================================
@dataclass
class GantryConfig:
    # --- motion transmission -------------------------------------------
    fullStepsPerRev: int = 200          # NEMA 17
    microstepping: int = 16             # A4988 MS1/MS2/MS3 = 1/16
    mmPerRevX: float = 40.0             # GT2 2mm pitch x 20 teeth
    mmPerRevY: float = 40.0
    # --- work area ------------------------------------------------------
    boundsMinXmm: float = 0.0
    boundsMinYmm: float = 0.0
    boundsMaxXmm: float = 200.0
    boundsMaxYmm: float = 200.0
    originXmm: float = 10.0             # where text starts on the bed
    originYmm: float = 180.0
    # --- speeds ---------------------------------------------------------
    drawFeedMmMin: float = 900.0        # pen-down feed
    travelFeedMmMin: float = 2400.0     # pen-up rapid
    accelMmS2: float = 400.0            # trapezoidal accel limit
    minSegmentMm: float = 0.12          # collapse shorter moves
    # --- pen actuator (one-way gear motor, 90 deg per toggle) -----------
    penPulseMs: float = 260.0           # time for one 90-degree rotation
    penSettleMs: float = 90.0           # dwell after a toggle
    penUpCode: str = 'M5'               # firmware maps these to the pulse
    penDownCode: str = 'M3'
    penStartsUp: bool = True            # assumed state at power-on

    @property
    def microstepsPerRev(self):
        return self.fullStepsPerRev * self.microstepping

    @property
    def stepsPerMmX(self):
        return self.microstepsPerRev / self.mmPerRevX

    @property
    def stepsPerMmY(self):
        return self.microstepsPerRev / self.mmPerRevY


# ===========================================================================
# Trajectory -> machine coordinates
# ===========================================================================
def ToMachine(traj, cfg, fitToBounds=True, optimizeOrder=True):
    """Places the trajectory on the bed at (originX, originY) with y DOWN
    the page, and (optionally) scales it down if it would leave the work
    area. Returns (strokesXY, scale)."""
    x0, y0, x1, y1 = traj.Bounds()
    w, h = max(1e-6, x1 - x0), max(1e-6, y1 - y0)
    scale = 1.0
    if fitToBounds:
        availW = cfg.boundsMaxXmm - cfg.originXmm
        availH = cfg.originYmm - cfg.boundsMinYmm
        scale = min(1.0, availW / w, availH / h)
    out = []
    for s in traj.strokes:
        # both frames have y UP (standard plotter convention): the text's
        # top row lands at originY and later rows descend from there
        q = [(cfg.originXmm + (x - x0) * scale,
              cfg.originYmm - (y1 - y) * scale)
             for (x, y) in s]
        out.append(_Decimate(q, cfg.minSegmentMm))
    if optimizeOrder:
        out = OptimizeStrokeOrder(out)
    return out, scale


def _Decimate(poly, minSeg):
    if len(poly) < 2:
        return poly
    out = [poly[0]]
    for p in poly[1:-1]:
        if math.dist(p, out[-1]) >= minSeg:
            out.append(p)
    out.append(poly[-1])
    return out


def OptimizeStrokeOrder(strokes, rowTolMm=None):
    """Greedy nearest-neighbour ordering of the pen-down strokes, so the
    pen stops criss-crossing the page between them. Only the ORDER changes
    -- the same strokes are drawn, so the result is pixel-identical (the
    simulation check verifies this) but the pen-up travel and the job time
    drop substantially.

    Strokes are grouped into text rows first and each row is ordered
    left-to-right-ish within itself, so the machine still writes the page
    in reading order rather than wandering between lines."""
    if len(strokes) < 3:
        return strokes
    ys = [sum(p[1] for p in s) / len(s) for s in strokes]
    if rowTolMm is None:
        heights = [max(p[1] for p in s) - min(p[1] for p in s)
                   for s in strokes]
        rowTolMm = max(1.0, 1.2 * float(np.median(heights)))
    order = sorted(range(len(strokes)), key=lambda i: ys[i])
    rows, cur = [], [order[0]]
    for i in order[1:]:
        if abs(ys[i] - ys[cur[-1]]) <= rowTolMm:
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
                k = min(remaining, key=lambda i: strokes[i][0][0])
            else:
                k = min(remaining,
                        key=lambda i: min(math.dist(pen, strokes[i][0]),
                                          math.dist(pen, strokes[i][-1])))
            remaining.remove(k)
            s = strokes[k]
            # entering from whichever end is closer is free for a plotter
            if pen is not None and \
                    math.dist(pen, s[-1]) < math.dist(pen, s[0]):
                s = s[::-1]
            out.append(s)
            pen = s[-1]
    return out


def _Clamp(strokes, cfg):
    lo = (cfg.boundsMinXmm, cfg.boundsMinYmm)
    hi = (cfg.boundsMaxXmm, cfg.boundsMaxYmm)
    clipped = 0
    out = []
    for s in strokes:
        q = []
        for (x, y) in s:
            cx = min(max(x, lo[0]), hi[0])
            cy = min(max(y, lo[1]), hi[1])
            if (cx, cy) != (x, y):
                clipped += 1
            q.append((cx, cy))
        out.append(q)
    return out, clipped


# ===========================================================================
# Pen state machine: one-way motor, 90 deg per toggle
# ===========================================================================
class PenController:
    """The gear motor cannot reverse and has no position feedback, so pen
    state is tracked here: each state CHANGE emits exactly one 90-degree
    pulse. Emitting a pulse when no change is needed would invert the pen
    for the rest of the job, so every transition goes through this class."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.isUp = cfg.penStartsUp
        self.pulses = 0

    def SetPen(self, wantUp):
        if wantUp == self.isUp:
            return []
        self.isUp = wantUp
        self.pulses += 1
        code = self.cfg.penUpCode if wantUp else self.cfg.penDownCode
        dwell = (self.cfg.penPulseMs + self.cfg.penSettleMs) / 1000.0
        return [f"{code}   ; pen {'UP' if wantUp else 'DOWN'} "
                f"(one 90deg pulse of the one-way gear motor)",
                f"G4 P{dwell:.3f}   ; wait for the 90deg rotation to finish"]


# ===========================================================================
# G-code
# ===========================================================================
def WriteGcode(traj, cfg, path, title=''):
    strokes, scale = ToMachine(traj, cfg)
    strokes, clipped = _Clamp(strokes, cfg)
    pen = PenController(cfg)
    L = []
    L.append(f"; Handwriting-Robot G-code{(' -- ' + title) if title else ''}")
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
    L += pen.SetPen(True)
    L.append(f"G0 F{cfg.travelFeedMmMin:.0f}")
    for s in strokes:
        if len(s) < 2:
            continue
        L += pen.SetPen(True)
        L.append(f"G0 X{s[0][0]:.3f} Y{s[0][1]:.3f}")
        L += pen.SetPen(False)
        L.append(f"G1 F{cfg.drawFeedMmMin:.0f}")
        for (x, y) in s[1:]:
            L.append(f"G1 X{x:.3f} Y{y:.3f}")
    L += pen.SetPen(True)
    L.append(f"G0 X{cfg.originXmm:.3f} Y{cfg.originYmm:.3f}   ; park")
    L.append('M2')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    return dict(lines=len(L), strokes=len(strokes), penPulses=pen.pulses,
                scale=scale, clipped=clipped, path=str(path))


# ===========================================================================
# Step / direction schedule
# ===========================================================================
def _TrapezoidTime(dist, vMax, accel):
    """Time to move `dist` mm under a trapezoidal profile from rest to rest."""
    if dist <= 0:
        return 0.0
    dAcc = vMax * vMax / (2.0 * accel)
    if dist < 2 * dAcc:                     # triangular
        vPeak = math.sqrt(dist * accel)
        return 2.0 * vPeak / accel
    return 2.0 * vMax / accel + (dist - 2 * dAcc) / vMax


def WriteStepSchedule(traj, cfg, path):
    """Bresenham-style per-segment step counts with a trapezoidal time
    budget: the format is (t_us, dxSteps, dySteps, penState) per move plus
    explicit PEN rows for the 90-degree pulses -- i.e. exactly what a
    step/dir firmware clocks out."""
    strokes, scale = ToMachine(traj, cfg)
    strokes, _ = _Clamp(strokes, cfg)
    pen = PenController(cfg)
    rows = ['t_us,type,x_steps,y_steps,dir_x,dir_y,pen_state']
    tUs = 0.0
    curX = cfg.originXmm
    curY = cfg.originYmm
    sx = cfg.stepsPerMmX
    sy = cfg.stepsPerMmY
    totalSteps = 0

    def move(x, y, drawing):
        nonlocal tUs, curX, curY, totalSteps
        dxMm, dyMm = x - curX, y - curY
        dist = math.hypot(dxMm, dyMm)
        if dist < 1e-9:
            return
        dxS = int(round(dxMm * sx))
        dyS = int(round(dyMm * sy))
        if dxS == 0 and dyS == 0:
            curX, curY = x, y
            return
        feed = (cfg.drawFeedMmMin if drawing else cfg.travelFeedMmMin) / 60.0
        dt = _TrapezoidTime(dist, feed, cfg.accelMmS2)
        tUs += dt * 1e6
        rows.append(f"{tUs:.0f},{'DRAW' if drawing else 'TRAVEL'},"
                    f"{abs(dxS)},{abs(dyS)},{1 if dxS >= 0 else -1},"
                    f"{1 if dyS >= 0 else -1},{'UP' if pen.isUp else 'DOWN'}")
        totalSteps += abs(dxS) + abs(dyS)
        curX, curY = x, y

    def penTo(up):
        nonlocal tUs
        if pen.SetPen(up):
            tUs += (cfg.penPulseMs + cfg.penSettleMs) * 1000.0
            rows.append(f"{tUs:.0f},PEN_PULSE_90DEG,0,0,1,1,"
                        f"{'UP' if up else 'DOWN'}")

    penTo(True)
    for s in strokes:
        if len(s) < 2:
            continue
        penTo(True)
        move(s[0][0], s[0][1], False)
        penTo(False)
        for (x, y) in s[1:]:
            move(x, y, True)
    penTo(True)
    move(cfg.originXmm, cfg.originYmm, False)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows) + '\n')
    return dict(rows=len(rows) - 1, seconds=tUs / 1e6, steps=totalSteps,
                penPulses=pen.pulses, path=str(path))


# ===========================================================================
# Preview: rasterize the EMITTED G-code (round-trip check)
# ===========================================================================
def ParseGcode(path):
    """Returns pen-down polylines in machine mm, by replaying the file."""
    strokes = []
    cur = []
    penDown = False
    x = y = 0.0
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.split(';')[0].strip()
            if not line:
                continue
            up = line.upper()
            code = up.split()[0]
            if code in ('M5',):
                if penDown and len(cur) >= 2:
                    strokes.append(cur)
                cur, penDown = [], False
                continue
            if code in ('M3',):
                penDown = True
                cur = [(x, y)]
                continue
            if code in ('G0', 'G1'):
                nx = re.search(r'X(-?\d+\.?\d*)', up)
                ny = re.search(r'Y(-?\d+\.?\d*)', up)
                if nx:
                    x = float(nx.group(1))
                if ny:
                    y = float(ny.group(1))
                if nx or ny:
                    if penDown:
                        cur.append((x, y))
    if penDown and len(cur) >= 2:
        strokes.append(cur)
    return strokes


def RenderGcodePreview(gcodePath, cfg, pxPerMm=6.0, strokePx=2,
                       showTravel=True, path=None):
    strokes = ParseGcode(gcodePath)
    W = int((cfg.boundsMaxXmm - cfg.boundsMinXmm) * pxPerMm)
    H = int((cfg.boundsMaxYmm - cfg.boundsMinYmm) * pxPerMm)
    img = Image.new('RGB', (max(8, W), max(8, H)), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W - 1, H - 1], outline=(210, 210, 210))

    def P(pt):
        # machine Y is up, image Y is down
        return ((pt[0] - cfg.boundsMinXmm) * pxPerMm,
                (cfg.boundsMaxYmm - pt[1]) * pxPerMm)

    if showTravel:
        for a, b in zip(strokes, strokes[1:]):
            d.line([P(a[-1]), P(b[0])], fill=(235, 170, 170), width=1)
    for s in strokes:
        if len(s) >= 2:
            d.line([P(p) for p in s], fill=(0, 0, 0), width=strokePx,
                   joint='curve')
    if path:
        img.save(path)
    return img, strokes


def RasterizeStrokes(strokes, pxPerMm=8.0, strokePx=2, bounds=None):
    """Binary raster of a stroke set (used by the simulation check)."""
    if bounds is None:
        pts = [p for s in strokes for p in s]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bounds = (min(xs), min(ys), max(xs), max(ys))
    x0, y0, x1, y1 = bounds
    W = max(8, int((x1 - x0) * pxPerMm) + 4)
    H = max(8, int((y1 - y0) * pxPerMm) + 4)
    img = Image.new('L', (W, H), 255)
    d = ImageDraw.Draw(img)
    for s in strokes:
        pts = [((x - x0) * pxPerMm + 2, (y1 - y) * pxPerMm + 2) for (x, y) in s]
        if len(pts) >= 2:
            d.line(pts, fill=0, width=strokePx, joint='curve')
    return np.array(img) < 128, bounds


def SimulateAndCompare(traj, cfg, gcodePath, pxPerMm=8.0, strokePx=2):
    """Rasterizes the synthesized trajectory and the G-code that was
    actually emitted, and reports their agreement (IoU + mean nearest-
    neighbour deviation in mm). This is the digital->machine loss."""
    machineStrokes, _ = ToMachine(traj, cfg)
    machineStrokes, _ = _Clamp(machineStrokes, cfg)
    gStrokes = ParseGcode(gcodePath)
    if not gStrokes or not machineStrokes:
        return dict(iou=0.0, meanDevMm=float('inf'), maxDevMm=float('inf'))
    allPts = [p for s in (machineStrokes + gStrokes) for p in s]
    xs = [p[0] for p in allPts]
    ys = [p[1] for p in allPts]
    bounds = (min(xs), min(ys), max(xs), max(ys))
    a, _ = RasterizeStrokes(machineStrokes, pxPerMm, strokePx, bounds)
    b, _ = RasterizeStrokes(gStrokes, pxPerMm, strokePx, bounds)
    inter = float((a & b).sum())
    union = float((a | b).sum())
    iou = inter / max(1.0, union)

    # geometric deviation: nearest-point distance from every emitted G-code
    # vertex to the synthesized polyline vertices
    src = np.array([p for s in machineStrokes for p in s])
    dst = np.array([p for s in gStrokes for p in s])
    if len(src) > 6000:
        src = src[np.linspace(0, len(src) - 1, 6000).astype(int)]
    if len(dst) > 4000:
        dst = dst[np.linspace(0, len(dst) - 1, 4000).astype(int)]
    devs = []
    for i in range(0, len(dst), 256):
        chunk = dst[i:i + 256]
        dd = np.sqrt(((chunk[:, None, :] - src[None, :, :]) ** 2).sum(-1))
        devs.append(dd.min(axis=1))
    devs = np.concatenate(devs) if devs else np.array([0.0])
    return dict(iou=round(iou, 4), meanDevMm=round(float(devs.mean()), 4),
                maxDevMm=round(float(devs.max()), 4),
                nStrokesSynth=len(machineStrokes), nStrokesGcode=len(gStrokes))
