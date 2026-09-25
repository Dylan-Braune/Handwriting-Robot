"""
odroid_direct_drive.py -- ODROID N2+ direct-GPIO gantry test driver.

This runs entirely on the SBC: it bit-bangs the X/Y stepper STEP/DIR pins
itself via gpiod, no microcontroller in between. That means the emergency
stop check below sits in the exact same loop as every single step pulse --
there is no serial link or second board to add latency, so "always check
end-stops before every step" is a real, same-process guarantee, not just a
best-effort one.

WIRING (all switches active-HIGH: "in"/triggered = 3.3V, "out"/released = 0V):
    Each switch input needs its own pull-down resistor (~10k to GND) --
    they connect the pin to 3.3V when triggered and leave it floating
    otherwise, so without a pull-down the pin has no defined state when
    released. Done in hardware here, not in software.
    4x END-STOP switches (X_MIN, X_MAX, Y_MIN, Y_MAX) at the gantry's four
       travel extremes. ANY of these going HIGH is an immediate stop --
       this is checked at the top of every loop iteration, i.e. before
       every step pulse in both RPM and CIRCLE mode, and always wins over
       whatever mode/command is active.
    1x PEN switch -- tells you the pen-lift mechanism's ACTUAL position:
       HIGH ("in") = pen UP, LOW ("out") = pen DOWN. This is feedback,
       not a safety interlock.
    1x pen-lift MOTOR output -- plain digital output (3.3V/0V) driving
       the pen motor's control line through your own transistor/driver.
       Per your mechanism description (a cam/eccentric that flips pen
       height ~90 degrees each time the motor runs), this is a TOGGLE,
       not a "drive to position" motor: turning it on always changes
       whichever state you're currently in. set_pen() below runs it only
       until the switch confirms the target state was reached, instead
       of trusting a fixed timer -- this is exactly the fix
       motion_executor.ino's own comments called out as future work
       ("add a limit switch... and have this loop hold the motor on
       until the switch trips instead of trusting a fixed delay").

A NOTE ON POLARITY: you specified HIGH = triggered/switch-is-in for all
5 switches, so that's what this implements. Worth knowing for later: the
conventional way to wire a SAFETY-CRITICAL limit switch is the opposite
("normally closed", so triggered = LOW, and a snapped/disconnected wire
ALSO reads as "triggered" = fails safe). With active-HIGH switches, a
wire falling off a limit switch reads as "not triggered" and the estop
silently stops protecting that axis. Not changing this since it's your
explicit spec, just flagging it as something to weigh.

GCODE
    `g <path>` runs a G-code file, one line at a time, standard-fashion:
    each move is executed and completed before the next line is read (no
    lookahead/blending). Supports the subset this project's own
    WriteGCode.py emits, plus enough of the standard dialect to run other
    simple files:
        G0 / G1  X.. Y.. F..    linear move (G0/G1 both just move here --
                                 there's no separate "rapid" positioning
                                 mode on this hardware, F sets feed rate
                                 in mm/min and persists until changed)
        G2 / G3  X.. Y.. I.. J.. F..   clockwise / counter-clockwise arc,
                                 centre at (current + I, current + J) --
                                 the standard arc format. Interpolated into
                                 small linear segments (this hardware has
                                 no native arc mode, same as every other
                                 move in this file).
        G4 P..                  dwell for P seconds (end-stops still
                                 checked throughout, same as everywhere)
        G90 / G91               absolute / relative positioning
        G20 / G21               inches / millimetres
        M3                      pen DOWN (matches WriteGCode.py's
                                 penDownCode)
        M5                      pen UP (matches WriteGCode.py's penUpCode)
        M2 / M30                program end
    Feed rate (F, mm/min) sets how fast each move actually runs, converted
    to a per-step delay the same way motion_planner.py does (distance /
    feed = time, divided across however many steps that move needs) --
    not a fixed demo speed like RPM/CIRCLE mode use. An end-stop trip
    aborts the remaining file immediately (checked before every line and
    inside every move, same priority as everywhere else in this script)
    and requires 'r' to clear before anything else will run, same as always.

Commands (via stdin, same as before):
    calibrate  home both axes, measure real steps-per-mm, move to the
               safe start corner (see CALIBRATION section). Must be run
               once per session before 'g' will run a file.
    <number>   RPM mode at that RPM (0 = stop)
    c          circle demo mode
    g <path>   run a G-code file (see GCODE above)
    u          pen up
    d          pen down
    r          clear a tripped emergency stop (only works if every
               end-stop currently reads clear -- never auto-clears)
    q          quit
"""
import math
import re
import sys
import threading
import time

# gpiod (and the actual /dev/gpiochip0 device) only exist on the Odroid --
# importing this module on your laptop for testing/server-side use must NOT
# crash just because the hardware library isn't installed there. Every
# actual hardware access is deferred to connect()/GantryController below,
# so pure functions (parse_gcode_line, etc.) stay usable without a gantry.
try:
    import gpiod
    from gpiod.line import Direction, Value
    _GPIOD_IMPORT_ERROR = None
except ImportError as _e:
    gpiod = None
    Direction = Value = None
    _GPIOD_IMPORT_ERROR = _e


class GantryNotConnectedError(RuntimeError):
    """Raised by connect() when the gpiod library or the GPIO chip itself
    isn't available -- i.e. this isn't running on the Odroid, or the
    Odroid's GPIO device is inaccessible. Callers (server.py) should catch
    this specifically and report a clean "gantry not connected" message
    instead of a stack trace."""
    pass


# Hardware Pin Assignments for ODroid N2+ (gpiochip0)
CHIP_PATH = "/dev/gpiochip0"

# Fixes the "writes backwards" (mirror-image) symptom: the X axis's
# physical wiring direction is inverted relative to WriteGCode.py's
# coordinate convention (X increases left-to-right on the page), so every
# commanded +X move was stepping the carriage the wrong way. Flip this
# back to False if this over-corrects (i.e. it now writes backwards the
# OTHER way) -- that would mean the actual issue was elsewhere and this
# made it worse, not better.
INVERT_X = True
INVERT_Y = True
DIR1_PIN = 64   # Physical Pin 7 (moved from offset 62 after rewiring)
STEP1_PIN = 68  # Physical Pin 11
DIR2_PIN = 81   # Physical Pin 12
STEP2_PIN = 69  # Physical Pin 13

# New: 4 end-stops + pen switch (inputs) + pen motor (output). Picked from
# gpiochip0 offsets `gpioinfo` reports as free, named 40-pin-header lines
# (PIN_xx) -- not already used above, and avoiding the conventional I2C
# (PIN_3/PIN_5), UART (PIN_8/PIN_10) and SPI (PIN_19/21/23/24/26) header
# pins in case you want those free later. Change any of these if they
# clash with how you've already wired something else.
X_MIN_PIN = 72   # Physical Pin 15
X_MAX_PIN = 65   # Physical Pin 16
Y_MIN_PIN = 66   # Physical Pin 18
Y_MAX_PIN = 67   # Physical Pin 22
PEN_SWITCH_PIN = 70   # Physical Pin 33 (input, pen up/down feedback)
PEN_MOTOR_PIN = 71    # Physical Pin 35 (output, drives the pen motor)

ENDSTOP_PINS = {"X_MIN": X_MIN_PIN, "X_MAX": X_MAX_PIN,
                "Y_MIN": Y_MIN_PIN, "Y_MAX": Y_MAX_PIN}

# Kinematics Configuration
MICROSTEPS = 16
STEPS_PER_REV = 200 * MICROSTEPS
STEPS_PER_MM = 80.0   # assumed value ONLY until calibrate() measures the real one
DEFAULT_CIRCLE_RADIUS_MM = 8.0
CIRCLE_STEP_DELAY_S = 0.000002   # per-step pulse HIGH time, same as the old code used

# ---------------------------------------------------------------------------
# Calibration -- measured switch-to-switch travel (button FULLY pressed to
# button FULLY pressed), from the physical gantry:
#   X: 204mm fully pressed (first click ~202mm -- switch compliance, not
#      used directly, see calibrate()'s docstring)
#   Y: 262mm fully pressed (first click ~259mm)
# EDGE_TOLERANCE_MM is kept clear of BOTH switches on both axes, so the
# usable writing area is switch-to-switch minus 2x this value -- exactly
# your specified 194mm x 252mm (19.4cm x 25.2cm) working area.
X_SWITCH_TRAVEL_MM = 204.0
Y_SWITCH_TRAVEL_MM = 262.0
EDGE_TOLERANCE_MM = 5.0
HOMING_STEP_DELAY_S = 0.0012  # slower than normal drawing -- gentler contact with the switches.
                              # Lower this further only with caution: too fast risks skipped
                              # steps (throwing off the measured travel) or a harder impact
                              # on the switch itself.
HOMING_MAX_STEPS = 200_000    # safety cap: a switch that never triggers is a fault, not a reason to spin forever

# Populated by calibrate(). Nothing may write to the gantry until
# calibration["done"] is True -- see write_gcode_file()'s check below.
calibration = {
    "done": False,
    "steps_per_mm_x": STEPS_PER_MM,
    "steps_per_mm_y": STEPS_PER_MM,
    "usable_width_mm": X_SWITCH_TRAVEL_MM - 2 * EDGE_TOLERANCE_MM,
    "usable_height_mm": Y_SWITCH_TRAVEL_MM - 2 * EDGE_TOLERANCE_MM,
}

# System State -- starts STOPPED. target_rpm must default to 0.0: the
# main loop pulses the motors the instant it sees a nonzero target_rpm,
# so a nonzero default here would mean the gantry starts moving on its
# own the moment the script runs, before you've even had a chance to
# hit an end-stop or emergency-stop it.
current_mode = "RPM"
target_rpm = 0.0
step_delay_s = 0.0
running = True

theta = 0.0
current_x_steps = 0
current_y_steps = 0
last_circle_time = time.perf_counter()
circle_interval_s = 0.0015
circle_radius_mm = DEFAULT_CIRCLE_RADIUS_MM
circle_radius_steps = circle_radius_mm * STEPS_PER_MM

# End-stop / estop state
estopped = False
tripped_switch = None


def calculate_delay(rpm):
    global step_delay_s
    abs_rpm = abs(rpm)
    if abs_rpm < 0.1:
        step_delay_s = 0
        return
    step_delay_s = (60.0 / (abs_rpm * STEPS_PER_REV)) / 2.0


ENDSTOP_DEBOUNCE_S = 0.004  # real switch stays HIGH; motor-noise spikes don't


def triggered_endstop():
    """Returns the name of the first end-stop currently HIGH (triggered),
    or None. Cheap enough (a handful of gpiod reads) to call every loop
    iteration without meaningfully affecting step timing.

    Debounced: a pin must still read ACTIVE after ENDSTOP_DEBOUNCE_S before
    it's trusted. This only costs time on an actual trigger (rare) -- the
    common case of "nothing triggered" returns immediately. Added because
    switching the pen motor without a flyback diode / with no shared
    ground to the ODROID induces brief noise spikes on these floating
    input pins; fix that in hardware too, this only masks the symptom."""
    for name, pin in ENDSTOP_PINS.items():
        if req.get_value(pin) == Value.ACTIVE:
            time.sleep(ENDSTOP_DEBOUNCE_S)
            if req.get_value(pin) == Value.ACTIVE:
                return name
    return None


def _axis_dir_value(invert, positive):
    """The gpiod Value to send an axis's DIR pin to move in its
    positive (True) or negative (False) direction, respecting that
    axis's INVERT_X/INVERT_Y flag. SINGLE SOURCE OF TRUTH for this
    project's direction convention -- both bresenham_move() (normal
    drawing) and _home_axis()/calibrate() (homing) call this, so they
    can never disagree with each other about which pin value drives
    which physical direction. They used to compute this separately
    (homing had its own hardcoded Value.INACTIVE/ACTIVE guess) and that
    mismatch was a real bug: swapping INVERT_X to fix drawing direction
    did nothing for homing, and vice versa."""
    forward = positive != invert   # XOR: invert flips which value means "forward"
    return Value.ACTIVE if forward else Value.INACTIVE


def bresenham_move(dx, dy, step_delay_s=CIRCLE_STEP_DELAY_S):
    """Steps BOTH axes from the current position by (dx, dy) steps, using
    the same Bresenham interpolation motion_planner.py/motion_executor.ino
    use for straight moves -- every step actually needed gets issued (not
    just one per axis per call), which is what makes the traced path
    match the commanded shape instead of lagging into cut corners. Global
    current_x_steps/current_y_steps are updated as it goes. Returns early
    (without finishing the segment) if an end-stop trips mid-move -- the
    outer loop's own check handles latching the emergency-stop state.

    step_delay_s: per-step pulse HIGH time. Defaults to the fixed CIRCLE
    demo value; G-code moves pass a feed-rate-derived delay instead (see
    _gcode_linear_move) so a slow F word actually draws slowly rather
    than always running at this hardcoded rate."""
    global current_x_steps, current_y_steps
    ax, ay = abs(dx), abs(dy)
    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1
    req.set_value(DIR1_PIN, _axis_dir_value(INVERT_X, sx > 0))
    req.set_value(DIR2_PIN, _axis_dir_value(INVERT_Y, sy > 0))

    def pulse(step_pin):
        req.set_value(step_pin, Value.ACTIVE)
        time.sleep(step_delay_s)
        req.set_value(step_pin, Value.INACTIVE)

    if ax >= ay:
        err = ax // 2
        for _ in range(ax):
            if triggered_endstop() is not None:
                return
            pulse(STEP1_PIN)
            current_x_steps += sx
            err -= ay
            if err < 0:
                pulse(STEP2_PIN)
                current_y_steps += sy
                err += ax
    else:
        err = ay // 2
        for _ in range(ay):
            if triggered_endstop() is not None:
                return
            pulse(STEP2_PIN)
            current_y_steps += sy
            err -= ax
            if err < 0:
                pulse(STEP1_PIN)
                current_x_steps += sx
                err += ay


def pen_is_up():
    """HIGH ("in") = up, LOW ("out") = down."""
    return req.get_value(PEN_SWITCH_PIN) == Value.ACTIVE


PEN_UP_OVERRUN_S = 0.04  # extra run time after the switch first trips, to
                          # let the pen mechanism finish seating -- tune
                          # up/down based on how early your switch triggers


def set_pen(target_up, timeout_s=3.0):
    """Runs the pen motor only until the switch confirms the target
    state (or `timeout_s` elapses, which means a real fault -- motor
    stalled, switch not wired/triggering -- and is reported, not
    silently swallowed). Checks the end-stops first: pen movements don't
    override an emergency stop either."""
    if triggered_endstop() is not None:
        print(f"Refusing pen move -- end-stop {tripped_switch} still tripped.")
        return False
    if pen_is_up() == target_up:
        return True  # already there, the cam-toggle motor must not run
    req.set_value(PEN_MOTOR_PIN, Value.ACTIVE)
    deadline = time.time() + timeout_s
    reached = False
    while time.time() < deadline:
        if triggered_endstop() is not None:
            break  # estop wins even mid pen-actuation
        if pen_is_up() == target_up:
            reached = True
            break
        time.sleep(0.002)
    if reached and target_up:
        # Switch trips early on the up stroke -- keep the motor running a
        # little longer so the pen actually finishes lifting, instead of
        # cutting power the instant the switch first makes contact.
        time.sleep(PEN_UP_OVERRUN_S)
    req.set_value(PEN_MOTOR_PIN, Value.INACTIVE)
    if not reached:
        print("Pen move did NOT confirm via switch -- check motor/wiring.")
    return reached


# ---------------------------------------------------------------------------
# G-code: parse one line at a time, execute it completely, move on -- no
# lookahead/blending, matching how this simple a controller can honestly
# run a file. See the module docstring's GCODE section for the supported
# command subset.
# ---------------------------------------------------------------------------
GCODE_MIN_FEED_MM_S = 0.1     # floor so a stray F0 doesn't divide by zero
_GCODE_WORD_RE = re.compile(r'([A-Za-z])\s*(-?[0-9]*\.?[0-9]+)')


def parse_gcode_line(raw):
    """One line -> (command, {letter: value}) or None for blank/comment
    lines. Comments are ';' to end of line or anything in parentheses,
    same as every common G-code dialect. Assumes one G/M word per line
    (what WriteGCode.py and virtually every simple gcode generator emit);
    the first G or M word found becomes the command, every other letter
    is a parameter."""
    line = raw.split(';', 1)[0]
    line = re.sub(r'\([^)]*\)', '', line).strip()
    if not line:
        return None
    cmd, params = None, {}
    for letter, num in _GCODE_WORD_RE.findall(line):
        letter = letter.upper()
        val = float(num)
        if cmd is None and letter in ('G', 'M'):
            cmd = f"{letter}{int(val)}"
        else:
            params[letter] = val
    return (cmd, params) if cmd else None


def _clamp_to_usable_area(x_mm, y_mm):
    """Keeps a commanded point inside the calibrated safe working
    rectangle (0..usable_width_mm, 0..usable_height_mm in the drawing's
    own coordinate frame, i.e. already inset from both switches by
    EDGE_TOLERANCE_MM). This is a SECOND line of defence on top of the
    physical end-stops -- a bad/uploaded G-code file with an out-of-range
    coordinate gets pulled back into bounds instead of ever reaching a
    switch. Prints a warning when it actually clamps something, so a
    genuinely bad file doesn't fail silently."""
    w, h = calibration["usable_width_mm"], calibration["usable_height_mm"]
    clamped_x = min(max(x_mm, 0.0), w)
    clamped_y = min(max(y_mm, 0.0), h)
    if clamped_x != x_mm or clamped_y != y_mm:
        print(f"[calibration] clamped out-of-bounds target ({x_mm:.1f}, {y_mm:.1f}) "
              f"-> ({clamped_x:.1f}, {clamped_y:.1f}) mm")
    return clamped_x, clamped_y


def _gcode_linear_move(target_x_mm, target_y_mm, feed_mm_min):
    """Moves to an ABSOLUTE mm position. Step delta is computed from
    absolute mm->step rounding referenced to the CURRENT actual step
    count (not accumulated from a separately-tracked float), so repeated
    small moves can't drift the way naively summing deltas would --
    same principle motion_planner.py's carry-forward rounding protects
    against. Per-step delay comes from the feed rate, not a fixed demo
    speed, so a slow F word really does draw slowly.

    Uses the CALIBRATED steps-per-mm for each axis (not the assumed
    STEPS_PER_MM constant), and the target is clamped into the safe
    working area first -- see _clamp_to_usable_area()."""
    target_x_mm, target_y_mm = _clamp_to_usable_area(target_x_mm, target_y_mm)
    spmm_x, spmm_y = calibration["steps_per_mm_x"], calibration["steps_per_mm_y"]
    # EDGE_TOLERANCE_MM offset: mm=0 in the drawing's own frame is NOT the
    # X_MIN/Y_MIN switch itself, it's already inset from it -- see calibrate().
    x0_mm = current_x_steps / spmm_x - EDGE_TOLERANCE_MM
    y0_mm = current_y_steps / spmm_y - EDGE_TOLERANCE_MM
    dist_mm = math.hypot(target_x_mm - x0_mm, target_y_mm - y0_mm)
    target_x_steps = round((target_x_mm + EDGE_TOLERANCE_MM) * spmm_x)
    target_y_steps = round((target_y_mm + EDGE_TOLERANCE_MM) * spmm_y)
    dx_steps = target_x_steps - current_x_steps
    dy_steps = target_y_steps - current_y_steps
    if dx_steps == 0 and dy_steps == 0:
        return
    major_steps = max(abs(dx_steps), abs(dy_steps), 1)
    feed_mm_s = max(feed_mm_min / 60.0, GCODE_MIN_FEED_MM_S)
    time_s = dist_mm / feed_mm_s
    step_delay_s = max(time_s / major_steps, CIRCLE_STEP_DELAY_S)
    bresenham_move(dx_steps, dy_steps, step_delay_s=step_delay_s)


def _gcode_arc_move(x0_mm, y0_mm, x1_mm, y1_mm, i_mm, j_mm, clockwise, feed_mm_min):
    """Standard G2/G3 arc: centre is (start + I, start + J). No native arc
    mode on this hardware (same as every other move here), so this
    interpolates the arc into short linear segments and runs each one
    through _gcode_linear_move -- consistent feed-rate timing, consistent
    end-stop checking, no separate code path to keep in sync."""
    cx, cy = x0_mm + i_mm, y0_mm + j_mm
    radius = math.hypot(i_mm, j_mm)
    if radius < 1e-6:
        _gcode_linear_move(x1_mm, y1_mm, feed_mm_min)
        return
    start_angle = math.atan2(y0_mm - cy, x0_mm - cx)
    end_angle = math.atan2(y1_mm - cy, x1_mm - cx)
    if clockwise:
        while end_angle >= start_angle:
            end_angle -= 2.0 * math.pi
    else:
        while end_angle <= start_angle:
            end_angle += 2.0 * math.pi
    angle_span = end_angle - start_angle
    arc_len_mm = abs(angle_span) * radius
    n_segments = max(4, int(arc_len_mm / 0.4))   # ~0.4mm chord length
    for k in range(1, n_segments + 1):
        if triggered_endstop() is not None:
            return
        a = start_angle + angle_span * (k / n_segments)
        _gcode_linear_move(cx + radius * math.cos(a), cy + radius * math.sin(a), feed_mm_min)


def run_gcode_file(path):
    """Runs a G-code file to completion, or until an end-stop aborts it.
    Sets the global estop state (not just a local return) so the rest of
    the script -- the main loop's own check, the 'r' command -- reacts to
    an abort exactly the same way it reacts to any other trigger.

    Refuses to run at all if calibrate() hasn't been done this session --
    checked here (not just in write_gcode_file()) so this applies equally
    whether a job comes from the web server or from typing 'g <file>' at
    this script's own command prompt."""
    global estopped, tripped_switch
    if not calibration["done"]:
        raise NotCalibratedError("Run calibration ('calibrate' command) before running a G-code file.")
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except OSError as e:
        print(f"Couldn't open '{path}': {e}")
        return

    abs_mode = True
    feed_mm_min = 900.0    # matches WriteGCode.py's drawFeedMmMin default
    cur_x_mm = current_x_steps / STEPS_PER_MM
    cur_y_mm = current_y_steps / STEPS_PER_MM
    print(f"Running G-code: {path} ({len(lines)} lines)")

    for lineno, raw in enumerate(lines, 1):
        hit = triggered_endstop()
        if hit is not None:
            if not estopped:
                estopped = True
                tripped_switch = hit
                print(f"\n!!! EMERGENCY STOP -- end-stop {hit} triggered !!!")
            print(f"G-code job ABORTED at line {lineno}.")
            return
        if estopped:
            print(f"G-code job ABORTED at line {lineno} -- already emergency-stopped.")
            return

        parsed = parse_gcode_line(raw)
        if parsed is None:
            continue
        cmd, p = parsed

        if cmd in ("G0", "G1"):
            if "F" in p:
                feed_mm_min = p["F"]
            if abs_mode:
                target_x = p.get("X", cur_x_mm)
                target_y = p.get("Y", cur_y_mm)
            else:
                target_x = cur_x_mm + p.get("X", 0.0)
                target_y = cur_y_mm + p.get("Y", 0.0)
            _gcode_linear_move(target_x, target_y, feed_mm_min)
            cur_x_mm, cur_y_mm = target_x, target_y

        elif cmd in ("G2", "G3"):
            if "F" in p:
                feed_mm_min = p["F"]
            if abs_mode:
                target_x = p.get("X", cur_x_mm)
                target_y = p.get("Y", cur_y_mm)
            else:
                target_x = cur_x_mm + p.get("X", 0.0)
                target_y = cur_y_mm + p.get("Y", 0.0)
            _gcode_arc_move(cur_x_mm, cur_y_mm, target_x, target_y,
                            p.get("I", 0.0), p.get("J", 0.0), cmd == "G2", feed_mm_min)
            cur_x_mm, cur_y_mm = target_x, target_y

        elif cmd == "G4":
            deadline = time.time() + p.get("P", 0.0)
            while time.time() < deadline:
                if triggered_endstop() is not None:
                    break
                time.sleep(0.005)

        elif cmd == "G90":
            abs_mode = True
        elif cmd == "G91":
            abs_mode = False
        elif cmd == "G20":
            print(f"[Line {lineno}] G20 (inches) seen -- this interpreter assumes "
                  f"mm throughout; values will be used as-is, NOT converted.")
        elif cmd == "G21":
            pass

        elif cmd == "M3":
            if not set_pen(False):
                print(f"[Line {lineno}] Pen-down did not confirm -- continuing anyway.")
        elif cmd == "M5":
            if not set_pen(True):
                print(f"[Line {lineno}] Pen-up did not confirm -- continuing anyway.")
        elif cmd in ("M2", "M30"):
            print(f"[Line {lineno}] Program end ({cmd}).")
            return

        else:
            print(f"[Line {lineno}] Skipping unsupported command: {cmd}")

    print(f"G-code job '{path}' complete.")


def input_listener():
    global current_mode, target_rpm, running, theta, current_x_steps, current_y_steps
    global estopped, tripped_switch, circle_radius_mm, circle_radius_steps

    while running:
        raw_cmd = sys.stdin.readline().strip()
        if not raw_cmd:
            continue
        cmd = raw_cmd.lower()   # only for matching single-letter commands --
                                 # a 'g' filename keeps its original case below

        if cmd == "q":
            running = False
            break
        elif cmd == "calibrate":
            if estopped:
                print("Cannot calibrate while emergency-stopped -- clear with 'r' first.")
                continue
            # CALIBRATING mode makes the main loop skip its automatic
            # emergency-stop reaction (see the main loop below) -- without
            # this, that reaction fires on every switch calibrate() hits
            # ON PURPOSE, forcing STEP1_PIN/STEP2_PIN to INACTIVE from a
            # SEPARATE thread while _step_axis() is mid-pulse on those same
            # pins. That race corrupted the step count and caused exactly
            # the confusing, contradictory failures seen during testing.
            current_mode = "CALIBRATING"
            try:
                result = calibrate()
                print(f"Calibrated. steps/mm: X={result['steps_per_mm_x']:.3f} "
                      f"Y={result['steps_per_mm_y']:.3f}  usable area: "
                      f"{result['usable_width_mm']:.1f} x {result['usable_height_mm']:.1f} mm")
            except RuntimeError as e:
                print(f"Calibration failed: {e}")
            finally:
                if not estopped:
                    current_mode = "RPM"
                    target_rpm = 0.0
        elif cmd == "r":
            hit = triggered_endstop()
            if hit is not None:
                print(f"Cannot clear -- end-stop {hit} is still triggered.")
            elif estopped:
                estopped = False
                tripped_switch = None
                current_mode = "RPM"
                target_rpm = 0.0
                print("Emergency stop cleared. Mode: Stopped.")
            else:
                print("Not stopped.")
        elif cmd == "u":
            print("Pen up..." if set_pen(True) else "Pen up FAILED.")
        elif cmd == "d":
            print("Pen down..." if set_pen(False) else "Pen down FAILED.")
        elif cmd == "c" or cmd.startswith("c "):
            if estopped:
                print("Cannot change mode while emergency-stopped -- clear with 'r' first.")
                continue
            parts = cmd.split()
            if len(parts) > 1:
                try:
                    new_r = float(parts[1])
                    if new_r <= 0:
                        print("Radius must be positive -- keeping previous radius.")
                    else:
                        circle_radius_mm = new_r
                except ValueError:
                    print(f"Couldn't parse '{parts[1]}' as a radius -- keeping previous radius.")
            circle_radius_steps = circle_radius_mm * STEPS_PER_MM
            current_mode = "CIRCLE"
            theta, current_x_steps, current_y_steps = 0.0, 0, 0
            print(f"Mode: Circle (radius {circle_radius_mm} mm)")
        elif cmd == "g" or cmd.startswith("g "):
            if estopped:
                print("Cannot start a G-code job while emergency-stopped -- clear with 'r' first.")
                continue
            path = raw_cmd[1:].strip()   # original case preserved, unlike `cmd`
            if not path:
                print("Usage: g <path-to-gcode-file>")
                continue
            current_mode = "GCODE"   # main loop idles while this thread drives the moves itself
            run_gcode_file(path)
            if not estopped:
                current_mode = "RPM"
                target_rpm = 0.0
        else:
            try:
                target_rpm = float(cmd)
                if estopped:
                    print("Cannot change mode while emergency-stopped -- clear with 'r' first.")
                    continue
                current_mode = "RPM"
                if abs(target_rpm) < 0.1:
                    target_rpm = 0.0
                    print("Mode: Stopped")
                else:
                    # Route through _axis_dir_value() -- the SAME function
                    # bresenham_move()/calibrate() use -- so "positive RPM"
                    # here means the same physical direction as "positive"
                    # means everywhere else. This used to set both DIR pins
                    # directly, ignoring INVERT_X/INVERT_Y entirely: a THIRD
                    # disconnected direction convention, on top of the
                    # bresenham_move/_home_axis mismatch already fixed --
                    # meaning a direction observed via a plain RPM test told
                    # you nothing reliable about what calibrate() or drawing
                    # would actually do.
                    req.set_value(DIR1_PIN, _axis_dir_value(INVERT_X, target_rpm > 0))
                    req.set_value(DIR2_PIN, _axis_dir_value(INVERT_Y, target_rpm > 0))
                    calculate_delay(target_rpm)
                    print(f"Mode: RPM ({target_rpm})")
            except ValueError:
                pass


# req holds the live gpiod line request once connect() succeeds. None
# means "not connected yet" -- every function above that touches hardware
# (bresenham_move, set_pen, triggered_endstop, run_gcode_file) assumes
# connect() has already been called successfully before it runs.
req = None


def connect():
    """Opens the GPIO chip and claims every pin this module drives. Safe to
    call more than once (a no-op if already connected). Raises
    GantryNotConnectedError -- NOT whatever raw exception gpiod/the OS
    throws -- if the gpiod library isn't installed or the chip can't be
    opened, so callers (server.py in particular) can catch ONE clear
    exception type regardless of which of those two things went wrong."""
    global req
    if req is not None:
        return
    if gpiod is None:
        raise GantryNotConnectedError(
            f"gpiod is not installed (this isn't the Odroid): {_GPIOD_IMPORT_ERROR}"
        )
    try:
        req = gpiod.request_lines(
            CHIP_PATH,
            consumer="stepper-control",
            config={
                DIR1_PIN: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.ACTIVE),
                STEP1_PIN: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE),
                DIR2_PIN: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.ACTIVE),
                STEP2_PIN: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE),
                # Pull-down is done in hardware (resistor on each switch line to
                # GND), not here -- see the module docstring.
                X_MIN_PIN: gpiod.LineSettings(direction=Direction.INPUT),
                X_MAX_PIN: gpiod.LineSettings(direction=Direction.INPUT),
                Y_MIN_PIN: gpiod.LineSettings(direction=Direction.INPUT),
                Y_MAX_PIN: gpiod.LineSettings(direction=Direction.INPUT),
                PEN_SWITCH_PIN: gpiod.LineSettings(direction=Direction.INPUT),
                PEN_MOTOR_PIN: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE),
            },
        )
    except Exception as e:
        req = None
        raise GantryNotConnectedError(f"Could not open {CHIP_PATH}: {e}") from e
    calculate_delay(target_rpm)


def disconnect():
    """Releases the GPIO lines (motors/pen off first). Safe to call even
    if never connected."""
    global req
    if req is None:
        return
    try:
        req.set_value(STEP1_PIN, Value.INACTIVE)
        req.set_value(STEP2_PIN, Value.INACTIVE)
        req.set_value(PEN_MOTOR_PIN, Value.INACTIVE)
    finally:
        req.release()
        req = None


def is_connected():
    return req is not None


class NotCalibratedError(RuntimeError):
    """Raised by write_gcode_file() when calibrate() hasn't been run yet
    this session. The assumed STEPS_PER_MM constant and switch-travel
    figures are estimates -- writing before calibrating risks the wrong
    distance being drawn, or (worse) drifting into a switch over a long
    job. Calibration must be redone every time the process restarts;
    it is intentionally NOT persisted to disk, since a mechanical bump
    between sessions would silently invalidate a saved value."""
    pass


def _step_axis(step_pin, dir_pin, dir_value, stop_when, ignore=frozenset()):
    """Steps ONE axis, slowly, with dir_pin held at dir_value, until
    `stop_when(triggered_name)` returns True for some triggered
    end-stop. Any OTHER triggered end-stop not in `ignore` is treated as
    a real fault (crossed axis wiring) and raises immediately. Returns
    (name_that_stopped_it, steps_taken).

    This is direction-AGNOSTIC on purpose: it never assumes which
    physical direction dir_value drives, so a wrong INVERT_X/INVERT_Y or
    a backwards motor/switch wiring can't make it grind into a switch it
    doesn't recognise -- it just finds out empirically which switch this
    direction leads to and reports that back."""
    req.set_value(dir_pin, dir_value)
    steps = 0
    while True:
        hit = triggered_endstop()
        if hit is not None:
            if stop_when(hit):
                return hit, steps
            if hit not in ignore:
                raise RuntimeError(f"Unexpected end-stop {hit} triggered -- check wiring (crossed axes?).")
        if steps >= HOMING_MAX_STEPS:
            raise RuntimeError("No expected end-stop triggered during homing -- check wiring.")
        req.set_value(step_pin, Value.ACTIVE)
        time.sleep(HOMING_STEP_DELAY_S)
        req.set_value(step_pin, Value.INACTIVE)
        time.sleep(HOMING_STEP_DELAY_S)
        steps += 1


def _calibrate_axis(step_pin, dir_pin, min_name, max_name):
    """Direction-agnostic axis calibration: drives one way (arbitrarily
    picked as Value.ACTIVE) until EITHER of this axis's two end-stops
    triggers -- whichever one that turns out to be -- then reverses and
    drives until the OTHER one triggers, counting the steps between
    them. Never needs to know in advance which DIR value means "toward
    MIN"; it discovers that empirically every run, so it can't be broken
    by an INVERT_X/INVERT_Y mismatch, a backwards motor wire, or swapped
    switch wires the way the old direction-specific homing could.

    Returns (first_hit, second_hit, steps_between)."""
    print(f"[calibrate] driving until either {min_name} or {max_name} is activated...")
    first_hit, _ = _step_axis(step_pin, dir_pin, Value.ACTIVE,
                               stop_when=lambda h: h in (min_name, max_name))
    print(f"[calibrate] {first_hit} has been activated. Now reversing toward "
          f"{max_name if first_hit == min_name else min_name}...")
    other = max_name if first_hit == min_name else min_name
    # ignore=first_hit: that switch may still read triggered for the
    # first few steps of the reverse move (release lag), which is
    # expected, not a wiring fault.
    second_hit, travel_steps = _step_axis(step_pin, dir_pin, Value.INACTIVE,
                                           stop_when=lambda h: h == other,
                                           ignore={first_hit})
    print(f"[calibrate] {second_hit} has been activated. {travel_steps} steps measured between the two switches.")
    return first_hit, second_hit, travel_steps


def calibrate():
    """Full calibration sequence -- must be run before any writing job.

    1. Drives X in an arbitrary direction until EITHER X switch triggers
       (whichever one that is), then reverses until the OTHER X switch
       triggers, counting steps between them -- see _calibrate_axis().
       Doesn't need to know in advance which way is "toward X_MIN".
    2. Same for Y.
    3. Derives the ACTUAL steps-per-mm for each axis from that measured
       step count and the known physical switch-to-switch distance,
       instead of trusting the assumed STEPS_PER_MM constant.
    4. Moves to the safe starting corner -- EDGE_TOLERANCE_MM in from
       both switches -- so writing always begins from a known, repeatable
       position with headroom on every side.

    Refuses to run if an end-stop is already tripped (clear it with 'r'
    /a fresh connect first) -- calibrating while already jammed against
    a switch would produce a nonsense measurement.

    NOTE on X/Y_SWITCH_TRAVEL_MM: these use the FULLY-pressed distance,
    not the ~2mm-earlier "first click" figure -- the switch's own
    compliance (the bit of give between first contact and full press)
    is small and roughly constant, so it mostly cancels out between the
    two ends of the same axis. EDGE_TOLERANCE_MM (5mm, 10x that
    compliance) is what actually keeps every real move away from the
    switches, not precision in this constant."""
    global current_x_steps, current_y_steps

    if triggered_endstop() is not None:
        raise RuntimeError(f"Cannot calibrate -- end-stop {tripped_switch} already tripped.")

    # Direction-agnostic: doesn't need INVERT_X/INVERT_Y to be correct at
    # all, see _calibrate_axis()'s docstring.
    print("[calibrate] === X axis ===")
    _x_first, x_second, x_travel_steps = _calibrate_axis(STEP1_PIN, DIR1_PIN, "X_MIN", "X_MAX")
    current_x_steps = x_travel_steps if x_second == "X_MAX" else 0

    print("[calibrate] === Y axis ===")
    _y_first, y_second, y_travel_steps = _calibrate_axis(STEP2_PIN, DIR2_PIN, "Y_MIN", "Y_MAX")
    current_y_steps = y_travel_steps if y_second == "Y_MAX" else 0

    calibration.update({
        "done": True,
        "steps_per_mm_x": x_travel_steps / X_SWITCH_TRAVEL_MM,
        "steps_per_mm_y": y_travel_steps / Y_SWITCH_TRAVEL_MM,
        "usable_width_mm": X_SWITCH_TRAVEL_MM - 2 * EDGE_TOLERANCE_MM,
        "usable_height_mm": Y_SWITCH_TRAVEL_MM - 2 * EDGE_TOLERANCE_MM,
    })

    # Move to the safe starting corner (mm=0,0 in the drawing's own
    # frame), ready for the first real move of a writing job.
    _gcode_linear_move(0.0, 0.0, feed_mm_min=600.0)

    return dict(calibration)


def write_gcode_file(path):
    """One-shot entry point for server.py: connects if needed, runs the
    file, and always leaves the connection open afterward (a Flask process
    stays alive across requests, so there's no need to reconnect every
    time -- call disconnect() yourself at shutdown if you want to release
    the pins). Raises GantryNotConnectedError if there's no gantry here,
    or NotCalibratedError if calibrate() hasn't been run this session --
    callers should catch both specifically to report a clean error
    instead of a 500. (The calibration check itself lives in
    run_gcode_file(), so it applies the same way from the CLI too.)"""
    connect()
    run_gcode_file(path)
    return {
        "estopped": estopped,
        "tripped_switch": tripped_switch,
    }


if __name__ == "__main__":
    connect()
    threading.Thread(target=input_listener, daemon=True).start()
    print("Ready. Commands: 'calibrate' (required before 'g'), 'c' (circle), 'g <file>' (run G-code), [number] (RPM/stop), "
          "'u'/'d' (pen), 'r' (clear estop), 'q' (quit)")

    try:
        while running:
            # End-stops ALWAYS win, checked before every single step pulse in
            # either mode -- this is the actual emergency-stop priority the
            # user asked for, not just a polite suggestion to the motors.
            #
            # EXCEPT during CALIBRATING: calibrate() (running in the OTHER
            # thread, input_listener) deliberately drives INTO switches on
            # purpose and handles that itself in _step_axis(). If this loop
            # also reacted here, it would force STEP1_PIN/STEP2_PIN to
            # INACTIVE from a second thread while _step_axis() is mid-pulse
            # on those same pins -- a real race that corrupted step counts
            # and caused confusing, contradictory calibration failures.
            if current_mode == "CALIBRATING":
                time.sleep(0.01)
                continue
            hit = triggered_endstop()
            if hit is not None:
                req.set_value(STEP1_PIN, Value.INACTIVE)
                req.set_value(STEP2_PIN, Value.INACTIVE)
                if not estopped:
                    estopped = True
                    tripped_switch = hit
                    target_rpm = 0.0
                    print(f"\n!!! EMERGENCY STOP -- end-stop {hit} triggered !!!")
                time.sleep(0.005)
                continue
            elif estopped:
                # Switch went clear on its own without an explicit 'r' -- stay
                # stopped anyway; a human decision is required to resume,
                # never an automatic one.
                time.sleep(0.005)
                continue

            if current_mode == "RPM":
                if target_rpm != 0.0 and step_delay_s > 0:
                    req.set_value(STEP1_PIN, Value.ACTIVE)
                    req.set_value(STEP2_PIN, Value.ACTIVE)
                    time.sleep(step_delay_s)
                    req.set_value(STEP1_PIN, Value.INACTIVE)
                    req.set_value(STEP2_PIN, Value.INACTIVE)
                    time.sleep(step_delay_s)
                else:
                    time.sleep(0.01)

            elif current_mode == "CIRCLE":
                now = time.perf_counter()
                if now - last_circle_time >= circle_interval_s:
                    last_circle_time = now

                    theta += 0.008
                    if theta >= 2.0 * math.pi:
                        theta -= 2.0 * math.pi

                    target_x = round(circle_radius_steps * math.sin(theta))
                    target_y = round(circle_radius_steps * (1.0 - math.cos(theta)))

                    # Issue EVERY step needed to reach this waypoint exactly
                    # (Bresenham-interpolated), not just one per axis -- this
                    # is what actually traces a circle instead of a diamond,
                    # and what makes the reached radius match the commanded
                    # one instead of chronically undershooting it.
                    bresenham_move(target_x - current_x_steps, target_y - current_y_steps)

            elif current_mode == "GCODE":
                # The input_listener thread runs the whole file itself (see
                # the 'g' command) and drives the pins directly -- this loop
                # just idles so it never ALSO tries to step in the meantime.
                time.sleep(0.01)

    finally:
        disconnect()
        print("Exited.")
