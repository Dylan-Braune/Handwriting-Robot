"""
odroid_direct_drive.py -- ODROID N2+ direct-GPIO gantry driver.

Bit-bangs the X/Y stepper STEP/DIR pins over gpiod, no microcontroller in
between. End-stops are checked in the same loop as every step pulse, so
"stop immediately on any trigger" is a real guarantee, not best-effort.

WIRING
    All 5 switches are active-HIGH (triggered = 3.3V), each with its own
    ~10k pull-down resistor to GND (done in hardware, not here).
    - X_MIN, X_MAX, Y_MIN, Y_MAX: travel-limit switches.
    - PEN switch: feedback only. HIGH = pen up, LOW = pen down.
    - PEN motor: a digital output driving a cam/eccentric that TOGGLES
      pen height each time it runs (not "move to position"). set_pen()
      runs it only until the switch confirms the target was reached.

GCODE SUPPORTED
    G0/G1 X.. Y.. F..        linear move (no separate rapid mode)
    G2/G3 X.. Y.. I.. J.. F. arc, centre at (start + I, start + J)
    G4 P..                   dwell P seconds
    G90/G91                  absolute/relative positioning
    G20/G21                  inches (values used as-is, NOT converted) / mm
    M3 / M5                  pen down / pen up
    M2 / M30                 program end
    F sets feed rate (mm/min), persists until changed. An end-stop trip
    aborts the file and requires 'r' to clear before anything else runs.

CLI COMMANDS (via stdin)
    calibrate   home both axes, measure real steps-per-mm, park at the
                safe start corner. Required once per session before 'g'.
    <number>    RPM test mode (0 = stop)
    c [radius]  circle demo
    g <path>    run a G-code file
    u / d       pen up / down
    r           clear a tripped emergency stop (never auto-clears)
    q           quit
"""
import math
import re
import sys
import threading
import time

# gpiod only exists on the Odroid. Importing this module elsewhere (a
# laptop, the web server) must not crash -- hardware access only happens
# inside connect() and the functions that require it.
try:
    import gpiod
    from gpiod.line import Direction, Value
    _GPIOD_IMPORT_ERROR = None
except ImportError as _e:
    gpiod = None
    Direction = Value = None
    _GPIOD_IMPORT_ERROR = _e


class GantryNotConnectedError(RuntimeError):
    """No gpiod / no GPIO chip available -- not the Odroid, or the
    device couldn't be opened. Callers should report this cleanly
    instead of a raw exception."""


class NotCalibratedError(RuntimeError):
    """calibrate() hasn't succeeded yet this session. Never persisted to
    disk: a mechanical bump between sessions would silently invalidate a
    saved value, so it must be redone every restart."""


# --- Pin assignments (ODROID N2+, gpiochip0 offsets) ------------------------
CHIP_PATH = "/dev/gpiochip0"

DIR1_PIN, STEP1_PIN = 64, 68   # X motor -- physical pins 7, 11
DIR2_PIN, STEP2_PIN = 81, 69   # Y motor -- physical pins 12, 13

X_MIN_PIN, X_MAX_PIN = 72, 65  # physical pins 15, 16
Y_MIN_PIN, Y_MAX_PIN = 66, 67  # physical pins 18, 22
PEN_SWITCH_PIN = 70            # physical pin 33 (input, feedback)
PEN_MOTOR_PIN = 71             # physical pin 35 (output)

ENDSTOP_PINS = {"X_MIN": X_MIN_PIN, "X_MAX": X_MAX_PIN,
                "Y_MIN": Y_MIN_PIN, "Y_MAX": Y_MAX_PIN}

# True/False per axis: flip if that axis moves the wrong physical way
# for a commanded positive/negative step delta. Only ever matters for
# DRAWING direction -- calibrate() below never needs these to be correct.
INVERT_X = True
INVERT_Y = True

# --- Kinematics --------------------------------------------------------------
MICROSTEPS = 16
STEPS_PER_REV = 200 * MICROSTEPS
STEPS_PER_MM = 80.0   # only used before calibrate() has run
CIRCLE_STEP_DELAY_S = 0.000002
DEFAULT_CIRCLE_RADIUS_MM = 8.0

# --- Calibration --------------------------------------------------------------
# Switch-to-switch travel, fully pressed to fully pressed (measured).
X_SWITCH_TRAVEL_MM = 204.0
Y_SWITCH_TRAVEL_MM = 262.0
EDGE_TOLERANCE_MM = 5.0        # kept clear of both switches on both axes
HOMING_STEP_DELAY_S = 0.0008   # per-step pulse time while homing (both axes)
HOMING_MAX_STEPS = 200_000     # a switch that never triggers is a fault
ENDSTOP_DEBOUNCE_S = 0.004     # filters brief noise spikes, not real triggers

calibration = {
    "done": False,
    "steps_per_mm_x": STEPS_PER_MM,
    "steps_per_mm_y": STEPS_PER_MM,
    "usable_width_mm": X_SWITCH_TRAVEL_MM - 2 * EDGE_TOLERANCE_MM,
    "usable_height_mm": Y_SWITCH_TRAVEL_MM - 2 * EDGE_TOLERANCE_MM,
}

# --- Runtime state -------------------------------------------------------------
req = None                # live gpiod line request once connect() succeeds
current_mode = "RPM"      # RPM | CIRCLE | GCODE | CALIBRATING
target_rpm = 0.0          # must start at 0: nonzero here would move the
step_delay_s = 0.0        # gantry the instant the main loop starts running
running = True

current_x_steps = 0
current_y_steps = 0
theta = 0.0
last_circle_time = time.perf_counter()
circle_interval_s = 0.0015
circle_radius_mm = DEFAULT_CIRCLE_RADIUS_MM
circle_radius_steps = circle_radius_mm * STEPS_PER_MM

estopped = False
tripped_switch = None


# =============================================================================
# Connection
# =============================================================================
def connect():
    """Opens the GPIO chip and claims every pin. Safe to call repeatedly.
    Raises GantryNotConnectedError for any failure reason (missing
    gpiod, chip busy, etc.) so callers only need to catch one type."""
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
                DIR1_PIN: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.ACTIVE),
                STEP1_PIN: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE),
                DIR2_PIN: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.ACTIVE),
                STEP2_PIN: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE),
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
    """Releases the GPIO lines (motors/pen off first). Safe if never connected."""
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


# =============================================================================
# Low-level motion primitives
# =============================================================================
def calculate_delay(rpm):
    global step_delay_s
    if abs(rpm) < 0.1:
        step_delay_s = 0
    else:
        step_delay_s = (60.0 / (abs(rpm) * STEPS_PER_REV)) / 2.0


def triggered_endstop(ignore=frozenset()):
    """Name of the first end-stop currently HIGH that isn't in `ignore`,
    or None. `ignore` SKIPS those pins during the scan (not just filters
    the result), so a switch legitimately still triggered elsewhere
    (e.g. the other axis resting at its own limit) can never hide a
    genuinely different switch further down the scan.

    Debounced: a pin must still read ACTIVE after ENDSTOP_DEBOUNCE_S
    before it's trusted, filtering brief noise (e.g. from switching the
    pen motor) without adding real delay to the common "nothing
    triggered" case."""
    for name, pin in ENDSTOP_PINS.items():
        if name in ignore:
            continue
        if req.get_value(pin) == Value.ACTIVE:
            time.sleep(ENDSTOP_DEBOUNCE_S)
            if req.get_value(pin) == Value.ACTIVE:
                return name
    return None


def axis_dir_value(invert, positive):
    """The Value to send an axis's DIR pin to move positive (True) or
    negative (False), respecting that axis's INVERT flag. Single source
    of truth: every direction decision in this file (drawing, homing,
    the RPM test command) goes through this, so they can never disagree
    with each other about which pin value means which physical direction."""
    forward = positive != invert   # XOR
    return Value.ACTIVE if forward else Value.INACTIVE


def bresenham_move(dx, dy, step_delay_s=CIRCLE_STEP_DELAY_S):
    """Steps both axes by (dx, dy) steps using Bresenham interpolation,
    so a diagonal move traces a straight line instead of one axis
    finishing early. Stops early (without finishing) if an end-stop
    trips mid-move -- the caller's own loop handles the estop state."""
    global current_x_steps, current_y_steps
    ax, ay = abs(dx), abs(dy)
    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1
    req.set_value(DIR1_PIN, axis_dir_value(INVERT_X, sx > 0))
    req.set_value(DIR2_PIN, axis_dir_value(INVERT_Y, sy > 0))

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


# =============================================================================
# Pen
# =============================================================================
PEN_UP_OVERRUN_S = 0.04  # extra run time after the switch first trips, so the
                         # pen mechanism finishes seating instead of stopping
                         # the instant the switch first makes contact


def pen_is_up():
    """HIGH ("in") = up, LOW ("out") = down."""
    return req.get_value(PEN_SWITCH_PIN) == Value.ACTIVE


def set_pen(target_up, timeout_s=3.0):
    """Runs the pen motor only until the switch confirms the target
    state, or timeout_s elapses (a real fault, reported not swallowed).
    Refuses if an end-stop is already tripped."""
    if triggered_endstop() is not None:
        print(f"Refusing pen move -- end-stop {tripped_switch} still tripped.")
        return False
    if pen_is_up() == target_up:
        return True  # already there -- the toggle motor must not run

    req.set_value(PEN_MOTOR_PIN, Value.ACTIVE)
    deadline = time.time() + timeout_s
    reached = False
    while time.time() < deadline:
        if triggered_endstop() is not None:
            break
        if pen_is_up() == target_up:
            reached = True
            break
        time.sleep(0.002)
    if reached and target_up:
        time.sleep(PEN_UP_OVERRUN_S)
    req.set_value(PEN_MOTOR_PIN, Value.INACTIVE)

    if not reached:
        print("Pen move did NOT confirm via switch -- check motor/wiring.")
    return reached


# =============================================================================
# Calibration -- direction-agnostic: never needs to know in advance which
# way is "toward MIN". Drives one way until EITHER switch on an axis
# triggers, reverses to find the OTHER one, and measures the steps between.
# =============================================================================
def _step_axis(step_pin, dir_pin, dir_value, stop_when, ignore=frozenset()):
    """Steps one axis at dir_value until stop_when(triggered_name) is
    True. Any other triggered switch not in `ignore` is a real fault
    (crossed wiring) and raises immediately. Returns (name, steps)."""
    req.set_value(dir_pin, dir_value)
    steps = 0
    while True:
        hit = triggered_endstop(ignore=ignore)
        if hit is not None:
            if stop_when(hit):
                return hit, steps
            raise RuntimeError(f"Unexpected end-stop {hit} triggered -- check wiring.")
        if steps >= HOMING_MAX_STEPS:
            raise RuntimeError("No expected end-stop triggered during homing -- check wiring.")
        req.set_value(step_pin, Value.ACTIVE)
        time.sleep(HOMING_STEP_DELAY_S)
        req.set_value(step_pin, Value.INACTIVE)
        time.sleep(HOMING_STEP_DELAY_S)
        steps += 1


def _calibrate_axis(step_pin, dir_pin, min_name, max_name, other_axis_names=frozenset()):
    """Finds both switches on one axis without assuming which direction
    leads to which. other_axis_names: the other axis's switches, which
    may still read triggered if it was just calibrated (its own normal
    resting position, not a fault here)."""
    print(f"[calibrate] driving until either {min_name} or {max_name} is activated...")
    first_hit, _ = _step_axis(step_pin, dir_pin, Value.ACTIVE,
                               stop_when=lambda h: h in (min_name, max_name),
                               ignore=other_axis_names)
    other = max_name if first_hit == min_name else min_name
    print(f"[calibrate] {first_hit} activated. Reversing toward {other}...")
    # ignore first_hit too: it may still read triggered for the first
    # few steps of the reverse move (release lag), not a real fault.
    second_hit, travel_steps = _step_axis(step_pin, dir_pin, Value.INACTIVE,
                                           stop_when=lambda h: h == other,
                                           ignore={first_hit} | set(other_axis_names))
    print(f"[calibrate] {second_hit} activated. {travel_steps} steps measured.")
    return first_hit, second_hit, travel_steps


def calibrate():
    """Homes both axes, derives real steps-per-mm from the measured
    travel and the known switch-to-switch distance, then parks at the
    safe starting corner (EDGE_TOLERANCE_MM in from both switches).
    Refuses if an end-stop is already tripped."""
    global current_x_steps, current_y_steps

    if triggered_endstop() is not None:
        raise RuntimeError(f"Cannot calibrate -- end-stop {tripped_switch} already tripped.")

    print("[calibrate] === X axis ===")
    _, x_second, x_travel = _calibrate_axis(STEP1_PIN, DIR1_PIN, "X_MIN", "X_MAX")
    current_x_steps = x_travel if x_second == "X_MAX" else 0

    print("[calibrate] === Y axis ===")
    _, y_second, y_travel = _calibrate_axis(
        STEP2_PIN, DIR2_PIN, "Y_MIN", "Y_MAX", other_axis_names={"X_MIN", "X_MAX"})
    current_y_steps = y_travel if y_second == "Y_MAX" else 0

    calibration.update({
        "done": True,
        "steps_per_mm_x": x_travel / X_SWITCH_TRAVEL_MM,
        "steps_per_mm_y": y_travel / Y_SWITCH_TRAVEL_MM,
        "usable_width_mm": X_SWITCH_TRAVEL_MM - 2 * EDGE_TOLERANCE_MM,
        "usable_height_mm": Y_SWITCH_TRAVEL_MM - 2 * EDGE_TOLERANCE_MM,
    })
    _gcode_linear_move(0.0, 0.0, feed_mm_min=600.0)  # park at the safe start corner
    return dict(calibration)


# =============================================================================
# G-code
# =============================================================================
GCODE_MIN_FEED_MM_S = 0.1   # floor so a stray F0 doesn't divide by zero
_GCODE_WORD_RE = re.compile(r'([A-Za-z])\s*(-?[0-9]*\.?[0-9]+)')


def parse_gcode_line(raw):
    """One line -> (command, {letter: value}) or None for blank/comment
    lines. Assumes one G/M word per line; every other letter is a param."""
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
    """Clamps a target into the calibrated safe rectangle -- a second
    line of defence on top of the physical switches, so a bad/uploaded
    G-code coordinate gets pulled back in bounds instead of ever
    reaching one."""
    w, h = calibration["usable_width_mm"], calibration["usable_height_mm"]
    cx, cy = min(max(x_mm, 0.0), w), min(max(y_mm, 0.0), h)
    if (cx, cy) != (x_mm, y_mm):
        print(f"[gcode] clamped out-of-bounds target ({x_mm:.1f}, {y_mm:.1f}) -> ({cx:.1f}, {cy:.1f}) mm")
    return cx, cy


def _gcode_linear_move(target_x_mm, target_y_mm, feed_mm_min):
    """Moves to an absolute mm position using the CALIBRATED steps-per-mm
    per axis. Step delta is computed from the current actual step count
    (not an accumulated float), so repeated small moves can't drift."""
    target_x_mm, target_y_mm = _clamp_to_usable_area(target_x_mm, target_y_mm)
    spmm_x, spmm_y = calibration["steps_per_mm_x"], calibration["steps_per_mm_y"]
    # mm=0 in this coordinate frame is EDGE_TOLERANCE_MM in from the
    # X_MIN/Y_MIN switch, not the switch itself -- see calibrate().
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
    step_delay_s = max((dist_mm / feed_mm_s) / major_steps, CIRCLE_STEP_DELAY_S)
    bresenham_move(dx_steps, dy_steps, step_delay_s=step_delay_s)


def _gcode_arc_move(x0_mm, y0_mm, x1_mm, y1_mm, i_mm, j_mm, clockwise, feed_mm_min):
    """G2/G3 arc, centre at (start + I, start + J). No native arc mode,
    so this interpolates into short linear segments (~0.4mm chords)."""
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
    n_segments = max(4, int(abs(angle_span) * radius / 0.4))
    for k in range(1, n_segments + 1):
        if triggered_endstop() is not None:
            return
        a = start_angle + angle_span * (k / n_segments)
        _gcode_linear_move(cx + radius * math.cos(a), cy + radius * math.sin(a), feed_mm_min)


def run_gcode_file(path):
    """Runs a G-code file to completion or until an end-stop aborts it.
    Refuses to run at all until calibrate() has succeeded this session
    -- checked here so it applies from both the web server and the CLI."""
    global estopped, tripped_switch
    if not calibration["done"]:
        raise NotCalibratedError("Run calibration ('calibrate') before running a G-code file.")
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except OSError as e:
        print(f"Couldn't open '{path}': {e}")
        return

    abs_mode = True
    feed_mm_min = 900.0
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
            target_x = p.get("X", cur_x_mm) if abs_mode else cur_x_mm + p.get("X", 0.0)
            target_y = p.get("Y", cur_y_mm) if abs_mode else cur_y_mm + p.get("Y", 0.0)
            _gcode_linear_move(target_x, target_y, feed_mm_min)
            cur_x_mm, cur_y_mm = target_x, target_y

        elif cmd in ("G2", "G3"):
            if "F" in p:
                feed_mm_min = p["F"]
            target_x = p.get("X", cur_x_mm) if abs_mode else cur_x_mm + p.get("X", 0.0)
            target_y = p.get("Y", cur_y_mm) if abs_mode else cur_y_mm + p.get("Y", 0.0)
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
            print(f"[Line {lineno}] G20 (inches) -- values used as-is, NOT converted.")
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


def write_gcode_file(path):
    """server.py's entry point: connects if needed, runs the file,
    leaves the connection open afterward. Raises GantryNotConnectedError
    or NotCalibratedError -- callers should catch both for a clean
    error instead of a 500."""
    connect()
    run_gcode_file(path)
    return {"estopped": estopped, "tripped_switch": tripped_switch}


# =============================================================================
# Interactive CLI
# =============================================================================
def input_listener():
    global current_mode, target_rpm, running, theta, current_x_steps, current_y_steps
    global estopped, tripped_switch, circle_radius_mm, circle_radius_steps

    while running:
        raw_cmd = sys.stdin.readline().strip()
        if not raw_cmd:
            continue
        cmd = raw_cmd.lower()  # a 'g' filename keeps its original case below

        if cmd == "q":
            running = False
            break

        elif cmd == "calibrate":
            if estopped:
                print("Cannot calibrate while emergency-stopped -- clear with 'r' first.")
                continue
            # CALIBRATING mode makes the main loop skip its own estop
            # reaction below -- calibrate() drives into switches on
            # purpose and handles that itself.
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
                    current_mode, target_rpm = "RPM", 0.0

        elif cmd == "r":
            hit = triggered_endstop()
            if hit is not None:
                print(f"Cannot clear -- end-stop {hit} is still triggered.")
            elif estopped:
                estopped, tripped_switch = False, None
                current_mode, target_rpm = "RPM", 0.0
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
            path = raw_cmd[1:].strip()  # original case preserved, unlike cmd
            if not path:
                print("Usage: g <path-to-gcode-file>")
                continue
            current_mode = "GCODE"  # main loop idles; this thread drives the moves
            run_gcode_file(path)
            if not estopped:
                current_mode, target_rpm = "RPM", 0.0

        else:
            try:
                target_rpm = float(cmd)
            except ValueError:
                continue
            if estopped:
                print("Cannot change mode while emergency-stopped -- clear with 'r' first.")
                continue
            current_mode = "RPM"
            if abs(target_rpm) < 0.1:
                target_rpm = 0.0
                print("Mode: Stopped")
            else:
                req.set_value(DIR1_PIN, axis_dir_value(INVERT_X, target_rpm > 0))
                req.set_value(DIR2_PIN, axis_dir_value(INVERT_Y, target_rpm > 0))
                calculate_delay(target_rpm)
                print(f"Mode: RPM ({target_rpm})")


if __name__ == "__main__":
    connect()
    threading.Thread(target=input_listener, daemon=True).start()
    print("Ready. Commands: 'calibrate' (required before 'g'), 'c' (circle), 'g <file>' (run G-code), "
          "[number] (RPM/stop), 'u'/'d' (pen), 'r' (clear estop), 'q' (quit)")

    try:
        while running:
            # End-stops always win -- EXCEPT during CALIBRATING, where
            # calibrate() drives into switches on purpose and handles
            # that itself. Reacting here too would force STEP1_PIN/
            # STEP2_PIN off from this thread while _step_axis() (the
            # other thread) is mid-pulse on those same pins.
            if current_mode == "CALIBRATING":
                time.sleep(0.01)
                continue

            hit = triggered_endstop()
            if hit is not None:
                req.set_value(STEP1_PIN, Value.INACTIVE)
                req.set_value(STEP2_PIN, Value.INACTIVE)
                if not estopped:
                    estopped, tripped_switch, target_rpm = True, hit, 0.0
                    print(f"\n!!! EMERGENCY STOP -- end-stop {hit} triggered !!!")
                time.sleep(0.005)
                continue
            elif estopped:
                # A switch clearing on its own never auto-resumes --
                # only an explicit 'r' does.
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
                    theta = (theta + 0.008) % (2.0 * math.pi)
                    target_x = round(circle_radius_steps * math.sin(theta))
                    target_y = round(circle_radius_steps * (1.0 - math.cos(theta)))
                    bresenham_move(target_x - current_x_steps, target_y - current_y_steps)

            elif current_mode == "GCODE":
                time.sleep(0.01)  # input_listener thread drives this directly

    finally:
        disconnect()
        print("Exited.")
