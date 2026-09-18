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

Commands (via stdin, same as before):
    <number>   RPM mode at that RPM (0 = stop)
    c          circle demo mode
    u          pen up
    d          pen down
    r          clear a tripped emergency stop (only works if every
               end-stop currently reads clear -- never auto-clears)
    q          quit
"""
import math
import sys
import threading
import time
import gpiod
from gpiod.line import Direction, Value

# Hardware Pin Assignments for ODroid N2+ (gpiochip0)
CHIP_PATH = "/dev/gpiochip0"
DIR1_PIN = 62   # Physical Pin 7
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
STEPS_PER_MM = 80.0
DEFAULT_CIRCLE_RADIUS_MM = 8.0
CIRCLE_STEP_DELAY_S = 0.000002   # per-step pulse HIGH time, same as the old code used

# System State
current_mode = "RPM"
target_rpm = 30.0
step_delay_s = 0.005
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


def triggered_endstop():
    """Returns the name of the first end-stop currently HIGH (triggered),
    or None. Cheap enough (a handful of gpiod reads) to call every loop
    iteration without meaningfully affecting step timing."""
    for name, pin in ENDSTOP_PINS.items():
        if req.get_value(pin) == Value.ACTIVE:
            return name
    return None


def bresenham_move(dx, dy):
    """Steps BOTH axes from the current position by (dx, dy) steps, using
    the same Bresenham interpolation motion_planner.py/motion_executor.ino
    use for straight moves -- every step actually needed gets issued (not
    just one per axis per call), which is what makes the traced path
    match the commanded shape instead of lagging into cut corners. Global
    current_x_steps/current_y_steps are updated as it goes. Returns early
    (without finishing the segment) if an end-stop trips mid-move -- the
    outer loop's own check handles latching the emergency-stop state."""
    global current_x_steps, current_y_steps
    ax, ay = abs(dx), abs(dy)
    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1
    req.set_value(DIR1_PIN, Value.ACTIVE if sx > 0 else Value.INACTIVE)
    req.set_value(DIR2_PIN, Value.ACTIVE if sy > 0 else Value.INACTIVE)

    def pulse(step_pin):
        req.set_value(step_pin, Value.ACTIVE)
        time.sleep(CIRCLE_STEP_DELAY_S)
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


def set_pen(target_up, timeout_s=1.0):
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
    req.set_value(PEN_MOTOR_PIN, Value.INACTIVE)
    if not reached:
        print("Pen move did NOT confirm via switch -- check motor/wiring.")
    return reached


def input_listener():
    global current_mode, target_rpm, running, theta, current_x_steps, current_y_steps
    global estopped, tripped_switch, circle_radius_mm, circle_radius_steps

    while running:
        cmd = sys.stdin.readline().strip().lower()
        if not cmd:
            continue

        if cmd == "q":
            running = False
            break
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
                    dir_val = Value.ACTIVE if target_rpm > 0 else Value.INACTIVE
                    req.set_value(DIR1_PIN, dir_val)
                    req.set_value(DIR2_PIN, dir_val)
                    calculate_delay(target_rpm)
                    print(f"Mode: RPM ({target_rpm})")
            except ValueError:
                pass


# Initialize GPIO lines using libgpiod v2 API
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

calculate_delay(target_rpm)

threading.Thread(target=input_listener, daemon=True).start()
print("Ready. Commands: 'c' (circle), [number] (RPM/stop), 'u'/'d' (pen), 'r' (clear estop), 'q' (quit)")

try:
    while running:
        # End-stops ALWAYS win, checked before every single step pulse in
        # either mode -- this is the actual emergency-stop priority the
        # user asked for, not just a polite suggestion to the motors.
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

finally:
    req.set_value(STEP1_PIN, Value.INACTIVE)
    req.set_value(STEP2_PIN, Value.INACTIVE)
    req.set_value(PEN_MOTOR_PIN, Value.INACTIVE)
    req.release()
    print("Exited.")
