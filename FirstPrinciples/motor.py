"""
motor.py -- turns a pen trajectory (list of x, y points) into stepper
motor pulses on the gantry (FU 2.8, 3.1, 3.2). gpiod is a GPIO HARDWARE
library (same allowed category as the camera driver) -- the actual
path-following logic (Bresenham's line algorithm, step timing) is
written out by hand below, not provided by any library.

Pin numbers match the physical gantry wiring already established for
this project (ODROID N2+, gpiochip0).
"""
import time

# gpiod only exists on the Odroid -- importing this file on a laptop
# for testing the rest of the pipeline must not crash.
try:
    import gpiod
    from gpiod.line import Direction, Value
except ImportError:
    gpiod = None

CHIP_PATH = "/dev/gpiochip0"
DIR_X, STEP_X = 62, 68
DIR_Y, STEP_Y = 81, 69
STEPS_PER_MM = 80.0
STEP_DELAY_S = 0.0005

req = None
current_x_steps = 0
current_y_steps = 0


def connect():
    """Opens the GPIO chip and claims the 4 motor pins. Call this once
    before write_trajectory()."""
    global req
    if gpiod is None:
        raise RuntimeError("gpiod not available -- this isn't the Odroid")
    req = gpiod.request_lines(
        CHIP_PATH, consumer="handwriting-robot",
        config={
            DIR_X: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE),
            STEP_X: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE),
            DIR_Y: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE),
            STEP_Y: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE),
        },
    )


def _pulse(step_pin):
    req.set_value(step_pin, Value.ACTIVE)
    time.sleep(STEP_DELAY_S)
    req.set_value(step_pin, Value.INACTIVE)
    time.sleep(STEP_DELAY_S)


def move_to(target_x_mm, target_y_mm):
    """Moves in a straight line to an absolute mm position, using
    Bresenham's line algorithm -- issues the right mix of X and Y step
    pulses so the pen travels in a straight diagonal, not a staircase."""
    global current_x_steps, current_y_steps
    target_x = round(target_x_mm * STEPS_PER_MM)
    target_y = round(target_y_mm * STEPS_PER_MM)
    dx, dy = target_x - current_x_steps, target_y - current_y_steps

    req.set_value(DIR_X, Value.ACTIVE if dx >= 0 else Value.INACTIVE)
    req.set_value(DIR_Y, Value.ACTIVE if dy >= 0 else Value.INACTIVE)
    steps_x, steps_y = abs(dx), abs(dy)
    sign_x = 1 if dx >= 0 else -1
    sign_y = 1 if dy >= 0 else -1

    if steps_x >= steps_y:
        error = steps_x // 2
        for _ in range(steps_x):
            _pulse(STEP_X)
            current_x_steps += sign_x
            error -= steps_y
            if error < 0:
                _pulse(STEP_Y)
                current_y_steps += sign_y
                error += steps_x
    else:
        error = steps_y // 2
        for _ in range(steps_y):
            _pulse(STEP_Y)
            current_y_steps += sign_y
            error -= steps_x
            if error < 0:
                _pulse(STEP_X)
                current_x_steps += sign_x
                error += steps_y


def write_trajectory(points, origin_x_mm, origin_y_mm, mm_per_pixel=0.2):
    """Moves the pen through every point of one character's trajectory,
    offset to start at (origin_x_mm, origin_y_mm) on the page."""
    for px, py in points:
        move_to(origin_x_mm + px * mm_per_pixel, origin_y_mm + py * mm_per_pixel)
