"""
motion_planner.py

SBC-side motion planner for the handwriting gantry (FU3 in the project
proposal). Runs entirely on the single-board computer -- per your call on
the SBC/ESP32 split, the ESP32 does zero path decisions. Everything here
decides WHAT to draw and HOW FAST; the ESP32 (see esp32_firmware/) only
executes the exact step counts, directions, and timings this file hands it.

Pipeline (matches FU3.2/FU3.3/FU3.4 + the pen-mapping algorithm your first
semester report already committed to):

    strokes (mm, from your trajectory-reconstruction/DSD-style model)
        -> Douglas-Peucker simplification per stroke   [FU3.2 accuracy budget]
        -> mm waypoints -> step-space waypoints         [steps-per-mm config]
        -> Bresenham-interpolated MOVE blocks            [coordinated X/Y]
        -> PEN_UP / PEN_DOWN blocks between strokes      [FU3.3]
        -> binary frames over serial to the ESP32        [FU3.1/FU3.4]

Nothing here depends on OpenCV/numpy for the geometry -- first-principles,
same convention as the rest of this repo's CV code. numpy is only used for
the (optional) simulator's plotting, never for the planning logic itself.

WHAT YOU STILL NEED TO FILL IN (see CONFIG below): the mm-per-step figures
for your actual belt/pulley or leadscrew, and the pen-lift motor's timing.
Everything else is ready to run against the simulator with no hardware.
"""

import math
import struct


# ============================================================================
# CONFIG -- measure these on your actual gantry before trusting real output.
# ============================================================================

class GantryConfig:
    # NEMA17 full-step angle. If you add microstepping on the driver (e.g.
    # A4988/DRV8825 at 1/16), multiply STEPS_PER_REV by the microstep factor
    # and this whole file needs no other changes -- everything downstream is
    # expressed in "steps", not "full steps".
    STEP_ANGLE_DEG = 1.8
    MICROSTEPPING = 1  # e.g. 16 for 1/16 microstepping -- set to match your driver's MS pins
    STEPS_PER_REV = int(round((360.0 / STEP_ANGLE_DEG) * MICROSTEPPING))

    # TODO: measure these. For a GT2 belt on a 20-tooth pulley, belt pitch is
    # 2mm so one revolution moves the carriage 20*2 = 40mm -> MM_PER_REV=40.
    # For a leadscrew, MM_PER_REV = the screw's lead (mm advanced per turn).
    MM_PER_REV_X = 40.0
    MM_PER_REV_Y = 40.0

    STEPS_PER_MM_X = STEPS_PER_REV / MM_PER_REV_X
    STEPS_PER_MM_Y = STEPS_PER_REV / MM_PER_REV_Y

    # Feed rate while the pen is down (mm/s). Start conservative -- your
    # sketch_LocatingMaxSpeed.ino test is exactly how to find the real ceiling
    # (the RPM before the motor stalls/skips), then convert: mm/s = (RPM/60) * MM_PER_REV.
    DRAW_FEED_MM_S = 25.0
    # Pen-up travel can safely go faster since positional error there doesn't
    # matter for the 0.5mm accuracy spec.
    TRAVEL_FEED_MM_S = 60.0

    # Douglas-Peucker tolerance. This IS your accuracy budget against the
    # proposal's 0.5mm absolute positional error spec (Requirement 5) -- keep
    # it comfortably under 0.5mm since it isn't the only source of error
    # (mechanical backlash, step rounding, etc. all eat into the same budget).
    # Rule of thumb used here: leave ~40% margin.
    SIMPLIFY_EPSILON_MM = 0.25

    # Pen-lift motor: yours has no driver and just rotates ~90 degrees in one
    # direction per actuation (per your description -- a cam/eccentric flips
    # the pen height each quarter turn, driven open-loop by timing). TODO:
    # time this empirically on the real mechanism and set it here. Open-loop
    # timing WILL drift over hundreds of actuations per page -- see the note
    # in esp32_firmware/ about adding a limit switch or slotted-disc optical
    # flag so the ESP32 can confirm the flip instead of trusting a timer.
    PEN_ACTUATE_MS = 150


# ============================================================================
# Douglas-Peucker (first principles -- no shapely/rdp dependency)
# ============================================================================

def _point_segment_distance(p, a, b):
    """Perpendicular distance from point p to the line segment a-b (mm)."""
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    projX, projY = ax + t * dx, ay + t * dy
    return math.hypot(px - projX, py - projY)


def douglas_peucker(points, epsilon):
    """Classic recursive DP simplification. points: list of (x,y) mm tuples.
    Returns a reduced list of (x,y) that stays within `epsilon` mm of the
    original polyline -- this is the exact algorithm your first semester
    report already names as the intended pen-mapping technique [ref 2 in
    that report's literature review]."""
    if len(points) < 3:
        return list(points)

    start, end = points[0], points[-1]
    maxDist, maxIdx = -1.0, -1
    for i in range(1, len(points) - 1):
        d = _point_segment_distance(points[i], start, end)
        if d > maxDist:
            maxDist, maxIdx = d, i

    if maxDist > epsilon:
        left = douglas_peucker(points[:maxIdx + 1], epsilon)
        right = douglas_peucker(points[maxIdx:], epsilon)
        return left[:-1] + right
    return [start, end]


# ============================================================================
# Motion blocks: SBC decides everything, ESP32 just executes
# ============================================================================

class MoveBlock:
    """One coordinated straight-line move. stepsX/stepsY are SIGNED (sign =
    direction). stepIntervalUs is the tick period for whichever axis takes
    MORE steps (the "major" axis) -- the minor axis is Bresenham-interpolated
    against it. This is the same decomposition GRBL/Marlin use; the
    difference here is the SBC computes the block, not the microcontroller."""
    __slots__ = ("stepsX", "stepsY", "stepIntervalUs")

    def __init__(self, stepsX, stepsY, stepIntervalUs):
        self.stepsX = stepsX
        self.stepsY = stepsY
        self.stepIntervalUs = stepIntervalUs


class PenBlock:
    __slots__ = ("down",)

    def __init__(self, down):
        self.down = down  # True = pen down, False = pen up


def _mm_to_steps_with_carry(dxMm, dyMm, cfg, carry):
    """Converts an mm delta to an integer step delta WITHOUT resetting the
    rounding remainder every segment. Necessary because Douglas-Peucker
    produces many short segments -- naive per-segment round() would let sub-
    step rounding error accumulate into visible drift over a whole word.
    `carry` is a mutable [carryX, carryY] in mm, updated in place."""
    exactX = dxMm * cfg.STEPS_PER_MM_X + carry[0]
    exactY = dyMm * cfg.STEPS_PER_MM_Y + carry[1]
    stepsX = int(round(exactX))
    stepsY = int(round(exactY))
    carry[0] = exactX - stepsX
    carry[1] = exactY - stepsY
    return stepsX, stepsY


def build_motion_plan(strokes, cfg=GantryConfig):
    """strokes: list of strokes, each a list of (x_mm, y_mm) points, already
    in gantry coordinates (i.e. your trajectory source -- reconstructed
    stroke order from the scanned page, or a DSD-style model output -- has
    already placed them where they should land on the physical page). Pen is
    implicitly up between strokes and down within one.

    Returns a flat list of MoveBlock / PenBlock ready to serialize."""
    blocks = []
    carry = [0.0, 0.0]
    cursor = (0.0, 0.0)

    for stroke in strokes:
        if len(stroke) < 2:
            continue
        simplified = douglas_peucker(stroke, cfg.SIMPLIFY_EPSILON_MM)

        # Travel move (pen up) from wherever we are to the start of this stroke
        blocks.append(PenBlock(down=False))
        blocks.extend(_segment_to_blocks(cursor, simplified[0], cfg.TRAVEL_FEED_MM_S, cfg, carry))
        blocks.append(PenBlock(down=True))

        for i in range(len(simplified) - 1):
            blocks.extend(_segment_to_blocks(simplified[i], simplified[i + 1], cfg.DRAW_FEED_MM_S, cfg, carry))

        cursor = simplified[-1]

    blocks.append(PenBlock(down=False))
    return blocks


def _segment_to_blocks(a, b, feedMmS, cfg, carry):
    dx, dy = b[0] - a[0], b[1] - a[1]
    stepsX, stepsY = _mm_to_steps_with_carry(dx, dy, cfg, carry)
    if stepsX == 0 and stepsY == 0:
        return []

    distMm = math.hypot(dx, dy)
    timeS = distMm / feedMmS if feedMmS > 0 else 0.0
    majorSteps = max(abs(stepsX), abs(stepsY), 1)
    stepIntervalUs = max(1, int(round((timeS * 1_000_000) / majorSteps)))
    return [MoveBlock(stepsX, stepsY, stepIntervalUs)]


# ============================================================================
# Serial protocol -- fixed-size binary frames, ESP32 side just deserializes
# ============================================================================
#
# Frame layout (little-endian, matches struct format below):
#   byte    0      : command  (0x01=MOVE, 0x02=PEN, 0x03=END)
#   MOVE payload   : int32 stepsX, int32 stepsY, uint32 stepIntervalUs   (9 bytes)
#   PEN payload    : uint8 down (1 or 0)                                  (1 byte)
#   END payload    : (none)
#   last byte      : XOR checksum of every byte before it (incl. command)
#
# Kept as fixed, simple structs on purpose -- there is no "planning" left
# to do on receipt, only "which struct is this and what do the fields say",
# which is why this satisfies your "no processing on the ESP32" constraint.

CMD_MOVE = 0x01
CMD_PEN = 0x02
CMD_END = 0x03


def _checksum(payload_bytes):
    c = 0
    for b in payload_bytes:
        c ^= b
    return c


def serialize_block(block):
    if isinstance(block, MoveBlock):
        body = bytes([CMD_MOVE]) + struct.pack("<iiI", block.stepsX, block.stepsY, block.stepIntervalUs)
    elif isinstance(block, PenBlock):
        body = bytes([CMD_PEN, 1 if block.down else 0])
    else:
        raise TypeError(f"Unknown block type: {type(block)}")
    return body + bytes([_checksum(body)])


def serialize_plan(blocks):
    """Concatenates every block's frame plus a trailing END frame. This is
    what actually gets written to the serial port to the ESP32."""
    out = bytearray()
    for block in blocks:
        out += serialize_block(block)
    endBody = bytes([CMD_END])
    out += endBody + bytes([_checksum(endBody)])
    return bytes(out)


# ============================================================================
# Sanity checks you can run right now, no hardware required
# ============================================================================

def estimate_worst_case_error_mm(cfg=GantryConfig):
    """Two independent error sources stack: the DP simplification tolerance,
    and the +-0.5 step rounding error from _mm_to_steps_with_carry (bounded
    because of the carry term, not just per-segment). Reports both against
    the proposal's 0.5mm spec (Requirement 5) so you can see the margin."""
    stepErrX = 0.5 / cfg.STEPS_PER_MM_X
    stepErrY = 0.5 / cfg.STEPS_PER_MM_Y
    total = cfg.SIMPLIFY_EPSILON_MM + max(stepErrX, stepErrY)
    print(f"DP simplification tolerance : {cfg.SIMPLIFY_EPSILON_MM:.3f} mm")
    print(f"Worst-case step rounding    : {max(stepErrX, stepErrY):.3f} mm "
          f"(steps/mm X={cfg.STEPS_PER_MM_X:.2f}, Y={cfg.STEPS_PER_MM_Y:.2f})")
    print(f"Combined worst case         : {total:.3f} mm  (spec: 0.5 mm)")
    if total > 0.5:
        print("!! Over budget -- tighten SIMPLIFY_EPSILON_MM or increase MICROSTEPPING.")
    else:
        print(f"OK -- {0.5 - total:.3f} mm of margin left for mechanical backlash etc.")


if __name__ == "__main__":
    print("=== motion_planner.py self-check (no hardware needed) ===\n")
    estimate_worst_case_error_mm()

    print("\nBuilding a demo plan for a simple triangle stroke...")
    demoStrokes = [[(0.0, 0.0), (10.0, 0.0), (5.0, 8.0), (0.0, 0.0)]]
    plan = build_motion_plan(demoStrokes)
    moveCount = sum(1 for b in plan if isinstance(b, MoveBlock))
    penCount = sum(1 for b in plan if isinstance(b, PenBlock))
    print(f"Generated {len(plan)} blocks ({moveCount} moves, {penCount} pen actions)")

    raw = serialize_plan(plan)
    print(f"Serialized size: {len(raw)} bytes -- this is exactly what gets written to the ESP32's serial port")

    choice = input("\nRun the matplotlib visual simulator on this demo triangle? [Y/n]: ").strip().lower()
    if choice not in ("n", "no"):
        from motion_simulator import simulate_and_plot
        simulate_and_plot(demoStrokes, plan)
