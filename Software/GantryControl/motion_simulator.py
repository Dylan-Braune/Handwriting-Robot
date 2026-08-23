"""
motion_simulator.py

Software-only stand-in for the ESP32: replays the exact same MOVE/PEN blocks
motion_planner.py would serialize and send over serial, using the identical
Bresenham tick logic the real firmware uses (see esp32_firmware/motion_
executor/motion_executor.ino) -- so what you see plotted here is what the
gantry will actually draw, not an idealized version of it.

Lets you check the pipeline end-to-end (Douglas-Peucker -> steps -> Bresenham
-> reconstructed path) against your original input strokes before any
hardware is involved, and reports the actual measured error against the
0.5mm spec instead of just the theoretical budget in
estimate_worst_case_error_mm().
"""

import math

from motion_planner import MoveBlock, PenBlock, GantryConfig


def _bresenham_ticks(stepsX, stepsY):
    """Mirrors the ESP32 firmware's execution loop exactly (see
    motion_executor.ino) -- whichever axis has more steps is "major" and
    gets a step almost every tick; the other is interpolated in via the
    standard Bresenham error-accumulator so the move traces a straight line
    instead of an L-shape. Yields (dx_step, dy_step) pairs of -1/0/+1."""
    ax, ay = abs(stepsX), abs(stepsY)
    sx = 1 if stepsX > 0 else -1
    sy = 1 if stepsY > 0 else -1

    if ax >= ay:
        err = ax // 2
        for _ in range(ax):
            yield sx, 0
            err -= ay
            if err < 0:
                yield 0, sy
                err += ax
                # NOTE: the y-step above is yielded as a *separate* tick,
                # matching how a two-pin STEP/DIR driver actually works
                # (it can't step both axes in literally the same pulse
                # unless you wire them to a shared STEP line, which this
                # project doesn't). This slightly serializes diagonal
                # moves in time but not in final position.
    else:
        err = ay // 2
        for _ in range(ay):
            yield 0, sy
            err -= ax
            if err < 0:
                yield sx, 0
                err += ay


def replay_plan(plan, cfg=GantryConfig):
    """Executes the plan exactly as the ESP32 would and returns
    (path, penDownPath, finalPos) in mm, for plotting/measurement."""
    x, y = 0.0, 0.0
    penDown = False
    path = [(x, y, penDown)]

    for block in plan:
        if isinstance(block, PenBlock):
            penDown = block.down
            path.append((x, y, penDown))
        elif isinstance(block, MoveBlock):
            for dxStep, dyStep in _bresenham_ticks(block.stepsX, block.stepsY):
                x += dxStep / cfg.STEPS_PER_MM_X
                y += dyStep / cfg.STEPS_PER_MM_Y
                path.append((x, y, penDown))

    return path


def measure_error(originalStrokes, reconstructedPath):
    """For every pen-down point on the reconstructed (post Douglas-Peucker +
    step-rounding + Bresenham) path, finds the nearest point on the ORIGINAL
    input stroke and reports the max/mean deviation in mm -- the real,
    measured number to compare against the proposal's 0.5mm spec, not just
    the theoretical estimate in motion_planner.estimate_worst_case_error_mm."""
    origPoints = [p for stroke in originalStrokes for p in stroke]
    if not origPoints:
        return 0.0, 0.0

    downPoints = [(x, y) for x, y, down in reconstructedPath if down]
    if not downPoints:
        return 0.0, 0.0

    errors = []
    for x, y in downPoints:
        best = min(math.hypot(x - ox, y - oy) for ox, oy in origPoints)
        errors.append(best)

    return max(errors), sum(errors) / len(errors)


def simulate_and_plot(originalStrokes, plan, cfg=GantryConfig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = replay_plan(plan, cfg)
    maxErr, meanErr = measure_error(originalStrokes, path)

    fig, ax = plt.subplots(figsize=(8, 6))

    for stroke in originalStrokes:
        xs, ys = zip(*stroke)
        ax.plot(xs, ys, "o--", color="lightgray", linewidth=1, markersize=3, label="_nolegend_")

    downSegX, downSegY = [], []
    upSegX, upSegY = [], []
    prevDown = None
    for x, y, down in path:
        if down:
            downSegX.append(x)
            downSegY.append(y)
            upSegX.append(None)
            upSegY.append(None)
        else:
            upSegX.append(x)
            upSegY.append(y)
            downSegX.append(None)
            downSegY.append(None)

    ax.plot(downSegX, downSegY, "-", color="black", linewidth=1.2, label="pen-down (drawn)")
    ax.plot(upSegX, upSegY, ":", color="tab:blue", linewidth=0.8, label="pen-up (travel)")
    ax.plot([], [], "o--", color="lightgray", label="original input stroke")

    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Simulated gantry output  |  max error {maxErr:.3f}mm, mean {meanErr:.3f}mm  (spec: 0.5mm)")

    outPath = "motion_sim_preview.png"
    fig.savefig(outPath, dpi=150, bbox_inches="tight")
    print(f"\nSaved simulator preview to {outPath}")
    print(f"Measured max error: {maxErr:.3f} mm | mean error: {meanErr:.3f} mm | spec: 0.5 mm "
          f"({'PASS' if maxErr <= 0.5 else 'OVER BUDGET'})")
    return outPath, maxErr, meanErr
