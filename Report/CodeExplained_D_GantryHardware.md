# Code Explained — Block D: Gantry Hardware Control

**File:** `Software/GantryControl/odroid_direct_drive.py` (816 lines)

## Role in the pipeline

This module is the fourth and final functional block of the system, sitting
downstream of everything else the project does (handwriting-style synthesis,
glyph/stroke generation, and G-code generation from the synthesised image).
It receives a finished G-code file — a plain-text sequence of plotter
instructions describing where the pen should travel and when it should be
up or down — and is responsible for turning that abstract instruction stream
into physical motion: two NEMA-class stepper motors driving an X/Y gantry,
and a one-way geared cam motor that lifts and lowers a pen.

Architecturally this is a **direct-GPIO bit-banging driver**: there is no
microcontroller (no Arduino, no GRBL board) between the software and the
motors. The ODROID N2+ single-board computer's own GPIO pins are toggled
directly from Python via the Linux `gpiod` character-device API, with STEP
and DIR lines driven in tight software loops and end-stop (limit switch)
pins polled in the very same loop that drives the motors. This is a
deliberate design choice: because the end-stop check and the step pulse are
issued from the same thread without anything else in between, "stop
immediately on any end-stop trigger" is a hard real-time guarantee relative
to the loop's own timing, not a best-effort check that could be delayed by
a separate controller's command queue or serial buffering.

The module exposes three ways to be driven: (1) an interactive line-based
CLI over stdin for manual testing/jogging/calibration, (2) a direct
Python-level entry point (`write_gcode_file`) that the project's web
server calls to run a finished G-code file end-to-end, and (3) a free-running
circle demo and raw-RPM test mode used for isolated hardware testing without
needing a real G-code file. All three ultimately fall through to the same
small set of low-level primitives: `bresenham_move` for coordinated
two-axis stepping, `calibrate`/`_calibrate_axis`/`_step_axis` for homing,
and `set_pen` for the pen-lift actuator.

---

## Module-level setup and constants

### `gpiod` import guard (lines 46–56)

**Purpose:** Allow this file to be imported on machines that are not the
ODROID (e.g. the web server process, or a developer's laptop) without
crashing, while still giving a clean, catchable error the moment hardware
access is actually attempted.

**Mechanism:** The import of `gpiod` (and `Direction`/`Value` from
`gpiod.line`) is wrapped in a `try/except ImportError`. On failure, `gpiod`,
`Direction`, and `Value` are all set to `None`, and the original exception
is stashed in `_GPIOD_IMPORT_ERROR`. No stub/mock GPIO *class* is
implemented in this file — the fallback is simply "the names are `None`".
Nothing dereferences them until `connect()` is called, and `connect()`
explicitly checks `if gpiod is None` and raises `GantryNotConnectedError`
with the original import error message embedded, rather than letting
import machinery raise `AttributeError`/`NameError` deep inside hardware
code. This means the "off-hardware" story for this file is: **the module
degrades gracefully by refusing hardware operations with a clear
exception, not by emulating a fake plotter.** Anything that needs to test
G-code parsing, Bresenham stepping maths, or coordinate transforms off
real hardware must do so by calling the pure-logic functions directly
(e.g. `parse_gcode_line`, `bresenham_move`'s arithmetic) or by monkey-
patching `req`/`gpiod` in a test harness — there is no built-in simulated
line request object. This is flagged again in the Limitations section
below, since it affects how the report should describe testing
methodology.

**Literature connection:** Standard defensive-import / optional-hardware-
dependency pattern; no canonical citation, this is conventional Python
engineering practice.

### Custom exceptions

#### `GantryNotConnectedError` (line 59)
**Purpose:** A single, specific exception type that every hardware-access
failure path raises, so callers (CLI, web server) can catch one type
instead of a bare `Exception` and still report something meaningful to the
user. Raised by `connect()` for both "gpiod not importable" and "chip open
failed" cases.

#### `NotCalibratedError` (line 65)
**Purpose:** Raised by `run_gcode_file()` if a G-code job is attempted
before `calibrate()` has succeeded in the current process lifetime.
**Design rationale (from the docstring):** this state is *deliberately
never persisted to disk*. A mechanical bump, re-homing by hand, or a power
cycle between sessions would silently invalidate a previously-saved
steps-per-mm/travel measurement, and a stale calibration is more dangerous
than no calibration (it would let the gantry run full-speed into a wall
without the second line of defence `_clamp_to_usable_area` even knowing
anything is wrong). Forcing recalibration every process start trades
convenience for safety.

### Pin assignments and physical wiring constants (lines 71–117)

These are not functions but are essential to document because every
function below operates in terms of them:

- `CHIP_PATH = "/dev/gpiochip0"` — the Linux `gpiod` character device
  representing the ODROID N2+'s GPIO bank.
- `DIR1_PIN, STEP1_PIN = 64, 68` (X motor), `DIR2_PIN, STEP2_PIN = 81, 69`
  (Y motor) — BCM/offset-style line numbers within that chip, annotated
  with the corresponding physical header pin numbers (7/11 and 12/13) in
  the comments.
- `X_MIN_PIN, X_MAX_PIN, Y_MIN_PIN, Y_MAX_PIN` — four travel-limit switches,
  one per axis end, **active-HIGH** (3.3 V when triggered) via external
  10 kΩ pull-down resistors in hardware (not configured in software).
- `PEN_SWITCH_PIN` — a feedback-only input; HIGH = pen fully up, LOW = pen
  fully down. This is read, never driven.
- `PEN_MOTOR_PIN` — a digital output driving a cam/eccentric mechanism.
  Critically, per the module docstring, this motor is a **toggle
  actuator**, not a position actuator: running it moves the pen from
  whichever state it is currently in to the other state (it does not
  accept "go to position X"). This directly explains why `set_pen()`
  (below) must poll `PEN_SWITCH_PIN` as a stopping condition rather than
  running the motor for a fixed, pre-computed duration.
- `INVERT_X = True`, `INVERT_Y = True` — per-axis booleans that flip the
  meaning of "positive direction" to compensate for whichever way the
  belts/pulleys happen to be mechanically mounted. The comment is explicit
  that these only matter for *drawing* direction; `calibrate()` is
  direction-agnostic and never needs them to be correct, because it always
  measures "whichever switch it hits first" rather than assuming X+ means
  "toward X_MAX."

### Kinematics constants (lines 91–96)

- `MICROSTEPS = 16`, `STEPS_PER_REV = 200 * MICROSTEPS = 3200` — a 1.8°/step
  (200 steps/rev) motor driven at 16× microstepping, giving 3200 "micro-
  steps" per full mechanical revolution.
- `STEPS_PER_MM = 80.0` — a nominal/fallback linear conversion factor
  (steps of motor rotation per millimetre of carriage travel, determined
  by belt pitch and pulley tooth count), used only *before* `calibrate()`
  has run for the session. This is explicitly a provisional value; the
  real, trusted steps/mm value per axis is derived empirically inside
  `calibrate()` and stored in the `calibration` dict.
- `CIRCLE_STEP_DELAY_S = 0.000002` — a 2 µs default step-pulse half-period
  used as a floor/default in the Bresenham line-stepper, far faster than
  any real feed rate; it functions as a safety floor so computed delays
  never divide down to an unrealistically short period.
- `DEFAULT_CIRCLE_RADIUS_MM = 8.0` — default radius for the built-in test
  "draw a circle" demo mode.

### Calibration constants (lines 98–117)

- `X_SWITCH_TRAVEL_MM = 204.0`, `Y_SWITCH_TRAVEL_MM = 262.0` — physically
  *measured* (not derived) switch-to-switch travel distances for each axis,
  hard-coded from a one-time manual measurement of the specific physical
  build. **This is a per-build constant** — see Limitations.
- `EDGE_TOLERANCE_MM = 5.0` — a safety margin kept clear of both end-stops
  on both axes; the "usable drawing area" is the full switch-to-switch
  travel minus `2 * EDGE_TOLERANCE_MM` (margin at both ends).
- `HOMING_STEP_DELAY_S = 0.0008` — 0.8 ms per-half-pulse delay used only
  while homing (a conservative, slow, reliable speed — distinct from the
  much faster drawing-speed delays computed per G-code move).
- `HOMING_MAX_STEPS = 200_000` — a hard ceiling on how many steps a homing
  move may take before it's treated as a wiring/mechanical fault (switch
  never triggered) rather than running forever.
- `ENDSTOP_DEBOUNCE_S = 0.006`, `ENDSTOP_CONFIRM_SAMPLES = 3` — software
  debounce parameters for all end-stop reads (used by `triggered_endstop`,
  documented below). The comment is explicit about *why*: the motors have
  no flyback diodes and share a ground return, which is stated to
  electrically inject noise capable of holding a switch input HIGH for
  several milliseconds — long enough to fool a single-recheck debounce.
  Three confirmations spaced 6 ms apart (18 ms total worst case) are used
  instead.
- `RETRACT_MM = 5.0` — distance stepped back off a tripped switch during
  emergency-stop recovery, so the switch is not left physically latched
  HIGH (which would otherwise make `r` unable to ever clear the estop,
  since `triggered_endstop()` would immediately report it as still
  triggered the instant the estop flag was cleared).

### `calibration` dict and runtime globals (lines 119–143)

A single mutable dict holds the live calibration result
(`done`, `steps_per_mm_x`, `steps_per_mm_y`, `usable_width_mm`,
`usable_height_mm`), defaulting to the nominal `STEPS_PER_MM` and the raw
switch-travel constants until `calibrate()` overwrites them with measured
values. A block of module-level globals (`req`, `current_mode`,
`target_rpm`, `step_delay_s`, `running`, `current_x_steps`,
`current_y_steps`, `theta`, circle-demo state, `estopped`,
`tripped_switch`) hold all mutable runtime state — this file uses a
module-as-singleton style rather than a class, appropriate for a
single-process, single-gantry controller.

---

## Connection management

### `connect()` (lines 149–178)

**Purpose:** Open the `gpiod` character device and claim every pin used by
this module in one atomic request, configuring each as INPUT or OUTPUT
with safe initial values.

**Inputs:** None (operates on module-level pin constants).

**Outputs:** Side effect only — sets the module-global `req` to a live
`gpiod` line-request object, or raises. No return value.

**Inner workings:** Idempotent guard (`if req is not None: return`) makes
it safe to call repeatedly (e.g. from both the CLI entry point and the web
server's `write_gcode_file`). If `gpiod` failed to import, raises
`GantryNotConnectedError` immediately with the stashed import error.
Otherwise calls `gpiod.request_lines(...)` with a `config` dict mapping
each pin to a `gpiod.LineSettings`: `DIR1_PIN`/`DIR2_PIN` default to
OUTPUT with initial value `Value.ACTIVE`, `STEP1_PIN`/`STEP2_PIN` and
`PEN_MOTOR_PIN` default to OUTPUT with initial value `Value.INACTIVE`
(motors must not move or run the instant the pins are claimed), and the
four end-stop pins plus the pen switch are INPUT. Any exception during the
request (chip busy, permission denied, wrong path) is caught, `req` is
reset to `None`, and re-raised as `GantryNotConnectedError` with the
original exception chained via `raise ... from e` (preserves the original
traceback for debugging while presenting one exception type to callers).
Finally calls `calculate_delay(target_rpm)` to initialise `step_delay_s`
consistent with whatever `target_rpm` happens to be (0 at startup, per the
global's comment).

**Literature connection:** Standard resource-acquisition pattern
(try/except/raise-chained); no named algorithm.

### `disconnect()` (lines 181–192)

**Purpose:** Safely release the GPIO lines, ensuring motors are commanded
off *before* the line request is released (so the hardware doesn't briefly
float/glitch mid-shutdown).

**Inputs:** None. **Outputs:** Side effect — sets `STEP1_PIN`, `STEP2_PIN`,
`PEN_MOTOR_PIN` to `Value.INACTIVE`, then calls `req.release()`, then sets
`req = None`. Guarded by `if req is None: return` so it's safe to call even
if never connected. The motor-off writes are wrapped in `try/...finally:
req.release()` so that even if setting a value somehow raises, the line
request is still released and `req` is still cleared — the release always
happens.

### `is_connected()` (lines 195–196)

**Purpose:** Trivial predicate, `return req is not None`, used by callers
that want to check connection state without triggering a `connect()`
attempt.

---

## Low-level motion primitives

### `calculate_delay(rpm)` (lines 202–207)

**Purpose:** Converts a desired motor speed in RPM into the per-half-step
delay used by the raw RPM test mode in the main loop.

**Inputs:** `rpm: float` — signed revolutions-per-minute target (sign
only affects direction elsewhere; this function uses `abs(rpm)`).

**Outputs:** Side effect — writes the module-global `step_delay_s`
(seconds, float).

**Inner workings / maths:** If `abs(rpm) < 0.1`, treated as stopped:
`step_delay_s = 0`. Otherwise:

```
step_delay_s = (60.0 / (|rpm| * STEPS_PER_REV)) / 2.0
```

Derivation: at `rpm` revolutions per minute, one revolution takes
`60/rpm` seconds, and one revolution is `STEPS_PER_REV` steps, so the time
per *full step cycle* (one STEP pulse HIGH-then-LOW cycle) is
`60 / (rpm * STEPS_PER_REV)` seconds. Each step cycle in this code is
implemented as two separate `time.sleep(step_delay_s)` calls (once with
the pin HIGH, once with it LOW — see the main loop's RPM branch), so the
delay *per half-cycle* is that value divided by 2. This is a textbook
open-loop stepper-timing formula; there is no acceleration ramping here,
it's a fixed-rate square wave for as long as that RPM is requested.

**Literature connection:** Standard embedded motor-control technique
(open-loop constant-rate step-pulse generation), no single canonical
citation — this is conventional stepper-driver arithmetic, not a named
published algorithm.

### `triggered_endstop(ignore=frozenset())` (lines 210–234)

**Purpose:** The single, shared, debounced "is anything dangerous
currently triggered" check used by *every* motion primitive in the file
(homing, drawing, RPM mode, pen actuation). Centralising this in one
function is what gives the module its "checked in the same loop as every
step pulse" real-time safety property described in the module docstring.

**Inputs:** `ignore: frozenset[str]` — names of end-stops (keys of
`ENDSTOP_PINS`) to skip entirely during this scan, not merely to filter
from the result. The docstring is explicit about why this distinction
matters: skipping means a different, independently and legitimately
triggered switch elsewhere (e.g. the other axis still resting at its own
limit after homing) can never mask detection of a *new*, different switch
trip later in the same scan.

**Outputs:** Returns the `str` name of the first confirmed-triggered
switch not in `ignore` (iteration order follows `ENDSTOP_PINS` dict
order: `"X_MIN", "X_MAX", "Y_MIN", "Y_MAX"`), or `None` if none are
triggered. No side effects beyond the debounce sleeps.

**Inner workings / maths — debounce state machine:** For each
non-ignored pin: read it once (`req.get_value(pin)`); if not
`Value.ACTIVE`, move to the next pin immediately (zero added latency for
the common "all clear" case — the costly debounce loop below only runs
once a pin is seen HIGH at all). If it *is* ACTIVE, enter a confirmation
loop that runs `ENDSTOP_CONFIRM_SAMPLES - 1` (= 2) more times: sleep
`ENDSTOP_DEBOUNCE_S` (6 ms), re-read the pin, and `break` out of the
`for...else` construct immediately if it ever reads non-ACTIVE (meaning
the initial HIGH was noise, not a real trigger). If the loop completes
all iterations without breaking (Python's `for...else: else` clause fires
only when the loop was *not* broken), the pin is trusted as genuinely
triggered and its name is returned. This is a classic "N consecutive
samples must agree" software debounce, parametrised as
3 samples × 6 ms = up to 18 ms worst-case confirmation latency.

**Literature connection:** Standard digital-input debouncing technique
(majority-of-N-consecutive-samples), a conventional embedded-systems
practice — no single canonical citation.

### `_all_triggered_endstops()` (lines 237–242)

**Purpose:** Unlike `triggered_endstop`, which returns only the *first*
match, this returns *every* currently-triggered switch name — needed
because more than one can be legitimately active simultaneously (e.g. the
gantry resting in a corner where both an X and a Y switch are pressed at
once).

**Inputs:** None. **Outputs:** `list[str]` of triggered switch names.

**Inner workings:** A list comprehension over `ENDSTOP_PINS`, calling
`triggered_endstop` once per candidate name with every *other* name
ignored (`ignore=set(ENDSTOP_PINS) - {name}`), keeping `name` only if that
targeted, single-switch check confirms it. This reuses the same debounce
logic rather than duplicating it, at the cost of running the full debounce
procedure once per switch (acceptable since this is only called during
the already-slow emergency-stop recovery path, not the hot step loop).

### `axis_dir_value(invert, positive)` (lines 245–252)

**Purpose:** The single source of truth translating "I want this axis to
move in the logical positive/negative direction" into the actual
`Value.ACTIVE`/`Value.INACTIVE` to write to a DIR pin, respecting that
axis's `INVERT_X`/`INVERT_Y` flag. Every direction decision anywhere in
the file (drawing moves, homing, RPM test mode) routes through this one
function so they can never disagree about which physical direction a
given DIR value produces.

**Inputs:** `invert: bool` (the axis's invert flag), `positive: bool`
(whether the caller wants the logical-positive direction).

**Outputs:** Returns `Value.ACTIVE` or `Value.INACTIVE` (pure function, no
side effects).

**Inner workings / maths:**

```
forward = positive != invert     # boolean XOR
return Value.ACTIVE if forward else Value.INACTIVE
```

`!=` on two Python bools implements exclusive-or: the DIR pin is ACTIVE
exactly when "requested positive" and "axis inverted" disagree (one true,
one false), which is precisely the XOR relationship needed so that
flipping `invert` flips the physical meaning of `Value.ACTIVE` without
needing to change call sites.

### `_RETREAT_SIGN` dict (line 257)

Not a function, but a necessary lookup table: maps each end-stop name to
`+1` or `-1`, i.e. which direction (in the same sign convention as
`current_x_steps`/`current_y_steps`, positive = away from MIN toward MAX)
constitutes "retreating" from that specific switch. `X_MIN`→`+1` (move
positive/away), `X_MAX`→`-1`, `Y_MIN`→`+1`, `Y_MAX`→`-1`.

### `_retract_off_switch(hit, ignore=frozenset())` (lines 260–290)

**Purpose:** Step a single axis a fixed `RETRACT_MM` distance away from a
specific end-stop it just hit, so the switch is not left physically
latched HIGH after a stop. Used both by `calibrate()` (after finding each
switch) and by `_trigger_emergency_stop` (after an unplanned trip).

**Inputs:** `hit: str` — the end-stop name just triggered (e.g.
`"X_MAX"`). `ignore: frozenset[str]` — other switches expected to still
read triggered right now and which must not be mistaken for a *new*
fault during the retract (e.g. the other axis, mid-recovery from a corner
hit).

**Outputs:** Side effects — pulses the relevant STEP pin and updates
`current_x_steps` or `current_y_steps` by the signed number of steps
actually moved. No return value.

**Inner workings / maths:** Determines which axis (`is_x =
hit.startswith("X")`), looks up the retreat sign from `_RETREAT_SIGN`,
selects that axis's DIR/STEP pins and invert flag, and computes:

```
target_steps = max(1, round(RETRACT_MM * steps_per_mm))
```

using the *current calibrated* steps/mm for that axis (falls back to the
nominal `STEPS_PER_MM` value in `calibration` if calibration hasn't run
yet). Sets the DIR pin via `axis_dir_value(invert, sign > 0)`, then loops
up to `target_steps` times: before each step, checks
`triggered_endstop(ignore={hit} | set(ignore))` — if some *other*,
unexpected switch trips mid-retract, the loop breaks early without
raising (treated as a genuine mechanical problem the caller should see
via the switch still being tripped, not pushed through). Each iteration
pulses the STEP pin ACTIVE, sleeps `HOMING_STEP_DELAY_S`, sets it
INACTIVE, sleeps `HOMING_STEP_DELAY_S` again (a symmetric square-wave
pulse), and accumulates `moved += sign`. After the loop, `current_x_steps`
or `current_y_steps` is incremented by the actual signed `moved` count
(which may be less than `target_steps` if it broke early) — this keeps
the tracked position consistent with reality even on early exit.

### `_trigger_emergency_stop(hit)` (lines 293–312)

**Purpose:** Idempotently latch the system into an emergency-stop state
the moment any end-stop trips unexpectedly, immediately halt both
motors, and retract off *every* currently-triggered switch (not only the
one originally detected) so the gantry isn't left mechanically wedged
against a corner requiring a manual power-cycle.

**Inputs:** `hit: str` — the end-stop name that triggered detection.

**Outputs:** Side effects — sets globals `estopped = True`,
`tripped_switch = hit`; writes `Value.INACTIVE` to `STEP1_PIN` and
`STEP2_PIN` immediately (motors stop stepping at once); prints status
messages; calls `_retract_off_switch` for each currently-triggered
switch.

**Inner workings:** Guard `if estopped: return` makes this safe to call
from multiple places (the main loop and a running G-code job) without
double-triggering. Computes `all_hits = _all_triggered_endstops() or
[hit]` — i.e. re-scans for every genuinely triggered switch right now
(handles the corner case of two switches active simultaneously), falling
back to just `[hit]` if the re-scan somehow finds nothing (a switch that
bounced back below threshold in the interim). For each triggered switch
`h`, calls `_retract_off_switch(h, ignore=set(all_hits) - {h})` so the
other simultaneously-triggered switches are not treated as new faults
during each individual retract.

**Literature connection:** This is a standard "software emergency-stop /
limit-switch interlock" pattern common in CNC/plotter firmware (e.g. the
limit-switch handling found in GRBL-style controllers); no single
academic citation applies, but it is worth citing as a widely-used
convention in open-source CNC firmware if the report wants a comparison
point.

---

## `bresenham_move(dx, dy, step_delay_s=CIRCLE_STEP_DELAY_S)` — the key algorithm (lines 315–355)

**Purpose:** Given a desired integer step displacement on each axis
simultaneously, step both motors so that the combined path traces as
straight a line as possible between the two axes, rather than one axis
finishing all its steps first and then the other axis finishing
separately (which would produce an L-shaped, not diagonal, path).

**Inputs:** `dx: int`, `dy: int` — signed step deltas for X and Y
respectively (not millimetres — these are already in motor microsteps).
`step_delay_s: float` — the half-pulse delay (seconds) to use for *every*
step pulse issued during this move; defaults to the fast
`CIRCLE_STEP_DELAY_S` constant but is overridden by G-code moves with a
feed-rate-derived value (see `_gcode_linear_move` below).

**Outputs:** Side effects only. Sets `DIR1_PIN`/`DIR2_PIN` once at the
start according to the sign of `dx`/`dy`; pulses `STEP1_PIN`/`STEP2_PIN`
some number of times; updates `current_x_steps`/`current_y_steps` as it
goes (so the tracked absolute position is always correct even on an
early abort). Returns early (no return value, no exception) without
completing the commanded delta if an end-stop trips mid-move — the
caller's own loop is documented as responsible for reacting to
`estopped`/`tripped_switch` state afterward.

### The maths: Bresenham's line algorithm, exactly as implemented

This is the single most important algorithm in the file and deserves full
derivation.

**The problem it solves:** two independent stepper motors must each emit
a whole number of discrete step pulses, but together they must trace a
diagonal line whose slope is, in general, not a ratio the two motors can
represent exactly step-for-step (e.g. 7 steps on X for every 3 on Y).
Naively stepping axis X seven times then axis Y three times would draw a
staircase with one long horizontal segment followed by one long vertical
segment — visibly not a straight line. The motors must be *interleaved*
so that, at every point along the move, the ratio of steps-so-far on each
axis approximates the ratio of the full deltas as closely as integer
arithmetic allows. This is exactly the problem Bresenham's algorithm was
invented to solve.

**Decision-variable setup, as implemented:**

```python
ax, ay = abs(dx), abs(dy)
sx = 1 if dx >= 0 else -1
sy = 1 if dy >= 0 else -1
```

`ax`, `ay` are the unsigned magnitudes; `sx`, `sy` are the direction
signs, applied once to the DIR pins and then used only to increment the
position counters.

The code branches on which axis has the larger magnitude — this is the
"major axis" (the axis that must step once on every iteration; the other
is the "minor axis" that only steps on some iterations):

**Case `ax >= ay` (X is major axis):**

```python
err = ax // 2
for _ in range(ax):
    pulse(STEP1_PIN)              # X always steps
    current_x_steps += sx
    err -= ay
    if err < 0:
        pulse(STEP2_PIN)          # Y steps only when the error rolls over
        current_y_steps += sy
        err += ax
```

**Case `ay > ax` (Y is major axis)** is the exact mirror image, with the
roles of X and Y swapped.

This is the standard integer incremental-error form of Bresenham's line
algorithm. The accumulator `err` represents (twice) the perpendicular
distance of the "ideal" continuous line from the current integer step
position, tracked *without ever using floating-point division*. On every
major-axis step, `err` is decremented by the minor-axis delta (`ay`). When
`err` goes negative, it means the accumulated error has grown large enough
that the line has drifted more than half a minor-step away from the
stepped path, so a compensating minor-axis step is taken, and `err` is
corrected back upward by the major-axis delta (`ax`). This keeps the error
term bounded to roughly `±ax/2`, i.e. the actual stepped path never
deviates from the ideal straight line by more than half a step in the
minor axis — the theoretically optimal result for an integer stepper
algorithm. The initial `err = ax // 2` (rather than starting at 0) centres
the error term so deviation is distributed symmetrically about the ideal
line rather than biased to one side at the start of the move.

Each iteration also calls `triggered_endstop()` as the very first action,
*before* issuing any pulse — meaning the loop aborts mid-move (via
`return`) the instant an end-stop is detected, never issuing a stepper
pulse past that point in the same pass. The inner helper `pulse(step_pin)`
sets the pin `Value.ACTIVE`, sleeps `step_delay_s`, then sets it
`Value.INACTIVE` — note that, unlike `_retract_off_switch`'s symmetric
pulse, here only one `step_delay_s` sleep happens per edge inside
`pulse`, but `pulse` is called up to twice per loop iteration (once for
the major axis, optionally once for the minor axis) with no additional
delay between them; the DIR pins are only touched once, outside the loop,
since a single call never changes direction mid-move.

**Why Bresenham is the correct choice here — literature connection:**
This is a textbook application of Bresenham's algorithm, originally
published as **Bresenham, J.E. (1965), "Algorithm for computer control of
a digital plotter," IBM Systems Journal, 4(1), 25–30**. This citation is
unusually apt for this report: Bresenham's original paper was *literally*
about driving a two-axis digital pen plotter — the same physical problem
this code solves (two independently-steppable axes must trace a straight
line to a pen on paper), decades before it became famous as a general
computer-graphics line-rasterisation technique. The report's literature
review can cite this paper directly and accurately as the origin of the
exact technique used to coordinate the X and Y stepper motors in this
project, rather than needing to reach for a generic "computer graphics"
citation.

---

## Pen control

### `pen_is_up()` (lines 366–368)

**Purpose:** Read the physical pen-position feedback switch.

**Inputs:** None. **Outputs:** `bool` — `True` if `PEN_SWITCH_PIN` reads
`Value.ACTIVE` (per the wiring docstring: HIGH = pen fully retracted/up),
`False` otherwise (LOW = pen down). Pure read, no side effects, no
debounce (this switch is not treated as safety-critical the way the
end-stops are, so it is read raw).

### `set_pen(target_up, timeout_s=3.0)` (lines 371–397)

**Purpose:** Drive the one-way geared pen-lift cam motor just long enough
to reach the requested pen state, confirmed by the feedback switch,
rather than for a blind fixed duration — this is necessary precisely
because (per the wiring docstring) the pen motor is a *toggle* mechanism:
running it moves the pen from whatever state it is in now to the other
state, and running it for the wrong duration either stops short of
fully seating or (if a fixed pulse were used and the mechanism's timing
varied) could over-run past the intended state back toward the original
one.

**Inputs:** `target_up: bool` — desired end state (`True` = pen up).
`timeout_s: float = 3.0` — maximum time to wait for the switch to confirm
before giving up and reporting failure.

**Outputs:** Returns `bool` — `True` if the target state was confirmed via
the switch (or was already there), `False` if refused (end-stop tripped)
or timed out. Side effects: drives `PEN_MOTOR_PIN` ACTIVE then INACTIVE;
prints a diagnostic on failure.

**Inner workings / maths — the pulse/poll state machine:**
1. Refuse immediately (`return False`) if `triggered_endstop()` is
   non-`None` — the pen must not be actuated while an axis fault is live.
2. If `pen_is_up() == target_up` already, return `True` immediately
   *without running the motor* — the docstring is explicit that the
   toggle motor "must not run" in this case, since running a toggle
   actuator that's already at the target would flip it *away* from the
   target.
3. Otherwise: set `PEN_MOTOR_PIN` to `Value.ACTIVE` (motor starts
   running) and compute `deadline = time.time() + timeout_s`.
4. Poll in a `while time.time() < deadline` loop, every 2 ms
   (`time.sleep(0.002)`): break out early (fault) if an end-stop trips
   mid-actuation, or break out (success, `reached = True`) the instant
   `pen_is_up() == target_up`.
5. If the target was reaching *up* specifically (`reached and
   target_up`), sleep an additional fixed **`PEN_UP_OVERRUN_S = 0.04`**
   (40 ms) *before* switching the motor off. The inline comment explains
   this precisely: the switch "first makes contact" slightly before the
   mechanism has fully seated, so cutting power the instant the switch
   first trips would leave the pen mechanically under-seated; running
   40 ms longer lets the cam finish its travel. No equivalent overrun is
   applied when lowering the pen (`target_up == False`), implying the
   seating asymmetry only matters in the up direction for this particular
   mechanism (e.g. gravity assists seating when lowering).
6. Unconditionally (success or timeout) sets `PEN_MOTOR_PIN` back to
   `Value.INACTIVE` — this always executes, so a timed-out motor is never
   left running indefinitely.
7. If `reached` is `False`, prints a diagnostic that the move did not
   confirm (likely wiring/motor fault) but the function still returns
   `False` rather than raising, leaving the decision of how to handle a
   failed pen move to the caller (both call sites — the CLI's `u`/`d`
   commands and `run_gcode_file`'s `M3`/`M5` handling — choose to print a
   warning and continue rather than abort the whole job).

**Literature connection:** This is a one-way/ratchet-style actuator driven
by a timed/switch-confirmed pulse — standard embedded mechatronics
practice for toggle-type solenoid or cam actuators with position
feedback, no single canonical citation. The general mechanical principle:
a one-way gear motor (or ratchet mechanism) can only usefully be driven in
short bursts because continuous drive would either run the mechanism past
its intended end position (if it is a bistable cam/toggle with no natural
stopping point other than a switch) or attempt to drive it past a hard
mechanical stop, which is why pulsed, feedback-confirmed drive (rather
than continuous drive, and rather than open-loop fixed-duration pulses)
is the appropriate control strategy — this code implements the stronger
of those two options (feedback-confirmed, not fixed-duration).

---

## Calibration / homing

The docstring at lines 401–404 states the design goal precisely:
calibration is **direction-agnostic** — it never assumes in advance which
physical direction corresponds to "toward MIN" for either axis. Instead it
drives one way until *either* switch on an axis triggers, then reverses to
find the *other* switch, measuring the number of steps travelled between
the two. This makes the whole calibration procedure insensitive to wiring
mistakes in motor phase order or `INVERT_X`/`INVERT_Y` sign errors (those
flags are documented elsewhere as only mattering for drawing, never for
homing).

### `_step_axis(step_pin, dir_pin, dir_value, stop_when, ignore=frozenset())` (lines 405–423)

**Purpose:** The lowest-level homing primitive — steps one axis at a
fixed, slow, conservative rate in one fixed direction until a
caller-supplied stopping predicate is satisfied, treating any other
unexpected trigger as a wiring fault.

**Inputs:** `step_pin`, `dir_pin: int` — the GPIO line numbers for this
axis. `dir_value: Value` — the direction to drive (already resolved to
`Value.ACTIVE`/`Value.INACTIVE`, not a logical sign — the caller is
responsible for direction semantics here). `stop_when: Callable[[str],
bool]` — a predicate over the currently-triggered end-stop name,
returning `True` if that trigger is the expected stopping condition for
this call. `ignore: frozenset[str]` — switches to ignore during the scan
(passed through to `triggered_endstop`).

**Outputs:** Returns `(name: str, steps: int)` — the end-stop name that
satisfied `stop_when`, and how many step pulses were issued before it did.
Side effects: sets the DIR pin once; pulses the STEP pin repeatedly.
Raises `RuntimeError` if an *unexpected* end-stop triggers (one for which
`stop_when` returns `False` — interpreted as crossed/incorrect wiring),
or if `HOMING_MAX_STEPS` is exceeded without any expected trigger
(interpreted as "no switch exists or is wired at all" — a fault, not
silently looping forever).

**Inner workings:** Sets `dir_pin` once, then loops: check
`triggered_endstop(ignore=ignore)` first — if something triggers, either
return success (`stop_when(hit)` true) or raise (otherwise). Then check
the step ceiling and raise if exceeded. Otherwise pulse `step_pin` ACTIVE,
sleep `HOMING_STEP_DELAY_S`, pulse INACTIVE, sleep again, and increment
`steps`. Note this does **not** update `current_x_steps`/
`current_y_steps` itself — that bookkeeping is left to the caller
(`_calibrate_axis`/`calibrate`), since during the "find first switch"
phase the starting position isn't yet known in calibrated terms anyway.

### `_calibrate_axis(step_pin, dir_pin, min_name, max_name, other_axis_names=frozenset())` (lines 426–443)

**Purpose:** Fully home one axis: find one end, then find the other end,
without assuming which physical direction is which, and report the raw
step count between them (the measurement `calibrate()` uses to derive
real-world steps/mm).

**Inputs:** `step_pin`, `dir_pin: int`. `min_name`, `max_name: str` — the
two end-stop names for this axis (e.g. `"X_MIN"`, `"X_MAX"`).
`other_axis_names: frozenset[str]` — the *other* axis's switch names,
which may legitimately still read triggered (e.g. if that axis was
already homed and is resting against its own switch) and so must be
ignored as "not a fault here."

**Outputs:** Returns `(first_hit: str, second_hit: str, travel_steps:
int)`.

**Inner workings — the homing state machine, step by step:**
1. Drive forward (`Value.ACTIVE`) via `_step_axis`, stopping when the hit
   is *either* `min_name` or `max_name` (`stop_when=lambda h: h in
   (min_name, max_name)`), ignoring the other axis's switches. This finds
   whichever end happens to be in the direction the motor was arbitrarily
   started moving — direction-agnostic by construction.
2. Compute `other = max_name if first_hit == min_name else min_name` —
   whichever switch was *not* just hit is the target for the reverse pass.
3. Drive in reverse (`Value.INACTIVE`) via a second `_step_axis` call,
   stopping only when `h == other`, this time ignoring both the other
   axis's switches *and* `first_hit` itself — the comment explains this
   ignore is needed because the just-left switch may still read triggered
   for the first few steps of the reverse move due to mechanical/
   electrical release lag, which must not be mistaken for a real fault.
4. The step count from this second `_step_axis` call, `travel_steps`, is
   the measured switch-to-switch distance in raw motor steps for this
   axis.

### `calibrate()` (lines 446–480)

**Purpose:** The top-level, user-invoked homing routine: finds both
switches on both axes, derives true steps-per-mm for each axis from the
measured travel versus the known physical switch-to-switch distance,
updates the usable drawing area, and parks the gantry at a safe starting
corner.

**Inputs:** None (operates on globals and the hard-coded
`X_SWITCH_TRAVEL_MM`/`Y_SWITCH_TRAVEL_MM` constants).

**Outputs:** Returns `dict(calibration)` (a shallow copy of the final
calibration dict). Side effects: updates globals `current_x_steps`,
`current_y_steps`; updates `calibration["steps_per_mm_x"]`,
`["steps_per_mm_y"]`, `["usable_width_mm"]`, `["usable_height_mm"]`,
`["done"] = True`; drives both motors through the full homing sequence;
finally performs one G-code-style linear move to park. Raises
`RuntimeError` if an end-stop is already triggered before starting (would
make the direction-agnostic logic ambiguous/unsafe).

**Inner workings / maths — full homing sequence:**
1. Precondition check: refuse if `triggered_endstop()` is not `None`.
2. **X axis:** call `_calibrate_axis(STEP1_PIN, DIR1_PIN, "X_MIN",
   "X_MAX")`. Set `current_x_steps = x_travel if x_second == "X_MAX" else
   0` — i.e. the step counter is defined as 0 at `X_MIN` and
   `x_travel` at `X_MAX`, so whichever switch the axis ended homing on
   determines whether the counter starts at the high or low end. Compute
   the real conversion factor:

   ```
   steps_per_mm_x = x_travel_steps / X_SWITCH_TRAVEL_MM
   ```

   i.e. empirically measured steps divided by the independently,
   physically measured millimetre distance between the two switches —
   this is the calibration step that replaces the nominal `STEPS_PER_MM
   = 80.0` constant with a value specific to this particular belt/pulley/
   microstepping combination. Then immediately retract off `x_second` by
   `RETRACT_MM` via `_retract_off_switch` — necessary, per the docstring,
   because otherwise the axis is left resting switch-down, which would
   trip the main loop's own emergency-stop check the instant calibration
   mode ends.
3. **Y axis:** identical procedure via `_calibrate_axis(STEP2_PIN,
   DIR2_PIN, "Y_MIN", "Y_MAX", other_axis_names={"X_MIN", "X_MAX"})` —
   note X's switches are now passed as `other_axis_names` since X is
   already homed and resting near (just retracted from) its own switch,
   and must not be mistaken for a Y-axis fault.
4. Set `calibration["done"] = True` and recompute
   `usable_width_mm`/`usable_height_mm` as
   `SWITCH_TRAVEL_MM - 2 * EDGE_TOLERANCE_MM` per axis (restating the
   module-level initial values, now authoritative).
5. Call `_gcode_linear_move(0.0, 0.0, feed_mm_min=600.0)` — reuses the
   ordinary G-code linear-move machinery (which internally calls
   `bresenham_move`) to park the gantry at local coordinate (0, 0), which
   per `_gcode_linear_move`'s own coordinate convention (see below) is
   `EDGE_TOLERANCE_MM` in from both MIN switches — the safe starting
   corner, not the switches themselves.
6. Returns a copy of the final `calibration` dict to the caller for
   display.

**Literature connection:** This is a standard CNC/3D-printer-style
"home-to-switch, measure, derive scale factor" calibration procedure —
directly comparable to the homing cycles implemented in open-source
firmware such as GRBL or Marlin, though those typically assume known
direction and only measure one direction's travel; the direction-
agnostic double-probe approach here is a more defensive variant. No
single academic citation applies; this is standard CNC engineering
practice.

---

## G-code parsing and execution

### `parse_gcode_line(raw)` (lines 490–505)

**Purpose:** Convert one raw text line of a G-code file into a structured
`(command, params)` pair the interpreter can act on, stripping comments
and ignoring blank lines.

**Inputs:** `raw: str` — one line of text from the G-code file, exactly
as read (may include a trailing newline, inline comments, parenthetical
comments).

**Outputs:** Returns `(cmd: str, params: dict[str, float])` — e.g.
`("G1", {"X": 10.0, "Y": 5.0, "F": 900.0})` — or `None` if the line is
blank/comment-only or has no recognised G/M word.

**Inner workings:** Strips `;`-style line comments
(`raw.split(';', 1)[0]`), strips `(...)`-style parenthetical comments via
regex `re.sub(r'\([^)]*\)', '', line)`, then `.strip()`s whitespace; if
empty, returns `None`. Otherwise tokenises the remaining text with
`_GCODE_WORD_RE = re.compile(r'([A-Za-z])\s*(-?[0-9]*\.?[0-9]+)')`, which
matches a single letter immediately followed (optionally with whitespace)
by a signed decimal number — this is the standard G-code "word" format
(a letter + a number, e.g. `X12.5`, `F900`, `G1`). For each matched
`(letter, num)` pair, uppercases the letter and parses the number as
`float`. The *first* letter encountered that is `G` or `M` becomes the
command, formatted as `f"{letter}{int(val)}"` (so `G1.0` would parse to
`"G1"` — the integer part only, consistent with G-code's use of integer
command numbers even though the regex allows decimals). Every other
letter/number pair (or any G/M word after the first) is stored in the
`params` dict keyed by letter. Returns `None` if no `cmd` was ever set
(e.g. a line with only parameter words and no G/M word — malformed/
unsupported line, silently dropped). The code's own docstring notes the
simplifying assumption that there is exactly one G/M word per line.

**Literature connection:** G-code itself derives from EIA RS-274 (and
its later variants, e.g. NIST RS274/NGC); the parser here is a minimal,
project-specific subset implementation, not an attempt at full RS-274
compliance. Worth citing RS-274 as the standard this is a dialect of, if
the report wants to position the supported command subset against the
full standard.

### `_clamp_to_usable_area(x_mm, y_mm)` (lines 508–517)

**Purpose:** A second, software line of defence (independent of the
physical end-stop switches) that pulls any out-of-bounds target
coordinate back inside the calibrated safe drawing rectangle before it is
ever acted on — protecting against a bad or malicious G-code file, not
just against calibration or mechanical error.

**Inputs:** `x_mm`, `y_mm: float` — a requested target position in the
calibrated local coordinate frame (0,0 = safe corner, see below).

**Outputs:** Returns `(cx, cy): tuple[float, float]` — the clamped
coordinate. Side effect: prints a warning if clamping actually changed
the value.

**Inner workings / maths:**

```
w, h = calibration["usable_width_mm"], calibration["usable_height_mm"]
cx = min(max(x_mm, 0.0), w)
cy = min(max(y_mm, 0.0), h)
```

A straightforward two-sided clamp (`max` then `min`) per axis into
`[0, usable_width_mm] × [0, usable_height_mm]`.

### `_gcode_linear_move(target_x_mm, target_y_mm, feed_mm_min)` (lines 520–540)

**Purpose:** The core "move to an absolute millimetre position at a given
feed rate" primitive underlying both `G0`/`G1` and the per-segment
stepping used inside arc interpolation (`G2`/`G3`) — converts a
millimetre-space target and a feed rate into a step-space Bresenham call
with a computed per-step delay.

**Inputs:** `target_x_mm`, `target_y_mm: float` — target position,
absolute, in the calibrated local coordinate frame (mm). `feed_mm_min:
float` — feed rate in millimetres per minute (standard G-code `F` word
units).

**Outputs:** Side effect only (drives `bresenham_move`, which updates
`current_x_steps`/`current_y_steps` and may abort early on an end-stop).
No return value.

**Inner workings / maths — coordinate transform and feed-rate timing:**
1. Clamp the target via `_clamp_to_usable_area`.
2. Read the *calibrated*, per-axis steps/mm: `spmm_x =
   calibration["steps_per_mm_x"]`, similarly for Y.
3. Recover the *current* position in mm from the tracked step counters,
   explicitly correcting for the coordinate-frame offset:

   ```
   x0_mm = current_x_steps / spmm_x - EDGE_TOLERANCE_MM
   y0_mm = current_y_steps / spmm_y - EDGE_TOLERANCE_MM
   ```

   The comment clarifies the coordinate convention precisely: "mm=0" in
   G-code/local space is defined as `EDGE_TOLERANCE_MM` *in from* the
   X_MIN/Y_MIN switch, not at the switch itself (since `current_x_steps =
   0` corresponds to sitting exactly on the X_MIN switch per
   `calibrate()`'s convention). So the forward step→mm transform is `mm =
   steps/steps_per_mm - EDGE_TOLERANCE_MM`, and correspondingly the
   inverse mm→step transform (used next) is `steps = (mm +
   EDGE_TOLERANCE_MM) * steps_per_mm`.
4. Compute the straight-line distance to travel (for feed-rate timing,
   not for stepping itself):

   ```
   dist_mm = hypot(target_x_mm - x0_mm, target_y_mm - y0_mm)
   ```

   (Euclidean distance, `math.hypot` avoiding manual
   `sqrt(dx**2+dy**2)` overflow/precision pitfalls.)
5. Convert both the current and target mm positions to absolute step
   counts using the inverse transform above, rounding to the nearest
   integer step:

   ```
   target_x_steps = round((target_x_mm + EDGE_TOLERANCE_MM) * spmm_x)
   target_y_steps = round((target_y_mm + EDGE_TOLERANCE_MM) * spmm_y)
   dx_steps = target_x_steps - current_x_steps
   dy_steps = target_y_steps - current_y_steps
   ```

   The docstring stresses that the delta is computed from the *actual
   current integer step count* each time, not from an accumulated
   floating-point position — this avoids compounding rounding error
   across many small successive moves (a classic bug class in CNC/plotter
   firmware where repeated float accumulation causes slow positional
   drift; computing the delta fresh from the authoritative integer step
   counters each call eliminates it).
6. If there's no integer step change on either axis, return immediately
   (no-op move).
7. Compute per-step timing from the feed rate:

   ```
   major_steps = max(|dx_steps|, |dy_steps|, 1)
   feed_mm_s = max(feed_mm_min / 60.0, GCODE_MIN_FEED_MM_S)
   step_delay_s = max((dist_mm / feed_mm_s) / major_steps, CIRCLE_STEP_DELAY_S)
   ```

   `feed_mm_min / 60.0` converts the G-code feed rate from mm/min to
   mm/s, floored at `GCODE_MIN_FEED_MM_S = 0.1` mm/s so a stray `F0` in a
   file can never cause a division by zero or an infinite delay.
   `dist_mm / feed_mm_s` is the total wall-clock time the whole move
   should take to honour the requested feed rate; dividing by
   `major_steps` (the number of pulses the major axis will issue — i.e.
   the number of discrete timing "ticks" the move will take) gives the
   per-step delay needed so the whole move takes the right total time.
   This value is floored again at `CIRCLE_STEP_DELAY_S` as an absolute
   minimum step period (a hardware/software speed ceiling).
8. Finally hands off to `bresenham_move(dx_steps, dy_steps,
   step_delay_s=step_delay_s)` to actually perform the interleaved
   two-axis stepping at that timing.

**Literature connection:** The feed-rate-to-step-delay conversion is
standard constant-velocity G-code interpolation (no acceleration/
deceleration ramping — this is a trapezoidal- or S-curve-free, simple
fixed-speed-per-segment implementation); worth noting explicitly in the
report that no velocity ramping/jerk control is implemented, in contrast
to firmware like GRBL which implements full trapezoidal velocity
planning. No named algorithm beyond Bresenham (used internally).

### `_gcode_arc_move(x0_mm, y0_mm, x1_mm, y1_mm, i_mm, j_mm, clockwise, feed_mm_min)` (lines 543–565)

**Purpose:** Implements `G2` (clockwise arc) / `G3` (counterclockwise arc)
by approximating the true circular arc as a sequence of short straight
line segments, since there is no native circular-interpolation stepping
mode — each segment is simply handed to `_gcode_linear_move`.

**Inputs:** `x0_mm, y0_mm: float` — arc start point (current position).
`x1_mm, y1_mm: float` — arc end point (target). `i_mm, j_mm: float` — the
standard G-code I/J offset format: the arc's centre is specified
*relative to the start point*, not as absolute coordinates. `clockwise:
bool` — `True` for G2, `False` for G3. `feed_mm_min: float` — feed rate,
passed through to each linear sub-segment unchanged.

**Outputs:** Side effect only — issues a sequence of
`_gcode_linear_move` calls. Returns early if an end-stop trips mid-arc.

**Inner workings / maths — arc parameterisation and discretisation:**
1. Resolve the absolute centre from the I/J offsets:
   `cx, cy = x0_mm + i_mm, y0_mm + j_mm` — this is exactly the standard
   G-code I/J convention (centre = start point + I/J offset, as opposed
   to the less common R-word radius format, which this parser does not
   support).
2. `radius = hypot(i_mm, j_mm)` — the radius is derived, not given
   directly, as the distance from the start point to the computed centre.
3. Degenerate-arc guard: if `radius < 1e-6`, treat it as a zero-radius
   arc (malformed G-code) and fall back to a plain linear move to the
   target instead of attempting arc maths on a degenerate circle.
4. Compute the start and end angles of the arc relative to the centre
   using `atan2`:

   ```
   start_angle = atan2(y0_mm - cy, x0_mm - cx)
   end_angle   = atan2(y1_mm - cy, x1_mm - cx)
   ```

5. Normalise the end angle so that sweeping from `start_angle` to
   `end_angle` goes the *correct rotational direction*:
   - If clockwise (`G2`): while `end_angle >= start_angle`, subtract
     `2π` — ensures the end angle is reached by *decreasing* angle
     (clockwise in standard mathematical convention, since angle
     increases counterclockwise).
   - If counterclockwise (`G3`): while `end_angle <= start_angle`, add
     `2π` — ensures it's reached by *increasing* angle.

   This loop-based normalisation handles the case where the naive
   `atan2` difference would otherwise imply sweeping the "short way"
   around in the wrong rotational sense, or wrapping past ±π incorrectly.
6. `angle_span = end_angle - start_angle` — the (now correctly signed)
   total angle to sweep.
7. Determine how many straight-line segments to approximate the arc
   with, targeting an approximately fixed chord length:

   ```
   n_segments = max(4, int(|angle_span| * radius / 0.4))
   ```

   `|angle_span| * radius` is the arc length in mm (angle in radians ×
   radius = arc length); dividing by `0.4` targets roughly a 0.4 mm chord
   per segment (stated directly in the function's docstring: "~0.4mm
   chords"), with a floor of 4 segments regardless of how short/small the
   arc is (so even a tiny arc is still polygonally approximated rather
   than collapsing to a single segment with excessive angular error).
8. Step through the arc as a parametric angle sweep: for `k` from `1` to
   `n_segments`, compute the fractional progress `k / n_segments` and the
   corresponding angle `a = start_angle + angle_span * (k / n_segments)`,
   then the corresponding point on the circle `(cx + radius*cos(a), cy +
   radius*sin(a))`, and issue a `_gcode_linear_move` to that point at the
   same feed rate. An end-stop check happens once per segment, so an arc
   can be aborted mid-sweep just like a straight move, just with
   segment-level (not single-step-level) granularity for the abort check
   — though note that each `_gcode_linear_move` call internally invokes
   `bresenham_move`, which itself re-checks the end-stop on every single
   step within the segment, so the overall real granularity of the abort
   check is still per-step, just that *this* function's own check is
   coarser (defence in depth, not the only check).

**Literature connection:** This is a standard engineering approximation
of circular arcs as a polygon of short chords for digital
plotting/stepper systems (effectively a piecewise-linear interpolation
in polar form around the arc centre) — a conventional CNC technique, not
a single named published algorithm. If the report wants a citation
alongside Bresenham for completeness, this would be described as a
"polygonal/chordal arc approximation," contrasted with native circular
interpolation implemented in hardware motion controllers (e.g. via
Bresenham's own circle-drawing variant, which this code does *not* use —
it uses trigonometric sampling instead, which is simpler to implement but
costs more floating-point computation per segment than an integer
circle-Bresenham would).

### `run_gcode_file(path)` (lines 568–646)

**Purpose:** The top-level G-code interpreter loop: reads a whole file,
walks it line by line, maintains interpreter state (absolute/relative
mode, current feed rate, current position), dispatches each parsed
command to the appropriate handler, and checks for end-stop faults before
every single line.

**Inputs:** `path: str` — filesystem path to a G-code text file.

**Outputs:** No return value (callers read module globals `estopped`/
`tripped_switch` afterward, as `write_gcode_file` does explicitly). Side
effects: drives all gantry/pen motion for the whole file; prints
per-line progress/diagnostic/error messages; may set `estopped` via
`_trigger_emergency_stop`.

**Inner workings — interpreter state machine:**
1. Precondition: raises `NotCalibratedError` if `calibration["done"]` is
   `False` — the docstring notes this is checked here specifically so it
   applies uniformly whether the caller is the CLI or the web server.
2. Reads the whole file into `lines` up front (`f.readlines()`); any
   `OSError` (file not found, permissions) is caught, printed, and the
   function returns without raising further (a soft failure for file-
   access problems, as opposed to the harder `NotCalibratedError`/
   `GantryNotConnectedError` raised elsewhere).
3. Initialises interpreter state: `abs_mode = True` (G-code conventionally
   starts in absolute positioning mode), `feed_mm_min = 900.0` (a default
   feed rate used until an `F` word is seen), and `cur_x_mm`/`cur_y_mm`
   derived from the *current* step counters via the plain (uncorrected
   for `EDGE_TOLERANCE_MM`!) conversion `current_x_steps / STEPS_PER_MM` —
   note this specific line uses the module-level nominal `STEPS_PER_MM`
   constant, **not** the calibrated `calibration["steps_per_mm_x"]`, which
   is a minor inconsistency worth flagging (see Limitations) relative to
   `_gcode_linear_move`'s own internal recomputation of position from
   calibrated values and the edge-tolerance offset — in practice this only
   affects the interpreter's *locally tracked* `cur_x_mm`/`cur_y_mm`
   (used for computing relative-mode deltas and arc start points), while
   the actual motor stepping always goes through `_gcode_linear_move`'s
   own correct, calibrated step-based computation.
4. Main loop over every line, numbered from 1:
   - **Fault check first:** calls `triggered_endstop()`; if tripped,
     calls `_trigger_emergency_stop(hit)` and aborts the whole file
     immediately (`return`). Also checks the `estopped` flag separately
     (covers the case where a previous iteration already latched it,
     e.g. mid-arc via `_gcode_arc_move`'s own internal check) and aborts
     similarly.
   - Parses the line via `parse_gcode_line`; skips (`continue`) if
     `None` (blank/comment/unsupported-format line).
   - **`G0`/`G1`** (treated identically — this driver does not
     distinguish rapid vs. feed-rate moves, noted explicitly in the
     module docstring: "linear move, no separate rapid mode"): updates
     `feed_mm_min` if `F` present; resolves target X/Y either as the
     absolute param value (if given, else stays at current) in `abs_mode`,
     or as current + delta in relative mode; calls
     `_gcode_linear_move`; updates `cur_x_mm`/`cur_y_mm` to the
     (unclamped!) requested target — note this means the interpreter's
     locally tracked position can drift from the actual clamped physical
     position if `_clamp_to_usable_area` ever altered the real target;
     subsequent relative moves would then be computed from the
     *requested*, not actual, position. This is a subtlety worth flagging
     for the report's limitations discussion.
   - **`G2`/`G3`**: identical target resolution, then delegates to
     `_gcode_arc_move` with `cmd == "G2"` passed as the `clockwise` flag.
   - **`G4`**: dwell. Computes `deadline = time.time() + p.get("P", 0.0)`
     and busy-waits in a `while time.time() < deadline` loop, sleeping
     5 ms per iteration and checking `triggered_endstop()` each
     iteration so a dwell can also be interrupted by a fault.
   - **`G90`/`G91`**: set `abs_mode` True/False respectively.
   - **`G20`**: prints a warning that inch values are used *as-is, not
     converted to mm* — i.e. this G-code dialect does not actually
     implement unit conversion despite recognising the command; a file
     using inches would produce wrong physical distances unless its
     numbers happen to already be in mm. This is an explicit, documented
     gap (also called out in the module docstring) and should be
     mentioned in the report's limitations/expected-failure-conditions
     section.
   - **`G21`**: no-op (`pass`) — millimetres is the implicit native unit
     throughout this file, so `G21` requires no action.
   - **`M3`**: pen down — calls `set_pen(False)`; on failure, prints a
     warning but does **not** abort the job (continues drawing with
     whatever pen state resulted).
   - **`M5`**: pen up — calls `set_pen(True)`; same non-aborting failure
     handling.
   - **`M2`/`M30`**: program end — prints and `return`s immediately,
     ending the file early by design (these mean "stop here" in G-code
     regardless of remaining lines).
   - **Anything else:** printed as an explicitly skipped/unsupported
     command, and the loop continues (no abort) — a deliberately
     permissive stance toward unrecognised commands.
5. After the loop completes naturally (file exhausted without an `M2`/
   `M30` or a fault), prints a completion message.

### `write_gcode_file(path)` (lines 649–656)

**Purpose:** The externally-facing entry point used by the project's web
server component to run a finished G-code file — wraps connection
management and execution into one call with a clean return value.

**Inputs:** `path: str`.

**Outputs:** Returns `{"estopped": bool, "tripped_switch": str | None}` —
a small status dict the caller (web server) can use to report success/
failure to its own client without needing to inspect module globals
directly. Side effects: calls `connect()` (idempotent — connects only if
not already connected), then `run_gcode_file(path)` (full side effects as
above), and *deliberately leaves the connection open afterward* (no
`disconnect()` call) so a subsequent job doesn't pay the connect overhead
again and so the gantry/pen state stays whatever it was left at. Can
propagate `GantryNotConnectedError` (from `connect()`) or
`NotCalibratedError` (from `run_gcode_file`) — the docstring tells callers
to catch both for a clean error response instead of a raw 500.

---

## Interactive CLI

### `input_listener()` (lines 662–759)

**Purpose:** Runs as a background daemon thread (started in `__main__`)
reading newline-terminated commands from stdin and dispatching them,
while the main thread's loop (described next) concurrently drives
whatever `current_mode` is currently active. This split exists so a
long-running G-code job or circle demo doesn't block the ability to type
new commands, and so RPM/circle mode stepping (driven by the *main*
thread) can run continuously without needing to poll stdin itself.

**Inputs:** None directly — reads from `sys.stdin` in a loop. Each raw
line is both kept in original case (`raw_cmd`, needed for file paths after
`g `) and lower-cased (`cmd`, used for command matching).

**Outputs:** Side effects per command — too numerous to repeat in full
after the sections above, since most commands simply call the functions
already documented (`calibrate()`, `set_pen()`, `bresenham_move()` via
G-code or circle mode, `triggered_endstop()`/estop-clear logic). Notable
points not yet covered:

- **`calibrate`** command: refuses if already `estopped`. Sets
  `current_mode = "CALIBRATING"` *before* calling `calibrate()` —
  the comment explains this specific mode exists so the **main loop**
  (below) skips its own end-stop-triggered emergency-stop reaction while
  calibration is in progress, since `calibrate()` deliberately drives
  into switches on purpose and already handles that itself via its own
  `_step_axis` logic; if the main thread's loop reacted too by zeroing
  `STEP1_PIN`/`STEP2_PIN` mid-pulse, it would race with the input-listener
  thread's own in-flight pulse on those same pins (a real, called-out
  thread-safety hazard specific to the dual-thread design). After
  `calibrate()` returns (success or `RuntimeError`), the `finally` block
  resets `current_mode` back to `"RPM"` with `target_rpm = 0.0`, but only
  `if not estopped` (a fault during calibration should leave the system
  latched, not silently reset to idle RPM mode).
- **`r`** (clear estop): refuses if any end-stop is still physically
  triggered (`triggered_endstop()` non-`None`) — printed as "still
  triggered" — this prevents clearing the flag while the switch itself is
  still mechanically pressed, which (absent `_retract_off_switch` having
  already run) would otherwise immediately re-trigger. Only actually
  clears `estopped`/`tripped_switch` and resets to `"RPM"`/`0.0` if
  `estopped` was true in the first place; otherwise just reports "Not
  stopped."
- **`u` / `d`**: direct pen up/down via `set_pen`, printing success/
  failure.
- **`c [radius]`**: enters circle demo mode. Parses an optional radius
  argument (validated positive and float-parseable; invalid/non-positive
  input is reported and the previous radius kept rather than crashing).
  Recomputes `circle_radius_steps = circle_radius_mm * STEPS_PER_MM`
  (again using the nominal, not calibrated, conversion — consistent with
  this being a pure hardware-test demo mode rather than a calibrated
  drawing operation). Resets `theta`, `current_x_steps`,
  `current_y_steps` all to 0 (circle mode treats its own start point as
  the new step-space origin, not the calibrated drawing-area origin).
- **`g <path>`**: sets `current_mode = "GCODE"` (so the main loop idles
  while *this* thread drives the whole job synchronously) and calls
  `run_gcode_file(path)` directly from the input-listener thread itself —
  i.e. G-code execution blocks the input-listener thread for the whole
  job's duration (stdin commands typed during a running job are queued by
  the OS but not read/acted on until the job function returns). After it
  returns, resets to `"RPM"`/`0.0` unless now `estopped`.
- **Anything else (bare number):** attempts `float(cmd)`; silently
  ignores (`continue`) anything that doesn't parse as a number at all
  (not treated as an error — just not a recognised command). If
  `estopped`, refuses. Otherwise: if `abs(target_rpm) < 0.1`, snaps to
  exactly `0.0` and reports "Stopped"; otherwise sets both DIR pins via
  `axis_dir_value` (same sign for both axes — this test mode drives X and
  Y the same logical direction simultaneously, used purely to verify
  motors/wiring work, not to draw anything meaningful), recalculates
  `step_delay_s` via `calculate_delay(target_rpm)`, and reports the new
  mode.

**Literature connection:** Standard command-line/REPL dispatch loop;
no named algorithm. The producer/consumer-free, single-reader-thread
plus single-driver-thread concurrency model is a simple, pragmatic
design choice appropriate for a single-operator test/control tool, not
a general concurrent-systems pattern requiring citation.

### `if __name__ == "__main__":` main loop (lines 762–816)

**Purpose:** The actual real-time driving loop — this is what repeatedly
calls `connect()`, starts the input-listener thread, and then continuously
checks end-stops and drives whichever mode is active (`RPM`, `CIRCLE`,
`GCODE`, or `CALIBRATING`) until `running` becomes `False` or the process
is interrupted, always calling `disconnect()` on the way out via a
`finally` block (so Ctrl+C, an unhandled exception, or a clean `q` all
leave the GPIO lines released and motors off).

**Inner workings, per iteration:**
1. If `current_mode == "CALIBRATING"`: sleep 10 ms and `continue` — the
   comment reiterates the thread-safety rationale given above for why
   this loop must stay completely hands-off during calibration (the
   input-listener thread, via `calibrate()`/`_step_axis`, is the sole
   driver of `STEP1_PIN`/`STEP2_PIN` during this window).
2. Otherwise check `triggered_endstop()`. If tripped:
   `_trigger_emergency_stop(hit)` is called from *this* (main) thread,
   `target_rpm` is zeroed, and the loop sleeps 5 ms before continuing —
   this is the "stop immediately on any trigger" guarantee from the
   module docstring realised concretely: the check happens on every pass
   of this loop, interleaved with whichever stepping mode is active,
   rather than via an interrupt callback.
3. Else if already `estopped` (but nothing newly triggered this instant):
   sleep 5 ms and `continue` — explicitly, a switch releasing on its own
   does **not** auto-clear the estop; only the explicit `'r'` command in
   `input_listener` does, by design (preventing an intermittent/bouncing
   switch from silently resuming motion).
4. **`RPM` mode:** if `target_rpm != 0.0` and `step_delay_s > 0`, pulses
   *both* `STEP1_PIN` and `STEP2_PIN` together: set both ACTIVE, sleep
   `step_delay_s`, set both INACTIVE, sleep `step_delay_s` again — this
   realises the half-period timing computed by `calculate_delay` (two
   sleeps of `step_delay_s` per full step cycle, matching that function's
   division by 2 noted above). If stopped, sleeps a coarser 10 ms instead
   (no need to tight-loop when idle).
5. **`CIRCLE` mode:** implements the free-running circle demo using a
   fixed-rate parametric angle update rather than Bresenham-at-circle-
   resolution-per-tick: every `circle_interval_s` (1.5 ms) elapsed on
   `time.perf_counter()`, increments `theta` by a fixed `0.008` radians
   (wrapped modulo `2π`), computes the target point on the circle as

   ```
   target_x = round(circle_radius_steps * sin(theta))
   target_y = round(circle_radius_steps * (1 - cos(theta)))
   ```

   This is a standard parametric circle through the origin: at `theta=0`,
   `(x,y) = (0,0)`; the `(1 - cos(theta))` form (rather than plain
   `cos(theta)`) shifts the circle so it starts and is tangent at the
   origin rather than centred there, convenient for a "draw a circle
   starting from wherever the pen currently is" demo. It then calls
   `bresenham_move(target_x - current_x_steps, target_y - current_y_steps)`
   — i.e. each ~1.5 ms tick, Bresenham is invoked fresh with the delta
   from the *current* actual step position to the newly computed point on
   the ideal circle, so each tick draws one more short Bresenham-
   interpolated segment approximating the circle, broadly the same
   "polygon of short chords" idea used in `_gcode_arc_move`, just driven
   by a fixed angular increment and timer tick instead of a feed-rate/
   chord-length computation.
6. **`GCODE` mode:** the main loop does nothing but sleep 10 ms per
   iteration — the comment clarifies this mode exists purely so the main
   loop *doesn't* try to also drive motors while the input-listener
   thread is synchronously running `run_gcode_file`.
7. On loop exit (`running` becomes `False`, or an exception propagates
   out of the `try`), the `finally` block unconditionally calls
   `disconnect()` and prints "Exited."

---

## Known limitations / honest caveats

These are offered specifically for the report's "testing methodology,"
"expected failure conditions," and "unsolved problems" discussions.

1. **No real off-hardware simulation.** The `gpiod` import guard (lines
   46–56) only prevents an *import-time* crash on non-ODROID machines; it
   does **not** provide a mock/stub GPIO backend. `gpiod`, `Direction`,
   and `Value` are simply set to `None`, and any attempt to actually use
   hardware (`connect()`) raises `GantryNotConnectedError` cleanly rather
   than silently simulating pin states. This means: (a) the module is
   safe to `import` from the web server or any non-ODROID host without
   crashing the whole process, which is the actual purpose of the guard;
   but (b) there is no built-in way to exercise the real stepping/timing
   code paths (`bresenham_move`, the main loop's stepping branches,
   `set_pen`'s switch-polling loop) without a real `req` object backed by
   real or mocked hardware. Testing done without the physical gantry
   therefore necessarily exercised only the pure-logic layers directly —
   `parse_gcode_line` (regex/text parsing), the Bresenham error-term
   arithmetic in isolation, the mm↔step conversion formulas, and the arc
   angle/segment-count maths in `_gcode_arc_move` — by calling those
   functions standalone or re-deriving their formulas, not by running the
   module end-to-end. The report should describe testing honestly along
   these lines: algorithmic/parsing logic was unit-testable off-hardware;
   actual motor timing, end-stop debounce behaviour, and pen-mechanism
   timing could only be validated on the physical ODROID + gantry.

2. **Hard-coded, per-build mechanical constants.** Several constants are
   measurements of one specific physical assembly and would need
   re-measurement on any rebuild, re-wiring, or belt/pulley change:
   - `X_SWITCH_TRAVEL_MM = 204.0`, `Y_SWITCH_TRAVEL_MM = 262.0` — manually
     measured switch-to-switch distances; any change in switch mounting
     position invalidates these and silently produces wrong steps/mm
     after the next `calibrate()` run (calibrate() does not detect that
     these reference constants might be stale — it trusts them
     completely).
   - `STEPS_PER_MM = 80.0` — a nominal fallback based on assumed belt
     pitch/pulley teeth/microstepping; only used before calibration
     completes (circle-demo mode and `run_gcode_file`'s initial
     `cur_x_mm`/`cur_y_mm` seed also use this nominal constant rather than
     the calibrated value — see point 4 below).
   - `MICROSTEPS = 16` — a driver-board DIP-switch/jumper setting assumed
     in software; if the physical driver is ever reconfigured to a
     different microstepping setting, this constant must be updated
     manually or all RPM/timing calculations silently become wrong.
   - `HOMING_STEP_DELAY_S = 0.0008`, `PEN_UP_OVERRUN_S = 0.04`,
     `ENDSTOP_DEBOUNCE_S`/`ENDSTOP_CONFIRM_SAMPLES`, `RETRACT_MM` — all
     empirically tuned timing constants specific to the particular
     motors, switches, and pen mechanism used; a different pen motor or
     noisier wiring could require retuning.
   - `INVERT_X`/`INVERT_Y` — specific to how this particular build's
     motor phases/belts happen to be wired; flipped on any rewiring.

3. **`G20` (inch mode) is recognised but not implemented.** The
   interpreter explicitly prints a warning and uses inch values *as
   literal millimetre values* rather than converting — a G-code file
   that actually uses inch units would be drawn at the wrong physical
   scale (25.4× too small). This is a known, documented gap rather than a
   silent bug, but is still a real functional limitation worth naming in
   "expected failure conditions."

4. **Minor position-tracking inconsistency in `run_gcode_file`.** The
   interpreter's initial `cur_x_mm`/`cur_y_mm` (line 583–584) are derived
   using the nominal `STEPS_PER_MM` constant rather than the calibrated
   `calibration["steps_per_mm_x"/"y"]` values that `_gcode_linear_move`
   itself uses internally and that are the actual source of truth after
   `calibrate()` has run. Because every actual move still routes through
   `_gcode_linear_move` (which recomputes everything correctly from the
   calibrated values and the true step counters), the *physical* motion
   is unaffected; however, the interpreter's locally-tracked
   `cur_x_mm`/`cur_y_mm` used for computing **relative-mode (`G91`)**
   deltas and arc start points could be very slightly off from the true
   calibrated position, especially right after a session where
   calibration measured steps/mm noticeably different from the nominal
   80.0 fallback. Additionally, `cur_x_mm`/`cur_y_mm` are updated after
   `G0`/`G1`/`G2`/`G3` to the *requested* target, not the
   *clamped* target actually reached if `_clamp_to_usable_area` altered
   it — so a G-code file that commands an out-of-bounds move followed by
   a relative move could compound a small positional error. Neither issue
   causes a hard failure, but both are worth a sentence in the report's
   "unsolved problems" discussion as sources of potential small
   cumulative positional drift in edge cases (out-of-bounds targets,
   mixed-mode files), distinct from the robust, drift-free step-counter-
   based approach used for the actual motor-driving maths.

5. **No velocity ramping/acceleration profiling.** Both the feed-rate-
   derived step delay in `_gcode_linear_move` and the fixed-rate RPM test
   mode use a constant step period for the whole move/segment — there is
   no trapezoidal or S-curve acceleration/deceleration ramp as found in
   full CNC firmware (e.g. GRBL). Every move effectively starts and stops
   at full commanded speed instantly, which on real hardware risks
   skipped steps (loss of position without any error being raised) if a
   commanded feed rate is too aggressive for the motor/mechanism's torque
   at that instant. This is a legitimate "unsolved problem"/future-work
   item for the report.

6. **Pen mechanism asymmetry is undocumented beyond the overrun
   constant.** `PEN_UP_OVERRUN_S` is applied only when raising the pen,
   not lowering it; the code does not explain *why* lowering needs no
   equivalent overrun (plausibly gravity assists seating when lowering),
   so this is inferred mechanical reasoning rather than something stated
   in the code — flagged here so the report doesn't overstate certainty
   about the exact mechanical cause.
