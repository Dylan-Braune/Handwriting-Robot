"""Write-up for reproduce.py: drawing the handwriting on the real machine."""

MODULES = {
 "reproduce.py": dict(intro=[
   "**Functional units 7 and 8, requirements R1 and R5.** This file turns a "
   "pen path into motion. It has two halves and they are independent of each "
   "other.",

   ("The upper half converts a trajectory in millimetres into a g-code file. "
    "It is pure arithmetic, runs on any computer, and is the half every "
    "measured result in the project comes from. The lower half drives the "
    "machine itself: stepper pulses, limit switches, homing and the pen "
    "lift. It needs the Raspberry Pi and the wiring, so its imports are "
    "guarded and the file still loads without them."),

   ("Nothing in this file knows anything about handwriting. It receives a "
    "list of strokes and draws them, and it would behave identically if "
    "they came from tracing a real page rather than from synthesis. That "
    "separation is deliberate: it means a fault in the drawing can never be "
    "a fault in the handwriting model, and it is why the g-code round trip "
    "can be measured independently of everything upstream."),

   ("Two constraints from the physical build shape the design more than "
    "anything else. The pen actuator is a one-way gear motor that turns 90 "
    "degrees per pulse, so the pen cannot be commanded to a position, only "
    "toggled — its state has to be tracked in software. And the axes are "
    "driven by stepper motors through toothed belts, so position is counted "
    "in whole steps from a known home, which means homing has to establish "
    "that origin before any absolute move is meaningful."),
 ]),
}

FUNCS = {

 "GantryConfig": dict(
   inputs="Constructed with no arguments in every current call site; every "
          "field has a default.",
   output="A dataclass holding the machine's physical parameters, with three "
          "derived properties.",
   body=[
     "Every physical number about the machine in one place: motor steps per "
     "revolution, microstepping, belt pitch times pulley teeth, the work "
     "area, where text starts on the bed, the drawing and travel feed rates, "
     "and the pen actuator's timing.",
     ("sub", "The derived quantities"),
     "Steps per millimetre is not typed in; it is computed from the gearing:",
     ("formula", ["microstepsPerRev = fullStepsPerRev * microstepping",
                  "stepsPerMm       = microstepsPerRev / mmPerRev",
                  "                 = (200 * 16) / 40  =  80 steps/mm"]),
     "The reason this matters is maintenance rather than elegance. Changing "
     "a pulley or a driver's microstepping jumper changes one field and "
     "everything downstream follows; a typed-in steps-per-millimetre would "
     "have to be recalculated by hand and would silently disagree with the "
     "hardware if anyone forgot.",
     ("sub", "Two observations about this class"),
     "It is never constructed with overridden values anywhere in the "
     "project — ten call sites, all `GantryConfig()`. The configurability is "
     "therefore available but unused.",
     "And the field `accelMmS2` is declared and **never read**. There is no "
     "acceleration limiting anywhere in the g-code; it is left entirely to "
     "the firmware. Any claim that the trajectory is acceleration-limited "
     "would not be supported by this code.",
   ]),

 "ToMachine": dict(
   inputs="`traj`, a trajectory. `cfg`, the machine. `fitToBounds`, whether "
          "to shrink to fit. `optimizeOrder`, whether to reorder strokes.",
   output="`(strokesXY, scale)` — the strokes in bed coordinates and the "
          "scale factor applied.",
   body=[
     "Places the drawing on the bed. Three things happen.",
     ("sub", "Translation"),
     "The trajectory's own bounding box is moved so that its top-left "
     "corner sits at the configured origin, which is where text starts on "
     "the bed. Both frames have y upwards, so the first line of text lands "
     "at the origin and later lines descend from it.",
     ("sub", "Scaling, downwards only"),
     ("formula", ["scale = min( 1 , availableWidth/w , availableHeight/h )"]),
     "The `min` with 1 is the important part: the drawing is shrunk if it "
     "would otherwise run off the work area, and never enlarged. Asking for "
     "small writing therefore gets small writing rather than writing blown "
     "up to fill the bed, which is what a naive fit-to-page would do.",
     ("sub", "Decimation and ordering"),
     "Each stroke is passed through `_Decimate` and then the whole set "
     "through `OptimizeStrokeOrder`.",
   ]),

 "_Decimate": dict(
   inputs="`poly`, a stroke in bed coordinates. `minSeg`, the shortest move "
          "worth making.",
   output="The stroke with sub-resolution moves merged.",
   body=[
     "Removes points closer together than the machine can usefully resolve. "
     "A traced or finely resampled stroke can carry a point every fraction "
     "of a millimetre; each becomes a line of g-code and a separate move "
     "command.",
     "The threshold is set by what the machine can actually do. At 80 steps "
     "per millimetre one microstep is 12.5 micrometres, and the default "
     "minimum segment of 0.12 mm is about ten microsteps — short enough to "
     "be invisible in the drawing, long enough that the file does not "
     "contain thousands of moves the machine will execute as a single "
     "step.",
     "This matters for motion quality as well as file size: a controller "
     "asked to execute very many very short moves spends its time "
     "decelerating and accelerating between them and the result is slower "
     "and rougher than the same path with fewer points.",
   ]),

 "OptimizeStrokeOrder": dict(
   inputs="`strokes`, the strokes in bed coordinates. `rowTolMm`, how far "
          "apart two strokes can be vertically and still count as the same "
          "band.",
   output="The strokes reordered.",
   body=[
     ("sub", "The problem"),
     "Between strokes the pen is up and the machine is moving without "
     "drawing, which is wasted time. Drawing a page in the order the strokes "
     "happen to arrive means the pen criss-crosses the page. Minimising the "
     "total pen-up distance over all possible orderings is the travelling "
     "salesman problem, which is NP-hard, so an exact solution is not "
     "available for a page with hundreds of strokes.",
     ("sub", "The heuristic"),
     "Two standard ideas combined. Strokes are first grouped into horizontal "
     "bands, using a tolerance derived from the median stroke height, and "
     "the bands are processed top to bottom. Within a band, the nearest "
     "unvisited stroke is taken next.",
     "Banding first is what keeps the result sensible: nearest-neighbour "
     "alone will happily jump between the top and bottom of the page when "
     "two strokes happen to be close in one axis. Banding also matches how "
     "writing is naturally organised, so the machine draws roughly the way a "
     "person would, which makes watching it comprehensible and makes a fault "
     "obvious.",
     "Nearest-neighbour is typically within a few tens of per cent of "
     "optimal, which is ample here: the pen-up travel is a fraction of the "
     "pen-down travel once the order is reasonable, so further optimisation "
     "would be saving a small share of a small quantity.",
   ]),

 "_Clamp": dict(
   inputs="`strokes`, `cfg`.",
   output="The strokes with every point inside the work area.",
   body=[
     "A final safety net. Everything upstream should already keep the "
     "drawing inside the bed, so this should never change anything — and "
     "that is why it is here. Driving the gantry into its end of travel is "
     "a mechanical event, not a software one, and the cost of a redundant "
     "check is nothing against the cost of being wrong once.",
   ]),

 "NewPenState": dict(
   inputs="`cfg`.",
   output="A small mutable state: whether the pen is believed to be up, and "
          "how many toggles have been emitted.",
   body=[
     "Starts tracking the pen. The initial value comes from "
     "`cfg.penStartsUp`, the assumed state at power-on, because there is "
     "nothing in the g-code path that can read the pen switch — that only "
     "exists when driving the machine directly.",
   ]),

 "SetPenLines": dict(
   inputs="`state`, the tracked pen state. `cfg`. `wantUp`, the state "
          "wanted.",
   output="The g-code lines needed to get there, which is an empty list when "
          "the pen is already in the wanted state.",
   body=[
     ("sub", "Why this is not trivial"),
     "The pen is raised and lowered by a one-way gear motor that turns 90 "
     "degrees per pulse. It cannot be told to go up; it can only be told to "
     "turn, which alternates between up and down. There is no absolute "
     "command and, in the file-writing path, no feedback.",
     "So the software has to hold a model of where the pen is and emit a "
     "pulse only when that model needs to change. If it emitted a pen-up "
     "before every stroke regardless, every alternate stroke would be drawn "
     "with the pen up and the drawing would come out as every other stroke.",
     ("sub", "The dwell"),
     "After each toggle a dwell is emitted:",
     ("formula", ["dwell = (penPulseMs + penSettleMs) / 1000  seconds"]),
     "The pulse time is how long the 90 degree rotation takes and the settle "
     "time allows the mechanism to stop moving. Without the dwell the "
     "controller would begin the next move while the pen was still "
     "descending, which draws an arc into the page as it lands.",
   ]),

 "WriteGcode": dict(
   inputs="`traj`, `cfg`, `path`, and an optional `title` for the header "
          "comment.",
   output="None; writes the file.",
   body=[
     "Turns a trajectory into a g-code file. The structure per stroke is: "
     "move to the start with the pen up at travel feed, lower the pen, draw "
     "the stroke at drawing feed, raise the pen.",
     "Pen-up moves use the faster travel feed because nothing is being "
     "drawn, and the drawing feed is slower because stroke quality depends "
     "on it. Separating the two is most of why a page does not take several "
     "times longer than necessary.",
     "The header records the configuration the file was generated with, "
     "including the derived steps per millimetre, so that a file found later "
     "can be checked against the machine it was meant for.",
     "**Measured result.** Writing a trajectory and reading it back with the "
     "independent parser reproduces the geometry to a mean vertex error of "
     "0.0004 mm and a maximum of 0.0007 mm, with an intersection-over-union "
     "of 0.997 to 0.999, for every one of the thirteen writers. Against "
     "requirement R5's 0.5 mm tolerance that is a factor of about 700, so "
     "the file format contributes essentially nothing to the error budget.",
   ]),

 "ParseGcode": dict(
   inputs="`path`, a g-code file.",
   output="A list of strokes recovered from it.",
   body=[
     "Reads a g-code file back into strokes: it follows the moves, tracks "
     "the pen state from the pen commands, and collects the points drawn "
     "while the pen is down.",
     ("sub", "Why this is written separately on purpose"),
     "It would be shorter to share code with the writer. It is deliberately "
     "not shared, because the point of the round-trip measurement is to test "
     "the file. If the reader and writer shared a representation, a mistake "
     "in that shared representation would cancel out and the test would "
     "report perfect agreement while both were wrong. Two independent "
     "implementations agreeing is evidence; one implementation agreeing with "
     "itself is not.",
     "What the round trip does and does not measure is worth being precise "
     "about. It measures everything the file format can lose: coordinate "
     "rounding, stroke reordering, decimation and clamping. It measures "
     "nothing about the machine — no backlash, no belt stretch, no missed "
     "steps. The 0.0004 mm figure is a statement about the file, not about "
     "the pen on the paper.",
   ]),

 "GantryNotConnectedError": dict(
   output="An exception type.",
   body=["Raised when something that needs the hardware is called without "
         "it. A distinct type rather than a generic error so a caller can "
         "distinguish 'no machine attached' from 'the machine refused'."]),

 "NotCalibratedError": dict(
   output="An exception type.",
   body=[
     "Raised when an absolute move is requested before homing. Until homing "
     "has run, the step counters have no fixed relationship to the bed, so "
     "an absolute coordinate is meaningless and acting on it would drive the "
     "axis to an arbitrary place. Refusing is the only safe response."
   ]),

 "Connect": dict(
   output="None; claims the GPIO pins.",
   body=[
     "Claims the pins and puts them in a known safe state: drives disabled, "
     "pen motor off. Setting the state explicitly rather than leaving the "
     "pins as found matters because a pin's power-on state is not "
     "guaranteed, and an enabled driver with an undefined direction line is "
     "a motor that may move when the first pulse arrives."
   ]),

 "Disconnect": dict(output="None; releases the pins.",
   body=["Releases the claim so another process can use the hardware, and "
         "so the pins return to their default state."]),

 "IsConnected": dict(output="Whether the pins are currently claimed.",
   body=["Lets callers check rather than catching an exception."]),

 "CalculateDelay": dict(
   inputs="`rpm`, the wanted motor speed.",
   output="The delay in seconds between step pulses.",
   body=[
     "Converts a wanted speed into the pause between pulses, which is how "
     "speed is actually controlled when stepping in software:",
     ("formula", ["stepsPerSecond = rpm/60 * microstepsPerRev",
                  "delay          = 1 / stepsPerSecond"]),
     "This is a software-timed stepper drive, so the achievable speed is "
     "bounded by how precisely the delay can be held, not by the motor. That "
     "is a known limitation of driving steppers from a general-purpose "
     "operating system."
   ]),

 "TriggeredEndstop": dict(
   inputs="`ignore`, switches to disregard.",
   output="The name of a triggered switch, or None.",
   body=[
     ("sub", "Why reading the pin once is not enough"),
     "A mechanical switch bounces: the contacts chatter for a few "
     "milliseconds on closing. The wiring also runs beside stepper cables "
     "carrying switched current, which induces noise. A single read can "
     "therefore show a trigger that is not real, and a false trigger during "
     "a drawing stops the job.",
     ("sub", "The treatment"),
     "Each switch is read `ENDSTOP_CONFIRM_SAMPLES` times with a short "
     "interval and is only believed if every read agrees. Requiring "
     "consecutive agreement rejects both bounce and induced spikes, since "
     "neither persists, while a genuinely pressed switch stays pressed.",
     "The `ignore` set exists for the retract operation, which deliberately "
     "runs while a switch is pressed and must not be stopped by it."
   ]),

 "_AllTriggeredEndstops": dict(output="Every switch currently reading "
          "triggered.",
   body=["Used by homing and by the safety check at startup, where the "
         "question is whether any axis is at a limit rather than which one "
         "stopped a particular move."]),

 "AxisDirValue": dict(
   inputs="`invert`, whether the axis is wired backwards. `positive`, the "
          "wanted direction.",
   output="The logic level for the direction pin.",
   body=[
     "Resolves the direction pin from the wanted direction and the axis's "
     "inversion flag. The flag exists because which way a motor turns for a "
     "given level depends on how the coils are wired, and that is settled at "
     "build time, not in software. Keeping it in one function means the rest "
     "of the code can speak in terms of positive and negative travel."
   ]),

 "_RETREAT_SIGN": dict(output="Which direction comes off each switch.",
   body=["A minimum switch is escaped by moving positive and a maximum "
         "switch by moving negative."]),

 "_RetractOffSwitch": dict(
   inputs="`hit`, the switch that is pressed. `ignore`.",
   output="None; moves the axis until the switch releases.",
   body=[
     "Backs an axis off a pressed switch. A pressed switch means the axis is "
     "at its limit, and nothing else may be attempted from there — a move in "
     "the wrong direction would drive the carriage into the end stop.",
     "The retract therefore runs first, in the known-safe direction, with "
     "the pressed switch in the ignore set so its own state does not abort "
     "the move, until the switch releases and then a little further by "
     "`RETRACT_MM` so it is clear of the trip point rather than resting on "
     "it."
   ]),

 "_TriggerEmergencyStop": dict(
   inputs="`hit`, the switch responsible.",
   output="None.",
   body=[
     "Stops everything immediately: drive disabled, pen raised, the machine "
     "marked stopped, and the responsible switch recorded.",
     "The pen is raised rather than left where it is because a stationary "
     "pen resting on paper bleeds ink into one spot. Recording which switch "
     "fired is what makes the fault diagnosable afterwards, and the stopped "
     "state latches so that nothing resumes until it is explicitly cleared "
     "— an automatic recovery after an unexplained limit trip would risk "
     "repeating whatever caused it."
   ]),

 "BresenhamMove": dict(
   inputs="`dx`, `dy`, the move in whole steps. `stepDelayS`, the pause "
          "between pulses.",
   output="None; pulses the motors.",
   cite="Bresenham1965",
   body=[
     ("sub", "The problem"),
     "Two stepper motors can each only take whole steps. To draw a straight "
     "diagonal line, the two have to be stepped in the right interleaved "
     "pattern — for a line 100 steps across and 30 up, roughly one Y step "
     "every 3.33 X steps, which is not a whole number.",
     ("sub", "Bresenham's solution"),
     "Carry an integer error term that accumulates the fractional part "
     "[Bresenham1965]. For the case where X is the major axis:",
     ("formula", ["err = 2*|dy| - |dx|",
                  "for each step along x:",
                  "    step x",
                  "    if err > 0:",
                  "        step y",
                  "        err -= 2*|dx|",
                  "    err += 2*|dy|"]),
     "The error term tracks how far the ideal line has drifted from the "
     "stepped path, scaled by 2*|dx| so that everything stays an integer. "
     "When it goes positive the minor axis is a step behind and is stepped "
     "to catch up. The path never deviates from the true line by more than "
     "half a step.",
     ("sub", "Why it is the right choice here"),
     "No floating point, no division inside the loop, and no accumulated "
     "rounding error over a long move. On a small single-board computer "
     "pulsing motors in software, where the loop body's duration directly "
     "sets the feed rate, both the cost and the predictability matter. The "
     "algorithm dates from plotter control [Bresenham1965], which is exactly "
     "what this machine is.",
     "The limit switches are polled inside the loop, so a move stops within "
     "one step of a trigger rather than running to completion first."
   ]),

 "PEN_UP_OVERRUN_S": dict(output="How long a pen pulse may run before it is "
          "treated as failed.",
   body=["A one-way motor with a jammed mechanism would otherwise run "
         "indefinitely."]),

 "PenIsUp": dict(output="Whether the pen switch reads up.",
   body=["The only feedback the pen mechanism provides, and the reason "
         "`SetPen` can verify a toggle while the file-writing path can only "
         "assume."]),

 "SetPen": dict(
   inputs="`targetUp`, the wanted state. `timeoutS`.",
   output="None; raises if the pen never reaches the state.",
   body=[
     "Raises or lowers the pen and confirms it actually happened. The motor "
     "is pulsed and the switch watched; if the expected state does not "
     "arrive within the timeout, an error is raised rather than continuing.",
     "Continuing would be worse than stopping. A failed pen-down draws "
     "nothing and a failed pen-up drags the pen across the page, and because "
     "the actuator is one-way, a missed toggle inverts the pen state for the "
     "whole of the rest of the job: every subsequent stroke would be drawn "
     "in the opposite sense."
   ]),

 "_StepAxis": dict(
   inputs="The pins, a direction, a stopping condition, and switches to "
          "ignore.",
   output="The number of steps taken.",
   body=["Steps one axis until a condition is met, which is the shared "
         "mechanism behind homing and retracting. Taking the stopping "
         "condition as a callable keeps one stepping implementation rather "
         "than one per purpose."]),

 "_CalibrateAxis": dict(
   inputs="The pins and switch names for one axis.",
   output="The measured travel of that axis in steps.",
   body=[
     "Drives one axis to its minimum switch, zeroes the counter, then drives "
     "to its maximum switch and records the count. That count is the usable "
     "travel.",
     "Measuring rather than assuming matters because the usable travel "
     "depends on the build: where the switches ended up, how the carriage "
     "sits, how the belt was tensioned. A figure from the drawing would be "
     "wrong by a few millimetres on each axis."
   ]),

 "Calibrate": dict(
   output="None; sets the module-level calibration.",
   body=[
     "Homes both axes and establishes the work area.",
     ("sub", "The switch travel correction"),
     "A microswitch does not trigger at the exact point of contact; its "
     "lever travels a measurable distance first. The raw triggering "
     "positions therefore lie outside the true usable area by that amount at "
     "each end, so `X_SWITCH_TRAVEL_MM` and `Y_SWITCH_TRAVEL_MM` are "
     "subtracted, with a further `EDGE_TOLERANCE_MM` kept as margin.",
     "These are measured properties of the parts used, and they are the "
     "difference between the software's idea of the bed and the actual bed.",
     ("sub", "Why homing is slow on purpose"),
     "`HOMING_STEP_DELAY_S` is longer than the drawing delay. A homing move "
     "ends by driving into a hard stop, and doing that quickly risks "
     "skipping steps on impact, which corrupts the very origin the operation "
     "exists to establish. `HOMING_MAX_STEPS` bounds the move so a "
     "disconnected switch produces an error instead of a motor running until "
     "something gives."
   ]),

 "GCODE_MIN_FEED_MM_S": dict(output="The slowest feed the runner accepts.",
   body=["A guard against a file asking for a feed so low the job would "
         "never finish."]),

 "_GCODE_WORD_RE": dict(output="The pattern splitting a g-code word into its "
          "letter and number.",
   body=["G-code is a sequence of letter-number words, so one pattern "
         "handles every command."]),

 "ParseGcodeLine": dict(
   inputs="`raw`, one line of g-code.",
   output="A dictionary from letter to number.",
   body=[
     "Splits a line into its words, discarding comments. Keeping parsing "
     "separate from acting on the result is what lets the same parser serve "
     "both the machine runner and the round-trip verification."
   ]),

 "_ClampToUsableArea": dict(
   inputs="`xMm`, `yMm`.",
   output="The point clamped into the homed work area.",
   body=["The last check before a move. The area comes from homing, not from "
         "configuration, so it reflects the machine as built."]),

 "_GcodeLinearMove": dict(
   inputs="The target in millimetres and the feed rate.",
   output="None; performs the move.",
   body=[
     "Converts a target in millimetres into a step count relative to the "
     "current position, sets the delay from the feed rate, and hands the "
     "move to `BresenhamMove`.",
     "Position is tracked in steps rather than millimetres throughout, "
     "because steps are what the machine actually does. Converting to "
     "millimetres and back each time would accumulate rounding error over "
     "thousands of moves and the machine's idea of where it is would drift "
     "from the truth."
   ]),

 "_GcodeArcMove": dict(
   inputs="Start and end points, the I and J centre offsets, direction, and "
          "feed.",
   output="None; performs the arc.",
   body=[
     "Implements G2 and G3 arcs, which the machine cannot execute directly "
     "because it only knows how to move in straight lines.",
     ("sub", "Finding the arc"),
     "In g-code, I and J give the centre as an offset from the start point, "
     "so the centre and radius follow directly, and the start and end angles "
     "come from the two endpoints.",
     ("formula", ["cx, cy = x0 + I , y0 + J",
                  "radius = sqrt(I^2 + J^2)",
                  "a0 = atan2(y0 - cy, x0 - cx)",
                  "a1 = atan2(y1 - cy, x1 - cx)"]),
     ("sub", "The direction problem"),
     "Those two angles do not determine the arc. `atan2` returns a value in "
     "a fixed range, so the sweep from a0 to a1 could be either way round "
     "and could be the short way or the long way. The code forces the sweep "
     "to be strictly decreasing for a clockwise arc and strictly increasing "
     "otherwise by adding or subtracting whole turns until it is. Without "
     "that, an arc crossing the discontinuity is drawn the wrong way round, "
     "which is a common defect in hand-written g-code interpreters.",
     ("sub", "Chord subdivision"),
     ("formula", ["nSegments = max( 4 , |angleSpan| * radius / 0.4 )"]),
     "The arc is split so that each straight chord spans about 0.4 mm of arc "
     "length. The number of segments therefore scales with the arc's actual "
     "length rather than with its angle, so a large gentle arc and a small "
     "tight one are both drawn to the same accuracy. The deviation of a "
     "chord from the arc is approximately `L^2 / (8*radius)`, so at 0.4 mm "
     "chords on a 10 mm radius the error is about 2 micrometres, far below "
     "anything the machine can resolve.",
     "Limit switches are polled between chords, so an arc can be interrupted "
     "like any other move.",
   ]),

 "RunGcodeFile": dict(
   inputs="`path`, a g-code file.",
   output="None; draws the file.",
   body=[
     "Reads and executes a file line by line: linear moves, arcs, pen "
     "commands and dwells, stopping immediately if a limit switch trips.",
     "State is carried between lines — the current position, the feed rate, "
     "and the pen — because g-code is modal: a line that gives only "
     "coordinates inherits the last feed rate and motion mode. Treating each "
     "line independently would be a misreading of the format."
   ]),

 "WriteGcodeFile": dict(output="None; writes the test pattern.",
   body=["Saves the built-in test geometry as g-code, which gives a known "
         "file for checking the machine without involving any of the "
         "handwriting pipeline."]),

 "InputListener": dict(output="None; runs until stopped.",
   body=[
     "The keyboard control loop for manual testing: jog the axes, toggle the "
     "pen, home, run a file. It exists because bringing up a machine "
     "requires moving one axis a little and watching what happens, and doing "
     "that through the full pipeline would confound a wiring fault with a "
     "software one."
   ]),
}
