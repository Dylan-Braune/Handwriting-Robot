"""reproduce.py, part 1: pen path to G-code.

The half of the file that is pure arithmetic: placing a drawing on the bed,
ordering the strokes, tracking the pen actuator, and writing and reading a
G-code file.  Part 2, in codetheory_reproduce2.py, covers driving the real
machine.
"""

INTRO = [
 ("p", "This file draws. It takes a list of strokes in millimetres and "
       "turns them into motion, and nothing in it knows anything about "
       "handwriting: the strokes may have come from synthesis or from "
       "tracing a real page, and the file behaves the same either way. That "
       "separation is deliberate, and it is what allows the drawing half to "
       "be tested without any of the recognition half being involved."),
 ("h", "The two halves, and why they are kept apart"),
 ("items", [
   "**The top half** converts a pen path into a G-code file. It is pure "
   "arithmetic, runs on any computer, and is the half this project's "
   "measured geometry results were taken on.",
   "**The bottom half** drives the real gantry: stepper motors, limit "
   "switches, homing and the pen lift. It needs the single-board computer "
   "and the wiring, so its imports are guarded and the top half still works "
   "without them.",
 ]),
 ("p", "The guard matters more than it appears. Without it, importing this "
       "module on a laptop would fail at the hardware import, and since "
       "every test and every evaluation script imports it, none of them "
       "could run anywhere but on the machine itself."),
 ("h", "Background: what G-code is"),
 ("p", "**G-code** is the standard language for telling a machine tool "
       "where to move. A line is a command letter and number, such as G0 or "
       "G1, followed by parameters: `G1 X12.5 Y40.0` means move in a "
       "straight line to that position. It dates from numerically "
       "controlled milling and is what nearly every plotter, 3-D printer "
       "and mill accepts, which is why it is used here rather than a "
       "private format."),
 ("items", [
   "**G0** is a rapid move, used with the pen up, where only the "
   "destination matters.",
   "**G1** is a controlled move at a feed rate, used with the pen down, "
   "where the path matters.",
   "**G2 and G3** are circular arcs, clockwise and anticlockwise.",
   "**G4** is a dwell: wait for a stated time.",
   "**M3 and M5** are auxiliary commands, here mapped to the pen "
   "actuator.",
   "**G21 and G90** declare millimetres and absolute coordinates.",
 ]),
 ("h", "Units"),
 ("p", "The top half works in millimetres throughout. The bottom half works "
       "in motor steps, converting at the boundary using a steps-per-"
       "millimetre figure that calibration measures rather than assumes."),
]


FUNCS = {

 "GantryConfig": dict(
   output="a record of every physical number describing the machine, with "
          "derived step scaling available as properties.",
   theory=[
     ("h", "What it is for"),
     ("p", "Collects everything physical about the machine into one place: "
           "the gearing, the work area, the feed rates and the pen "
           "actuator's timing. A physical change to the machine, a "
           "different pulley or a larger bed, is then one edit here rather "
           "than a search for the number wherever it was used."),
     ("h", "Why steps per millimetre is derived and not typed in"),
     ("p", "This is the one piece of real reasoning in the record. Steps "
           "per millimetre is the number that converts a distance into "
           "motor steps, and it could simply be stated. It is instead "
           "computed from the parts the machine is actually built from."),
     ("eq", r"\mathrm{stepsPerMm} \;=\; "
            r"\frac{\mathrm{fullStepsPerRev} \times "
            r"\mathrm{microstepping}}{\mathrm{mmPerRev}}",
      [(r"\mathrm{stepsPerMm}", "how many motor steps move the axis one "
                                "millimetre"),
       (r"\mathrm{fullStepsPerRev}", "the motor's own step count per "
                                     "revolution, 200 for the NEMA 17 "
                                     "used"),
       (r"\mathrm{microstepping}", "how many electrical sub-steps the "
                                   "driver divides each full step into, 16 "
                                   "here"),
       (r"\mathrm{mmPerRev}", "how far the axis travels per motor "
                              "revolution, set by the belt and pulley")],
      None),
     ("p", "With the values given this is 200 times 16 divided by 40, or 80 "
           "steps per millimetre. The reason for deriving it is that each "
           "of the three inputs is a fact someone can check against the "
           "hardware: the motor's label says 1.8 degrees per step, the "
           "driver's jumpers say 1/16, and the belt is 2 mm pitch with a "
           "20-tooth pulley, giving 40 mm per revolution. A single stated "
           "80 could not be checked that way, and if a pulley were changed "
           "it would be silently wrong, with the only symptom being a "
           "drawing of the wrong size."),
     ("h", "Background: what microstepping is"),
     ("p", "A stepper motor moves in fixed increments, 1.8 degrees for a "
           "200-step motor. **Microstepping** is the driver's ability to "
           "hold the rotor between two full steps by driving the coils at "
           "intermediate currents, dividing each step into smaller ones. "
           "At 1/16 the resolution becomes 0.1125 degrees, which for this "
           "gearing is 12.5 micrometres of travel. Finer resolution is not "
           "the main benefit: full stepping produces an abrupt jump "
           "between positions that makes the motor vibrate audibly and "
           "visibly, and microstepping smooths that into nearly continuous "
           "motion, which matters when the thing being drawn is a curve a "
           "few millimetres across."),
     ("h", "Why the pen actuator has timing in the config"),
     ("p", "The pen is raised and lowered by a geared motor that turns 90 "
           "degrees per pulse, so the time one pulse takes is a physical "
           "property that the G-code must wait for. The pulse and settling "
           "times are recorded here and used to emit a dwell of the right "
           "length. Getting that wrong does not fail cleanly: too short a "
           "dwell means drawing begins while the pen is still descending, "
           "which produces a stroke that starts in the air."),
     ("note", "The acceleration limit is declared in this record and is "
              "never read: no acceleration limiting is implemented. Motion "
              "steps at a constant rate computed from the feed, so a move "
              "begins at full speed. For the light gantry and modest feeds "
              "used that is tolerable, but a reader should not take the "
              "field's presence as evidence that a trapezoidal profile is "
              "applied."),
   ]),

 "ToMachine": dict(
   inputs=[("traj", "the trajectory to place"),
           ("cfg", "the machine configuration"),
           ("fitToBounds", "whether to shrink the drawing to fit the work "
                           "area, default true"),
           ("optimizeOrder", "whether to reorder the strokes, default true")],
   output="a pair: the strokes in bed coordinates, and the scale factor "
          "applied.",
   theory=[
     ("h", "What it is for"),
     ("p", "A trajectory is positioned wherever synthesis happened to put "
           "it, with its origin at the start of the first line. The bed has "
           "its own coordinates, its own extent, and a chosen place where "
           "text should begin. This function moves the drawing from one to "
           "the other."),
     ("h", "The transformation"),
     ("eq", r"x_{\mathrm{bed}} \;=\; \mathrm{originX} + (x - x_0)\,k, "
            r"\qquad y_{\mathrm{bed}} \;=\; \mathrm{originY} - "
            r"(y_1 - y)\,k",
      [(r"x_{\mathrm{bed}},\, y_{\mathrm{bed}}", "the position on the bed, "
                                                 "in millimetres"),
       ("x,\\,y", "the position in the trajectory"),
       ("x_0", "the trajectory's leftmost extent"),
       ("y_1", "its topmost extent"),
       (r"\mathrm{originX},\, \mathrm{originY}", "where text is to begin on "
                                                 "the bed"),
       ("k", "the scale factor")],
      None),
     ("p", "Both frames have y increasing upward, so no flip is needed, "
           "unlike the rasteriser's conversion. What the y expression does "
           "instead is measure downward *from the top* of the drawing: the "
           "topmost row of text lands exactly at the chosen origin and "
           "later rows descend from there. Anchoring at the top rather than "
           "the bottom is what makes a one-line and a ten-line page both "
           "start in the same place, which is how a person writes on a "
           "sheet."),
     ("h", "Why it never enlarges"),
     ("eq", r"k \;=\; \min\!\left(1,\; "
            r"\frac{\mathrm{availW}}{w},\; "
            r"\frac{\mathrm{availH}}{h}\right)",
      [("k", "the scale factor actually applied"),
       (r"\mathrm{availW}", "the bed width available to the right of the "
                            "origin"),
       (r"\mathrm{availH}", "the height available below it"),
       ("w,\\,h", "the drawing's own width and height"),
       ("1", "the cap, which is what prevents enlargement")],
      None),
     ("p", "The 1 in the minimum means the drawing is shrunk when it would "
           "not fit and is otherwise left at the size asked for. That is "
           "the right behaviour because the size was chosen deliberately: "
           "the caller specified an x-height in millimetres, so the writing "
           "is meant to be that size. Scaling up to fill the bed would "
           "override that choice, and a request for small writing would "
           "produce large writing. Scaling down is different in kind: it is "
           "not overriding the request but avoiding an impossibility, since "
           "a drawing larger than the bed cannot be drawn at all."),
     ("h", "Why both axes get the same factor"),
     ("p", "A single factor is taken as the smaller of the two ratios, "
           "rather than each axis being scaled to fit independently. "
           "Independent scaling would stretch the writing, changing the "
           "proportions of every letter, which is both visibly wrong and "
           "destroys the property the whole pipeline maintains, that the "
           "stored shapes are reproduced faithfully."),
   ]),

 "_Decimate": dict(
   inputs=[("poly", "a path in millimetres"),
           ("minSeg", "the shortest segment to keep, in millimetres")],
   output="the path with points closer together than that merged away.",
   theory=[
     ("h", "What it is for"),
     ("p", "A path may carry far more points than the machine can act on. "
           "A traced stroke has a point per pixel, and resampling during "
           "layout placed points every 0.35 mm. This removes the points "
           "that are closer together than the machine can resolve."),
     ("h", "The method, and why it is not Douglas-Peucker"),
     ("p", "Each point is kept only if it is at least the minimum distance "
           "from the last point kept. This is a different rule from the "
           "path simplification in style.py, which keeps points according "
           "to how far they stray from a straight line, and the difference "
           "is appropriate to the different purpose. Here the question is "
           "not whether a point carries shape information but whether the "
           "machine can act on it at all: a commanded move shorter than the "
           "minimum segment is below the resolution of the motion, so it "
           "costs a line in the file and produces nothing."),
     ("h", "Why the comparison is against the last kept point"),
     ("p", "Comparing against the last kept point rather than the previous "
           "input point is what makes the rule accumulate correctly. "
           "Against the previous input point, a long run of points each "
           "slightly under the threshold would all be dropped, however far "
           "the path travelled in total. Against the last kept point, the "
           "distance accumulates until the threshold is reached, so a point "
           "is emitted every minimum segment along the path regardless of "
           "how finely the input was sampled."),
     ("note", "The first and last points are always kept, whatever the "
              "spacing. The endpoints define where a stroke begins and "
              "ends, so dropping one would shorten the stroke, and the "
              "final point in particular is where the pen lifts and where "
              "the next stroke's travel is measured from."),
   ]),

 "OptimizeStrokeOrder": dict(
   inputs=[("strokes", "the strokes in bed coordinates"),
           ("rowTolMm", "how far apart two strokes may sit vertically and "
                        "still count as the same row; derived if not "
                        "given")],
   output="the same strokes in an order that reduces pen-up travel.",
   theory=[
     ("h", "What it is for"),
     ("p", "The strokes may be drawn in any order, since each is "
           "independent. The order determines how far the pen travels "
           "raised between them, and that travel is pure waste: it takes "
           "time and draws nothing. For a page of handwriting with "
           "hundreds of strokes the difference between a good order and a "
           "poor one is substantial."),
     ("h", "Background: this is a travelling salesman problem"),
     ("p", "Finding the order that minimises total travel is the "
           "**travelling salesman problem**: given a set of places, find "
           "the shortest route visiting all of them. It is one of the "
           "classic hard problems in computing, with no known method that "
           "finds the exact best route in reasonable time as the number of "
           "places grows. With several hundred strokes an exact answer is "
           "out of reach, so a **heuristic** is used: a rule that gives a "
           "good answer quickly with no guarantee of being best."),
     ("h", "The heuristic: bands, then nearest neighbour"),
     ("p", "Two ideas are combined. The strokes are first grouped into "
           "horizontal bands by their vertical centres, and the bands are "
           "processed from top to bottom. Within a band, the nearest "
           "unvisited stroke is taken next."),
     ("items", [
       "**Banding** exploits the structure of the problem. Text is "
       "arranged in lines, and almost all the strokes of one line of "
       "writing are near each other, so a route that finishes one line "
       "before starting the next is close to optimal. A pure nearest-"
       "neighbour search without banding can wander between lines, since "
       "a stroke on the line below may happen to be nearer than the next "
       "stroke along.",
       "**Nearest neighbour** within a band is the standard simple "
       "heuristic: from where the pen is, go to the closest place not yet "
       "visited. It is greedy, so it can paint itself into a corner and "
       "leave a long move to the last stroke, but within one line of text "
       "the strokes are nearly collinear and the greedy route is close to "
       "the best.",
     ]),
     ("h", "Why the band tolerance is derived from the strokes"),
     ("eq", r"\mathrm{rowTol} \;=\; \max\big(1,\; 1.2 \cdot "
            r"\mathrm{median}\{\,h_s\,\}\big)",
      [(r"\mathrm{rowTol}", "how far apart two strokes' vertical centres "
                            "may be and still share a band, in "
                            "millimetres"),
       ("h_s", "the height of stroke s, from its lowest point to its "
               "highest"),
       ("1.2", "a margin above the typical stroke height"),
       ("1", "a floor in millimetres, for a drawing of very small strokes")],
      None),
     ("p", "A fixed tolerance would be wrong at every size but one. "
           "Deriving it from the median stroke height ties it to the "
           "writing: a band should be about as tall as a line of text, and "
           "a typical stroke is about the height of a letter. The median "
           "again rather than the mean, so that one tall stroke spanning "
           "several lines, which a long descender or an underline can "
           "produce, does not widen every band."),
     ("h", "Reversing a stroke to enter from the nearer end"),
     ("p", "A stroke can be drawn from either end, and the result is "
           "identical ink. So when the far end is closer to the pen than "
           "the near end, the stroke is reversed. The code comment notes "
           "that this is free for a plotter, and it is worth saying why: "
           "unlike a milling cut or an extrusion, a pen stroke has no "
           "direction-dependent behaviour, so nothing is lost. It can "
           "nearly halve the travel to a long stroke."),
     ("note", "Fewer than three strokes are returned unchanged. With two "
              "strokes there is only one meaningful order, and the banding "
              "statistics would be computed from a sample too small to "
              "describe anything."),
   ]),

 "_Clamp": dict(
   inputs=[("strokes", "the strokes in bed coordinates"),
           ("cfg", "the machine configuration")],
   output="a pair: the strokes with every point forced inside the work "
          "area, and how many points were moved.",
   theory=[
     ("h", "What it is for"),
     ("p", "A final safety net. Every point is forced inside the machine's "
           "declared bounds, and the number of points that had to be moved "
           "is counted and reported."),
     ("h", "Why it exists when ToMachine already fits the drawing"),
     ("p", "The fitting scales the drawing to the available area, so in "
           "principle nothing can lie outside it. This clamp exists for the "
           "cases where that reasoning does not hold: the fitting can be "
           "switched off by the caller, the origin can be set such that the "
           "available area is computed wrongly, and a stroke can be "
           "reversed or reordered by code that has no knowledge of the "
           "bounds. A point outside the work area is not a cosmetic "
           "problem; it commands the machine to drive into its own frame."),
     ("h", "Why the count is returned"),
     ("p", "Clamping silently would hide a real fault. A drawing that "
           "needed clamping was mispositioned, and the clamped version is "
           "not what was asked for: the affected strokes are flattened "
           "against the boundary, which is visible as a straight edge where "
           "writing should be. Returning the count lets the writer emit a "
           "warning into the file, so the condition is recorded where "
           "someone will see it rather than inferred from the drawing."),
     ("note", "Clamping each coordinate independently is not geometrically "
              "principled: a stroke crossing the boundary diagonally is "
              "bent rather than cut, so its shape changes. Properly "
              "clipping the path to the rectangle would preserve the shape "
              "of the part inside. That is not done because the condition "
              "is a fault to be reported rather than a case to be handled "
              "well, and the simple form cannot itself fail."),
   ]),

 "NewPenState": dict(
   inputs=[("cfg", "the machine configuration")],
   output="a small mutable record holding whether the pen is up and how "
          "many pulses have been emitted.",
   theory=[
     ("h", "Why the pen's state must be tracked at all"),
     ("p", "This is the central peculiarity of the machine and it shapes "
           "the whole of the G-code writing. The pen is raised and lowered "
           "by a geared motor that turns 90 degrees per pulse. It is a "
           "**one-way actuator**: it accepts \"move\" and nothing else. "
           "There is no command meaning \"be up\", and nothing in the "
           "G-code can ask where the pen currently is."),
     ("p", "The consequence is that the writer must know the pen's state "
           "by remembering it. The state begins at a known value, assumed "
           "at power-on, and every pulse emitted flips it. Emitting a "
           "toggle when the pen is already in the wanted state would move "
           "it to the wrong one, and from then on every pen command in the "
           "file would be inverted: the machine would travel with the pen "
           "down and draw with it up."),
     ("h", "Why it is a list rather than a plain value"),
     ("p", "The record is a two-element list so that the function updating "
           "it can modify it in place and the caller sees the change. A "
           "tuple or an integer would be copied, and the caller would keep "
           "the old state. This is the smallest possible piece of shared "
           "mutable state and it exists because the pen genuinely has a "
           "state that persists across many lines of output."),
     ("note", "The starting state is an assumption, not a measurement. If "
              "the machine is powered on with the pen down, every pen "
              "command in the file is inverted. The hardware half resolves "
              "this properly, with a switch that reports the pen's actual "
              "position, but a written file cannot: it has no way to ask."),
   ]),

 "SetPenLines": dict(
   inputs=[("state", "the pen state record, updated in place"),
           ("cfg", "the machine configuration"),
           ("wantUp", "the state wanted")],
   output="the G-code lines needed to reach that state, which is nothing at "
          "all when it is already current.",
   theory=[
     ("h", "What it does"),
     ("p", "Compares the wanted state with the tracked one. If they agree, "
           "no lines are emitted. If they differ, the state is flipped, the "
           "pulse count incremented, and two lines are produced: the toggle "
           "command and a dwell."),
     ("h", "Why emitting nothing is the important case"),
     ("p", "The empty return is what makes the mechanism correct. Because "
           "the actuator toggles, a redundant command is not merely "
           "wasteful, it is wrong. The writer calls this function before "
           "every travel move and before every stroke, which means it is "
           "called many times with a state that is already current, and "
           "every one of those calls must produce nothing. Placing that "
           "check inside the function rather than at each call site is what "
           "guarantees it, since a single forgotten check at one call site "
           "would invert the rest of the file."),
     ("h", "Why a dwell follows every toggle"),
     ("eq", r"\mathrm{dwell} \;=\; \frac{\mathrm{penPulseMs} + "
            r"\mathrm{penSettleMs}}{1000}",
      [(r"\mathrm{dwell}", "the wait emitted after a toggle, in seconds"),
       (r"\mathrm{penPulseMs}", "how long one 90-degree rotation takes"),
       (r"\mathrm{penSettleMs}", "extra time for the mechanism to come to "
                                 "rest"),
       ("1000", "the conversion from milliseconds to the seconds a G-code "
                "dwell expects")],
      None),
     ("p", "G-code is executed line by line, and the machine would begin "
           "the next move as soon as the toggle command was issued, while "
           "the pen was still travelling. Drawing would then start before "
           "the pen reached the paper, losing the beginning of the stroke, "
           "or a travel move would begin before the pen was clear, dragging "
           "a line across the page. The dwell blocks until the motion is "
           "complete. The settling time is added on top of the rotation "
           "time because the mechanism has backlash and the pen oscillates "
           "slightly as it lands."),
     ("note", "The pulse count is accumulated and reported by the writer. "
              "It is a useful figure because each pulse is a mechanical "
              "operation taking about a third of a second: a file with "
              "twice the necessary pulses takes noticeably longer to draw "
              "and wears the actuator twice as fast. It is also how a fault "
              "in the stroke ordering shows up, since a poor order lifts "
              "the pen more often than necessary."),
   ]),

 "WriteGcode": dict(
   inputs=[("traj", "the trajectory to write"),
           ("cfg", "the machine configuration"),
           ("path", "where to write the file"),
           ("title", "an optional title for the header comment")],
   output="a dictionary summarising what was written: line and stroke "
          "counts, pen pulses, the scale applied, and how many points were "
          "clamped.",
   theory=[
     ("h", "What it is for"),
     ("p", "The entry point of the top half. It places and orders the "
           "strokes, clamps them, and writes the file."),
     ("h", "The pattern for one stroke"),
     ("items", [
       "**Pen up,** if it is not already, so the travel does not draw.",
       "**G0 to the stroke's first point,** at travel feed. A rapid move, "
       "since the path taken does not matter with the pen raised.",
       "**Pen down,** with its dwell.",
       "**G1 at drawing feed,** then one G1 per remaining point.",
     ]),
     ("p", "The two feed rates are set once rather than repeated on every "
           "line, because G-code feed is modal: it persists until changed. "
           "Repeating it would lengthen the file by a third for no effect."),
     ("h", "Why the header carries the machine's own numbers"),
     ("p", "The header comments record the writer, the stroke count, the "
           "scale applied, the steps per millimetre for both axes, and how "
           "the pen codes are to be interpreted. None of it affects the "
           "machine, since comments are ignored. It is there because a "
           "G-code file outlives the program that wrote it, and a file "
           "whose scale and gearing are unknown cannot be checked against "
           "the machine it is about to be run on. Recording the steps per "
           "millimetre in particular makes it possible to tell afterwards "
           "whether a drawing came out the wrong size because the file was "
           "wrong or because the machine was configured differently."),
     ("h", "The warning comment"),
     ("p", "Where points had to be clamped, a warning comment is written "
           "into the file. That places the diagnosis where the evidence is: "
           "someone examining a drawing with a flattened edge looks at the "
           "file, and the explanation is in it. A warning printed to the "
           "console at the time of writing would have scrolled away long "
           "before the drawing was made."),
     ("h", "Parking at the end"),
     ("p", "The file ends by raising the pen, moving back to the origin, "
           "and issuing the program-end command. Parking matters "
           "practically: it leaves the machine in a known position, so the "
           "next file starts from the same place, and it takes the pen off "
           "the drawing, where a pen left resting bleeds ink into the "
           "paper."),
   ]),

 "ParseGcode": dict(
   inputs=[("path", "a G-code file to read")],
   output="the strokes recovered from it, as lists of points in "
          "millimetres.",
   theory=[
     ("h", "What it is for, and why it is written separately"),
     ("p", "Reads a G-code file back into strokes, by following the pen "
           "state and the coordinates through the file and collecting the "
           "points visited with the pen down. Its purpose is verification: "
           "comparing what comes back with what went in measures whether "
           "the file faithfully represents the intended geometry."),
     ("p", "The code comment makes the design decision explicit: it shares "
           "no code with the writer, deliberately. That is the whole value "
           "of the test. If both used a common routine to format and "
           "interpret coordinates, a fault in that routine would cancel "
           "between them and the comparison would pass while the file was "
           "wrong. Two independent implementations can only agree if the "
           "file itself is right, which is the property being tested."),
     ("h", "What the test can and cannot detect"),
     ("items", [
       "**It detects** rounding error from writing coordinates to three "
       "decimal places, a stroke reordering or reversal that lost a point, "
       "a clamp that moved geometry, and any error in the pen-state "
       "tracking, since a mistracked pen makes the reader collect the wrong "
       "points entirely.",
       "**It cannot detect** anything about the machine. The feed rates, "
       "the dwells, the acceleration and every mechanical effect are "
       "invisible to it. The measured agreement of better than 0.999 by "
       "area overlap is a statement about the file, not about the drawing.",
     ]),
     ("p", "Being clear about that boundary is the point of describing the "
           "test at all. A round-trip agreement this close is easy to "
           "misread as a statement that the robot draws accurately, and it "
           "is not: it says the arithmetic between a pen path and a G-code "
           "file loses essentially nothing, which locates any remaining "
           "error in the mechanics rather than in the software."),
     ("h", "How the reader tracks state"),
     ("p", "It maintains a current position and a pen flag. Comments are "
           "stripped, the command letter is taken from the first word, and "
           "coordinates are extracted by pattern. A pen-down command starts "
           "a new stroke at the current position; a pen-up command ends it; "
           "a move updates the position and, if the pen is down, appends "
           "the point."),
     ("p", "Starting the stroke at the current position on pen-down, rather "
           "than at the first move after it, is what makes the recovered "
           "stroke begin where the writer's travel move left the pen. "
           "Without it every stroke would be missing its first point, and "
           "the comparison would show a systematic error that was the "
           "reader's and not the file's."),
     ("note", "Strokes of fewer than two points are discarded, which "
              "matches the writer, since it skips such strokes too. A "
              "stroke still open at the end of the file is closed and kept, "
              "so a file that ends without a final pen-up does not lose its "
              "last stroke."),
   ]),

 "GantryNotConnectedError": dict(
   output="an error type raised when something needs the hardware and the "
          "hardware is not available.",
   theory=[
     ("h", "What it is for"),
     ("p", "Raised when the GPIO library is missing or the chip device "
           "cannot be opened, which is the ordinary situation on any "
           "computer that is not the machine's own controller."),
     ("h", "Why a named error rather than letting the import fail"),
     ("p", "An unguarded import of the hardware library would make this "
           "module unimportable off the machine, and since the test and "
           "evaluation scripts all import it, none of them could run "
           "anywhere else. A distinct error type lets the hardware "
           "functions fail cleanly when called, while the arithmetic half "
           "of the file remains usable."),
     ("p", "Being a distinct type rather than a generic error is what lets "
           "a caller distinguish \"there is no machine here\" from \"the "
           "machine is there and something went wrong\". The first is "
           "expected and should be reported as a plain message; the second "
           "is a fault."),
   ]),

 "NotCalibratedError": dict(
   output="an error type raised when an absolute move is attempted before "
          "homing.",
   theory=[
     ("h", "Why calibration is a precondition and not a convenience"),
     ("p", "A stepper motor has no idea where it is. It moves a commanded "
           "number of steps from wherever it happens to be, and nothing in "
           "the motor or the driver reports a position. So a command to "
           "move to a particular place on the bed is meaningless until the "
           "machine has established where that place is relative to its "
           "current position, which is what homing does."),
     ("p", "Executing such a command anyway is not a silent error. The "
           "machine drives toward a position computed from an unknown "
           "origin, which is as likely as not to be outside the bed, and it "
           "will drive into its own frame until the limit switches catch "
           "it. A distinct error type refusing the command is the "
           "appropriate response to a request that cannot be carried out "
           "safely."),
     ("note", "This is one of the few places in the project where a "
              "function refuses rather than falling back. The usual pattern "
              "is to return a degraded result, but there is no degraded "
              "version of a physical move: it either goes to the right "
              "place or it drives the machine somewhere arbitrary."),
   ]),

 "CHIP_PATH": dict(
   output="the device path of the GPIO controller.",
   theory=[
     ("h", "Background: what a GPIO chip is"),
     ("p", "**GPIO** stands for general-purpose input/output: pins on a "
           "single-board computer that software can drive high or low, or "
           "read as high or low. They are how a program controls hardware "
           "that has no other interface, which here means the stepper "
           "drivers, the limit switches and the pen motor."),
     ("p", "On Linux the pins are reached through a device file "
           "representing the controller. Opening it yields a handle through "
           "which individual lines are requested and then set or read. The "
           "path names which controller, and a board with several exposes "
           "several."),
     ("note", "This is a Linux path, which is part of why the hardware half "
              "must be guarded: on any other system the file does not "
              "exist, and the library that opens it is not installed."),
   ]),

 "PEN_SWITCH_PIN": dict(
   output="the GPIO line number of the switch that reports the pen's "
          "position.",
   theory=[
     ("h", "Why the pen has a sensor when the G-code path only tracks it"),
     ("p", "The written file has no choice but to remember the pen's state, "
           "because a file cannot ask a question. The machine can. This "
           "switch closes when the pen is raised, so the hardware half "
           "knows the pen's actual position rather than its assumed one."),
     ("p", "The difference is what allows the hardware to be correct where "
           "the file can only be consistent. SetPen compares the wanted "
           "state with the measured one and does nothing if they already "
           "agree, so a pen that starts in an unexpected position is "
           "handled rather than inverting everything that follows. It also "
           "allows a failure to be detected: if the motor runs and the "
           "switch never changes, something is mechanically wrong, and that "
           "can be reported instead of the machine continuing to draw with "
           "a pen that never came down."),
     ("note", "The switch reports only one of the two states directly: it "
              "is closed when up and open otherwise. That is sufficient, "
              "since there are only two states, but it means a fault in "
              "which the pen sticks halfway reads as \"down\". The overrun "
              "time below is what guards against the related case of "
              "stopping the motor the instant the switch first trips."),
   ]),

 "PEN_MOTOR_PIN": dict(
   output="the GPIO line number that drives the pen actuator.",
   theory=[
     ("h", "How the actuator is driven"),
     ("p", "Setting this line active runs the geared motor; setting it "
           "inactive stops it. There is no position control and no "
           "direction: the motor turns one way only, and a quarter turn of "
           "its cam moves the pen between up and down."),
     ("h", "Why the software must decide when to stop it"),
     ("p", "Because the motor has no limit of its own, running it too long "
           "carries the cam past the position wanted and begins the next "
           "quarter turn, which returns the pen to where it started. So the "
           "software runs the motor while watching the switch and stops it "
           "when the wanted state is reached. That is a closed loop, in "
           "contrast with the written file's open-loop timed dwell, and it "
           "is why the hardware path is more robust than the file path for "
           "the same operation."),
   ]),

 "ENDSTOP_PINS": dict(
   output="a mapping from each limit switch's name to its GPIO line number.",
   theory=[
     ("h", "What a limit switch is for"),
     ("p", "An **endstop** or limit switch is a mechanical switch at the "
           "end of an axis's travel, closed when the carriage reaches it. "
           "Four are fitted, at the minimum and maximum of each axis, and "
           "they serve two distinct purposes."),
     ("items", [
       "**Homing.** Since the motors cannot report their position, the "
       "only way to establish one is to drive until a switch is reached, "
       "which is a known place. Calibration drives to both ends of each "
       "axis, which also measures the travel between them.",
       "**Protection.** During any motion, a switch closing means the "
       "carriage has reached a limit it was not supposed to reach, which "
       "stops the machine. Without that, a mistaken coordinate drives the "
       "carriage into the frame and the motor stalls, skipping steps and "
       "losing the position entirely.",
     ]),
     ("h", "Why they are named rather than numbered"),
     ("p", "The switches are held in a mapping keyed by name, so the code "
           "refers to X_MIN rather than to a pin number. The names carry "
           "the information the logic needs: which axis a switch belongs to "
           "and which direction to retreat when it trips. The retreat "
           "direction table is keyed by the same names, so the two cannot "
           "drift out of correspondence, which a pair of parallel numeric "
           "lists could."),
   ]),

 "INVERT_X": dict(
   output="true, meaning the X axis's direction signal is reversed.",
   theory=[
     ("h", "What it corrects"),
     ("p", "A stepper driver has a direction pin: one level turns the motor "
           "one way and the other the opposite. Which physical direction "
           "each level produces depends on how the motor's coils happen to "
           "be wired to the driver and on how the motor is mounted relative "
           "to the axis. There is no convention that settles it, and the "
           "wiring is a four-conductor connector that can be fitted either "
           "way round."),
     ("p", "So the mapping between the signal and the physical direction is "
           "determined by trying it, and the result is recorded here. The "
           "alternative, rewiring the motor until the default convention "
           "happens to be right, achieves nothing a single flag cannot."),
     ("h", "How the flag is applied"),
     ("p", "AxisDirValue combines the wanted direction with this flag using "
           "an exclusive-or, so a wanted positive direction with inversion "
           "set gives the inactive level. Keeping the correction in one "
           "function means every motion path inherits it, and a new motion "
           "routine cannot forget it as long as it asks for a direction by "
           "meaning rather than by signal level."),
   ]),

 "INVERT_Y": dict(
   output="true, meaning the Y axis's direction signal is reversed as well.",
   theory=[
     ("h", "Why each axis has its own flag"),
     ("p", "The two axes are wired and mounted independently, so the "
           "correction each needs is independent. Both happen to require "
           "inversion on this machine, which is coincidence rather than "
           "something that could be assumed: a single shared flag would "
           "work here and would be wrong on a machine where only one axis "
           "needed it."),
   ]),

 "MICROSTEPS": dict(
   output="16, the microstepping division set on the drivers.",
   theory=[
     ("h", "What it records"),
     ("p", "The driver divides each of the motor's full steps into this "
           "many sub-steps, which is set by jumpers on the driver board. "
           "The value here must match those jumpers: the software cannot "
           "read them, so a mismatch produces motion scaled by a factor of "
           "two or more with no error of any kind, only a drawing of the "
           "wrong size."),
     ("h", "Why sixteen"),
     ("p", "It is the finest division the A4988 driver used offers, and "
           "the reason to take the finest is smoothness rather than "
           "resolution. At 1/16 the motor advances in increments small "
           "enough that the motion is effectively continuous, which "
           "matters for drawing curves a few millimetres across: coarser "
           "stepping is audible as a buzz and visible as faceting on a "
           "curve. The cost is that sixteen times as many step pulses must "
           "be issued per millimetre, which for a software-timed stepper "
           "sets the maximum speed."),
     ("note", "This constant duplicates the microstepping field in "
              "GantryConfig, because the two halves of the file were "
              "written to be independent and each carries its own copy of "
              "the machine's numbers. The duplication is a real hazard: the "
              "two must be changed together, and nothing enforces it."),
   ]),

 "STEPS_PER_REV": dict(
   output="3200, the microsteps per motor revolution.",
   theory=[
     ("h", "How it is obtained"),
     ("p", "The motor's 200 full steps per revolution multiplied by the "
           "microstepping. Computing it rather than stating 3200 keeps the "
           "two factors visible, each of which is a checkable fact about "
           "the hardware, and means a change to the microstepping jumpers "
           "is one edit."),
     ("note", "200 appears here as a literal rather than as a named "
              "constant, which is the one piece of the derivation not "
              "spelled out. It is the motor's step angle of 1.8 degrees "
              "divided into 360."),
   ]),

 "STEPS_PER_MM": dict(
   output="80.0, a provisional steps-per-millimetre figure.",
   theory=[
     ("h", "What it is for, and why it is provisional"),
     ("p", "A placeholder used only before calibration has run. Once the "
           "machine has homed, it has *measured* its own steps per "
           "millimetre, by counting the steps between the two limit "
           "switches and dividing by the known distance between them, and "
           "the measured figure replaces this one."),
     ("h", "Why a measurement beats the calculation"),
     ("p", "The calculated 80 assumes the gearing is exactly as specified, "
           "and it is not quite. A GT2 belt's pitch varies slightly with "
           "tension and temperature, a pulley's effective radius depends on "
           "where the belt sits in its teeth, and both are manufactured to "
           "a tolerance. Measuring the actual travel captures all of that "
           "at once, without needing to know which part contributed what. "
           "The measured value is typically within a percent or two of the "
           "calculated one, which over a 200 mm page is a few millimetres."),
     ("note", "The calculated value remains in use in one place: "
              "RunGcodeFile reads the starting position using this constant "
              "rather than the calibrated figure, so the position it begins "
              "from is slightly off even after calibration. The subsequent "
              "moves use the calibrated value, so the error does not "
              "accumulate, but the first move of a job is affected."),
   ]),

 "CIRCLE_STEP_DELAY_S": dict(
   output="0.000002, the shortest delay between step pulses, in seconds.",
   theory=[
     ("h", "What it bounds"),
     ("p", "A step pulse is produced by setting a pin high, waiting, and "
           "setting it low. This is the wait, and it therefore sets the "
           "fastest the motor can be driven. Two microseconds between edges "
           "is near the practical floor for a driver's minimum pulse width "
           "and well beyond what the motor can follow."),
     ("h", "Why it is used as a floor rather than a speed"),
     ("p", "Motion is normally timed from the requested feed rate: the "
           "delay is computed so that the move takes the time the feed "
           "implies. This constant is the lower bound on that computation, "
           "so a feed rate too high to be achieved is silently reduced to "
           "the fastest the machine can step instead of producing a "
           "negative or zero delay."),
     ("note", "The delay is produced by asking the operating system to "
              "sleep, which on a general-purpose Linux system is accurate "
              "to perhaps tens of microseconds rather than to two. So at "
              "short delays the actual rate is set by the operating "
              "system's scheduling rather than by this value, and the real "
              "maximum speed is lower and less regular than the constant "
              "suggests. Step timing of this kind properly belongs in "
              "dedicated firmware or a hardware timer."),
   ]),

 "DEFAULT_CIRCLE_RADIUS_MM": dict(
   output="8.0, the radius of the test circle in millimetres.",
   theory=[
     ("h", "What it is for"),
     ("p", "The machine has a test mode that drives a continuous circle, "
           "and this is its default radius. A circle is a useful test "
           "pattern because it exercises both axes continuously and in "
           "varying proportion, so it reveals faults that a straight line "
           "does not: a mismatch between the two axes' steps per "
           "millimetre shows as an ellipse, and backlash in a belt shows as "
           "a flat spot where an axis reverses."),
   ]),

 "X_SWITCH_TRAVEL_MM": dict(
   output="204.0, the measured distance between the two X limit switches.",
   theory=[
     ("h", "Why this number is the key to calibration"),
     ("p", "Calibration counts the steps between the two switches, which "
           "gives a travel in steps. Converting that into steps per "
           "millimetre requires knowing the travel in millimetres, and that "
           "is this constant, measured once from the physical machine with "
           "a rule."),
     ("eq", r"\mathrm{stepsPerMm}_X \;=\; "
            r"\frac{\mathrm{travelSteps}_X}{\mathrm{X\_SWITCH\_TRAVEL\_MM}}",
      [(r"\mathrm{stepsPerMm}_X", "the calibrated scaling for the X axis"),
       (r"\mathrm{travelSteps}_X", "the steps counted between the two "
                                   "switches during homing"),
       (r"\mathrm{X\_SWITCH\_TRAVEL\_MM}", "this constant, the same "
                                           "distance in millimetres")],
      None),
     ("p", "So the accuracy of every subsequent move rests on this one "
           "hand-measured figure. An error of one percent in it scales the "
           "whole drawing by one percent, and nothing in the system can "
           "detect that, because the machine has no other reference for "
           "absolute distance. It is the single value in the project that "
           "must be got right by measurement rather than by reasoning."),
     ("note", "The distance is between the switches' *trip points*, not "
              "between the switch bodies or the ends of the rail. A switch "
              "trips a little before the carriage reaches it, by however "
              "much its lever travels, so the figure has to be taken from "
              "where motion actually stops."),
   ]),

 "Y_SWITCH_TRAVEL_MM": dict(
   output="262.0, the same distance for the Y axis.",
   theory=[
     ("h", "Why the two axes differ"),
     ("p", "The axes are physically different lengths, 204 and 262 "
           "millimetres, which is simply how the machine is built. Each is "
           "calibrated against its own figure, giving each its own steps "
           "per millimetre. Keeping them separate also allows the two axes "
           "to come out with slightly different scaling, which they will "
           "even with identical gearing, because belt tension and pulley "
           "tolerance differ between them. Forcing a single shared value "
           "would make circles come out as ellipses."),
   ]),

 "EDGE_TOLERANCE_MM": dict(
   output="5.0, the margin kept clear of every limit switch, in "
          "millimetres.",
   theory=[
     ("h", "What it is for"),
     ("p", "After homing, the usable area is defined as the measured travel "
           "less this margin at each end of each axis, and the coordinate "
           "origin is placed this far in from the minimum switches. So the "
           "machine never deliberately drives within five millimetres of "
           "any switch."),
     ("h", "Why a margin is necessary"),
     ("p", "Driving to a position where a switch is about to trip is "
           "hazardous in a specific way. A limit switch tripping during "
           "normal motion is treated as an emergency: the drive stops, the "
           "pen lifts and the job aborts. If the usable area reached the "
           "switches exactly, then ordinary rounding, a step lost to a "
           "stiff belt, or the thermal expansion of the frame would "
           "occasionally trip a switch during a legitimate move, and a job "
           "would abort partway through a drawing for no real fault."),
     ("p", "Five millimetres is comfortably more than any of those effects "
           "and costs ten millimetres of travel on each axis, under five "
           "percent of the bed. That is a cheap price for the difference "
           "between a boundary that is occasionally crossed and one that is "
           "crossed only when something is genuinely wrong."),
     ("eq", r"\mathrm{usableWidth} \;=\; \mathrm{switchTravel} \;-\; "
            r"2 \cdot \mathrm{EDGE\_TOLERANCE\_MM}",
      [(r"\mathrm{usableWidth}", "the width the machine will address"),
       (r"\mathrm{switchTravel}", "the measured distance between the "
                                  "switches"),
       ("2", "the margin applies at both ends")],
      None),
     ("note", "Because the origin is offset by this margin, the conversion "
              "between a coordinate and a step count adds it back: a "
              "commanded 0 corresponds to the margin's worth of steps from "
              "the minimum switch, not to zero steps. Both _GcodeLinearMove "
              "and the position readback have to apply that offset, and a "
              "mismatch between them would put every move out by five "
              "millimetres."),
   ]),

 "HOMING_STEP_DELAY_S": dict(
   output="0.0008, the per-step delay used while homing, in seconds.",
   theory=[
     ("h", "Why homing is slow"),
     ("p", "This delay is four hundred times the minimum, making homing "
           "much slower than ordinary motion. Three reasons require that."),
     ("items", [
       "**The switch must be detected before the carriage reaches the "
       "frame.** At high speed many steps elapse between consecutive "
       "switch readings, and the carriage can travel several millimetres "
       "past the trip point, which on an axis whose switch sits close to "
       "the frame means driving into it.",
       "**The stop must not lose steps.** Stopping a fast-moving axis "
       "abruptly can make the motor skip, and a skipped step during homing "
       "corrupts the measurement that calibration exists to produce.",
       "**The direction is unknown at the start.** Homing drives until "
       "something stops it, with no knowledge of how far away that is, so "
       "the motion is deliberately cautious.",
     ]),
     ("p", "Since homing happens once per session and takes a few seconds, "
           "the cost is negligible, which makes this an easy trade: all the "
           "speed in exchange for reliability on the one measurement "
           "everything else depends on."),
   ]),

 "HOMING_MAX_STEPS": dict(
   output="200000, the most steps homing will take before giving up.",
   theory=[
     ("h", "What it protects against"),
     ("p", "Homing drives until a switch trips. If no switch ever trips, "
           "because one is disconnected, miswired or mechanically stuck, "
           "the loop would run forever while the motor drives the carriage "
           "into the frame and stalls there. This bound turns an endless "
           "fault into a reported one."),
     ("h", "Why this many"),
     ("p", "At 80 steps per millimetre, 200000 steps is 2500 millimetres, "
           "about ten times the longest axis. So a correctly functioning "
           "machine cannot approach the limit however far from the switch "
           "it starts, while a broken one reaches it in under a minute at "
           "the homing delay. The margin is deliberately generous because "
           "the consequence of setting it too low, a false failure on a "
           "machine that was simply far from its switch, is worse than "
           "taking longer to report a real one."),
     ("note", "Exceeding it raises an error naming the wiring as the likely "
              "cause, which is the right message: by far the commonest "
              "reason for a switch never tripping is a connector that has "
              "worked loose, and the person reading the message is standing "
              "next to the machine."),
   ]),

 "ENDSTOP_DEBOUNCE_S": dict(
   output="0.006, the gap between confirmation samples of a switch, in "
          "seconds.",
   theory=[
     ("h", "Background: contact bounce and electrical noise"),
     ("p", "A mechanical switch does not close cleanly. Its contacts strike "
           "and separate several times over a few milliseconds before "
           "settling, so a single reading taken during that window may "
           "catch it either way. This is **contact bounce**. Separately, "
           "the wires running to the switches are long and run alongside "
           "the motor cables, which carry switched currents of an amp or "
           "more, and that induces voltage spikes on the switch lines."),
     ("p", "Either effect can make a switch read as closed when it is not. "
           "Since a switch reading closed triggers an emergency stop, a "
           "spurious reading aborts a drawing that was proceeding "
           "correctly, which on a long page is a real cost."),
     ("h", "The remedy: sample repeatedly, spaced out in time"),
     ("p", "A switch is believed only if it reads closed on several "
           "consecutive samples with this delay between them. The spacing "
           "is what makes it work: bounce and induced noise are brief, "
           "lasting microseconds to a millisecond or two, so samples six "
           "milliseconds apart are effectively independent of one another "
           "and a transient cannot be present in all of them. Sampling "
           "rapidly instead would read the same disturbance several times "
           "and confirm it."),
   ]),

 "ENDSTOP_CONFIRM_SAMPLES": dict(
   output="3, how many consecutive readings are required before a switch is "
          "believed.",
   theory=[
     ("h", "The trade-off"),
     ("p", "More samples reject noise more reliably and delay a genuine "
           "detection. The delay matters because the axis keeps moving "
           "while the switch is being confirmed: with three samples six "
           "milliseconds apart the confirmation takes twelve milliseconds, "
           "during which a carriage travelling at the drawing feed covers "
           "about 0.2 millimetres. Three is enough that a transient "
           "surviving all three is very unlikely, while the overshoot stays "
           "well inside the five-millimetre margin kept clear of the "
           "switches."),
     ("h", "Why it exits early on a disagreement"),
     ("p", "The check stops as soon as any sample reads open, concluding "
           "that the switch is not pressed. This is the asymmetry that "
           "makes the cost acceptable: the expensive path, taking all three "
           "samples, is only followed when the first reading is closed, "
           "which is rare. In the normal case of a switch reading open the "
           "function returns after one reading, so the debounce costs "
           "nothing during ordinary motion, and it is called before every "
           "single step."),
   ]),

 "RETRACT_MM": dict(
   output="5.0, how far to back off a tripped switch, in millimetres.",
   theory=[
     ("h", "Why a retract is needed after a stop"),
     ("p", "When a switch trips, the carriage is resting against it. "
           "Leaving it there is unworkable for two reasons. The switch "
           "continues to read closed, so every subsequent check sees a "
           "tripped endstop and refuses to move, including the move needed "
           "to get clear: the machine is stuck. And a switch held closed "
           "under pressure can be damaged over time."),
     ("p", "So after an emergency stop the axis is stepped away from the "
           "switch until it releases, by this distance. Five millimetres "
           "matches the edge margin, which is sensible: it is the distance "
           "that puts the carriage back inside the usable area, so the "
           "machine ends up where it is allowed to be rather than just "
           "clear of the switch."),
     ("h", "Why the retract watches the other switches"),
     ("p", "The retract loop checks for any switch other than the one being "
           "retracted from, and stops if one trips. That case means "
           "something is genuinely wrong, a switch stuck closed or an axis "
           "shorter than believed, and continuing to drive would push "
           "against a second limit. Stopping and reporting is the correct "
           "response, since the situation cannot be resolved by moving."),
   ]),

 "PEN_UP_OVERRUN_S": dict(
   output="0.04, extra running time after the pen switch first trips, in "
          "seconds.",
   theory=[
     ("h", "The problem it solves"),
     ("p", "The switch trips the moment the pen reaches the up position, "
           "and stopping the motor there leaves the cam exactly at the "
           "point where the switch just closed. That is the worst place to "
           "stop. A small amount of backlash, a vibration from the next "
           "move, or the pen's own weight can rotate the cam back past the "
           "trip point, and the pen drops onto the page during a travel "
           "move, drawing a line across the drawing."),
     ("p", "Running on for a further 40 milliseconds carries the cam past "
           "the trip point into the middle of its up region, where neither "
           "backlash nor vibration can bring it back. The principle is "
           "general: when a detector marks the boundary of a safe region, "
           "stop inside the region rather than on its edge."),
     ("h", "Why only on the way up"),
     ("p", "The overrun is applied when raising the pen and not when "
           "lowering it. Down is the position gravity and the pen's spring "
           "favour, so it is stable without help, whereas up is held "
           "against them. Overrunning on the way down would also press the "
           "pen harder into the paper than intended, which broadens the "
           "stroke and can damage the nib."),
   ]),

 "GCODE_MIN_FEED_MM_S": dict(
   output="0.1, the slowest feed rate accepted, in millimetres per second.",
   theory=[
     ("h", "What it protects against"),
     ("p", "The step delay is computed by dividing the move's duration by "
           "its step count, and the duration is the distance divided by the "
           "feed. A feed of zero therefore divides by zero. A file "
           "containing `F0`, whether through a typing error or a "
           "generator's fault, would crash the job partway through a "
           "drawing."),
     ("h", "Why a floor rather than a rejection"),
     ("p", "Clamping to a very slow but finite rate means such a line "
           "produces a slow move instead of a failure. That is the better "
           "behaviour here because the alternative, aborting, loses the "
           "whole drawing, whereas one unusually slow move is a visible "
           "pause and nothing more. At a tenth of a millimetre per second "
           "the move is clearly wrong and clearly deliberate, so it is "
           "noticeable without being destructive."),
   ]),

 "_GCODE_WORD_RE": dict(
   output="the compiled pattern that splits a G-code line into letters and "
          "numbers.",
   theory=[
     ("h", "What a G-code word is"),
     ("p", "A G-code line is a sequence of **words**, each a letter "
           "followed by a number: `G1`, `X12.5`, `F900`. This pattern "
           "matches one such word, capturing the letter and the number "
           "separately, and finding every match in a line splits it into "
           "its parts."),
     ("h", "Why the pattern allows space between them"),
     ("p", "It permits whitespace between the letter and the number, so "
           "`X 12.5` parses as well as `X12.5`. Both forms occur, because "
           "different generators format differently and some files are "
           "hand-edited. A reader of a standard interchange format should "
           "accept the variations the standard permits rather than only the "
           "form it writes itself."),
     ("h", "Why the number pattern is as it is"),
     ("p", "It accepts an optional minus sign, any number of digits, an "
           "optional decimal point and more digits, so it matches `5`, "
           "`-5`, `5.25` and `.5` alike. The last of those is why the "
           "integer part is optional: a coordinate written as `.5` is "
           "valid and would otherwise be read as a word with no number at "
           "all."),
     ("note", "The pattern is compiled once at module level rather than on "
              "each use. A compiled pattern is parsed into its matching "
              "machinery once instead of on every call, which matters here "
              "because this is applied to every line of a file that may run "
              "to tens of thousands."),
   ]),

 "_RETREAT_SIGN": dict(
   output="a mapping from each switch's name to the direction to move to "
          "get off it.",
   theory=[
     ("h", "What it encodes"),
     ("p", "Which way is away from each switch: positive for the minimum "
           "switches and negative for the maximum ones. A minimum switch "
           "sits at the low end of the axis, so escaping it means moving "
           "toward higher coordinates, and the reverse for a maximum."),
     ("h", "Why it is a table and not computed from the name"),
     ("p", "The sign could be derived by testing whether the name ends in "
           "MIN. A table states it instead, which is better for a value "
           "with physical consequences: driving the wrong way off a switch "
           "pushes the carriage harder into its limit. A table is read "
           "directly against the machine, whereas a derivation depends on a "
           "naming convention holding, and a switch named differently would "
           "silently produce the wrong direction rather than a missing-key "
           "error."),
   ]),

 "Calibration": dict(
   output="a dictionary holding the measured machine geometry and whether "
          "homing has been done.",
   theory=[
     ("h", "What it holds"),
     ("p", "The measured steps per millimetre for each axis, the usable "
           "area in millimetres, and a flag saying whether calibration has "
           "run. Everything in it is a measurement rather than a "
           "configuration: the values are produced by driving the machine "
           "and counting, not by being stated."),
     ("h", "Why the flag is checked rather than the values"),
     ("p", "Code requiring calibration tests the flag, not whether the "
           "steps-per-millimetre values look plausible. That is the right "
           "test because the values have defaults that are perfectly "
           "plausible, being the calculated 80: there is no value of them "
           "that means \"not yet measured\". An explicit flag can express "
           "that where the data cannot."),
     ("note", "This is module-level mutable state, shared by every function "
              "in the hardware half, which is the pattern the whole bottom "
              "half uses. It is defensible here in a way it would not be "
              "elsewhere: there is exactly one physical machine, so a "
              "single shared record of its state is an accurate model of "
              "the situation, and two independent copies would be able to "
              "disagree about a machine that can only be in one state."),
   ]),
}


#the rest of the chapter, in order
for _part in ("codetheory_reproduce2",):
    try:
        _m = __import__(_part)
    except ModuleNotFoundError:
        continue
    FUNCS.update(_m.FUNCS)
