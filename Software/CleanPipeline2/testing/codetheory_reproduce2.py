"""reproduce.py, part 2: driving the machine.

The live state, connecting to the pins, the safety checks, stepping the
motors, homing, and executing a G-code file on the real gantry.
"""

FUNCS = {

 "req": dict(
   output="the live GPIO line request once a connection has been made, or "
          "nothing before that.",
   theory=[
     ("h", "What it holds"),
     ("p", "The handle through which every pin is set and read. It is "
           "obtained by requesting a set of lines from the GPIO chip, and "
           "until that has happened it is nothing, which is also how the "
           "module knows whether it is connected."),
     ("h", "Why requesting lines is exclusive"),
     ("p", "The Linux GPIO interface grants a requested line exclusively to "
           "the process that asked for it: no other program can drive the "
           "same pin while the request is held. That is essential here, "
           "since two programs both driving a stepper's step pin would "
           "produce motion neither intended. It also means the request must "
           "be released on shutdown, or the pins remain claimed by a "
           "process that has exited and the next run cannot acquire them."),
     ("note", "Using the handle's presence as the connection flag means "
              "there is one piece of state rather than two, so the flag "
              "cannot disagree with reality. A separate boolean would be "
              "able to say connected while the handle was absent, which is "
              "the kind of inconsistency that produces an error far from "
              "its cause."),
   ]),

 "CurrentMode": dict(
   output="which of the manual control modes the machine is in.",
   theory=[
     ("h", "What the modes are"),
     ("p", "The hardware half includes a small interactive console for "
           "testing, and this says what it is currently doing: turning the "
           "motors at a set speed, drawing a test circle, running a G-code "
           "file, or homing. The main loop reads it each pass and acts "
           "accordingly."),
     ("h", "Why a mode variable rather than separate loops"),
     ("p", "A single loop that consults a mode can be interrupted and "
           "redirected from the keyboard thread at any moment, because the "
           "thread only has to change one variable. Separate loops per mode "
           "would each need their own mechanism for being told to stop, and "
           "switching between them would mean leaving one loop and entering "
           "another, during which the motors would be in an undefined "
           "state."),
     ("note", "The whole manual console exists for bringing the machine up "
              "and diagnosing it, not for drawing. A drawing job is run by "
              "RunGcodeFile, which does not consult the mode at all."),
   ]),

 "targetRpm": dict(
   output="0.0, the wanted motor speed in revolutions per minute.",
   theory=[
     ("h", "Why the initial value matters"),
     ("p", "The code comment is explicit that this must start at zero, and "
           "the reason is a real hazard rather than tidiness. The main loop "
           "begins stepping as soon as it starts, at whatever speed this "
           "says. A non-zero initial value would move the gantry the "
           "instant the program ran, before the operator had asked for "
           "anything and quite possibly before homing, so the machine would "
           "drive from an unknown position toward an unknown one."),
     ("p", "A device that moves on startup is a genuine safety problem, "
           "and the general principle is that a machine's initial state "
           "should be the one in which it does nothing. The same reasoning "
           "sets the step pins inactive and the pen motor off when the "
           "pins are first claimed."),
   ]),

 "stepDelayS": dict(
   output="0.0, the pause between step pulses derived from the wanted "
          "speed.",
   theory=[
     ("h", "What it is"),
     ("p", "The delay the main loop waits between step edges, computed from "
           "the requested speed by CalculateDelay. A value of zero means "
           "stopped, which is consistent with the initial speed of zero."),
     ("note", "A delay and a speed are inversely related, so this variable "
              "and targetRpm hold the same information in two forms. The "
              "derived form is cached because it is needed in the innermost "
              "loop, where recomputing a division per step would be "
              "wasteful. The hazard of a cached derivative is that the two "
              "can disagree, which is why the only function that sets the "
              "speed also sets this."),
   ]),

 "running": dict(
   output="true while the main loop should continue.",
   theory=[
     ("h", "How a thread is asked to stop"),
     ("p", "The main loop tests this each pass and exits when it becomes "
           "false, which the keyboard thread sets on a quit command. This is "
           "a **cooperative shutdown**: the loop is asked to stop and does "
           "so at a point of its own choosing, rather than being terminated "
           "from outside."),
     ("h", "Why that matters for a machine"),
     ("p", "Terminating a thread part way through a step pulse would leave "
           "a step pin high, so the driver would hold the coil energised "
           "indefinitely, heating the motor. Letting the loop finish its "
           "current iteration guarantees every pulse completes and the "
           "shutdown path can then set the pins to a safe state and release "
           "them. For software a forced stop is usually recoverable; for a "
           "machine holding a physical state it is not."),
   ]),

 "CurrentXSteps": dict(
   output="0, the X axis position in steps from the homing reference.",
   theory=[
     ("h", "Why the position is tracked in software"),
     ("p", "This is the machine's only knowledge of where it is. A stepper "
           "motor has no position feedback: it moves when pulsed and "
           "reports nothing. So the position is maintained by counting, "
           "incremented or decremented on every single step issued by every "
           "motion routine."),
     ("h", "What that implies, and why it is accepted"),
     ("p", "The count is the position only as long as every pulse produced "
           "a real step. A motor that stalls, from an obstruction or an "
           "over-fast acceleration, skips steps that were counted, and the "
           "recorded position silently becomes wrong by that amount with no "
           "way to detect it. This is **open-loop** control, and it is "
           "universal in low-cost motion systems because closing the loop "
           "requires an encoder on every axis."),
     ("p", "What makes it workable is that the error does not accumulate "
           "unnoticed indefinitely: homing re-establishes the position from "
           "a physical reference, and the limit switches catch the case "
           "where the error is large enough to drive into a limit. Within "
           "one drawing, the loads are light and the feeds modest, so "
           "stalling is unlikely."),
     ("note", "A step count is an integer, so counting cannot accumulate "
              "rounding error the way tracking a position in millimetres "
              "would. That is the reason the count is kept in steps and "
              "converted to millimetres only when a coordinate is needed."),
   ]),

 "CurrentYSteps": dict(
   output="0, the Y axis position in steps.",
   theory=[
     ("h", "Why the axes are counted separately"),
     ("p", "Each axis has its own motor, its own switches and its own "
           "calibrated scaling, so each needs its own count. Keeping them "
           "separate also means the two can be converted with different "
           "steps-per-millimetre figures, which they must be: the "
           "calibration measures each axis independently and the two differ "
           "slightly even with identical gearing."),
   ]),

 "theta": dict(
   output="0.0, the current angle of the test circle in radians.",
   theory=[
     ("h", "What it is for"),
     ("p", "The circle test mode advances this angle a little each pass and "
           "moves to the corresponding point on the circle, so the carriage "
           "traces a continuous circle. It persists between loop passes "
           "because the position must continue from where it reached, not "
           "restart."),
     ("note", "This is a convenience for mechanical testing rather than "
              "part of drawing. A circle run continuously is the most "
              "useful single test pattern for a gantry, since it exercises "
              "both axes in constantly varying proportion and makes axis "
              "mismatch and belt backlash visible as a distorted shape."),
   ]),

 "LastCircleTime": dict(
   output="the time the circle test last advanced.",
   theory=[
     ("h", "Why the time is recorded rather than the loop simply sleeping"),
     ("p", "The circle must advance at a steady rate, and the main loop "
           "does other work whose duration varies. Recording when the last "
           "advance happened and comparing against the current time lets "
           "the loop advance when enough time has elapsed, independently of "
           "how long the rest of the pass took. Sleeping for a fixed "
           "interval instead would make the real period the interval plus "
           "the work, so the circle's speed would depend on what else the "
           "loop was doing."),
     ("p", "A high-resolution performance counter is used rather than "
           "wall-clock time, because it is monotonic: it cannot jump "
           "backwards if the system clock is adjusted, which would make an "
           "elapsed-time comparison give a negative result."),
   ]),

 "CircleIntervalS": dict(
   output="0.0015, how often the test circle advances, in seconds.",
   theory=[
     ("h", "What it sets"),
     ("p", "The interval between successive points of the circle, which "
           "together with the angular step sets how fast the circle is "
           "traced. At 1.5 milliseconds the motion appears continuous."),
   ]),

 "CircleRadiusMm": dict(
   output="the current test circle radius in millimetres.",
   theory=[
     ("h", "Why it is separate from the default"),
     ("p", "It starts from the default constant and can be changed from the "
           "keyboard during testing, so the current value and the default "
           "are distinct: the default is what the machine starts with and "
           "this is what it is doing now. Varying the radius during a test "
           "is useful because the faults a circle reveals show up "
           "differently at different sizes, backlash being most visible on "
           "a small circle and axis mismatch on a large one."),
   ]),

 "CircleRadiusSteps": dict(
   output="the same radius converted into steps.",
   theory=[
     ("h", "Why it is precomputed"),
     ("p", "The circle loop needs the radius in steps on every pass, and "
           "converting once avoids repeating the multiplication inside the "
           "motion loop."),
     ("note", "The conversion uses the uncalibrated constant rather than "
              "the measured scaling, so a circle drawn after calibration is "
              "slightly the wrong size. For a mechanical test pattern that "
              "does not matter, since what is being looked at is the shape "
              "rather than the dimensions, but it is the same duplication "
              "hazard noted under STEPS_PER_MM."),
   ]),

 "estopped": dict(
   output="false, becoming true once an emergency stop has fired.",
   theory=[
     ("h", "What it enforces"),
     ("p", "Once an emergency stop has happened, this stays set until "
           "someone clears it from the console. Every motion path checks "
           "it, so nothing moves in the meantime."),
     ("h", "Why the stop latches instead of clearing itself"),
     ("p", "This is the important property. After the retract, the switch "
           "is no longer pressed, so a test that only looked at the "
           "switches would conclude all was well and allow motion to "
           "resume. But the reason the switch tripped has not been "
           "addressed: the coordinate was wrong, or the belt slipped, or "
           "the work is not where it was thought to be. Resuming would "
           "drive into the same limit again."),
     ("p", "A **latching** fault requires a human to acknowledge it, which "
           "is the standard arrangement for a machine's emergency stop and "
           "for the same reason: the stop indicates that the machine's idea "
           "of the world is wrong, and only a person can establish whether "
           "it is right again."),
   ]),

 "TrippedSwitch": dict(
   output="which switch caused the emergency stop, or nothing.",
   theory=[
     ("h", "Why the identity is kept"),
     ("p", "Knowing that a stop occurred is not enough to act on it. Which "
           "switch tripped says which axis and which end, which tells the "
           "retract which way to move and tells the operator where to look. "
           "The message printed on a stop names the switch for exactly that "
           "reason."),
   ]),

 "Connect": dict(
   output="nothing; it claims the GPIO lines or raises if it cannot.",
   theory=[
     ("h", "What it does"),
     ("p", "Requests every pin the machine uses from the GPIO chip in a "
           "single call, declaring each as an input or an output and giving "
           "each output its initial level."),
     ("h", "Why all the pins are requested at once"),
     ("p", "One request for the whole set is better than ten separate ones. "
           "It is atomic: either every pin is obtained or none is, so the "
           "module cannot end up holding half the pins it needs, with some "
           "motion possible and some not. And it establishes every initial "
           "level in the same operation, so there is no window during which "
           "some outputs are configured and others are floating."),
     ("h", "The initial levels, and why they are what they are"),
     ("items", [
       "**Step pins inactive.** A step is a transition, so holding them "
       "low means no step is in progress and none is commanded.",
       "**Pen motor inactive.** The actuator is off, so the pen stays "
       "wherever it is rather than beginning to turn.",
       "**Direction pins active.** The level is arbitrary, since every "
       "motion sets the direction before stepping, and a direction pin by "
       "itself causes no movement.",
       "**Every switch an input.** Driving a switch line as an output "
       "would fight the switch and could damage the pin.",
     ]),
     ("p", "The common thread is that the initial state must be one in "
           "which nothing moves, which is the same principle that puts the "
           "initial speed at zero."),
     ("h", "Why failure is translated"),
     ("p", "Any failure to open the chip is caught and re-raised as the "
           "module's own not-connected error, with the original attached. "
           "The handle is set back to nothing first, so a failed connection "
           "does not leave the module believing it is connected. "
           "Translating the error lets a caller distinguish an absent "
           "machine, which is expected on a laptop, from a fault on a "
           "machine that is present, without having to know what the GPIO "
           "library raises."),
     ("note", "Calling it twice returns immediately rather than requesting "
              "the lines again, which would fail since the lines are held "
              "exclusively by this same process. Making repeat calls "
              "harmless lets every entry point call it without tracking "
              "whether someone else already has."),
   ]),

 "Disconnect": dict(
   output="nothing; it sets the outputs safe and releases the lines.",
   theory=[
     ("h", "The order of operations"),
     ("p", "The step pins and the pen motor are set inactive first, and "
           "only then are the lines released. The order is necessary: once "
           "a line is released, its level is no longer controlled, and a "
           "line released while active may stay active. Releasing a step "
           "pin while high would leave the driver holding a coil "
           "energised, which heats the motor with nothing running to notice."),
     ("h", "Why the release is in a finally clause"),
     ("p", "The setting of safe levels is wrapped so that the release "
           "happens whether or not it succeeded. If a pin cannot be set, "
           "because the chip has gone away or the device was unplugged, the "
           "lines must still be released, or they stay claimed by a process "
           "that has exited and the next run cannot acquire them. Releasing "
           "a resource must not depend on the success of the tidying done "
           "before it."),
   ]),

 "IsConnected": dict(
   output="whether the GPIO lines are currently claimed.",
   theory=[
     ("h", "What it is for"),
     ("p", "Tests whether the handle exists. Callers use it to report the "
           "machine's availability without attempting an operation that "
           "would raise, which is what a user interface needs: it must say "
           "whether the machine is there, not find out by trying to move "
           "it."),
   ]),

 "CalculateDelay": dict(
   inputs=[("rpm", "the wanted motor speed in revolutions per minute")],
   output="nothing; it sets the module's step delay.",
   theory=[
     ("h", "The conversion"),
     ("eq", r"t_{\mathrm{delay}} \;=\; \frac{1}{2} \cdot "
            r"\frac{60}{|\mathrm{rpm}| \cdot \mathrm{stepsPerRev}}",
      [(r"t_{\mathrm{delay}}", "the pause between step pin edges, in "
                               "seconds"),
       (r"\mathrm{rpm}", "the wanted speed in revolutions per minute"),
       (r"\mathrm{stepsPerRev}", "microsteps per revolution, 3200 here"),
       ("60", "seconds per minute, converting the speed into revolutions "
              "per second"),
       (r"\tfrac{1}{2}", "the halving, explained below")],
      None),
     ("p", "Read it from the inside out. The speed in revolutions per "
           "minute divided by 60 is revolutions per second; multiplied by "
           "the steps per revolution it is steps per second; its reciprocal "
           "is seconds per step. That is the time one whole step must "
           "occupy."),
     ("h", "Why the result is halved"),
     ("p", "A step is produced by two transitions: the pin goes high, waits, "
           "goes low, waits. So the delay is used twice per step, and each "
           "wait must be half the step period for the step to take the time "
           "intended. Omitting the halving would run the motor at half the "
           "requested speed, which is the kind of error that looks like a "
           "mechanical problem."),
     ("h", "The dead band at low speeds"),
     ("p", "A requested speed below 0.1 revolutions per minute sets the "
           "delay to zero, meaning stopped. Without it, a speed approaching "
           "zero gives a delay approaching infinity, and the loop would "
           "wait essentially forever between steps with no way to interrupt "
           "it. Treating a very small speed as stopped is both what the "
           "operator means and the only behaviour that remains "
           "controllable."),
   ]),

 "TriggeredEndstop": dict(
   inputs=[("ignore", "switch names to disregard")],
   output="the name of a switch confirmed to be pressed, or nothing.",
   theory=[
     ("h", "What it is for"),
     ("p", "The safety check, called before essentially every step of every "
           "motion. It reads each switch and returns the name of the first "
           "confirmed to be pressed."),
     ("h", "How the debounce is arranged"),
     ("p", "A switch reading open is dismissed immediately. One reading "
           "closed is then sampled twice more, spaced by the debounce "
           "delay, and is believed only if all three agree. The reasoning "
           "behind the spacing and the count is given under "
           "ENDSTOP_DEBOUNCE_S and ENDSTOP_CONFIRM_SAMPLES."),
     ("h", "Why the asymmetry is what makes this affordable"),
     ("p", "This function runs before every step pulse, so its cost is "
           "multiplied by the tens of thousands of steps in a drawing. The "
           "early dismissal is what keeps that cost down: in the normal "
           "case every switch reads open and the function performs four "
           "pin reads and returns, with no delays at all. The twelve "
           "milliseconds of confirmation are spent only when a switch "
           "actually reads closed, which happens once per job at most."),
     ("h", "Why some switches can be ignored"),
     ("p", "The ignore set exists because there are legitimate situations "
           "in which a switch is pressed and should not stop the machine. "
           "Retracting off a switch necessarily begins with that switch "
           "pressed, and homing one axis may hold a switch while the other "
           "axis is moved. Without the ability to disregard a named switch, "
           "the only remedy would be to omit the safety check entirely "
           "during those operations, which would remove it exactly where "
           "the machine is deliberately being driven against its limits."),
     ("note", "The check returns at the first switch it confirms rather "
              "than examining all four. For stopping that is sufficient, "
              "since one tripped switch is reason enough. The separate "
              "_AllTriggeredEndstops exists for the emergency stop, which "
              "does need the full list so that it can retract off every one "
              "of them."),
   ]),

 "_AllTriggeredEndstops": dict(
   output="a list of every switch currently confirmed as pressed.",
   theory=[
     ("h", "Why a complete list is needed"),
     ("p", "An emergency stop must retract off every pressed switch, not "
           "just the one that was noticed first. Two can be pressed at "
           "once: a carriage driven into a corner reaches an X and a Y "
           "limit together, and retracting off only one leaves the machine "
           "still held against the other, so the next motion check still "
           "finds a tripped switch and nothing can move."),
     ("h", "How it is built from the single check"),
     ("p", "Each switch is tested by calling the single check with every "
           "*other* switch in the ignore set, so the answer concerns that "
           "switch alone. It is a roundabout construction, but it reuses "
           "the debounce logic rather than duplicating it, which matters "
           "for a safety check: two implementations of a debounce could "
           "diverge, and the one in the rarely exercised path is the one "
           "that would not be noticed."),
   ]),

 "AxisDirValue": dict(
   inputs=[("invert", "whether this axis's direction signal is reversed"),
           ("positive", "whether the wanted motion is in the positive "
                        "direction")],
   output="the level to set on the direction pin.",
   theory=[
     ("h", "What it resolves"),
     ("p", "Turns a wanted direction into a pin level, accounting for the "
           "axis's wiring inversion. It is the single place the inversion "
           "is applied, so every motion routine inherits it by asking for a "
           "direction by meaning rather than by level."),
     ("eq", r"\mathrm{forward} \;=\; \mathrm{positive} \;\oplus\; "
            r"\mathrm{invert}",
      [(r"\mathrm{forward}", "whether the direction pin should be driven "
                             "active"),
       (r"\mathrm{positive}", "whether the wanted motion is toward "
                              "increasing coordinates"),
       (r"\mathrm{invert}", "this axis's wiring inversion flag"),
       (r"\oplus", "**exclusive or**: true when exactly one of the two is "
                   "true")],
      None),
     ("p", "Exclusive or is exactly the right operation, and it is worth "
           "checking the four cases. Wanting positive motion on an "
           "uninverted axis drives the pin active; wanting negative on the "
           "same axis drives it inactive. On an inverted axis both answers "
           "swap. So the output is active when the two inputs disagree, "
           "which is what exclusive or computes. Written as nested "
           "conditions it would be four branches that must each be right; "
           "as an exclusive or it is one expression that cannot be half "
           "correct."),
   ]),

 "_RetractOffSwitch": dict(
   inputs=[("hit", "the name of the pressed switch"),
           ("ignore", "other switches to disregard while retracting")],
   output="nothing; it steps the axis clear and updates the position count.",
   theory=[
     ("h", "What it does"),
     ("p", "Steps the affected axis away from the switch by the retract "
           "distance, at homing speed, updating the step count as it goes. "
           "The direction comes from the retreat table and the axis and "
           "pins from the switch's name."),
     ("h", "Why it watches the other switches"),
     ("p", "Each step checks for any switch other than the one being "
           "escaped and the ones explicitly ignored, and stops if one "
           "trips. The code comment calls that a real fault, and it is: "
           "moving away from one limit should not approach another, so if "
           "it does, the machine's geometry is not what is believed, "
           "perhaps because a switch is stuck closed or an axis is shorter "
           "than its recorded travel. Continuing to drive would push "
           "against a second limit."),
     ("h", "Why the position count is updated"),
     ("p", "The retract moves the carriage, so the recorded position must "
           "follow, or every subsequent absolute move is out by the retract "
           "distance. The count is accumulated during the loop and applied "
           "to the right axis afterwards, which also means a retract cut "
           "short by another switch updates by the distance actually "
           "travelled rather than the distance intended."),
     ("note", "Retracting at homing speed rather than drawing speed is "
              "deliberate. The machine is at a limit, in a state it was not "
              "supposed to reach, and the geometry is not trusted. Moving "
              "slowly means the other-switch check has time to act before "
              "much further travel occurs."),
   ]),

 "_TriggerEmergencyStop": dict(
   inputs=[("hit", "the switch that caused the stop")],
   output="nothing; it stops the drive, retracts and latches the fault.",
   theory=[
     ("h", "The sequence, and why it is in this order"),
     ("items", [
       "**Latch the fault first,** before anything else. Setting the flag "
       "and recording the switch immediately means that any other code "
       "checking it from this point on sees the stop, including the retract "
       "that follows.",
       "**Set the step pins inactive.** This ends any pulse in progress so "
       "no coil is left energised mid-step.",
       "**Find every pressed switch,** since more than one may be.",
       "**Retract off each of them,** so the machine is clear of its "
       "limits and able to move again once the fault is cleared.",
       "**Report,** naming the switches and the retract distance.",
     ]),
     ("h", "Why re-entry is refused"),
     ("p", "The function returns immediately if the fault is already "
           "latched. Without that, a second switch tripping during the "
           "retract would trigger another stop, which would retract again, "
           "which could trip again, and the recursion would continue until "
           "the program ran out of stack. The guard makes the stop happen "
           "once and the retract complete, which is the behaviour wanted: "
           "the first stop has already established that the machine is to "
           "halt, and a second detection adds nothing."),
     ("h", "Why the pen is raised"),
     ("p", "Raising the pen is part of stopping, not a courtesy. A pen left "
           "down during an abort rests on the paper at the point of "
           "failure, where the ink bleeds into a visible blot, and if the "
           "machine is subsequently moved by hand it drags a line across "
           "the drawing. The drawing is already spoiled by the abort; the "
           "object is to avoid spoiling it further."),
   ]),

 "BresenhamMove": dict(
   inputs=[("dx", "the X displacement in steps"),
           ("dy", "the Y displacement in steps"),
           ("stepDelayS", "the pause between step edges")],
   output="nothing; it steps both motors so the carriage travels in a "
          "straight line.",
   theory=[
     ("h", "The problem"),
     ("p", "Two motors must move together so that the carriage travels "
           "along a straight line from one point to another. Each can only "
           "take whole steps, so the path is necessarily a staircase; the "
           "task is to arrange the staircase so that it stays as close to "
           "the true line as whole steps permit. A move of 300 steps in X "
           "and 100 in Y must take one Y step for every three X steps, "
           "spread evenly, not all 100 at the start."),
     ("h", "The naive approach, and why it is avoided"),
     ("p", "The obvious method computes, for each X step, the exact Y "
           "position on the line and rounds it. That requires a floating-"
           "point multiplication and a rounding per step, and this loop "
           "runs for every one of tens of thousands of steps in a drawing. "
           "It also accumulates rounding error, since the exact position is "
           "recomputed from a division that is not exact."),
     ("h", "Bresenham's method"),
     ("p", "The algorithm \\cite{bresenham1965} keeps a running error term "
           "instead, representing how far the minor axis has fallen behind "
           "the ideal line, and uses only integer addition and comparison."),
     ("eq", r"\begin{aligned} "
            r"e &\leftarrow \lfloor a_x / 2 \rfloor \\[3pt] "
            r"\text{for each major step:} \quad "
            r"e &\leftarrow e - a_y \\[3pt] "
            r"\text{if } e < 0: \quad "
            r"&\text{step the minor axis, } \; e \leftarrow e + a_x "
            r"\end{aligned}",
      [("e", "the error term: how far the minor axis is behind"),
       ("a_x", "the major axis displacement in steps, taken as positive"),
       ("a_y", "the minor axis displacement, positive"),
       (r"\lfloor a_x/2 \rfloor", "the starting value, which centres the "
                                  "error so the first minor step falls "
                                  "half-way through the first run rather "
                                  "than at its start")],
      "bresenham1965"),
     ("p", "The reasoning is a scaled comparison. The true slope is the "
           "minor displacement over the major, and testing whether the "
           "accumulated fraction has passed a half-step would need "
           "division. Multiplying the whole test through by the major "
           "displacement removes the division: subtracting the minor "
           "displacement each major step accumulates the same quantity "
           "scaled up, and comparing against zero is the same test as "
           "comparing the unscaled fraction against its threshold. Adding "
           "the major displacement back when the minor axis steps is the "
           "scaled equivalent of subtracting one whole step."),
     ("p", "So the loop contains one subtraction, one comparison and "
           "occasionally one addition, all on integers. There is no "
           "division, no floating point and no accumulated rounding error: "
           "the error term is exact at every step, so the path is the "
           "closest staircase to the true line by construction rather than "
           "by approximation."),
     ("h", "Why the major axis is chosen first"),
     ("p", "The loop runs over whichever axis has the greater displacement, "
           "with the other as the minor. That is necessary rather than "
           "cosmetic: the minor axis steps at most once per major step, so "
           "if the shorter axis were driven as the major the longer one "
           "could not keep up and the line would come out at the wrong "
           "angle. The two branches of the code are mirror images, "
           "differing only in which pin is pulsed and which count is "
           "updated."),
     ("h", "The safety check inside the loop"),
     ("p", "Every iteration checks the endstops and returns immediately if "
           "one has tripped. Returning part way leaves the move incomplete "
           "and the carriage somewhere along the line, which is why the "
           "step counts are updated as the loop proceeds rather than at the "
           "end: the recorded position is correct for where the carriage "
           "actually stopped, not for where it was told to go."),
     ("note", "Bresenham's algorithm was published for drawing lines on "
              "raster displays, where the problem is identical: choose "
              "which pixels best approximate a straight line between two "
              "points. A stepper gantry is the same problem with the pixels "
              "replaced by step positions, which is why an algorithm from "
              "computer graphics is the standard solution for motion "
              "control."),
   ]),

 "PenIsUp": dict(
   output="whether the pen switch currently reads as up.",
   theory=[
     ("h", "What it is for"),
     ("p", "Reads the pen position switch. This is the capability the "
           "written G-code file lacks: the machine can ask where the pen "
           "is, so it never has to assume."),
     ("note", "The read is not debounced, unlike the endstop check. That is "
              "acceptable because of how it is used: SetPen polls it "
              "repeatedly in a loop while the motor runs, so a single "
              "spurious reading advances the loop by one iteration and the "
              "next reading corrects it. A spurious endstop reading, by "
              "contrast, aborts the whole job, which is why that one is "
              "confirmed and this one is not."),
   ]),

 "SetPen": dict(
   inputs=[("targetUp", "the pen position wanted"),
           ("timeoutS", "how long to wait for confirmation, default 3 "
                        "seconds")],
   output="true if the pen reached the wanted position, false otherwise.",
   theory=[
     ("h", "The closed loop"),
     ("p", "The motor is started, the switch is polled until it reports the "
           "wanted position or the timeout expires, and the motor is "
           "stopped. This is **closed-loop** control: the action continues "
           "until the measurement confirms the result, in contrast with the "
           "written file's timed dwell, which commands a pulse and assumes "
           "it worked."),
     ("h", "Why it checks the position first"),
     ("p", "If the pen already reads as being where it is wanted, the "
           "function returns immediately without running the motor. The "
           "code comment is emphatic, and the reason is the one-way "
           "actuator: running it when the pen is already correct turns the "
           "cam a further quarter turn and moves the pen to the wrong "
           "position. For a toggle, a redundant command is not merely "
           "wasteful but actively wrong."),
     ("h", "Why it refuses while an endstop is tripped"),
     ("p", "A pen move is declined outright if any limit switch is pressed. "
           "The machine is in a fault state, the pen motor draws current "
           "from the same supply as the steppers, and moving the pen does "
           "nothing to resolve the fault. Refusing keeps the machine still "
           "until the fault has been dealt with, which is what a latched "
           "stop is for."),
     ("h", "Why a timeout, and what its expiry means"),
     ("p", "If the switch never reports the wanted position within three "
           "seconds, the motor is stopped and the function reports failure. "
           "A toggle takes about a third of a second, so three seconds is "
           "ample and its expiry means something is mechanically wrong: the "
           "motor is not turning, the switch is disconnected, or the "
           "linkage has broken. Without a timeout the motor would run "
           "indefinitely, turning the cam round and round and overheating. "
           "Reporting the failure lets the caller abort rather than "
           "continuing to draw with a pen in an unknown position."),
     ("h", "The overrun"),
     ("p", "On a successful raise the motor runs a further 40 milliseconds "
           "past the switch trip, for the reason given under "
           "PEN_UP_OVERRUN_S: stopping exactly on the trip point leaves the "
           "cam where backlash or vibration can drop the pen onto the page "
           "during a travel move."),
   ]),

 "_StepAxis": dict(
   inputs=[("stepPin", "the step pin for this axis"),
           ("dirPin", "its direction pin"),
           ("dirValue", "the level to drive the direction pin at"),
           ("stopWhen", "a test saying which switch is the expected one"),
           ("ignore", "switches to disregard")],
   output="the switch that stopped the motion and how many steps were "
          "taken.",
   theory=[
     ("h", "What it is for"),
     ("p", "Drives one axis in one direction until a switch stops it, "
           "counting the steps. It is the primitive the homing and "
           "calibration are built from, and the step count it returns is "
           "the measurement that calibration converts into steps per "
           "millimetre."),
     ("h", "Why the stopping condition is a test rather than a name"),
     ("p", "The caller supplies a function saying which switch is the "
           "expected one, rather than naming it. That is necessary because "
           "homing begins without knowing which end of the axis it will "
           "reach: the carriage may be anywhere, so driving in one "
           "direction may bring it to either switch. The first call accepts "
           "either, and the second accepts only the opposite one."),
     ("h", "Why an unexpected switch is an error rather than a stop"),
     ("p", "A switch that trips which is not the expected one raises an "
           "error naming the wiring, rather than being treated as a normal "
           "stop. The distinction matters because the step count is about "
           "to be used as a measurement: stopping on the wrong switch means "
           "the count does not measure what it is believed to, and a "
           "calibration built on it would scale every subsequent move "
           "wrongly. A measurement known to be invalid should fail rather "
           "than be used."),
     ("h", "Why the step limit is also an error"),
     ("p", "Exceeding the step bound raises an error too. A correctly "
           "functioning machine reaches a switch in a fraction of that "
           "distance, so the bound being reached means no switch is going "
           "to trip, which is the disconnected-wiring case. Again the "
           "appropriate response is to stop and say so, since there is no "
           "sensible way to proceed without a reference position."),
   ]),

 "_CalibrateAxis": dict(
   inputs=[("stepPin", "the step pin"), ("dirPin", "the direction pin"),
           ("minName", "the minimum switch's name"),
           ("maxName", "the maximum switch's name"),
           ("OtherAxisNames", "the other axis's switches, to disregard")],
   output="the first switch reached, the second, and the steps measured "
          "between them.",
   theory=[
     ("h", "The measurement"),
     ("p", "The axis is driven in one direction until either switch trips, "
           "then reversed until the opposite one does, counting the steps "
           "of the second move. That count is the full travel of the axis "
           "in steps, which divided by the known travel in millimetres "
           "gives the steps per millimetre."),
     ("h", "Why the count comes from the second move"),
     ("p", "The first move starts from wherever the carriage happened to be "
           "and ends at a switch, so its length is arbitrary and carries no "
           "information. The second starts at a known position, one switch, "
           "and ends at another known position, the other switch, so its "
           "length is the distance between two physical references. Only a "
           "move between two known points measures anything."),
     ("h", "Why the first switch is ignored during the reverse"),
     ("p", "The reverse move begins with the first switch still pressed, "
           "since the carriage is resting against it, and it stays pressed "
           "for the first few steps until the lever releases. The code "
           "comment calls this release lag and notes it is not a fault. "
           "Without adding that switch to the ignore set, the reverse move "
           "would immediately detect it, decide it was unexpected, and "
           "raise an error on every calibration."),
     ("h", "Why the other axis is ignored throughout"),
     ("p", "The other axis may be resting against one of its own limits "
           "while this one is being calibrated, which is a perfectly normal "
           "state and nothing to do with this axis's motion. Ignoring its "
           "switches lets each axis be calibrated regardless of where the "
           "other happens to be parked."),
   ]),

 "Calibrate": dict(
   output="a copy of the calibration record: the measured scaling for both "
          "axes and the usable area.",
   theory=[
     ("h", "What it establishes"),
     ("p", "Two things the machine cannot otherwise know: where it is, and "
           "how far a step moves it. Both come from driving to physical "
           "references, since the motors report neither."),
     ("h", "The sequence"),
     ("items", [
       "**Refuse if a switch is already pressed.** Calibration drives "
       "until a switch trips, so starting against one makes the first move "
       "meaningless.",
       "**Measure the X axis** between its two switches, and set its "
       "steps per millimetre from the count.",
       "**Retract off the switch** it finished against, so it is clear of "
       "its limit.",
       "**Measure the Y axis** the same way, ignoring the X switches.",
       "**Retract** again, record the usable area, and park at the safe "
       "corner.",
     ]),
     ("h", "Where the position count comes from"),
     ("p", "After each axis, the step count is set to the full travel if "
           "the axis finished at its maximum switch, and to zero if at its "
           "minimum. That is the whole point of homing: the carriage is at "
           "a switch, the switch is at a known place, so the count can be "
           "set rather than guessed. Every subsequent absolute move is "
           "measured from it."),
     ("h", "The usable area and the switch-travel allowance"),
     ("eq", r"\mathrm{usable} \;=\; \mathrm{switchTravel} \;-\; "
            r"2 \cdot \mathrm{edgeTolerance}",
      [(r"\mathrm{usable}", "the area the machine will address, per axis"),
       (r"\mathrm{switchTravel}", "the hand-measured distance between the "
                                  "two switches"),
       (r"\mathrm{edgeTolerance}", "the margin kept clear at each end")],
      None),
     ("p", "The usable area is deliberately smaller than the measured "
           "travel, for the reasons given under EDGE_TOLERANCE_MM. The "
           "coordinate origin is placed at the margin's distance in from "
           "the minimum switches, so a commanded zero is a position the "
           "machine is allowed to occupy rather than the switch itself."),
     ("h", "Why it parks at the end"),
     ("p", "Calibration finishes with a move to the origin of the usable "
           "area. That leaves the machine at a known, legal position, "
           "clear of every switch, which is what a drawing job expects to "
           "start from. Finishing while resting against a limit would mean "
           "the first motion check of the next job found a tripped switch."),
   ]),

 "ParseGcodeLine": dict(
   inputs=[("raw", "one line of a G-code file")],
   output="the command and its parameters, or nothing for a line with no "
          "command.",
   theory=[
     ("h", "What it does"),
     ("p", "Strips comments, splits the line into words, takes the first G "
           "or M word as the command, and collects the rest as parameters "
           "keyed by their letters."),
     ("h", "The two comment conventions"),
     ("p", "G-code has two: a semicolon makes the rest of the line a "
           "comment, and text in parentheses is a comment wherever it "
           "appears, including in the middle of a line. Both are handled, "
           "the first by truncating at the semicolon and the second by "
           "removing every parenthesised span. Handling only the first "
           "would leave comment text to be parsed as words, and a comment "
           "containing a letter followed by a digit would be read as a "
           "parameter."),
     ("h", "Why the first G or M word becomes the command"),
     ("p", "A line has one command and any number of parameters, and the "
           "command is identified by being a G or M word. Taking the first "
           "such word rather than the first word generally is what lets a "
           "line with a leading line number be parsed, and the test that "
           "the command slot is still empty is what stops a later G word "
           "being mistaken for a second command."),
     ("note", "The command's number is converted to an integer while "
              "parameters stay as floating point. That is correct for each: "
              "G1 and G01 denote the same command, so an integer "
              "normalises them, while a coordinate of 12.5 must keep its "
              "fraction."),
   ]),

 "_ClampToUsableArea": dict(
   inputs=[("xMm", "the target X coordinate"),
           ("yMm", "the target Y coordinate")],
   output="the coordinates forced inside the homed usable area.",
   theory=[
     ("h", "Why the machine clamps as well as the writer"),
     ("p", "The G-code writer already clamps to the configured bounds. This "
           "clamps again, against the *measured* usable area, and the two "
           "are not the same check. The writer's bounds are configuration, "
           "stated in advance; this one is what calibration actually "
           "found, which may be smaller, and it applies to any file "
           "whatever its origin, including a hand-written one or one from "
           "another program."),
     ("p", "The principle is that the machine must protect itself rather "
           "than trusting its input. A clamp at the point of motion is the "
           "last line of defence and the only one that cannot be bypassed "
           "by using a different generator."),
     ("h", "Why it prints when it acts"),
     ("p", "A clamp that fires means the file asked for something outside "
           "the machine, so the drawing will be wrong: strokes are "
           "flattened against the boundary. Printing the original and "
           "clamped coordinates makes that visible while the job runs, so "
           "the operator can stop it rather than discovering afterwards "
           "that part of the drawing is a straight line along one edge."),
   ]),

 "_GcodeLinearMove": dict(
   inputs=[("targetXMm", "the destination X coordinate"),
           ("targetYMm", "the destination Y"),
           ("feedMmMin", "the feed rate in millimetres per minute")],
   output="nothing; it moves the carriage there at that rate.",
   theory=[
     ("h", "What it does"),
     ("p", "Converts a destination in millimetres into a step displacement, "
           "works out the step timing from the feed rate, and hands the "
           "move to the Bresenham stepper."),
     ("h", "The coordinate conversion"),
     ("eq", r"\mathrm{targetSteps}_X \;=\; \mathrm{round}\big( "
            r"(x_{\mathrm{mm}} + \mathrm{edgeTol}) \cdot "
            r"\mathrm{stepsPerMm}_X \big)",
      [(r"\mathrm{targetSteps}_X", "the absolute step position wanted"),
       ("x_{\\mathrm{mm}}", "the commanded coordinate"),
       (r"\mathrm{edgeTol}", "the margin, added because a commanded zero "
                             "sits that far in from the minimum switch"),
       (r"\mathrm{stepsPerMm}_X", "the calibrated scaling for this axis")],
      None),
     ("p", "The margin must be added here and subtracted when the current "
           "position is read back, and the two must agree. If one applied "
           "the offset and the other did not, every move would be out by "
           "the margin, and because both appear in the same function the "
           "error would be a constant five-millimetre shift of the whole "
           "drawing."),
     ("h", "The step timing"),
     ("eq", r"t_{\mathrm{step}} \;=\; \max\!\left( "
            r"\frac{d_{\mathrm{mm}} / v}{\mathrm{majorSteps}},\; "
            r"t_{\min} \right), \qquad "
            r"v = \max\!\left(\frac{\mathrm{feed}}{60},\; v_{\min}\right)",
      [(r"t_{\mathrm{step}}", "the delay between step edges"),
       ("d_{\\mathrm{mm}}", "the straight-line distance of the move"),
       ("v", "the feed rate in millimetres per second"),
       (r"\mathrm{majorSteps}", "the step count on the axis that moves "
                                "furthest, which is how many loop "
                                "iterations the move takes"),
       ("t_{\\min}", "the minimum step delay"),
       ("v_{\\min}", "the minimum feed, guarding against a zero feed rate"),
       ("60", "the conversion from per-minute to per-second")],
      None),
     ("p", "Dividing by the major axis step count rather than the total is "
           "correct because the Bresenham loop iterates once per major-axis "
           "step, taking a minor step inside some of those iterations. So "
           "the major count is the number of delays the move will actually "
           "contain, and dividing the intended duration by it gives the "
           "delay each must be."),
     ("note", "No acceleration is applied: the move begins and ends at full "
              "speed. For a light gantry at these feeds that is tolerable, "
              "though it is the likeliest source of lost steps on a fast "
              "travel move, and it is why the configuration's acceleration "
              "field, which is never read, would be the first thing to "
              "implement."),
   ]),

 "_GcodeArcMove": dict(
   inputs=[("x0Mm", "the arc's start X"), ("y0Mm", "its start Y"),
           ("x1Mm", "its end X"), ("y1Mm", "its end Y"),
           ("iMm", "the centre's X offset from the start"),
           ("jMm", "the centre's Y offset from the start"),
           ("clockwise", "the direction of travel"),
           ("feedMmMin", "the feed rate")],
   output="nothing; it draws the arc as a sequence of short straight moves.",
   theory=[
     ("h", "Why arcs must be subdivided"),
     ("p", "The machine can only move in straight lines, because that is "
           "what the Bresenham stepper does. An arc is therefore "
           "approximated by a sequence of short chords. This is what every "
           "machine controller does; the only question is how short."),
     ("h", "How G-code specifies an arc"),
     ("p", "The command gives the end point and the **centre offset**, the "
           "vector from the start point to the centre, conventionally "
           "written as I and J. The radius is then the length of that "
           "offset, and the start and end angles follow from the two "
           "endpoints relative to the centre. Specifying the centre by "
           "offset rather than absolutely is the convention because it "
           "makes the command independent of where the arc sits."),
     ("h", "Forcing the sweep to go the right way"),
     ("p", "The two angles alone do not determine the arc: the same pair of "
           "endpoints can be joined by a short arc one way or a long one "
           "the other. The direction flag settles it, and the code enforces "
           "it by adjusting the end angle by whole turns until the sweep "
           "runs strictly the right way. For a clockwise arc the end angle "
           "is decreased until it is below the start; for anticlockwise, "
           "increased until above. Without that, an arc crossing the point "
           "where the angle measurement wraps would be traced the long way "
           "round, sweeping nearly a full circle instead of a small arc."),
     ("h", "How many chords"),
     ("eq", r"n \;=\; \max\!\left(4,\; \left\lfloor "
            r"\frac{|\Delta\theta| \cdot r}{0.4} \right\rfloor \right)",
      [("n", "the number of straight segments the arc becomes"),
       (r"|\Delta\theta|", "the angle swept, in radians"),
       ("r", "the arc's radius in millimetres"),
       ("0.4", "the target chord length in millimetres"),
       (r"\max(4,\cdot)", "a floor, so even a tiny arc is drawn as a curve "
                          "rather than a straight line")],
      None),
     ("p", "The numerator is the arc's length, since arc length is radius "
           "times angle in radians, so the expression is simply the arc "
           "length divided by the wanted chord length. Fixing the chord "
           "length rather than the number of segments is what makes the "
           "approximation uniformly good: a large arc gets proportionally "
           "more segments, so the deviation from the true curve is the same "
           "whatever the size."),
     ("h", "Why 0.4 millimetres"),
     ("p", "The relevant quantity is the **sagitta**, how far the chord "
           "departs from the arc at its midpoint, which for a short chord "
           "is approximately the chord length squared divided by eight "
           "times the radius. For a 0.4 mm chord on a 5 mm radius that is "
           "about 4 micrometres, far below the pen's line width and well "
           "below what the machine can position to. So the approximation is "
           "invisible, and making the chords shorter would add moves "
           "without changing the drawing."),
     ("note", "Arcs are included for completeness of the G-code reader "
              "rather than because this project emits them: handwriting is "
              "written out as straight segments throughout. They matter "
              "because the reader is meant to run any ordinary G-code file, "
              "and arcs are among the commonest commands in files from "
              "other sources."),
   ]),

 "RunGcodeFile": dict(
   inputs=[("path", "the G-code file to execute")],
   output="nothing; it draws the file or aborts.",
   theory=[
     ("h", "What it does"),
     ("p", "Reads the file and executes it line by line, maintaining the "
           "current position, the feed rate and whether coordinates are "
           "absolute or relative, and carrying out each move."),
     ("h", "Why calibration is required first"),
     ("p", "The function refuses with a dedicated error if homing has not "
           "run. A file's coordinates are absolute positions on the bed, "
           "and an absolute position is meaningless until the machine knows "
           "where it is. Executing one anyway would drive from an unknown "
           "origin toward a position computed as though it were known."),
     ("h", "The safety check before every line"),
     ("p", "Each line begins by checking the endstops and the latched fault "
           "flag, and aborts on either. Checking per line as well as per "
           "step gives two different guarantees: the per-step check inside "
           "the stepper stops a move in progress, and this one stops the "
           "job rather than merely the move. Without it, a move cut short "
           "by a switch would be followed by the next line of the file "
           "being executed from the wrong position."),
     ("p", "The abort reports the line number. That is what makes a failure "
           "diagnosable: it says how far the drawing got and which command "
           "was being executed, so the file can be examined at that point "
           "rather than searched."),
     ("h", "Modal state"),
     ("p", "G-code is **modal**, meaning several settings persist until "
           "changed: the feed rate, absolute or relative coordinates, and "
           "the units. The interpreter must therefore carry that state "
           "across lines, which is why the feed and the absolute-mode flag "
           "are variables of the loop rather than parameters of each move. "
           "A line consisting only of coordinates inherits the feed set "
           "many lines earlier, which is how G-code files are normally "
           "written."),
     ("note", "The starting position is read using the uncalibrated "
              "steps-per-millimetre constant rather than the measured "
              "value, so the position the job believes it starts from is "
              "slightly wrong even after calibration. Subsequent moves use "
              "the calibrated figure, so the error does not accumulate, but "
              "the first move of a job is displaced by the difference "
              "between the calculated and measured scaling."),
   ]),

 "WriteGcodeFile": dict(
   inputs=[("path", "where to write")],
   output="nothing; it writes a small test file.",
   theory=[
     ("h", "What it is for"),
     ("p", "Emits a short G-code file for exercising the machine during "
           "bring-up, so the execution path can be tested without the "
           "synthesis half being involved. It is a diagnostic convenience, "
           "not part of drawing handwriting: a real drawing is written by "
           "WriteGcode from a trajectory."),
   ]),

 "InputListener": dict(
   output="nothing; it runs the keyboard control loop until told to stop.",
   theory=[
     ("h", "What it is for"),
     ("p", "A console for operating the machine by hand: home it, run a "
           "file, set a speed, draw a test circle, clear a latched fault, "
           "quit. It is what the machine is brought up and diagnosed "
           "with."),
     ("h", "Why it runs in its own thread"),
     ("p", "Reading a line from the keyboard blocks until something is "
           "typed. If that happened in the main loop, the machine would "
           "stop stepping while waiting for input, so motion would proceed "
           "only while the operator was typing. Running the console in a "
           "separate thread lets it wait on the keyboard while the main "
           "loop continues to step the motors."),
     ("h", "How the two threads communicate"),
     ("p", "Entirely through the module-level state: the console sets the "
           "mode, the speed, the fault flag, and the loop reads them. There "
           "are no locks. That is sound here for a specific reason rather "
           "than by luck: every shared value is a single machine word, "
           "whose assignment cannot be observed half-complete, and each is "
           "written by only one of the two threads. Nothing requires two "
           "values to change together, which is the situation locks exist "
           "to protect."),
     ("h", "Why the thread is a daemon"),
     ("p", "A **daemon** thread does not keep the program alive: when the "
           "main thread finishes, the process exits whether or not the "
           "daemon is still running. That is wanted here because the "
           "console is blocked reading the keyboard most of the time, so if "
           "it had to finish before the program could exit, a quit command "
           "issued from anywhere else would hang until a further key was "
           "pressed."),
     ("note", "The clear command for a latched fault is deliberately a "
              "manual action. It is the acknowledgement described under "
              "the latching flag: the machine stopped because its idea of "
              "the world was wrong, and only a person standing at it can "
              "establish that it is right again."),
   ]),
}
