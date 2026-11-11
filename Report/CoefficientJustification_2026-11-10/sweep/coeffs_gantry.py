#u23539195 coefficient justification: the G-code and gantry coefficients
#these change the pen path and the machine numbers, not the handwriting, so the
#panels show the toolpath and the figures carry the job statistics
import probes as P


def C(**kw):
    return kw


PROFILE = r"C:/Users/braun/AppData/Local/Temp/claude/C--Users-braun-OneDrive-Documents-GitHub-Handwriting-Robot/685927e6-23ae-495d-95f3-1824d845cbe0/scratchpad/out/prof_owen.json"

GC_RUN = dict(profile=PROFILE,text=P.SENTENCE_SHORT,seed=0,
              mmPerXh=4.0,jitter=0.15)

PATH_PANEL = dict(kind="path",showTravel=True,panelHeight=0.95)


ENTRIES = [

C(key="join_strokes_mm",
  title="Gap the pen draws through instead of lifting",
  symbol=r"$d_{\text{join}}$",
  subsystem="G-code generation",
  module="reproduce", where="GantryConfig.joinStrokesMm",
  default=0.6, values=[0.0,0.25,0.6,1.5,4.0],
  patches=lambda v: [("reproduce","joinStrokesMm: float = 0.6",
                      "joinStrokesMm: float = %s" % v)],
  task="gcode", params=GC_RUN, panel=PATH_PANEL,
  metrics=["strokes","penPulses","joinedStrokes","joinedInkMm","travelMm",
           "estimatedSeconds"],
  trendMetrics=["penPulses","estimatedSeconds"],
  source=r"Minimising non-productive moves and tool changes is the standard "
         r"objective of toolpath generation; the pen-up and pen-down commands "
         r"themselves are the \texttt{M3}/\texttt{M5} spindle words of "
         r"\cite{iso6983}.",
  reason=r"Lifting the pen costs far more than drawing a short line: each lift "
         r"is two pen pulses, each with a $350$ ms dwell, so $0.7$ s, while "
         r"drawing $0.6$ mm at the $900$ mm/min drawing feed costs $0.04$ s. "
         r"Where the next stroke begins very close to where the last one ended, "
         r"drawing straight into it is therefore about twenty times faster, and "
         r"it also removes the risk that the pen fails to re-seat. The cost is a "
         r"visible mark: the gap is drawn as ink, so $d_{\text{join}}$ must stay "
         r"below the width at which that mark is noticeable. At $0.6$ mm the "
         r"added line is comparable with the pen's own $0.7$ mm trace and reads "
         r"as part of the stroke.",
  understanding=r"The statistics make the trade explicit: \texttt{joinedStrokes} "
                r"counts how many lifts were avoided and \texttt{joinedInkMm} "
                r"how much ink was added to avoid them. Raising "
                r"$d_{\text{join}}$ draws ever longer false connections across "
                r"the writing; at $0$ every stroke gets its own lift.",
  alternatives=r"Reordering strokes so that consecutive ones start near each "
               r"other, which \texttt{OptimizeStrokeOrder} already does, or a "
               r"faster pen actuator, which is a hardware change.",
  worked=r"One avoided lift saves $2 \times 350 = 700$ ms of dwell plus the "
         r"travel move, and costs at most $0.6$ mm of ink drawn at "
         r"$900$ mm/min, which is $0.04$ s.",
  ),

C(key="close_ring_mm",
  title="Gap at which a stroke is closed into a ring",
  symbol=r"$d_{\text{ring}}$",
  subsystem="G-code generation",
  module="reproduce", where="GantryConfig.closeRingMm",
  default=0.6, values=[0.0,0.25,0.6,1.5,4.0],
  patches=lambda v: [("reproduce","closeRingMm: float = 0.6",
                      "closeRingMm: float = %s" % v)],
  task="gcode", params=GC_RUN, panel=PATH_PANEL,
  metrics=["strokes","closedRings","drawMm","estimatedSeconds"],
  trendMetrics=["closedRings","drawMm"],
  source=r"Closing a nearly closed contour is the standard repair for a traced "
         r"loop whose ends do not meet after thinning \cite{lam1992}.",
  reason=r"A letter with a loop, such as \texttt{o}, \texttt{a} or \texttt{e}, "
         r"is traced from a skeleton whose ends rarely meet exactly: thinning "
         r"and simplification leave a gap of a fraction of a millimetre. Drawn "
         r"as-is the loop has a visible notch. If a stroke's own two ends are "
         r"within $d_{\text{ring}}$, the first point is appended so the ring "
         r"closes. The limit is the same $0.6$ mm as the join limit for the same "
         r"reason: it is about one pen width, so closing is invisible.",
  understanding=r"The test applies only to a stroke's own ends and only when it "
                r"has more than three points, so it cannot close a short "
                r"two-point dash into a degenerate ring. Too large and a genuine "
                r"\texttt{c}, whose ends are meant to be apart, is closed into "
                r"an \texttt{o}, which is a legibility error, not just a "
                r"cosmetic one.",
  alternatives=r"Closing in the glyph store rather than at G-code time, which "
               r"would record the closure in the profile and make it visible to "
               r"the shape descriptor; or never closing and accepting the notch.",
  worked=r"At $4$ mm per x-height an \texttt{o} is about $3$ mm across, so "
         r"$0.6$ mm is a fifth of the letter: a \texttt{c} with a normal opening "
         r"is well outside the limit.",
  ),

C(key="min_segment_mm",
  title="Shortest move kept in the G-code",
  symbol=r"$\ell_{\min}$",
  subsystem="G-code generation",
  module="reproduce", where="GantryConfig.minSegmentMm",
  default=0.12, values=[0.0,0.05,0.12,0.3,1.0],
  patches=lambda v: [("reproduce","minSegmentMm: float = 0.12",
                      "minSegmentMm: float = %s" % v)],
  task="gcode", params=GC_RUN, panel=PATH_PANEL,
  metrics=["lines","strokes","drawMm","estimatedSeconds"],
  trendMetrics=["lines","drawMm"],
  source=r"Chord tolerance and minimum segment length are standard parameters of "
         r"toolpath post-processing; the machine resolution they are set against "
         r"follows from the microstepping of the driver \cite{a4988ds} and the "
         r"motor step angle \cite{stepperds}.",
  reason=r"There is no point emitting a move the machine cannot resolve. One "
         r"microstep is $40/(200 \times 16) = 0.0125$ mm, so moves below about "
         r"that are rounded to nothing by \texttt{\_GcodeLinearMove}, which "
         r"rounds target positions to whole steps. $\ell_{\min} = 0.12$ mm is "
         r"about ten microsteps: comfortably resolvable, but small against a "
         r"$4$ mm x-height, so discarding shorter moves does not visibly change "
         r"a letter. The benefit is file size and interpreter time, which matter "
         r"because the G-code is parsed line by line in Python on the ODROID.",
  understanding=r"\texttt{\_Decimate} always keeps the first and last point of a "
                r"stroke, so decimation cannot shorten a stroke or move its "
                r"endpoints; it only drops intermediate points. Raising "
                r"$\ell_{\min}$ progressively straightens curves, which is "
                r"visible on a letter before it is visible in the statistics.",
  alternatives=r"A chord-deviation tolerance, as Douglas--Peucker "
               r"\cite{douglas1973} applies in the glyph store, which is "
               r"shape-aware rather than length-based and would be the better "
               r"criterion here too; or no decimation.",
  worked=r"At $0.12$ mm and a drawing feed of $900$ mm/min, one retained segment "
         r"takes $8$ ms. The probe job emits the recorded number of G-code lines "
         r"at each setting.",
  ),

C(key="pen_dwell_ms",
  title="Dwell after a pen-down pulse",
  symbol=r"$t_{\text{down}}$",
  subsystem="Pen actuator",
  module="reproduce", where="GantryConfig.penDownDwellMs",
  default=350.0, values=[100.0,200.0,350.0,500.0,800.0],
  patches=lambda v: [("reproduce","penDownDwellMs: float = 350.0",
                      "penDownDwellMs: float = %s" % v)],
  task="gcode", params=GC_RUN, panel=PATH_PANEL,
  metrics=["penPulses","dwellSeconds","estimatedSeconds","drawSeconds"],
  trendMetrics=["dwellSeconds","estimatedSeconds"],
  source=r"A dwell is the \texttt{G4} word of \cite{iso6983}. The need for one "
         r"follows from the actuator being an uncontrolled one-way gear motor "
         r"rather than a servo: there is no position feedback to wait on, so the "
         r"wait must be open-loop.",
  reason=r"The pen is raised and lowered by a gear motor that turns $90^\circ$ "
         r"per pulse and cannot be commanded to a position. The controller "
         r"therefore issues the pulse and waits long enough for the rotation to "
         r"finish and the nib to settle before any motion command follows. If "
         r"the wait is too short the carriage starts moving while the pen is "
         r"still descending and the first few millimetres of the stroke are "
         r"drawn faint or not at all, which is the pen-pressure fault this "
         r"project has recorded. The down dwell is held equal to the up dwell at "
         r"$350$ ms, with the comment in the source noting it is the longer case "
         r"because the nib must seat.",
  understanding=r"The dwell is pure cost: it is paid twice per pen cycle and the "
                r"estimate counts $\text{pulses} \times "
                r"(t_{\text{up}}+t_{\text{down}})/2$. On a job with $58$ pulses "
                r"it accounts for $58 \times 0.35 = 20.3$ s of a $34.1$ s "
                r"estimate, that is $60\%$ of the predicted job time. That makes "
                r"it the single largest term in the time budget and the first "
                r"thing to attack for speed.",
  alternatives=r"Reading the pen switch and continuing as soon as it confirms, "
               r"which the code already does inside \texttt{SetPen} with a "
               r"$3$ s timeout and a $40$ ms overrun, so the dwell is a second, "
               r"open-loop guard on top of a closed-loop wait; a sprung pen "
               r"holder, which removes the need to wait at all; or a servo with "
               r"position feedback.",
  worked=r"On the probe job the submitted setting gives "
         r"$48 \times (350+350)/2000 = 16.8$ s of dwell against $8.9$ s of "
         r"drawing and $3.8$ s of travel, that is $57\%$ of a $29.5$ s "
         r"estimate.",
  notes=r"This sweep is arithmetic on the generated job, not a measurement of "
        r"the machine: no gantry was run. Whether $350$ ms is actually long "
        r"enough for the nib to seat is a hardware question that needs a timed "
        r"test on the gantry, and that test has not been done.",
  ),

C(key="draw_feed",
  title="Feed rate while drawing",
  symbol=r"$F_{\text{draw}}$",
  subsystem="Motion",
  module="reproduce", where="GantryConfig.drawFeedMmMin",
  default=900.0, values=[300.0,600.0,900.0,1500.0,2400.0],
  patches=lambda v: [("reproduce","drawFeedMmMin: float = 900.0",
                      "drawFeedMmMin: float = %s" % v)],
  task="gcode", params=GC_RUN, panel=PATH_PANEL,
  metrics=["drawSeconds","travelSeconds","dwellSeconds","estimatedSeconds"],
  trendMetrics=["drawSeconds","estimatedSeconds"],
  source=r"Feed rate is the \texttt{F} word of \cite{iso6983}. The upper bound "
         r"is set by the step rate the driver and motor can follow "
         r"\cite{a4988ds,stepperds}; the trapezoidal profiling that would "
         r"normally bound acceleration is described by Biagiotti and Melchiorri "
         r"\cite{biagiotti2008} and is \emph{not} implemented here.",
  reason=r"$900$ mm/min is $15$ mm/s, which at $80$ steps/mm is a step rate of "
         r"$1200$ Hz, well inside what an A4988 at $1/16$ microstepping will "
         r"pass. It is deliberately below the $2400$ mm/min travel feed because "
         r"the drawing moves are the short, direction-changing ones: the "
         r"controller issues steps at a constant delay with no acceleration "
         r"ramp, so every corner is taken at full speed, and a stepper that is "
         r"commanded to reverse at speed loses steps. A lower drawing feed is "
         r"the compensation for having no acceleration profile.",
  understanding=r"Because the time model ignores acceleration, the estimate it "
                r"produces is a floor on the real time rather than a prediction, "
                r"and the source says so. Raising $F_{\text{draw}}$ shortens the "
                r"estimate linearly but risks lost steps on the real machine, "
                r"which would show as a drawing that drifts out of registration "
                r"rather than as a slower job.",
  alternatives=r"Implementing trapezoidal acceleration \cite{biagiotti2008} and "
               r"then raising the feed, which is the proper fix; or look-ahead "
               r"cornering, which needs the whole path in memory.",
  worked=r"$900$ mm/min $= 15$ mm/s; at $\text{stepsPerMm} = 80$ that is "
         r"$1200$ steps/s, a step interval of $833\ \mu$s against the "
         r"$2\ \mu$s floor in \texttt{MIN\_STEP\_DELAY\_S}.",
  ),

C(key="ruling_xh_frac",
  title="X-height as a share of the ruled line spacing",
  symbol=r"$f_{xh}$",
  subsystem="Page layout",
  module="reproduce", where="GantryConfig.rulingXhFrac",
  default=0.46, values=[0.25,0.35,0.46,0.60,0.80],
  patches=lambda v: [("reproduce","rulingXhFrac: float = 0.46",
                      "rulingXhFrac: float = %s" % v)],
  task="gcode", params=dict(GC_RUN,useXhForRuling=True), panel=PATH_PANEL,
  metrics=["mmPerXh","drawMm","estimatedSeconds","strokes"],
  trendMetrics=["mmPerXh","drawMm"],
  source=r"The ratio of x-height to line pitch is a basic typographic and "
         r"palaeographic measure of a hand, and is measured per writer in the "
         r"profile as \texttt{xhOverPitch}; the same ratio appears as a style "
         r"feature in writer identification \cite{bulacu2007}.",
  reason=r"The machine writes on ruled paper with an $8.4$ mm ruling, and the "
         r"writing must sit on those lines at a plausible size. The x-height is "
         r"therefore set as a fraction of the ruling. The fraction is only a "
         r"\emph{fallback}: \texttt{XhForRuling} prefers the writer's own "
         r"measured \texttt{xhOverPitch} and falls back to $f_{xh}$ only when "
         r"that is missing or outside $[0.15,0.90]$. $0.46$ is close to the "
         r"middle of the measured range for these writers, so the fallback "
         r"produces a typical hand rather than an extreme one.",
  understanding=r"Everything downstream scales with it: the pen width, the "
                r"collision clearance, the drift bound and the ligature sag are "
                r"all in x-heights, so changing $f_{xh}$ changes the physical "
                r"size of the whole drawing and hence the drawn length and the "
                r"job time. A layout that then exceeds the ruling cap "
                r"\texttt{rulingMaxExtent} is shrunk again by a second pass.",
  alternatives=r"Setting an absolute x-height in millimetres and ignoring the "
               r"ruling, which detaches the writing from the paper; or fitting "
               r"the measured ascender-to-descender extent to the ruling "
               r"directly, which the probe pass approximates.",
  worked=r"At the $8.4$ mm ruling, $f_{xh} = 0.46$ gives "
         r"$x_h = 3.86$ mm, and Owen's line step is "
         r"$(1.51+0.68+0.85) \times 3.86 = 11.7$ mm, which exceeds one ruling, "
         r"so the writing spans more than one ruled line unless the cap pulls it "
         r"down.",
  ),

C(key="endstop_debounce",
  title="Interval between the samples that confirm an end-stop",
  symbol=r"$t_{\text{db}}$",
  subsystem="End-stops",
  module="reproduce", where="ENDSTOP_DEBOUNCE_S",
  default=0.006, values=[0.001,0.003,0.006,0.012,0.025],
  patches=lambda v: [("reproduce","ENDSTOP_DEBOUNCE_S = 0.006",
                      "ENDSTOP_DEBOUNCE_S = %s" % v)],
  task="bresenham", params=dict(dx=1,dy=0),
  panel=dict(kind="path",panelHeight=0.5), skipFigure=True,
  metrics=["steps"],
  source=r"Mechanical switch contacts bounce for a period that depends on the "
         r"switch; sampling until a reading is stable is the standard software "
         r"debounce, and the contact-bounce behaviour of the microswitch type "
         r"used is given in its data sheet \cite{stepperds}.",
  reason=r"A limit switch on a long wire beside two stepper motors picks up both "
         r"contact bounce and electrical noise. A single reading is therefore not "
         r"trusted: the pin must read triggered \texttt{ENDSTOP\_CONFIRM\_SAMPLES} "
         r"times in a row, $t_{\text{db}}$ apart. The interval must span a bounce "
         r"so that the samples are genuinely independent, and the total "
         r"confirmation time $(n-1)\,t_{\text{db}} = 12$ ms must be short against "
         r"how long the carriage takes to travel the overtravel distance of the "
         r"switch. At the $2400$ mm/min travel feed the carriage covers "
         r"$40 \times 0.012 = 0.48$ mm in that time, which is inside the "
         r"overtravel of the microswitch and well inside the $5$ mm edge "
         r"tolerance.",
  understanding=r"The two coefficients multiply: confirmation latency is "
                r"$(n-1)\,t_{\text{db}}$ and noise rejection improves with both. "
                r"A longer interval rejects longer noise bursts but lets the "
                r"carriage run further past the switch before the stop takes "
                r"effect. This is a safety path, so the latency is the term that "
                r"must stay bounded.",
  alternatives=r"A hardware RC filter or a Schmitt trigger on the switch line, "
               r"which removes the software cost entirely and is the better "
               r"engineering answer; or an interrupt with a dead time.",
  worked=r"$(3-1) \times 6\text{ ms} = 12$ ms of confirmation; at $40$ mm/s the "
         r"carriage moves $0.48$ mm, against $5$ mm of edge tolerance, so the "
         r"stop has about ten times the margin it needs.",
  notes=r"\textbf{No sweep was run for this coefficient.} It can only be "
        r"exercised on the gantry, with a switch, and the gantry was not run for "
        r"this study. The numbers above are the arithmetic of the confirmation "
        r"path, not a measurement, and the right test is to record the switch "
        r"line on a scope while the carriage closes it.",
  noSweep=True,
  ),

]
