# Handover — Report comparison + gantry stepper-motor torque check

Context dump of this conversation for continuation in a new chat. Project: handwriting-robot final report (`Report/FinalReport_v5/`) plus a side task on gantry stepper-motor sizing.

---

## 1. Report comparison task (completed)

**Request:** Compare the student's `Report/FinalReport_v5` against two reference final-year reports:
- `C:\Users\braun\Downloads\23531003_Report_v2.pdf` — "Neural Network: Analogue Implementation" by A. Sreekumar (111 pages). Described by the user as "very good, should be used as a good reference."
- `C:\Users\braun\Downloads\aston_not_finished.pdf` — "Multi-robot capture the flag" by A.W. Du Toit (68 pages). Unfinished: Results/Discussion/Conclusion are empty headings; Lit Study, Approach and most of Design are complete and polished.

Comparisons were done assuming all placeholder figures/images/diagrams in the student's report get filled in correctly — the focus was on writing, structure, rigor, and reasoning, not missing pictures.

### Full-report pass (Results/Discussion/Conclusion included)

**Overall verdict:** the student's report is already more rigorous than both references, especially in Results/Discussion — it defines every metric formally with confidence intervals, explicitly separates "search-judge" (circular) from "independent" evidence, and states causal failure-mode reasoning the references don't attempt. `aston_not_finished.pdf` is not a useful style reference for Results/Discussion since that part is empty in it.

**Concrete gaps found (independent of images):**
1. `3_Main_Report/5_Discussion.tex` — table `tab:dis-compare` (lines ~85–114) has **five `\mustdo{figure}` cells** in the published-literature comparison table (citations already present: `shi2017`, `kizilirmak2023`, `bulacu2007`, `kang2020`, `dai2024`). This is desk research only — read the five already-cited papers and fill in the real accuracy/CER/WER figures. Highest-value single fix found.
2. `3_Main_Report/design/3_1_summary.tex` — Design Summary tables (`tab:des-summary`, `tab:des-summary2`) have roughly half their rows marked "Incomplete" with inline `\mustdo`s (camera specs, CAD files, power budget, consent documentation, ethics clearance). Not a missing-figure problem — needs text answers the student has.
3. `6_Conclusion.tex:21` — flagged self-contradiction: real-handwriting word accuracy is quoted as 70.5% in one place and 71.2% in `4_Results.tex`'s qualification-test table, both against the same 89.0% character figure. Needs reconciling which run each number came from.

**Where the student's report already wins:** formal metric definitions with Wilson/bootstrap intervals (neither reference does this), explicit circularity-of-evaluation flagging (search judge vs. independent judge, tabulated per metric), causal "Strong points"/"Unsolved problems"/"Expected failure conditions" sections tied to specific design decisions rather than generic statements.

### Scoped re-pass: Literature Study + Approach + Design only

Files read in full for this pass: `3_Main_Report/1_Literature_Study.tex`, `3_Main_Report/2_Approach.tex`, `3_Main_Report/3_Design_and_Implementation.tex` (just the `\input` list), `3_Main_Report/design/3_1_summary.tex`, `3_Main_Report/design/3_3_hardware.tex` (full), `3_Main_Report/design/3_4_segmentation.tex` (full). Reference PDFs: Sreekumar pages 1–32, 53–70 (lit study, approach, early design, PCB design, full results/discussion); aston pages 1–35 (preamble, proposal, full lit study, full approach, start of design).

**Literature Study:**
- `aston_not_finished.pdf` has the best model here despite being unfinished overall: every citation is explicitly tied to a named requirement number and used to *derive* a spec value (e.g., the 8% path-length tolerance derived from a cited discretisation bound, forcing Requirement 5). Closes with a per-citation "Application of the literature to this project" bullet list.
- Student's `1_Literature_Study.tex` is organised by sub-problem (not by paper), has a `\new{}` tagging convention marking corrected/non-original wording (an honesty device neither reference uses — this is a declared-attribution requirement specific to this project, see Memory section below), and closes with an "Intended contribution and discussion" section (lines ~131–139) doing the same job as aston's closing section but in prose rather than a citation-by-citation requirement map.
- Sreekumar's lit review is a straight topic survey with no requirement-mapping at all — weaker than both the student's and aston's on this axis.

**Approach:**
- Student's `2_Approach.tex` is the strongest of the three structurally. Its `tab:app-tradeoffs` (alternatives-considered table, 3 columns: Decision / Alternatives considered / Preferred and reason) is a genuine formal table — aston does the same thing as a bullet list (readable but not tabulated), Sreekumar has no formal alternatives comparison in Approach at all (trade-offs scattered as asides in Design).
- Sreekumar's Approach is thin: five short paragraphs, no alternatives analysis.
- **Open gaps found:** `2_Approach.tex:29` has three live `\mustdo`s inside otherwise-finished prose (computer used for training, reason for the pen-lift mechanism change, FU4 supply voltages/current ratings). `2_Approach.tex:67` has one more (`\mustdo{list the mechanical parts I designed and made.}`). These are text answers the student already knows, not missing diagrams.

**Design:**
- Structural difference (not a flaw): both references cut Design into many short numbered sub-subsections (one mechanism, one derivation each — e.g. Sreekumar's 3.5.1/3.5.2/3.5.3, aston's 3.5.1/3.5.2/3.5.3). The student's `3_Design_and_Implementation.tex` groups by **component** instead (3.3 hardware, 3.4 segmentation, 3.5 classification, 3.6 style synthesis), and — per explicit instruction given earlier this session (see Memory section) — the internal Theory/Design/Tests phase-label markers were deliberately stripped for flowing prose. Reads as four continuous essays rather than many short lab-notebook entries. Trade-off: harder to skim/cross-reference, but reads as one coherent argument, which neither reference achieves (Sreekumar's legibility comes from small chunking, not from the whole hanging together).
- Both references narrate real bugs inline as part of Design prose, not hidden in Discussion (Sreekumar: ReLU-vs-prior-implementation concern; aston: an 8-bit-overflow silent-failure bug, an exposure-clipping bug) — the student's hardware section already does this too (the "Known faults" paragraph, the missing-clamp-diode/missing-base-pull-down admission in `3_3_hardware.tex:213`) — a match, not a gap.
- **Concrete gaps found:** `3_1_summary.tex:27` has an unresolved `\mustdo` on the training computer (duplicate of the Approach one). The Design Summary table itself still marks ~half its rows "Incomplete" (same as full-pass finding above).

### Judged specifically on: completion / flow / demonstrated understanding / proving the choice is best *for this system*

This was the final framing the user asked for, applied specifically to the Design section (`3_3_hardware.tex` and `3_4_segmentation.tex`, both read in full).

**Completion:** uneven by component. Segmentation (3.4) is close to finished — only one real gap (R4 timing figure needs updating to 4.9–9.7s measured value, plus photographs). Hardware (3.3) carries substantive open gaps: no torque margin was stated at all before this conversation's calculation work (`:73-74`), the 24V→5V/3.3V conversion stage is undocumented, the X-axis direction-pin fault (blocks R5) is unresolved, four figure placeholders (gantry CAD, repeatability overlay, rectangle test, G-code verification overlay) have precise captions already written, just no image yet.

**Flow:** both subsections read as one argument, not a stitched list. 3.4 establishes a shared vocabulary (baseline, x-height, pitch) before any equation uses it; 3.3 explains *why* the torque equation is harder to satisfy than the carriage mass alone suggests before plugging in numbers. The only flow break is where an unresolved fact forces an interruption (e.g. `:213-216` breaking from circuit description to flag design faults) — correct to do, but a genuine register shift.

**Demonstrated understanding — strongest axis, with specific evidence:**
- Distinguishes holding torque vs. pull-out torque and explains *why* only pull-out is a safe design basis (`:32`) — exactly the distinction this conversation's torque calculation exercise worked through in practice.
- Explains *why* squaring before summing makes the deskew objective work (`eq:seg-skew`) — the actual mathematical content, not a citation dressed up.
- Reports measuring that direct-pixel-to-row mapping was 31× faster than rotating the mask, an optimisation justified by the student's own measurement.
- Same "two thresholds beat one" (hysteresis) concept applied correctly and independently in both 3.3 (switch debounce) and 3.4 (ink threshold) — sign of transferred understanding, not copied derivations.

**Proving the choice is best for *this* system — both subsections argue from quantified counterfactuals, not generic taste:**
- `tab:seg-alt`: local-mean threshold would need a tuned offset; the deployed global Otsu threshold works specifically *because* the student's own earlier lighting-division step already removed the reason a local threshold would be needed — argument chains back to the student's own prior design decision.
- One motor on a cross-shaft vs. two independent motors: justified by naming the actual failure mode of the alternative (synchronising two open-loop steppers in lock-step is itself a design problem) rather than dismissing it as "more complex" (`:18`).
- The retired-segmenter comparison (`:260`) is the clearest system-specific argument found anywhere: the old 2200-line/60-function segmenter is named, its capabilities (diagram tagging, per-line deskew, wider-angle search) are explicitly listed as capabilities *given up*, justified against the new design goals — admitting a capability loss as the cost of better justification, which neither reference does.

**Net conclusion given to the user:** flow and demonstrated understanding are already at a finished standard in Design; completion is genuinely behind in 3.3 (hardware) specifically because real physical measurements were missing (this is what the subsequent torque-calculation work in this conversation was closing); the "best for my system" argument is consistently made through quantified counterfactuals rather than taste.

---

## 2. Test paragraph for gantry reproduction testing

User asked for a short paragraph to have each author write by hand and then have the gantry reproduce in their style, for Qualification Test 1 (reproduction fidelity, R1). Minimal punctuation requested, themed around the project.

Iteration 1 (pangram-style, all 26 letters forced in) was offered but rejected as too complex/forced.

**Final version the user approved the style of** (asked for similar word-choice variety to this reference sentence they provided: *"This handwriting robot is designed to classify the text and author of any page of text, as well as reproduce any piece of text in the style of any writer the system was trained off of."*):

> This handwriting robot was built to read the words and identify the writer from any photographed page, then reproduce fresh sentences in that same person's style using a gantry and pen. Over 13 different writers were used to train and test the system, each contributing several pages so the network could learn how they loop their letters, slant their lines and space their words.

This is the version to actually use for the writer-sample collection, unless the user asked for further changes after this (not captured in this conversation).

---

## 3. GT2 belt/pulley terminology check

Confirmed correct usage: "GT2 belt" (or "GT2 timing belt") and "pulleys" (not "belt pulley," which is slightly redundant/odd phrasing) are the standard terms. The student's existing report text ("GT2 belt, 2mm pitch, 20-tooth pulleys") is already correct as written — no change needed.

---

## 4. Stepper motor torque-margin calculation (the main working thread)

### Context / goal
The report currently has an open gap (`3_3_hardware.tex:73-74`, flagged in Section 1 above): *"neither the carriage mass nor the friction force was measured, and the pull-out torque at the rates actually used was not obtained, so no torque margin can be stated."* This conversation worked through closing that gap with real numbers.

### System facts established over the conversation
- Gantry is belt-driven (GT2, 2mm pitch, 20-tooth pulleys), **not** the lead-screw linear actuator datasheet the user pasted partway through (17LS19-1684E-200G-AR4, Tr8x8 screw, from an AR4-MK2 robot arm kit) — that datasheet was flagged as being for a different mechanism (closed-loop lead-screw actuator) than the student's actual open-loop GT2-belt design, and was not used for the final calculation. It was however useful as an example of what a stepper datasheet contains (holding torque, phase resistance 1.65Ω, inductance 2.8mH — used to explain electrical time constant / chopping behaviour).
- Operating profile: **constant velocity, no acceleration ramp** (the design has none — flagged elsewhere in the report as a limitation, since the generator's G-code time model assumes a trapezoidal profile with a=400mm/s² but the driver doesn't actually execute it).
- Friction: **near-zero**, because the carriage rides on linear bearings.
- Moving mass: **950 g total** = 400g motor + 400g chromed steel rods + remainder (~150g) in 3D-printed carriage parts.
- Belt pulley: 2mm pitch, 20 teeth → exact pitch-circle travel per revolution = 40mm/rev (this is **exact**, from tooth engagement, not approximated from a measured diameter — confirmed better than the π×12.5mm≈39.27mm the user initially proposed from a measured 12.5mm bearing/shaft diameter, since 12.5mm is likely the bore/outer diameter, not the true pitch diameter, which for a 20T GT2 pulley works out to ≈12.73mm).
- Target max speed: 60 mm/s, rounded up to 4000 mm/min for margin.
- Pull-out torque source: a manufacturer curve for motor **17HM19-1684S** (24V, 1.68A, "200 Microstep" label on the chart — x-axis in full-step KPPS, y-axis 0–600 RPM using 200 full steps/rev), found at https://manuals.plus/asin/B00PNEQMLY. Chart image was pasted by the user (not re-describable here beyond the data points used below). This curve was read as: (60 RPM, 7 N·cm), (120 RPM, 21 N·cm), (150 RPM, 20 N·cm), (180 RPM, 35 N·cm, peak region), (240 RPM, 35 N·cm), (300 RPM, 33 N·cm), (360 RPM, 32 N·cm), (450 RPM, 28 N·cm), (600 RPM, 15 N·cm). Shape: holding-torque-like at very low speed → dip toward ~7 N·cm around 60 RPM (resonance trough) → rises to a peak ~35 N·cm around 180–240 RPM → falls off at high speed due to inductance limiting current rise time.
- **Important caveat raised and not yet resolved:** `\mustdo` was added recommending the student confirm whether the existing `stepperds` bibliography citation key already covers this specific pull-out-torque curve/datasheet, or whether a separate citation entry is needed, since the earlier-pasted datasheet (holding torque 44 N·cm, R=1.65Ω, L=2.8mH) and this pull-out-torque-curve datasheet may or may not be the same source document.

### Final calculation chain (ready to paste into the report, LaTeX given below was the last thing produced)

**Step 1 — speed to RPM**, using the exact belt pitch travel (40mm/rev), not a diameter-based approximation:
$$n = \frac{v}{\ell_{\mathrm{rev}}}\times 60 = \frac{4000\ \mathrm{mm/min}}{40\ \mathrm{mm/rev}} = 100\ \mathrm{RPM}$$

**Step 2 — pull-out torque at that speed**, linear interpolation on the manufacturer curve between (60 RPM, 7 N·cm) and (120 RPM, 21 N·cm):
$$T_{\mathrm{po}}(n) = T_1 + \frac{n-n_1}{n_2-n_1}(T_2-T_1) = 7 + \frac{100-60}{120-60}(21-7) \approx 16.3\ \mathrm{N\,cm}$$

**Step 3 — required torque**, from the 950g mass, an assumed linear-ball-bearing rolling-friction coefficient μ≈0.004 (**not yet sourced from a bearing manufacturer's catalog — flagged as a `\mustdo`**), and the pulley's half-measured-diameter radius r=6.25mm (kept deliberately distinct from the belt pitch radius r=6.37mm already in the report's Eq. gan-radius — these are two different measured quantities, shaft/bearing diameter vs. belt pitch diameter, and should not be conflated):
$$T_{\mathrm{req}} = r\,F_f = r\,\mu\,m\,g = 0.00625 \times 0.004 \times 0.950 \times 9.81 \approx 0.029\ \mathrm{N\,cm}$$

**Step 4 — margin:**
$$\frac{T_{\mathrm{po}}}{T_{\mathrm{req}}} = \frac{16.3}{0.029} \approx 560\times$$

**Conclusion given to the user:** the motor is not torque-limited at any speed used by this machine — pull-out torque at the worst-case (highest) commanded speed exceeds the actual requirement by roughly 560×. The real constraint, if any, is the **resonance dip** visible in the low-speed region of the curve (torque can drop well below even the plotted 7 N·cm minimum in the unplotted region below 60 RPM, which is where the system's *draw* speed of ~22.5 RPM actually falls) — this is a separate electromechanical self-oscillation failure mode, independent of load, that can look identical from the outside to the already-documented X-axis direction-pin fault (motor appears to simply stop advancing correctly). Flagged as worth checking empirically (listen/watch for a rough patch while slowly sweeping the axis through its full speed range) and worth mentioning alongside the direction-pin fault in the report's Unsolved Problems list as an alternative/contributing explanation for any unexplained step loss.

### Full ready-to-paste LaTeX block (the final artifact produced in this conversation)

```latex
\paragraph{Speed-to-torque margin check.}
The commanded linear feed is converted to a motor speed, read against the manufacturer's measured pull-out torque curve, and compared with the torque the moving mass actually demands.

\begin{equation}
n = \frac{v}{\ell_{\mathrm{rev}}}\times 60 = \frac{4000\ \mathrm{mm/min}}{40\ \mathrm{mm/rev}} = 100\ \mathrm{RPM}
\label{eq:gan-rpm}
\end{equation}

where $n$ is the motor speed in revolutions per minute, $v=4000$\,mm/min is the commanded feed (60\,mm/s rounded up for margin), and $\ell_{\mathrm{rev}}=40$\,mm is the belt travel per revolution already established in Equation~\ref{eq:gan-radius} from the GT2 pulley's pitch ($p=2$\,mm, $N_t=20$ teeth).

\begin{equation}
T_{\mathrm{po}}(n) = T_1 + \frac{n-n_1}{n_2-n_1}\bigl(T_2-T_1\bigr) = 7 + \frac{100-60}{120-60}(21-7) \approx 16.3\ \mathrm{N\,cm}
\label{eq:gan-pullout}
\end{equation}

where $T_{\mathrm{po}}(n)$ is the pull-out torque available at speed $n$, found by linear interpolation between the two bracketing points read off the manufacturer's pull-out torque curve~\cite{stepperds}: $(n_1,T_1)=(60\ \mathrm{RPM},\,7\ \mathrm{N\,cm})$ and $(n_2,T_2)=(120\ \mathrm{RPM},\,21\ \mathrm{N\,cm})$. \mustdo{confirm \texttt{stepperds} in the bibliography is this specific pull-out torque curve (24\,V, 1.68\,A, from manuals.plus); add a separate citation if it is a different datasheet from the one already cited for the motor's basic ratings.}

\begin{equation}
T_{\mathrm{req}} = r\,F_f = r\,\mu\,m\,g = 0.00625 \times 0.004 \times 0.950 \times 9.81 \approx 0.029\ \mathrm{N\,cm}
\label{eq:gan-treq}
\end{equation}

where $T_{\mathrm{req}}$ is the torque needed to move the carriage at constant velocity, $r=6.25$\,mm is the pulley pitch radius (half of $\ell_{\mathrm{rev}}/\pi$ is \emph{not} used here; $r=p N_t/(2\pi)$ already appears as 6.37\,mm in Equation~\ref{eq:gan-radius} from the pitch diameter --- the 6.25\,mm figure instead comes from half the measured 12.5\,mm bearing/shaft diameter and is kept separate because it is a different measured quantity, not the belt pitch radius), $\mu\approx0.004$ is an assumed linear-ball-bearing rolling-friction coefficient \mustdo{cite a bearing manufacturer's catalog value for $\mu$ used here; this is currently an assumed, not a sourced, figure}, $m=0.950$\,kg is the moving mass (0.400\,kg motor + 0.400\,kg chromed steel rods + remainder in 3D-printed carriage parts), and $g=9.81$\,m/s$^2$.

\begin{equation}
\frac{T_{\mathrm{po}}}{T_{\mathrm{req}}} = \frac{16.3}{0.029} \approx 560
\label{eq:gan-margin}
\end{equation}

The pull-out torque available at the highest commanded speed exceeds the torque the carriage actually demands by a factor of about 560, so the drive is not torque-limited at any feed used by this machine.
```

**Still open / not yet resolved by end of conversation:**
1. Confirm the `stepperds` citation key actually points to this pull-out-torque-curve datasheet (manuals.plus, 17HM19-1684S); add a distinct bib entry if it's a different document from whatever `stepperds` currently cites.
2. Source a real linear-bearing friction coefficient from a bearing manufacturer's catalog instead of the assumed μ≈0.004.
3. Decide whether/how to mention the resonance-dip risk in the report (Unsolved Problems list is the natural place, alongside the existing X-axis direction-pin fault item).
4. This closes the `\mustdo` at `3_3_hardware.tex:74` once pasted in — but the *other* `\mustdo`s in that same subsection (motor datasheet part number/revision, driver carrier part number, supply voltage/current ratings, calibrated steps/mm, CAD photo, repeatability overlay, rectangle test, G-code verification overlay) are still open and were not addressed in this conversation.

---

## 5. Conceptual side-questions answered (stepper motor fundamentals)

These were explanatory, not report edits, but useful context if continuing to discuss the hardware design:

- **What a winding is:** a coil of wire around a stator pole piece; the student's NEMA17 is two-phase (two separate windings/coils); current through a winding creates an electromagnet that pulls the rotor's permanent-magnet teeth into alignment.
- **Chopping vs. stepping frequency:** these are independent. Stepping happens at the controller's STEP pulse rate (the student's 1200–3200 steps/s). Chopping happens continuously underneath, at tens of kHz, set by the driver chip's own current-control loop — it's how the driver holds a precise target current in an inductive winding (including while the motor is sitting still at one microstep position), and it's what makes each of the 16 intermediate microstep current ratios actually achievable.
- **Why a stepper motor suits this design:** open-loop position control with no encoder needed (trusts the step count — exactly what R5's tolerance is built on), good low-speed torque (opposite of a brushed DC motor), simple digital (STEP/DIR) driver interface matching the GPIO-pin constraint, precision from counting rather than measuring. Trade-off: a missed step is silent and unrecoverable with no feedback to detect it — which is the whole reason the torque-margin work in this conversation matters.
- **How a step physically happens:** the driver energises a winding current pattern → the rotor's permanent-magnet teeth are pulled toward the nearest matching magnetic "valley" in that field → the driver shifts the current pattern by one microstep's worth → the field's valley shifts → the rotor's teeth follow it. Holding torque is literally a measure of how much external force it takes to pull the rotor out of its currently locked valley.

---

## 6. Relevant persistent memory (already saved, not duplicated into report files)

These exist in the assistant's long-term memory for this project and inform how prior/future report work should be approached — listed here so a new chat without memory access still has the key constraints:

- The report's content rule: no third-party libraries/AI used in the actual pipeline (only Flask/web-plumbing and training/eval scripts are exempted); the report must state honest gaps, not paper over them.
- Never claim the student derived published algorithms himself — frame as building from cited papers (the recogniser architecture is `kizilirmak2023`).
- `REPORT_REQUIREMENTS.md` and `ProjectHistoryDigest.md` (in `Report/`) are the condensed specs to read instead of the original source PDFs for report-writing sessions.
- All equations must be cited per the project's own Appendix 4 rule (clarified in a dedicated memory file) — relevant to why the stepper torque equations above need the `stepperds` citation resolved.
- `Report/FinalReport_v5/REFERENCE_AUDIT.md` tracks citation page-locators and corrected sources — check this before adding/relying on any new citation key.
- The `\new{}` tagging convention in the Lit Review marks non-original wording per a declared-attribution requirement specific to this project (not a general report-writing device).
