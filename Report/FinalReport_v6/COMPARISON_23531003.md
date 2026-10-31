# What 23531003's report does that mine does not

Read end to end: all 111 pages of `23531003_Report_v2.pdf` (A. Sreekumar, *Analogue
Neural Network for digit recognition*, "Synara"). This file records what I found, why
his report is stronger, and exactly where mine must change. It is written to be acted
on, not admired.

---

## 0. The two numbers that matter most

| | His | Mine (v5/v6) |
|---|---|---|
| Figures in the design section | **37, all real** | 23, of which **22 are placeholders** |
| Unresolved "MUST STILL BE COMPLETED BY ME" markers | **0** | **50** in Section 3 alone |
| Main report length | ~70 pp (at the cap) | **85 pp (15 over a hard cap)** |

Everything else in this document is secondary to these three. A design section whose
evidence is 22 empty red boxes cannot score well however good the prose is, and a report
15 pages over a hard cap is marked down before it is read. **Fixing figures and length
outranks every stylistic point below.**

Content density is *not* my problem: I have 58 equations, 16 tables, 38 subsubsections
and 74 citations in Section 3. He has comparable counts. Mine is not thin. It is
unfinished and unevidenced.

---

## 1. He closes the loop; I only open it

This is the deepest structural difference and the one worth copying first.

His pattern, repeated perhaps fifteen times:

> **predict with the formula → measure → compare → declare the model validated (or not)**

Verbatim example (§3.6.3):

> "Substituting the resistor values into Equation 31 gives a Thevenin resistance of
> Rth=5kΩ. The corresponding input voltages applied were 1.17V, 1.42V, and 1.67V. The
> frequencies calculated using Equation 35 are: 102.73 kHz, 163.39 kHz and 218.45 kHz.
> The values were measured in simulation to be 93.93 kHz, 156.5 kHz, and 216.83 kHz,
> which are in 10% tolerance. **As such, Equation 35 can be considered to model the
> relationship between the voltage and the frequency correctly.**"

My report states equations and then moves on. I have 58 equations in Section 3 and I
validate almost none of them against a measurement. An unvalidated equation is an
assertion; a validated one is a result.

**Where I must do this:** `eq:gan-spm` (steps/mm) against the calibration;
`eq:gan-latency` (13 ms trip-to-stop) against a scope trace; `eq:gc-trap` (the time
model) against a stopwatch; `eq:seg-pitch` against hand-counted line spacing;
`eq:gc-verify` (IoU) — this last one I already have the machinery for and never ran.

---

## 2. Alternatives are compared **with numbers in a table**, then chosen

He never writes "X was considered and rejected" in prose alone. Every comparison is a
table with measured values:

- Table 3: 3 input-reduction methods × 3 activation functions, each `mean ± std` over
  30 repeats → feature extraction chosen.
- Table 6: 9 values of τ_syn → "does not impact accuracy, so choose on component
  availability".
- Table 7: 5 E-series → E96.
- Table 8: 6 merge tolerances → ±3%, "only a 0.58% drop in exchange for halving the
  number of distinct resistors".

My "Alternatives considered" subsubsections (3.4.9, 3.5.5, 3.6.7) are **prose with no
numbers**. That is the single easiest upgrade available to me: I already *have* the
measurements for several of these (the four-arm C1–C4 join comparison, the namer
configuration sweep, the segmentation parameter choices). They must become tables with
a stated winner and a stated cost.

---

## 3. Every constant is justified at the point it is introduced

Not one number appears unexplained. Samples:

- "a 1nF capacitor was selected **because it is widely available**"
- "3.2V was chosen as the upper value so the voltages could comfortably match what the
  DAC could output (it was able to output up to 5V [20], but **an overly conservative
  safety factor was used to not damage the DAC**)"
- "The minimum was set to 0.1 as **0 would be the same as a lack of information**"
- "The threshold was set at 1MΩ because **the 1% tolerance at this resistance is equal
  to the RC value chosen during neuron design**"
- "A safety factor of 15 was applied... **to ensure the continuous nature of the analogue
  implementation can be captured while keeping training times low**"

This is why merging my appendices into Section 3 is the right call: my constants
currently sit in appendix tables (`Appendix_seg` "Constants of the minor operations",
`Appendix_syn` "Constants of the synthesiser") **away from the decision they justify**.
A reader hits the number in Section 3 with no reason attached. That is exactly backwards
from his report.

---

## 4. The report has a **plot**, including a failure and a pivot

§3.9 "Pulse summing" is an entire subsection documenting a design that **did not work**:

- required slew rate 8.13×10⁶ V/s → "impossible";
- required gain–bandwidth product `GBP = 400 kHz × 14.11 = 5.644 MHz` → not available
  from JLCPCB within budget;
- a fallback RC averager was built and the bench trace showed oversaturation;
- → "**As such, the SNN was not chosen to be implemented in the analogue domain.**"

Then §3.10 opens: "In order to manage the time constraint and the GBP problems mentioned
in section 3.9, the decision was made to rather use a hybrid neural network."

That is a story: **attempt → measured obstacle → quantified pivot → result.** The reader
is carried.

My report buries its failures in "Known faults and limitations" at the *end* of each
subsection, which reads as confession rather than narrative. I have at least three real
pivots that deserve this treatment and currently do not get it:

1. **Character segmentation → line segmentation + CTC** (FU 2.2 abandoned). Currently one
   paragraph in 3.1. This is a genuine measured pivot and should be a chapter.
2. **The ink namer → the shape namer** (the writer-ID failure on an out-of-domain
   photograph, diagnosed by a 15-page × 5-perturbation robustness study). Currently in
   3.7 "Design reconsiderations". This is my single best piece of engineering and it is
   told flatly.
3. **Training on the SBC → offline training** (FU 2.4–2.6 abandoned).

**Action:** promote these into the narrative spine of Section 3 with the obstacle
quantified, as he does with the GBP number.

---

## 5. Expected-versus-actual is declared up front as a budget

> "The system **was expected to lose 10% accuracy** when converting from the digital
> simulation to the analogue domain... This value was expected as Koen's implementation
> [4] lost 12% accuracy during his conversion (89.0% in software vs 77% in hardware)."

and he states his decision rule once, globally:

> "the heuristic used to make all decisions was to **select the option that maximised
> accuracy**."

Then in the Discussion he returns to it: "a drop of 10% was expected... this proved not
to be the case as the final network achieved 88.3%, **which is only a drop of 0.9%**."

I have no declared decision heuristic and no declared error budget. I should state both
in 3.1 — my heuristic is genuinely different from his and worth saying plainly
(something like *prefer the option that survives a change of pen, camera and lighting,
even at a cost in headline accuracy* — which is exactly what the shape-namer decision
did).

---

## 6. Requirements as numbered lists, then the design derived from them

§3.6.1 opens:

> "The neuron must satisfy two requirements:
> 1. It must reset the membrane potential to 0 V after every firing event...
> 2. It must produce spikes with a consistent width, while the firing frequency varies..."

and the rest of the subsection discharges them in order. Several of my subsubsections
open with context instead of obligations. Cheap, high-value fix.

---

## 7. One concrete worked example, carried end to end

Figure 10 and the text around it take **a single MNIST sample (a '4')** and explain
three of its sixteen feature voltages — why vertical spread is low, why the skeleton has
four endpoints, why loop_count sits at its 0.1 V floor — and then says what the feature
*cannot* do: "it rules out '0', '8', '9', '6', but cannot distinguish a '4' from a '1',
'3', '5', or '7'."

Later, the same device at the other end: MNIST test entry 9478 is the one sample the
board classifies into *no* output, and he tracks it down with a table of its summer
voltages and explains it by the shifted operating window.

I have **no single worked example carried through my pipeline.** I should take one real
page — one of the webcam pages — and carry it through conditioning → line crops →
reading → naming → style profile → synthesis → G-code → drawn output, as a running
example with a figure at each stage. This would also fill a large share of my 22 empty
figure slots with material I can generate today.

---

## 8. Citation practice: denser, and with locators

He cites nearly every factual sentence, and uses page/chapter locators routinely:
`[23, pp. 135]`, `[23, pp. 666]`, `[26][ch.8]`, `[35, p. 453]`, `[29]` for a datasheet
interpolation. He cites **personal communication** properly ([16] is an email, dated and
timed) and web sources with access dates.

He has 52 references for a 70-page report. I have **38 distinct keys used out of 97 in
the .bib** — so I am citing less than half of what I have collected, which is odd and
looks like under-citation. My `REFERENCE_AUDIT.md` already imposes a locator rule; it is
not yet applied everywhere.

**Action:** audit Section 3 for uncited factual claims, and add locators to the textbook
and datasheet citations.

---

## 9. Literature study earns the design decisions

§1.4 "Other implementations of Spiking Neural Networks" evaluates each prior work
**against his own numbered requirements** and rejects it for a stated, specific reason:

> "their implementation used 81 inputs as opposed to the maximum of 16 that this system
> is allowed to use... their system used 180nm CMOS transistors and a complex neuron
> that is not feasible... uses a Binary Exponential Charge Injector which would require
> complex circuitry... **Overall, this implementation only shows feasibility, however
> their use of the Schmitt Trigger to generate the spikes was noted.**"

Note the last clause: he takes one specific idea from each rejected paper. My literature
study surveys methods but does not consistently say *"this is rejected for this project
because of requirement R_n"*.

---

## 10. Qualification tests follow a rigid repeated template

Every test: **Objectives → Equipment used → Test setup and experimental parameters →
Steps followed (numbered) → Results or measurements → Observations (numbered) →
Statistical Analysis.**

Specific things he does that I do not:

- **Defines each metric before using it**, with the rationale for any threshold: "Two
  results are considered close when the gap between the winning class and the
  second-highest is less than 10% of the winner's output. 10% was chosen as..."
- **Engineers statistical validity in**: 6 hardware runs across **2 boards**, software
  averaged over 100 runs, "to increase the statistical validity of the test".
- **Bounds the result**: "Accuracy after worst-case drop" assumes every close call is
  wrong → 82.36%, **still above spec**. This is a far stronger claim than a point
  estimate. I should do the same with my per-line writer-ID agreement.
- **Observations are a numbered list of 13 specific readings** off the confusion matrix,
  not prose.
- **Per-unit forensic analysis**: all ten output neurons analysed individually, hardware
  versus software, each with a causal explanation.
- **Compares against published prior art on the same benchmark, with himself in the
  ranking** (Table 16: seven analogue MNIST implementations, his at 88.31%), and
  explains honestly why the better ones are better (more inputs, transistor-level).

I have no such comparison table against published work. I could build one for
handwriting synthesis/writer-ID and it would be strong.

---

## 11. His appendix rule (which answers "what goes in the appendix later")

Observed from what he actually put there:

- **A** — the *same* worked example repeated for all ten digits (one is in the main text).
- **B** — the *same* calculation repeated for the remaining neurons (one is in main).
- **C** — a **subsystem that is not part of the deliverable**, the test rig (15 pp).
- **D** — all six confusion matrices (the worst one is in main).
- **E** — the full bill of materials.

The rule: **the main report carries one of each kind of thing, fully explained; the
appendix carries the remaining instances, the full enumerations, and anything that is
not the system itself.** Nothing a reader needs in order to *accept a design decision*
is in his appendix.

By that rule my merge is correct — the derivations and constants belong in Section 3 —
and what should move *out* of my main report is: full constant tables (keep the ones
that justify a decision), repeated per-writer breakdowns, and prepared-but-unused
worksheets.

---

## 12. Things my report already does as well or better — keep these

Being fair, so I do not throw away strengths while copying him:

- **The 3.1 deliverable/implementation/completion table** — I already have exactly his
  Table 1 device, over two tables.
- **Honesty instrumentation** — the `\mustdo` / `\mustcomplete` macros are a better
  discipline than anything in his report; they just must all be *resolved and deleted*
  before submission.
- **Separation-of-concerns argument** for G-code (generator knows no electronics,
  interpreter knows no handwriting) is a genuinely good piece of design writing.
- **Library-free claim** — "no machine-learning framework supplies any layer, gradient or
  optimiser" is a stronger claim than his (he used PyTorch's Adam and says so).
- **The from-first-principles gradient check** is a verification he has no equivalent of.
- **Algorithm blocks** — he has none; Aston's report uses them well and I already have 6.

---

## 13. Ordered action list

**Blocking (do first):**

1. Cut the main report from 85 pp to ≤70. This is a hard cap, "not allowed under any
   circumstances". After merging the appendices in, decide what leaves.
2. Produce real artwork for the 22 placeholder figures, or delete the claims that depend
   on them. Start with the ones I can generate from data I already have today:
   G-code verification overlay, segmentation stages, per-line report, drawn output.
3. Resolve or delete all 50 `\mustdo` markers in Section 3.

**High value, low effort:**

4. Turn every "Alternatives considered" into a comparison **table with numbers** and a
   stated winner and cost.
5. Close the loop on the equations I can: predict → measure → compare → verdict.
6. Open subsubsections with numbered requirements.
7. State the decision heuristic and the error budget once, in 3.1.

**Structural:**

8. Promote the three pivots (character→line+CTC, ink→shape namer, on-board→offline
   training) into the narrative with the obstacle quantified.
9. Carry one worked example — a single real page — through the whole pipeline.
10. Add a comparison table against published work on the same task.
11. Audit Section 3 citations: add locators, cite the unused half of the .bib or drop it.
