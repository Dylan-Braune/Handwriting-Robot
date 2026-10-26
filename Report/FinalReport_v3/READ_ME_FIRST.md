# Final report — what is here and what you still have to do

| File | What it is |
|---|---|
| `main.pdf` = `CompleteVersion_final.pdf` | **The current report. 141-page main report.** Every piece of credit-bearing content sits in the numbered sections; the Appendix is switched off because nothing was left that belonged in it. 52 figures, 42 tables, 148 equations, all cross-referenced. |
| `CompleteVersion_112pages.pdf` | Snapshot before the final accuracy pass. Superseded. |
| `FullVersion_over_70_pages.pdf` | An earlier 90-page snapshot. Superseded. |
| `../FinalReport_v3_checkpoint_76pages.pdf` | A 76-page snapshot from when supporting material still sat in the appendix. |

## What the final accuracy pass changed

The report was audited three ways: against Appendix 4, against the live `Software/CleanPipeline/`
code, and for sourcing. The code audit found that large parts were accurate — every parameter count,
the architecture, the charset, every gradient-check figure, every optimiser and augmentation
parameter, all six stroke-normalisation steps, the whole of §3.7, all of §3.8/§3.9 kinematics and
all 18 server routes were verified correct. It also found real defects, now fixed:

- **§3.3 was describing the wrong program.** All 15 pages documented the retired
  `Software/CNN/SegmentPage.py`; only 2 of ~130 quoted constants existed in the deployed
  `SegmentLean.py`, and it claimed row projection had been *rejected* when the live code uses it.
  Rewritten against the deployed module, with the nine equations it does not evaluate explicitly
  marked as retained only because Section 4's timings were measured on the old one.
- **The qualification tests silently covered ten writers, not thirteen** — the harness
  (`qtp_scripts/qtp_common.py:36`) excludes `abhinav`, `owen` and `thiya`, which are the three
  weakest on the shape judge. Now disclosed by name in three places.
- **A figure that existed in no data file** — "98.6% confidence over 28 detected lines" was the
  character accuracy mis-copied as a writer confidence. Replaced with the real single-line 89.4%.
- **A measurement declared missing that was in `qtp_data/`** — page classification timing, now
  reported (40.3–100.1 s against 26.4–43.9 s budgets).
- **§6.2 claimed 93.67% was "above the 95% target"**. It is 1.33 points below.
- **§5 said recognition time exceeded its spec** when it is 2.8× inside it.
- **Two false first-principles claims**: "no part of training depends on an external library" (SciPy
  is on the live trainer path) and "no training library runs on the board" (start-up imports torch
  through `WebRenderHelpers`). Both narrowed to what is true and defensible.
- **§3.5's "all 84 backbone tensors are bit-identical"** — 0 of 84 match.
- **λ is absent from all 13 profiles**, so the deployed path runs at λ = 0; §3.6's table and §3.7's
  anchor argument now say so.
- **The profile builder cannot run** (`BuildStyleProfile.py:312` imports a name that does not
  exist), so the stored profiles cannot be regenerated. Now stated.
- **`tab:sty-writers` rebuilt from the 13 JSON files** — every row was stale and three writers were
  missing.
- **16 backward-pass equations added to §3.4/§3.5.** The report claimed every gradient was derived
  by hand but contained exactly one; the derivations now match the code and tie to the 78-
  configuration gradient check.
- **Sourcing**: 73 cited entries, 41 journal articles, 1:1 correspondence, four mis-citations fixed,
  datasheets cited for the component values.

`FinalReport_v2/` is untouched, so nothing was lost.

## The page count

The main report is **112 pages against Appendix 4's hard maximum of 70**, which the document
says is "not allowed under any circumstances". That is a deliberate result of your instruction to
put all necessary content in the sections and keep only credit-free extras in the appendix: there
is no arrangement that satisfies both the 70-page cap and that instruction.

For scale, the main report now holds 52 figures, 41 tables and 115 equations. When the appendix
was carrying the design, Section 3 held 59 equations, 7 figures and 3 tables and the appendix held
48, 32 and 31 — the appendix was larger than Section 3, and 103 of its 117 numbered items were
cross-referenced from the main report, so the main report depended on optional reading.

Section lengths against the guidelines: Literature study 6 (4-8), Approach 3 (1-2), Design and
implementation 74 (30-40), Results 18 (10-15), Discussion 8 (6-8), Conclusion 3 (1-4).

Deciding what to do about this is yours. The honest options are: submit over the cap and accept
the consequence; cut scope from Section 3 (the four subsystems are what make it 74 pages); or put
the most reference-like material back into the appendix and accept that it is optional reading.
`python measure_pages.py` reports the live figure after any change.

Build with `latexmk -pdf main.tex`. Check the page budget with `python measure_pages.py`
(the v2 copy of that script was broken — it matched the table of contents instead of the body
headings and reported a 2-page main report).

---

## 1. Fix this first — it fails the project on its own

**Part 2 renders blank.** `2_Project_Definition/project_proposal.pdf` is an unflattened PDF form.
Its 56 fields hold your answers, but `\includepdf` does not draw form-field appearances, so the
project description, technical challenges, functional analysis, requirement table, field conditions
and student tasks all print as empty boxes — and so do the title, student number, study leader and
the Feedback block. Proof: the string "The need for handwriting still exists" appears on zero pages
of the built PDF.

**Fix:** open that PDF in Acrobat or Edge, use *Print to PDF* (which flattens the fields into the
page content), replace the file, rebuild, and confirm page 1 shows the title, student number, study
leader and Feedback block.

## 2. Declarations you must complete and sign

In `1_Preamble/1_Preamble.tex`, marked with red boxes:

- **Project history** — I wrote what the project files support (no predecessor project; bought
  components; borrowed layer pattern; public database; libraries). Confirm the first sentence.
- **AI usage declaration** — the template's blanket denial was **deleted because it is not true**.
  Appendix 4 requires this text "modified to reflect the actual facts" and requires you to "clearly
  indicate if any AI tools were used for any aspect of the work or writing". A replacement
  disclosure is written; check it word by word and make it exactly true.
- **Software library declaration** — the required exceptions paragraph is written out (NumPy,
  Pillow, SciPy in two named places, Flask, OpenCV for the camera, libgpiod for the pins, Hugging
  Face `datasets` once, PyTorch for the writer-ID weights). Confirm, and settle the one open
  question: whether the deployed recogniser's weights were trained from random initialisation or
  warm-started from the PyTorch checkpoint.
- **Language editor** name, signature and date — the report is rejected without this.
- **Your signature and date** — the report is rejected without this.

## 3. The red markers

Every outstanding item is marked in red and must be resolved and the macro deleted before you
submit:

- **`MUST STILL BE COMPLETED BY ME`** — a boxed red block, used where the consequence is severe.
- **`[MUST STILL BE COMPLETED BY ME: …]`** — the same thing inline, where a box would break a table
  or a sentence.
- **`FIGURE MUST STILL BE INSERTED BY ME`** — a red framed box reserving space for a figure.
- **`[NOT YET MEASURED]`** — a measurement the protocol defines but which was not taken.

Count them with:
`grep -rn 'mustcomplete{\|mustdo{\|figplaceholder{\|nym{}' --include=*.tex .`

The largest group is in §3.9, and it is where the most marks are available: Appendix 4 gives **zero
credit** for a circuit diagram presented without design equations and calculated component values.
The equations are written and the circuits drawn; what is missing is your measurements — the A4988
sense-resistor value and the Vref you actually set, the N20 motor's running and stall current,
whether a clamp diode and base pull-down are fitted, the 24 V supply's rating and how it is
converted to the board and pen-motor rails, the carriage masses for the torque margin, and the CAD
drawings and photographs.

## 4. What changed in this version

- **§3.9 rewritten** with a component table giving the reason for every choice, and design equations
  for the belt resolution and pulley radius, the torque requirement, the A4988 coil-current setting
  and its thermal derating, the switch pull-downs with their bounds and RC settling, the BC337-40
  base drive and saturation margin, and the bulk-capacitor sag. Two new figures: a wiring diagram
  and a two-panel circuit figure for the switch input and the pen-motor stage.
- **The gradient check is now real.** `Software/CleanPipeline/GradCheckNumpyNN.py` and its log are
  committed. 78 configurations, 77 passed, agreement between 4.9e-12 and 1.4e-10. §3.4.7 previously
  claimed "22 configurations" with no code behind it; it now reports what the run actually produced.
  The one failure is a genuine bug — the CTC loss raises a shape error on an empty transcription —
  reported in §3.4.7 and §5.1.3.
- **Truthfulness corrections.** R6 downgraded from "Met" to "Partially met" (the proposal's
  random-writer reproduction was never carried out). The "Equipment used" blocks now say the
  development laptop, not the ODROID N2+, because that is where every figure was measured. The
  proposal's ticked undertaking that all processing would be on an embedded platform is now
  confronted in §5.1.3. The writer count is corrected from "eleven IAM and two personal" to eight
  and five.
- **Observations blocks rewritten to be objective** — Appendix 4 forbids interpretation there, and
  all six contained verdicts. The verdicts moved to §5.1.
- **Faults found and fixed:** unescaped `%` signs that silently deleted text from the built PDF;
  corrupted `\\` row breaks inside the LSTM and CTC equations; a `\caption` outside a float that
  broke the build; a capacitor drawn to nothing and a backwards base-drive arrow in my own figures;
  the §3.10 claim that 60 characters at 25 ms is 3.5 s (it is 1.5 s, or 3.5 s including the 2 s
  allowance).

## 5. Things I could not settle

- Whether the deployed recogniser's weights were warm-started (affects §3.4 and declaration 6).
- Whether the recogniser was fine-tuned on two personal writers or five — the appendix data table
  still describes the two-writer set; flagged in place.
- The genuine handwriting word accuracy is 70.5 % in §6.2 and 71.2 % in §4 against the same 89.0 %
  character accuracy; one is wrong and I could not tell which.
- Ethics clearance status for the handwriting samples, and documented consent from the five
  contributors.
