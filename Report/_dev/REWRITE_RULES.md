# Section 3 compression brief — read this fully before editing

Goal: Section 3 goes from 72 printed pages to **under 40**, without trivialising the
engineering. The student's instructions and Appendix 4 agree on how, and both are binding.

## 1. The framing, which several earlier passes got wrong

The point of the project is to **follow the engineering process: build a system from trusted
published approaches**, not to invent algorithms. Research papers publish an *approach*, not
source code, so implementing a published approach from its description is the engineering
work and is exactly what earns credit.

Appendix 4: *"you will only get credit for your OWN first principles design and
implementation. Off the shelf solutions, copies of designs from (e.g.) blogs or data sheets,
code found in libraries, user interfaces, data bases — none of these require engineering
design effort and therefore do not carry any credit."*

So for every method, the report must make three things plain, in this spirit:

1. **Where the approach came from**, cited at the point of use.
2. **What was decided for this project** — geometry, parameters, ordering, thresholds,
   integration, deviations and why.
3. **That the implementation is the student's**, with no library doing the arithmetic.

### Hard prohibitions

- **Never claim the student derived, invented or originated an algorithm or an equation.**
  The recogniser architecture in particular is **not** his design: it follows Kizilirmak and
  Yanikoglu, *CNN-BiLSTM model for English Handwriting Recognition*, 2023
  (`\cite{kizilirmak2023}`, arXiv:2307.00664) — 12 conv layers (32/64/128 filters, 2+4+6 per
  stage), 2x2 max-pool after stages 1 and 2 only, height collapsed by adaptive max-pool, a
  2-layer BiLSTM of hidden width 256 with no dropout, and a linear plus log-softmax CTC head.
  His own choices there are the 64x640 input (halved from the paper's 100x960 because full
  size cost about an hour per epoch on a CPU), the data pipeline, the label cleaning, the
  augmentation and the staged training recipe.
- **Do not write "this is from first principles" or "this satisfies R1 and R2" as a refrain.**
  State provenance once, where the method is introduced, and move on. The student explicitly
  objected to that padding.
- **Do not plant unsupported claims.** If the project files do not say it, do not write it;
  put a `\mustdo{...}` instead.

## 2. Appendix 4 rules that bind this rewrite

- **No first person anywhere in Part 3.** *"This is the only section of the report written in
  the first person. No other section of the report may be written in the first person."*
  (That is Part 1.) Use impersonal/passive constructions. `verify_report.py` checks this.
- **No long program listings.** *"No long program listings are allowed other than in the
  technical documentation. However, you may (and should) reproduce sections of code."*
  So: short excerpts of **5 to 15 lines** showing the lines that matter, not whole functions.
  This is the single biggest page saving available — Section 3 currently carries 389 verbatim
  lines across 21 blocks, roughly 10 pages.
- **Spend the space on the most important aspects**, which *"would typically be the first
  number of requirements in your approved Project Proposal (as the requirements should be in
  order of importance)."* So R1 and R2 content earns its pages; a minor helper does not.
- **No credit for engineering basics** ("e.g. Ohm's law"). Standard derivations go.
- Every equation and every float must still be referred to in the text; floats may not run
  across a page break; captions below, in bold.
- If a figure is taken from another source, credit it in the caption.

## 3. What to cut, in order

1. **Derivations, in full.** Keep the *result* that the code evaluates, cite its source, drop
   the algebra that gets to it. This is both shorter and more honest, since the derivations
   are not the student's.
2. **Long snippets to short excerpts.** Keep the lines that show the design decision; replace
   the surrounding boilerplate with one sentence. Keep the `\texttt{File.py}` attribution.
3. **Compact the floats.** Smaller fonts (8 pt minimum, 6 pt allowed for circuit component
   values), tighter `\tabcolsep` and `\arraystretch`, merged near-duplicate tables, TikZ
   `scale=` and fewer label words. A table that only restates prose goes.
4. **Summarise, do not delete, the engineering.** If a segmentation stage needs a particular
   algorithm, that algorithm stays — named, cited and with its parameters. Compress the
   narration around it instead. Three sentences of explanation can become one; a stage cannot
   become zero.
5. **Only then** drop coverage of small functions that are not core to the project's
   functionality. Say nothing rather than half-say it.

## 4. What must survive

- Every algorithm the deployed code actually runs, with its parameters and constants.
- Every design equation the code evaluates, and the hardware calculations with worked values.
- The complete circuit diagram with component values, and the design summary tables.
- Every honest admission and known fault. These are GA9 marks: the lambda = 0 live path, the
  judge that also drives the search, the X-axis direction fault, the missing clamp diode and
  base pull-down, the unmeasured R5, the 28 per cent ink classifier on drawn output, the
  ten-of-thirteen writer coverage in the test harness, the CTC empty-label fault.
- Every `\mustdo` and `\mustcomplete` marker, unless the thing it asks about has been removed
  from the report entirely.
- All externally referenced `\label`s. Grep the whole tree before deleting any label.

## 5. Before handing back

Run `python verify_report.py` from `Report/FinalReport_v4/` and report its output. Do not run
`latexmk` (the parent session builds) and do not touch git, `main.tex`, Part 1, Part 2,
Section 1, Section 2, or Sections 4 to 6.
