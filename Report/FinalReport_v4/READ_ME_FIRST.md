# Final report, v4 — state and what is still yours to do

Build with `latexmk -pdf main.tex`. The folder now holds only what the build needs: `main.tex`,
`epr400.cls`, the six `.bib` files, the four content directories and `Figures/`.

Everything else lives in `_dev/` and is not part of the build. To use one of those scripts, copy it
up to this directory first — each works out its paths from where the script file itself sits:

- `_dev/measure_pages.py` — real printed pages per section, read off the folios. **This is the
  only trustworthy page count.**
- `_dev/measure_sources.py` — a fast estimate from the sources, for use while iterating. Good in
  aggregate, noisy per file by up to about half a page.
- `_dev/verify_report.py` — eleven structural checks. All eleven pass as of this writing.
- `_dev/tools_flatten_form.py` — the Part 2 form flattener (section 1 below).
- `_dev/` also holds `REWRITE_RULES.md`, the figure to-do notes, the abbreviation drafts and the
  `qtp_data/` and `qtp_scripts/` directories that Section 4 quotes by filename.

`FinalReport_v2/` and `FinalReport_v3/` are untouched, so nothing is lost.

## 1. Part 2 is FIXED — it no longer needs anything from you

Part 2 used to print as empty boxes, which would have failed the report on its own. The approved
proposal is a PDF form whose fields were never flattened, and `\includepdf` imports page content
but not annotations — and a form field's filled-in value lives in its widget's appearance stream,
which is an annotation. So all the content was in the file and none of it drew.

`tools_flatten_form.py` now draws each widget's own appearance stream into the page content. The
report includes the flattened copy, `2_Project_Definition/project_proposal_flat.pdf`; your original
`project_proposal.pdf` is kept beside it, unaltered. **Nothing was retyped or edited** — the
appearances drawn are the ones the form itself carried — so Part 2 is still the unaltered approved
proposal, as Appendix 4 requires. Regenerate with:

    python -I _dev/tools_flatten_form.py \
      2_Project_Definition/project_proposal.pdf \
      2_Project_Definition/project_proposal_flat.pdf

**Please still eyeball those eight pages against the AMS copy** and confirm every answer, the title
block and the Feedback block appear. That is the one `\mustdo` left in Part 2.

## 2. What has been done to Sections 1 to 3

**Section 1, Literature study.** Rebuilt from your actually-submitted first-semester report
(`Reports/FirstSemesterReport23539195(1).pdf`). All nine of your real sources are cited, and all 45
topics of your original review were verified present by script, plus the new reproduction material
(writer identification, the three generation routes, CTC, LSTM, motion control, centre-line
tracing). Split into the two parts Appendix 4 prescribes — background and context, then how it was
applied — which is also your original "Existing Concepts" / "Intended Contribution" shape. Eight
journal articles, so Appendix 4's six-journal minimum is met.

One number differs from your submitted report on purpose: **IAM has 657 authors, not 627.** 657 is
what Marti and Bunke report and we now cite them directly. Be ready to explain if anyone compares.

**Section 2, Approach.** Two pages, with the concept, the design alternatives and the trade-offs
Appendix 4 requires. Its duplicate functional-unit table was removed in favour of the fuller one in
Section 3.2.

**Section 3.** Compressed hard from 74 printed pages. Every design equation the code evaluates
survives, the complete circuit keeps every component value, every honest fault admission survives,
and every red marker survives. Code excerpts were cut to 5–15 line extracts because Appendix 4
forbids long program listings but asks for sections of code.

**Attribution was corrected throughout, and this matters.** The report had credited your
recogniser's layer pattern to `marti2002`, the IAM *database* paper. It actually follows
**Kizilirmak and Yanikoglu, arXiv:2307.00664** (`kizilirmak2023`), as `CleanPipeline/TrainText.py:1`
documents. Claims that you derived the gradients or designed the architecture have been rewritten:
the backward passes are now described as the standard published results for those layers,
implemented here without a framework, with each method cited where it is used. That is the honest
and the stronger position — Appendix 4 gives credit for your own first-principles *implementation*,
and building from trusted papers is the engineering process, not a shortcut.

**Appendix 4 rules now enforced by `verify_report.py`:** no first person outside Part 1; every
equation and float referred to in the text; no section opening with a float; no citation off the
keep-list; plus the four checks that each caught a real defect in earlier versions (captions outside
floats, corrupted `\\` row breaks in maths, unescaped `%` silently eating text, duplicate labels).

## 3. Still yours to do

1. **Check Part 2's eight pages** against the AMS copy (above).
2. **Sections 4, 5 and 6 are yours** and untouched. Results is over its guideline and Discussion
   slightly over. Section 6 line 25 still says "Every hand-derived backward pass" — that is the
   claim you asked to drop, and it is in your section, so I left it.
3. **The declarations** in `1_Preamble/1_Preamble.tex`: project history, the AI usage declaration
   (the template's blanket denial was deleted because it is not true; Appendix 4 requires this text
   "modified to reflect the actual facts"), the software library declaration, the language editor's
   name and signature, and your own signature and date. The report is rejected without the last two.
4. **The red markers.** Every outstanding item is marked in red and must be resolved and the macro
   deleted before you submit. Count them with:
   `grep -rn 'mustcomplete{\|mustdo{\|figplaceholder{' --include=*.tex .`
   The largest group is in Section 3.9, and it is where the most marks are available: Appendix 4
   gives **zero** credit for a circuit diagram without calculated component values. The equations
   are written and the circuit drawn; what is missing is your measurements — the A4988 sense
   resistor and the Vref you actually set, the N20's running and stall current, whether a clamp
   diode and base pull-down are fitted, the supply's rating, the carriage masses for the torque
   margin, and the CAD drawings and photographs.
5. **One open question I could not settle:** whether the deployed recogniser's weights were trained
   from random initialisation or warm-started from the PyTorch checkpoint. It changes what Section
   3.4 and declaration 6 may claim.

## 4. If you ever switch the Appendix back on

The Appendix is disabled with `\iffalse` in `main.tex` (around line 181), because under the rule
you set — credit-bearing content in the numbered sections, only credit-free extras in the Appendix
— nothing was left that belonged there.

If you do bring it back, note that `4_Appendices/Appendix_hw.tex:4` and `Appendix_more.tex:5`
mention `fig:gan-calib` and `fig:sys-arch` in LaTeX **comments**, and both labels have since been
deleted. They are harmless now (comments, in files that are not even input), but if you write that
appendix out from those notes the references will dangle. The calibration listing is now the middle
excerpt of `fig:gan-code` in Section 3.9, and the architecture figure's content was folded into
`fig:sys-flow` in Section 3.10 and `fig:des-hw`/`fig:des-sw` in Section 3.2.

## 5. The page budget

Appendix 4's own per-section maxima sum to 77 against its hard cap of 70, so they cannot all be met
at once. Section 3's guideline is 30–40.

Note that the red markers are themselves worth roughly a page of Section 3, and they disappear when
you resolve them — so the version an examiner sees is shorter than what `measure_pages.py` reports
now. Run it after any change for the live figure.
