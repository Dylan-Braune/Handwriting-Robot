# Design revision for Prism

Full Design review, citation and calculation polish, 1 November 2026. It follows the agreed seven-subsection
outline and starts from the manually supplied Prism version. It does not replace
the baseline export, v6, Results, Discussion, Conclusion or any pipeline code.

## Reading copy

`design_preview.pdf` compiles only Section 3 and its cited references, using an
unchanged copy of the supplied `epr400.cls`. Pagination is local to this preview.
Blue completion notes and labelled figure requests are intentional draft markers,
not statements that missing measurements or illustrations have been completed.
The exact final Design, reference and appendix counts are recorded in `VALIDATION.json`.
It includes supporting calculations and a separate appendix of new offline subsystem
evidence. Full-report import and replacement of remaining placeholders can change pagination.

## Import into the current Prism project

1. Keep a copy of the current Prism project before importing.
2. Replace `3_Main_Report/3_Design_and_Implementation.tex` and add its `design/`
   subdirectory from this package.
3. Add `design_support.tex` at the project root. Include
   `\input{design_support}` once before `\begin{document}` in the report's main file.
   If these packages are already loaded, merge package declarations instead of
   loading the same package twice with different options.
4. Add the supplied Figures. The additional requested figures have exact filenames
   in `FIGURES_AND_EVIDENCE.md`. When a file arrives at that name, the draft slot
   automatically displays it without changing the body text.
5. Merge bibliography entries with the project's bibliography. The original keys
   are retained; the A4988 entry has a verified URL and revision added. Do not
   replace a newer bibliography without comparing it.
6. Add `Appendices/design_implementation_details.tex` and
   `Appendices/design_subsystem_evidence.tex` to the existing appendix.
   Its section label is referenced from Design; merge its heading into the full
   report's appendix structure instead of calling `\appendix` a second time.
7. Rebuild the full report and recheck page numbers, cross-references and layout.
   `design_preview.tex` is a review wrapper, not a replacement for the main report.

The submitted report must remove resolved completion markers, include the real
figures, and report only verified final configuration and measurements.

## What changed

- Preserved the structure and wording of the user's mechanics explanation, and
  the opening stepper discussion as closely as technical correctness allows.
- Replaced planning notes with complete prose across Hardware, segmentation,
  networks, style extraction/reproduction, and integration.
- Corrected the 200/400-step contradiction, RPM units, torque radius and inference
  from a friction-only calculation; separated resolution from physical accuracy.
- Explained ink-to-shape writer identification as a design decision in 3.5.4.
- Separated shared training methodology, first-principles/framework implementation,
  numerical correspondence, and final-model performance.
- Included archived numerical correspondence results with their actual test scope.
- Kept the geometric drawing as commissioning evidence and excluded formal system
  qualification tests and final system performance tables.
- Audited the explanation against Appendix 4, Sreekumar's report, Aston's draft
  and the current main code paths; see `DESIGN_COVERAGE_AUDIT.md`.
- Added numbered algorithm blocks, representation contracts, explicit CTC
  and convolution derivatives, actual final glyph examples and ruled/free placement.
- Added a real stored-glyph worked example with reproducible provenance.
- Incorporated supplied training provenance: final weights were framework-trained;
  numerical inference and the implemented first-principles trainer are distinct.
- Included the eleven-writer width-selection study as subsystem design evidence,
  without treating validation accuracy as a full-system qualification result.
- Added supporting calculations and field meanings in an importable appendix.

## Evidence and limits

This pass adds line-numbered, indented pseudocode with explicit conditions,
loops and returns; 11 algorithm blocks appear across Design and the appendices.
It fills the processing overview, network architecture, segmentation stages,
actual camera capture comparison, original-line glyph alignment and diagnostic
candidate-selection figures. Extended four-profile file renders and a real robot
page photograph appear in the evidence appendix.

New offline checks were run, including NumPy model inference, independent
component/seam references, CTC enumeration/finite differences and G-code file
reconstruction. No training or hardware motion occurred. The archived camera
sequence includes failures; the report retains them without assigning unmeasured
lux values or treating these examples as qualification tests.

To reproduce the new evidence from the repository and original local files:

1. Run `run_visual_bench.py` (12 explicitly synthetic/reduced page cases, profile
   renders, small numerical checks, archived crop alignment and candidate scores).
2. Run `make_camera_evidence.py` (actual camera-roll photographs and native page).
3. Run `extra_subsystem_checks.py` (independent seam and conversion fixtures).
4. Run `make_processing_diagrams.py` (code-derived diagrams).

`Evidence/bench.json` and `Evidence/camera_bench.json` retain settings, exact
outputs and provenance. `EVIDENCE_NOTES.md` explains the limits. Input photos,
models and datasets are not copied into the source package; figure assets and
small generated evidence are included. Helper continuation scripts record how
the initial exploratory run was completed; the main bench now selects the
visually matched archived crop directly, rather than its failed reduced-page crop.

See `FIGURES_AND_EVIDENCE.md`. No new model training or gantry execution occurred.
The stroke-normalisation illustration was generated from existing crops with the
live image primitives at width fraction 0.17, matching the selected eleven-writer
checkpoint's supplied setting. The reference comparison is an archived
CleanPipeline2 test, not a newly run CleanPipeline3 test.

`SOURCE_AUDIT.md` records checked in-text page locators and seven referenced sources
still requiring manual source/page checks. Those sites carry blue markers in the
PDF. Unpaginated online documents use exact section headings. This pass reuses
the user's existing literature/bibliography wherever appropriate; the current
main-report union has 41 cited external keys; no code/internal-record bibliography citations remain.

The final glyph libraries are taken as fixed demo inputs. Manual curation is not
described as an operating feature; historical artifact filenames remain only in
private provenance records. The selected normalisation width is 0.17.

## Build

From this directory, with MiKTeX installed:

```powershell
pdflatex -disable-installer -interaction=nonstopmode -halt-on-error -no-shell-escape design_preview.tex
bibtex design_preview
pdflatex -disable-installer -interaction=nonstopmode -halt-on-error -no-shell-escape design_preview.tex
pdflatex -disable-installer -interaction=nonstopmode -halt-on-error -no-shell-escape design_preview.tex
```

The optional `make_design_examples.py` recreates the stroke illustration using
the repository's existing crop images, NumPy/Pillow, ReportLab and pypdfium2.
It does not require model weights or hardware.

## Latest full review

See `FULL_REVIEW.md` for the Appendix 4/proposal mapping, reference-report practices,
corrections and remaining conditions, assuming planned diagrams are complete.
All forty numbered equations remain. Generic scalar examples have been replaced with real-input traces or removed where no actual run values were available. Borrowed methods
use research citations with verified page locators where available. Seven source/page
checks remain visibly flagged. Repeated citations to the project's code and records
have been removed throughout; occasional code-module pointers remain in prose.
Own design choices, derivations and experiments are identified as project work,
with evidence provenance retained in the source package. See `SOURCE_AUDIT.md`.

The original four-width training sweep is retained in Design. A new post-design
seam sensitivity check has eight coefficient settings across six fixed real-page
gaps (48 runs); all default paths match the unmodified implementation. Details
are in the evidence appendix and `Evidence/coefficient_sensitivity.json`. It
demonstrates a trade-off without inventing an optimal coefficient or historical
selection rationale. No pipeline code or settings were changed.

Detailed CTC/training and motion-file pseudocode and the supporting reference
tables are now in Appendix A. Main Design retains five algorithms; the complete
package contains eleven. All forty numbered equations remain; their real-input trace or derivation basis is recorded in `EQUATION_AUDIT.json`. See `SWEEP_INVENTORY.md` for included studies and older artifacts lacking
enough metadata for a controlled comparison.

Final reading-copy pagination is recorded in `VALIDATION.json`; full-report import changes it.

Current pagination is recorded in `VALIDATION.json`.

## Real system calculations

The manuscript no longer uses `Calculated example:` labels or invented scalar
values as design evidence. Read-only traces use Owen01, its archived first original
line, the actual trained reader, the selected eleven-writer classifier, and final
Owen/Dylan profiles. The generated Owen sentence is a new offline job, not a
historical physical execution. The trace catches and corrects the reader-head
count (74 alphabet characters plus blank = 75 outputs) and uses the actual job
feeds (900/2400 mm/min). Generic optimizer/derivative scalars with no real run
record were removed; the supported theory remains. Hardware gaps were not filled
with invented homing counts, load coefficients or Vref measurements.

Run `trace_real_calculations.py` to reproduce the execution traces. Full-precision
records, code/model/input hashes and settings are in `Evidence/real_calculations.json`.
No optimizer step, training epoch or hardware command is involved.

Section 3.2 now has separate hardware and software system-diagram placeholders
plus the required systems-level description. Drawing guidance is in
`FIGURES_AND_EVIDENCE.md`; both figures remain to be supplied. The existing
processing sketch is a planning aid, not the final software block diagram.

Final verified reading copy: 43 Design pages, 3 reference pages and 14 appendix pages (60 total).
