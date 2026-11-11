# Design revision for Prism

Second Design pass, 1 November 2026. It follows the agreed seven-subsection
outline and starts from the manually supplied Prism version. It does not replace
the baseline export, v6, Results, Discussion, Conclusion or any pipeline code.

## Reading copy

`design_preview.pdf` compiles only Section 3 and its cited references, using an
unchanged copy of the supplied `epr400.cls`. Pagination is local to this preview.
Blue completion notes and labelled figure requests are intentional draft markers,
not statements that missing measurements or illustrations have been completed.
The reviewed preview contains 39 Design pages, two reference pages and four
supporting appendix pages. Real figures and full-report import can change pagination.

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
6. Add `Appendices/design_implementation_details.tex` to the existing appendix.
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
- Added eight numbered algorithm blocks, representation contracts, explicit CTC
  and convolution derivatives, actual final glyph examples and ruled/free placement.
- Added a real stored-glyph worked example with reproducible provenance.
- Incorporated supplied training provenance: final weights were framework-trained;
  numerical inference and the implemented first-principles trainer are distinct.
- Included the eleven-writer width-selection study as subsystem design evidence,
  without treating validation accuracy as a full-system qualification result.
- Added supporting calculations and field meanings in an importable appendix.

## Evidence and limits

See `FIGURES_AND_EVIDENCE.md`. No new model training or gantry execution occurred.
The stroke-normalisation illustration was generated from existing crops with the
live image primitives at width fraction 0.17, matching the selected eleven-writer
checkpoint's supplied setting. The reference comparison is an archived
CleanPipeline2 test, not a newly run CleanPipeline3 test.

`SOURCE_AUDIT.md` records checked in-text page locators and eight referenced sources
still requiring manual source/page checks. Those sites carry blue markers in the
PDF. Unpaginated online documents use exact section headings. This pass reuses
the user's existing literature/bibliography wherever appropriate; the current
main-report union has 40 cited keys, including the internal code record.

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
