# Integrated report: Literature, Approach, Design, Results and Discussion

Entry point: `report_preview.tex`. Run `python build_report.py` here with NumPy,
Pillow, Matplotlib and MiKTeX available. The original epr400 class is unchanged.

This version contains Sections 1-5, the cited bibliography and supporting
appendices. It builds on the reviewed Design/Results/Discussion revision and
uses the author's original Prism export for the Literature Review and Approach.
The original export and previous report versions remain unchanged.

## Editorial changes

- Retained the Literature Review's voice and descriptive wording where accurate.
  Corrected technical claims and linked the topics to their project uses.
- Removed Intended contribution; topic-level application paragraphs and a brief
  Application of the literature summary now supply the rationale.
- Reworked Approach around the proposal's functions, subsystem dependencies,
  alternatives and the final architecture. Removed unsupported chronology,
  multiplied subsystem-accuracy claims and blanket no-library assertions.
- Included both original hardware and software diagrams unchanged, each with
  numbered explanations beneath it.
- Simplified in-text citations to the source and verified page(s); omitted the
  page where unverified. No chapter, section, algorithm or PDF qualifiers remain.
- Corrected existing Ma and Hemanth bibliography records and consolidated all
  cited material in one bibliography. No code-file citations were introduced.
- Added equation-source links and identified project-defined calculations and
  derivations without pretending an external paper supplied the exact rule.
- Removed inline editorial comments, blue checks and absent-figure placeholders.
  Their instructions are preserved in `AUTHOR_COMPLETION_LIST.md`.

## Checks and evidence

`INTEGRATED_REVIEW.json` records build/reference checks, section lengths and visual
review. `EQUATION_SOURCE_AUDIT.md` records the basis of all 42 numbered equations.
`SourceChecks/` records newly consulted literature and the source-page cleanup.
Downloaded papers are working research copies; source URLs are supplied in the
manifest rather than bundling those copies in the delivery ZIP.

The original measurement archive remains in
`ResultsEvidence/measurements/RESULTS_PROVENANCE.md`. Aggregate values are
rechecked by `review_results.py`; the figure script reads preserved local data,
not a session scratchpad. Original image/model evaluation provenance is retained.
Four measured checkpoint copies have unique evidence names and full hashes.
No server checkpoint or runtime setting was changed.

Selected demonstration models remain the library-trained joint reader and the
eleven-writer classifier at normalisation width 0.17. The NumPy models remain
separate comparisons. Physical pages, offline image scoring and ODROID timing
are identified by their actual evidence; no unmeasured pass was invented.

## Scope of this assembly

This is the assembled Sections 1-5 review, not the complete submission. Front
matter, the approved proposal and Conclusion are not added by this task. The
whole-report 70-page main-text cap still requires a final compression pass;
section lengths are recorded honestly in `INTEGRATED_REVIEW.json`. Missing
author records are in the completion list, outside the typeset report.

Copied historical Design audit notes describe earlier revisions. This README,
the current review JSON, equation audit and completion list take precedence.
