# Results, Discussion and appendix revision, 2 November 2026

Based on your supplied SubmissionRevision_2026-11-02.zip. The revised source is Report/ResultsDiscussionRevision_2026-11-02/full_report.tex. The older SubmissionRevision folder and your open segmentation document were left alone.

## Scope

All wording before Results is preserved. Two missing math delimiters in the reproduction pseudocode were repaired so the supplied source compiles. A separate compatibility file defines the supplied unknown command \inP and disables pdfTeX-only microtype options under XeTeX. No Design narrative, headings, equations, diagram assets, Literature Review, Approach, hardware wording, proposal or Conclusion was rewritten.

Results now uses the six qualification tests, summary tables, eleven new blue and silver plots, direct photograph examples and appendix references. QTP1 contains classifier results and the natural-variation geometric test. The older descriptor-correlation experiment is omitted from the revised Results and appendices. The Discussion covers measured behaviour, engineering trade-offs, practical improvements and Appendix 4's required considerations. Appendices retain existing worked substitutions and referenced design diagrams, plus the detailed qualification records. They introduce no new theoretical method.

## Numbers retained and added

- Physical reader score 84.41%, physical writer identification 159/184 (86.41%), pooled page identification 14/14.
- Original validation reader 94.06%, writer 153/156 (98.08%). Existing checkpoint provenance is retained once in the evidence appendix. Existing results are not relabelled as new first-principles or ODROID measurements.
- Direct geometric agreement relative to the writer's own natural variation averages 86.68% with equal writer weight. It uses 83 cross-page repeated words across eleven writers. This is a relative geometric ratio, not recognition accuracy, a Pearson correlation, or proof that 85% of arbitrary handwriting is correct. The conditional word-type bootstrap interval is 83.65 to 89.65%. The nearest-reference check gives 87.89%.
- All 70 supplied feature lengths from ten square-and-triangle samples are within 0.5 mm. Maximum absolute error 0.49 mm, mean absolute error 0.297 mm, RMS error 0.327 mm. These are feature lengths, not absolute bed coordinates. No instrument least count or uncertainty was invented.
- All 25 explicitly completed reproduction requests meet 4 s per character. Classification totals remain 46.26 s and 55.49 s. Supporting host acceleration is separated from embedded measurement.

## Items outside the authorised scope

The unchanged Conclusion still quotes the older 0.37 mm maximum error and 0.697 descriptor correlation. These need reconciliation before submission. The new 86.68% geometric ratio must not simply replace 0.697 as if both were the same metric.

The protected Design's first-principles writer comparison lists 97.51%, while the archived evaluation used for the comparable writer table records 149/156 (95.51%). The supplied console training log peaks at 96.8% on a 155-line validation partition. These are different records and populations. Their wording was not silently changed or combined.

Instrument model, least count, absolute placement/repeatability overlays and a calibrated illuminance series remain unestablished by the supplied records. The new plots use existing measurements only. No training or robot operation was started. No commit, push or deployment was performed.

## Build and checks

Run python build_report.py in the source folder, or compile full_report.tex using pdfLaTeX, BibTeX and two more pdfLaTeX passes. The supplied template remains unchanged. The built-in editor compiler reaches its 20 MiB preview limit for this project, so the delivered PDF uses the successful local pdfLaTeX build.

The PDF was rendered and inspected. All eleven newly generated scientific plots embed Times New Roman and print at least 10 pt in the report. Source labels and references were checked. The source audit and original input hashes are included in the evidence package.
