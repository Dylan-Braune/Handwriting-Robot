# Expanded code theory manual

The existing 335-page manual has been expanded to 373 PDF pages. The original
per-function chapter source files and original bibliography are unchanged.
Explanatory additions sit inside their corresponding software chapters, with
a final chapter for measurement and verification. The main PDF at
`Software/CleanPipeline2/testing/results/CodeTheory/main.pdf` is updated too.

## Files

- `Handwriting_Robot_Code_Theory_Expanded.pdf`: expanded manual.
- `Handwriting_Robot_Code_Theory_Source.zip`: complete standalone LaTeX source.
- `Original_335_page_manual.pdf`: preserved pre-edit PDF.

## Start here

The PDF has one unnumbered title page: PDF viewer page is printed page plus one.
Its table of contents and bookmarks link to the added sections.

| Subject | Section | Printed page |
| --- | --- | ---: |
| Reading map and evidence conventions | 1.1 | 14 |
| Segmentation, integral sums, Otsu, deskew, seams, ownership and crop rejection | 4.47 | 92 |
| CNN feature sequence, dimensions, normalisation, BiLSTM, CTC and writer recognition | 5.42 | 140 |
| Character cuts, thinning, tracing, simplification, shear, descriptors, medoids and profiles | 6.44 | 184 |
| Synthesis coordinates, drift, anchors, joins, joint scoring and candidate repair | 7.51 | 231 |
| Torque, quantisation, electronics, all documented pins, pen/debounce, calibration and motion | 8.72 | 277 |
| Partitions, sampling, losses, gradients, optimisers, schedules and training records | 9.103 | 355 |
| Recorded calculations, numerical verification, metrics, geometry, timing and evidence gaps | Chapter 10 | 359 |
| Source audit | 10.13 | 364 |
| Additional jargon definitions | 10.14 | 365 |
| Persistent method-source links | 10.15 | 367 |

## Evidence boundary

The report used is the unsuffixed Downloads file
`23539195_BrauneDR_FINALREPORT.pdf`, created 3 November 2026, 108 PDF pages.
SHA-256: `A7C44C245CD34032CE6B57FF54C35221B8E85D8C318BA27D45863E02DB9E52E0`.

Read-only source inspection covers CleanPipeline3's paths, segment, recognise,
style, synthesise, reproduce and train files. Original manual prose remains
historical; the reading guide explains the version boundary. No module was
imported or executed. No experiment, coefficient sweep, training, inference,
gantry operation, physical repair or demo preparation was performed. PDF text
extraction, LaTeX/BibTeX compilation and PDF layout rendering were used solely
to author and verify this document.

The additions distinguish algebraic derivations, published methods, reported
observations, source-comment intent and unsupported heuristic optima. They do
not incorporate the other agent's ongoing sweeps. The stroke-width selection
table's page-versus-line inconsistency, missing torque-curve provenance,
conditional hardware assumptions and blocked source access remain identified.
No claim is made that all original manual/report assertions or references have
been independently verified.

The source and report hashes checked against the initial snapshot matched
after authoring. Existing uncommitted work was preserved. No commit or push.

## Build the source

Extract the source ZIP to a directory with a LaTeX installation. Compile
`main.tex` with pdfLaTeX, run BibTeX on `main`, then run pdfLaTeX twice more.
The source bundle requires no project software or model weights.

The final compile has no undefined references/citations or overfull boxes.
Representative added pages, equations, the pin table and references were rendered
for visual review. The PDF contains 373 pages and 54 cited bibliography entries.
