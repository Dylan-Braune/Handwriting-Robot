# Handwriting robot report revision, 2 November 2026

Open `full_report.tex` as the main document.

Build with `python build_report.py`, or pdfLaTeX, BibTeX, pdfLaTeX, pdfLaTeX.
The build writes to `.build/`. Figures are supplied and compilation does not run
experiments, regenerate measurements, train models or operate hardware.

The main report from Literature through Conclusion is 67 pages. Design is 40 pages.
Front matter, the unchanged approved proposal, references and appendices are excluded.
The complete PDF has 139 pages.

The figures were enlarged and reset in Times on 2 November 2026, which added those
pages; no text, number, table or citation changed. Every generated figure is drawn
at the width of the text column, 455.24 pt, and included at `width=\linewidth`, so
LaTeX does not rescale it and the type sizes set in the figure are the sizes on the
page. The six subsystem illustrations drawn on a 720 pt canvas are the exception:
LaTeX scales those by 455.24/720 and their type is set that factor larger to
compensate. `tmp/audit_figures.py` in the repository checks this on the finished
PDFs.

This folder contains only compilation dependencies and these build instructions.
The separate supporting-evidence ZIP preserves raw records, derived CSVs, console
log transcriptions and scripts. It is not needed to compile the report.
