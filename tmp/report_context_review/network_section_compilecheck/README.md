# Compile the full report

Entry point: `full_report.tex`.

With Python and MiKTeX installed:

    python build_report.py

The PDF and temporary LaTeX files are written to `.build/`, keeping this source folder clean. Alternatively compile `full_report.tex` with pdfLaTeX, BibTeX, pdfLaTeX, pdfLaTeX in your LaTeX editor.

Only referenced source, bibliography, class and graphics are included. Evidence, generation scripts, audits, old builds and unused figures were moved to the sibling `ReportWorkingArchive_2026-11-01_VisualTests` folder. Existing project data are preserved there. Figures are already generated; compiling does not run tests, train models or operate hardware.
