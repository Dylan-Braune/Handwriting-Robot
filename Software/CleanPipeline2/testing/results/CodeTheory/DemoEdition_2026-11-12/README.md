# Code theory manual: current demo edition

Updated 12 November 2026. 445 pages.

The current PDF is ../main.pdf. Its standalone editable LaTeX source is this
directory's main.tex. The parent directory's old main.tex and codetheory_*.tex
are the historical source, preserved rather than overwritten. The previous
373-page PDF is preserved under output/Code_Theory_Demo_2026-11-11/Previous_373_page_manual.pdf.

Build from this directory: pdflatex main.tex; bibtex main; pdflatex main.tex;
pdflatex main.tex. These commands build documentation only.

Scope: static inspection of fourteen Software/DemoCode Python modules and the
startup script, including web/server.py and web/lineworkers.py. The latter was
refreshed during final review after a concurrent external edit. Exact captured
file hashes are in SOURCE_MANIFEST.json. The source ZIP includes the captured
source so the reference can be checked independently of later code edits.
346 top-level functions/classes and 57 class methods are inventoried.

The report-to-code index links all 35 numbered report equations to their
appropriate code-stage explanations. Module chapters follow input-to-output
stages, with relevant report theory, methods, alternatives and library
comparisons beside them. Shared methods are cross-referenced. Training has its
own chapter outside live inference. The function index and coverage CSV locate
current names and source lines.

Checks completed: citation-key resolution; LaTeX cross-reference resolution;
complete callable inventory; all 35 equation homes; current source hash match;
PDF rendering and visual sampling/contact-sheet review. No robot code was
executed, no inference/training or hardware tests were run, and neither the
report nor the implementation was edited by this documentation task.

The source audit flags unsupported ignoreRuling use in ApiGenerate, unsupported
profile/uniformInk keywords to RenderTrajectory, and SegmentLines(timer=...)
in retracing. These are static findings, not repaired or runtime-certified paths.
Published method sources are distinguished from project heuristics, derived
arithmetic and historical measurements. Unverified torque-curve points and
component/current assumptions remain explicitly qualified. Performance
comparisons state semantic and implementation differences without invented
timing or accuracy measurements. This is not a certification of deployed
ODROID files, checkpoint arrays, or every historical report reference.
