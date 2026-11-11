from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'Report/DesignVisualRevision_2026-11-01'
p=HERE/'Appendices/design_subsystem_evidence.tex';s=p.read_text(encoding='utf-8')
s=s.replace('\\subsection{Independent component and seam references}',r'''\subsection{Independent component and seam references}
Algorithm~\ref{alg:des-reference-checks} states the two independent reference comparisons.''')
s=s.replace('\\subsection{CTC probability and derivative check}',r'''\subsection{CTC probability and derivative check}
Algorithm~\ref{alg:des-ctc-reference} gives the exhaustive probability and derivative procedure.''')
s=s.replace('Nine controlled fixtures combined',r'Algorithm~\ref{alg:des-file-reference} describes the serialization check. Nine controlled fixtures combined')
p.write_text(s,encoding='utf-8')
p=HERE/'README.md';s=p.read_text(encoding='utf-8').replace('The preview now includes the supporting calculations and a separate appendix of\nnew offline subsystem evidence. Full-report import can change pagination.','The preview contains 42 Design pages, two reference pages and nine appendix pages.\nIt includes supporting calculations and a separate appendix of new offline subsystem\nevidence. Full-report import and replacement of remaining placeholders can change pagination.')
p.write_text(s,encoding='utf-8')
p=HERE/'FIGURES_AND_EVIDENCE.md';s=p.read_text(encoding='utf-8')
s='''# Figure and evidence register: visual revision

This pass fills six formerly pending assets using real data or code-derived diagrams:
processing overview, network architecture, native-page segmentation stages, actual
camera-roll capture comparison, original-line glyph alignment and three-candidate
scoring. It adds four-profile G-code renders and an actual robot-page/crop example
in the appendix. See `EVIDENCE_NOTES.md` for provenance and limitations.

Still pending: hardware overview, assembly photo, complete installed circuit,
camera-arrangement photo, labelled glyph-filtering cases and a genuine joining
example. The hardware overview remains an important user drawing with guidance.
The historical checklist below is retained as the detailed drawing brief; its
older status cells are superseded by this paragraph.

'''+s
p.write_text(s,encoding='utf-8')
p=HERE/'DESIGN_COVERAGE_AUDIT.md';s=p.read_text(encoding='utf-8')
s='''# Visual and pseudocode follow-up

The following prior coverage audit still applies. This pass adds 11 total
line-numbered pseudocode blocks (eight design procedures and three independent
verification procedures), fills six figure slots and adds two appendix figures.
New evidence is recorded in `Evidence/bench.json` and `Evidence/camera_bench.json`.
It includes failures and explicitly separates software inference, synthetic
perturbations, archived real captures and existing physical robot output.
The code's current `rulingMaxExtent` default is 1.25; the previously stated 1.60
was corrected in the Design. No pipeline code, model or final profile was changed.

'''+s
p.write_text(s,encoding='utf-8')
