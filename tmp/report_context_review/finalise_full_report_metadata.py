from pathlib import Path
import re,json
R=Path(__file__).resolve().parents[2];B=R/'Report/IntegratedReport_2026-11-01'
for p in [*B.glob('3_Main_Report/**/*.tex'),*B.glob('Appendices/*.tex')]:
 t=p.read_text(encoding='utf-8')
 t=re.sub(r'\\designasset\{([^}]+)\}\{([^}]+)\}\{([^{}]*)\}',lambda m:'\\includegraphics[width='+m[2]+'\\linewidth]{Figures/'+m[1]+'}',t)
 p.write_text(t,encoding='utf-8')
p=B/'design_support.tex';t=p.read_text();t=re.sub(r'\\newcommand\{\\designasset\}[^\n]*\n','',t);p.write_text(t)

p=B/'README.md';p.write_text('''# Integrated report: Literature, Approach, Design, Results and Discussion

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
''',encoding='utf-8')

alltext='\n'.join(p.read_text(encoding='utf-8') for p in [*B.glob('3_Main_Report/**/*.tex'),*B.glob('Appendices/*.tex')])
unpaged=sorted({k for keys in re.findall(r'\\cite\{([^}]+)\}',alltext) for k in keys.split(',')})
p=B/'AUTHOR_COMPLETION_LIST.md';t=p.read_text(encoding='utf-8');t+='\n## Citation-only sources retained without a page\n\n'+', '.join('`'+k+'`' for k in unpaged)+'.\n\nWhere an article was checked only through its abstract or bibliographic record, no precise page is supplied. In particular, obtain the cited edition for Acarnley, BC337, Chellapilla, Douglas-Peucker, Otsu, Postl, Marti, Vinciarelli and the IQS technical page before adding locators.\n';p.write_text(t,encoding='utf-8')
(B/'SourceChecks/README.md').write_text('''# Consulted source versions

`manifest.json` gives the primary download URLs for Chen, Ma, Hemanth, Bulacu,
LeCun and Shi. The consulted LeCun author version is numbered 1-46, so the report
uses that version's pages 3-4 and 6-8 where applicable. The bibliography retains
the journal publication details and points to the author-hosted record.

Chen p. 39 supports recognition terminology, p. 40 the EMNIST data and pp. 40-41
normalisation. Ma pp. 268-270 supports offline recognition and hybrid features;
its existing bibliography pages were corrected to 268-273. Hemanth's paper is
dated October 2021, rather than January 2022; its printed pp. 2459-2460 describe
segmentation and give the expanded IAM counts. Bulacu p. 701 describes texture
and allographic features. Shi's consulted author version pp. 2-3 describes CRNN.

Previously checked Design sources remain identified by their existing records.
JCGM's official 2008 guide pp. 18-19 supports first-order sensitivity and numerical
perturbation; this supplies a source for the derived scale-error and derivative
checks, not a claim that the guide describes this robot's exact equations.

Primary URLs checked additionally:
- https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf
- https://arxiv.org/abs/2003.07212 (FragNet author abstract)
- https://publications.idiap.ch/authors/show/672/title (Vinciarelli record)
- https://www.nist.gov/itl/products-and-services/emnist-dataset
- https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

Page locators are not guessed for inaccessible sources. These working PDFs/texts
are excluded from the delivery ZIP; URLs and the audit are included.
''',encoding='utf-8')
print('Clean source metadata and completion list ready.')
