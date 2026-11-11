from pathlib import Path
import re,json
ROOT=Path(__file__).resolve().parents[2]; B=ROOT/'Report/DesignPolished_2026-11-01'
for p in list((B/'3_Main_Report/design').glob('*.tex'))+list((B/'Appendices').glob('*.tex')):
 t=p.read_text(encoding='utf-8')
 # Cite substantive introductory paragraphs that share a block with a heading.
 blocks=re.split(r'(\n\s*\n)',t)
 module={'3_1_summary.tex':'project design summary','3_2_system.tex':'web/server.py',
 '3_3_hardware.tex':'reproduce.py','3_4_segmentation.tex':'segment.py',
 '3_5_classification.tex':'recognise.py; train.py; trainlibrary.py',
 '3_6_reproduction.tex':'style.py; synthesise.py; reproduce.py',
 '3_7_integration.tex':'web/server.py; reproduce.py',
 'design_implementation_details.tex':'train.py; segment.py; style.py',
 'design_subsystem_evidence.tex':'subsystem manifests'}[p.name]
 for i,s in enumerate(blocks):
  if not re.search(r'\\(?:subsection|subsubsection|section)\{',s):continue
  if '\\cite' in s or '\\begin{' in s:continue
  prose=[l for l in s.splitlines() if re.match('[A-Za-z]',l)]
  if not prose:continue
  if p.name=='design_subsystem_evidence.tex':cite=r'~\cite[subsystem manifests]{brauneevidence2026}'
  elif p.name=='3_3_hardware.tex':cite=r'~\cite[hardware clarifications]{brauneevidence2026}\cite[reproduce.py]{braunepipeline2026}'
  else:cite=r'~\cite['+module+r']{braunepipeline2026}'
  blocks[i]=s.rstrip()+cite+'\n'
 t=''.join(blocks)
 t=t.replace(r'\ell\prime_s',r"\ell'_s")
 p.write_text(t,encoding='utf-8')
p=B/'3_Main_Report/design/3_6_reproduction.tex';t=p.read_text(encoding='utf-8')
anchor='Algorithm~\\ref{alg:des-profile} gives their place in the complete profile-building procedure.'
t=t.replace(anchor,anchor+r' For a chosen descriptor dot product of 0.9, prior distance is $1-0.9=0.1$. Three variants with pairwise $L_1$ distances 0.2, 0.6 and 0.5 have summed distances 0.8, 0.7 and 1.1, so the second is the medoid; it remains an observed glyph, rather than an averaged path.')
p.write_text(t,encoding='utf-8')
p=B/'SOURCE_AUDIT.md';t=p.read_text(encoding='utf-8')
t+='''
## Citation and calculation pass requested during final review

Every numbered equation has an immediately adjacent citation to its implemented
convention, input record or primary source, and a numerical substitution (existing
substitutions are retained). Borrowed mathematical derivations also retain their
primary references and inspected page locators in the neighbouring explanation.
Unverified pages now explicitly say `page to verify` inside the cite command as
well as retaining the visible blue source flag. No page was fabricated to make
the report appear complete.

Implementation claims throughout the prose and captions now cite the project
source with file/function locators; recorded hardware/model/experiment facts cite
the accompanying unpublished project records. This introduces one internal
evidence reference, not another outside publication. Chosen calculation examples
are explicitly identified and must not be described as measurements.

The new coefficient sweep is in Evidence/coefficient_sensitivity.json and its
script. It is post-design sensitivity, not proof of historical selection or an
independent segmentation-accuracy result. Input and code hashes are retained.
''';p.write_text(t,encoding='utf-8')
p=B/'DESIGN_COVERAGE_AUDIT.md';t=p.read_text(encoding='utf-8')
t=t.replace("Appendix 4 PDF pp. 12–16 are the main Design authority. Its example software heading\ndoes not require reorganising this user's accepted three-stage explanation into a\nsingle generic Software design subsection.","Appendix 4 pp. 12–17 are the main Design authority. Its text calls for software-design headings as well as allowing project-specific organisation. The user explicitly requested three stage headings in place of a generic Software design heading; this revision follows that choice. The software content is covered, but exact heading conformity is not claimed.")
t='# Superseded by the full polish review\n\nRead `FULL_REVIEW.md` for the current verdict, all Appendix 4 conditions, citation/calculation pass and new coefficient sensitivity. The historical audit below records earlier passes and must not override that review.\n\n'+t
p.write_text(t,encoding='utf-8')
p=B/'README.md';t=p.read_text(encoding='utf-8')
t=t.replace('Visual evidence and pseudocode pass, 1 November 2026.','Full Design review, citation and calculation polish, 1 November 2026.')
t=re.sub(r'The preview contains 42 Design pages, two reference pages and nine appendix pages\.', 'The exact final Design, reference and appendix counts are recorded in `VALIDATION.json`.',t)
t+='''
## Latest full review

See `FULL_REVIEW.md` for the Appendix 4/proposal mapping, reference-report practices,
corrections and remaining conditions, assuming planned diagrams are complete.
All forty numbered equations now have a nearby implementation/input citation and
numerical worked values. External methods retain page-numbered citations where
verified. Eight unavailable page checks remain explicitly flagged, including
`page to verify` inside the cite command. `EQUATION_AUDIT.json` records coverage.

The new `PROJECT_DESIGN_RECORDS.md` supports citations to supplied hardware and
training records and the manifests. It adds one internal evidence reference;
Haines and One-DM rationale reuse existing Literature Review sources. The complete
current baseline-section/Design reference union is 41, including project records.

The original four-width training sweep is retained in Design. A new post-design
seam sensitivity check has eight coefficient settings across six fixed real-page
gaps (48 runs); all default paths match the unmodified implementation. Details
are in the evidence appendix and `Evidence/coefficient_sensitivity.json`. It
demonstrates a trade-off without inventing an optimal coefficient or historical
selection rationale. No pipeline code or settings were changed.
''';p.write_text(t,encoding='utf-8')
