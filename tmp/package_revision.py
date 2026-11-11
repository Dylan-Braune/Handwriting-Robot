from pathlib import Path
import shutil,json,re,hashlib,zipfile
P=Path('Report/SubmissionRevision_2026-11-02').resolve(); T=Path('tmp/final_report_review').resolve(); out=Path('output').resolve();out.mkdir(exist_ok=True)
# Archive the new derived records before cleaning the compilation folder.
D=T/'derived-evidence';D.mkdir(exist_ok=True)
for f in (P/'Evidence').glob('*'):
 if f.suffix in ['.csv','.txt']:shutil.copy2(f,D/f.name)
# Keep only observed compiler inputs plus the bibliography and build entry points.
used=set()
for line in (P/'.build/full_report.fls').read_text().splitlines():
 if not line.startswith('INPUT '):continue
 q=Path(line[6:]);q=q if q.is_absolute() else P/q
 q=q.resolve()
 if q.is_relative_to(P) and '.build' not in q.relative_to(P).parts and q.is_file():used.add(q)
used.update(P/x for x in ['references.bib','build_report.py','README.md'])
unused=T/'unused-source-assets';moved=[]
for f in list(P.rglob('*')):
 if not f.is_file() or '.build' in f.relative_to(P).parts or f in used:continue
 rel=f.relative_to(P);dst=unused/rel;dst.parent.mkdir(parents=True,exist_ok=True)
 assert f.resolve().is_relative_to(P) and dst.resolve().is_relative_to(T)
 shutil.move(str(f),str(dst));moved.append(str(rel))
(T/'source-cleanup.json').write_text(json.dumps(moved,indent=2))
# No deletions, the user's input ZIP and all earlier report folders are untouched.
(P/'README.md').write_text('''# Handwriting robot report revision, 2 November 2026

Open `full_report.tex` as the main document.

Build with `python build_report.py`, or pdfLaTeX, BibTeX, pdfLaTeX, pdfLaTeX.
The build writes to `.build/`. Figures are supplied and compilation does not run
experiments, regenerate measurements, train models or operate hardware.

The main report from Literature through Conclusion is 63 pages. Design is 37 pages.
Front matter, the unchanged approved proposal, references and appendices are excluded.
The complete PDF has 133 pages.

This folder contains only compilation dependencies and these build instructions.
The separate supporting-evidence ZIP preserves raw records, derived CSVs, console
log transcriptions and scripts. It is not needed to compile the report.
''')
print('Compile files',len(used),'unused archived',len(moved))
# Record scope and outstanding author checks outside the report.
review=out/'Report_Revision_Notes_2026-11-02.md'
review.write_text('''# Report revision notes, 2 November 2026

## Scope and size

The baseline was the supplied `IntegratedReport_2026-11-01.zip`.
The main report is 63 pages from Literature through Conclusion, compared with
82 pages in the supplied baseline. Design is 37 pages, compared with 50.
The complete PDF is 133 pages including the proposal, front matter, references
and expanded evidence appendices. There are 48 printed references.

All existing main subsection, subsubsection and paragraph headings are retained.
Literature and Approach are byte-for-byte unchanged. The approved proposal PDF
and its wrapper are unchanged. Hardware prose is retained, with the permitted
calculation reductions, CAD restoration and brief figure explanations.

## Changes

- Simplified the explanations from segmentation onwards while retaining general
  formulas, variables, design choices and worked values.
- Moved detailed substitutions into Appendix A, with links from the main text.
- Retained two seam panels and Owen's first feature row in Design. Full originals
  are in Appendix B. Other Design figures and algorithms remain.
- Added panel labels and a short functional table for the eight CAD parts.
- Put explanations before every Design and Results figure. Main figures use fixed
  placement so they follow their introducing text. Captions are shorter.
- Added row-normalised writer matrices for original and physical lines. Each
  reference-writer row sums to 100 percent and shows its actual line count.
  No lines were duplicated or discarded to force equal class sizes.
- Retained 159/184 physical line identifications and 153/156 original identifications.
  The unweighted physical diagonal mean is 85.431 percent, distinct from the
  line-pooled 86.413 percent. The report retains the original pooled score.
- Added a complete Owen photograph in Results, all fourteen physical pages in
  Appendix C, all 184 paired-line scores and all 25 completed ODROID timings.
  Photo pixels are unchanged. LaTeX rotates their display to reading orientation.
- Kept the training-route comparison in Design and transcribed the supplied
  first-principles logs in Appendix C, including the incomplete end of each excerpt.
- Completed the acronyms and all four existing Conclusion headings.
- Corrected the blanket AI and software-library declarations. They previously
  contradicted the documented assistance and implementation. Signatures remain blank.

## Evidence and interpretation

No new physical trials, training runs or gantry operation were performed. New
matrices and page summaries are arithmetic regroupings of the saved prediction
records. All measured accuracies and times are retained. Checkpoint provenance is
recorded in the appendix rather than repeatedly discussed in Results.

The physical page accuracy records and embedded timing records have no verified
one-to-one page/job mapping. They remain separate tables. Development-host timing
improvements are not presented as measured ODROID passes. Feature-length accuracy
is not relabelled as absolute coordinate accuracy. Missing lux measurements and
word overlays were not invented.

## Checks to finish yourself

- Add the complete gantry and measurement-setup photographs. The current 63-page
  main report leaves four pages below the 67-page maximum.
- Confirm the fitted motor specification and the exact `stepperds` source. The
  inherited bibliography record does not identify a supplier or document revision.
  The torque curve and friction assumption remain explicitly conditional.
- Check the full Dziedzic paper for the 10 to 60 mm/s speed range and its page.
  The publisher/DOI record was verified, but the full text was unavailable.
  DOI https://doi.org/10.1016/j.forsciint.2026.112896
- Check the GA12-N20 variant supplier specifications against the installed motor.
  The restored reference is the existing Phipps Electronics listing. Check the
  DML31P listing against the installed switch as well.
- Other existing source page numbers were retained, not newly verified against
  every source PDF. No new page number was invented for unavailable material.
- Obtain the caliper model and least count, fitted Vref and institutional clearance
  or exemption reference. Consent signatures should remain private supporting records.
- Read the corrected declarations against your university's required wording
  before signing. This revision does not certify institutional compliance.
- The protected Approach still says step counts assure accuracy without an
  encoder. That overstates open-loop accuracy. It was left because you expressly
  protected Approach, while Design states the missed-step assumption correctly.
- The protected Literature retains its existing sentence distinguishing selected
  reference-trained weights. It was not silently edited or contradicted elsewhere.

## Validation

The report builds with installed pdfLaTeX and BibTeX. The final build has no
undefined citations, missing references, duplicate labels or overfull boxes.
The main heading hierarchy and retained figure/formula labels were checked against
the baseline. Literature, Approach and proposal preservation were checked separately.
Figures and pages were rendered for visual review. The app compiler timed out
while downloading its TeX environment, so the successful local compiler supplied
this PDF. No tab or the separate open segmentation document was changed.
''')
# Preserve evidence records, without duplicating model binaries or the personal raw dataset.
E=Path('Report/ReportWorkingArchive_2026-11-01_VisualTests/ResultsEvidence')
ev=out/'Handwriting_Robot_Supporting_Evidence_2026-11-02.zip'
with zipfile.ZipFile(ev,'w',zipfile.ZIP_DEFLATED) as z:
 for f in E.rglob('*'):
  if not f.is_file() or 'checkpoints' in f.relative_to(E).parts and f.suffix.lower() in ['.npz','.pt','.pth']:continue
  z.write(f,'ResultsEvidence/'+f.relative_to(E).as_posix())
 for root,sub in [(Path('Report/ReportWorkingArchive_2026-11-01_VisualTests/Evidence'),'DesignEvidence'),(Path('Report/DesignVisualTests_2026-11-01/scripts'),'VisualTestScripts'),(D,'DerivedRecords')]:
  for f in root.rglob('*'):
   if f.is_file() and '__pycache__' not in f.parts:z.write(f,sub+'/'+f.relative_to(root).as_posix())
 z.write('tmp/build_added_evidence.py','RevisionScripts/build_added_evidence.py')
 z.write(review,'REVISION_NOTES.md')
 z.writestr('README.md','''# Supporting report evidence

ResultsEvidence/measurements/RESULTS_PROVENANCE.md is the original result index.
ResultsEvidence/measurements/outputs contains the measured JSON records and
ResultsEvidence/measurements/raw contains the preserved ODROID log.
DerivedRecords contains both confusion matrices as raw counts and row percentages,
all physical paired-line records and transcriptions of the supplied training logs.
The writer order is 150, 152, 153, 384, 552, 588, A, B, O, T, Y.
A=abhinav, B=dylan, O=owen, T=thiya, Y=yeukita.

Model binaries are not duplicated here. The manifest retains their SHA-256 hashes.
The original personal/IAM datasets remain in the project. Full physical-page images
are included in the compilation source ZIP as the appendix figure dependencies.
Historical scripts retain their original data-path assumptions. RevisionScripts
records the report derivation script and is intended to run from the project root,
not as part of compilation. No script needs to run to compile the supplied source.
''')
print('Evidence ZIP MB',round(ev.stat().st_size/1e6,2))
