from pathlib import Path
import json,hashlib,zipfile,shutil
B=Path('Report/GeometricFidelity_2026-11-02');O=Path('output');raw=json.loads((B/'raw_results.json').read_text());st=json.loads((B/'statistics.json').read_text());assert st['independent_reference_check']['max_f1_difference']==0 and raw['checks']['no_neural_modules_loaded']
# Strict JSON finite-value validation.
for name in ['raw_results.json','statistics.json','summary.json']:
 json.loads((B/name).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
# Verify that copies of all as-run scripts match the hashes recorded during measurement.
for fn,h in raw['code_sha256'].items():assert hashlib.sha256((B/'source'/(Path(fn).stem+'_as_run.py')).read_bytes()).hexdigest()==h
# No report-writing operations were performed by this study. Existing edits are preserved.
# Preserve the completed analysis as a separate package.
archive=O/'Handwriting_Geometric_Comparison_2026-11-02.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in B.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:z.write(p,'GeometricFidelity_2026-11-02/'+p.relative_to(B).as_posix())
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
shutil.copy2(B/'FINDINGS.md',O/'Handwriting_Geometric_Findings_2026-11-02.md')
# Durable local context, without changing the authoritative report.
p=Path('Report/REPORT_COLLABORATION_CONTEXT.md');txt=p.read_text();note='''## Classifier-free matched-text geometric study, 2 November 2026

Separate analysis only, `Report/GeometricFidelity_2026-11-02/FINDINGS.md`. No report
LaTeX or deployed code changed. Compared the source-text blocks of all five personal
writers with the corresponding original handwritten sentences. Used preserved line
crops, exact case-sensitive transcript matching, pre-score word-boundary review and
fixed geometric processing. Four small source-annotation corrections are recorded.
No recognition model, writer classifier, training or gantry operation was used.

74 word pairs and 65 unambiguously separated character pairs were evaluated. At the
prespecified 0.10 x-height distance tolerance, mean centreline F-scores are Owen
87.09 percent (25 chars in 7 words), Yeukita 83.26 (19), Dylan 73.97 (18), Abhinav
48.78 (3, too small for a writer-wide claim), and Thiya unavailable (0 separable).
All 65 pool to a mean 80.57 percent. Owen's pooled stroke-point F-score is 85.28
percent. Owen strict foreground IoU is 39.59 and fixed-width IoU 56.99 percent.
Do not relabel F-score as IoU, character transcription accuracy, a count of correct
characters or a Pearson style-correlation coefficient. Only 17/25 Owen characters
individually have F>=85; the reported 87.09 is the mean score, not a pass fraction.

Owen wrong-character control averages 52.24 percent. On 17 cases with a same-letter
other-writer reference, true-writer mean is 88.04 and control mean 74.88. Independent
cKDTree checks agree with raster distance-transform scores on all 65 pairs. Word
resampling gives Owen descriptive 95-percent range 82.72 to 91.83. This is bounded
same-source reconstruction evidence on a separable-character subset, not proof all
writers or future texts meet 85. Whole-word scores are much lower and retained in
full, with normalisation/layout limitations documented.

The proposal's actual demonstration wording counts visibly matching characters.
Prepared anonymised full physical-block read-back sheets and an unscored response
CSV in `blind_readback/`. Its separate answer key must be withheld from readers.
`score_blind_readback.py` requires every line and computes edit-based per-writer
accuracy after actual responses exist. No human scores have been invented. Human
style judgement remains separate from read-back. Source references include Chen
et al. (2022), p.5, adaptive IoU, and the official Tanks and Temples F-score definition.

Deliverables `output/Handwriting_Geometric_Comparison_2026-11-02.zip` and
`output/Handwriting_Geometric_Findings_2026-11-02.md`. Figures, all pairs, controls,
annotation records, input hashes and exact as-run script copies are in the analysis
folder/package. This study did not edit the report. The last delivered main report was 63 pages.

'''
if '## Classifier-free matched-text geometric study' not in txt:txt=txt.replace('# Report collaboration context\n','# Report collaboration context\n\n'+note,1)
p.write_text(txt)
print('Archive MB',round(archive.stat().st_size/1e6,2));print('Analysis validated. No report edits performed.')
