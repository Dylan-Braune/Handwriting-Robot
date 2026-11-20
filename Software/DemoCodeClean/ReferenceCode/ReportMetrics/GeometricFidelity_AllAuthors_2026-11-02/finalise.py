from pathlib import Path
import ast, collections, hashlib, importlib.metadata, json, platform, shutil, sys, zipfile
import numpy as np

B=Path(__file__).resolve().parent
ROOT=B.parents[1]
x=json.loads((B/'results.json').read_text())
c=json.loads((B/'candidates.json').read_text())
accepted=json.loads((B/'accepted_candidates.json').read_text())
quality=json.loads((B/'quality_review.json').read_text())
W=c['writers']
names={w:w if w.isdigit() else w.capitalize() for w in W}
rows=x['results'];n=len(rows)
improved=sum(r['word_F']>r['line_F'] for r in rows)
declined=sum(r['word_F']<r['line_F'] for r in rows)
notes=['# All-author comparison of two whole-word normalisations', '',
'2 November 2026. Offline measurements on existing photographs. No report, model, server or gantry changes.', '',
f"Both methods were evaluated on the same **{n} words across all eleven authors** with recorded physical reproduction pages. There were {len(c['physical'])} eligible physical words across 44 source-passage lines. The other {len(x['unsupported'])} have no accepted exact same-writer reference token and remain explicitly unassessed. Single-character and non-word tokens are outside this whole-word test.", '',
'The writer-balanced mean F-score is **49.63% using line x-height** and **74.78% using whole-word height**. The word-weighted means are 51.25% and 75.48%. The second method gives a higher writer mean for all eleven writers. Only writer 150 has a whole-word mean above 85%, at 89.88%.', '',
'## Full results', '',
'All numbers below are centreline F-scores in percent. The word counts are the common denominator for the two methods.', '',
'| Author | Assessed / eligible words | Line x-height mean F | Word-height mean F | Words at F >= 85, line / word |',
'|---|---:|---:|---:|---:|']
for w,s in x['summary'].items():
    notes.append(f"| {names[w]} | {s['scored_words']} / {s['eligible_proposals']} | {100*s['line']['mean_F']:.2f} | {100*s['word']['mean_F']:.2f} | {s['line']['at_least_85']} / {s['word']['at_least_85']} |")
notes += ['',
f"At the individual-word level, {x['aggregates']['line']['word_count_at_least_85']}/{n} line-height scores and {x['aggregates']['word']['word_count_at_least_85']}/{n} word-height scores reach F = 0.85. A mean F-score is not the fraction of correct words. Word-height normalisation raises the individual score on {improved} words and lowers it on {declined}; ties account for the remainder.", '',
'per_author_summary.csv additionally reports medians and strict foreground IoU. word_scores.csv records every accepted physical word, chosen references, precision, recall, registration offsets, skeleton-point counts, IoU and scores at two other tolerances. all_reference_comparisons.csv retains all bank comparisons rather than only winners.', '',
'## What the methods compare', '',
'Method 1 uses the existing image-processing body-band estimate as the line x-height. Each word is scaled isotropically with the same factor as its line to a nominal 48-pixel x-height. Method 2 tightly crops that normalised word and rescales its total ink height to 48 pixels, also isotropically. The second method preserves the word aspect ratio but removes its absolute height relative to the original line. Ascenders, descenders and punctuation affect total word height.', '',
'Both methods thin the foreground masks, centre the words and search translations from minus five to plus five pixels on each axis. Translation minimises the symmetric mean nearest-centreline distance. Precision is the fraction of physical skeleton points within 4.8 pixels of the original skeleton, and recall reverses the point sets. F = 2PR/(P + R). The point-set precision, recall and harmonic mean follow the [Tanks and Temples evaluation tutorial](https://tanksandtemples.org/tutorial/), Appendix B. The two-dimensional handwriting application, height normalisations, tolerance and reference bank are project-specific choices.', '',
'The tolerance is 0.10 of the chosen 48-pixel height. Its physical meaning differs between the two methods. A high F means nearby strokes at this tolerance. It is not exact pixel overlap, transcription accuracy, a correct-character fraction, writer-identification accuracy or stylistic correlation. Neither method is independently calibrated to human judgements of acceptable reproduction. The results alone do not prove the proposal specifications.', '',
'Strict IoU is foreground intersection area divided by foreground union area, exported separately. It is not interchangeable with F. No background pixels receive matching credit. There is no independent width scaling, slant correction, deformable registration or change of distance tolerance.', '',
'## Corpus and reference selection', '',
'The physical population comprises every eligible token in the 44 recorded block C source-passage lines. It includes all six IAM writers and all five personal writers with physical reproduction pages. It does not include the generic block A paragraph or block B pangram, which are different test populations and often lack the same text in a writer\'s genuine pages. Exact case-sensitive tokens are retained, including punctuation. A missing comma, different capitalisation or a missing cached source crop can therefore leave a word unassessed.', '',
f"The source search used {c['source_lines']} label-checked cached original lines, produced {c['reference_candidate_count']} candidate occurrences and selected {len(c['references'])} reference entries before similarity scoring. Selection used page diversity and source order, with up to eight variants per writer and token. Canonical page, line and word index deduplicate the previously reviewed originals against their cached counterparts. Twenty-two reference crops were rejected after visual inspection for a wrong or incomplete word or foreign neighbouring ink, leaving {len(accepted['references'])} accepted references. All {len(accepted['physical'])} physical proposals were retained, including faint strokes and poor joins. The frozen quality-review record predates scoring.", '',
'For each physical word, the highest F in its fixed same-writer, same-token bank is selected independently for each method. Thus the two methods can choose different original occurrences. These are optimistic nearest-reference reconstruction fits. Original pages may also have supplied the synthesis glyph profiles. The experiment is not a held-out test of new reference pages or future writing.', '',
'The earlier quick pilot used fewer physical words and a different reference population, including duplicate representations of some source occurrences. The present full test uses canonical occurrence deduplication and its own pre-score review. Preserve the earlier studies as separate experiments rather than replacing their recorded numbers.', '',
'## Unassessed words', '',
'These are not counted as geometric failures or silently replaced with references from another writer. They define the coverage limitation of the reported means.', '',
'| Author | No accepted exact same-writer reference |',
'|---|---|']
for w in W:
    missing=[r['token'] for r in x['unsupported'] if r['writer']==w]
    notes.append('| '+names[w]+' | '+(', '.join(missing) if missing else 'None')+' |')
notes += ['',
'unsupported_words.csv retains the physical IDs and locations of these cases. candidates.json records additional single-character/non-word exclusions and cache-label disagreements. An unassessed word is not evidence of a synthesis failure. Conversely, omitting it means the reported average does not describe the entire passage.', '',
'## Interpretation', '',
'Whole-word-height normalisation improves the writer-balanced mean by 25.15 percentage points. This shows that the geometric assessment is sensitive to the height definition and associated registration scale. It does not establish that the reproduction itself improved. Relative word width, individual letter spacing, joins and letterform differences remain visible in the overlays. This experiment does not isolate how much error comes from source cropping, synthesis, image capture or gantry motion.', '',
'Writer 150 gives the strongest mean fit, with ten of fourteen assessed words reaching F = 0.85 under word-height normalisation. Yeukita and Dylan have means of 83.36% and 81.13%, respectively. Thiya remains the weakest at 55.18%; none of the eight assessed cursive words reaches 0.85. These observations support successful whole-word reconstruction in some styles and cases, but do not justify an 85% claim across all writers.', '',
'The earlier 89.29% result measures individually isolated characters for five writers. It should remain separate. Whole-word scores retain accumulated spacing and join differences and use a different sample population. A lower word score does not invalidate the earlier calculation, and the earlier character mean cannot substitute for these full-word findings.', '',
'## Visual evidence and validation', '',
'figures/all_author_summary.png, PDF and SVG compare the complete per-author means. Each comparison_<writer> PNG/PDF shows the same three physical words under both methods. Examples are selected after scoring as strongest, middle-ranked and weakest by the word-height score. The displayed original photograph crop is the word-height winner. The line-height overlay can use a different original, whose ID is recorded in word_scores.csv. All-word sheets display both original selections and the physical crop for every scored word. Original contact sheets and all 44 physical line snapshots remain available for audit.', '',
f"All {x['validation']['selected_pairs_checked_both_methods']} selected comparisons, one for each word and method, exactly match the original full raster scorer and an independent cKDTree nearest-neighbour calculation on registered skeletons. Maximum F difference is zero. No recognition, training or PyTorch module was loaded. This validates numerical implementation, not the scientific equivalence of F and the proposal's accuracy requirements.", '',
'## Reproduction of the analysis', '',
'Install requirements.txt and run measure.py, then make_figures.py. They resolve saved masks and the frozen image toolkit relative to themselves and require no model weights, original dataset cache or hardware. prepare_inputs.py is extraction provenance and requires the original repository, its earlier reviewed boundaries and source caches. freeze_quality_review.py records the visual review decisions. finalise.py creates these notes and packages the evidence. environment.json records software versions and ARCHIVE_MANIFEST.json hashes every packaged file.', '',
'## Possible concise report wording', '',
'A direct geometric comparison was performed on 141 reproduced words across all eleven writers, using fixed banks of genuine same-writer reference words. The writer-balanced mean centreline F-score was 49.63% after line-height normalisation and 74.78% after normalising each word by its total height. Writer 150 achieved a mean of 89.88% with word-height normalisation. The improvement under the second normalisation shows the sensitivity of whole-word overlap to the chosen height reference. The remaining differences include word proportions, letter spacing and joins. Twenty-five eligible words had no accepted exact reference and were left unassessed. These measurements provide supporting evidence of word-shape reproduction on the assessed subset, rather than a measure of character correctness or stylistic correlation.', '',
'No LaTeX, application code, checkpoint, deployment or hardware state was changed. Existing studies and their delivered packages are preserved.']
(B/'FINDINGS.md').write_text('\n'.join(notes)+'\n',encoding='utf8')

(B/'README.md').write_text('''# All-author whole-word geometric evidence

Read FINDINGS.md first. Eleven writers, 141 assessed words out of 166 eligible
source-passage words, both normalisations applied to the same physical cases.
Writer-balanced mean F is 49.63% for line x-height and 74.78% for whole-word height.
These are optimistic nearest-reference geometric fits, not character accuracy,
writer accuracy, exact IoU, a count of correct words or stylistic correlation.

Key files

* per_author_summary.csv contains means, medians, coverage, IoU and threshold counts.
* word_scores.csv contains every selected result for both methods.
* all_reference_comparisons.csv contains every scored reference pairing.
* unsupported_words.csv contains all 25 unassessed physical words.
* results.json includes all the above and numerical validation.
* figures/ contains the overview, per-writer comparisons and all-word audit sheets.
* review/ contains the pre-score visual contact sheets and selected full-line context.
* inputs/ contains source grayscale crops, native masks and normalised masks.
* line_snapshots/ contains the 44 physical source-text lines.

Regenerate numerical results and plots using the saved arrays.

```
python measure.py
python make_figures.py
```

Install requirements.txt first. source/segment.py freezes the project's image
toolkit. No original cache, classifier, training, model weights or gantry is needed
for these two commands. prepare_inputs.py requires the original repository and
dataset caches. Quality decisions are recorded in freeze_quality_review.py and
quality_review.json and were fixed before measurement. PROTOCOL.md preserves the
experimental choices. ARCHIVE_MANIFEST.json hashes the package files.

This experiment does not edit the report. Earlier character-level and cursive-pilot
studies remain separate and unchanged.
''',encoding='utf8')
(B/'requirements.txt').write_text('numpy\nscipy\npillow\nopencv-python\nmatplotlib\n')
environment={'python':sys.version,'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','pillow','opencv-python','matplotlib']},'hardware_scope':'Offline Windows analysis of existing physical photographs, not ODROID timing','no_report_or_application_changes':True}
(B/'environment.json').write_text(json.dumps(environment,indent=2))
# Include the context-review helper as optional repository-only extraction provenance.
helper=ROOT/'tmp/all_author_context_review.py'
if helper.exists():shutil.copy2(helper,B/'review_source_context.py')
OUT=ROOT/'output';OUT.mkdir(exist_ok=True)
shutil.copy2(B/'FINDINGS.md',OUT/'Handwriting_All_Authors_Both_Methods_2026-11-02.md')
for p in B.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
files=[p for p in B.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ARCHIVE_MANIFEST.json']
manifest={p.relative_to(B).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)}
(B/'ARCHIVE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
archive=OUT/'Handwriting_All_Authors_Both_Methods_2026-11-02.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files+[B/'ARCHIVE_MANIFEST.json']:z.write(p,p.relative_to(B))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert all(hashlib.sha256(z.read(p)).hexdigest()==v['sha256'] for p,v in manifest.items())
print('Evidence package saved and hashes verified',archive.stat().st_size,'bytes',len(files)+1,'files',flush=True)
