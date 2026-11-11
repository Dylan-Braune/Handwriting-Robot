from pathlib import Path
import json, sys, platform, hashlib, zipfile, shutil, importlib.metadata
import numpy as np

B = Path(__file__).resolve().parent
ROOT = B.parents[1]
x = json.loads((B / 'results.json').read_text())
S = x['summary']
W = ['dylan', 'yeukita', 'owen', 'abhinav', '150']
names = dict(zip(W, ['Dylan', 'Yeukita', 'Owen', 'Abhinav', 'IAM 150']))
selected = json.loads((B / 'selected_separability.json').read_text())['selected']

lines = ['# Five-writer geometric comparison, 2 November 2026', '',
'The equally weighted mean across the five writers is **89.29% nearest-reference centreline F-score**, using 193 separable characters cropped from photographs of actual robot-written source-text blocks. Each physical character is compared with its closest variant in a fixed, same-writer, same-letter bank of up to eight genuine examples. No text recogniser or writer classifier is used.', '',
'This is an optimistic reference-bank reconstruction test. It supports a bounded statement about glyph shape on this subset. It does not establish 89.29% transcription accuracy, foreground IoU, stylistic correlation or a percentage of characters judged correct. The proposal requirements cannot be declared passed solely from this average.', '',
'## Full assessed sample', '',
'| Writer | Characters | Letter types | Mean F at 0.10 x-height | Characters individually at F >= 0.85 | Strict foreground IoU |',
'|---|---:|---:|---:|---:|---:|']
for w in W:
    z = S[w]
    rr = [r for r in x['results'] if r['writer'] == w]
    lines.append(f"| {names[w]} | {z['scored']} | {z['letters']} | {100*z['mean_f1']:.2f}% | {sum(r['f1'] >= .85 for r in rr)}/{len(rr)} | {100*z['mean_iou']:.2f}% |")
lines += ['',
'The writer-balanced mean is 89.29%. The character-weighted mean is 91.90%. The first gives each writer equal weight despite different sample sizes. Four writer means exceed the 85% reference level. Abhinav remains at 75.10% across his 14 assessed characters. His three assessed lowercase c characters average 86.77%, which is letter-specific evidence and does not replace his full mean.', '',
'## Method and selection', '',
'1. Use existing photographs of robot output and genuine original handwriting. Keep the physical population to the source-text block already reproduced for each writer. IAM 150 uses the passage beginning “This is not a filmed play”. Case-sensitive transcripts identify characters.',
'2. Propose disconnected spans. A word supplies characters when its span count agrees with its transcript. In a partly joined word, propose its outer spans and visually check that each contains one complete letter. Review original and physical crops before scoring. Exclude joined glyphs, fragments and isolated noise. Retain genuine unusual letterforms and damaged robot strokes.',
'3. Select original banks by page diversity and source order, with at most eight variants per writer and letter, before scoring. Do not prune banks by score or replace quality rejections with stronger matches.',
'4. Estimate the line body height and apply one isotropic scale to make it 48 pixels. Centre tight character crops. Search translations of at most five pixels in either direction, choosing the smallest symmetric mean nearest-centreline distance. Do not stretch width, change slant or apply deformable registration.',
'5. Thin the masks. Precision is the fraction of physical skeleton points within 4.8 pixels of the original skeleton. Recall reverses the two sets. Calculate F = 2PR/(P + R), using the same 0.10 x-height tolerance as the earlier study. Select the largest F within the fixed genuine bank for each physical character.',
'6. Average all assessed characters within each writer, then the five writer means. Keep low scores. Record the selected-reference ID, reference count, alignment, point counts, strict IoU and sensitivity values for every pair.', '',
'The thresholded point-set precision, recall and harmonic mean follow the [Tanks and Temples evaluation tutorial](https://tanksandtemples.org/tutorial/), Appendix B. Applying this to two-dimensional handwriting skeletons, choosing the tolerance and constructing reference banks are project-specific choices. The 0.10 x-height tolerance has not been independently calibrated against human acceptability.', '',
'Strict IoU is intersection area divided by union area. Chen et al., [Complex Handwriting Trajectory Recovery](https://arxiv.org/pdf/2210.15879), p. 5, discuss stroke-width sensitivity and introduce adaptive IoU. The exported adaptive-overlap diagnostic uses a finite Euclidean radius search and is not an exact reproduction of their iterative kernel procedure. The headline result is F, not adaptive IoU.', '',
'A score of 100% means every skeleton point is within the tolerance in both directions. It does not mean identical handwriting or identical foreground pixels.', '',
'## Coverage', '',
'The source search examined 640 cached genuine lines and proposed 2,964 reference glyphs. Page-diverse, source-ordered bank selection retained 792 proposals before visual checking. Review rejected 84 ambiguous reference crops and 18 of 212 physical proposals. There are 708 accepted references and 194 accepted physical characters. One Yeukita q has no accepted genuine reference, leaving 193 scored characters. The unassessed q is preserved in results.json. No score-based exclusions were made.', '',
'The scored characters come from 71 physical word groups. They are a separable-character subset of these existing source-text blocks, not all characters on the pages. Connected handwriting is underrepresented. Reference pages may also have supplied glyphs to synthesis profiles. This is reconstruction evidence and does not assess unseen reference pages. Searching several genuine variants is more favourable than using one fixed handwritten occurrence.', '',
'## Same physical characters as the earlier test', '',
'All 65 earlier physical masks are unchanged, verified by exact array equality. Only their reference comparison changes.', '',
'| Writer | Same characters | Fixed genuine occurrence | Closest bank variant |',
'|---|---:|---:|---:|']
baseline = json.loads((B / 'matched_baseline.json').read_text())
for w, z in baseline['per_writer'].items():
    lines.append(f"| {names[w]} | {z['n']} | {100*z['fixed_occurrence_mean']:.2f}% | {100*z['nearest_reference_mean']:.2f}% |")
lines += ['',
'On those same 65 characters, the character-weighted mean changes from 80.57% to 91.11%. This isolates the reference-bank effect from the expanded physical sample and added IAM writer. It does not imply that the robot output changed. The earlier results remain preserved.', '',
'## Selected strongest visual matches', '',
'These pairs are deliberately selected after scoring. They illustrate achievable shape agreement and are not the sample used to calculate writer means.', '',
'| Writer | Letter | Best-pair F | Physical word |',
'|---|---|---:|---|']
for w in W:
    z = S[w]['best_pair']
    r = next(r for r in x['results'] if r['id'] == z['id'])
    lines.append(f"| {names[w]} | {z['char']} | {100*z['f1']:.2f}% | {r['word']} |")
lines += ['',
'See figures/selected_best_five_writers.png or its PDF. Each comparison_<writer> additionally shows a strongest, middle-ranked and weakest case. The all_pairs_<writer> sheets retain every assessed character, with the genuine reference above the robot crop, including poor matches.', '',
'## Examples that distinguish styles', '',
'Compare same-letter banks from all four other writers. Match own and other bank sizes to their minimum size separately for each rival. For an exploratory gallery, require own F >= 0.85 in every comparison, then select the character with the largest minimum own-minus-rival margin. The table shows the least favourable rival comparison for each selected character. All four comparisons are saved in selected_separability.json.', '',
'| Writer | Letter | Own F | Rival writer | Rival F | Minimum advantage |',
'|---|---|---:|---|---:|---:|']
for a in selected:
    r = a['physical']
    q = a['least_favourable_matched_control']
    lines.append(f"| {names[r['writer']]} | {r['char']} | {100*q['own_f1']:.2f}% | {names[q['other_writer']]} | {100*q['other_f1']:.2f}% | {100*q['difference']:.2f} pp |")
lines += ['',
'See figures/selected_distinguishable_letters.png or its PDF. These are post-selected descriptive examples, not a held-out writer-identification test. Simple letters can agree strongly across writers. Full controls, including unfavourable comparisons, are retained in controls.csv.', '',
'Across full control populations, own-writer means exceed same-letter other-writer means for every writer. These size-matched means are Dylan 81.61% versus 77.14%, Yeukita 93.21% versus 82.54%, Owen 94.14% versus 82.77%, Abhinav 75.10% versus 65.61%, and IAM 150 95.49% versus 81.53%. These use matched bank sizes and available control characters, so their own means differ from the headline table.', '',
'## Sensitivity and verification', '',
'References remain fixed for this tolerance comparison.', '',
'| Writer | F at 0.05 x-height | F at 0.10 x-height | F at 0.15 x-height |',
'|---|---:|---:|---:|']
for w in W:
    rr = [r for r in x['results'] if r['writer'] == w]
    lines.append(f"| {names[w]} | {100*np.mean([r['f1_0.05'] for r in rr]):.2f}% | {100*S[w]['mean_f1']:.2f}% | {100*np.mean([r['f1_0.15'] for r in rr]):.2f}% |")
lines += ['',
'The cached scorer agrees exactly with the original full raster scorer on 28 diverse reference pairs and all 193 selected winners. An independent cKDTree nearest-neighbour calculation also reproduces every selected F-score exactly. This validates the numerical implementation, not a specification pass criterion. No neural modules were loaded.', '',
'Resampling physical word groups 10,000 times with seed 23539195 gives a descriptive 95% range of 87.45% to 91.06% for the five-writer mean. This is conditional on the five writers, chosen crops and fixed banks. It excludes uncertainty from bank selection, segmentation decisions and new-page variation, and is not an independent generalisation interval.', '',
'## Suggested report wording', '',
'Photographs of the reproduced source-text passages were compared directly with original handwriting using a classifier-free geometric test. A fixed bank of up to eight genuine variants was formed for each writer and character. After line-height normalisation and bounded translation, each robot-written character was compared with its closest same-writer variant. The mean centreline F-score at a tolerance of 0.10 x-height was 89.29% when the five writers were weighted equally, over 193 separable characters. Four writer means exceeded 85%. Abhinav averaged 75.10%, although his three assessed c characters averaged 86.77%. These results support the reproduction of individual glyph shapes on the assessed subset. The nearest-reference selection and incomplete coverage prevent this score from establishing full-page character accuracy or the stylistic-correlation target.', '',
'No report source, trained checkpoint or application code was changed. Measurements were made offline on the available Windows computer.']
(B / 'FINDINGS.md').write_text('\n'.join(lines) + '\n', encoding='utf8')

(B / 'README.md').write_text('''# Five-writer direct geometric evidence

Start with FINDINGS.md. The main result is 89.29% equally writer-weighted mean nearest-reference centreline F-score over 193 separable physical characters at 0.10 x-height. It is not IoU, transcription accuracy, a character pass fraction or style correlation. The bank search is intentionally optimistic. Best-case galleries are selected after scoring. Abhinav's full mean is 75.10%.

Per-writer comparisons are in figures/. The PDF and PNG versions use the same data. per_writer_summary.csv gives all five means. per_character.csv records every selected pair. all_reference_comparisons.csv retains all evaluated bank comparisons. controls.csv records size-matched other-writer tests.

Install requirements.txt and regenerate the numerical results and figures from the saved masks.

```
python measure_reference_bank.py
python make_comparisons.py
python check_baseline_and_separation.py
python validate_distances.py
```

These commands resolve inputs relative to themselves. No dataset cache, checkpoint, classifier, training or robot is needed. source/segment.py freezes the project's image toolkit. glyphs/ contains native masks, normalised masks and original grayscale crops. line_snapshots/ preserves genuine line context from the source search. baseline_source/ saves the earlier 65 physical masks and their records. Historical metadata includes repository paths, but numerical regeneration uses saved arrays.

build_candidates.py records extraction and requires the original repository, earlier study and raw caches. save_quality_review.py records visual identity/separability decisions made before scoring. It is not an automatic quality classifier. candidates.json, quality_review.json and accepted_candidates.json retain the complete selection record.

PROTOCOL.md records the scoring rule. statistics.json contains conditional resampling. independent_validation.json records the independent distance check. selected_separability.json preserves all four rival comparisons for each exploratory distinguishing example. ARCHIVE_MANIFEST.json records SHA-256 for packaged files and environment.json records software versions.

No report LaTeX was edited. Comparisons use real source and robot handwriting crops.
''', encoding='utf8')

env = {'python': sys.version, 'platform': platform.platform(),
       'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'pillow', 'matplotlib', 'opencv-python']},
       'hardware_scope': 'Offline Windows analysis, not an ODROID timing measurement',
       'report_modified': False}
(B / 'environment.json').write_text(json.dumps(env, indent=2))
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)
shutil.copy2(B / 'FINDINGS.md', OUT / 'Handwriting_Five_Writer_Geometry_Findings_2026-11-02.md')
files = [p for p in B.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'ARCHIVE_MANIFEST.json']
manifest = {p.relative_to(B).as_posix(): {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)}
(B / 'ARCHIVE_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
archive = OUT / 'Handwriting_Five_Writer_Geometry_2026-11-02.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in files + [B / 'ARCHIVE_MANIFEST.json']:
        z.write(p, p.relative_to(B))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print('Saved', archive, 'bytes', archive.stat().st_size, 'files', len(files) + 1)
