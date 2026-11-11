# All-author whole-word geometric evidence

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
