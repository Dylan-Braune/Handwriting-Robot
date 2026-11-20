# Matched natural-writing baseline

Read FINDINGS.md for the result and its scope. With cross-page matched reference
banks and whole-word-height normalisation, genuine writing gives mean F 86.79%,
robot writing 76.47%, and the relative robot/natural ratio is 88.11%. Averaging all
pair scores without nearest-reference selection gives 79.45%, 69.00% and 86.85%.
Both ratios concern 83 repeated words across 11 writers, not every passage word.
They are not character accuracy, Pearson correlation or a specification pass.

Use per_writer_summary.csv and word_baselines.csv for tables. All pair scores,
matched folds, exclusions, statistics, selected figure IDs and source crops are
preserved. figures/ contains both average comparisons, all-word scatter and an
illustrative genuine/robot comparison for every writer.

Regenerate from the saved arrays after installing requirements.txt.

```
python measure.py
python make_evidence.py
python finalise.py
```

Inputs and the frozen image toolkit are included. No weights, model inference,
training or gantry are needed. If accepted_candidates.json is absent, measure.py
imports the existing all-author evidence from its repository path. The packaged
copy already contains that file and all required arrays. Keep PROTOCOL.md and
the exact same frozen inputs when reproducing these results.

No report or application changes were made. Earlier studies remain unchanged.
