# Five-writer direct geometric evidence

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
