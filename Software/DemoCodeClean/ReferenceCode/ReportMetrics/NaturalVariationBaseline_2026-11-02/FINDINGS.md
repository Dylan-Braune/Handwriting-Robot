# Natural handwriting variation and robot reproduction

2 November 2026. A new matched-baseline analysis of the existing data. Earlier studies and the report are unchanged.

## Main finding

The robot reaches **88.11% of the measured same-writer natural baseline** under the existing whole-word-height geometric method. Genuine words average 86.79% centreline F-score against other-page references. The corresponding robot words average 76.47% against those exact reference banks. These are equally weighted averages across all eleven writers.

A second calculation uses every reference pairing rather than selecting the best. Genuine-to-genuine agreement averages 79.45%, while robot-to-genuine agreement averages 69.00%. The robot therefore reaches **86.85% of that ordinary average baseline**. Both results support describing reproduction relative to measured natural variation. Neither ratio is a character accuracy or a Pearson correlation coefficient.

## Matched results for each author

Cross-page, whole-word-height normalisation. Each genuine probe and robot probe have the same reference bank in every comparison. Scores are averaged over probes and physical words before calculating the per-writer ratio.

| Author | Physical words | Genuine-to-genuine F | Robot-to-genuine F | Robot / natural baseline |
|---|---:|---:|---:|---:|
| 150 | 8 | 97.32% | 94.65% | 97.26% |
| 152 | 8 | 96.61% | 80.79% | 83.63% |
| 153 | 6 | 94.40% | 78.26% | 82.90% |
| 384 | 6 | 86.60% | 77.07% | 88.99% |
| 552 | 8 | 87.34% | 72.96% | 83.54% |
| 588 | 5 | 83.52% | 76.30% | 91.36% |
| Abhinav | 12 | 82.90% | 68.31% | 82.40% |
| Dylan | 10 | 80.91% | 75.92% | 93.83% |
| Owen | 5 | 85.30% | 83.17% | 97.50% |
| Thiya | 4 | 73.69% | 51.83% | 70.34% |
| Yeukita | 11 | 86.13% | 81.91% | 95.09% |

| Aggregation rule | Genuine-to-genuine F | Robot-to-genuine F | Robot / natural baseline |
|---|---:|---:|---:|
| Matched nearest-reference banks | 86.79% | 76.47% | 88.11% |
| All reference pairings, no best selection | 79.45% | 69.00% | 86.85% |

The ordinary average row is useful for the question about normal variation in a writer's own words. The nearest-reference row matches the selection principle in the previous robot experiment. Both rows use the same physical-word cohort and equal writer weighting. The aggregate ratio is the ratio of the two writer-balanced means, rather than the mean of eleven ratios.

## How the baseline was constructed

Use only the frozen, visually reviewed same-writer original reference banks from the all-author study. No new crops, tolerance settings, alignment options or score-based exclusions are introduced. Exact tokens, including case and punctuation, are required. Canonical source occurrences were already deduplicated. A further check finds no identical grayscale reference crops.

For each genuine probe occurrence of a word, remove every reference from that probe's page. This leaves a bank of the same word written by the same author on other pages. Score the genuine probe against that bank. Score the physical robot word against exactly the same bank. Both probes therefore have the same number of possible reference matches. A genuine probe is never compared with itself.

For nearest-reference agreement, take each probe's highest F in the common bank. For ordinary average agreement, average every F in that common bank. Average these matched folds over genuine probes for each physical word, then over physical words within each writer. Finally average the eleven writer means. Full pair scores, bank membership, selected originals and intermediate averages are saved.

All eleven authors can be included, with 83 physical words covering 77 distinct writer-and-token groups. These are the earlier test words with at least two genuine source pages. Four more words have repeats only within a single page and appear in the separate leave-one-occurrence-out check. The original full word study had 141 assessed robot words out of 166 eligible passage words. This repeat-baseline subset cannot replace that full coverage statement.

Only repeated words can have a measured natural-repeat baseline. The resulting subset contains more common, shorter words. Its mean token length is 3.63 characters, compared with 4.73 across the earlier 141 words. Owen contributes five words and Thiya four to the cross-page analysis. Their ratios should not be treated as complete-page estimates.

## Formula and real numerical substitution

The unchanged geometric score uses nearest-centreline precision P and recall R within a fixed distance, combined as F = 2PR/(P + R). The point-set definition follows the [Tanks and Temples evaluation tutorial](https://tanksandtemples.org/tutorial/), Appendix B. Applying it to handwriting, defining the normalisation and constructing the matched natural baseline are choices for this project.

The relative-baseline measure is defined for this experiment as Q = 100 times mean robot F divided by mean natural F. It is a descriptive ratio, not a published accuracy metric. For the writer-balanced cross-page nearest-reference result, Q = 100(0.7647033887 / 0.8679351508) = 88.11%. For the ordinary average result, Q = 100(0.6899640174 / 0.7944743192) = 86.85%.

The 100% reference means equal measured geometric agreement to this natural baseline. It is not a perfect-writing score or a theoretical ceiling. Individual ratios can exceed 100% when a robot word happens to match the available references more closely than another genuine instance does. Fifteen of the 83 physical words do so under the nearest-reference calculation, and those values are retained without capping.

Pearson correlation is a different statistic that measures linear association between two arrays, as defined in the [SciPy pearsonr documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html). The present numbers must be called geometric agreement and relative-to-baseline ratios. No new Pearson style-correlation result has been created.

## Both normalisations and both reference policies

These checks were specified before measuring the natural baseline. Both absolute scores and ratios remain visible.

| Probe exclusion | Normalisation | Words | Genuine best-bank F | Robot best-bank F | Relative best-bank score | Relative ordinary-pair score |
|---|---|---:|---:|---:|---:|---:|
| cross page | Line x-height | 83 | 67.22% | 54.49% | 81.05% | 79.68% |
| cross page | Word height | 83 | 86.79% | 76.47% | 88.11% | 86.85% |
| leave one occurrence out | Line x-height | 87 | 68.68% | 54.43% | 79.26% | 78.35% |
| leave one occurrence out | Word height | 87 | 86.94% | 76.96% | 88.52% | 87.05% |

Word-height normalisation tightly crops each word and isotropically scales its total ink height to 48 pixels. Line-height normalisation retains the earlier estimated 48-pixel body height. Both centre the masks, allow at most five pixels of translation per axis, and measure stroke proximity within 4.8 pixels. The chosen height reference changes the physical meaning of that pixel tolerance. There is no width-only stretch, slant correction or deformable warp.

The line-height method gives 81.05% of its natural nearest-reference baseline, below the word-height relative result. It is retained rather than hidden. Allowing distinct same-page occurrences gives a word-height relative result of 88.52%, close to the cross-page 88.11%. The no-best-selection cross-page result remains 86.85%.

## Uncertainty and numerical checks

Paired resampling of word-type groups within each of the eleven fixed writers gives a descriptive 95% range of 85.16% to 90.83% for the nearest-reference ratio. The ordinary average ratio has a range of 83.68% to 89.73%. These use 10,000 draws with seed 23539195 and keep all reference banks fixed. They do not include uncertainty from source selection, new pages, image segmentation or unseen writers. They do not establish a population-level 85% guarantee.

All 4480 ordered original-original pair scores and 1828 matched folds are retained across methods and exclusion policies. One genuine pair in every repeated-word bank was checked with both normalisations, giving 162 full-raster and independent cKDTree checks with maximum F difference zero. The previous all-author study independently checked all 282 selected robot pairs. No neural modules were loaded.

## What this supports in the report

A fair statement is that the robot retained approximately 87% to 88% of measured within-writer word-shape agreement on the repeated-word subset, with the two aggregation rules reported explicitly. Keep the raw scores, coverage and normalisation beside that statement. This explains why imperfect pixel or stroke overlap can still represent substantial reproduction fidelity when genuine handwriting also varies.

It does not establish 87% to 88% character correctness or fulfil the proposal's 85% stylistic-correlation requirement. It also does not remove the weaker result for Thiya. His nearest-reference ratio is 70.34%, compared with 97.50% for Owen, 95.09% for Yeukita and 93.83% for Dylan. This measured baseline gives context while retaining real differences between the reproduced styles.

Suggested wording follows.

Natural variation was measured by comparing repeated instances of the same word from each writer's original pages. Each genuine instance and its corresponding robot-written word were evaluated against the same references from other pages. Across 83 reproduced words from all eleven writers, the writer-balanced centreline F-score was 86.79% for genuine handwriting and 76.47% for the robot output. The robot therefore retained 88.11% of the measured natural baseline. Averaging all reference comparisons instead of choosing the closest match gave a relative score of 86.85%. These results support word-shape reproduction relative to natural variation on the repeated-word subset. They remain separate from character-recognition accuracy and stylistic correlation.

## Saved evidence

figures/natural_vs_robot compares the matched nearest-reference results. figures/average_pairwise_baseline shows ordinary average agreement. figures/word_scatter shows all 83 physical words. Each comparison_<writer> figure shows one illustrative matched fold, selected for closeness to that writer's two average scores. The genuine and robot probes have identical bank membership, but can choose different closest originals. illustrated_folds.json identifies the exact panels and their source IDs.

per_writer_summary.csv contains every author, method and exclusion policy. word_baselines.csv contains all per-word values. matched_folds.csv records reference membership and each fold. original_original_pairs.csv and robot_original_pairs.csv retain all pair scores. excluded_words.csv lists words without a natural repeat baseline. statistics.json records conditional resampling. Frozen masks and source grayscale crops are in inputs/.

To regenerate numerical results, run measure.py. Then run make_evidence.py and finalise.py for figures, notes and packaging. With the saved arrays present these commands need no original dataset cache, model weights, training or hardware. requirements.txt and environment.json record dependencies. ARCHIVE_MANIFEST.json hashes the package. No LaTeX, runtime pipeline, model, deployment or gantry state was changed.
