# Classifier-free comparison of original and robot handwriting

Measured on 2 November 2026 from saved images on the Windows development host.
The handwriting was already physically produced by the gantry. This work involved
no training, neural inference, synthesis, deployment or gantry motion.

## Finding

Owen exceeds the 85-percent reference on the declared centreline-agreement test.
The mean per-character F-score is **87.09%** on 25 geometrically separable character
pairs from seven matched words. Pooling all their centreline points gives **85.28%**.
These are observed geometric scores, not percentages of correctly transcribed
characters, strict IoU, Pearson correlation or a universal 85-percent guarantee.

Yeukita's corresponding mean is **83.26%**, close to the reference. The complete
separable-character set averages **80.57%** over 65 characters.

## Character results

| Writer | Matched words inspected | Separable characters | Mean character F-score | Pooled point F-score | Strict foreground IoU | Fixed-width IoU |
|---|---:|---:|---:|---:|---:|---:|
| Owen | 21 | 25 | 87.09% | 85.28% | 39.59% | 56.99% |
| Yeukita | 16 | 19 | 83.26% | 81.64% | 33.61% | 51.72% |
| Dylan | 11 | 18 | 73.97% | 73.16% | 29.69% | 44.03% |
| Thiya | 8 | 0 | Not measured | Not measured | Not measured | Not measured |
| Abhinav | 18 | 3 | 48.78% | 47.83% | 17.85% | 26.63% |
| Pooled | 74 | 65 | 80.57% | 78.20% | 34.10% | 50.46% |

Abhinav's three-character subset is too small to represent that writer's general
performance. No Thiya character was forced out of joined writing. Both writers
remain in the full word comparison and their lack of character coverage is explicit.

The separable characters cover 25/109 alphabetic characters in Owen's matched words,
19/75 for Yeukita, 18/49 for Dylan, 3/90 for Abhinav and 0/51 for Thiya. This is an
easy-to-separate subset, not a random sample of all produced handwriting.

## Method and calculation

Transcripts identify corresponding source and physically reproduced words. Matching
is exact and case-sensitive and follows transcript order, with no search for the
best-looking source variant. Word boundaries were visually reviewed before scores
were calculated. The small geometric annotation corrections and one unavailable
source word are recorded in word_annotations.json and pair_manifest.json.

Each line's estimated x-height is scaled isotropically to 48 pixels. A character is
accepted only when both words contain exactly the expected number of horizontally
separated ink groups. All 65 resulting character pairs were visually inspected.
The masks are centred, and translation is limited to approximately 0.10 x-height.
There is no independent width stretching, slant correction or elastic deformation.

Original and reproduced centreline sets are G and P. At distance tolerance tau,
precision is the fraction of P within tau of G, and recall is the fraction of G
within tau of P. F = 2PR/(P+R). This adapts a standard nearest-point geometric
F-score to two-dimensional handwriting. Here tau = 0.10 estimated x-height,
or 4.8 pixels after normalisation. It is a declared engineering tolerance, not a
published universal handwriting pass threshold. The full 0.05 to 0.20 sensitivity
curve is supplied, with 0.10 retained as the primary setting.

For Owen, pooled point precision is 83.5673% and recall is 87.0721%. Their harmonic
mean is 85.2837%. Taking each character's F-score first and averaging the 25 values
gives 87.0859%. The two aggregation rules answer slightly different questions.

Pixel IoU counts only foreground ink. Background agreement is never included.
Fixed-width IoU dilates both centrelines to the same nominal width. Adaptive IoU
widens the physical centreline against the fixed original ink mask, following the
width-adjustment idea of Chen et al. The adaptation compares photographed images,
not their original predicted-trajectory experiment.

Sources for the metric definitions

- [Tanks and Temples official evaluation, Appendix B](https://tanksandtemples.org/tutorial/).
  Its distance-threshold precision, recall and F-score are used here in 2D.
- [Chen et al., Complex Handwriting Trajectory Recovery, p. 5](https://arxiv.org/pdf/2210.15879).
  This supports the adaptive stroke-width IoU comparison.

## Controls and uncertainty

Owen's wrong-character control averages 52.24%, compared with 87.09% for the correct
pairs. For the 17 characters having another-writer reference of the same letter,
the matched correct-writer mean is 88.04% and the other-writer control is 74.88%.
The control uses one fixed original occurrence per available other writer, not the
most dissimilar example. High agreement is therefore stronger than an arbitrary
letter match, although common Latin letter shapes still agree between writers.

An independent cKDTree implementation reproduces all 65 primary F-scores exactly,
compared with the Euclidean distance-transform implementation. Identical masks
score 100%. The scorer confirms no recognition, training or PyTorch module was loaded.

Resampling whole source-word groups 10000 times gives a descriptive 95% range of
82.72% to 91.83% for Owen's mean. This does not establish that the population mean
is at least 85%. The observed point estimates exceed 85% in the sampled Owen case. Seventeen of the 25 individual Owen characters have F-scores at least 85%. Thus a mean score of 87.09% must not be described as 87.09% of characters passing a threshold.

## Whole-word results and limitations

The same-text whole-word comparison retains 74 pairs. At the same 0.10 x-height
tolerance, mean word F-scores are 51.83% for Owen, 49.34% for Yeukita, 40.61% for Dylan,
10.54% for Thiya and 36.78% for Abhinav. These scores retain spacing, width and
proportion differences removed by separate character positioning. They are also
sensitive to the line-based body-height estimator, particularly on cursive writing.
The positive character result must not replace or conceal these word-level results.

The source sentences were part of the available handwriting material. This is a
same-source reconstruction comparison, not proof of unseen-writer generalisation.
Character matching does not test word spacing, baseline layout or temporal stroke
order. It also does not establish that a reader can identify 85% of all characters.

## Relationship to the proposal

The proposal says the demonstration will count characters that can visibly be seen
to match the intended text in the chosen writer's style. A mean geometric F-score
is not that count. The current Owen result is useful independent supporting evidence,
but it cannot silently replace either character transcription accuracy or an 0.85
style-correlation coefficient in the qualification summary.

The blind_readback folder supplies anonymised sheets for all five complete physical
source-text blocks, a blank response CSV and a separate answer key. Give readers only
the P01 to P05 images and responses.csv, without the answer key. Preserve case and
punctuation, and put ? for unreadable characters. The scorer requires every line,
so a favourable partial response cannot masquerade as the complete test.

Run `python score_blind_readback.py path/to/completed_responses.csv` once actual
responses exist. It reports edit-based character accuracy per writer. No human
response or human accuracy has been invented in this study. A separate blinded
style judgement would still be needed for a human style claim.

## Suggested report wording

A separate classifier-free comparison was performed on matching original and
robot-drawn text. After isotropic x-height normalisation and bounded translation,
Owen's 25 geometrically separable characters achieved a mean centreline F-score
of 87.09% at a distance tolerance of 0.10 x-height. The pooled centreline-point
F-score was 85.28%. The matched other-writer control was lower, supporting retention
of writer-specific shape in this sample. This is a character-shape result on a
restricted subset and does not establish the same percentage for complete-word
layout, transcription accuracy or the full writer population.

## Files and reproducibility

- PROTOCOL.md records the choices fixed before similarity measurement.
- input_manifest.json records cached/source label correspondence and input hashes.
- word_annotations.json records geometric annotation corrections.
- pair_manifest.json identifies every included and unmatched token.
- character_scores.csv, word_scores.csv and controls_scores.csv retain all cases.
- per_writer_summary.csv and statistics.json contain the aggregate results.
- masks contains the exact normalised binary inputs used for scoring.
- figures contains the summary, tolerance curve and all 25 Owen overlays.
- blind_readback contains the unscored human transcription packet.

From the project root, run prepare_inputs.py, annotate_words.py, prepare_pairs.py,
measure_geometry.py, verify_statistics.py and make_evidence.py in that order.
Initial preparation needs the existing personal-line cache and preserved physical
line crops. Measurement and verification use the saved masks and image-toolkit module.
The analysis is separate from the report. No report LaTeX or deployed code was edited.
