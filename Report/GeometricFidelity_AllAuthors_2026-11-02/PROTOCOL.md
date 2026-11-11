# All-author whole-word geometry protocol

Compare the two normalisations from the cursive pilot on all eleven writers with
physical reproduction pages (150, 152, 153, 384, 552, 588, Abhinav, Dylan, Owen,
Thiya and Yeukita). Use every source-passage (block C) token with at least two
characters and at least one letter, across all 44 recorded physical lines. Record
single-letter and non-word exclusions and every unsupported word. Do not replace
unavailable same-writer original words with another writer's handwriting.

Method 1 uses the existing 48-pixel line x-height, isotropic scale and bounded
translation of +/-5 pixels. Method 2 tightly crops that mask and isotropically
rescales total word height to 48 pixels. Both use nearest-centreline precision,
recall and F within 4.8 pixels after translation selected by symmetric mean
nearest distance. The same pixel threshold refers to different height definitions.
Do not describe either whole-word result as transcription or writer accuracy.

For each writer and exact case-sensitive transcript token, select up to eight
original occurrences by page diversity and source order, before scores. Preserve
punctuation. Reuse previously reviewed source/physical boundaries where available.
Deduplicate occurrences by canonical page, line and word index. Check selected
banks and physical words visually before scoring. Reject only identity errors,
foreign ink, missing word fragments or unusable boundaries, not poor geometric
matches. Keep weak real reproductions. Freeze all quality decisions before scoring.

For each physical word, choose its highest F reference independently under each
normalisation. Score both methods on exactly the same accepted physical population.
Retain every individual reference comparison, low score, bank size and missing case.
Report full per-writer means, medians, sample coverage and fractions above F=0.85.
Give writer-balanced and word-weighted aggregates. These are optimistic same-source
reference-bank fits. No tolerance tuning, width stretching, slant correction,
deformable fitting or new model inference, training or gantry operation.

Save per-writer best, middle and worst examples for both methods plus all-word audit
sheets. Numerically validate selected pairs with the original raster implementation
and independent point-neighbour distances. Keep all prior studies and report sources
unchanged. This is offline analysis, not an ODROID runtime measurement.
