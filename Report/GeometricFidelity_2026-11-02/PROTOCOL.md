# Matched physical handwriting geometry test

Protocol fixed before calculating similarity scores, 2 November 2026.

The primary material is the source-text block on the five personal writers' physical
pages, paired with its original handwritten source sentences. All five writers are
reported. No recognition or writer classifier is loaded or used. Existing line
crops and human-supplied transcripts locate the text. Cached original line labels
must match the current source label files. All inputs are hashed.

1. Propose word boundaries using horizontal gaps in foreground ink. Visually check
   the word assignments before calculating similarity. Corrections are geometric
   annotations, recorded separately, not selected using overlap scores.
2. Match exact, case-sensitive transcript tokens monotonically, not by image score.
   Record unmatched words and source numbering/bullets. No best-match search over
   repeated original letters or stored glyph variants is permitted.
3. Normalise line x-height to 48 pixels isotropically. Do not stretch individual word
   widths, correct slant independently or apply elastic/nonrigid shape warping.
4. Centre corresponding word masks and allow translation within +/- 0.10 x-height,
   using minimum symmetric skeleton distance to choose the offset. Apply the same
   alignment to all metrics. Report original/reproduced physical pen masks as well
   as centreline comparisons.
5. Report strict foreground IoU and Dice, fixed-width skeleton IoU (stroke diameter
   approximately 0.17 x-height), and symmetric skeleton precision/recall/F1 within
   0.05, 0.10, 0.15 and 0.20 x-height. The 0.10 tolerance is the primary distance
   result. Larger tolerances are a sensitivity curve, not a replacement chosen to
   reach a target. Blank background does not contribute to any success score.
6. Also report adaptive stroke-width IoU, an adaptation of Chen et al. (2022),
   section 3.1, p.5. Skeletonised physical output replaces their predicted
   trajectory, so this is explicitly an image-to-image adaptation. The original
   mask stays fixed while the recovered skeleton is dilated, taking the maximum
   foreground IoU over widths. It does not change the skeleton's geometry.
7. For per-character analysis, use only words where both masks have exactly the
   expected number of geometrically separable character groups. Merge overlapping
   x-ranges (e.g. a dot with its stem), then pair left-to-right. Visually inspect
   the resulting pairs. Report this as a separable-character subset, with coverage
   and exclusions. Connected characters are not assigned invented boundaries.
8. Compare wrong-character controls and available same-character other-writer
   controls with identical processing. This tests whether a high score is specific
   enough to support the intended interpretation.
9. Report all per-writer means, sample counts, the pooled and equal-writer means,
   and word-group bootstrap intervals. A geometric F1 or IoU percentage is not
   character transcription accuracy, stylistic correlation or human judgement.
   An 85-percent finding applies only to its named metric and sample population.

Sources
Chen et al., Complex Handwriting Trajectory Recovery, ACCV 2022, section 3.1, p.5.
https://arxiv.org/pdf/2210.15879
Huttenlocher et al., Comparing Images Using the Hausdorff Distance, IEEE TPAMI
15(9), 850-863, 1993. https://people.eecs.berkeley.edu/~malik/cs294/Huttenlocher93.pdf
