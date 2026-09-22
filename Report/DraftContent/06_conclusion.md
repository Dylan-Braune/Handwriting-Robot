# 6. Conclusion (DRAFT CONTENT — third person, past tense)

## 6.1 Summary of the work completed

A system was designed and implemented that reads scanned or photographed
handwritten pages, identifies which of ten known authors produced a
given line of handwriting, learns a per-author model of that person's
letterforms and writing statistics from a small number of real samples,
generates new handwritten text in a chosen author's style for arbitrary
input text, and converts the result into G-code that a two-axis gantry
executes with a physical pen. The recognition and identification
components were built as a CNN-BiLSTM-CTC network and a convolutional
classifier respectively, both trained from first principles; the
synthesis component was built as an exemplar-based glyph-library
generator; and the physical motion system was built around direct GPIO
stepper control with a same-process end-stop safety check.

## 6.2 Summary of observations and findings

Both learned models exceeded their accuracy targets on real handwriting:
the text recognizer reached 89.4% character accuracy and the writer-
identification classifier reached 96.7%-100% accuracy (depending on
variant) on real, held-out handwriting from the ten target authors. The
synthesis stage, after per-author tuning of its legibility/authenticity
blend parameter and a subsequent change to the synthesis selection
method itself (choosing the best of several candidate draws by a joint
measure of legibility and distinctiveness together, rather than a single
draw scored on legibility alone), reached a final mean
writer-identification accuracy of 97.5% direct and 95.8% via G-code —
comfortably exceeding the 85% target on both paths, including the G-code
path that reflects what the physical gantry would actually draw — while
mean synthesised-text character accuracy (81.6%/81.4%) remained just
below the same target overall, though the eight authors excluding the
two most heavily connected, steeply-slanted ones (both already at the
maximum setting of the tuned parameter) average approximately 86.8%,
meeting it. Three significant data/design defects were found and
corrected during the project: a page-labelling method that
systematically mis-assigned line-level ground truth for authors who
wrapped a shared source paragraph differently from a reference author
(corrected by replacing proportional apportioning with
dynamic-programming word-width alignment, which raised measured
real-handwriting text accuracy from 42.7% to 89.4% with no change to the
recognizer itself); a writer-identification classifier that relied on
ink-density cues the physical gantry cannot reproduce, corrected by
training a second, stroke-normalised classifier variant, which raised
synthesised-output writer identification from approximately 20% to 79.2%
before further tuning; and a synthesis-parameter bug in which a
per-author legibility setting was defined but never actually applied
by the code path used for evaluation, corrected as part of the tuning
work described above. Two further attempted fixes — adjusting a global
letterform-connection threshold to resolve a specific, persistent
confusion between two authors, and increasing one author's legibility
parameter beyond its tuned value — each validated against smaller,
self-constructed sentence pools but regressed the project's own official
verification script and were reverted, which is recorded as a specific
methodological finding, observed twice on two different kinds of
parameter: agreement between independently-built validation sets does
not by itself prove a change generalises.

## 6.3 Contribution

[DRAFT NOTE: this subsection specifically requires the student's own,
explicit account of what was new to them personally, what was done by the
student versus consulted from others, and the extent and nature of study
leader guidance received, per the exact requirement described in
Appendix 4 (section 6.3) of the study guide. This cannot be responsibly
drafted from the available project history alone and must be written or
reviewed directly by the student before submission.]

## 6.4 Suggestions for future work

Four directions for future work follow directly from the unsolved
problems identified in section 5.1.3. First, the legibility/distinguishability
trade-off was tuned per author manually, evaluating a fixed set of
candidate settings against the project's own verification script rather
than via a fully automatic search; a more principled approach would
search this (and related synthesis parameters, such as letter spacing
and connection probability) automatically and jointly across authors,
which was found to be practical in principle — both judges are cheap to
evaluate and require no retraining to score a candidate setting — but
was not implemented as a general-purpose search tool within this
project's timeframe. Second, the specific, persistent case of author 150
being page-misclassified as author "dylan" by the writer-identification
judge was traced to a plausible geometric cause (both being the least
connected, least slanted hands in the set) but not resolved; a dedicated
investigation into which specific geometric features the classifier is
and is not responding to for this pair, rather than the single global
parameter that was tried and found insufficient, would be a reasonable
next step. Third, a calibrated "none of the ten authors" rejection
threshold for the writer-identification classifier, using held-out
handwriting from authors outside the trained set to establish an
appropriate confidence cut-off, was identified as a desirable safeguard
but not yet implemented or validated, and would be required before the
writer-identification judgement could be trusted in any setting where an
unknown or out-of-set author's handwriting might be presented to the
system. Fourth, and more generally, this project's experience of a
parameter change validating against two self-built sentence pools and
then regressing against the official verification script suggests that
future work of this kind should validate against a substantially larger
and more diverse pool of evaluation sentences than the six used here, to
reduce the risk of a change appearing to generalise when it has in fact
only been tuned to the idiosyncrasies of a small evaluation set.
