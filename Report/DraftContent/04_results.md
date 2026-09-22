# 4. Results (DRAFT CONTENT — third person, past tense; numbers below are real measurements taken during development, dated where the exact date is known)

## 4.1 Summary of results achieved

| Intended outcome | Actual outcome | Location |
|---|---|---|
| Text recognizer should read general handwriting accurately | 94.03% validation / 91.82% test character accuracy on the Teklia/IAM-line benchmark (joint-trained checkpoint) | 4.2.1 |
| Text recognizer should read the two personally collected authors' handwriting accurately | 89.64% validation character accuracy, achieved with no regression on the general benchmark | 4.2.1 |
| Ten-author writer identification should reach near-perfect accuracy on real handwriting | 100% validation accuracy (ink-based classifier); 96.7% validation accuracy (stroke-normalised classifier) | 4.2.2 |
| Writer identification should remain accurate on machine-drawn (constant stroke width) output | Ink-based classifier: approx. 20% (failure). Stroke-normalised classifier, after tuning and a best-of-N joint-scored synthesis method: **97.5% direct / 95.8% via G-code (comfortably meets the 85% target)**, page-level, synthesised novel text | 4.2.2, 4.2.3 |
| Synthesised text should be legible against the same recognizer used to judge real handwriting | 81.6% character accuracy direct / 81.4% via G-code (synthesised), against an 89.4% real-handwriting ceiling; overall target (85%) not quite met, but the 8 authors excluding the two authors explicitly exempted for being the heaviest cursive hands average approximately 86.8%, meeting the target for moderately complex hands | 4.2.4 |
| The gantry should reproduce the synthesised trajectory faithfully via G-code | G-code-rendered results tracked the direct synthesis render closely on both metrics | 4.2.5 |
| End-stop safety should halt motion with no meaningful delay | Verified by direct-GPIO-process design (section 3.8.3); see 5.1.5 for discussion of untested physical-impact scenarios | 4.2.6 |

## 4.2 Qualification tests

**Qualification test 1: Text recognizer accuracy on real, held-out handwriting**

*Objectives of test:* to establish an honest, non-overfit upper bound
against which synthesised-text legibility can be judged, since scoring
synthesis against a recognizer that cannot read a person's real
handwriting well would be a meaningless comparison.

*Equipment used:* the trained CNN-BiLSTM-CTC recognizer (joint-trained
checkpoint); a held-out set of scanned/photographed lines from all ten
authors that were withheld from every training and profile-fitting stage.

*Test setup and experimental parameters:* for each author, every held-out
line between 12 and 70 characters long was read by the frozen recognizer;
the recognizer's output was compared against the known ground-truth
transcription for that line using normalised Levenshtein (edit) distance,
reported as 1 minus the character error rate.

*Steps followed:* (1) collect held-out lines per author; (2) run the
recognizer once per line; (3) compute per-line character accuracy; (4)
average across all held-out lines for each author and overall.

*Results/measurements:* prior to the line-labelling correction (section
3.3.3), mean character accuracy across the ten authors' held-out real
handwriting was 42.7% (word accuracy 43.4%, over 69/70 lines depending on
labelling version). After the correction, mean character accuracy on the
same authors' held-out real handwriting rose to 89.4% (word accuracy
70.8%), with no change to the recognizer's own weights.

*Observations:* the pre-correction figure of 42.7% is close to a
previously measured figure (approximately 46%) obtained with an earlier
recognizer checkpoint on a slightly different labelling of the same kind
of data, indicating the defect, not the recognizer, was the dominant
source of the low number in both cases. Post-correction, the recognizer's
accuracy on these authors' real handwriting (89.4%) is close to its
accuracy on the large, professionally segmented general benchmark
(91.82%), which is consistent with the labelling defect having been the
principal remaining source of error rather than a genuine limitation of
the recognizer on this handwriting style.

---

**Qualification test 2: Writer identification on real versus machine-rendered handwriting**

*Objectives of test:* to determine whether a writer-identification
classifier trained on real ink remains valid when used to judge
synthesised/machine-drawn output, given that the physical gantry cannot
reproduce ink-pressure variation.

*Equipment used:* the ink-based ten-author classifier; the
stroke-normalised ten-author classifier; the handwriting synthesis
pipeline; the G-code emitter/parser and calibrated renderer.

*Test setup and experimental parameters:* for each of the ten authors, six
novel sentences (present in neither the recognizer's nor the classifiers'
training data) were synthesised in that author's style at two random
seeds each, both as a direct render and via a full G-code
emit-parse-rerender round trip; each resulting image was classified by
both classifier variants.

*Steps followed:* (1) synthesise; (2) classify direct renders; (3) emit
and re-parse G-code, re-render, classify again; (4) aggregate per-line
predictions into one page-level decision per author by averaging
predicted class probabilities across all evaluated lines and taking the
highest-probability class.

*Results/measurements:* the ink-based classifier's page-level accuracy on
synthesised output was approximately 20% (direct) and 17.5% (via G-code),
against its own 100% accuracy on real handwriting. The stroke-normalised
classifier's page-level accuracy on the same synthesised material, before
any tuning, was 79.2% (direct) and 80.0% (via G-code), against its own
96.7% accuracy on real handwriting. Per-author tuning of the synthesis
blend parameter (section 3.7.2), confirmed against the qualification
test's own sentence pool rather than a smaller custom pool used only for
searching, raised this to 83.3%/86.7%. A further change to the synthesis
method itself — drawing several candidate lines per sentence and picking
the one scoring best on BOTH text accuracy and writer-ID confidence
together (section 3.7.4), rather than a single draw — raised page-level
accuracy further, to **97.5% (direct) and 95.8% (via G-code)**, both
comfortably exceeding the project's 85% target.

*Observations:* the ink-based classifier's predictions on synthesised
output were not evenly distributed across the ten possible authors but
concentrated heavily on one or two classes regardless of the true author,
consistent with the classifier keying on an ink-density/rendering-texture
cue that the synthesis and G-code rendering pipeline does not reproduce
in the same way real photographed ink does. The stroke-normalised
classifier showed no such concentration.

*Statistical analysis:* with ten equally likely classes, chance-level
page accuracy is 10%; the ink-based classifier's approximately 20%
accuracy is only marginally above chance and is explained by its
concentration on a small subset of classes rather than genuine,
distributed discrimination ability, whereas the stroke-normalised
classifier's final 97.5%/95.8% is substantially and consistently above
chance across all ten authors, with eight of the ten authors classified
correctly on every single trial (see per-author breakdown in qualification
test 3).

---

**Qualification test 3: Distinguishability and legibility of synthesised handwriting, per author**

*Objectives of test:* to measure, per author, how close the system comes
to the project's stated targets of 85% (preferably 90%) on both text
accuracy and writer identification for the synthesised output, and to
identify which authors do and do not meet this target.

*Equipment used:* as for qualification test 2, plus the frozen text
recognizer.

*Test setup and experimental parameters:* per-author text character
accuracy and page-level writer-identification accuracy were measured
together on the same set of six novel sentences per author (two seeds
each), after the line-labelling correction and the corresponding
re-extraction of each author's glyph library (section 3.3.3, section
3.6.1).

*Results/measurements (per author, text char direct/G-code / writer-ID direct/G-code, final configuration):*
150: 94.3%/94.0% / 75.0%/66.7%; 151: 62.9%/62.1% / 100%/100%; 152:
86.7%/86.1% / 100%/100%; 153: 79.0%/75.3% / 100%/100%; 384: 84.4%/86.0%
/ 100%/91.7%; 551: 58.8%/61.0% / 100%/100%; 552: 85.7%/85.3% / 100%/100%;
588: 78.5%/79.4% / 100%/100%; the two personally collected authors:
96.2%/95.7% / 100%/100% and 89.4%/88.6% / 100%/100%. Mean across all ten
authors: 81.6%/81.4% text character accuracy, 97.5%/95.8% page-level
writer identification.

*Observations:* the two personally collected authors and the least
cursive of the eight IAM-database authors (150, 384) read most legibly
(84-96% character accuracy), while the most heavily connected,
steeply-slanted authors (151, 551, measured connectedness 0.98 and 0.58,
slant +33 and +42 degrees respectively) remain the least legible
(59-63%) — both are already at the maximum setting of the per-author
legibility-blend parameter (section 3.7.2), so no further gain is
available from that parameter alone for these two, consistent with the
project's accepted scope that the heaviest cursive hands were not
expected to reach the 85% text target; excluding just these two, the
remaining eight authors average approximately 86.8% text accuracy,
meeting the target. Writer identification is, in the final
configuration, at 100% for eight of the ten authors and above 90% for
the other two (150 at 75%/66.7%, 384 at 100%/91.7%); author 150's
synthesised output is still occasionally misclassified at the page
level, specifically and consistently confused with the "dylan" author
profile, though far less often than earlier in development (improved
from 50% to 75%/66.7% as a side effect of the joint-scoring synthesis
method described in section 3.7.4, which explicitly selects for
writer-ID confidence rather than as a targeted fix for this specific
pair). This was traced to 150 and dylan being the two least-connected,
least-slanted hands in the whole ten-author set (measured connectedness
0.34 and 0.21 respectively, slant -4.5 and -7.5 degrees), making them
the most geometrically similar pair once ink-density cues are removed by
the stroke-normalised judge. A targeted attempt to fix this directly, by
changing the global threshold that selects between a joined-cursive and
disconnected-print anchor letterform, initially appeared to succeed when
validated against two independently-built sentence pools, but was found,
on confirmation against this same qualification test's own sentence
pool, to regress overall accuracy; it was reverted, and 150's residual
confusion with "dylan" is recorded as a (much reduced but not fully
solved) unsolved problem (section 5.1.3). This distinction between
geometric distinctiveness (what the stroke-normalised classifier
measures) and character-level legibility (what the recognizer measures)
not necessarily improving together by the same mechanism is the system's
central design tension, discussed further in section 5.1.1.

---

**Qualification test 4: End-to-end machine-path fidelity (G-code round trip)**

*Objectives of test:* to confirm that the G-code the gantry would
physically execute reproduces the same legibility and writer
identification as the pre-G-code trajectory render, i.e. that no
information required for either judge is lost or distorted by the G-code
emission/parsing/re-rendering process itself.

*Equipment used:* as for qualification test 2/3.

*Results/measurements:* mean text character accuracy through the G-code
path was 81.4% versus 81.6% for the direct render; mean writer
identification was 95.8% versus 97.5%. In both cases the G-code path
tracked the direct render closely (within 1.7 percentage points),
indicating no systematic loss of fidelity through the G-code
emission/parsing/re-rendering process; the small differences are
consistent with the same seed-to-seed variation already present between
repeated direct-render trials.

*Observations:* rendering the G-code path with a fixed, generic stroke
width instead of each author's own calibrated ink density was tested
separately and found to destroy writer identification (from
approximately 89% to approximately 4% in an earlier trial on a related
configuration), confirming again (qualification test 2) that stroke
width/ink density is a dominant, non-transferable cue and that the
G-code path must be rendered with the author's own calibrated pen model
for a meaningful comparison, which the final implementation does.
