# 3. Design and implementation (DRAFT CONTENT — third person, past tense)

## 3.1 Design summary

| Deliverable / task | Implementation | Completion | Section |
|---|---|---|---|
| Line segmentation and labelling of scanned handwriting pages | Implemented from first principles: Otsu binarisation, ink-row clustering for line boxes, OCR of the printed source text, dynamic-programming word-width alignment for per-line transcription | Completed (after a documented redesign, see 5.1.2) | 3.3 |
| Handwritten text recognition (CNN-BiLSTM-CTC) | Implemented and trained from first principles; PyTorch used as the numerical/autograd framework only, network architecture and CTC forced-alignment code written by the student | Completed | 3.4 |
| Writer identification (10-author classifier) | Implemented and trained from first principles; two variants (ink-based and stroke-normalised) | Completed | 3.5 |
| Author style profile extraction (per-character glyph library) | Implemented from first principles: CTC Viterbi forced alignment, glyph pooling and cross-author letter prior | Completed | 3.6 |
| Handwriting synthesis (text-to-trajectory generation) | Implemented from first principles: exemplar-based glyph selection, connectivity/legibility blending | Completed; distinguishability and legibility targets not fully met for the most cursive authors (see 4 and 5.1.3) | 3.7 |
| G-code generation and parsing | Implemented from first principles: linear and arc (chord-interpolated) motion commands, pen up/down control | Completed | 3.8 |
| Gantry motion control (direct GPIO stepper driving, end-stop safety) | Implemented from first principles: Bresenham-coordinated multi-axis stepping, same-process end-stop interrupt | Completed | 3.9 |

## 3.2 System block diagram

*(Figure placeholder: page image/photograph -> line segmentation ->
labelled line images -> [A] text recognizer training branch, [B] writer
identification training branch -> per-author forced alignment -> glyph
library / style profile -> handwriting synthesis -> trajectory ->
G-code -> gantry. The frozen text recognizer and the frozen writer-ID
classifier are also used a second time, downstream of synthesis, purely
as evaluation judges, shown as a feedback path into the results/verification
stage.)*

The system is organised as a pipeline with two data sources feeding a
shared architecture: a general-purpose handwriting corpus (for
recognizer pre-training) and ten specific authors' own pages/photographs
(eight drawn from the IAM database, two personally collected). Both
sources are reduced to the same internal representation — a labelled
line image — before any author-specific processing begins, which allowed
the same segmentation, recognition and classification code to operate
identically on IAM scans and on personally photographed pages without a
parallel implementation.

## 3.3 Dataset preparation and line segmentation

### 3.3.1 Design alternatives

Two page-segmentation and labelling strategies were implemented over the
course of the project.

The first strategy detected physical handwriting lines directly from ink
row-density (rows with a sustained run of dark pixels, after Otsu
binarisation, were clustered into line bands), and assigned each
detected line a text label by apportioning the page's known printed
source paragraph across the lines *proportionally to the number of
ink-columns (approximate word count) detected in each line*. This is
computationally simple and requires only the count of lines detected and
the count of words in the source text.

The second strategy, adopted after a labelling defect was diagnosed
(section 5.1.2), instead performs a true alignment: it OCRs the page's
own clean, printed header text with Tesseract, detects word-shaped ink
runs on each physical line using an Otsu-thresholded gap size (so word
boundaries are separated from letter boundaries by the same
non-parametric method used for pixel binarisation), and aligns the
resulting sequence of ink-runs against the sequence of expected words
using a dynamic-programming edit distance formulated over *pixel width*
rather than character identity.

### 3.3.2 Design procedure followed

Binarisation used Otsu's method [ref 6, section 1], which selects the
threshold $t^*$ that minimises the within-class variance of pixel
intensities:

    t* = argmin_t  w_bg(t) * var_bg(t)  +  w_fg(t) * var_fg(t)     (1)

where `w_bg`/`w_fg` are the background/foreground pixel-count fractions
at threshold `t`, and `var_bg`/`var_fg` are the corresponding intensity
variances. This threshold was computed once per page (not per line), on
the assumption of reasonably uniform page-wide lighting, which held for
both the scanned IAM pages and the photographed personal pages.

Word-width alignment models each known word `w_j` (from the OCR'd
header) with an *expected pixel width* derived from a per-character width
table (narrow characters such as `i`, `l`, `t` are assigned a smaller
unit width than wide characters such as `m`, `w`, following typographic
convention), scaled by a single page-wide pixels-per-character-unit
constant estimated from the ratio of total detected ink-run width to
total expected character-width units across the whole page. Given `M`
detected ink runs of width `r_i` and `N` expected words of width `e_j`,
alignment is solved as a shortest-path problem over a `(M+1) x (N+1)`
grid, permitting three transition types per step: a direct 1:1 match
(one run assigned to one word, cost proportional to the squared relative
error between `r_i` and `e_j`), a 2:1 merge (two adjacent runs assigned
to one word, for a word whose letters were mis-split into two ink runs),
and a 1:2 split (one run assigned to two words, for two short words
whose ink runs merged), each carrying a small fixed penalty to discourage
over-use of merges/splits relative to the far more common 1:1 case. A
hyphen-aware sub-word split routine additionally allows a single ink run
to be divided between the last word of one physical line and the first
word (or word fragment, for a hyphenated word) of the next.

Individual character glyphs required for the style-profile stage (section
3.6) were obtained by re-using the trained text recognizer in forced
alignment mode: the recognizer's per-timestep character-probability
output, together with the known ground-truth transcription for a line,
was decoded with a Viterbi search over the CTC blank-interleaved label
sequence, giving the maximum-likelihood assignment of image-column spans
to characters.

### 3.3.3 Outcome and measured effect

The first (proportional) labelling strategy produced a systematic
defect: for four authors that had copied an identical source paragraph
(the IAM c03 prompt), each writer wrapped the paragraph into a different
number of physical lines depending on their own handwriting size, but
the proportional method's assumption of positionally-matching line
breaks across writers was violated whenever a writer's wrap points
differed from the reference. The visible symptom was a text recognizer
reading output that closely resembled the *correct words for a
neighbouring physical line*, shifted forward by roughly one phrase per
line and compounding down the page (see the discussion of this finding
in section 5.1.2, with the numerical evidence in section 4). Re-running
the alignment-based strategy across all ninety-eight labelled pages
changed the assigned line *count* for zero pages (verified with a
dry-run pass before committing any change), meaning the existing,
independently-built pixel-crop cache for these lines remained valid, and
the correction was purely to the text assigned to each already-correctly-cut
crop. The measured character-accuracy improvement on real (non-synthesised)
handwriting held out during training rose from 42.7% to 89.4% purely from
this relabelling, with no change to the recognizer's weights (section 4).

A secondary defect was found and corrected in the same pass: the OCR
step used for the alignment method produces Unicode "smart quote"
characters (e.g. the right single quotation mark, U+2019) that were not
members of the fixed 70-character alphabet the recognizer was trained
against. The text-encoding routine used throughout the project silently
discards any character outside this alphabet rather than raising an
error, so every apostrophe introduced by the new labelling method would
otherwise have been dropped from the training/scoring target text
without warning, artificially and permanently reducing measured accuracy
on every line containing a contraction or possessive. This was corrected
by normalising OCR-introduced typographic punctuation to its
plain-ASCII equivalent before the labels were used for either training
or evaluation.

## 3.4 Text recognition model

### 3.4.1 Architecture and training objective

The recognizer follows a CNN-BiLSTM-CTC design (section 1.1.1): a
convolutional stack extracts a sequence of column feature vectors from
the input line image (fixed height, aspect-ratio-preserving width), a
bidirectional LSTM models context along that sequence in both reading
directions, and a final linear-plus-softmax layer produces, at every
timestep, a probability distribution over the alphabet plus one
additional "blank" symbol required by CTC. Training minimises the
negative log-likelihood of the target label sequence under the CTC loss,
which sums the probability of every possible frame-to-label alignment
consistent with the target (including repeated characters separated by
blanks, to allow a character to occupy more than one timestep) via a
forward-backward dynamic-programming recursion, avoiding the need to
commit to any one alignment during training.

### 3.4.2 Design procedure followed: three training-data regimes

Three successive text-recognizer checkpoints were produced over the
course of the project, each addressing a limitation found in the
previous one.

The first checkpoint was trained only on lines derived from this
project's own (at that time uncorrected) page segmentation, on the order
of 790 lines. Because this dataset is small and drawn only from the ten
target authors' own, often unusual, handwriting, its ability to
generalise to ordinary handwriting was limited.

The second checkpoint was trained on 6,480 lines from the Teklia/IAM-line
derivative of the IAM database — a large, professionally pre-segmented,
general-purpose handwriting corpus with no writer-identity field
attached, which is why it could only ever be used for recognizer
training, not for the writer-identification task. This measurably
improved general handwriting-reading ability (mean character accuracy on
a held-out set of scanned pages rose from 77.2% to 83.7%, with the
largest single-author improvement, for the most cursive writer in the
original ten, from 82.1% to 95.1%), but had not yet been specialised to
the two personally collected authors' handwriting.

The third, and final, checkpoint addressed the two personally collected
authors specifically. A straightforward sequential fine-tune of the
second checkpoint on the personal authors' own lines was attempted
first: this raised personal-author accuracy (to 90.2% full fine-tune,
88.5% with the convolutional backbone frozen) but caused a substantial
regression on the general Teklia benchmark (from 93.87%/91.60% down to
84.26%/80.56% validation/test character accuracy), a textbook instance of
catastrophic forgetting — the network's weights moved to satisfy the
new, narrow training distribution at the expense of the previously
learned, broader one. This was resolved by *joint* training instead of
sequential fine-tuning: every training epoch drew a weighted mixture of
Teklia and personal-author lines (a fixed personal-line fraction was
maintained using a weighted random sampler, oversampling the small
personal set without literally duplicating it), so the network was never
allowed to lose exposure to the general-domain data while it learned the
personal-author-specific data. This produced a checkpoint that improved
on personal-author accuracy (89.64% validation character accuracy) with
no regression, and in fact a small improvement, on the general Teklia
benchmark (94.03% validation, 91.82% test) relative to the pre-fine-tune
checkpoint — confirming that the forgetting problem, not a fundamental
data or capacity limitation, was the cause of the earlier regression.

## 3.5 Writer identification

### 3.5.1 Architecture

The writer-identification classifier reuses the recognizer's
convolutional feature-extraction backbone architecture (though trained
with independent weights, warm-started from the recognizer's own trained
weights rather than from random initialisation), followed by an
adaptive pooling stage that reduces the variable-width feature sequence
to a single fixed-length vector per line, and a linear classification
head over the (in the final configuration) ten author classes. Reusing
an architecturally-identical, and initially weight-identical, backbone
to the recognizer avoided a second independent convolutional-network
design and training effort, and gave the classifier a head start of
already-learned, general handwriting-shape features before any
author-specific training began.

### 3.5.2 A significant design correction: ink density versus reproducible geometry

The first, ink-based, ten-author classifier trained to a measured 100%
validation accuracy on real, correctly-segmented pages. When the same
classifier was subsequently used to judge *machine-drawn* output (either
the direct synthesis render or the same content parsed back out of
generated G-code and re-rendered), accuracy collapsed to approximately
20%, heavily biased toward guessing one or two of the ten classes
regardless of the true author. The cause was established by constructing
a second classifier, identical in architecture, but trained on
*stroke-normalised* input — every training line binarised, reduced to its
skeleton (topological centreline), and re-drawn at one constant pen
width, identical for every author — which forces the classifier to learn
slant, letter proportions, spacing and connection habits rather than any
cue derived from ink thickness or pressure variation. This
stroke-normalised classifier reached 96.7% validation accuracy on real
handwriting, and, critically, did not collapse when judging synthesised
output, because the gantry draws every author with the same physical pen
at the same constant stroke width and ink density is consequently not a
style channel the machine can reproduce; a judge trained to rely on it is
measuring something the system is not designed to produce. This finding,
and the corresponding architectural fix, is documented further in section
5.1.2 as an example of a critical design correction driven directly by an
evaluation result rather than anticipated at design time.

A further, easily overlooked implementation requirement follows directly
from this design: because the stroke-normalised classifier's
convolutional backbone was calibrated on stroke-normalised images during
training, any image submitted to it for classification at evaluation
time — whether a real photograph or a synthesised render — must be passed
through the identical stroke-normalisation transform before
classification, or the classifier receives input from a different
distribution than the one its weights were fitted to and its output
becomes unreliable. This requirement was initially missed when the
stroke-normalised classifier was first substituted into the evaluation
pipeline, and its correction is documented as a second, related fix in
section 5.1.2.

### 3.5.3 Author selection

Ten authors were used: eight drawn from the IAM database (chosen for
having among the largest per-writer page counts available, and, within
that constraint, for stylistic diversity, discussed below) and two
personally collected. Two additional IAM-database candidate writers were
excluded from the final set because they had copied the same source
paragraph as three of the eight retained writers and were judged, by
direct visual comparison of a cropped mid-page band from each candidate's
real handwriting, to be redundant with an already-represented stroke-style
cluster, adding comparatively little additional style diversity to the
ten-author set while consuming an equal share of classifier-training
and glyph-extraction effort. Per-author style statistics measured during
profile construction (section 3.6) — slant angle and a measured
letter-connectedness fraction — confirmed that the retained eight IAM
authors span a wide style range (measured slant from -4.5 to +42 degrees;
measured connectedness from 0.34 to 0.98), which was the deliberate
selection criterion.

## 3.6 Author style profile construction

### 3.6.1 Glyph extraction

For each author, every non-holdout training line was forced-aligned
(section 3.3.2, section 3.4.1's CTC decoding applied in alignment rather
than free-recognition mode) against its own known transcription, and
each aligned character span was cropped out as an individual glyph image
together with its stroke-level vector representation (obtained by
skeletonising the cropped ink and tracing the resulting centreline into
an ordered point sequence, i.e. converting the raster crop into the
sequence-of-strokes representation needed for later trajectory
synthesis, section 3.7). Alignments scoring below the 25th percentile of
an author's own per-line alignment confidence were discarded, on the
basis that a low-confidence alignment indicates the character boundaries
it implies are not trustworthy, and a small number of additional
fragment-size gates rejected variants whose total pen-stroke length or
longest single stroke fell below a fixed minimum, removing broken or
partial glyph cuts before they could enter an author's variant library.

### 3.6.2 Cross-author letter prior and per-author profile

Because any single author's own handwriting sample (on the order of
60-90 usable lines) is too small to reliably distinguish a genuinely
unusual but valid letterform from a bad segmentation cut, an additional
pooling step builds a *cross-author* prior for what each letter
typically looks like, using every accepted glyph variant from all ten
authors together. This prior is used only to down-weight or reject an
author's own least-typical variants; it never substitutes another
author's letterform for a missing or malformed one, preserving the
requirement that every character drawn in an author's style is drawn
using that author's own extracted ink wherever a usable variant exists.

Each author's final profile records: a measured slant angle, a measured
connectedness fraction (the proportion of adjacent in-word letter pairs
that are physically joined in the author's real handwriting), a
measured x-height-normalised word-spacing statistic, ascender and
descender extents, and, for each character the author was observed to
write, up to twelve accepted glyph variants ranked by typicality (an
empirically measured setting: restricting selection to the two or three
most typical variants per character, rather than a larger pool, measured
higher writer-identification accuracy on synthesised text — 80.5% versus
75.5% — because averaging across many stylistically different variants
of the same letter dilutes the very letterform signature that makes the
author's writing identifiable).

## 3.7 Handwriting synthesis

### 3.7.1 Trajectory generation

Given arbitrary text and an author's profile, synthesis proceeds
character by character within each word: a glyph variant is selected
(weighted toward the author's own most typical variants, section 3.6.2),
scaled to the requested physical x-height, sheared according to the
author's measured slant (capped to avoid ascenders/descenders ramping
off a legible vertical range at extreme slant angles), and placed at the
current pen position. Adjacent letters within a word are optionally
joined by a short connecting stroke (a "ligature"), drawn with
probability equal to the author's measured connectedness fraction rather
than as an unconditional on/off rule — an unconditional joining rule was
tried first and found to fuse whole words into single, over-wide ink
blobs for highly connected authors, which is both visually wrong and
directly counter-productive for writer identification, since it distorts
exactly the letter-pitch and component-count statistics a
writer-identification classifier depends on. A minimum-clear-space
collision guard pushes a glyph's horizontal placement to avoid its ink
overlapping the immediately preceding glyph's body, independently of
whether the two are being visually joined.

### 3.7.2 The legibility/authenticity trade-off

Where an author's own extracted glyph for a given character is
ambiguous or easily confused with another letter (measured as its
distance from the cross-author prior for that character, section 3.6.2),
the design provides a tunable blend, per instance, towards a single
designed "legibility anchor" letterform for that character — either an
upright, print-style anchor or, for authors whose measured connectedness
exceeds a threshold, a joined cursive-skeleton anchor that can still
connect naturally into neighbouring letters. This blend is controlled by
a single scalar per author (nominally in the range 0 to 1), and
represents a direct, explicit trade-off: increasing it raises text
legibility (because ambiguous author-specific shapes are progressively
replaced by an unambiguous reference shape) at the cost of authenticity
and, consequently, of writer-identification accuracy, since the anchor
shape is by construction less distinctive than the author's own real
ink. Section 4 and section 5.1.1 report the measured effect of this
parameter and the point selected for each author.

### 3.7.3 Fallback method considered but not adopted as primary

A per-author sequence model (a small recurrent network trained only on
that author's own, small stroke dataset, rather than a large, general
corpus) was designed as a fallback synthesis method, to be used only if
the exemplar-based approach could not be tuned to an acceptable combined
legibility/distinguishability result. This fallback was not ultimately
required: the exemplar-based approach, combined with the selection
method described in section 3.7.4, reached the project's targets for
eight of the ten authors, and the per-author sequence model was
therefore not implemented in the final system. It remains a documented
option for future work (section 6.4) for the two authors whose text
accuracy did not reach the target.

### 3.7.4 Selection by best-of-N, scored jointly on legibility and distinctiveness

The single largest remaining improvement made to the synthesis stage was
not a change to how individual glyphs are drawn, but a change to how a
finished candidate line is *selected*. Because `SynthesizeText` draws
random per-instance jitter and, for ambiguous glyphs, a random choice
between the author's own variant and the legibility anchor (section
3.7.2), two draws of the same text in the same author's style are not
identical — some draws happen to read more clearly, others happen to
look more distinctively like the author's own hand.

The first attempt exploited this by drawing a line several times and
picking the draw the frozen text recognizer read most accurately (a
"best-of-N" scheme), optionally followed by a repair pass that
re-draws specifically the characters the recognizer still could not
read, pinning just those to the legibility anchor. This raised text
accuracy substantially, but it was found to *reduce* writer-identification
accuracy for several authors, because selecting purely for legibility
has no reason to also select for distinctiveness, and the repair pass in
particular replaces individual letterforms with a less author-specific
shape.

The adopted design instead scores every candidate draw by the harmonic
mean of two quantities together: the text recognizer's character
accuracy on that candidate, and the writer-identification classifier's
predicted probability that the candidate belongs to the intended author.
The draw with the best *joint* score is kept, with no repair pass. This
change is a genuine design correction, not a hyperparameter adjustment:
it recognises that legibility and distinctiveness are two properties of
the same rendered image, decided by two different frozen judges, and
that optimising for one without reference to the other will
systematically trade one away for the other rather than finding a good
compromise. Measured effect: text accuracy improved for every author
with no writer-identification regression for any author, and several
authors whose writer-identification accuracy had been in the 33-83%
range rose to 100%. Increasing the number of candidate draws considered
per line (from 6 up to 14) improved text accuracy further, monotonically
and without further writer-identification cost, since each additional
draw simply gives the joint-scoring selection a larger pool to choose
the best compromise from; this is a materially different, and safer,
kind of parameter than a fixed synthesis constant, because the selection
process re-runs fresh on whatever specific text is being rendered rather
than being tuned once on one sample of text and applied blindly
elsewhere (see the discussion of a related, less safe parameter change
in section 5.1.2). The computational cost is proportionally higher
(each candidate requires a full recognizer pass and a full classifier
pass), which was judged acceptable for generating a final line to be
drawn, though not for rapid, iterative parameter searching, for which
the cheaper single-draw synthesis was used instead, with any
improvement it suggested confirmed against the full evaluation
described in section 4 before being adopted.

## 3.8 G-code generation and motion control

### 3.8.1 G-code emission

The synthesised trajectory (a sequence of pen-down polylines and pen-up
transitions in physical millimetre coordinates) is converted directly
into a G-code program: `G0`/`G1` linear moves carry a modal feed rate `F`
(mm/min) computed from the physical distance to be travelled and a
minimum feed-rate floor; `M3`/`M5` toggle the pen-lift solenoid down and
up respectively at stroke boundaries; `G4 P<seconds>` inserts an explicit
dwell where required. `G2`/`G3` circular arcs (used, for example, to
produce test patterns and to validate the interpreter independently of
handwriting content) are supported via an `I`/`J` centre-offset
specification and are interpolated into short, roughly 0.4 mm, straight
chords, since the physical gantry has no native circular-interpolation
stepping mode.

### 3.8.2 G-code parsing and physical motion execution

A companion parser reads back an arbitrary G-code file (used both to
drive the physical gantry and, for verification purposes, to re-render
exactly the image the gantry would draw, section 4) and decomposes each
linear segment into individual stepper pulses using Bresenham's line
algorithm, which selects, at each iteration, the axis step that keeps
the accumulated position error smallest relative to the ideal line,
guaranteeing coordinated, straight multi-axis motion using only integer
arithmetic per step — appropriate for a single-board-computer GPIO
control loop where floating-point trigonometric evaluation per step
would be unnecessarily costly.

### 3.8.3 Safety-relevant design choice: direct GPIO control

Stepper and end-stop I/O is driven directly from GPIO on the same
single-board computer that runs the motion-planning software, in
preference to a networked or serial-connected microcontroller
architecture that had been used in an earlier iteration of the gantry
control software. The end-stop input is polled in the same process,
immediately before every individual step pulse is issued, which
guarantees that a triggered end-stop halts motion within one step
interval with no communication latency. A serial/networked microcontroller
architecture cannot offer the same guarantee, because a move, once
issued to the microcontroller, executes to completion (or until the next
polled status message is received and acted upon) without the ability
for the host to interrupt it mid-move without an additional round trip;
this was judged an unacceptable risk for a mechanism capable of driving
an axis into a physical hard stop.

## 3.9 Software implementation summary

All model architectures (the CNN-BiLSTM-CTC recognizer, the CNN writer
classifier and its stroke-normalised variant), the CTC forced-alignment
Viterbi decoder, the Otsu thresholding and word-width dynamic-programming
alignment routines, the glyph extraction and style-profile construction
code, the handwriting-synthesis trajectory generator, the G-code
emitter/parser, and the Bresenham-based motion executor were all
implemented from first principles for this project. PyTorch was used
only as a numerical tensor/autograd library (i.e., for the mechanics of
backpropagation and GPU-accelerated convolution/matrix operations), not
for any pre-built recognizer, classifier, alignment, or handwriting-
synthesis functionality; no third-party handwriting-recognition,
writer-identification, or handwriting-synthesis library or pre-trained
model was used anywhere in the pipeline. [DRAFT NOTE: verify and adjust
this paragraph to exactly match the student's own software-library
usage declaration for Part 1 before submission, including whether
Tesseract OCR (used only for reading a page's own printed header text,
never handwriting) needs to be listed as an explicit declared exception.]
