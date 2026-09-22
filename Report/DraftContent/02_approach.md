# 2. Approach (DRAFT CONTENT — third person, past tense)

The problem was decomposed into five sequential subsystems, each with an
independent, measurable success criterion, so that errors could be
localised to a specific stage rather than only observed as a single
end-to-end failure: (i) segmentation of a scanned handwriting page into
individual line images with correct transcriptions, (ii) a text
recognizer able to read an arbitrary handwritten line, (iii) a writer
identification classifier able to determine which of a known set of
authors produced a given line, (iv) a per-author style model capable of
generating new handwriting in that author's style for arbitrary text, and
(v) conversion of the generated strokes into G-code that a two-axis
gantry with a pen-lift solenoid could execute, subject to the constraint
that every author is drawn with the same physical pen and the same
constant stroke width.

## 2.1 Design alternatives

**Segmentation and labelling.** Two alternatives were considered for
obtaining line-level ground truth from scanned pages: (a) apportioning a
page's known printed source paragraph across its physical lines
proportionally by the number of ink columns detected per line, and (b) a
dynamic-programming alignment that matches each detected word-ink-run
against an expected pixel width derived from the word's own character
widths. Alternative (a) was implemented first, because it is simpler and
requires no additional dependency; it was later found to fail
systematically whenever different authors wrapped an identical source
paragraph into a different number of physical lines than the reference
author (see section 4 and section 5.1), and was replaced with alternative
(b) once this was diagnosed.

**Text recognition.** A segmentation-based approach (recognise isolated,
pre-cut characters) was rejected at the outset in favour of a CTC-based,
alignment-free sequence recognizer (section 1.1.1, section 3.4), because
correct character segmentation cannot be guaranteed for cursive,
variably-connected handwriting, and an error in that segmentation would
propagate directly into a wrong reading, with no way for the recognizer to
recover.

**Writer identification.** An unsupervised, hand-engineered feature
approach (measuring slant, stroke width, and letter proportions directly
and comparing against per-author reference statistics with a distance
metric) was considered, but a supervised CNN classifier was preferred
because its accuracy could be measured and iterated directly on held-out
data, and because it could reuse the convolutional backbone already
trained for text recognition, reducing the amount of independent model
development required. A single classifier trained on raw ink was found,
during testing, to be inadequate for judging machine-drawn (constant
stroke width) output, which led to the addition of a second,
stroke-normalised classifier (section 3.5.2) specifically for judging
synthesised and G-code-rendered lines; this second classifier was not
part of the original design and was introduced only once the first
classifier's failure mode had been diagnosed.

**Handwriting synthesis.** An end-to-end recurrent handwriting generation
network (Graves-style, section 1.1.3) was considered and rejected as the
primary method because of the small amount of per-author training data
available (tens of real lines per author, not the hundreds to thousands
typically required to specialise such a model convincingly), and because
it would generate raster or unstructured stroke output rather than the
explicitly parameterised, per-character vector strokes needed to drive a
gantry predictably. An exemplar/glyph-library approach — extracting real
character shapes from each author's own writing and recombining them
according to measured statistics of that author's hand — was selected
instead as the primary method, with a small, per-author-trained sequence
model retained only as a fallback to be used if the exemplar approach
could not be tuned to an acceptable accuracy (see section 3.7 and section
5.1.3 for the circumstances under which this fallback was or was not
required).

**Physical motion.** A microcontroller-mediated (ESP32, serial) motion
architecture was considered against direct GPIO stepper control from the
single-board computer running the rest of the software stack. Direct
GPIO control was preferred because it allows an end-stop safety check to
be performed in the same process, immediately before every single step
pulse, giving a guaranteed-immediate stop; a networked/serial
microcontroller architecture cannot interrupt an already-issued
in-progress move without an additional communication round trip, which
introduces a safety-relevant latency that direct control avoids
entirely.

## 2.2 Preferred solution

The preferred solution pipeline, in the order data flows through it, is:
(1) scanned/photographed pages are segmented into line images and
correctly labelled using dynamic-programming word-width alignment; (2) a
CNN-BiLSTM-CTC recognizer is trained, first on a large general-purpose
handwriting corpus and then jointly fine-tuned with a small amount of
project-specific data so that it can read both general handwriting and
the specific authors' hands without forgetting either; (3) the same
recognizer is reused, frozen, to forced-align each author's own lines and
extract individual character glyphs; (4) those glyphs are pooled into a
per-author style profile (slant, connectedness, spacing, ascender/descender
proportions, and a library of accepted letterform variants); (5) new
text is synthesised for an author by selecting and joining glyphs from
that profile according to the measured statistics, with a tunable blend
towards a cleaned-up "legibility anchor" shape where an author's own
extracted variants are ambiguous; (6) two independent classifiers/readers
score the result — the writer-identification classifier (stroke-normalised
variant) and the text recognizer — before the trajectory is converted to
G-code and physically drawn. Sections 3 and 4 describe each of these six
stages, and the corresponding measured results, in detail.
