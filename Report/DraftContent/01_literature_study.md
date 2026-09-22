# 1. Literature study (DRAFT CONTENT — third person, past tense, to be pasted into the report template)

## 1.1 Background and context of the problem

A system that reproduces an individual's handwriting on paper by physical
means combines three largely independent research problems: (i) machine
reading of handwritten text (handwritten text recognition, HTR), (ii)
identification of the writer of a given sample (writer identification,
WI), and (iii) generation of new, plausible handwritten strokes in a given
person's style (handwriting synthesis). Each of these has an established
literature, and the project's design decisions at every stage were guided
by results reported in that literature, adapted to the specific
constraint that the final output had to be executable as a sequence of
physical pen strokes by a two-axis gantry, not merely a rendered image.

### 1.1.1 Handwritten text recognition

The dominant architecture for line-level handwriting recognition since the
mid-2010s combines a convolutional feature extractor with a recurrent
sequence model and a Connectionist Temporal Classification (CTC) output
layer, avoiding the need for pre-segmented characters [1], [2]. Graves et
al. [1] introduced CTC as a way to train recurrent networks on sequence
labelling problems where the alignment between the input (a sequence of
image columns) and the output (a sequence of characters) is unknown and
variable in length, which is precisely the situation in cursive
handwriting, where character boundaries are not well defined by
segmentation alone. Shi et al. [2] demonstrated that a CNN feature
extractor followed by a bidirectional LSTM and a CTC loss (CRNN) gives
strong results for scene text and, by extension, offline handwriting,
without requiring an explicit character segmentation stage. This
CNN-BiLSTM-CTC pattern was adopted directly for the text recognizer built
for this project (see section 3.4), because it removes the need to solve
character segmentation before text recognition can be trained at all — a
property that turned out to be important, since the project's own line
segmentation later proved to be a significant source of error (see
section 4 and the discussion in section 5).

Handwriting recognition research depends on standardised, labelled
corpora. The IAM Handwriting Database [3] is the most widely used
English-language offline handwriting corpus, and supplied both the
general-purpose training data (via the pre-segmented Teklia/IAM-line
derivative used in this project, see section 3.4.1) and the per-writer
page scans used to build the ten author profiles (section 3.5, 3.6). The
IAM database's own construction methodology — full-page scans of a
constrained set of prompt paragraphs, each written by many different
individuals — was itself the reason the ten authors could be chosen: six
of the original candidate writers had copied the identical printed
paragraph (form c03), which permitted a direct, text-independent
comparison of stroke style between writers (see section 3.5.3).

### 1.1.2 Writer identification

Writer identification is typically framed either as a texture/statistical
problem (measuring slant, curvature, stroke width and connected-component
statistics directly from ink, as surveyed by Bulacu and Schomaker [4]) or
as a learned-representation problem using a convolutional classifier
trained directly on line or word images. This project used the latter
approach for practical reasons: a CNN classifier could reuse the same
convolutional backbone architecture as the text recognizer (see section
3.5.1), and its accuracy could be measured directly and iteratively
improved, whereas hand-engineered texture features would have required
separate feature-selection work with no guarantee of matching the
downstream, more specific requirement of this project — namely,
distinguishing a small, closed set of ten known writers, rather than
open-set identification against an unconstrained population.

An important finding reported in stroke-based forensic and biometric
writer-identification literature, and independently confirmed during this
project (section 3.5.2), is that ink density and pen pressure are
strong, but physically non-transferable, style cues: a classifier trained
on real photographed or scanned ink implicitly learns to use stroke
thickness variation as a discriminative feature, because it correlates
with genuine writer-specific pen pressure and grip. This is a problem
specific to a system whose final output is drawn by a machine with a
single pen at a constant nominal pressure, because a writer-identification
judge trained on that cue cannot meaningfully evaluate machine-drawn
output — the physical mechanism destroys precisely the signal the judge
relies on. This required a second, purpose-built classifier trained on
stroke-normalised (skeletonised and re-inked at constant width) input,
discussed in detail in section 3.5.2, so that writer-identification
accuracy could be measured on a feature set the gantry could physically
reproduce.

### 1.1.3 Handwriting synthesis

Two broad families of handwriting synthesis exist in the literature.
The first is sequence generation with recurrent networks, in which a
network is trained end-to-end on real pen-stroke coordinate sequences and
samples new stroke sequences conditioned on a text string, as introduced
by Graves in the widely cited "Generating Sequences with Recurrent Neural
Networks" [5]. This approach can produce highly natural-looking cursive
output, but it requires substantial per-writer training data (typically
hundreds to thousands of lines) to specialise convincingly to one
individual's hand, which was not available for any of the ten authors
used in this project (each of whom contributed on the order of 60-90
usable training lines, see section 3.6). More recent GAN-based approaches
(e.g. GANwriting-style few-shot style transfer) reduce this data
requirement but introduce their own instability and evaluation
difficulties, and, being learned end-to-end at the pixel or stroke level,
provide comparatively little direct control over the individual geometric
parameters (slant, connectivity, x-height, spacing) that a physical
gantry needs calibrated explicitly for the drawing to be geometrically
valid G-code.

The second family, and the one adopted for this project, is glyph-library
(exemplar-based) synthesis: individual character shapes are extracted
directly from a small number of real handwriting samples belonging to one
author, and new text is generated by selecting, connecting and mildly
perturbing those extracted shapes according to measured statistics of
the author's hand (slant, connectedness, spacing, and so on). This is
substantially more compatible with a small, per-author dataset (tens
rather than thousands of lines), is directly interpretable (a
malformed output can be traced to a specific extracted glyph or a specific
statistic), and produces vector stroke data as a natural intermediate
representation, which maps directly onto G-code motion commands (section
3.8) without requiring an additional image-to-vector reconstruction step
after synthesis, which would be necessary with a purely image-generative
GAN or diffusion approach.

### 1.1.4 Forced alignment and image preprocessing

Extracting individual character shapes from a line image, given only the
line's transcription and no character-level bounding boxes, is a forced
alignment problem. The CTC loss used for the text recognizer produces, as
a side effect of its training objective, a per-timestep probability
distribution over characters that can be decoded with the Viterbi
algorithm to recover the most likely alignment between the label sequence
and time (image-column) steps [1]. This project reused the trained
recognizer for exactly this purpose (section 3.6.1): a frozen recognizer
that had already learned to read general handwriting was run once per
line in forced-alignment mode against the known ground-truth transcription
for that line, which yields approximate per-character pixel spans. This
is standard practice in HTR pipelines that need character-level data
without character-level annotation, and avoided the alternative of manual
character-boundary labelling, which would not have been feasible at the
scale required (thousands of characters across ten authors).

Binarisation of scanned handwriting was performed with Otsu's method [6],
a classical, non-parametric global thresholding technique that selects
the threshold minimising intra-class pixel-intensity variance. This
was chosen over adaptive/local thresholding because the scanned IAM pages
and the project's own photographed personal pages both have reasonably
uniform, well-lit backgrounds, for which Otsu's method is known to
perform robustly and requires no free parameters to tune per page.

## 1.2 How the literature was applied

The literature review above directly explains four of the most consequential
design decisions taken in this project:

1. The text recognizer was built as a CNN-BiLSTM-CTC network (following
   [1], [2]) rather than as a segmentation-then-classification pipeline,
   because CTC's alignment-free training removes the dependency on a
   correct upstream character segmentation — a dependency that, when
   later introduced indirectly through a flawed *line*-level
   segmentation step, was in fact found to be the dominant source of
   measured error in the project (see sections 4 and 5.1).
2. Writer identification used a learned CNN classifier reusing the
   recognizer's own backbone, but a second, stroke-normalised variant
   was required once it was established (consistent with general
   findings that ink/pressure cues dominate CNN writer classifiers) that
   the first classifier's accuracy collapsed to near-chance when judging
   machine-rendered, constant-width output rather than the real,
   pressure-varying ink it was trained on (section 3.5.2, section 4).
3. Exemplar/glyph-library synthesis was chosen over end-to-end recurrent
   or generative synthesis specifically because of the small per-author
   sample size available (tens, not thousands, of lines) and because a
   vector stroke representation is required directly as an intermediate
   product for G-code generation, whereas an image-generative approach
   would need an additional vectorisation step not required by this
   design.
4. CTC forced alignment ([1]) was reused, rather than manual annotation,
   to solve the otherwise-intractable problem of obtaining character-level
   training data for the glyph library from only line-level
   transcriptions.

## References (draft numbering, to be finalised against the report's chosen style)

[1] A. Graves, S. Fernández, F. Gomez and J. Schmidhuber, "Connectionist
    Temporal Classification: Labelling Unsegmented Sequence Data with
    Recurrent Neural Networks," in *Proc. 23rd Int. Conf. Machine
    Learning (ICML)*, 2006.

[2] B. Shi, X. Bai and C. Yao, "An End-to-End Trainable Neural Network
    for Image-based Sequence Recognition and Its Application to Scene
    Text Recognition," *IEEE Trans. Pattern Analysis and Machine
    Intelligence*, vol. 39, no. 11, 2017.

[3] U.-V. Marti and H. Bunke, "The IAM-database: an English sentence
    database for offline handwriting recognition," *Int. J. Document
    Analysis and Recognition*, vol. 5, pp. 39-46, 2002.

[4] M. Bulacu and L. Schomaker, "Text-Independent Writer Identification
    and Verification Using Textural and Allographic Features," *IEEE
    Trans. Pattern Analysis and Machine Intelligence*, vol. 29, no. 4,
    2007.

[5] A. Graves, "Generating Sequences With Recurrent Neural Networks,"
    *arXiv:1308.0850*, 2013.

[6] N. Otsu, "A Threshold Selection Method from Gray-Level Histograms,"
    *IEEE Trans. Systems, Man, and Cybernetics*, vol. 9, no. 1, 1979.

NOTE TO SELF (remove before submission): add the student's own first-semester
report references here once that document is available, and expand this
section with any project-specific gantry/G-code kinematics literature
(e.g. Bresenham's line algorithm original paper, CNC G-code standards)
which is currently referenced only in section 3.8's design text, not here.
