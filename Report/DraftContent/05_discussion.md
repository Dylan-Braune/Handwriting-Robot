# 5. Discussion (DRAFT CONTENT — third person, past tense)

## 5.1 Critical evaluation of the design

### 5.1.1 Interpretation of results

The results in section 4 show a system that performs very well on some
sub-tasks and only partially meets its own targets on others. Text
recognition of real handwriting (89.4% character accuracy after the
labelling correction) and writer identification of real handwriting
(96.7%-100% depending on classifier variant) both comfortably exceed the
project's 85%/90% targets, which indicates the two learned models
(recognizer and classifier) are themselves adequate for the task. The
synthesis stage's shortfall was substantially larger before two rounds of
improvement: per-author parameter tuning (section 3.7.2), followed by a
change to the synthesis selection method itself, choosing the
best-scoring of several candidate draws by a joint measure of legibility
and distinctiveness rather than a single draw (section 3.7.4). In the
final configuration, mean synthesised writer identification (97.5%
direct, 95.8% via G-code) comfortably exceeds the 85% target — indeed
approaching the "preferably 90%" aspiration stated for this metric —
while mean synthesised text character accuracy (81.6% direct, 81.4% via
G-code) remains just short of the 85% target overall, though the eight
authors excluding the two hardest cursive hands average approximately
86.8%, meeting it. This is an important distinction to draw honestly,
because it would be easy to describe the system as simply "not accurate
enough" without identifying that writer identification is essentially
solved and the remaining bottleneck is specifically the character-level
legibility of the two most heavily cursive authors' generated
handwriting.

The per-author breakdown (qualification test 3) shows that this
shortfall is not evenly distributed: eight of the ten authors reach or
approach the text-accuracy target, while the two most heavily connected,
most steeply slanted authors (151 and 551) remain well below it — both
are already at the maximum available setting of the per-author
legibility parameter, so no further improvement is available from that
specific mechanism for those two authors without a more fundamental
change. This matches the project's own accepted scope, agreed before
this tuning work began, that heavily cursive authors were not expected
to reach the same target as moderately complex ones.

### 5.1.2 Critical evaluation

Two design decisions in hindsight were not correct on the first attempt
and required a documented correction once their consequences were
measured, and both are instructive about the risk of trusting a metric
without inspecting the underlying data or model behaviour that produced
it.

The first was the original, proportional line-labelling method (section
3.3.1). It was a reasonable first design given its simplicity, and it was
not obviously wrong from its own logic; its failure mode (a
one-phrase-per-line drift in read-back text) was only found by directly
inspecting individual recognizer outputs against their labelled ground
truth for a handful of lines, rather than by looking at the aggregate
accuracy number alone, which merely looked disappointingly low without
indicating why. This is a specific instance of a more general lesson
applied throughout the project: an unexpectedly low or high accuracy
number was, on more than one occasion, evidence of a data or
labelling defect rather than of genuine model performance, and the
project's practice of re-measuring a suspicious result against raw,
directly-inspected examples before accepting it as a true model
limitation proved necessary rather than merely cautious.

The second was the assumption, implicit in the first writer-identification
classifier's design, that a classifier which performs well on real
photographed handwriting would necessarily perform well on synthesised,
machine-rendered handwriting of the same content. This assumption is
false specifically because the classifier was never constrained,
during training, to ignore ink-density cues that the machine cannot
reproduce — nothing in a standard supervised training objective
discourages a classifier from using whatever cue best separates the
training classes, including cues that happen not to transfer to the
deployment distribction. The correction (a stroke-normalised classifier
variant, section 3.5.2) was successful, but it was reactive rather than
anticipated, and represents additional development time that a more
careful initial analysis of what a physical pen-plotter can and cannot
reproduce would have avoided.

Where implementation choices are considered to have been good ones in
hindsight: separating the pipeline into independently measurable stages
(section 2) allowed both of the defects above to be isolated to a
specific stage rather than only observed as one confusing end-to-end
number, and made both corrections tractable to diagnose and to verify
(via before/after re-measurement) once found.

### 5.1.3 Unsolved problems

The central unsolved problem is the trade-off between synthesised-text
legibility and writer-identification distinguishability (section 3.7.2):
increasing the legibility blend toward the designed anchor letterforms
raises text accuracy but reduces how distinctively an author's own
handwriting is reproduced, and the reverse holds for reducing it. This
parameter was tuned per author (rather than globally) against the
project's own verification script, giving an intermediate mean
writer-identification accuracy of 83.3% direct / 86.7% via G-code and
mean text accuracy of 77.2%/77.8%. A further change to the synthesis
selection method itself — described in section 3.7.4, choosing the
best of several candidate draws by a joint measure of legibility and
distinctiveness rather than legibility alone — raised the final mean
writer-identification accuracy to 97.5% direct / 95.8% via G-code
(comfortably exceeding the 85% target) and mean text accuracy to 81.6%
direct / 81.4% via G-code (short of 85% overall, though the eight
authors excluding the two hardest cursive hands average approximately
86.8%, meeting it). For the two authors with the most extreme measured
connectedness and slant (151 and 551), the legibility parameter is at
its maximum available setting (1.0) and their text accuracy remains in
the 59-63% range even with the improved selection method — no setting of
this specific parameter, nor a better selection among candidate draws,
could bring them further, which is consistent with the project's
accepted scope that the heaviest cursive hands were not expected to
reach 85%. The per-author RNN fallback described in section 3.7.3 was
not ultimately required to reach the writer-identification target and
was not implemented; whether it would meaningfully help these two
remaining low-text-accuracy authors beyond what was achieved here was
not investigated and is suggested as future work (section 6.4).

A second unsolved problem is that author 150 — one of the most legible
synthesised authors — is still occasionally page-misclassified by the
writer-identification judge (consistently confused with author "dylan"
specifically, never with any other author) even in the final
configuration, though considerably less often than earlier in
development (its own writer-identification accuracy rose from 50% to
75%/66.7% as an incidental side effect of the joint-scoring selection
method, which was not designed as a targeted fix for this specific
pair). This was traced to a plausible cause: 150 and dylan are the two
least-connected, least-slanted hands in the whole ten-author set
(measured connectedness 0.34 and 0.21 respectively, slant -4.5 and -7.5
degrees), making them the most geometrically similar pair once the
stroke-normalised judge removes ink-density cues (section 3.5.2). A
targeted fix was attempted: lowering the connectedness threshold that
selects between a joined-cursive and a disconnected-print anchor
letterform, so that 150 (whose own measured connectedness sits between
the old and new threshold) would receive cursive-anchor treatment like
the more connected authors. This fix validated cleanly against two
independently-constructed sentence pools built specifically to test it,
appearing to resolve the confusion without harming any other author —
but when confirmed against the project's own official verification
script, it in fact *regressed* overall accuracy (writer-identification
fell from 79.2% to 77.5%, and 150's own page-level accuracy fell
further, from 50% to approximately 17%). It was reverted. A second,
unrelated attempt to raise a different author's (153's) text accuracy by
increasing its own legibility parameter showed the identical pattern —
validated cleanly on a smaller custom sentence pool, then regressed that
author's own writer-identification accuracy from 100% to 91.7%/83.3% on
the official script — and was likewise reverted. Both episodes are
recorded honestly here because together they demonstrate a specific,
general risk worth stating plainly, on two different kinds of parameter
(one global, one per-author): agreement between self-constructed
validation sets is not sufficient evidence that a change generalises,
particularly when the sentence pools used for validation are small, and
every candidate improvement in this project was therefore confirmed
against the full official verification script before being kept.
Author 150's residual confusion with "dylan" remains unsolved.

A third unsolved problem, not yet quantified, is that the writer-
identification classifiers were trained and validated only on the ten
authors in the closed set; no explicit "none of these ten authors"
rejection threshold has yet been calibrated and validated against real
handwriting from authors outside the set, though this was identified as
a desirable safeguard for how the classifier should behave outside its
training distribution.

### 5.1.4 Strong points of the design

The joint text-recognizer training scheme (section 3.4.2) is considered
a strong point: it directly diagnosed and solved a real catastrophic-forgetting
regression, with the improvement confirmed by independent re-measurement
rather than accepted on a single reported number, and it generalised
correctly — the resulting checkpoint improved slightly even on data
(the general Teklia benchmark) it was not specifically being fine-tuned
for, which is evidence the fix addressed the actual underlying mechanism
(loss of exposure to the general-domain distribution) rather than merely
trading one narrow improvement for another.

The stroke-normalised writer-identification classifier is considered a
strong point for a related reason: it was designed specifically around a
physical constraint of the output device (the gantry's single, constant-
width pen), rather than around the properties of the training data alone,
and it measurably restored a classifier's usefulness on exactly the
distribution (machine-rendered output) the system actually needs to be
judged on.

The direct-GPIO, same-process end-stop safety design (section 3.8.3) is
considered a strong point on safety grounds specifically: it removes an
entire class of communication-latency failure mode that a networked or
serial microcontroller architecture would retain, at the cost of tying
motion control to the same process as the rest of the software stack.

### 5.1.5 Expected failure conditions

The system is expected to fail, or to degrade gracefully rather than
catastrophically, under the following conditions. If the text recognizer
or writer-identification classifier is presented with handwriting or
rendered output substantially different in scale, contrast or noise
characteristics from its training distribution (e.g. a page scanned at a
very different resolution, or lit unevenly enough to defeat the
page-wide Otsu threshold), recognition and classification accuracy is
expected to degrade without any explicit warning being raised, since no
out-of-distribution detection is currently implemented for either model
on the input side (as distinct from the currently-unimplemented output-side
"none of the ten authors" rejection threshold noted in section 5.1.3).
If an end-stop is triggered mid-move, the direct-GPIO control loop halts
stepping within one step interval, but no test was performed of a
genuine physical hard-stop impact (as opposed to a normal, controlled
end-stop trigger during ordinary operation), so the mechanical
consequences of an undetected or very-late-triggered end-stop condition
(e.g. a failed switch) remain untested and are treated as a residual
risk rather than a verified-safe condition.

## 5.2 Considerations in the design

### 5.2.1 Ergonomics

[DRAFT NOTE: describe the physical layout of the gantry/pen mechanism,
any user-facing controls or interfaces (e.g. how a user selects an author
and enters text to be written), and any accessibility or ease-of-use
considerations built into the software's command interface. This section
requires details of the physical enclosure/mounting that are not fully
captured in the software-focused conversation history available for this
draft — please expand directly.]

### 5.2.2 Health and safety

The system involves a motorised gantry with moving parts and a pen
mechanism, and end-stop switches are used to prevent an axis from being
driven beyond its safe mechanical range of travel, checked in the same
process as motion generation to minimise the delay between a triggered
end-stop and motion actually halting (section 3.8.3). [DRAFT NOTE: add
details of enclosure/guarding around moving parts, any pinch-point
mitigation, and electrical safety of the stepper driver and power supply
wiring, which are physical-build details not captured in this
conversation's software-focused history.]

### 5.2.3 Environmental impact

The system does not itself consume disposable materials beyond ordinary
pen ink and paper, comparable to any handwriting activity it replaces or
augments; the electronic components (single-board computer, stepper
drivers, motors) are standard, commercially available parts with
established end-of-life recycling routes similar to other small
electromechanical hobbyist/prototyping equipment. [DRAFT NOTE: state the
specific power draw of the gantry/controller if measured, and address
whether any component selection was influenced by energy efficiency or
recyclability considerations.]

### 5.2.4 Social and legal impact

A system capable of reproducing a specific, real individual's
handwriting raises a legitimate concern around consent and potential
misuse (e.g. producing a handwritten document that a reader could
mistake for genuinely having been written by the person whose style was
modelled). [DRAFT NOTE: state explicitly what consent was obtained from
the two personally collected authors (yeukita and dylan) to have their
handwriting collected, modelled and reproduced by this system, and
whether any explicit statement on acceptable/intended use of the system
was defined as part of the project's scope — this is required content
for this subsection and should be written to reflect the actual
arrangement made.] No legislation specific to handwriting-imitation
systems is known to the author; general principles of data protection
(regarding the personally collected handwriting samples, which are a
form of personal biometric-adjacent data) and of fraud/forgery law
(regarding potential misuse of the system's output) are considered the
relevant legal context, though this project's own use of the data was
limited to model training and evaluation with the consent of the two
personal contributors. [DRAFT NOTE: confirm and finalise the exact legal
framing appropriate for the jurisdiction and confirm data handling
practice, e.g. whether personal handwriting samples were stored,
anonymised, or will be deleted after the project concludes.]

### 5.2.5 Ethics clearance

[DRAFT NOTE: state here whether ethics clearance was required for
collecting the two personal authors' handwriting samples and, if
obtained, the assigned clearance number; if not required, include an
explicit statement to that effect, following the exact requirement in
Appendix 4 of the study guide.]
