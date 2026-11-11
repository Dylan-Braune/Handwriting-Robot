# Proposed Design and implementation outline

Proposed on 31 October 2026 and accepted in principle by the user, with a subsequent
flow review and classifier-placement clarification. This is not an edit to the
report. v6 is supporting material, not the outline authority. Target approximately
40 pages initially.

## 3.1 Design summary

Required deliverable/implementation/completion table linked to the proposal.
Approximately 1 page.

## 3.2 System block diagram

As-built hardware and processing diagrams, modes, interfaces and offline/deployed
boundary explained in prose. Approximately 2 pages.

## 3.3 Hardware design

- 3.3.1 Plotter mechanics and mechanical construction
- 3.3.2 Stepper motor selection and drive resolution
- 3.3.3 Pen lift mechanism and drive circuit
- 3.3.4 End stops, homing and dimensional calibration
- 3.3.5 Electrical integration and power distribution
- 3.3.6 Image capture arrangement
- 3.3.7 Hardware verification and practical limitations

Approximately 8 pages. Put axis pulse timing/interpolation beside the drive design;
put trajectory-to-machine-coordinate conversion and G-code generation in 3.6.

## 3.4 Page conditioning and line segmentation

- 3.4.1 Input conditions and processing sequence
- 3.4.2 Lighting normalisation and ink extraction
- 3.4.3 Page orientation and skew correction
- 3.4.4 Line detection and separation
- 3.4.5 Ink ownership and classifier crop preparation
- 3.4.6 Segmentation verification and limitations

Approximately 6 pages. Explain shared image primitives at first meaningful use,
without turning each helper into a separate main-report subsection.

## 3.5 Neural network design, training and classification

- 3.5.1 Training data, labels and evaluation separation
- 3.5.2 Shared convolutional feature extractor
- 3.5.3 Text recognition using sequence modelling and CTC
- 3.5.4 Writer identification and stroke normalisation
- 3.5.5 Training procedure and model selection
- 3.5.6 First principles and framework implementations
- 3.5.7 Network verification and design limitations

Approximately 9 pages. Separate architecture attribution, implemented numerical
operations, actual training route, and inference. Keep detailed layer/gradient
catalogues in supplementary material where not essential to a design argument.

In 3.5.4 explain the substantive design progression: raw-ink writer identification
worked well on photographed handwriting, but the constant-width gantry cannot
preserve the same ink cues. Explain the move to stroke-normalised shape input,
its training/inference application and trade-off, supported by matched comparisons
when available. The user states the stroke-normalised writer classifier is now
used throughout. In 3.6.6 refer back to this design choice rather than repeating it.

Describe the architecture and training recipe once. In 3.5.6 distinguish the
implemented NumPy forward/backward/optimisation route from the framework route,
their purposes, weight compatibility, and actual inference/training provenance.
In 3.5.7 distinguish fixed-weight numerical agreement from separately trained
models' predictive performance. Neither alone proves the other.

## 3.6 Style extraction and handwriting reproduction

- 3.6.1 Character alignment and glyph extraction
- 3.6.2 Skeleton tracing and pen path construction
- 3.6.3 Glyph selection and library quality
- 3.6.4 Writer style measurement and storage
- 3.6.5 Text layout and connected handwriting synthesis
- 3.6.6 Candidate selection for legibility and writer identity
- 3.6.7 Machine coordinate mapping and G-code generation
- 3.6.8 Reproduction verification and design limitations

Approximately 11 pages. Follow image-to-glyph-to-trajectory-to-machine flow.
Best-of-N search scores are selection criteria, not independent qualification.

## 3.7 System integration and embedded operation

- 3.7.1 Classification and reproduction workflows
- 3.7.2 Embedded execution, interfaces and job control
- 3.7.3 Integration verification and remaining constraints

Approximately 3 pages. Show that the components work together; keep website tour,
HTTP route catalogues and generic deployment instructions brief or supplementary.

## Writing and evidence rules for this outline

Each subsection should connect its purpose, chosen method, mathematics and
parameters, implementation and relevant verification. These are prose functions,
not mandatory repeated Theory/Implementation/Results headings.

Include measured subsystem results where they validate design decisions. Keep the
formal integrated R1–R6 qualification tests in Results. Retain representative
successful and unsuccessful examples in the main text. Appendix material can
include extended variable tables, supplementary code analysis, repeated calculations,
full sweeps and additional examples. It must not carry the only explanation or
evidence needed to assess a core contribution.

Discuss a limitation through its cause, consequence, remedy and genuine reason it
remains. Establish reasons from evidence or the user; do not default to an invented
time-constraint explanation. Discuss historical changes only where they explain a
substantive final design choice.
