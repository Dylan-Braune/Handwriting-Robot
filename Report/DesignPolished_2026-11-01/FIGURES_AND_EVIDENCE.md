# Figure and evidence register: visual revision

This pass fills six formerly pending assets using real data or code-derived diagrams:
processing overview, network architecture, native-page segmentation stages, actual
camera-roll capture comparison, original-line glyph alignment and three-candidate
scoring. It adds four-profile G-code renders and an actual robot-page/crop example
in the appendix. See `EVIDENCE_NOTES.md` for provenance and limitations.

The user's original hardware overview is now included unchanged. Still pending:
software block diagram, assembly photo, complete installed
circuit, camera-arrangement photo, labelled glyph-filtering cases and a genuine
joining example. The historical checklist below is retained as a detailed drawing
brief; its older status cells are superseded by this paragraph.

# Figures and evidence to complete

This register records actual requests, not completed tests. The user will provide
diagrams/examples or decide that an illustration is unnecessary. Continue other
writing while requests are pending. Formal full-system results and QTPs remain in
Results, not here.

## Quick checklist — updated after the user's hardware reply

### System diagrams to prepare with guidance

- [x] Hardware system overview: user's supplied diagram, included unchanged;
  pin-level mapping is in Table~\ref{tab:des-pin-map}.
- [ ] Software block diagram: off-device preparation, ODROID page reading, and
  ODROID writing as three lanes, including stored files and the G-code boundary.
- [x] A simpler processing overview has been generated as a planning sketch; it
  does not replace the requested final software block diagram.
- [ ] Complete electrical schematic: actual installed wiring and values.
- [ ] Shared network architecture with separate reader and writer branches.

### Photographs to obtain later

- [ ] Complete gantry, labelled to explain the mechanical arrangement.
- [ ] Camera and page arrangement, showing mounting and lighting.

### Worked illustrations to prepare from real project data

The assistant can assemble these; the user need not draw them by hand.

- [ ] Page segmentation stages, from photograph to final line crops.
- [ ] Lighting and rotation examples for the same page, with selected crops.
- [x] Stroke normalisation stages (rebuilt at selected width 0.17).
- [ ] Transcript alignment and glyph extraction, using original handwritten inputs.
- [ ] Glyph variant filtering: retained/rejected examples and representative choice.
- [x] Actual final-profile glyph example: four Dylan `a` variants, real medoid
  and median prototype at the same scale. The final library is taken as given;
  no manual-curation feature or initial-versus-final comparison is presented.
- [ ] Glyph placement and joining: paths, detached marks and connector decisions.
- [ ] Candidate selection: a few real alternatives, actual scores and selection.

These are design explanations, not full-system results or QTPs. Existing axis,
pen mechanism, switch and calibration-path figures are retained; replace them only
if needed for correctness or legibility. This is the working figure set, not a
requirement to include every illustration regardless of page value.

The user identified `Software/CleanPipeline3/ReportResults/FullReproducedPages/`
as available material and recommended Owen, IAM 150, Yeukita and Dylan for clarity.
Use their actual reproduced pages for appropriate reproduction illustrations.
Alignment/profile-extraction illustrations must use original handwriting and its
known transcript; do not present analysis of robot output as original profile
construction. Prefer existing intermediate artefacts before generating new ones.

## High priority system overview

The user supplied a hardware sketch on 1 November 2026. Its architecture has been
redrawn as `Figures/design-hardware-overview.pdf` with corrected 3.3 V control,
24 V motor-rail and unverified adapter labels. The original attachment is retained
as `Evidence/user_hardware_diagram_input.png`; use the vector redraw in the report.
The electrical table `Table~\\ref{tab:des-pin-map}` maps physical header pins,
GPIO chip line offsets, signal levels, power paths and current evidence. The actual
ODROID adapter rating, installed ground topology, driver Vref and some pull-down
connections remain unverified. The detailed software block diagram remains pending.

### Hardware overview — Figure in 3.2

Filename: `Figures/design-hardware-overview.pdf` (completed vector redraw; see
`make_hardware_overview.py`).

Suggested arrangement: camera and page at left; ODROID in the middle; two A4988
drivers with X/Y motors at right; pen-motor switching circuit below the drivers;
four axis end stops and one pen-state switch below the ODROID; power supplies
along the bottom. Keep the physical pen/gantry visible as the output assembly.
Give the ODROID named USB, GPIO-header, network, DC-in, 3.3 V and GND ports so
the arrows terminate at meaningful interfaces. Use a solid style for electrical
links and a distinct dashed style for the Y cross-shaft and belt coupling. The
diagram is an overview rather than a circuit; reserve pin numbers and resistor
values for the complete circuit figure. A restrained key may distinguish custom
design elements from purchased modules without adding text to every block.

Show:

- Camera image data → ODROID, labelled USB.
- ODROID STEP/DIR → each A4988; each driver → its two motor windings.
- ODROID control → BC337/pen switching stage → geared motor and cam.
- X MIN, X MAX, Y MIN, Y MAX and PEN UP feedback → ODROID inputs.
- Confirmed 24 V rail (reported supply setting 2.11 A; do not present this as a
  measured A4988 phase-current limit) → motor-driver supply inputs.
- Supplied ODROID mains adapter → ODROID (DC output voltage still unknown).
- ODROID dedicated 3.3 V power rail → pen motor switching circuit and other
  installed 3.3 V loads; GPIO provides control signals, not motor supply power.
- Mark the pen motor as having no fitted flyback diode; keep this callout distinct
  from installed protection components.
- Laptop/browser ↔ server over the network as the operator interface.
- Common logic reference only if confirmed against the installed wiring; do not
  infer a shared ground from the separate-supply description alone.
- Legend distinguishing USB image data, GPIO/switch signals, electrical power
  and mechanical coupling. Keep the Y cross-shaft link visibly mechanical.
- No encoder path: show the end stops as homing/limit feedback, not continuous
  position measurement. The installed pen circuit has no flyback diode; call this
  out here only if it remains legible, otherwise show it in the circuit figure.

The network link describes the demonstrated operator interface; do not annotate
the ODROID as lacking an HDMI connector. It has a display connector even if the
robot is operated headlessly. Common ground and pen-rail return paths must follow
the actual installed circuit and remain a verification point until that drawing
is available.

This overview is not a complete circuit schematic. Leave resistor values and
physical pin numbering for the electrical drawing. Label the Y motor as driving
both side belts through the cross-shaft, not as two separate Y motors.

### Software block diagram — Figure in 3.2

Filename: `Figures/design-software-block-diagram.pdf`.

Use three lanes with stored artefacts between them. Keep the diagram to roughly
12–15 top-level blocks so the labels remain readable:

1. Labelled page/line images → preparation → network training → model weights;
   labelled lines + reader → glyph/style extraction → stored profiles.
2. Photograph → conditioning and line segmentation → reader + writer model →
   recognised text and writer, with per-line information.
3. Digital text + selected profile → glyph layout and candidate selection → pen
   trajectory → G-code file → gantry interpreter. Show reader/writer model scores
   feeding the implemented best-of-N candidate search.

Distinguish off-device training from ODROID request-time reading and writing. Do
not imply that an unlabelled classification request trains models or builds a new
profile. Draw model weights, line caches, writer profiles and generated jobs as
data objects. Label major blocks with proposal FU identifiers where they fit.
Mark the G-code file as the software/hardware boundary: the synthesiser ends there
and the gantry interpreter begins. Label best-of-N feedback as model scoring,
not physical validation. Reading a page and writing a new text are separate
requests; do not imply that page classification triggers reproduction.

## Other requested figures

| Location | Filename | Required content | Status |
|---|---|---|---|
| 3.3.1 | `design-gantry-assembly.jpg` | Whole assembled gantry with custom brackets, rails, shared shaft, belt fastening/tensioning and page restraint labelled | Requested |
| 3.3.5 | `design-complete-circuit.pdf` | Actual complete wiring, rails, GPIO mapping, driver configuration and component values, protection and decoupling | Requested |
| 3.3.6 | `design-camera-arrangement.jpg` | Actual camera mount, page support, working distance and illumination direction | Requested |
| 3.4.1 | `design-segmentation-stages.pdf` | One real page through ratio image, mask, cleaning, centres/seams and final crops | Pending source example |
| 3.4.6 | `design-segmentation-variation.pdf` | Same handwriting in normal/uneven lighting and small rotation; selected crops and actual observations | Requested, not a QTP |
| 3.5.2 | `design-network-architecture.pdf` | 64×640 input, 2/4/6 convolution stages, pooling, 128 features ×160 frames, reader and writer branches | Pending drawing |
| 3.5.4 | `design-stroke-normalisation.pdf` | Original, binary, skeleton and fixed-width crop segments | Rebuilt from existing data at selected fraction 0.17 |
| 3.6.1 | `design-glyph-alignment.pdf` | Known text, initial CTC locations, refined cuts, resulting glyphs; connected example | Asked whether existing figures are available |
| 3.6.3 | `design-glyph-quality.pdf` | Retained/rejected variants, reasons, medoid and accepted prototype if present | Pending real example |
| 3.6.3 | `design-glyph-profile-example.pdf` | Four actual final-profile variants and two representatives | Generated from stored Dylan profile; source hashes recorded |
| 3.6.5 | `design-glyph-joining.pdf` | Entry/exit, body versus detached mark, accepted and rejected connector cases | Asked for existing worked example |
| 3.6.6 | `design-candidate-selection.pdf` | Few candidates for fixed text/profile, A/P/J scores, selected candidate and repair/fallback sources | Pending logged example |

Existing axes, pen mechanism, prototype circuit, switch circuit and square/triangle
images were retained from the Prism export. Final circuit images must represent
the installed wiring, not a generic illustration or an assumed supplier variant.

## Hardware facts still needed

- Which axis is represented by the recorded 0.950 kg moving mass; actual resistance
  measurement; torque data for 42BYGH40-01 under applicable drive conditions.
- Confirm driver sense-resistor marking (user tentatively recalls R10, 0.1 ohm),
  measured Vref, current setting and cooling. Resistance alone does not establish
  the installed Vref or current limit.
- Pen-motor running/startup current and interface rating. User confirms power is
  from the ODROID dedicated 3.3 V power pin, switched using a GPIO control signal;
  no flyback diode is fitted. User reports 10 kOhm pull-down resistors, but their
  locations and whether one is on the transistor base remain unconfirmed.
- ODROID adapter DC output rating, supply distribution and local driver decoupling.
- Operating condition and included loads for the 0.35 A supply reading.
- Actual camera capture resolution, mounting distance and dimensions. User's
  supplied specification: 8 MP USB autofocus webcam, 78 degree field of view,
  CMOS IMX179, 3264 x 2488. This is not confirmation of software capture settings.
- Final calibrated step counts/scales if they are to be stated beyond code defaults.

The latest supplied hardware facts are incorporated into the second Design pass.
Unknown measurements remain labelled; a supplied fact is not an independent test.

## Classification evidence still needed

- Reader training counts, epochs, computer and actual historical run settings.
- Final installed checkpoint/preprocessing identity. Supplied records identify the
  selected eleven-writer `author_shape_11_pw170_weights.npz` (width 0.17, best
  epoch 43, validation 0.9935) and joint reader. Historical shipped weights were
  PyTorch-trained; the sweep used Colab T4 and a 1031/155 split. The report states
  the selected evaluation configuration without claiming older deployment defaults
  already match it. See `DESIGN_COVERAGE_AUDIT.md`.
- Matched ink/shape comparison on original and reproduced crops to demonstrate the
  design rationale; separate confidence from correctness and independent judgement.
- Matched trained-model comparison for first-principles/framework reader and writer
  models. The small archived study is not the final recognition result.
- Confirm independence of profile-building and evaluation inputs.

## Archived numerical comparison included

Producer: `Software/CleanPipeline2/testing/torchtrain.py`, `DoCheck`.
Output: `Software/CleanPipeline2/testing/results/trainvs/equivalence.json`.
The producer was inspected, but the experiment was not rerun in this revision.

Two random inputs of size 1×64×640; float64; matching initialized parameters;
text CTC and 13-class writer cross-entropy; dropout disabled for writer comparison.
`forward` is max absolute logit difference divided by the greater max magnitude
(floor 1e-8); `loss` is absolute loss difference; `worstGrad` ranks relative L2
gradient differences excluding tensors whose max gradient magnitude is below 1e-6.
The archive keeps those excluded tensors for inspection.

These details are stated in the draft so the table cannot be mistaken for a
final-model classification result. Relevance to the final submitted revision
still requires confirmation.

## Reasons for remaining limitations

The draft identifies remedies where supported by the design, but does not invent
why they were not implemented. Ask the user for the actual constraint when filling
those passages: data availability, measured trade-off, hardware constraints or
time may be relevant, but none should be asserted by default.
