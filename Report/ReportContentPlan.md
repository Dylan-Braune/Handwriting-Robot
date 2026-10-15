# Report Content Plan — Handwriting Robot (Part 3, target 68-70pg)

Synthesizes: REPORT_REQUIREMENTS.md (official rules), ProjectHistoryDigest.md (what's
actually built/measured), PersonalSummaryOfWhatIWantInMyReport.docx (your content wishlist),
and the approved Project Proposal (read in full this session — reqs/specs below are verbatim
from it). Use this as the single source of truth for section-by-section content before writing.

---

## Proposal requirements (verbatim, for Table 1 / Table 2 / QTP mapping)

| # | Core mission requirement | Target spec | Demonstration method (proposal's own words) |
|---|---|---|---|
| 1 | Reproduce digital text on paper in any trained writer's style | 85% char accuracy, 85% stylistic correlation to chosen style | Sentence reproduced in a user's style, >=85% chars visibly match |
| 2 | Decode handwritten text content + identify which trained writer wrote it | 95% overall classification accuracy | Predefined paragraph classified, text+writer match >=95% |
| 3 | Reproduce text at near real-time, practical speed | <=4 sec/character average (0-1% error budget) | x-character sentence reproduced within 4*x seconds |
| 4 | Identify writer + decode content in short practical time | <=25ms/character + 2s extra for image capture/conditioning | Page with x chars classified within 2+x*0.025s |
| 5 | Gantry precise pen movement, minimal positional error | <=0.5mm absolute positional error | Same two words redrawn twice, overlay, error <=0.5mm |
| 6 | Train/store >=10 writing styles permanently | Minimum 10 users' trajectory maps stored permanently | Display shows user count; any of the 10 reproducible on demand |

**Field conditions** (Section 5 of proposal):
1. (required) Well-lit lab/office, 400-1000 lux, minimal image-capture noise.
2. (optional, included) Stable, level, vibration-free surface for scanning + gantry writing.

**Student tasks (6.1, 8 bullets)** — map 1:1 to Table 1 rows: NN research (types/structure/layers/
feature extraction) -> image conditioning/noise removal/char separation/trajectory mapping research
-> CNN + image conditioning + char separation design/implementation -> classifier simulation on
existing datasets -> page-capture rig construction + NN/algorithms on SBC -> SBC testing with
captured images -> gantry CAD design/construction -> pen trajectory + stepper control design/
implementation -> full gantry+classifier integration testing.

**New knowledge (6.2)**: NN training/classification of character images; feature extraction from NN
layers; image processing for character isolation; pen trajectory map creation; 2D gantry design with
precise motor control; NN implementation on an SBC (not a PC).

**Proposal's own references** (already approved, carry into lit review + extend):
[1] Babushkin et al., "Analyzing handwriting legibility through hand kinematics," Frontiers in AI, 2025.
[2] Agduk & Aydemir, "Classification of Handwritten Text Signatures by Person and Gender," Acta
    Informatica Pragensia, 2022.
[3] Teikari, "Reproducing Handwriting with Position-Controlled Robots," MSc thesis, Tampere, 2016.
[4] Huang & Xiong, "Cost-Effective Robotic Handwriting System with AI Integration," IEEE LISAT, 2024.

---

## 1. Literature study (target 6-7pg; need >=6 journal refs total — have 4, need 2+ more)

Extend (not replace) the Sem1 lit review. Sem1 covered the problem domain generally; this needs to
also cover **every higher-level method actually used in the code**, each with 1-2 researched
alternatives (survey only — do not pick a winner here, that happens in Approach/Design).

Structure:
- **1.1 Background and context** (4-5pg), subheadings:
  - 1.1.1 Handwriting legibility/style and the case for automated reproduction (Sem1 base + [1])
  - 1.1.2 Handwritten text segmentation — classical approaches (projection profiling, connected
    components, seam carving) vs this project's DP-seam approach. Cite a segmentation survey +
    Otsu's original thresholding paper + Zhang-Suen skeletonization paper.
  - 1.1.3 Neural network architectures for character/writer classification — CNNs for image
    classification generally, + [2] (writer classification specifically) + an alternative
    (e.g. SVM-on-handcrafted-features, or a transformer-based classifier) as a surveyed alternative.
  - 1.1.4 Sequence modelling for text recognition — CTC loss + BiLSTM (cite **Graves et al. 2006,
    "Connectionist Temporal Classification"** — the actual paper behind `CtcForcedAlign`/greedy CTC
    decode used in this project) vs attention-based/transformer OCR as an alternative.
  - 1.1.5 Handwriting synthesis / glyph-based reproduction methods — glyph-library + forced-alignment
    extraction (this project's method) vs generative approaches: RNN-based online handwriting
    generation (Graves 2013, "Generating Sequences With Recurrent Neural Networks" — directly
    relevant, cite even though the project's own earlier RNN attempt was abandoned, as a surveyed
    alternative) vs diffusion-based handwriting synthesis (cite a DiffBrush/diffusion-handwriting
    paper — relevant since this was investigated and deliberately not used, a legitimate "considered
    alternative" to discuss). Mention "anchor methods" (anchor-based glyph positioning/exemplar
    matching) as another surveyed alternative per your note.
  - 1.1.6 Robotic/gantry handwriting reproduction — [3], [4], prior project context if this project
    builds on anything (state explicitly per Preamble rule — confirm with study leader whether any
    prior UP project is a direct predecessor; if genuinely novel, say so).
  - 1.1.7 Precision 2D gantry motion control — stepper motor control, Bresenham line interpolation
    for motion planning (cite Bresenham's original paper — directly used in `odroid_direct_drive.py`).
- **1.2 Summary of how this background was applied** (1-2pg) — explicit bridge: "the DP-seam approach
  in 1.1.2 was selected over X because...", "the CNN+CTC approach in 1.1.3/1.1.4 was selected over
  transformer OCR because [practical: SBC compute budget]...", etc. This is where you may now state
  preferences (lit review body itself stays neutral/surveying per your instruction).

**Reference count check**: proposal's 4 + Otsu + Zhang-Suen + Graves CTC + Graves RNN-synthesis +
Bresenham + diffusion-handwriting paper + a segmentation survey + a CNN-classification survey =
12-13 total, comfortably over the 6-journal-article minimum. Prioritize IEEE/journal sources for the
"6 journal articles" requirement specifically (Graves 2006 is a journal paper eligible; Bresenham is
a conference paper — fine as a non-journal technical reference).

---

## 2. Approach (target 1.5-2pg)

- One-paragraph storyline: problem split into 4 functional blocks (A: image processing &
  segmentation, B: training & classification, C: text reproduction & G-code, D: gantry hardware) —
  map directly onto proposal's FU1-FU4 functional block diagram (FU1 page-scanning, FU2 SBC
  processing, FU3 reproduction gantry, FU4 power).
- Concept alternatives considered and rejected, with reasons (this is where the diffusion-model
  route belongs — tried, worked, but rejected for violating the from-scratch/library-free project
  scope, not because it performed worse; be honest about *why* rejected, it's a legitimate design
  decision to defend).
- Why SegmentLean (DP-seam, from-scratch) was chosen over classical connected-component-only
  segmentation: measured, cite the "express" bug + fix from the digest as evidence of why a smarter
  seam approach was needed over naive row-cutting.
- Why glyph-library + forced-alignment + best-of-N joint scoring was chosen over an end-to-end
  generative model: controllability (exact text fidelity, required by spec 85% char accuracy) vs.
  a generative model's harder-to-guarantee text fidelity.
- High-level block diagram here can be simpler/conceptual (the detailed one is compulsory in 3.2).

---

## 3. Design and implementation (target 32-38pg — the core, most page budget here)

### 3.1 Design summary (compulsory, exact heading/sentence per REPORT_REQUIREMENTS, Table 1)
Table 1 must have one row per: (a) every deliverable in Proposal Section 4 (the 6 requirements'
"5. Your own design contribution" cells) AND (b) every task in Proposal Section 6.1 (8 bullets).
Draft rows (first-principles vs toolbox flagged honestly per digest Section 5 framing):

| Deliverable/task | Implementation (first-principles vs toolbox, be honest) | Status |
|---|---|---|
| Image conditioning + char/line separation from scratch (req1,2,4; task bullets 2-3) | `SegmentLean.py`: from-scratch numpy/PIL Otsu threshold, deskew search, DP-seam line segmentation, connected-component ownership rules. No cv2/skimage. | Completed. Sec 3.x |
| CNN classifier design+training for writer/text ID (req1,2,4; tasks 1,3) | `TrainText.py`/`TrainAuthor.py`: from-scratch-architecture CNN+BiLSTM+CTC, trained via PyTorch (training explicitly out of library-free scope per project rule — state this). Inference at runtime is from-scratch numpy (`np_inference/`), zero torch. | Completed. Sec 3.x |
| Classifier simulation/validation on existing datasets (task 4) | IAM dataset handling via `ExtractIAMLines.py`, validation accuracy reported per Results Sec 4. | Completed. Sec 4.x |
| Page-capture rig construction + NN on SBC (task 5) | Physical scanning rig built; `np_inference/` runs natively on ODROID SBC (no PC/GPU dependency), verified live boot. | Completed. Sec 3.x |
| SBC testing with captured images (task 6) | Live `/api/classify` tested against real photographed pages on the SBC. | Completed. Sec 4.x |
| Gantry CAD design + construction (task 7) | Physical 2D gantry built (state CAD tool used, if any — toolbox-assisted, flag honestly). | Completed/Incomplete — confirm current hardware status. Sec 3.x |
| Pen trajectory map + stepper motor control design/implementation (req3,5; task 8) | Glyph trajectory extraction (`BuildStyleProfile.py`, forced alignment) -> G-code (`WriteGCode.py`, first-principles) -> `odroid_direct_drive.py` (from-scratch Bresenham interpolation + bit-banged GPIO stepper control, no motor-control library). | Completed. Sec 3.x |
| Gantry+classifier full-system integration testing (req5,6; task 8) | End-to-end `/api/generate` pipeline tested; see QTPs. | Completed/Incomplete per latest gantry calibration status — **confirm with real measured mm-accuracy before writing this row's status.** |

(Confirm actual "Completed"/"Incomplete" wording and exact section numbers once the report is
drafted — never guess a status, must be true on submission day.)

### 3.2 System block diagram (compulsory)
Full current system diagram (hardware+software), descended from proposal's FU1-FU4 diagram but
reflecting the *actual* final implementation (e.g. `np_inference/` boundary, `ProfileIO.py` split,
`authors_config.py`, Flask server + web frontend as the real UI instead of a generic "display").

### 3.3+ Your own subsections (suggested order, matching the 4 functional blocks + personal summary)

**3.3 Image processing and segmentation (from scratch)**
- Maths: Otsu threshold derivation, lighting-normalisation (divide-by-blurred-background), deskew
  search objective function, DP shortest-path seam cost formulation, connected-component row-run +
  union-find labelling algorithm.
- Flow diagram of the 6 SegmentLean stages (digest Section 1A).
- Evidence of *why* each stage matters: before/after images for lighting normalisation, deskew,
  seam vs naive row-cut (use the "express" bug as a worked example of seam cost-map necessity).
- Honest scope note: `BuildStyleProfile.py`'s personal-glyph extraction still uses the older
  `SegmentPage.py`, not `SegmentLean` — mention as a design note here (ties to Discussion 5.1.3).

**3.4 Classifier design (writer-ID + text recognition)**
- Architecture diagram (conv stages, pooling, BiLSTM, heads) with maths: conv2d (im2col formulation),
  batchnorm (eval-mode affine from running stats), max/avg pooling, forward-prop equations,
  BiLSTM gate equations, backprop/gradient descent (brief, standard — don't over-credit, flag as
  "standard ML technique," per the "no credit for basic textbook content" rule — keep this part
  short, focus credit-earning detail on what's novel: the from-scratch numpy *inference* re-
  implementation).
- Two classifiers explained separately: ink classifier (raw photographed lines) vs shape classifier
  (stroke-normalised — explain `StrokeNormalize`: binarize->skeletonize->re-ink constant width, with
  the Zhang-Suen maths here).
- **The frozen-backbone bug (digest 2.5) belongs here as a worked design narrative**: what frozen-vs
  fine-tuned training means, why it broke Owen's classification specifically, the confusion-matrix
  evidence, the fix, and the measured improvement table. This is rich, genuinely first-principles
  content — don't compress it to one line.
- `np_inference/` section: explain the custom `.pt` pickle/zip parser (no torch import) and why this
  was necessary for the library-free inference requirement — strong first-principles credit claim.

**3.5 Feature extraction, glyph extraction and forced alignment**
- CTC forced alignment maths (Viterbi-style numpy alignment — `CtcForcedAlign`), explain what
  "forced alignment" buys you (known-text-to-pixel-boundary mapping) vs plain CTC decoding.
- Douglas-Peucker polyline simplification maths + the closed-curve bug (digest 2.3) as a full worked
  example: the geometric reason it fails (chord length -> 0), the fix (split at farthest point), and
  the before/after fragment-rate table. Strong, concrete, well-evidenced content.
- Fragment-length gating (digest 2.4) — per-character relative thresholding maths + the dotless-`i`
  evidence, tied forward to style notes (3b in digest) so examiners don't think it's a bug you missed.

**3.6 Text reproduction and G-code generation**
- `SynthesizeJointBestOf`: best-of-N candidate generation + harmonic-mean joint scoring formula
  (write this out as an actual equation — legibility score, writer-ID confidence score, harmonic
  mean combination).
- G-code emission maths: trajectory-to-motor-command mapping, arc/line move encoding.
- Flow diagram: profile load -> candidate draws -> scoring -> best selection -> G-code emit ->
  round-trip verification (`GcodeRoundTrip`).

**3.7 Gantry hardware (CAD, circuitry, motor control)**
- CAD design images, physical photos.
- Circuit diagrams (compulsory per formatting rules — component values, min 6pt font): stepper motor
  driver wiring, ODROID pin-out, microswitch end-stop circuit, pen-up/down transistor-driven clicker
  circuit, driver current/resistor setting calculations.
- Bresenham line-interpolation maths (from `odroid_direct_drive.py`), calibration/homing sequence
  description, G-code command parser (state machine for G0/G1/G2/G3/G4/G20/G21/G90/G91/M2/M3/M5/M30).
- Motor driver current/voltage design equations (whatever was actually calculated for the stepper
  driver — pull from your own hardware notes/build).

**3.8 Software architecture / web interface**
- Flask server routes, flow of a `/api/generate` request end-to-end, frontend->backend->gantry data
  flow diagram. Short code snippets as figures only (full code is the separate AMS submission —
  state this explicitly per the formatting rule).

---

## 4. Results (target 12-15pg)

### 4.1 Summary of results achieved (compulsory Table 2)
Two bold group rows: "Core mission requirements and specifications" (6 rows, reqs 1-6 above) and
"Field condition requirements and specifications" (2 rows). Draft actual-outcome column from
digest's real measured numbers — **do not invent numbers not in the digest**; where a spec's real
measurement isn't in the digest yet (gantry 0.5mm positional error test, 10-vs-20-author storage
proof, per-platform timing), flag as "measurement pending" rather than guessing:

| Requirement | Actual outcome (cite digest measurement) | 
|---|---|
| 1. 85% char acc + 85% style correlation | Matched-reference: synth char 81.6% vs real 89.0% (digest 3.5); per-author connectedness-stratified: Dylan/Owen/Yeukita (print-like) 96-98% char (digest 3.7) — **spec met for print-like styles, not fully met in pooled average; report both numbers honestly** |
| 2. 95% writer+text classification | Ink classifier 99.4-99.8% (digest B); shape classifier 93.7% post-fix (digest 2.5/3.3); official end-to-end `Evaluate.py rewrite` writer-ID 80.1% (digest 3.4) — **three different numbers depending on what's measured; must clearly state which pipeline stage each refers to** |
| 3. <=4s/char reproduction | ~1.5-2s/candidate draw, ~45-50s/line at production nTries=30 (digest C) — **convert to sec/char using known avg chars/line and compare directly against the 4s/char spec; likely needs an explicit pass/fail statement with nTries tradeoff discussion** |
| 4. <=25ms/char + 2s capture | Segmentation ~5.65s/page (digest A); classification not yet isolated to ms/char — **needs a dedicated timing measurement before this report can honestly fill this row (ties to old plan Task 5, not yet done)** |
| 5. <=0.5mm gantry positional error | **Not found in digest — needs an actual repeat-write/shape-draw calibration test before this row can be filled truthfully** |
| 6. 10 users stored permanently | 5 personal + IAM dataset authors trained, profiles stored (digest shows 13 authors in some measurements, 5 personal) — **confirm exact count of permanently stored, reproducible profiles before writing this row** |
| Field 1: 400-1000 lux | Not explicitly measured in digest — state qualitatively (lab conditions used) or add a quick lux-meter check |
| Field 2: stable/vibration-free surface | Qualitative — describe the physical setup used |

### 4.2 Qualification tests (5-6 tests, each with all 7 sub-subheadings, priority order = req 1-6 order)

1. **QTP1 — Reproduction accuracy and stylistic correlation vs connectedness** (covers req1).
   4 conditions: seen-sentence x {print-style author, cursive author} and unseen-sentence x {same
   two}. Use the digest's connectedness-stratified table (3.6/3.7) directly as "Results or
   measurements." Objectively note word-boundary accuracy gap (digest 3.5, -23.8pp) under
   Observations (no interpretation yet — save for Discussion).
2. **QTP2 — Writer + text classification accuracy** (covers req2). Sub-tests: known vs novel
   sentence, dataset (IAM) vs personal authors, cursive vs print-style authors, ink-head vs
   shape-head. Use digest 3.2/3.3/3.4 tables.
3. **QTP3 — Reproduction speed** (covers req3). Measure sec/char at varying nTries, ideally across
   ODROID vs laptop/desktop if hardware access allows (per your wishlist) — if only ODROID timing
   exists, present that and note the comparison as future work if not measured.
4. **QTP4 — Capture + classification speed** (covers req4). Needs the dedicated per-stage timing
   instrumentation (old plan's Task 5) — do this benchmarking work before the report, not in it.
5. **QTP5 — Gantry positional accuracy** (covers req5). Repeat-write-same-sentence overlay test +
   geometric shape (square/circle) draw test, both per your note that a shape test alone risks only
   testing a subsystem — use both and state this limitation explicitly under Observations.
6. **QTP6 — Multi-author storage and scalability** (covers req6). Demonstrate 10 (and optionally 20,
   per your "maybe also try 20" note) trained/stored profiles, each individually reproducible and
   classifiable on demand; stretch-test with 20 to show margin/degradation if any.

**Test equipment developed**: if a custom calibration jig/measurement rig was built (e.g. for the
0.5mm positional test), this may warrant its own "4.2 Test equipment developed" subsection ahead of
the QTPs, per the formatting rule — decide once you know what physical test setup you'll actually use.

---

## 5. Discussion (target 7-8pg — heavily weighted, ECSA GA9)

### 5.1 Critical evaluation
- **Interpretation of results**: explain the word-accuracy-vs-char-accuracy gap (digest 3.5) as
  evidence errors concentrate at word boundaries/spacing, not letterforms; explain the
  connectedness-correlation (digest 3.6/3.7) as the core evidence for why cursive was explicitly
  scoped as a stretch goal (quote the proposal's own spec target of 85% and show it's met for the
  print-style authors that are the actual target).
- **Critical evaluation**: honestly assess the DP-seam/forced-alignment design choice vs the
  rejected generative/diffusion alternative (Approach section) in hindsight — was first-principles
  glyph-library reproduction the right call given the measured ~82% char / ~49% word accuracy?
- **Unsolved problems** (use digest Section 4 verbatim, don't soften):
  1. `BuildStyleProfile.py` personal-glyph extraction still on `SegmentPage` not `SegmentLean` —
     explain measured impact (char 94.5%->98.2%, word 81.8%->92.1% elsewhere) as the expected gain
     if fixed, and why it wasn't done (time budget).
  2. Owen's residual shape-classifier confusion with Yeukita, and the two disagreeing
     re-measurements (99.3% vs 67.6%) — present both, state the measurement-protocol ambiguity
     honestly, do not pick a winner.
  3. Abhinav's synthesis writer-ID tension (official 16.7%-ish vs N-sweep showing 100% at N=15-20)
     — present as a genuinely open research question, two hypotheses (sentence-specific difficulty,
     seed sensitivity), neither confirmed.
  4. Gantry positional accuracy and per-stage timing (reqs 4,5) not yet isolated/measured — state
     plainly if the report goes to print before these are done.
- **Strong points**: first-principles `np_inference/` (verified zero-torch at runtime), the DP
  closed-curve bug fix and fragment-gate fix (both with strong quantitative before/after evidence),
  the frozen-backbone diagnosis-and-fix (a genuinely non-trivial ML debugging result), SegmentLean's
  ablation-verified minimal dependency footprint.
- **Expected failure conditions**: heavy cursive/high-connectedness styles (Thiya, 0.97) degrade
  to ~79% char / ~33-38% word — state this is a graceful degradation (text mostly legible, not a
  crash) correlated continuously with connectedness, not a cliff.

### 5.2 Considerations
- **Ergonomics**: web frontend + author dropdown for non-technical operation; physical rig loading
  ease (describe actual paper-loading/pen-mounting UX).
- **Health and safety**: moving gantry pinch points, stepper motor driver heat, mains-isolation via
  external PSU — state what guarding/precautions exist, or that none were needed at this scale.
- **Environmental impact**: minimal — low-power SBC + small steppers; note end-of-life
  electronics/e-waste briefly; no consumables beyond paper/ink.
- **Social and legal impact**: assistive-technology framing (helps people who can't write neatly/at
  all produce authentic-looking handwritten correspondence); data-privacy note (personal handwriting
  samples are biometric-adjacent — state how training data/consent was handled); no copyright issue
  (reproduces the user's own style, not another's copyrighted handwriting without consent).
- **Ethics clearance**: state explicitly whether UP ethics clearance was required/obtained for
  collecting personal handwriting samples from the 5 named contributors — **confirm actual status
  before writing this, do not guess; if no formal clearance was sought, say so and justify (informed,
  consenting participants, no vulnerable population, no sensitive data category).**

---

## 6. Conclusion (target 2-3pg, third person, no personal narrative)
- **6.1**: factual summary — 4-block pipeline built and integrated end-to-end on an SBC; literature
  survey completed; from-scratch segmentation, forced-alignment glyph extraction, numpy inference,
  and gantry control implemented; system demonstrated on 5 personal + IAM dataset authors.
- **6.2**: headline numbers — connectedness-correlated reproduction accuracy, classifier accuracy
  after the frozen-backbone fix, the matched-reference word-accuracy gap as the main remaining
  weakness, honest statement on which proposal specs are met vs not yet verified.
- **6.3**: contribution — name genuinely new knowledge per proposal 6.2 (NN training/classification
  from scratch, feature extraction via NN layers, SBC-native numpy inference, forced alignment,
  gantry/motor control), and explicitly separate PyTorch-assisted training from the library-free
  inference/segmentation/gantry work (per digest Section 5 framing) — be specific and honest about
  provenance, matching the Appendix 4 worked example's register.
- **6.4**: future work — fix the `SegmentPage`->`SegmentLean` holdout, resolve the Owen/Abhinav
  measurement discrepancies with a controlled re-measurement protocol, complete the timing/gantry
  QTPs (4,5) if not done before submission, consider extending beyond 10-20 authors, investigate the
  word-boundary accuracy gap specifically (spacing/kerning model) as the highest-leverage next step.

---

## 7. References
Target 12-15 total (comfortably over the 6-journal minimum): the proposal's 4 + Otsu + Zhang-Suen +
Graves CTC (2006) + Graves RNN handwriting generation (2013) + Bresenham + a diffusion-handwriting
synthesis paper + a handwriting/writer-classification survey + any CAD/stepper-motor-control
reference used for the hardware maths. Keep strict 1:1 citation<->reference-list correspondence.

---

## Open items to resolve before/while drafting (do not guess any of these)
1. Actual current hardware build status (gantry CAD/construction "Completed" vs "Incomplete").
2. Gantry positional accuracy (0.5mm spec) — needs a real measurement, not in the digest yet.
3. Per-stage timing (classification ms/char, capture+conditioning time) — needs dedicated
   benchmarking (ties to the old remaining-work plan's Task 5, not yet done).
4. Exact count/list of permanently stored, individually-reproducible author profiles (for the
   "10 users" requirement row).
5. Whether this project formally builds on a prior UP project (Preamble Section 2 is mandatory and
   must name one or explicitly deny it).
6. Language editor identity + signature/proof (Preamble item 3) — needed for a real submission; not
   needed for this reference-only draft.
7. Ethics clearance status for personal handwriting data collection.
8. True AI-usage and software-library declarations for whichever report is actually submitted (this
   draft is explicitly a comparison reference only, per your stated intent, not for submission).
