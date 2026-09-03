# Software/CNN -- pipeline overview

The top level holds only the files needed for the four real activities,
renamed so the name says what the file does. Everything else lives in
[`Archive/`](Archive/README.md).

Data flows through **three groups**:

```
  photo of a page ──► PREPROCESSING ──► line crops ──► READING ──► text / who wrote it
                                                                        │
  "some text" + author ──────────────► REWRITING ──► that hand ──► G-code for the gantry
```

---

## 1. Preprocessing -- cut a photographed page into line images

| File | Role |
|---|---|
| `SegmentPage.py` | Current line + MESS-block segmenter for **your own photographed pages** (not IAM). Entry point -- just run it. |
| `SegmentPageCore.py` | Baseline segmentation engine that `SegmentPage` builds on. |
| `RawImageOps.py` | Pure-numpy image-op library (threshold, morphology, projections) the segmenters and the style profiler are built on. |

IAM dataset pages use a different, simpler segmenter: `ExtractIAMLines.py`
(in the Reading group, since only the trainers and `ClassifyText` use it).

## 2. Reading -- train the models, then transcribe a page

| File | Role |
|---|---|
| `ExtractIAMLines.py` | Line segmentation for **IAM dataset** pages (`ExtractLinePatches`, `ReadLabelLines`). Shared by `TrainText` and `ClassifyText`. |
| `TrainText.py` | **Text-recognition** trainer -- CNN-BiLSTM-CTC (Kizilirmak & Yanikoglu, arXiv:2307.00664), trained on full IAM. Every other file imports its model/decoder/dataset helpers from here -- extend it in place, don't fork it. |
| `TrainAuthor.py` | **Writer-identification** trainer -- 10-author model (`AuthorClassifierCNN`) on `Data/Datasets/IAMpages10`. Imports the backbone from `TrainText`; can transfer-init from its trained weights. |
| `ClassifyText.py` | **Inference entry point** -- loads trained weights and transcribes a page (IAM or personal, picking `SegmentPage` for personal pages). Fully interactive, no CLI flags. |

Weights live in `NOGIT/weights/` (gitignored). Line caches rebuild
automatically on first run. To read the current text model's stored
validation numbers: `python Diagnostics/inspect_checkpoint.py` against
`NOGIT/weights/paper_cnn_bilstm_ctc_best.pt`.

## 3. Rewriting -- text in a chosen author's handwriting, as gantry motion

The text recognizer is used here **for inference only**, as a forced-alignment
and legibility-scoring tool. None of this touches `TrainText.py` or its weights.

| File | Role |
|---|---|
| `BuildStyleProfile.py` | Builds a per-author **style profile** from `Data/Datasets/IAMpages10`: CTC forced alignment → per-character x-spans, Zhang-Suen skeleton → ordered glyph polylines (several variants each), plus measured style params (slant, x-height, advances, word spacing, stroke width, connectedness). Fitted on non-holdout pages → `NOGIT/StyleProfiles10/<author>.json`. |
| `SynthesizeHandwriting.py` | Arbitrary text + a profile → a **pen trajectory** (ordered polylines with pen-up/pen-down structure). Cursive ligatures for connected hands, per-instance jitter + baseline drift. `SynthesizeLegible` does best-of-N scored by the frozen text reader. |
| `WriteGCode.py` | Trajectory → **G-code**, a **step/direction schedule**, and a plotter-preview raster. `SimulateAndCompare` rasterizes the emitted G-code and checks it against the trajectory. **All machine calibration is the `GantryConfig` dataclass at the top of this file** (see below). |
| `WriteAsAuthor.py` | **End-to-end entry point**: `python WriteAsAuthor.py "Hello world" 3` → profile → trajectory → preview + G-code + steps + simulation check, into `NOGIT/WriteJobs/<author>/`. |
| `VerifyRewrite.py` | "Right words, right hand?" -- synthesizes novel sentences, asks `TrainAuthor`'s model who wrote them and `TrainText`'s model what they say, both directly and through the full G-code round-trip. Also supplies the frozen-reader helpers `SynthesizeLegible` calls. |
| `EvaluateStyle.py` | Style-fidelity measurement: writer-ID accuracy on synthesized lines (target ≥ 85%), per-feature error vs the real hand, side-by-side comparison sheets. Dependency of `VerifyRewrite`. |

### Where the machine calibration lives

**All machine constants are in one place: the `GantryConfig` dataclass at the
top of `WriteGCode.py`.** Nothing else in the writing pipeline hardcodes a
hardware number.

* **X/Y steps per mm** are *derived*: NEMA 17 `fullStepsPerRev=200` ×
  A4988 `microstepping=16` = 3200 microsteps/rev, ÷ `mmPerRevX` / `mmPerRevY`.
  Set only mm-per-revolution -- GT2 2 mm belt on a 20-tooth pulley = 40 mm/rev
  (default, 80 steps/mm); for a leadscrew set it to the lead.
* **Work area / origin**: `boundsMin/MaxX/Ymm`, `originXmm`, `originYmm`.
  Anything outside is clamped and the G-code header reports how many points.
* **Speeds**: `drawFeedMmMin`, `travelFeedMmMin`, `accelMmS2` (trapezoidal
  timing in the step schedule).
* **Pen** (one-way gear motor, no driver): the motor turns one way and every
  90° toggles the pen, so pen state cannot be commanded or read back --
  `PenController` tracks it in software and emits exactly one 90° pulse per
  state *change*. `penPulseMs` = time for one 90° turn (measure once);
  `penUpCode` / `penDownCode` = the M-codes your firmware maps to that pulse;
  `penStartsUp` = assumed power-on state.

---

## Supporting folders (unchanged)

- `DatasetPrep/` -- one-time tools that generated the `_labels.txt` files the
  trainers read. Not on the live path.
- `Diagnostics/` -- scripts that check the training pipeline
  (`inspect_checkpoint.py`, `verify_training_pipeline.py`,
  `export_holdout_pages.py`).
- `Transfers/NonDatasetImages/` -- sample photographed pages + label files.
- `Archive/` -- retired / experimental files, see its own README.
- `NOGIT/` -- weights, caches, style profiles, write jobs (gitignored).

## Rename map (old → new)

| Old | New |
|---|---|
| `FullLineBoxMaker.py` | `ExtractIAMLines.py` |
| `train_paper_cnn_bilstm_ctc.py` | `TrainText.py` |
| `train_author_classifier.py` | `TrainAuthor.py` |
| `classify_page.py` | `ClassifyText.py` |
| `NonDatasetSegmenterFP2.py` | `SegmentPage.py` |
| `NonDatasetSegmenterFP.py` | `SegmentPageCore.py` |
| `fp_ops.py` | `RawImageOps.py` |
| `style_profile.py` | `BuildStyleProfile.py` |
| `synthesize_handwriting.py` | `SynthesizeHandwriting.py` |
| `gcode_writer.py` | `WriteGCode.py` |
| `write_as_author.py` | `WriteAsAuthor.py` |
| `verify_end_to_end.py` | `VerifyRewrite.py` |
| `evaluate_style.py` | `EvaluateStyle.py` |
