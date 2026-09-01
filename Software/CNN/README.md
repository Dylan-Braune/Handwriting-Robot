# Software/CNN -- what's here and why

This folder was reorganized to separate the current best pipeline from
one-time dataset-prep tools and diagnostics, without moving anything that's
actually imported by another file (checked via grep before moving anything
-- nothing here imports across the new subfolder boundaries).

## The core pipeline (this folder, top level)

These are the files that matter for a real end-to-end run today, in the
order data flows through them:

| File | Role |
|---|---|
| `FullLineBoxMaker.py` | Line segmentation for **IAM dataset** pages (`ExtractLinePatches`). Used by both trainers below and by `classify_page.py` when `is_dataset=True`. |
| `NonDatasetSegmenterFP.py` / `NonDatasetSegmenterCV.py` | Line + MESS-block segmentation for **your own photographed pages** (not IAM) -- first-principles numpy vs. OpenCV versions of the same algorithm. `NonDatasetSegmenterTest.py` is the strict scoring harness for both. `fp_ops.py` is the numpy image-op library the FP version is built on. |
| `NonDatasetPreprocessing.py` | Thin interactive wrapper around `NonDatasetSegmenterFP.py` for testing segmentation on the sample pages in `NOGIT/NonDatasetImages/`. |
| `train_paper_cnn_bilstm_ctc.py` | **Text recognition** trainer -- reproduces Kizilirmak & Yanikoglu's CNN-BiLSTM-CTC architecture (arXiv:2307.00664). Trains on the full IAM dataset. This is the file to treat as "the" text model; don't fork it, extend it in place. |
| `train_author_classifier.py` | **Writer identification** trainer -- NEW, added to replace the old joint model (see "What happened to the old author classifier" below). Trains only on `Data/Datasets/IAMpages10` (writers 150/151/152/153/154/155/384/551/552/588). Imports from `train_paper_cnn_bilstm_ctc.py` rather than duplicating it, and can optionally initialize its backbone from that file's trained weights (transfer learning). |
| `classify_page.py` | The actual deployment/inference entry point -- loads trained weights, transcribes a page (IAM or personal), and can save line-crop/model-input previews. Fully interactive, no CLI flags. |

Weights live in `NOGIT/weights/` (gitignored -- large binary files). Line
image caches (`line_cache_raw/`, `NOGIT/line_cache_authors10/`) are also
gitignored; both trainers rebuild them automatically on first run against a
given dataset path.

## What happened to the old author classifier

You were right that one existed. `train_handwriting_robot_v1_baseline.py`
had a `MultiTaskLineCRNN` with both a text head and an `author_head`,
trained jointly. It was deleted from the working tree in commit
`63ba8d9` ("FolderCleanupNOGIT") -- still recoverable via
`git show 63ba8d9~1:Software/CNN/OldCode/train_handwriting_robot_v1_baseline.py`
if you ever want to look at it again, but it isn't a file in this folder
any more.

`logs/train_10author.log` has that old joint model's actual results on the
same 10 IAM authors `train_author_classifier.py` now targets: **author-ID
accuracy reached 98.9-100% by around epoch 20 and stayed there for the rest
of the 200-epoch run**, while text accuracy stalled at **~28-29%
char-accuracy (70% CER)** the whole time, using the OLDER (pytesseract-
labelled, periodicity-segmented) line data.

`train_author_classifier.py` is the replacement: a separate, focused model
instead of a second head bolted onto the text model, trained on the CURRENT
line segmentation + label alignment (`DatasetPrep/regenerate_labels_with_
alignment.py`'s output), with an optional transfer-learning init from the
already-trained text model's backbone. Given writer-ID over only 10 classes
was already trivial for the old, weaker architecture, this should reach
similarly high accuracy quickly -- **run it and the real number will print
at the end of training**; nothing here should be taken as a claimed result
until you've actually run it, since this file has not been executed yet
(no torch in the dev sandbox this was written in).

To get the CURRENT text model's actual validation numbers (not found in any
log file during this reorganization -- `full_training.log`/`full_training_
v2.log` turned out to be from the old joint model and a label-alignment
run, not a completed `train_paper_cnn_bilstm_ctc.py` run), use
`Diagnostics/inspect_checkpoint.py` against `NOGIT/weights/paper_cnn_
bilstm_ctc_best.pt` -- it prints exactly the epoch/CER/char-acc stored
inside the checkpoint.

## DatasetPrep/

One-time (or re-run-when-the-dataset-changes) tools that produced the
`_labels.txt` files the trainers currently read. Not part of the live
training/inference path -- you'd only touch these again if you added new
pages or wanted to regenerate labels with a different alignment strategy.

- `generate_boxes_and_labels.py`, `generate_first50_lastpage.py` -- generate
  preview boxes + label files for chosen author folders.
- `regenerate_labels_with_alignment.py` -- the write-enabled version that
  actually overwrote your real dataset's labels using the width-based DP
  alignment method (~89% exact-line match on folder 150, per its own
  docstring).
- `scan_label_lengths.py` -- flags suspiciously long line labels (usually a
  sign of collapsed/merged line segmentation).

## Diagnostics/

Scripts you run to check the training pipeline is doing what you think it's
doing, not scripts that produce a deliverable themselves.

- `verify_training_pipeline.py` -- proves images and labels are actually
  paired correctly before trusting a loss number.
- `export_holdout_pages.py` -- copies held-out validation pages to their own
  folder so you can eyeball predictions on pages the model never trained on.
- `inspect_checkpoint.py` -- prints a checkpoint's stored metadata (see
  above -- this is how to get the real current text-model numbers).

## Loose ends noticed during this reorganization, not touched

- `GoodPreprocessing.zip` -- a manual backup of the segmenter files from
  Aug 19, before the pixel-completeness fixes made on Aug 20 (see
  `NonDatasetSegmenterFP.py`'s `HysteresisRecoverInk`/`RenderLine`). Now
  stale; safe to delete yourself once you've confirmed the current
  `NonDatasetSegmenterFP.py` is what you want to keep.
- `photo_8mp.jpg` -- a loose test photo, presumably from `ImageCapture.py`.
- `paper_cnn_bilstm_ctc_best.pt` / `paper_cnn_bilstm_ctc_checkpoint.pt` at
  this folder's TOP level -- these are stray duplicates. The real ones the
  code actually reads/writes live under `NOGIT/weights/`. These top-level
  copies aren't referenced by any script and were most likely saved here by
  an older version of the training script before it was pointed at
  `NOGIT/weights/`.

None of the above were deleted -- files in the synced project folder can't
be removed by the assistant, only overwritten, so these are flagged here for
you to clean up by hand rather than left unmentioned.

## The writing side: text -> that author's handwriting -> gantry motion

Added alongside the reading pipeline above; none of it touches
`train_paper_cnn_bilstm_ctc.py`, its weights, or how `classify_page.py`
reads text. The text recognizer is used here **for inference only**, as a
forced-alignment tool that says where each character sits inside a line.

| File | Role |
|---|---|
| `style_profile.py` | Builds a per-author **style profile** from `Data/Datasets/IAMpages10`: CTC forced alignment gives per-character x-spans, the ink is skeletonized (Zhang-Suen) to centerlines and traced into ordered polylines, and each character is stored as several **glyph variants** in a slant-removed baseline/x-height frame -- plus measured style parameters (slant, x-height, ascender/descender, letter advances, word spacing, stroke width, connectedness). Profiles are fitted on **non-holdout pages only** and saved to `NOGIT/StyleProfiles10/<author>.json`. |
| `synthesize_handwriting.py` | Arbitrary text + a profile -> a **pen trajectory** (ordered polylines with pen-up/pen-down structure). Composes from the glyph library for unseen words, adds cursive ligatures for connected hands, and applies per-instance jitter and baseline drift so repeated text isn't stamped. |
| `evaluate_style.py` | Measures the style claim: renders synthesized lines and asks `train_author_classifier.py`'s 10-author model who wrote them (**target: >= 85%**), compares feature distributions (slant / spacing / ink density) against the real hand, builds the real-vs-synth comparison sheet, and estimates the digital->physical loss with `MachineDistort` (microstep quantization + belt backlash + positioning noise). |
| `gcode_writer.py` | Trajectory -> **G-code**, a **step/direction schedule**, and a plotter preview; plus `SimulateAndCompare`, which rasterizes the emitted G-code and checks it against the synthesized trajectory. |
| `write_as_author.py` | **End-to-end entry point**: text + author -> profile -> trajectory -> preview + G-code + steps + simulation check. |

### Where the machine calibration lives

**All machine constants are in one place: the `GantryConfig` dataclass at
the top of `gcode_writer.py`.** Nothing else in the writing pipeline
hardcodes a hardware number.

* **X/Y steps per mm** are *derived*, not hardcoded: NEMA 17
  `fullStepsPerRev=200` x A4988 `microstepping=16` = 3200 microsteps/rev,
  divided by `mmPerRevX` / `mmPerRevY`. Set only the mm-per-revolution of
  your transmission -- GT2 2 mm belt on a 20-tooth pulley = 40 mm/rev
  (the default, giving 80 steps/mm); for a leadscrew set it to the lead.
* **Work area / origin**: `boundsMin/MaxX/Ymm`, `originXmm`, `originYmm`.
  Anything outside is clamped and the G-code header says how many points
  were clamped.
* **Speeds**: `drawFeedMmMin`, `travelFeedMmMin`, `accelMmS2` (used for the
  trapezoidal timing in the step schedule).
* **Pen (one-way gear motor, no driver)**: the motor only turns one way and
  every 90 degrees toggles the pen, so pen state cannot be commanded or
  read back -- `PenController` tracks it in software and emits exactly one
  90-degree pulse per state *change*. `penPulseMs` is how long your motor
  takes to turn 90 degrees (measure once); `penUpCode` / `penDownCode` are
  the M-codes your firmware should map to that pulse; `penStartsUp` is the
  assumed power-on state.
