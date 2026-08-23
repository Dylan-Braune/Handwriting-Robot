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
