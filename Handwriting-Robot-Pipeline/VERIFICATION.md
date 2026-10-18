# Clean pipeline copy — verification record

This folder is a self-contained copy of the handwriting pipeline. Nothing in the
original repository was modified.

## What was copied

| Item | Contents |
|---|---|
| `Software/CNN` | 12 pipeline modules + the 2 trainers + `ExtractIAMLines.py` |
| `Software/CNN/AuthorReproductionStuff` | `server.py`, `BuildStyleProfile.py`, `Evaluate.py`, `ProfileIO.py`, `SynthesizeHandwriting.py`, `web_render_helpers.py`, `camera_capture.py` |
| `Software/CNN/AuthorReproductionStuff/np_inference` | the 4 from-scratch numpy inference modules |
| `Software/GantryControl` | `odroid_direct_drive.py` |
| `Software/CNN/weights` | all 7 checkpoints |
| `Software/CNN/NOGIT/StyleProfiles10` | the 13 author style profiles |
| `Software/CNN/PersonalDataset` | the personal-writer pages and labels |
| `Software/CNN/Crops` | line crops + cut overlays |
| `Data/Datasets/IAMpages10` | the IAM pages the offline pipeline trains on |
| `Verification/` | 6 verification scripts + this record |

Deliberately excluded: `NOGIT` build caches (regenerable), the legacy segmenters
(`SegmentPage`, `SegmentHybrid`, `SegmentVariants`, `SegmentBubbles`,
`SegmentPaper`, `SegmentSimple`, `SegmentSpine`), `ReproductionImprovement`,
`StyleSynthesis`, `TestsForReport`, `DatasetPrep`, and the report/analysis trees.

## Change made during copying

`BuildStyleProfile.py` and `TrainAuthor.py` both imported the legacy
`SegmentPage` module at run time — the last two live dependencies on it.

- `BuildStyleProfile.build_personal_line_items` now calls
  `SegmentLean.SegmentLines` instead of `SegmentPage.ProcessPage`, so a personal
  author's training crops come from the same segmenter that serves run-time
  requests. The old TEXT/MESS tag filter is gone because SegmentLean crops are
  already line-level.
- `BuildStyleProfile`'s `LabelComponents` use now comes from `ProfileIO`.
- `TrainAuthor.StrokeNormalize` now takes `Dilate` from `ProfileIO` instead of
  `SegmentPage`. This is the function the shape classifier and the live server
  both call, so it had to move off SegmentPage as well.

No other logic was changed.

## Verification results (all run in this folder)

| Check | Result | Matches original |
|---|---|---|
| Segmentation, 30 pages | 669 lines, 216 s | yes (669) |
| Text recognition, 669 crops | char 98.614 %, word 94.205 % | yes, identical |
| Writer-ID, ink head, 1358 lines | 99.8 % | yes |
| Writer-ID, shape head, 1358 lines | 93.7 % | yes |
| numpy vs torch agreement | 100.00 % top-1, max diff 5.7e-06 | yes |
| End-to-end, 13 writers, G-code | writer-ID 92 %, char 79 %, word 44 % | yes |
| G-code round trip, 13 writers | writer-ID 90 %, ink control 28 %, char 79 % | yes |
| Server boot | 13 authors listed, torch never imported | yes |
| `/api/classify` one real crop | HTTP 200, correct writer | yes |
| `/api/generate` "test" (owen) | read back exactly, 100 % | yes |
| G-code sanity | M3 == M5, ends with M2 | yes |

## Running things

    # server
    cd Software/CNN/AuthorReproductionStuff && python server.py

    # verification (from this folder)
    python Verification/run_full_test.py
    python Verification/verify_text_accuracy.py
    python Verification/verify_author_classification.py
    python Verification/verify_gcode_v2.py
    python Verification/verify_end_to_end_fast.py
    python Verification/verify_server_api.py

Each verification script sets up its own `sys.path` from the folder it lives in,
so they run from the package root without further configuration.
