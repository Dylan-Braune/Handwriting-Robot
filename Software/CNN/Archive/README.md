# Archive/

Parked files that are **not** part of the current pipeline. Nothing in the
top-level folder imports anything in here. These are kept for reference /
history only -- their imports and their `sys.path` parent are now wrong
(they were written to sit one level up), so they will not run as-is.

| File | What it was | Why parked |
|---|---|---|
| `ImageCapture.py` | Webcam capture helper (OpenCV) | Standalone, never wired into a pipeline |
| `NonDatasetSegmenterCV.py` | OpenCV version of the page segmenter | Superseded by `SegmentPage.py` (first-principles numpy) |
| `NonDatasetSegmenterFP.py` | *(the OLD baseline, pre-split)* | Superseded -- current baseline is `SegmentPageCore.py` |
| `NonDatasetSegmenterTest.py` | Segmenter scoring harness | Already retired; scoring folded into the segmenters |
| `NonDatasetPreprocessing.py` | Interactive segmenter wrapper | Already retired stub |
| `match_author_style.py` | Nearest-real-line style experiment | One-off exploration |
| `train_author_fast.py` | Frozen-backbone writer-ID variant | Experiment / second-opinion model |
| `train_author_shape.py` | Shape-normalized writer-ID variant | Experiment |
| `verify_shape_style.py` | Verifier for `train_author_shape` | Depends on an archived experiment |
| `make_demo_check.py` | Demo-image generator | One-off |
| `paper_cnn_bilstm_ctc_best.pt`, `paper_cnn_bilstm_ctc_checkpoint.pt` | Stray duplicate weights | Real weights live in `NOGIT/weights/` |
| `GoodPreprocessing.zip` | Manual backup of the segmenter (Aug 2025) | Stale snapshot |
