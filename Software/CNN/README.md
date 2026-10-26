# Software/CNN -- which pipeline is which, and how to test each

There are **two separate, working implementations** of the same system (read a
photographed page, recognise the text, identify the writer, and reproduce text
in a chosen writer's hand as gantry G-code). They live in different folders,
share the same trained weights and `NOGIT/` data, and do **not** import each
other. Pick whichever one you mean to test and don't mix files between them.

```
  Software/CleanPipeline/            <- B. the new, from-scratch, library-free pipeline
  Software/CNN/AuthorReproductionStuff/
  Software/CNN/{TrainText,TrainAuthor,SegmentPage,authors_config}.py
  Software/GantryControl/            <- A. the old, most recently fully working pipeline
```

## A. The old pipeline -- `AuthorReproductionStuff/` (most recently fully working)

This is the version that was actually driven end to end (camera -> classify ->
synthesise -> G-code -> physical gantry) before the library-removal rewrite
below. It still uses **PyTorch** for the two neural networks, plus a
hand-written numpy-only re-implementation of those same two networks
(`np_inference/`) that the live server actually calls for inference, so no ML
library runs at request time even though training does.

| File | Role |
|---|---|
| `TrainText.py` (top level, **not** the file of the same name in `CleanPipeline/`) | PyTorch CNN-BiLSTM-CTC trainer, 4 subcommands: `base` (segment+train on raw IAM pages), `hf`/`hf` stage (public Teklia/IAM-line set), `joint` (warm-start + weighted joint training with the personal lines), `personal` (rejected fine-tune, kept for the record). Produces the `.pt` checkpoints in `weights/`. |
| `TrainAuthor.py` (top level) | PyTorch writer-ID trainer, 3 subcommands: `original`, `10authors`, `shape` (the stroke-normalised judge). Also defines `StrokeNormalize`, still imported by `AuthorReproductionStuff/server.py`. |
| `SegmentPage.py` (top level) | Self-contained page-to-line segmenter for real photographed pages (connected components, glyph-scale estimation, line chaining, text/non-text scoring). |
| `authors_config.py` (top level) | Single source of truth for the 13-writer set (`DATASET_AUTHORS`, `PERSONAL_AUTHORS`) and the checkpoint-filename convention. |
| `AuthorReproductionStuff/BuildStyleProfile.py` | Offline, run-by-hand: CTC forced alignment (via `np_inference`) cuts each line into per-character glyphs, skeletonises and normalises them, and writes one `NOGIT/StyleProfiles10/<author>.json` per writer. |
| `AuthorReproductionStuff/ProfileIO.py` | The small slice of the above that the *live* server needs: loading a saved profile, forced alignment, binarize/skeletonise/band primitives. |
| `AuthorReproductionStuff/SynthesizeHandwriting.py` | Text + profile -> pen trajectory -> G-code/step-schedule/plotter-preview (the former `WriteGCode.py` is merged in here). Best-of-$N$ search scored by the frozen recogniser + writer-ID judge. |
| `AuthorReproductionStuff/Evaluate.py` | All four evaluation scripts merged into one: `style`, `rewrite`, `shapestyle`, `legibility` subcommands (former `EvaluateStyle.py`/`VerifyRewrite.py`/`VerifyShapeStyle.py`/`EvaluateLegibility.py`). |
| `AuthorReproductionStuff/np_inference/` | `weight_io.py` (reads a `.pt` file's tensors without importing torch), `layers.py` (conv/batchnorm/pool/linear/LSTM forward only), `text_model.py`, `author_model.py` -- the inference-only numpy re-implementation the server actually runs. |
| `AuthorReproductionStuff/camera_capture.py`, `web_render_helpers.py` | Camera grab via OpenCV; PNG rendering helpers for the server's info pages. |
| `AuthorReproductionStuff/server.py` | Flask app, imports `odroid_direct_drive` from `Software/GantryControl/` for motor control. 18 `/api/...` routes (classify, generate, camera, gcode, gantry status/calibrate, model info, glyphs, samples). |
| `Software/GantryControl/odroid_direct_drive.py` | The gantry driver this server talks to. |

### How to test the old pipeline

```bash
cd Software/CNN
python AuthorReproductionStuff/server.py          # Flask app on :5000 (set PORT/WEBAPP_API_KEY env vars)
python AuthorReproductionStuff/Evaluate.py style        # writer-ID accuracy + style-feature agreement
python AuthorReproductionStuff/Evaluate.py rewrite       # "right words, right hand" on fixed novel sentences
python AuthorReproductionStuff/Evaluate.py shapestyle    # stroke-normalised judge accuracy
python AuthorReproductionStuff/Evaluate.py legibility --sentences 24   # headline char/word legibility metric
python AuthorReproductionStuff/BuildStyleProfile.py --10authors        # rebuild all 13 style profiles
```
`TrainText.py`/`TrainAuthor.py` need PyTorch installed and are slow (this is
the path with a GPU/vectorised-autograd backend, so it is the *faster* of the
two trainers below) -- only re-run them if you actually want a new `.pt`
checkpoint; the committed weights in `weights/` already reflect a finished
run.

---

## B. The new pipeline -- `Software/CleanPipeline/`

A from-scratch rewrite with **no machine-learning, dataset or scientific-ML
library anywhere** -- only numpy/scipy (maths), PIL (image I/O), Flask/
Flask-CORS (the web server), and hardware libraries used only where they must
be (`cv2` for the camera, `gpiod` for GPIO), both imported lazily inside the
one function that needs them. Verified with `grep`: zero `import torch`
anywhere in this folder. Training, validation and deployment all call the
*same* hand-written functions, not three separate implementations.

| File | Role |
|---|---|
| `NumpyNN.py` | The core library: hand-derived **forward and backward** passes for every layer (conv2d, batchnorm2d, relu, maxpool2d, adaptive max/avg pool, linear, dropout, bidirectional multilayer LSTM, CTC loss, cross-entropy loss) plus RMSprop/AdamW and two LR schedulers. Every primitive is gradient-checked against finite differences before being trusted. |
| `TrainScratch.py` | The one training program for **both** networks (`python TrainScratch.py {text,author}`), since they share the same conv backbone; builds it, trains it and checkpoints it using only `NumpyNN.py`. Supports `--resume` (continue from the last saved epoch after a crash) and `--warm-start PATH` (initialise from an existing `.npz` instead of random weights, to shorten the much slower CPU-only convergence versus a framework-based trainer). |
| `Classify.py` | The shared **inference** side: the same conv-backbone/LSTM/CTC forward math in eval mode, reading plain `.npz` checkpoints (no `.pt`/pickle parsing at all). Used both by `TrainScratch.py`'s own per-epoch validation and by the deployed server -- one implementation, not two to keep in sync. |
| `ConvertPtToNpz.py` | One-time, standalone: converts the old torch-trained `.pt` checkpoints in `weights/` to the exact same `.npz` format `TrainScratch.py` produces, so an old and a newly scratch-trained checkpoint are fully interchangeable. The only file in this folder that still parses the `.pt` format. |
| `TrainText.py` / `TrainAuthor.py` (**not** the torch files of the same name above) | Data-loading only: `IAMLineDatasetRaw`, `TrainEvalView`/`AuthorLabeledView`, `CollateFn`, augmentation, dataset splitting -- already pure PIL/numpy, reused as-is by `TrainScratch.py`. No model class, no torch, no training loop lives here any more. |
| `ExtractIAMLines.py` | Page-to-line segmentation for scanned IAM pages (shared by the data loaders above). |
| `SegmentLean.py` | Page-to-line segmentation for real photographed pages. |
| `Charset.py` | The 74-character set + blank, shared by training and CTC forced alignment. |
| `AuthorsConfig.py` | The 13-writer set and checkpoint-filename convention (mirrors `authors_config.py` above, kept in sync by hand). |
| `ExportTekliaLines.py` | One-time, standalone: downloads the pre-segmented Teklia/IAM-line dataset from HuggingFace. The **only** file in this whole folder that needs the `datasets` library -- not imported by anything else. |
| `ProfileIO.py` | Forced alignment + stroke-normalisation primitives the live server needs (same role as the old pipeline's file of the same name). |
| `BuildStyleProfile.py` | Offline style-profile builder (same role as the old pipeline's, rewritten against `Classify.py`/`NumpyNN.py`). |
| `SynthesizeHandwriting.py` | Text + profile -> trajectory -> G-code (same role as the old pipeline's file). |
| `Evaluate.py` | Same four evaluation subcommands as the old pipeline's `Evaluate.py`, rewired to the numpy models. |
| `Camera.py`, `Gantry.py`, `WebRenderHelpers.py` | Camera capture, ODROID GPIO gantry driver, and server image-rendering helpers -- the from-scratch equivalents of `camera_capture.py`/`odroid_direct_drive.py`/`web_render_helpers.py`. |
| `Server.py` | Flask app, same route shape as the old `server.py` (`/api/classify`, `/api/generate`, `/api/camera/...`, `/api/gantry/...`, `/api/gcode/...`, `/api/model_info`, etc.). |

### How to test the new pipeline

```bash
cd Software/CleanPipeline

# cheap smoke tests -- synthetic data, no dataset needed, run in seconds
python TrainScratch.py text --selftest
python TrainScratch.py author --selftest

# real training (slow -- no GPU, no vectorised autograd; prefer --warm-start)
python TrainScratch.py text   --source teklia --epochs 5 --warm-start ../CNN/weights/paper_cnn_bilstm_ctc_joint_best.npz
python TrainScratch.py author --epochs 5 --warm-start ../CNN/weights/author_classifier_10new_weights.npz
python TrainScratch.py text --resume     # continue a run that was interrupted

# evaluation, same 4 subcommands as the old pipeline
python Evaluate.py style
python Evaluate.py rewrite
python Evaluate.py shapestyle
python Evaluate.py legibility --sentences 24

python Server.py                          # Flask app on :5000 (same env vars as the old server)
python ConvertPtToNpz.py                  # (re-)generate .npz from the committed .pt weights, if missing
```

---

## Shared data (used by both pipelines)

- `weights/` -- tracked in git. Holds **both** formats side by side: the
  original `.pt` checkpoints (read by the old pipeline's PyTorch trainers,
  and once by `ConvertPtToNpz.py`) and the `.npz` checkpoints (read by
  `Classify.py` / the new pipeline, and by the old pipeline's `np_inference`).
  A `.pt` and its converted/retrained `.npz` counterpart are interchangeable.
- `NOGIT/` -- gitignored: `StyleProfiles10/` (the 13 built author profiles,
  shared), `GlyphCache10/`, line-image caches, web caches, run outputs. Not
  needed to start fresh -- both pipelines rebuild their own caches on first
  run, only `StyleProfiles10/` needs `BuildStyleProfile.py` run once.
- `PersonalDataset/`, `Data/Datasets/*` -- the personal-author photographs and
  public datasets; `Data/Datasets/*` is gitignored and must be transferred
  manually (e.g. to Google Drive for Colab training).

## If you're not sure which one to change

If the task is about **removing a library, or the from-scratch maths/
training itself**, work in `Software/CleanPipeline/`. If the task is about
**the one that was last driven on real hardware end to end**, that's
`AuthorReproductionStuff/` (+ `Software/GantryControl/`). The two folders are
independent; a fix in one does not apply to the other unless you port it by
hand.
