# Code Explained — Index

Four explainers covering the neural, segmentation, synthesis and hardware code:
what each part does, what its important variables mean, why its constants have
the values they do, and what the maths is doing.

> ### Which pipeline these describe
>
> The four blocks were written on **21 October 2026 against
> `Software/CleanPipeline`**, the first pipeline. The live system is now
> **`Software/CleanPipeline3`**, which reorganised the same code into fewer,
> larger modules under different names.
>
> **The explanations of the maths and the algorithms remain accurate** — most
> modules carried over nearly intact, several of them completely. What is out of
> date is the *file names*, the *commands*, and the fact that the v3
> segmenter has grown considerably beyond what block A covers. Use the
> translation table below to map any name in the four blocks onto the file it
> now lives in, and the "not yet covered" list to see what is missing.
>
> Checked against the live code on 2026-10-31.

| Block | File | Covers (v1 name → **live file**) |
|---|---|---|
| **A — Segmentation** | [CodeExplained_A_Segmentation.md](CodeExplained_A_Segmentation.md) | `SegmentLean.py`, `ProfileIO.py`, `MyImage.py` → **`segment.py`**; `ExtractIAMLines.py` → **`train.py`** |
| **B — Neural core** | [CodeExplained_B_TrainingClassification.md](CodeExplained_B_TrainingClassification.md) | `NumpyNN.py`, `TrainScratch.py`, `TrainText.py`, `TrainAuthor.py` → **`train.py`**; `Classify.py`, `Charset.py` → **`recognise.py`** |
| **C — Style and synthesis** | [CodeExplained_C_ReproductionGcode.md](CodeExplained_C_ReproductionGcode.md) | `BuildStyleProfile.py` → **`style.py`**; `SynthesizeHandwriting.py` → **`synthesise.py`** and **`reproduce.py`** |
| **D — Hardware and web** | [CodeExplained_D_GantryHardware.md](CodeExplained_D_GantryHardware.md) | `Gantry.py` → **`reproduce.py`**; `Server.py` → **`web/server.py`**; `Camera.py` → **`web/camera.py`**; `WebRenderHelpers.py` → **`web/render.py`** |

### Translation table, v1 module → live file

Measured by how many of each v1 module's top-level definitions survive by name.

| v1 module | Live file | Definitions carried |
|---|---|---|
| `Classify.py` | `recognise.py` | 25 of 25 |
| `Gantry.py` | `reproduce.py` | 24 of 24 |
| `TrainScratch.py` | `train.py` | 36 of 36 |
| `Camera.py` | `web/camera.py` | 6 of 6 |
| `WebRenderHelpers.py` | `web/render.py` | 4 of 4 |
| `NumpyNN.py` | `train.py` | 30 of 31 |
| `BuildStyleProfile.py` | `style.py` | 23 of 26 |
| `SynthesizeHandwriting.py` | `synthesise.py` (21), `reproduce.py` (7) | 28 of 31 |
| `ExtractIAMLines.py` | `train.py` | 10 of 12 |
| `SegmentLean.py` | `segment.py` | 9 of 12 |
| `ProfileIO.py` | `segment.py` (10), `recognise.py` (2) | 12 of 17 |
| `Server.py` | `web/server.py` | 14 of 23 |
| `MyImage.py` | `segment.py` | 7 of 12 |
| `TrainText.py` | `train.py` (11), `recognise.py` (3) | 14 of 18 |
| `AuthorsConfig.py` | — replaced by **`paths.py`** | — |
| `GradCheckNumpyNN.py` | **no successor in v3** | — |
| `ConvertPtToNpz.py`, `ExportTekliaLines.py`, `Evaluate.py` | no direct successor | — |

### Not yet covered by these four blocks

- **`segment.py` has 51 definitions; 25 of them are new since v1** and are
  described nowhere here: `Deskew`, `UprightPage`, `LineAxisScore`, `SeamCuts`,
  `LineCentres`, `RemovePrintedStructure`, `DropArtifactPieces`,
  `DropSolidCrops`, `AssignOwnership`, `PieceBounds`, `Integral`, `Otsu`,
  `RowRunLength`, `Elastic`, `PolylineLength`, and ten private helpers.
- **`recognise.py`** adds `StrokeNormalize`, `CtcForcedAlign`, `ReadTextScored`
  and `FrameXToPixel`.
- **New files with no v1 counterpart:** `paths.py` (the single definition of
  where data lives and who the writers are), `timing.py` (the stage timer),
  `curate.py`, `retrace.py`, `trainlibrary.py` (the same two networks in torch),
  `strokenorm_prepare.py` and `strokenorm_train.py`.
- `GradCheckNumpyNN.py` was **not carried into v3**, so the gradient-check
  result quoted in block B describes the v1 implementation of the same maths.

---

## How the system fits together

**The read path** — a photograph of handwriting in, text and an identity out:

```
photo
 └─ segment.SegmentLines                      (A.1)  page → line crops
     └─ recognise.PaperCRNNNumpy.forward      (B.7)  line → per-frame log-probs
         └─ recognise.DecodeCtc               (B.7)  → text
     └─ recognise.StrokeNormalize             (A.2.4) line → constant-width skeleton
         └─ recognise.AuthorClassifierCNNNumpy (B.7) → which writer
```

In the live code `segment.SegmentLines` first calls `Deskew` and
`RemovePrintedStructure`, then `LineCentres` and `SeamCuts`, none of which
block A describes. `web/server.py` stands the page on its correct axis with
`StandUp` before any of this runs.

**The write path** — a sentence and a writer in, pen motion out:

```
text + author
 └─ synthesise.SynthesizeJointBestOf        (C.3.6)  best of N by text × writer-ID
     └─ SynthesizeText                      (C.3.3)  glyphs + ligatures + jitter → polylines
         └─ profile JSON from style.py      (C.2)    glyph library + style parameters
     └─ reproduce.ToMachine / OptimizeStrokeOrder (C.3.7) → bed coordinates, reading order
     └─ reproduce.WriteGcode                (C.3.9)  → G-code
         └─ reproduce.RunGcodeFile          (D.1.7)  → step pulses
```

**The offline paths**, run by hand, that produce what those two depend on:

```
IAM / personal pages ─ train.IAMLineDatasetRaw (A.3) ─ train.py text (B.6) ─→ .npz weights
IAM / personal pages ─ segment + CTC forced alignment ─ style.py (C.2) ─→ profile JSON
```

The gradient-check script of block B.4 was not carried into the live pipeline;
the backward passes it verified now live in `train.py`.

`web/server.py` (D.2) exposes both live paths over HTTP and is what the web app talks to.

---

## The handful of ideas that carry the whole design

**`P`, the line pitch, is the unit of length in segmentation** (A.0). Almost every threshold in `segment.py` is a fraction of the pitch the page itself reports, so the segmenter is independent of resolution and handwriting size with no calibration step.

**Relative, not absolute, thresholding** (A.1.1). `ratio = gray / BoxMean(gray, r)` divides out the illumination field before Otsu runs, which is why a phone photo with one corner in shadow binarises correctly. Every box statistic in the project is an integral-image (Crow 1984) evaluation, so window size costs nothing.

**A seam, not a straight line, separates two text lines** (A.1.5). The minimum-cost path through an ink-cost image threads between lines and around descenders, which a horizontal cut cannot.

**Width is time for the recogniser and is averaged away for the writer-ID model** (B.2). Both models share the identical 12-layer convolutional backbone; they differ only in what they do with the width axis, and that difference is the difference between the two tasks.

**Stroke normalisation makes writer ID about geometry** (A.2.4, B.5.5). Real ink → centreline → re-inked at one constant width is the same transform the gantry performs when it redraws a hand with its own pen, so the classifier is trained on exactly what the robot can reproduce. Training and inference both apply it.

**The forward–backward and the forced alignment are one lattice under two semirings** (A.2.6, B.3.7). `train.CtcLoss` sums over the CTC predecessor set (Graves et al. 2006) to train; `recognise.CtcForcedAlign` maxes over the identical set (Viterbi 1967) to locate each character. The second is what makes glyph extraction from cursive possible at all.

**Character boundaries are snapped to thin, low ink columns** (C.2.2). A raw CTC boundary can fall mid-letter; the cheapest cut by `thin + 1.15*topAbove` is the ligature crossing, which is the honest boundary in joined writing.

**Thirteen authors pooled teach what a letter is; one author's pages cannot** (C.2.3). The cross-author shape prior is used only to reject an author's malformed variants, never to replace their letterforms.

**Connectedness is a probability per letter pair, not a threshold** (C.3.3). It is the measured fraction of adjacent in-word pairs the author actually joins — detected by a centreline crossing a character boundary, meaning the pen never lifted — and using it as a threshold fuses whole words into blobs.

**Best-of-N is scored by a harmonic mean** (C.3.6). The most legible of N random draws has no reason to also be the most distinctive, so a text-only score silently trades away writer identity. The harmonic mean is dominated by whichever axis is weaker, which forces a candidate that is genuinely good at both.

**One pen, so one stroke width** (C.3.5). The gantry writes every author with the same pen, so ink weight is not a style channel it can reproduce, and the renderer draws every author at one constant width — matching both the machine and the shape-only judge.

**The pen is a toggle with feedback, never a commanded position** (C.3.8, D.1.5). Every 90° pulse flips it, so software tracks its state and emits a pulse only on a genuine change, and the hardware driver runs the motor only until the switch confirms.

**End-stops are checked before every single step pulse** (D.1.4), debounced over three samples because the unsuppressed motors can hold an input high for milliseconds, and a trip retracts off the switch so the machine is not left latched.

**Calibration measures rather than assumes** (D.1.6). Homing is direction-agnostic — drive until either switch trips, reverse to find the other, count the steps — and the measured steps per millimetre replaces the nominal 80 from the pulley geometry.

---

## Reference points in the literature

| Idea | Source | Where used |
|---|---|---|
| CNN-BiLSTM-CTC recogniser architecture | Kizilirmak & Yanikoglu 2023, arXiv:2307.00664 | B.0, B.2 |
| CTC loss, forward–backward, the `y − occupancy` gradient | Graves et al. 2006 | B.3.7 |
| LSTM cell and the constant error carousel | Hochreiter & Schmidhuber 1997 | B.3.6 |
| BatchNorm | Ioffe & Szegedy 2015 | B.3.2 |
| im2col/GEMM convolution | Chellapilla et al. 2006 | B.3.1 |
| Elastic distortion augmentation | Simard et al. | B.5.2 |
| Otsu thresholding | Otsu 1979 | A.1.1, A.2.1 |
| Summed-area tables | Crow 1984 | A.1.1, A.2.1 |
| Connected components by runs and union–find | Rosenfeld & Pfaltz 1966 | A.2.2 |
| Zhang–Suen thinning | Zhang & Suen 1984 | A.2.3 |
| Seam carving as a line separator | Avidan & Shamir 2007 | A.1.5 |
| Viterbi decoding | Viterbi 1967 | A.2.6 |
| Polyline simplification | Douglas & Peucker 1973 | C.2.1 |
| Bresenham line interpolation | Bresenham 1965 | D.1.4 |
| IAM handwriting database | Marti & Bunke 2002 | A.3, B.5 |

---

## Running things

All commands are run from `Software/CleanPipeline3/`.

```
# training, from first principles, no ML framework
python train.py text   [--epochs N] [--batch N] [--source joint|teklia|pages]
python train.py author [--epochs N] [--pen-width 0.17] [--ink]                        [--writers 150 152 153 384 552 588 abhinav dylan owen thiya yeukita]

# the same two networks in torch, when time is short
python trainlibrary.py text   [--epochs N]
python trainlibrary.py author [--epochs N]

# the stroke-normalisation width sweep
python strokenorm_prepare.py                 # writes four .npz, about 12 minutes
python strokenorm_train.py --epochs 50 --batch 32 --data DIR --out DIR

# the web service, which exposes both live paths
python web/server.py

# the result scripts, each self-contained, in results/<experiment>/
python results/gantry_square/make.py
python results/gantry_square_triangle/make.py
python results/gantry_square_triangle/plot.py
```

`--writers`, `--ink` and `--pen-width` were added on 2026-10-31 and all default
to the previous behaviour. Without `--ink`, `train.py author` produces a
stroke-normalised SHAPE namer, not an ink namer.

There is no longer a standalone segmentation entry point, a gradient-check
script, a profile-builder command or an interactive gantry shell; segmentation
is called through `segment.SegmentLines`, the style profiles are built through
`style.py`, and the gantry is driven through `web/server.py`.
