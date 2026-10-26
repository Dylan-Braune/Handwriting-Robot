# Code Explained — Index

Four explainers covering **every file in `Software/CleanPipeline`**: what each one does, what its important variables mean, why its constants have the values they do, and what the maths is doing. Current as of 21 October 2026, written against the live code.

| Block | File | Covers |
|---|---|---|
| **A — Segmentation** | [CodeExplained_A_Segmentation.md](CodeExplained_A_Segmentation.md) | `SegmentLean.py`, `ProfileIO.py` (primitives), `ExtractIAMLines.py` |
| **B — Neural core** | [CodeExplained_B_TrainingClassification.md](CodeExplained_B_TrainingClassification.md) | `NumpyNN.py`, `GradCheckNumpyNN.py`, `TrainScratch.py`, `TrainText.py`, `TrainAuthor.py`, `Classify.py`, `Charset.py`, `AuthorsConfig.py`, `ConvertPtToNpz.py`, `ExportTekliaLines.py` |
| **C — Style and synthesis** | [CodeExplained_C_ReproductionGcode.md](CodeExplained_C_ReproductionGcode.md) | `BuildStyleProfile.py`, `SynthesizeHandwriting.py`, `Evaluate.py`, `ProfileIO.py` (alignment) |
| **D — Hardware and web** | [CodeExplained_D_GantryHardware.md](CodeExplained_D_GantryHardware.md) | `Gantry.py`, `Server.py`, `Camera.py`, `WebRenderHelpers.py` |

---

## How the system fits together

**The read path** — a photograph of handwriting in, text and an identity out:

```
photo
 └─ SegmentLean.SegmentLines                  (A.1)  page → line crops
     └─ Classify.PaperCRNNNumpy.forward       (B.7)  line → per-frame log-probs
         └─ Classify.DecodeCtc                (B.7)  → text
     └─ ProfileIO.StrokeNormalize             (A.2.4) line → constant-width skeleton
         └─ Classify.AuthorClassifierCNNNumpy (B.7)  → which writer
```

**The write path** — a sentence and a writer in, pen motion out:

```
text + author
 └─ SynthesizeHandwriting.SynthesizeJointBestOf  (C.3.6)  best of N by text × writer-ID
     └─ SynthesizeText                           (C.3.3)  glyphs + ligatures + jitter → polylines
         └─ profile JSON from BuildStyleProfile   (C.2)    glyph library + style parameters
     └─ ToMachine / OptimizeStrokeOrder           (C.3.7)  → bed coordinates, reading order
     └─ WriteGcode                                (C.3.9)  → G-code
         └─ Gantry.RunGcodeFile                   (D.1.7)  → step pulses
```

**The offline paths**, run by hand, that produce what those two depend on:

```
IAM / personal pages ─ ExtractIAMLines (A.3) ─ TrainScratch (B.6) ─→ .npz weights
IAM / personal pages ─ SegmentLean + CTC forced alignment ─ BuildStyleProfile (C.2) ─→ profile JSON
NumpyNN backward passes ───────────────────── GradCheckNumpyNN (B.4) ─→ verification table
```

`Server.py` (D.2) exposes both live paths over HTTP and is what the web app talks to.

---

## The handful of ideas that carry the whole design

**`P`, the line pitch, is the unit of length in segmentation** (A.0). Almost every threshold in `SegmentLean.py` is a fraction of the pitch the page itself reports, so the segmenter is independent of resolution and handwriting size with no calibration step.

**Relative, not absolute, thresholding** (A.1.1). `ratio = gray / BoxMean(gray, r)` divides out the illumination field before Otsu runs, which is why a phone photo with one corner in shadow binarises correctly. Every box statistic in the project is an integral-image (Crow 1984) evaluation, so window size costs nothing.

**A seam, not a straight line, separates two text lines** (A.1.5). The minimum-cost path through an ink-cost image threads between lines and around descenders, which a horizontal cut cannot.

**Width is time for the recogniser and is averaged away for the writer-ID model** (B.2). Both models share the identical 12-layer convolutional backbone; they differ only in what they do with the width axis, and that difference is the difference between the two tasks.

**Stroke normalisation makes writer ID about geometry** (A.2.4, B.5.5). Real ink → centreline → re-inked at one constant width is the same transform the gantry performs when it redraws a hand with its own pen, so the classifier is trained on exactly what the robot can reproduce. Training and inference both apply it.

**The forward–backward and the forced alignment are one lattice under two semirings** (A.2.6, B.3.7). `NumpyNN.CtcLoss` sums over the CTC predecessor set (Graves et al. 2006) to train; `ProfileIO.CtcForcedAlign` maxes over the identical set (Viterbi 1967) to locate each character. The second is what makes glyph extraction from cursive possible at all.

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

```
# segmentation on one page
python SegmentLean.py page.jpg outdir

# training, from scratch, no ML framework
python TrainScratch.py text   [--epochs N] [--batch-size N] [--source pages|teklia]
python TrainScratch.py author [--epochs N] [--batch-size N]

# verify every hand-written backward pass numerically
python GradCheckNumpyNN.py

# build the style profiles
python BuildStyleProfile.py              # IAM set
python BuildStyleProfile.py --10authors  # 8 dataset + personal authors

# one-time dataset / checkpoint conversion
python ExportTekliaLines.py [--out-dir PATH]
python ConvertPtToNpz.py

# the web service
python Server.py

# the gantry, interactively
python Gantry.py        # then: calibrate, g <file>, u/d, r, q
```
