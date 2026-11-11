# Design visual tests — 2026-11-01

41 figures for Section 3 (Design and Implementation), produced from live project
code and real project data. **Nothing here is a qualification test.** These make
a mechanism inspectable so a reader can see a stage work and see where it would
break; the specification measurements stay in Section 4.

Pick what you want. Everything is regenerable, and nothing in this folder is
referenced by the report yet.

```
figures/      41 figures, each as .pdf (for the report) and .png (to browse)
scripts/      the eight scripts that made them
data/         the numbers each script recorded, as JSON
MANIFEST.md   this file
INCLUDE.tex   ready-made \begin{figure} blocks, captions written
```

## How to run

```
cd scripts
python vt_seg.py          # T21-T29   segmentation
python vt_net.py          # T30-T40   network internals
python vt_grad.py         # T38-T39   trainer verification
python vt_style.py        # T19-T49   style extraction
python vt_synth.py        # T20-T51   synthesis
python vt_motion.py       # T52-T57   drive and machine mapping
python vt_blockdiagram.py # T15       software block diagram
python vt_jobstates.py    # T61       job states
python vt_system.py       # T15, T59-T61  integration (calls the two above)
python make_include.py    #           rebuild INCLUDE.tex after adding a figure
```

Each script imports `vt_common.py`, which sets the palette from
`Software/CleanPipeline3/results/palette.py` and resolves the data paths. Run
from inside `scripts/` (the scripts import each other by name, so `python -I`
will not find `vt_common`).

Roughly four minutes for the lot. Only `vt_system.py` and `vt_synth.py` take
more than a few seconds, because they run the networks forward many times.

## What these confirm

Ten quantities quoted in the Design draft reproduce **exactly** from live code:

| Draft says | Figure recomputes | Where |
|---|---|---|
| ratio threshold 0.803906, bin 171 | 0.8039062, bin 171 | T21 |
| ω₀=0.07069375, ω₁=0.92930625 | identical | T21 |
| μ₀=124.98356, μ₁=217.71913, variance 564.97942 | 124.98356, 217.71913, 564.97942 | T21 |
| ΣWX = −1.150897, b = −0.000435125, z = −1.151332 | identical to 7 figures | T30 |
| μ=0.2163021, σ²=0.02585394, γ=1.0193636, β=0.1324657 | identical | T30 |
| y ≈ −8.53619 | −8.53619 | T30 |
| x_L=146, b=107, h_x=38, lead 0.447 | identical | T43 |
| tan φ = −0.0787017 | slant −4.5°, tan −0.0787017 | T43 |
| Dylan medoid index 4, total 33.3413, others 34.4306–42.9091 | identical | T45 |
| connectedness 3/39 = 0.076923 | 3/39 = 0.0769 | T49 |

That is worth stating in the report: the figures and the worked arithmetic in
the text were produced independently and agree.

**Important scale note.** The draft's traces run on the page reduced to 1600 px
(`trace_real_calculations.py` thumbnails before calling `Deskew`). **T21 uses
that same reduced page** so its numbers match the text. Every other
segmentation figure runs at the page's own resolution, which is what the
segmenter sees in service — at 1600 px the line pitch falls to 25 px and the
geometry becomes too coarse to show. The stamp on each figure records which.

## The figures

Each is listed as: **number — what it shows** · the subsection it belongs to ·
what came out.

### Section 3.4 — Page conditioning and line segmentation

| # | Figure | For | Result |
|---|---|---|---|
| T21 | Otsu's criterion on the ratio histogram, with the page at the selected bin and ±40 bins | 3.4.2 | bin 171, variance 564.98; the maximum is isolated, not a plateau |
| T22 | The whole deskew search on five writers' pages | 3.4.3 | each page has a single sharp peak; selected angles ringed |
| T23 | The quarter-turn decision and its 1.35 margin | 3.4.3 | shows which pages clear the margin and which are left alone |
| T24 | Row profile, the 6% floor, the 0.6P rule and the pitch | 3.4.4 | Abhinav01: 29 peaks above the floor, 26 accepted, P=48 |
| T25 | **The seam cost field and the path through it** | 3.4.4 | a straight cut severs **64** ink pixels where the seam severs **1**; max departure 22 rows |
| T26 | The midpoint weight swept over six real gaps | 3.4.4 | at 0.012 the seam sits 0.4 px off the midline; at 0 it wanders |
| T27 | The three ownership rules, before and after, boxed | 3.4.5 | Abhinav01: 462 px reassigned across 3 cases |
| T29 | The 64×640 aspect-preserving fit on four writers | 3.4.5 | a 2170-px line scales to a=0.295 and uses 46 of 64 rows |

### Section 3.5 — Network design, training and classification

| # | Figure | For | Result |
|---|---|---|---|
| T30 | **One convolution written out**: the 9 inputs, 9 weights, their products, then BN and ReLU | 3.5.2 | z = −1.151332 → y = −8.53619 → ReLU 0; matches the text exactly |
| T31 | All 32 learned first-layer kernels | 3.5.2 | oriented light–dark pairs, i.e. stroke and edge detectors |
| T32 | Feature maps at true resolution through every stage | 3.5.2 | the shapes in the backbone table, as the arrays they describe |
| T33 | Max pooling on a real 4×4 window and a real map | 3.5.2 | H and W halve, channels do not |
| T34 | **The height collapse 128×16×160 → 128×1×160** (bottom panel of T32) | 3.5.2 | the decision that makes the sequence model possible |
| T35 | Which source pixels each of the 160 frames sees | 3.5.3 | 160 frames over 2170 px, 4 canvas columns each |
| T36 | **The CTC lattice with the forced path and the greedy path** | 3.5.3 | the reader decodes the line **exactly**; blank holds 46% of the mass |
| T37 | Per-frame posteriors as spikes | 3.5.3 | the characteristic spiky, blank-dominated output |
| T38 | **Gradient check**: 249 parameters against central differences | 3.5.6 | worst relative difference **7.5 × 10⁻⁷**; median ~10⁻¹⁰ |
| T39 | Shear, rotate, perspective and elastic on four writers | 3.5.5 | label-preserving geometry change |
| T40 | The stroke-width sweep shown, with the sweep accuracies | 3.5.4 | see the finding below |

### Section 3.6 — Style extraction and reproduction

| # | Figure | For | Result |
|---|---|---|---|
| T19 | The surviving glyph pools with each variant's width and prior distance | 3.6.3 | shows what the width and height-class gates act on |
| T41 | Skeleton → traced paths → simplified, each path shaded | 3.6.3 | 85 paths, 3738 → 974 vertices at ε=0.6 |
| T42 | **Cut-cost refinement**: the ink and height terms and where the boundary moves to | 3.6.2 | largest refinement on this line is 17 px over 63 characters |
| T43 | The stored glyph coordinate frame against image pixels | 3.6.3 | 'V': x_L=146, b=107, h_x=38, lead 0.447 — matches the text |
| T45 | **Variant cloud, medoid, prototype and the consensus totals**, three writers | 3.6.3 | Dylan medoid is index 4 at 33.3413 — matches the text |
| T46 | The 12×12 occupancy descriptor behind the prior | 3.6.3 | the same letter from four writers gives visibly different grids |
| T47 | The slant search on four writers | 3.6.4 | peaks clearly separated between hands |
| T48 | Jitter at 0 / 0.15 / 0.35 / 0.70 | 3.6.5 | where controlled variation becomes illegibility |
| T49 | **Connectedness counted on a real line** | 3.6.4 | 3 of 39 eligible boundaries crossed = 0.0769, matching the text |
| T50 | Style calibration before and after, three writers | 3.6.5 | the shear and extent corrections, per writer |
| T20 | Joining decided by each writer's own connection fraction | 3.6.5 | which profiles are joined and which are not |
| T51 | **Candidate selection** with A, P and the harmonic mean J (Owen and Dylan) | 3.6.6 | best J = 0.807 (Owen), 0.802 (Dylan); losing candidates shown |

### Section 3.3 — Drive, and 3.6.7 — machine mapping

| # | Figure | For | Result |
|---|---|---|---|
| T52 | **The Bresenham staircase** on a real (+6, −28) microstep move | 3.3.2 | worst departure 0.5 microsteps = 6.3 µm at 80 microsteps/mm |
| T53 | **The directional pulse penalty**, predicted and measured on 396 real segments | 3.3.2 | every segment lands on the predicted curve; mean 1.225, max exactly 2.0 at 45° |
| T54 | Stroke-order optimisation, pen-up path before and after | 3.6.7 | Owen 187.5 → 176.6 mm of air travel, ink unchanged |
| T55 | Decimation at five minimum-segment settings | 3.6.7 | vertex count against visible straightening |
| T56 | **The nominal time budget decomposed** | 3.6.7 | Owen 9.25 s drawing + 3.20 s travel + **20.3 s dwell** — dwell is 62% |
| T57 | A real job placed in the calibrated workspace with margins and bounds | 3.6.7 | 190.6 × 251.3 mm travel, 15/10 mm margins |

### Sections 3.2 and 3.7 — system

| # | Figure | For | Result |
|---|---|---|---|
| T15 | **The software block diagram** — three lanes, G-code marked as the boundary | 3.2 | fills one of the six missing `\designasset` placeholders |
| T59 | **The page vote**: per-line probabilities, ink weights, pooled result | 3.7.1 | Owen01: 20 of 20 crops agree, pooled 0.9961 |
| T60 | **The round trip**: trajectory → emitted G-code → re-parsed → a page the robot drew | 3.7.3 | (a) and (b) are indistinguishable; three visibly different hands |
| T61 | Job states and the interruption paths | 3.7.2 | generated / refused / completed / interrupted |

## Three things these turned up

**1. The stroke-width sweep is partly degenerate (T40).** The quantity that
changes the image is the odd dilation kernel `k`, not the fraction ρ. On Owen's
crop ρ=0.06 and ρ=0.09 both give k=3 and therefore the *identical* image; on
Abhinav's the same two collapse; on Dylan's ρ=0.09 and ρ=0.11 both give k=5.
The draft already warns that "nearby fractions can produce exactly the same
transformed image" — this shows it happening on the sweep's own settings, which
means some of the 98.06 / 97.42 / 98.06 spread is training noise between runs
on identical inputs, not a width effect. Worth a sentence beside Table 3.6.

**2. `ToMachine` returns a tuple, not a `Trajectory`.** `(strokes, scale)`. Not
a bug, but anything written against the `Trajectory` interface will fail on it,
and it cost me a pass here.

**3. Pen dwell is 62% of the nominal job budget (T56).** 20.3 s of 32.7 s for
one sentence, at the configured 350 ms each way and 58 transitions. The
Discussion treats the timing shortfall mostly as processing cost; this says the
largest single lever on the *drawing* side is the dwell constant and the number
of pen transitions, which stroke ordering and join distance both affect.

## Honest limits

- **Figures are illustrations, not measurements.** Captions are written to say
  so. T36, T51 and T59 are the easiest to overclaim from, because in each the
  system is being scored by the same models it uses — the captions state this.
- **Selection rules are stated, not hidden.** Where a figure shows "the" gap or
  "the" segment, the script chose it by a written rule (most ink avoided,
  largest departure, most peaks rejected) and the caption says which. Nothing
  was picked by eye.
- **T40's accuracies are quoted from your supplied sweep**, not measured here.
  The images are live; the bars are not.
- **T60 panel (c) is a different sentence** from (a) and (b). It is a real
  photographed page, not a drawing of that sentence, and the caption says so.
- **Three figures depend on checkpoints that are not in version control.**
  T40, T51 and T59 load `author_shape_11_pw*.npz` from `C:/Users/braun/Downloads/`.
  Same problem `measurements/RESULTS_PROVENANCE.md` §13a flags.
- **Nothing here touches the gantry.** T52 and T53 evaluate the stepping
  arithmetic on real generated moves; no motion was commanded.

## What I could not do

- `design-gantry-assembly.jpg` — needs a photograph of the assembled gantry.
- `design-camera-arrangement.jpg` — needs a photograph of the camera mount.
- `design-complete-circuit.pdf` — needs the installed wiring, which only you
  can confirm. T15 covers the *software* block diagram only.

Those three remain blue `\designasset` placeholder boxes in the built PDF.
