# Handwriting Robot — full project and report context

**Student:** D.R. Braune, u23539195, University of Pretoria, EPR400.
**Written:** 2026-10-31. Everything below is factual and sourced from the
repository, the measurements taken, or the student's own statements, which are
marked as such. Where something is unverified or assumed, it says so.

**Purpose:** to let another person or agent pick up work on the final report
without re-deriving anything. Read section 1 for where things live and how to
run them, section 2 onwards for what the project is and what has been measured.

---

## 1. Where everything is and how to run it

Repository root: `C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot`
Git remote: `https://github.com/Dylan-Braune/Handwriting-Robot.git`, branch `main`.

**The student pushes to git. Never push on their behalf.** Commit only when
asked.

### 1.1 The live system — `Software/CleanPipeline3/`

This is the current, deployed pipeline. Everything else under `Software/` is
superseded (see 1.6).

| File | What it does |
|---|---|
| `paths.py` | the single source of truth for where data lives and who the writers are |
| `segment.py` | page conditioning, deskew, line finding, seam cutting, crop building |
| `recognise.py` | numpy inference: `PaperCRNNNumpy` (reader), `AuthorClassifierCNNNumpy` (namer), `StrokeNormalize` |
| `synthesise.py` | glyph-library synthesis of new text in a writer's style |
| `style.py` | style profile extraction and storage |
| `reproduce.py` | gantry motion planning and G-code emission |
| `retrace.py`, `curate.py` | glyph recutting and library curation |
| `timing.py` | the stage timer used by the server |
| `train.py` | **from first principles** training, numpy only, no ML library |
| `trainlibrary.py` | the same two networks in torch, "for when time is short" |
| `web/server.py` | the Flask server and all API routes |
| `web/camera.py`, `web/render.py`, `web/frontend/` | capture, rendering, UI |

**Run the server:** from `Software/CleanPipeline3/`, the server is started on
the ODROID. Routes used in testing: `POST /api/classify` (image → text +
writer), `POST /api/generate` (text + author → job), `GET /api/gcode/<id>`,
`POST /api/write/<id>` (draw it), `GET /api/authors`, `POST /api/gantry/stop`,
`POST /api/gantry/calibrate`.

**Train from first principles:**
```
python train.py text   --epochs 40
python train.py author --epochs 50 --pen-width 0.17 \
    --writers 150 152 153 384 552 588 abhinav dylan owen thiya yeukita
```
`--writers`, `--ink` and `--pen-width` were added on 2026-10-31 and all default
to the previous behaviour. `--ink` trains on crops as photographed instead of
stroke-normalised; without it `train.py author` produces a SHAPE namer.
`--resume` continues from the checkpoint.

**Train with torch (much faster, same architecture and recipe):**
```
python trainlibrary.py author --epochs 40 --batch 8
```

**The stroke-normalisation sweep** (two files, added 2026-10-31):
```
python strokenorm_prepare.py          # local, ~12 min, writes 4 .npz
python strokenorm_train.py --epochs 50 --batch 32 --data /content/sn --out /content/sn
```
`strokenorm_prepare.py` does the skeletonising once per pen width and writes
`strokenorm_data_pw{060,090,110,170}.npz` (5.5–6.5 MB each, 24 MB total) to
`Software/CNN/NOGIT/StrokeNormSweep/`. `strokenorm_train.py` consumes those and
trains on GPU; it needs only those four files plus a clone of the repo.
Early stop: ctrl-C once skips to the next width, twice abandons; or drop
`skip.flag` / `stop.flag` into the output directory.

### 1.2 Data

| Path | Contents |
|---|---|
| `Data/Datasets/IAMpages10/` | **the writer corpus.** One folder per IAM writer (150, 151, 152, 153, 154, 155, 384, 551, 552, 588), 9–10 page PNGs each. Extracted into lines by the project's own extractor. |
| `Data/Datasets/TekliaIamLines/` | the pre-cut Teklia IAM line dataset, used to train the **reader** only |
| `Data/Datasets/IAMpages671/`, `IAMlines50/`, `IAMwordsFULL/` | larger/older IAM pulls, not used by the current pipeline |
| `Software/CNN/PersonalDataset/` | the five photographed contributors: `Abhinav/`, `Aston/`, `Dylan/`, `Owen/`, `Thiya/`, `Yeukita/`, each with `Pages/` and `Labels/` |
| `Software/CNN/weights/` | trained checkpoints, `.npz` for numpy inference and `.pt` alongside |

`Data/Datasets/` is gitignored. `Software/CNN/NOGIT/*` is gitignored except
`RNNStrokeCache10`.

### 1.3 Caches — `Software/CNN/NOGIT/`

| Path | Contents |
|---|---|
| `line_cache_authors10/` | extracted IAM line crops, sharded, with a manifest |
| `personal_lines_cache.pkl` | the five photographed writers cut into lines, 669 samples |
| `StyleProfiles10/` | one JSON style profile per writer, plus `_curated` variants (25 files) |
| `GlyphCache10/` | the per-writer glyph libraries used by the synthesiser |
| `StrokeNormSweep/` | the sweep's prepared `.npz` and its output weights |
| `SegCheck/`, `SegEdits/` | segmentation audit renders, and the folder for edited page scans |
| `WebJobs/` | on the ODROID: one folder per generate job, holding `job.gcode` |
| `CameraCaptures/` | server captures |

### 1.4 Results

| Path | Contents |
|---|---|
| `Software/CleanPipeline3/ReportResults/FullReproducedPages/` | **the 14 scanned reproduction pages** plus a `.txt` transcript for each, one line of text per written line |
| `Software/CleanPipeline3/ReportResults/SegmentedReproducedPageCrops/` | per page: `segmentation_overlay.png`, `all_crops_contact_sheet.png`, `line_NN.png` |
| `Software/CleanPipeline3/ReportResults/SegmentedReproducedPageCrops/_StandUpFailures_AsShipped/` | the three pages the orientation test got wrong, kept as evidence |
| `Software/CleanPipeline3/results/` | one folder per experiment, each with its own `make.py` |
| `Software/CleanPipeline3/results/palette.py` | **the one colour and style definition for every figure** |
| `Software/CleanPipeline3/results/gantry_square/`, `gantry_triangle/`, `gantry_square_triangle/` | the calibration figures and their G-code |
| `Report/FinalReport_v6/Figures/` | the figures the report includes |

`Software/CleanPipeline3/ReportResults/` sits inside a git-tracked folder and is
about 31 MB. It is **not** gitignored; decide whether to commit it.

### 1.5 The report — `Report/FinalReport_v6/`

| File | Contents |
|---|---|
| `main.tex` | the document, with the `\mustdo`, `\mustcomplete`, `\new`, `\figmaybe`, `\figplaceholder` macros |
| `epr400.cls` | the **official** template, byte-identical to the supplied one. Do not replace it. Defines `\QualificationTest` and `\QualificationSection` |
| `3_Main_Report/1_Literature_Study.tex` … `6_Conclusion.tex` | the six main sections |
| `3_Main_Report/design/3_1…3_10*.tex` | the design section, split by subsection |
| `4_Appendices/Appendix.tex` and `Appendix_*.tex` | the appendix; six of them are commented out (see 4.3) |
| `references.bib` | the bibliography |
| `verify_report.py` | **the compliance checker.** Run it from this directory |
| `make_results_figures.py` | regenerates every section 4 figure from `palette.py` |
| `REFERENCE_AUDIT.md` | the citation locator audit and its rule |
| `COMPARISON_23531003.md` | findings from reading another student's report end to end |
| `READ_ME_FIRST.md` | the student's own notes |

**Build:** `latexmk -pdf main.tex` from `Report/FinalReport_v6/`. If `main.pdf`
is locked by a viewer, build under another jobname:
`latexmk -pdf -jobname=v6 main.tex`.

**Check compliance:** `python verify_report.py`. It follows `\input` from
`main.tex` so it only judges files that are actually built, and reports files on
disk that are not.

### 1.6 Older versions — what they are and whether to trust them

| Path | Status |
|---|---|
| `Software/CleanPipeline/`, `CleanPipeline2/` | superseded pipelines. `CleanPipeline2` is what the 335-page code-theory document describes |
| `Software/CNN/` | the original training and segmentation work. `SegmentLean.py` is still imported by `train.py` for cutting personal pages. The 40-plus `SegmentLean_*.bak` files are a development history, not live code |
| `Software/CNN/TrainAuthor.py`, `TrainText.py` | the original trainers, superseded by `CleanPipeline3/train.py` |
| `Software/GantryControl/`, `ComputerVision/`, `StyleSynthesis/`, `WebDisplay/` | earlier subsystem work |
| `Report/FinalReport/` … `FinalReport_v5/` | earlier report revisions. v6 is current |
| `Report/_dev/verify_report.py` | the original checker; the v6 copy is the improved one |
| `Report/ProjectHistoryDigest.md` | a dense history of the pipeline, its bugs and its results. Read before writing |
| `Report/REPORT_REQUIREMENTS.md` | a condensed spec. **It gets several Appendix 4 rules wrong** — see 4.1 |
| `Report/CodeExplained_*.md` | four explainers of the code, in a settled house style |
| `FILE_INVENTORY.md`, `PROJECT_CONTEXT.md`, `WORKLOG.md`, `handover.md` | older top-level notes |

### 1.7 Analysis scripts written on 2026-10-31

These live in a session scratchpad and are **not** in the repository. If they
matter, move them in. Path:
`C:\Users\braun\AppData\Local\Temp\claude\C--Users-braun-OneDrive-Documents-GitHub-Handwriting-Robot\21096ef3-779e-4365-af8c-095a5dea04a3\scratchpad\`

| Script | What it produces |
|---|---|
| `blocks_master.py` | the 184-line per-line record of the 14 reproduction pages, split by block |
| `per_author.py` | accuracy per writer, connectedness bands, correlations |
| `cursivity.py` | connected components per character on each writer's real pages |
| `style_distance.py` | the writers placed in a style space, nearest-neighbour distances |
| `holdout_pw170.py` | the namer on its own 155 held-out lines |
| `holdout_text.py` | the reader on the same 155 lines |
| `text_boost.py` | reader comparison and the per-writer and database/photographed splits |
| `realign.py`, `look_384.py`, `audit_384.py` | the writer 384 investigation |
| `classify_gcode.py` | renders a `.gcode` and classifies it |
| `timing_sessions.py`, `match_sentences.py`, `attribute_final.py` | the timing-log attribution |
| `predict_drawtime.py` | per-writer ink and predicted draw time |
| `gantry_geom.py` | the square and triangle analysis |
| `segcheck.py`, `build_reportresults.py` | the segmentation renders and the ReportResults folder |

---

## 2. What the project is

A gantry robot that photographs a page of handwriting, reads the text, decides
which of the stored writers wrote it, and then draws new text on paper in that
writer's hand.

**Hard constraint on credit:** Appendix 4 gives credit for own first-principles
design and implementation. Off-the-shelf solutions, library code and copied
designs carry no credit. Implementing a published *approach* from a paper's
description **is** own implementation and does carry credit. The report must
never claim the student derived a published algorithm; it must cite the source
at the point of use and claim the implementation.

### 2.1 The writers

`paths.py` defines them:
```
DatabaseWriters = ["150","151","152","153","384","551","552","588"]
PersonalWriters = ["abhinav","dylan","owen","thiya","yeukita"]
```
**As of 2026-10-31 the student has withdrawn 151 and 551**, leaving **11
writers**. The new sweep classifiers are trained on those 11. `paths.py` has not
been edited — the `--writers` flag carries the restriction instead, because that
list is also read by the server, the reproducer and the style profiles.

In the report the five photographed contributors are labelled by initial:
**A = Abhinav, B = Braune, O = Owen, T = Thiya, Y = Yeukita.**

### 2.2 The two classifiers

- **Ink namer** `author_classifier_10new_weights.npz` — trained on crops as
  photographed, so it can read pen pressure.
- **Shape namer** `author_shape_10new_weights.npz` — trained on
  stroke-normalised crops, which redraw the writing at one constant pen width
  so thickness cannot be used.

`server.py` sets `PAGE_NAMER = SHAPE_MODEL` and `PAGE_NAMER_STROKE_NORM = False`.

**The ink namer is loaded at `server.py:181` and never used to classify
anything** — its only other appearance is lines 1024–1026, a stats panel
reporting its filename, parameter count and size. Every decision uses
`SHAPE_MODEL`: page naming (line 444), the synthesis best-of-N search (753), the
camera preview (781).

**There is a train/serve mismatch.** The shape namer was trained on
stroke-normalised crops but `PAGE_NAMER_STROKE_NORM = False` serves it
unnormalised ones. `ClassifyImage` normalises at the module default
`PenWidthXh = 0.14` and takes no width argument, so installing a model trained
at another width requires changing `recognise.PenWidthXh` as well.

### 2.3 The three-block test design

The student's own design, and the reason the 14 pages are laid out as they are:

- **Block A** — a 378-character paragraph, 8 to 17 written lines. Tests line
  segmentation and general text accuracy together.
- **Block B** — the 43-character pangram "The quick brown fox jumps over the
  lazy dog". Every letter, one or two lines, so it tests reading and writer
  identification without multi-line segmentation.
- **Block C** — a passage of the writer's **own** source text. The comparison
  and overfitting check: the only one of the three the writer actually wrote.

---

## 3. Measurements as of 2026-10-31

All taken on the development laptop (x86-64, CPython 3.14), not the ODROID.
Every figure states its sample.

### 3.1 The 14 reproduction pages

`Author150, Author152, Author1531, Author1532, Author3841, Author3842,
Author552, Author588, AuthorA1, AuthorA2, AuthorD, AuthorO, AuthorT, AuthorY`
— **11 of 13 writers; 151 and 551 have no scan.**

184 line samples. Measured with the 11-class shape namer
`author_shape_11_pw170_weights.npz` at pen width 0.17, and the reader
`paper_cnn_bilstm_ctc_joint_best.npz`.

| | Writer ID | Char (micro) | Word (micro) |
|---|---|---|---|
| Block A, 119 lines | 91.6 % [85.2, 95.4] | 84.88 % | 47.87 % |
| Block B, 21 lines | 76.2 % [54.9, 89.4] | 78.40 % | 41.41 % |
| Block C, 44 lines | 77.3 % [63.0, 87.2] | 85.32 % | 40.91 % |
| **All 184** | **86.4 % [80.7, 90.6]** | **84.41 %** | **45.97 %** |

Macro (mean of per-line percentages) over all 184: char 82.04 %, word 48.80 %.
Pages, lines pooled by ink: **14 of 14 correct**.

Per writer, ordered by character accuracy:

| Writer | Lines | Writer ID | Char | Word | Components/char |
|---|---|---|---|---|---|
| Y | 13 | 92.3 % | 97.01 | 74.16 | 0.739 |
| 150 | 13 | 53.8 % | 96.75 | 76.40 | 0.766 |
| O | 16 | 100 % | 94.25 | 58.51 | 0.951 |
| B | 16 | 93.8 % | 93.11 | 68.24 | 0.980 |
| 384 | 16 | 87.5 % | 86.93 | 43.18 | 0.681 |
| 552 | 15 | 80.0 % | 86.17 | 48.39 | 0.609 |
| 152 | 15 | 80.0 % | 85.74 | 49.44 | 0.512 |
| A | 27 | 100 % | 77.29 | 18.09 | 0.527 |
| 153 | 19 | 57.9 % | 75.72 | 31.46 | 0.255 |
| 588 | 18 | 94.4 % | 73.19 | 22.99 | 0.547 |
| T | 16 | 100 % | 60.97 | 13.41 | 0.208 |

The three largest confusions: **153 named as T on 6 lines, 150 named as Y on 5,
552 named as A on 3.**

### 3.2 Connectedness

Measured as **connected ink components per character on the writer's own real
pages**, 45 sampled crops per writer. A joined word is one component however
many letters, so lower is more cursive.

| Against | Pearson | Spearman |
|---|---|---|
| character accuracy | +0.855 | **+0.845** |
| word accuracy | +0.771 | +0.782 |
| writer identification | +0.170 | **−0.027** |

So connectedness strongly predicts how well text is **read** and has no
relationship with how well the writer is **identified**. The four least joined
hands (B, O, 150, Y) are exactly the four above 93 % character accuracy.

A style-proximity test was also run: each writer placed in a space of eight
standardised descriptors (slant, x-height, ascender, descender, stroke width,
ink fraction, aspect, mid-fraction), distance to nearest other writer against
identification gives Pearson −0.244, Spearman −0.100 — **no relationship**, and
none of the three big confusions is with the writer's nearest neighbour. So
identification difficulty is explained by neither cursiveness nor these
geometric similarity measures. `n = 11` throughout; no multiple-comparison
correction applied.

**An earlier version of this measure used pen lifts per character in the
synthesised G-code. That was wrong** — it measures how the glyph library was
cut, not how the writer joins letters, and it ranked 153 as the most separated
hand when its own pages are plainly joined. Do not use it.

### 3.3 The classifier on real handwriting (155 held-out lines)

Split is the project's own and seeded: for database writers the **last labelled
page**; for photographed writers a **seeded 15 % of lines**.

- **Writer identification: 154 of 155 = 99.35 %**, Wilson [96.4, 99.9].
  Database 55/55, photographed 99/100. Pages pooled by ink: **36/36**.
  The single error is a line of B named Y at 39.5 % confidence against a 97.0 %
  mean.
- **Text: 91.02 % character, 56.66 % word** with the shipped joint reader;
  **91.60 % / —** with the scratch reader. On the photographed lines only,
  **93.33 % joint, 94.42 % scratch.**

### 3.4 Reader comparison

| | All 155 | Database 55 | Photographed 100 |
|---|---|---|---|
| joint (shipped) | 91.02 % | 87.37 % | 93.33 % |
| **scratch (first principles)** | **91.60 %** | 87.15 % | **94.42 %** |

Scratch wins on 7 of 11 writers and on the photographed subset. It is **not
installed** — the server loads `joint_best`. The scratch checkpoint is at
`C:\Users\braun\Downloads\paper_cnn_bilstm_ctc_scratch_best.npz` (30 Oct 14:15);
the copy in `Software/CNN/weights/` is a **stale 19 Oct build**, a different
file.

### 3.5 Writer 384 — three separate data faults

1. **`h06-082.png` is on disk but absent from the dataset.** 9 of 10 pages load.
2. **10 merged crops across 6 of its 9 pages** (h06-079 line 5; h06-085 lines
   1, 6; h06-089 line 1; h06-092 lines 0, 4; j06-005 lines 0, 2, 6; j06-014
   line 8). Each merged crop desynchronises every label below it on that page.
   **10 of the 13 merged crops in the whole 98-page corpus belong to 384**; the
   others are one each on `151/c03-094b`, `154/c03-016e`, `154/c03-084e`.
3. **No line of 384 is in the reader's training data.** The reader trains on
   Teklia + personal, and Teklia's subset does not include this writer.

The **labels themselves are correct** — checked by eye against the images on
`j06-014` and `h06-079`. The fault is segmentation, not transcription.

Effect: 384 reads at 60.78 % per line but 82.70 % when the page is scored whole,
and it costs **1.89 points of the overall 155-line figure**.

### 3.6 Which writers the reader has actually seen

Checked per page against the Teklia arrow files:

| Writer | Held-out page in Teklia? |
|---|---|
| 150 | `c03-096a` 7/7 — **leaked** |
| 152 | `c03-096c` 7/7 — **leaked** |
| 153 | `c03-096d` 8/8 — **leaked** |
| 552 | `m06-106` 10/10 — **leaked** |
| 588 | `n06-140` 7/7 — **leaked** |
| 384 | `j06-014` 0/7, and 0 of all 9 pages — **never seen** |

So neither IAM group is in the intended condition. Five writers are tested on
lines the reader trained on; 384 is tested on a hand the reader has never seen.
**The five photographed writers are the only ones in the correct condition** —
seen hand, unseen lines — and they score 94.42 %.

Training the reader on the writers it will later read is the **design**, not a
flaw. The fault is that the reader's corpus (Teklia) and the namer's corpus
(IAMpages10) are different, so one writer is invisible to half the system.
The fix is to add `IAMpages10` lines to the reader's training, excluding each
writer's held-out page.

### 3.7 Stroke-normalisation sweep

`StrokeNormalize` quantises: `w = clip(round(penWidthXh*xh),2,9)`, then
`k = 2*(w//2)+1`. **Only kernels 3, 5, 7, 9 are reachable; k=1 is impossible**
because w is floored at 2. Measured over 400 crops the x-height runs 13–226 px,
median 39, so every setting at or below 0.06 gives k=3 on 98 % of crops.

Four widths trained on Colab with torch, 11 writers, 50 epochs each:

| penWidthXh | mean kernel | Holdout | Reproduced pages | Lines |
|---|---|---|---|---|
| 0.06 | 3.06 | 98.06 % | 10/14 | 130/184 |
| 0.09 | 4.14 | 97.42 % | **14/14** | 158/184 |
| 0.11 | 4.88 | 98.06 % | 12/14 | **160/184** |
| **0.17** | ~7.1 | **99.35 %** | **14/14** | 159/184 |
| *0.14 (current)* | *5.99* | *not trained* | — | — |

**0.17 is the current best choice.** 0.06 — the thinnest possible — is clearly
worst. The 0.14 control was never trained, so it is not yet established that
0.17 beats the current setting under identical conditions.

### 3.8 Timing

**R3, drawing.** The log holds 97 timing blocks over 13 sessions; 47 have a
drawing stage. Every attributable run met the 4 s/char allowance. Range **1.343
to 2.705 s per character**. Drawing is 83.7–98.0 % of each run.

**R4, classify.** All 14 pages **exceeded** the `2 + 0.025x` allowance, by 10.3
to 27.4 s against allowances of 3.4 to 15.1 s. Reading is 36–41 % of each page
and naming 17–25 %.

**Timing-to-writer attribution** (see 5.3 for method):

| Session | Time | Writer |
|---|---|---|
| 1 | 29 Oct 14:37 | owen, first run, **1200 mm/s — excluded** |
| 2 | 29 Oct 18:44 | thiya |
| 3 | 29 Oct 18:50 | dylan, then abhinav |
| 5 | — | 153, cancelled attempts, no paragraph drawn |
| 6 | 30 Oct 12:19 | 588 or yeukita cancelled, no paragraph drawn |
| 7 | 30 Oct 12:29 | **150** |
| 8 | 30 Oct 13:01 | **152** |
| 9 | 30 Oct 13:46 | **153** |
| 10 | 30 Oct 14:24 | **384** |
| 11 | 30 Oct 15:22 | **552** |
| 12 | 30 Oct 16:46 | **588** |
| 13 | 30 Oct 18:07 | **owen, final run** |

Session 4 (yeukita marker, 30 Oct 12:46) sits oddly among the IAM runs; its
marker reads "FAILED TEST ONE HARDER PRSSURE", so it is read as a later retry.

All runs except session 1 used 800 mm/min drawing and 4000 mm/min travel.
A constant **overhead factor of 1.355** (measured 8.7 % spread over three
independent anchors) relates idealised constant-feed time to measured time —
that is acceleration and deceleration across thousands of short segments.

### 3.9 Style signature — ink for identical text

Regenerating the same 378-character paragraph per writer:

| Writer | Pen-down mm | Pen lifts | | Writer | Pen-down mm | Pen lifts |
|---|---|---|---|---|---|---|
| abhinav | 3943 | 1044 | | owen | 2420 | 904 |
| 153 | 3494 | 1188 | | 150 | 2394 | 972 |
| 152 | 3185 | 1070 | | yeukita | 2367 | 968 |
| thiya | 3158 | **544** | | 588 | 2329 | 884 |
| 384 | 3084 | 914 | | 552 | 2313 | 804 |
| | | | | dylan | 2015 | 870 |

**A factor of 1.96 in ink for identical text.** Thiya lays down nearly as much
ink as 153 with **less than half the pen lifts**.

### 3.10 Gantry geometry — calibrated machine

**Measurements taken before calibration are excluded**; they describe a
configuration that no longer exists. n = 1 square and n = 1 triangle.

| Feature | Nominal | Measured | Error |
|---|---|---|---|
| square bottom (X) | 100.00 | 100.3 | +0.30 |
| square top (X) | 100.00 | 100.2 | +0.20 |
| square left (Y) | 100.00 | 100.1 | +0.10 |
| square right (Y) | 100.00 | 99.9 | −0.10 |
| triangle base (X) | 80.00 | 80.3 | +0.30 |
| triangle left sloped | 56.57 | 56.2 | −0.37 |
| triangle right sloped | 56.57 | 56.9 | +0.33 |

- **X axis scale +0.250 %, Y axis +0.000 %.** Mean absolute error 0.175 mm,
  worst side 0.30 mm, against R5's 0.5 mm limit.
- **Squareness error φ = −0.706°**, solved from the two sloped sides via
  `d = (l² − r²)/(2b)`, `φ = arctan(d/h)`. One free parameter, two
  measurements, residuals **0.02 mm on each** — at caliper resolution.
- The square's X scale predicts an 80.20 mm triangle base; 80.3 measured.
  Two sheets agreeing to 0.10 mm.
- `SQUARENESS_ERROR_DEG` in `reproduce.py` is documented as 1.0°; measured 0.706°.
- The same skew would split the square's diagonals to 142.30 and 140.55 mm
  — **an untaken measurement that would confirm it independently.**

**n = 1 supports the scale errors and the skew angle. It does not support a
repeatability figure, a standard deviation or a tolerance claim.** The 0.40 mm
range across the four sides is spread within one specimen, not variation
between repeated draws. The student plans more measurements on Monday
2026-11-02.

Travel constants currently in `reproduce.py`:
`X_SWITCH_TRAVEL_MM = 200.6`, `Y_SWITCH_TRAVEL_MM = 261.3`.

### 3.11 G-code classified without drawing

Rendering a `.gcode` to an image and classifying it separates synthesis loss
from gantry loss.

| Job | G-code lines | Text lines | Named | Conf | Agreeing |
|---|---|---|---|---|---|
| `2b7e03f1` | 9399 | 8 | O | 86.0 % | 7/8 |
| `99f136a0` | 5889 | 7 | O | 66.7 % | 6/7 |
| `bc8d7860` | 13747 | 17 | A | 75.6 % | 15/17 |
| `a6393598` | 9766 | 11 | B | **29.3 %** | **4/11** |
| `16d4c05b` | 1255 | 2 | B | 72.4 % | 2/2 |
| `bfa7a5dc` | 1288 | 1 | Y | 79.2 % | 1/1 |

`2b7e03f1` has 9399 lines, which is the count the log records for the run
annotated `Owen_curated` — so the classifier independently confirms a case where
the truth is known. `a6393598` is below any useful confidence and should not be
used.

---

## 4. The report

### 4.1 Appendix 4 — the rules that actually bind

Source of truth: `C:\Users\braun\Downloads\APPENDIX4_2026.pdf`. Extract with
`pdftotext -layout`. **`Report/REPORT_REQUIREMENTS.md` overstates several rules;
the PDF wins.**

Structure: Part 1 preamble, Part 2 the unaltered approved Project Proposal,
Part 3 the main report, optional appendix. **Page cap 70, main report sections
1–6 only**, excluding references and appendix. No abstract is required and none
should be invented.

**Section 4 Results specifically:**
- Commence on a new page. First describe *how* compliance was measured, then
  give every result, **including unsuccessful ones**.
- **4.1 Summary of results achieved** — a compulsory three-column table
  (Intended outcome / Actual outcome / Location in report) with rows grouped
  under *Core mission requirements and specifications* and *Field condition
  requirements and specifications*.
- **4.2 Qualification tests** — in **order of requirement importance**, focused
  on the **integrated system, not subsystems**, "seldom more than around five or
  six".
- Each test: a bold **unnumbered** `Qualification test N: title` heading, then
  *italic* subheadings — Objectives / Equipment used / Test setup and
  experimental parameters / Steps followed / Results or measurements /
  Observations / Statistical analysis.
- **Observations must contain no interpretation and no conclusions.** Those go
  in section 5.
- Equipment "in exact detail", **including software written to do the testing**.
  If that needs its own subsection it becomes 4.2 and the tests become 4.3.
- Guideline 10–15 pages.

**Figures and tables:**
- **Captions below** figures *and* tables. No title or description at the top as
  well.
- **Number and caption both bold**, `Figure 1.` with a period separator.
  `epr400.cls:377` already sets `labelfont=bf, textfont=bf, labelsep=period`.
- **Figures and tables numbered separately from 1.**
- **No frames around figures.**
- **Minimum 8 pt text on figures**, one font family and size across all of them.
  Sans-serif allowed on figures and captions, not in body text.
- Centred; not smaller than about 8 cm × 8 cm; **may not flow across a page
  break** — so no `longtable`.
- **Every figure, table and equation must be referred to in the text.**
- Credit figures taken from other sources in the caption.
- No long program listings; short excerpts only, handled as figures.

**Other rules worth knowing:**
- **First person is allowed only in Part 1.** Part 3 must be impersonal.
- "Never commence a **section** with a figure" — a section rule, not a page rule.
- Section 1 must be split into (i) background, 3–5 pages, and (ii) how it was
  applied, 1–3 pages, and must **cite at least six technical journal articles**
  — conference papers, theses, preprints and websites do not count.
- Section 1 must merge sources into a unified view, not summarise article by
  article.
- The design-summary table's third column must be a one-word status,
  *completed* or *incomplete*.
- Declarations 5 (AI usage) and 6 (software libraries) **must be modified to
  reflect the actual facts**. The template's blanket denials are placeholders.
  **AI tool use must be clearly indicated.**
- The per-section page guidelines sum to 77 against a cap of 70, so they cannot
  all be met.

### 4.2 Requirements being qualified

```
R1  Reproduce text in the style of any writer of the training set,
    at least 85 % character accuracy and 85 % stylistic correlation.
R2  Decode the text and identify the writer with an overall accuracy
    of at least 95 %.
R3  Reproduce text at no more than 4 s per character.
R4  Identify the writer and decode the text within 2 + 0.025x s for x characters.
R5  Absolute gantry positional error of at most 0.5 mm.
R6  At least 10 writing styles trained and stored permanently.
FC1 Operate on pages photographed by the system's own camera, indoor lighting.
FC2 Operate on ordinary unruled A4 with a ballpoint pen.
```

**Verdicts on current evidence:**

| | Verdict | Evidence |
|---|---|---|
| R1 | **Partially met** | char 84.41 %, **0.59 below target**; style side met at 86.4 % lines / 14 of 14 pages |
| R2 | **Partially met** | writer 99.35 % (exceeds); text 91.02 % char, 56.66 % word (below) |
| R3 | **Met** | worst 2.705 s/char against 4 |
| R4 | **Not met** | **14 of 14 pages exceeded**, by 10.3–27.4 s |
| R5 | **Met on n = 1** | worst side 0.30 mm against 0.5 |
| R6 | **Met** | 11 styles stored against ≥10 |

**Which evidence proves what.** R2 is proved on **real handwriting** — the 155
held-out lines — and qualifies the **classifier**. R1 is proved on **the robot's
own output** — the 184 lines of 14 pages — and qualifies the **synthesis and
gantry**; there the classifier is the measuring instrument and is only
admissible because R2 established it independently. Do not mix the two.

### 4.3 State of the v6 report

Section 4 was **rewritten on 2026-10-31** to the structure above, with six
qualification tests and a shared test-material subsection. Sections 1, 2, 3, 5
and 6 have not been rewritten against the new measurements.

The design section (section 3) absorbed the former appendices: `Appendix_hw`,
`Appendix_seg`, `Appendix_rec`, `Appendix_wid`, `Appendix_syn` and
`Appendix_more` are **commented out of `Appendix.tex` but still present on
disk**. `verify_report.py` now follows `\input` so it ignores them; anything
that scans the folder directly will report 47 phantom duplicate labels.

Figures for section 4 are generated by `make_results_figures.py` from
`results/palette.py`: `res_confusion_reproduced`, `res_confusion_holdout`,
`res_blocks`, `res_gantry_geometry`, `res_style_signature`, `res_per_author`,
`res_connectedness`.

**Colour palette, by the student's instruction:** black, silver, white and
shades of blue only. No red, green, amber or orange, **including for pass/fail**
— use dark blue for met and silver for unmet, distinguished by saturation, and
say which is which in the caption. Older section 3 figures still use red/green
/amber and are inconsistent with this.

**Known outstanding violations** from `verify_report.py` (built files only):

| Check | Count |
|---|---|
| `\caption` outside a float | 17 |
| equations never referred to | 3 |
| floats never referred to | 21 |
| citations off the keep-list | 4 (`iamdb` ×2, `duda1972` ×2) |

Also outstanding: the **length**. Before the section 4 rewrite the main report
was **117 pages against a hard cap of 70**, with the design section alone at
about 76.

### 4.4 Citations

`REFERENCE_AUDIT.md` sets the rule, and it must be obeyed:

> a locator is written only after the source was opened and the claim found.
> Nothing here is inferred from a page range, a search-result snippet or memory.

31 of 57 sources were obtained; 26 could not be. 33 of 81 design citations have
verified locators. **Do not invent page numbers.** Each design file ends with a
marker naming exactly which sources still need opening.

### 4.5 The comparison report

`COMPARISON_23531003.md` holds findings from reading another student's full
111-page report. The student's own assessment is that it is better than theirs
and that this is a fact, not an opinion; its structure, evidence density and
figure flow are the target to match.

---

## 5. Methods worth reusing

### 5.1 Statistics
- **Wilson score intervals** for proportions, not the normal approximation,
  which runs past 100 % at the sample sizes here.
- **Micro averaging** for character and word accuracy: one sum of edit distances
  over one sum of reference characters. The macro mean over-weights short lines
  — on the 184-line set the two differ by 2.4 points.
- Report **Pearson and Spearman together** at n = 11; where they disagree
  sharply the relationship is one point, not a trend.

### 5.2 Classifying G-code
Render the pen-down moves to an image, segment it like a page, classify the
crops. Identifies the writer of a job before anything is drawn, which separates
synthesis quality from gantry fidelity.

### 5.3 Attributing the timing log
Four independent keys, used together:
1. the student's own markers typed into the log;
2. **the character count of each writer's own closing sentence**, which is an
   exact integer key — but 150, 152 and 153 all share the same IAM sentence
   (82 chars) and 588 and yeukita are both 91, so it cannot separate those;
3. **feed-corrected drawing time** against per-writer ink regenerated from the
   synthesiser — only valid once the one run at a different feed is removed;
4. timestamps.

**A naive text search for writer names in the log fails badly** — it matches
`151.49 s total` as writer 151, `150.59 s to spare` as 150, and
`github.com/Dylan-Braune` as dylan, then propagates those through everything
below.

### 5.4 Detecting bad crops
A merged crop is roughly twice the height of its neighbours. Comparing each
crop's height against the page median at a 1.75× threshold found all 13 merged
crops in the 98-page corpus. **A page-level assertion that crop count equals
transcript line count would have caught every one of them** and is worth adding
permanently.

---

## 6. Known defects, open items and decisions outstanding

### 6.1 Defects found and not yet fixed

| Defect | Where | Effect |
|---|---|---|
| **`StandUp` mis-rotates sparse pages** | `segment.py:1320`, `AXIS_TURN_MARGIN = 1.35` | 3 of 14 pages. Dense pages score 2.26–2.69 on the axis ratio; the sparse ones 0.99, 0.42, 1.29. Costs **9/14 → 12/14** on writer ID. No threshold fixes all three; needs an ink-anchored second signal |
| **`SeamCuts` IndexError** | `segment.py:927` | `centres[0]` indexed without a non-empty check; `/api/classify` returns 500 instead of "no lines found" |
| **G-code clamping** | runtime | `clamped out-of-bounds target (5.1, 257.6) -> (5.1, 250.3)` recurs; something emits Y≈257 when usable travel ends at 250.3 |
| **Pen-state switch not confirming** | runtime | `Pen-up did not confirm -- continuing anyway` mid-job. If it fails during a travel move the pen draws across the page |
| **10 merged crops on writer 384** | extraction | see 3.5 |
| **`h06-082.png` never loaded** | dataset | 1 of 384's 10 pages missing |
| **Reader never trained on 384** | corpora | see 3.6 |
| **Train/serve stroke-norm mismatch** | `server.py:216–217` | shape namer served unnormalised crops |
| **`DecodeCrop` PNG/zlib** | `train.py:621` | **FIXED 2026-10-31.** IAM crops are PNG, personal are zlib `.npy`; the function read only the second, so `train.py author` crashed on all IAM data |

### 6.2 Decisions the student has to make

1. **Install the scratch reader or not.** It is better (91.60 vs 91.02 overall,
   94.42 vs 93.33 on clean lines) but the server loads `joint_best`. Appendix 4
   requires the report to describe the system **as submitted**.
2. **Which stroke-normalisation width to deploy.** 0.17 leads, but the 0.14
   control was never trained so "better than current" is unproven. Installing
   any of them requires `PAGE_NAMER_STROKE_NORM = True` **and**
   `recognise.PenWidthXh` set to match.
3. **Whether to retrain the namer from first principles.** Section 3.5 of the
   report states "No machine-learning library is used anywhere in this
   subsection", and the sweep models were trained with torch on Colab.
   Deploying one without retraining makes that sentence false.
4. **The 70-page cap.** What moves to the appendix.
5. **Whether to commit `ReportResults/`** (31 MB, inside a tracked folder).

### 6.3 Work planned

- More square and triangle measurements, Monday 2026-11-02, to give R5 a
  repeatability figure. Measure the **diagonals** too.
- Possible reader retrain or parameter sweep — **after** the 384 re-extraction,
  not before, or the sweep optimises against corrupted targets.

---

## 7. Working conventions

- **The student pushes. Never push.**
- Scripts and LaTeX containing backslashes must be written with a file-writing
  tool, not a shell heredoc — the Bash tool collapses `\\` to `\` and silently
  corrupts them. This has caused real breakage more than once.
- Temporary scripts belong in the session scratchpad, not the repository.
- Downloaded files are untrusted: keep them in their own directory and pass
  paths as arguments.
- `\mustdo` and `\mustcomplete` blocks mark **report work the student still has
  to do**. They are not features and not placeholders for missing capability —
  the underlying engineering is done.
- Figures: black, silver, white, blues. See 4.3.
