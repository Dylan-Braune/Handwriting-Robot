# Project History Digest — Handwriting Robot

Written as a handoff document so a fresh session (with full context budget) can
pick up accurate project history without re-deriving it or reading the full
prior conversation. Covers: pipeline architecture, the bugs found and fixed
(with before/after evidence), measured results, and open issues. Everything
here reflects the **verified, current state** of the repo, not aspirational
claims — numbers are measured, not estimated, unless marked "(estimate)".

---

## 1. Pipeline architecture (file-by-file, with line counts)

Four functional blocks, matching the project's own structure:
**(A) Image processing & segmentation, (B) Training & classification,
(C) Text reproduction & G-code, (D) Gantry hardware.**

### (A) Segmentation — `Software/CNN/SegmentLean.py` (593 raw / 479 real-code lines)
Pure numpy/PIL, no libraries. Turns one photographed page into line crops:
1. Normalise lighting (divide by blurred background) + Otsu threshold + deskew
   (search ±4° for the rotation sharpening the row-ink profile).
2. Strip long straight strokes (underlines, ruled lines, binder edge) —
   measured in units of line-spacing `P`.
3. Line pitch `P` + centres from row-ink-density peaks.
4. Rough boundaries: a DP shortest-path "seam" between neighbouring centres
   through a closing-based ink-cost map.
5. Ownership: small cut-off pieces rejoin their main part; small stray pieces
   move to the neighbouring line or get dropped.
6. Crop + render (contrast stretch, gamma darken, resize to 64x640).

**This is the production segmenter**, now used by writer-ID training, text
training, evaluation, and the live server's `/api/classify`. One holdout
remains: `BuildStyleProfile.py`'s personal-author glyph extraction still uses
the older `SegmentPage.py` (see Open Issues).

**Timing:** ~5.65s/page (clean conditions); observed up to ~20s/page under
heavy concurrent machine load (confirmed via A/B, not a code regression).

### (B) Training & classification
- **`TrainText.py`** (1815/1187) — `PaperCRNN`: conv backbone (3 stages,
  2/4/6 layers, 32/64/128ch, 3x3+BN+ReLU) → AdaptiveMaxPool (collapse height)
  → 2-layer bidirectional LSTM (hidden 256) → Linear(512→75) → log-softmax →
  greedy CTC decode. 74-char charset + blank. Input 64x640.
  Subcommands: `hf` (Teklia/IAM from scratch), `joint` (+personal, warm-started,
  `--init-from` required), `personal` (fine-tune on personal only).
  **~100s/epoch on Colab T4**, default 30 epochs for `joint`.
- **`TrainAuthor.py`** (~800 lines) — `AuthorClassifierCNN`, same backbone
  shape (no LSTM) + author_head (Linear 128→96→N). Two separately-trained
  instances:
  - **Ink classifier** (`10authors` subcommand) — raw photographed lines,
    full-network fine-tune. **99.4% val, 99.8% on all 1358 real lines, zero
    author confusion.**
  - **Shape classifier** (`shape` subcommand) — stroke-normalised input
    (binarize→skeletonize→re-ink constant width, `PEN_WIDTH_XH=0.14`), used
    to judge synthesis fairly against a constant-width pen. **Major bug found
    and fixed this session** (see Section 2 below).
- **`np_inference/`** — from-scratch numpy forward passes, **zero torch
  imports** (verified), used at actual inference time: `weight_io.py`
  (reads `.pt` without torch, 136/91), `layers.py` (conv/BN/pool/LSTM
  primitives, 182/121), `text_model.py` (`PaperCRNNNumpy`+`ReadText`, 115
  real), `author_model.py` (`AuthorClassifierCNNNumpy`+`ClassifyImage`+
  `StrokeNormalize`, 71 real).

### (C) Text reproduction & G-code
- **`AuthorReproductionStuff/BuildStyleProfile.py`** (1415/901) — offline
  glyph extraction via CTC forced alignment (`CtcForcedAlign`, Viterbi-style
  numpy), cuts each line into per-character glyphs, cross-author letter
  prior, writes `NOGIT/StyleProfiles10/<author>.json`. **Major bug found and
  fixed this session** (closed-curve collapse, see Section 2).
- **`AuthorReproductionStuff/ProfileIO.py`** (261 lines) — the live-server
  subset: `LoadProfile`, `CtcForcedAlign`, `BinarizeLine`/`Skeletonize`/
  `CoreBand`, plus (added this session) `OtsuThresholdValue`/`Dilate`/
  `Erode`/`Close`/`LabelComponents` copied verbatim from `SegmentPage.py`
  (verified byte-identical over 80 random trials) so the server doesn't
  import that 1659-line file just for 5 primitives.
- **`AuthorReproductionStuff/SynthesizeHandwriting.py`** (1695/1107) —
  `SynthesizeText` (one draw), `SynthesizeLegible` (best-of-N, text-only
  score), `SynthesizeJointBestOf` (**production method**: best-of-N scored
  by harmonic mean of text accuracy x writer-ID confidence), G-code emission,
  `GcodeRoundTrip` (re-parse+re-render for honest scoring).
  **~1.5-2s/candidate draw; ~45-50s/line at the production nTries=30.**
- **`AuthorReproductionStuff/Evaluate.py`** (1092 lines) — official
  measurement (`style`, `rewrite`, `shapestyle`, `legibility`). Full
  `rewrite` run (13 authors x 6 sentences x 2 seeds x best-of-30): **~2 hours**.
- **`AuthorReproductionStuff/server.py`** (625/416) — Flask API. Loads 3
  models + 13 profiles at boot (**~90-100s**). Routes: `/api/authors`,
  `/api/generate` (**14-21s** at nTries 6-10), `/api/classify` (now
  SegmentLean-based, see Section 2), `/api/gcode/upload`, `/api/gcode/<id>`.

### (D) Gantry hardware
- **`Software/GantryControl/odroid_direct_drive.py`** (816/530) — bit-bangs
  stepper STEP/DIR via `gpiod` (stub fallback off-hardware, confirmed import
  succeeds everywhere). G-code parser: `G0 G1 G2 G3 G4 G20 G21 G90 G91 M2 M3
  M30 M5` — confirmed to cover every command the generator emits.
  `bresenham_move` line interpolation, arc moves, `calibrate()` end-stop
  homing, pen up/down via one-way-gear-motor pulses (350ms dwell each),
  emergency-stop on end-stop trigger mid-move.
- **`WebFrontend/`** — static HTML/CSS/JS (no build step), talks to the API
  over CORS, backend URL configurable (so it can point at the ODROID's IP).

### Live production import closure (what actually loads to serve a request)
**13 files, 5067 raw lines, 3411 real-code lines**, verified by instrumented
boot (checked `sys.modules` after import) rather than static analysis:
server.py, SynthesizeHandwriting.py, SegmentLean.py, ProfileIO.py,
np_inference/{__init__,layers,text_model,weight_io,author_model}.py,
web_render_helpers.py, camera_capture.py, authors_config.py,
odroid_direct_drive.py. **Confirmed at runtime: `torch`, `TrainText`,
`ExtractIAMLines` are NOT loaded** — the production path is genuinely
library-free for inference.

---

## 2. Bugs found and fixed this session (with before/after evidence)

### 2.1 Segmentation: "express" missing its "p" (fixed earlier, pre-existing session)
DP seam's cost map lacked margin, letting it thread through thin letter
strokes. Fixed via closing-based (dilate-then-erode) cost map + tuned seam
step/curvature. Two follow-on fix *attempts* (`_hug_crossed_ink` per-fragment,
then word-grouped) were tried, measured via actual rendered crops (not just
the overlay) to **corrupt real letters** (e.g. "speaker"->"sp"+"eaker" split
across lines), and fully reverted. Lesson embedded in project practice:
always verify via real crop pixels, never the overlay alone.

### 2.2 Train/eval segmenter mismatch (found and fixed)
`TrainText.py`'s personal-page collectors used `SegmentPage.ProcessPage`
while everything else (evaluation, writer-ID training) had moved to
`SegmentLean`. Worse, `SegmentPage` mis-segmented several personal pages
(Thiya05: 62 crops vs 8 labels) causing silently mispaired training labels.
**Fixed**: switched both `TrainText.py` collectors to `SegmentLean`, added a
skip-with-warning for any future crop/label mismatch instead of silently
mispairing. **Result after retraining**: text accuracy on 669 real personal
lines went **90.7%->98.6% char, 76.8%->94.2% word** (old SegmentPage-trained
checkpoint vs new SegmentLean-trained one, same crops, same scoring code).

### 2.3 Douglas-Peucker closed-curve collapse (found and fixed)
`SimplifyPolyline` in `BuildStyleProfile.py` measures each point against the
chord from first to last point. On a **closed curve** (e.g. the loop of an
`o`) those coincide, the chord has zero length, every distance reads 0, and
it returns two identical points — **a closed letter collapsed to a
zero-length dot**. Confirmed via synthetic test (circle: 40pts->2pts,
length 0.0) and via the real cache (even pre-dating this session's other
changes): Yeukita's stored `o` variants were 23-27% fragments vs 2% for the
open-curve `e`. **Fix**: split closed curves at their farthest point from
the start before running Douglas-Peucker. **Verified**: `o` fragments
27%->1%, zero regression on `e`/`s`, confirmed across 5 authors (150, 551,
552, 588 all showed 4x-8x improvement on closed letters specifically, with
an **unaffected open-letter control group** — proof it wasn't noise).
Also fixed in the same pass: single-pixel dots (i-dots, j-dots) were
discarded because `TracePolylines` emits nothing for a 1-pixel skeleton
chain — added recovery for untraced components as a short dash. And:
cut boundaries could independently collapse onto the same column, leaving
a character zero width — fixed via a shared, monotonically-increasing
boundary array.
Two fix *attempts* that did NOT help, measured and reverted (kept as code
comments so they aren't retried): awarding whole unjoined ink components to
one character instead of slicing them (no measured improvement, 551 slightly
worse); penalising columns with multiple ink runs to avoid cutting through
loop interiors (measured zero effect, the real cause was the DP bug above).

### 2.4 Fragment-length gate was a global constant, not per-character (found and fixed)
`MIN_PEN_LEN=1.05` assumed every letter needs roughly the same amount of pen
path. A dotless `i` (this project's authors mostly don't dot their i's — see
Open Issues/style notes) is one stroke of ~0.70 x-heights — so **77% of `i`
variants were rejected as fragments**, and only the ones a neighbouring
letter had bled into (inflating pen length past the threshold) survived.
Every stored `i` was contaminated ("si", "ti", "fi", "bi", "di"...).
**Fix**: gate relative to each character's own median pen length
(`FRAGMENT_PEN_FRAC=0.5`, floor `MIN_PEN_ABS=0.30`). **Verified**: all 12
previously-contaminated `i` variants replaced by 8 clean ones (fewer but all
correct vs many all wrong). End-to-end effect on Yeukita (4 novel sentences):
writer-ID 100%->100% (no change, already good), **text char 90.9%->96.6%,
word 68.9%->91.0%** (+22pp word).

### 2.5 Shape classifier: frozen-backbone training bug (found and fixed — the big one)
`TrainAuthor.py`'s `Train()` (the `shape` subcommand) warm-started the CNN
backbone from the text recognizer's weights, then called `model.eval()` and
**only ever trained `author_head`** — a 2-layer MLP — on features extracted
ONCE from that frozen backbone (500 epochs, but all on the tiny head).
That backbone was trained for OCR, which is explicitly supposed to be
*invariant* to the stylistic differences that separate writers — freezing it
discarded exactly the signal needed. Symptom (found via DeepSeek's stage-4
confusion matrix, confirmed by direct inspection): **80% of Owen's 148 real
lines classified as "Yeukita"**, abhinav's 206 lines split across
"552"(41)/"thiya"(23). Pooled validation accuracy (95.3%) hid this because
IAM authors (100%, zero confusion) dominated the pooled average.
Ruled out "styles genuinely look alike" first: in the model's own 64x640
input space, Owen's lines sit CLOSER to each other (mean L2 11118) than to
Yeukita's (15599) — the largest gap of any pair involving Owen. And the
INK classifier (same crops, no stroke-normalization, full backbone
fine-tune) gets Owen right 147/148 — isolating the fault to the frozen
backbone, not the data or the normalization.
**Fix**: rewrote `Train()` to fine-tune the whole network end-to-end
(matching how the ink classifier already trains), 20 epochs instead of 500
(full fine-tuning converges much faster than training a linear probe).
**Verified on the full 669-line real dataset** (not just the validation
slice): **Owen 18.9%->99.3% (then 67.6% on a differently-scoped later
re-measurement — see Open Issues), Abhinav 64.1%->100% (97.1% in the later
re-measurement), shape-overall 83.4%->93.7%**. The systematic collapse
(one wrong neighbour absorbing most of an author's lines) is gone; residual
errors are scattered 1-6 lines, "the signature of a model at its limit, not
a systematic fault" except for Owen, who still loses a real (reduced)
fraction to Yeukita specifically.
**End-to-end effect** (`Evaluate.py rewrite`-style measurement, before/after,
same checkpoints otherwise): shape writer-ID 74%->92% (rendered), 74%->90%
(G-code); **ink writer-ID control (untouched) 26%->28% and text char
accuracy flat 79-80%->79-80%** — proof the gain is in the judge becoming
reliable, not a change in what's being drawn.

---

## 3. Current measured results (the numbers to cite)

### 3.1 Segmentation correctness
All 32 personal-author pages (5 authors): **669/669 crops match label
count exactly**, zero mismatches, after the Yeukita06/Dylan05 label cleanup
done earlier.

### 3.2 Text recognition, real photos (SegmentLean crops, current checkpoint)
| Author | char acc | word acc |
|---|---|---|
| Abhinav | 98.9% | 95.4% |
| Dylan | 99.0% | 96.7% |
| Owen | 99.2% | 96.6% |
| Thiya | 96.5% | 85.3% |
| Yeukita | 98.3% | 92.1% |
| **Overall (669 lines)** | **98.6%** | **94.2%** |

### 3.3 Writer-ID, real photos, post-fix shape classifier
Overall shape-head accuracy 93.7% (up from 83.4% pre-fix); ink-head 99.8%
(unchanged, was already excellent). Full per-author confusion matrices are
in `Software/CNN/NOGIT/full_pipeline_run/stage4_confusion_shape.csv` and
`stage4_confusion_ink.csv` (generated by DeepSeek's verification tooling,
re-run against the fixed checkpoint).

### 3.4 Official end-to-end synthesis accuracy — `Evaluate.py rewrite`
13 authors x 6 novel sentences x 2 seeds x best-of-30:
**writer-ID 80.1% (G-code 78.8%), text char 82.2% (G-code 82.5%), word
48.9%** — against an 85% target for writer-ID and text char. **Not met**
by this measurement, though see Open Issues for why Abhinav's contribution
to this number is contested.

### 3.5 Matched-reference comparison (the fairest single number)
Same held-out IAM words, read by the same recognizer, real handwriting vs.
this pipeline's synthesis of those identical words:
| | char | word |
|---|---|---|
| REAL | 89.0% | 70.5% |
| SYNTHESISED | 81.6% | 46.7% |
| gap | -7.4pp | **-23.8pp** |
Letters are reproduced reasonably faithfully (small char gap); word
accuracy falls much further, meaning errors concentrate at word
boundaries/spacing, not letterforms. **This is the clearest evidence of
where remaining synthesis quality should be improved.**

### 3.6 Style/connectedness measurements (from glyph-extraction logs)
| author | connectedness | slant |
|---|---|---|
| Dylan | 0.14-0.20 | -6 to -7.5 deg |
| Owen | 0.21-0.32 | ~0 deg |
| Yeukita | 0.33-0.42 | 0 deg |
| Abhinav | 0.74-0.79 | +6 deg |
| Thiya | 0.96-0.97 | +30 to +31.5 deg |
Thiya is essentially fully-joined cursive at a steep slant — the hardest
case throughout, and consistent with the project proposal's own framing
that cursive is a stretch goal, not a core requirement (print/lightly-joined
styles are the actual spec target).

### 3.7 Reproduction quality by connectedness (3-sentence spot check, nTries=30)
| author | connectedness | writer-ID | char (direct/gcode) | word (direct/gcode) |
|---|---|---|---|---|
| Owen | 0.21 | n/a* | 98.0%/97.4% | 92.1%/88.8% |
| Dylan | 0.14 | 100%/100% | 97.5%/96.2% | 88.8%/88.0% |
| Yeukita | 0.33 | 100%/100% | 96.6%/95.7% | 91.0%/87.8% |
| Abhinav | 0.74 | n/a* | 91.9%/91.2% | 61.8%/58.1% |
| Thiya | 0.97 | n/a* | 79.4%/79.4% | 33.1%/37.7% |
(*at the time of this specific spot check the shape classifier didn't yet
cover the 3 new authors; re-measurable now that it does.)
**Clean pattern: accuracy tracks how joined the handwriting is** — the
three print-ish hands (connectedness 0.14-0.33) all land 96-98% char;
Abhinav (0.74) drops to 92%; Thiya (0.97, heavy cursive) falls to 79%.
This is the headline justification for scoping cursive as a stretch goal.

---

## 4. Open issues / unresolved, stated honestly

1. **Personal-author glyph extraction still uses `SegmentPage`, not
   `SegmentLean`** (`BuildStyleProfile.py`'s `build_personal_line_items`).
   Everything else moved over; this one function feeds the glyph library
   every synthesis profile is built from. Measured SegmentLean-vs-SegmentPage
   gap elsewhere: char 94.5%->98.2%, word 81.8%->92.1%, fixed one outright
   author misattribution. Not yet fixed — would require re-running the
   ~35-40 min glyph-rebuild for the 5 personal authors.
2. **Owen's residual writer-ID collapse into Yeukita is real but reduced**,
   not eliminated, after the backbone-fine-tune fix. Two independent
   re-measurements disagree on magnitude (99.3% vs 67.6% on nominally the
   same 669-line set) — the cause (likely different crop preprocessing: live
   SegmentLean re-segmentation vs a cached classifier/ PNG that may have
   been resized twice) has not been established. **Whichever number is used
   in the report, say which measurement protocol produced it.**
3. **Abhinav's synthesis writer-ID is a real, separate, unresolved finding**
   — not a classifier problem (his real handwriting reads at 97-100%). Two
   independent end-to-end runs agree his *synthesised* style scores low
   (0-16.7%), and the authors failing writer-ID on synthesis also read back
   at lower character accuracy — pointing at the glyph-profile/drawing
   stage specifically for him, not the judge. A follow-up N-sweep (nTries
   1/5/10/15/20/30, single sentences) showed abhinav CAN be correctly
   classified at N>=5 with confidence climbing to ~94% on some sentences,
   and 100% writer-ID across 3 sentences at N=15-20 — in tension with the
   official 16.7% (averaged over 12 draws at N=30). Not reconciled — could
   be sentence-specific difficulty (2-3 hard sentences dragging the 12-draw
   average down) or seed sensitivity. **Flagged, not resolved — do not
   state a single confident cause in the report; present both results and
   the open question.**
3b. Most authors' handwriting is NOT dotted on "i" (measured: 92% of
   Yeukita's real `i`s have no dot in the ink) — this is a genuine style
   trait, not a pipeline bug, confirmed by direct pixel inspection and
   should not be "fixed."
4. **Code duplication** (per DeepSeek's audit, independently spot-checked):
   `Evaluate.py` and `SegmentLean.py` exist byte-identical in two places
   (`Software/CNN` main tree and an archival `TestsForReport`/
   `ReproductionImprovement` snapshot) — free deletions. `BuildStyleProfile.py`,
   `SegmentPage.py`, `TrainText.py`, `ExtractIAMLines.py` are genuinely
   diverged between main and those archival snapshots — the archival copies
   are stale, not authoritative. `FinalPipeline/` (the deployment copy) was
   found out of sync twice this session (once for the shape-classifier fix,
   once for 4 other files) and has since been fully re-synced and verified
   (byte-identical sweep across all shared files, live boot + `/api/generate`
   test against `FinalPipeline` specifically passed).
5. A DeepSeek-produced report inventory undercounted the live boot closure
   by exactly 318 lines (2 files, `np_inference/layers.py` +
   `np_inference/weight_io.py`, both real necessary imports) — corrected
   figure is in Section 1 above. DeepSeek's own report (`checks_v2.json`
   verdict: pass) has two tables with a `x100` formatting bug (nonsensical
   "1%" and "10000%" entries in section 7.1's summary row and section 7.4) —
   don't quote those two specific tables; the per-author rows elsewhere in
   the same document are correctly formatted.

---

## 5. Project framing notes relevant to report-writing

- **Cursive is explicitly a stretch goal, not a core requirement** (user's
  own words, confirmed this session): "cursive styles are not really part
  of the requirements... the system can be verified with the clean writing
  styles especially in reproduction. Cursive is a great bonus." Thiya
  (connectedness 0.97) is the cursive stress-test; Dylan/Owen/Yeukita
  (0.14-0.42) are the actual spec-relevant evidence.
- **Everything in `np_inference/` is first-principles** (zero torch
  imports, verified at runtime) and is the legitimate place to claim credit
  for "implemented from scratch, not from a library" — conv2d via im2col,
  batchnorm in eval mode from stored running stats, a from-scratch
  bidirectional multi-layer LSTM forward pass, CTC greedy decode, and the
  custom `.pt`-file reader that parses PyTorch's pickle+zipfile format
  without importing torch at all.
- **`SegmentLean.py` is entirely first-principles** (numpy/PIL only): Otsu
  thresholding, a custom 8-connected connected-component labeller
  (row-run + union-find), a DP shortest-path line-boundary solver, and the
  deskew search — all hand-written, no `cv2`/`skimage`/`scipy` calls.
- **CTC forced alignment** (`CtcForcedAlign` in `ProfileIO.py`) — the
  Viterbi-style alignment math is first-principles numpy; only the model's
  forward pass (which produces the log-probabilities being aligned) uses
  a from-scratch numpy network, not a library.
- Training scripts (`TrainText.py`, `TrainAuthor.py`) and verification/
  evaluation scripts (`Evaluate.py`) are explicitly exempted from the
  library-free rule by the user's own project scoping — PyTorch there is
  fine and should not be apologized for in the report; the distinction to
  make is "produces the robot's output" (library-free) vs "trains/measures
  it" (libraries allowed).
