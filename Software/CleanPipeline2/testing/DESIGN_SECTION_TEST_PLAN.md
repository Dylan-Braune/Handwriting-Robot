# Design-section test plan — master list

Cross-checked against the 364-function Code Theory manual (`testing/results/CodeTheory/main.pdf`)
module by module, plus the QTP draft and the earlier capture plan from this conversation.
"NEW" = found while reading the code-theory table of contents, not in the earlier plan.
Everything marked **no new capture** runs on photos/data you already have.

## 1. Hardware / gantry (3.3)

| Test | Est. time | Purpose | Proves |
|---|---|---|---|
| Bearing/rail friction write-up | done, write-up only | document existing result | mechanical design choice justified |
| Simple-shape verification (grid, calibration, endstops) | part of gantry day | — | see Part A below |
| Pen up/down timing write-up | done, write-up only | document existing result | pen mechanism tuned correctly |
| Motor step-speed table | 20 min, data exists | document tested speed range | driver/motor sized correctly |
| **Bresenham step interpolation diagram** (NEW) | 30 min | show the step-interpolation algorithm on one real diagonal move (ideal line vs actual stepped path) | the motion-control maths is first-principles and correctly approximates a straight line |

### Part A — one physical gantry day (~6-7 hrs on site)
Fix the X-axis direction-pin fault first; everything below is blocked on it.
1. Fresh calibration → draw the QT5 grid (≥40 crosses, 10mm spacing), repeat ×5 full cycles → scan each at 600dpi. Doubles as the 3.3.5 "known-dimension shape" check.
2. One sheet, same word written twice (QT5 repeatability overlay).
3. QT1(d): write one sentence in a chosen style, count visibly-matching characters (need ≥85%).
4. Stopwatch-time ~20 writing jobs at N=14 across the 5 sentence lengths (feeds QT3).

## 2. Page conditioning & segmentation (3.4)

| Test | Est. time | Purpose | Proves |
|---|---|---|---|
| Lighting/shadow/overlap photo sweep (10-15 photos, one page re-shot varying one factor at a time) | 45 min on-site | isolate what breaks segmentation | design choices (adaptive threshold, flat-field correction) are earned, not decorative |
| Before/after pipeline montage (page conditioning → binarize → deskew → rule removal → line-cut) | 1 hr, **no new capture** | visual step-by-step on one real photo | ties every equation in 3.4.1 directly to a real intermediate image |
| **Otsu threshold worked example** (NEW) | 20 min, no new capture | show the histogram and chosen threshold on one real crop | theory→calculation link for binarisation |
| **Deskew before/after** (NEW) | 20 min, no new capture | show a skewed real photo and its corrected version, with the measured `SkewScore` before/after | proves `EstimateSlantDeg`/`Deskew` works on real data, not just synthetic |
| **Ruled-line/printed-structure removal before/after** (NEW) | 20 min, no new capture | show a ruled page before and after `RemovePrintedStructure`/`DropArtifactPieces` | proves the structure-removal stage, which your draft mentions but doesn't yet have a dedicated figure for |
| **Line-separation seam visualisation** (NEW) | 30 min, no new capture | overlay the `LeastCostSeam`/`SeamCuts` cut path on a real page with touching/overlapping lines | this is your draft's "following tails and cutting" — currently only described in words |
| **Fragment ownership example** (NEW) | 20 min, no new capture | show a stray ink fragment (dot, diacritic, descender tail) before and after `AssignOwnership`/`_FragmentsToNeighbour` assigns it to the right line | proves the "mess tagging" step your draft lists but doesn't yet demonstrate |
| Line-split mismatch / coefficient comparison (your own draft item) | 1-2 hrs, mostly reusing old pipeline experiments | compare different tail-following/threshold coefficients | justifies the specific constants chosen |

## 3. Text & writer classification (not yet drafted)

| Test | Est. time | Purpose | Proves |
|---|---|---|---|
| Conv/pooling layer visualisation (2-3 stages, one real line) | 45 min | make "what a conv layer does" concrete | theory→implementation link |
| CTC output analysis for one real line | 30 min | show raw per-timestep probabilities and greedy-decode collapse | ties Equation 65/CTC theory to a real example |
| Disconnected vs. cursive classification comparison | 30 min | same analysis on a clean line and a cursive line side by side | honestly shows where/why recognition gets harder |
| Author-feature differentiation (2D projection of pooled features, 3-4 authors) | 45 min | visual proof the classifier learns author-distinguishing structure | not memorisation |
| Library vs. first-principles agreement | **0 — already done** | — | 100.00% top-1 agreement, max logit diff ~7.6×10⁻⁶ over 1358 real lines, already measured this project |
| Lighting/crop-condition effect on the classifier | 30 min, **reuses §2's photo sweep** | run the classifier on the same lighting-variation photos | second use of one capture session |
| **Stroke-normalisation before/after** (NEW) | 20 min, no new capture | show `StrokeNormalize` turning a real cursive line into constant-pen-width skeleton-redraw | proves the shape classifier's input pipeline, and doubles as evidence for §4 below |

## 4. Style extraction / glyph library (not yet drafted)

| Test | Est. time | Purpose | Proves |
|---|---|---|---|
| Style-measurement diagram (slant, x-height, spacing annotated on a real line) | 30 min | theory-to-calculation link | ties formulas to a real measured example |
| Connectedness measurement, cursive vs. disconnected | 20 min | show how the metric is counted on two real lines | concrete illustration of a core profile statistic |
| Forced-alignment cutting, clean vs. cursive | **0 — already done** | — | reuse this session's thiya cut-boundary visualisation and glyph-averaging sheets directly |
| Medoid vs. median prototype comparison | **0 — already done** | — | reuse this session's glyph-averaging sheets (abhinav/yeukita/thiya) |
| Cross-author prior / `priorD` quality filtering | **0 — already done** | — | reuse this session's `priorD` distribution plot and histogram |
| **Skeleton-to-polyline tracing chain** (NEW) | 30 min, no new capture | show one real glyph's skeleton, then traced, then `_PruneSpurs`/`_JoinAtJunctions` cleaning it, then `SimplifyPolyline` | proves the raw-skeleton-to-clean-stroke pipeline, currently undemonstrated |
| **Ligature-tail trimming before/after** (NEW) | 20 min, no new capture | show a cut cursive glyph before and after `_TrimLigatureTails` | proves a specific named cleanup step your style theory should mention |
| Final profile-statistics sanity check (NEW, optional) | 20 min | show one author's computed `BuildAuthorProfile` stats next to a direct measurement on their real page | cheap extra confidence, not essential |

## 5. Handwriting synthesis (not yet drafted)

| Test | Est. time | Purpose | Proves |
|---|---|---|---|
| Glyph merging/connection demo | 30 min | two raw glyph pieces before/after joining | joining mechanism works |
| Feature application (slant/spacing before vs. after) | 20 min | one glyph raw vs. styled | style-application step works |
| Small illustrative best-of-N sweep (2-3 authors, N∈{1,8,30}, one sentence) | 1 hr, reuses QT1 code | show the mechanism's trend, not the full proof | QT1 stays the rigorous version; this is for understanding |
| Anchor-glyph comparison (`_FB` vs. `_CURSIVE`) | **0 — already done** | — | reuse this session's anchor-glyph sheet and the hand-coded-vs-Hershey-font comparison |
| Legibility-level sweep (own glyphs → default → full anchor) | **0 — already done** | — | reuse this session's thiya legibility sweep and quantitative anchor-level test |
| Pen-plotting optimisation effect (travel distance before/after) | 20 min | one real line's stroke order before/after `OptimizeStrokeOrder` | optimisation has a measurable effect |
| G-code snippet figure | 15 min | short generated G-code next to its trajectory | satisfies the report's "code as figure" requirement |
| **Swap-probability curve** (NEW) | 15 min, no new capture | plot `_SwapProb` as a function of `priorD` | makes the anchor-substitution *decision rule* visible, not just its outcome |
| **Repair-pass before/after** (NEW) | 30 min, no new capture | show 2-3 weak characters (flagged by `_WeakChars`/forced alignment) before and after the repair pass fixes them | demonstrates a distinct mechanism (`SynthesizeLegible`'s repair rounds) you haven't shown yet, separate from the swap/anchor logic |
| Style calibration before/after (NEW, optional) | 20 min | one author with `StyleCalibration` on vs. off | shows the calibration step has a real, visible effect |

## Reuse map (unchanged from before, still the main time-saver)

- §2's lighting-sweep photos → reused for §3's lighting-robustness evidence.
- This session's thiya forced-alignment visualisation → directly becomes §4's "cursive breaks simple forced alignment" figure.
- This session's from-scratch-vs-framework agreement numbers → §3's library-comparison evidence.
- This session's anchor-glyph and legibility-sweep work → §5's anchor/legibility figures, in full.
- QT1's full infrastructure → both the official qualification test and §5's small best-of-N illustration.
- One physical gantry day → hardware 3.3.5, QT1(d), QT3's writing-time, and QT5, together.

## Net new work from this review

Everything marked NEW above needs **no new photography or gantry time** — it's all scripting against
data and models you already have. Realistic total for the NEW items alone: roughly 4-5 hours of
focused scripting/rendering, entirely separate from the one physical capture day in Part A.
