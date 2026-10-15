# Code Explained (C): Handwriting Reproduction, Glyph Extraction & G-code Generation

This document explains, function by function, every piece of code in `Software/CNN/AuthorReproductionStuff/` that extracts an author's handwriting style, synthesizes new text in that style, turns it into G-code, serves it over a web API, and measures how well the whole pipeline works. It is written so the author of this project can (a) re-derive their own design decisions from first principles and (b) cite the correct classical literature in the report's Design/Implementation and Literature Study sections.

Files covered, in pipeline order:

1. `BuildStyleProfile.py` — offline: raw page → per-character glyph library + style statistics (`NOGIT/StyleProfiles10/<author>.json`).
2. `ProfileIO.py` — the live-server subset of the above: profile loading, CTC forced alignment, image primitives, skeletonization.
3. `SynthesizeHandwriting.py` — profile + text → pen trajectory → G-code.
4. `server.py` — Flask API wrapping all of the above for the web app / gantry.
5. `Evaluate.py` — the four official measurement scripts (style, rewrite, shapestyle, legibility).
6. `web_render_helpers.py`, `camera_capture.py` — small supporting utilities.

---

## 1. `BuildStyleProfile.py` — offline glyph-library construction

### Role in the pipeline

This is the **training-time-equivalent** step for the synthesis half of the project (it is "offline" in the sense that it is run once per author, by hand, never during a live request). It takes an author's raw handwriting pages (`Data/Datasets/IAMpages10`, or page-segmented personal photos), runs the **frozen** text recognizer (`np_inference/text_model.PaperCRNNNumpy`, the same network trained in `TrainText.py`) in inference-only mode to get per-frame class probabilities, and uses **CTC forced alignment** (implemented in `ProfileIO.py`, imported here) to find out which x-pixel range of each line image corresponds to each character of the (already known) transcript. It then binarizes and skeletonizes the ink in each per-character window, traces the skeleton into ordered polylines, simplifies and normalizes those polylines into an upright, slant-removed, baseline-relative coordinate frame, and stores them as **glyph variants** in a per-author JSON profile. A battery of quality gates (shape-class checks, pen-length fragment gates, cross-author consensus) throws out cuts that are obviously wrong (fragments, contamination from a neighbouring letter, mis-aligned lines). Alongside the glyph library it also measures global style statistics (slant, x-height, ascender/descender reach, stroke width, word spacing, "connectedness" / cursive-joining rate) by direct measurement of the real ink.

The output JSON (`NOGIT/StyleProfiles10/<author>.json`) is the **only** thing `SynthesizeHandwriting.py` and `server.py` need at synthesis time — none of this file's expensive machinery runs live.

### Module-level constants and setup

- `MAX_VARIANTS_PER_CHAR = 12` — cap on how many example shapes of one character are kept per author (keeps profile JSON small and keeps `_GlyphSource`'s per-instance sampling from being swamped by near-duplicates).
- `ALIGN_CONF_PCT = 25` — lines whose CTC forced-alignment confidence (mean per-frame log-probability along the Viterbi path, see `CtcForcedAlign`) falls below the 25th percentile of that author's own lines are dropped from glyph extraction; a badly-aligned line has untrustworthy cut boundaries, so keeping it would inject garbage glyphs.
- `FRAGMENT_PEN_FRAC = 0.5`, `MIN_PEN_ABS = 0.30`, `MIN_LONGEST_STROKE = 0.55` — the per-character-relative fragment gate; see the dedicated worked example in **Known limitations / honest caveats** and in `_PenLength`/the variant-pruning loop in `BuildAuthorProfile` below.
- `DOT_MIN_AREA = 0.004` — smallest ink blob (as a fraction of x-height²) still treated as a deliberate dot (i/j dot, full stop) rather than scanner speckle, used by the dot-recovery step in `ExtractLineGlyphs`.
- `MAX_GLYPH_W = 2.1`, `MAX_GLYPH_W_WIDE = 2.9`, `_WIDE_CHARS = {'m','w','M','W'}` — absolute backstop against a stored "glyph" that is actually two letters glued together by a bad cut; wide letters (m, w) get a larger allowance since they are legitimately broader than one x-height.

---

### Skeleton → ordered polylines

#### `TracePolylines(skel)`

**Purpose.** Converts a binary skeleton image (1-pixel-wide centerlines, from `Skeletonize`) into a list of ordered polylines — the actual pen-path geometry — by walking the skeleton's pixel-adjacency graph.

**Inputs.** `skel`: 2-D boolean numpy array, `True` where the 1px-thin centerline exists.

**Outputs.** `list[list[(x, y)]]` — each inner list is one traced chain of skeleton pixel centres, in float (x, y) pixel coordinates, ordered along the chain (not necessarily left-to-right at this stage).

**Inner workings.**
1. Collects skeleton pixels into a Python `set` of `(row, col)` tuples for O(1) adjacency lookup.
2. `nbrs(p)` returns the up-to-8 neighbouring skeleton pixels of `p`, but **suppresses redundant diagonal steps**: if both orthogonal neighbours of a diagonal move are also ink, the diagonal is dropped. This avoids creating spurious degree-3 "staircase" junctions purely from how a diagonal line rasterizes — without this, every diagonal stroke would be mis-read as branching.
3. `deg(p)` = the pixel's graph degree (number of kept neighbours). Degree-2 pixels are "mid-stroke"; degree ≠ 2 pixels are endpoints (degree 1) or junctions (degree ≥ 3).
4. `walk(start, first)` follows a chain from a seed edge, extending through degree-2 pixels only, until it hits an endpoint, a junction, or an edge already used. This is a straightforward **graph traversal / Eulerian-path-style walk** over the adjacency graph — not a named classical algorithm, but a standard technique in skeleton-based vectorization (e.g. the stroke-extraction stage of systems like `potrace`).
5. Seeds are chosen as the non-degree-2 pixels (endpoints/junctions) first, so chains between junctions are captured, and any leftover pure loops (all degree-2, no junction at all — e.g. a perfect 'o') are picked up afterward.
6. Each chain is converted from `(row, col)` to `(x, y)` floats, then passed through `_PruneSpurs` (drop thinning artefacts) and `_JoinAtJunctions` (merge chains that cross through a junction in the same direction, i.e. one continuous pen stroke that happens to self-intersect, like the loop-crossing of a cursive 'e' or 'l').

**Literature connection.** Skeleton tracing/vectorization of thinned raster images is a standard step following **Zhang & Suen (1984)** thinning (see `Skeletonize` in `ProfileIO.py`); the junction-graph walk here is a hand-rolled instance of the general "trace centerlines from a topological skeleton" family used in vectorization tools.

#### `_Dir(pts, fromEnd, n=4)`

**Purpose.** Computes the unit tangent direction at one end of a polyline chain, pointing *away* from the chain (used to decide whether two chains continue "straight through" a junction).

**Inputs.** `pts`: list of (x,y). `fromEnd`: bool, which end. `n=4`: how many points back to measure the direction over (smooths out single-pixel staircase noise).

**Outputs.** A 2-vector unit direction (`np.array([dx, dy])`), or `[1,0]` if the segment is degenerate (near-zero length).

**Maths.** `v = b - a` where `a, b` are points `n` apart near the chosen end; direction = `v / ||v||` (Euclidean norm, `np.hypot`).

#### `_PruneSpurs(chains, minLen=2.5)`

**Purpose.** Deletes very short chains that dangle off a junction with only one shared endpoint — these are thinning artefacts (a 1-pixel "hair" that `Zhang & Suen` thinning sometimes leaves at a junction), not real pen strokes.

**Inner workings.** Counts, for every chain endpoint, how many chains terminate there (`ends` dict). A chain is dropped if its arc length (summed `np.hypot` over consecutive point differences) is below `minLen` **and** at least one of its endpoints is used by only this one chain ("dangling").

#### `_JoinAtJunctions(chains, maxAngleDeg=75.0)`

**Purpose.** Merges pairs of chains that meet at (near-)coincident endpoints and continue in nearly opposite directions (i.e. one is the continuation of the other) — this is what turns two broken half-strokes at a pen self-crossing (the loop of an 'e', the crossing of an 'l' or 't') back into one continuous stroke.

**Inner workings.** Greedy iterative merge: for every ordered pair of chains and every combination of their two ends, if the endpoints are within 1.5px of each other, compute `cosang = -dot(dir_i, dir_j)` (continuing straight through means the two outward tangents point in *opposite* directions, hence the negation) and the angle via `arccos`. The best (smallest-angle) pair under `maxAngleDeg` is merged (concatenated, with orientation flipped as needed), and the loop repeats until no more merges qualify. This is a simple greedy heuristic, not an optimal matching — acceptable here because junctions are rare and the chain count is small per glyph window.

---

### `SimplifyPolyline(poly, eps=0.6)` — Douglas–Peucker, with the closed-curve bug fixed

**Purpose.** Reduces a dense, pixel-level polyline (traced from the skeleton, so it has one point roughly per pixel) down to a small number of vertices that still represent the curve's shape within tolerance `eps` — i.e. classic **polyline simplification**.

**Inputs.** `poly`: list of (x, y) pixel coordinates (floats). `eps`: maximum allowed perpendicular deviation, in the same units (pixels).

**Outputs.** A shorter list of (x, y) points — a subset of the input vertices — such that no discarded point is farther than `eps` from the simplified segment that replaces it.

**The classical Douglas–Peucker algorithm (Douglas & Peucker, 1973, "Algorithms for the reduction of the number of points required to represent a digitized line or its caricature").** Given a polyline from point `p[0]` to `p[-1]`:
1. Treat the straight chord from `p[0]` to `p[-1]` as the candidate replacement.
2. Find the point `p[i]` in between with the **largest perpendicular distance** from that chord.
3. If that maximum distance `d[i] ≤ eps`, the whole chord is an acceptable simplification — return just `[p[0], p[-1]]`.
4. Otherwise, `p[i]` must be kept: **recursively simplify** the two sub-polylines `p[0..i]` and `p[i..end]` separately, and concatenate the results (dropping the duplicate `p[i]` at the join).

This is exactly what the `else` branch of the function implements:
```python
a, b = p[0], p[-1]
ab = b - a
L = np.hypot(*ab) + 1e-9
rel = p - a
d = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / L
i = int(np.argmax(d))
if d[i] > eps:
    return SimplifyPolyline(poly[:i+1], eps)[:-1] + SimplifyPolyline(poly[i:], eps)
return [poly[0], poly[-1]]
```

**The point-to-line-segment (chord) distance formula, worked out.** For a point `P` and a line through `A` and `B`, the perpendicular distance is the magnitude of the cross product of `(B-A)` and `(P-A)`, divided by the length of `(B-A)`:
```
d(P) = | (B.x - A.x)(P.y - A.y) - (B.y - A.y)(P.x - A.x) |  /  ||B - A||
```
This is exactly the 2-D cross-product-based point-to-line distance (the numerator is twice the signed area of triangle `A,B,P`; dividing by the base length `||B-A||` gives the triangle's height from `P`, i.e. the perpendicular distance). In the code, `ab = B - A`, `rel = P - A`, numerator `= ab.x*rel.y - ab.y*rel.x`, denominator `= L = ||ab||`.

**Why this breaks on a closed curve, worked out precisely.** If the polyline is closed (or nearly so), `A = p[0] ≈ p[-1] = B`, so `ab = B - A ≈ (0, 0)` and `L = ||ab|| ≈ 0` (the `+1e-9` only prevents an actual division-by-zero crash, it does not fix the geometry). The point-to-line formula above is only meaningful when `A` and `B` are distinct — it measures distance to the *line through A and B*, and when `A = B` there is no well-defined line/direction, only a single point. Numerically, as `L → 0`, the numerator `(ab.x·rel.y − ab.y·rel.x)` also `→ 0` at the same rate (since `ab → 0` multiplies every term), so the ratio `d = numerator / L` collapses toward 0 for **every** point on the curve, regardless of how far that point actually is from `A`. Concretely: every candidate point then satisfies `d[i] ≤ eps` trivially (all distances read as ~0), so the **very first call** immediately short-circuits to `return [poly[0], poly[-1]]` — i.e. the entire closed loop (e.g. the full circular stroke of a lowercase 'o') is collapsed to the single repeated point `p[0] ≈ p[-1]`, destroying the loop shape completely. The code's own comment records the measured symptom: *"An 'o' came out as a zero-length dot — measured at 23% of Yeukita's stored 'o' variants, against 2% for the open-curve 'e'"* — i.e. closed letterforms (o, e's loop, a's bowl) were disproportionately destroyed while open curves (most of e's outline) were simplified correctly, because an open curve's start and end points are genuinely far apart so its chord has real length and direction.

**The fix.** Before running ordinary Douglas–Peucker, detect a (near-)closed curve: compute `fromStart[k] = ||p[k] - p[0]||` for every point, and check whether the curve's *return distance* `fromStart[-1]` (distance from end back to start) is small relative to the curve's own extent (`max(fromStart)`):
```python
fromStart = np.hypot(*(p - p[0]).T)
if fromStart[-1] <= max(1.5, 0.25 * fromStart.max()):
    far = int(np.argmax(fromStart))
    if 0 < far < len(poly) - 1:
        return SimplifyPolyline(poly[:far+1], eps)[:-1] + SimplifyPolyline(poly[far:], eps)
```
It finds the point `far` that is **farthest from the start** (the "far side" of the loop — e.g. for a circular 'o' traced starting at the top, this is roughly the bottom), and splits the polyline there into two **open** arcs: `p[0..far]` and `p[far..end]`. Each arc now has genuinely distinct, well-separated endpoints, so its own chord has real length and direction, and ordinary Douglas–Peucker on each half behaves correctly — the two simplified arcs are then concatenated (dropping the duplicate midpoint) to give a faithful simplification of the full loop. The threshold `max(1.5, 0.25 * fromStart.max())` says "closed" means the curve comes back to within either 1.5px or a quarter of its own span of where it started — loose enough to catch loops that don't close up pixel-perfectly (as skeletons rarely do).

**Literature connection.** Douglas & Peucker (1973). The closed-curve failure mode and split-at-farthest-point fix is a standard, documented workaround in computational-geometry practice for applying open-curve line-simplification algorithms to closed polygons/contours (conceptually identical to how closed GIS polygons are simplified by picking two anchor vertices and simplifying each half separately).

---

### Glyph extraction from one aligned line

#### `_TrimLigatureTails(strokes, connL, connR, lowY=0.42, minKeep=0.18, climbY=1.6, climbSlope=2.2)`

**Purpose.** A per-character CTC cut in cursive handwriting lands mid-ligature — the stored glyph arrives with half of the connecting stroke to the previous/next letter still attached on each side. This function trims that connector off, **but only on a side that was independently confirmed to be pen-joined** (`connL`/`connR`, computed from the skeleton-crossing test in `ExtractLineGlyphs`, not guessed from shape).

**Inputs.** `strokes`: list of polylines (left-to-right oriented) making up the glyph window. `connL`, `connR`: bools, whether the pen was continuously down crossing into this glyph from the left/right. `lowY`, `minKeep`, `climbY`, `climbSlope`: geometric thresholds described below.

**Outputs.** `(trimmedStrokes, entryPoint, exitPoint)`.

**Inner workings.** `runLen(s, connected, maxHeight, maxSlope)` counts how many leading points of a stroke form a "carrier": points that (a) stay below `maxHeight` in normalized y, (b) keep moving forward (`dx ≥ 0`), and (c) never rise faster than `maxSlope` per unit x. This is tried twice per side with two different ceilings: `lowY=0.42` (a near-horizontal connector between two ordinary x-height letters) and `climbY=1.6, climbSlope=2.2` (a taller, steeper connector climbing into a tall letter, e.g. the join from a low letter into "T" or "h" — the code's comment explains that a single low/gentle threshold never matched this case, leaving "an unmistakable stray diagonal baked into every affected glyph"). The run is only trimmed if it's at least `minKeep` (18%) of the glyph's total x-span — so a genuinely short leading stroke of the letter itself (not a long carrier) is left alone. The function only ever attempts trimming on a side flagged `connL`/`connR = True`: a confirmed pen-lift before the letter means everything there is the letter's own ink.

#### `ExtractLineGlyphs(gray, text, model)`

**Purpose.** The central per-line extraction routine: given one grayscale line-crop image and its ground-truth transcript, returns a per-character glyph dictionary plus line-level statistics. This is the function that chains together CTC forced alignment, ink binarization, skeleton tracing, boundary-column scoring/snapping, per-character cutting, ligature trimming, dot recovery, and normalization into the stored coordinate frame.

**Inputs.** `gray`: grayscale numpy image of one handwriting line. `text`: the known transcript string. `model`: a `PaperCRNNNumpy` instance (frozen, inference only).

**Outputs.** `(glyphs, stats)` where `glyphs` is a list (one entry per character position in `text`, including spaces) of either `None` (unusable) or a dict with keys `char, strokes, width, advance, lead, top, bot, entryX, entryY, exitX, exitY, connL, connR`; `stats` carries `slant, xh, alignConf, strokeW, cuts`. Returns `(None, None)` if alignment or ink banding fails.

**Inner workings, step by step.**

1. **Run the frozen recognizer.** The line is resized with `resize_line_image_fixed` (fit-and-pad, not stretch — see the comment on `frame_x_to_pixel` below) and fed through `model.forward`, giving `(T, C)` log/probability-like outputs over `T` timesteps and `C` classes.
2. **CTC forced alignment** (`CtcForcedAlign`, from `ProfileIO.py` — detailed fully in Section 2) maps the known `text` onto a `(t0, t1)` timestep span per character, plus a confidence `conf` = mean per-frame log-prob along the chosen path.
3. **Ink extraction.** `BinarizeLine(gray)` Otsu-thresholds and rule-line-strips the ink; `CoreBand(ink)` finds the x-height band (baseline `base`, topline `top`); `xh = base - top`. `EstimateSlantDeg(ink)` measures the author's slant angle; `shear = tan(slant)` is used later to de-slant stored coordinates.
4. **Column cut-cost map.** Rather than trusting the recognizer's raw per-character boundaries directly (CTC peaks are typically only 1–2 frames wide and lag the actual ink, so naive boundary columns clip letters to slivers), the code builds a per-column "cut cost" that is cheap exactly where a vertical cut would fall in the *middle of a ligature* (the honest place to cut connected cursive writing): `thin` = how much thinner this column's ink is than the 60th-percentile column (favouring the thin neck of a connecting stroke), `topAbove` = how high above the baseline the column's topmost ink reaches (penalizing cutting through a tall ascender/descender), combined as `cutCost = thin + 1.15*topAbove`, with true gaps (`colProf == 0`) forced to exactly 0 (always the cheapest possible cut). The code's own note explains two *rejected* alternatives that were tried and measured worse: cutting in the de-slanted frame (regressed writer-ID 81.8%→68.6%), and penalizing columns that cross a loop (measured no change — the real cause of broken loop letters was the `SimplifyPolyline` bug above, not cut placement).
5. **Boundary snapping (`_Snap`)** finds, within `snapR = 0.5·xh` of a candidate boundary `x`, the column with the lowest `cutCost + 0.6·(distance from x)/snapR` — i.e. prefer a true gap or a thin/low ligature neck, but don't drift too far from where the recognizer said the boundary should be. (A tried-and-rejected alternative — searching further than `snapR` for a true gap — is noted as measured worse: 12%→18% dropped-glyph rate, because internal word boundaries then got dragged onto distant word-gap columns.)
6. **Frame-to-pixel mapping.** `frame_x_to_pixel(canvas_x, rawW, origH)` (from `TrainText.py`) is the correct inverse of the *fit-and-pad* resize used for recognition — the code flags that a naive `"* rawW / T"` scaling was only valid under an older stretch-to-fill resize and is now wrong because padding means the same canvas x can map to different original x depending on how much of the canvas is real content vs. padding.
7. **A single shared, strictly increasing boundary array** (`bnd[0..nV]`) is built once for the whole line rather than snapping each character's left/right edge independently — the comment explains that independent snapping could let two neighbouring boundaries land on the same column, leaving the character between them an empty, un-glyphed cell.
8. **Whole-line skeletonization and tracing** (`Skeletonize(ink)` then `TracePolylines`) is done **once per line**, not per column-slice, specifically because thinning a truncated column slice on its own creates a spurious horizontal spine artefact right at the cut edge.
9. **Segment assignment and crossing detection.** Each traced polyline is walked and split at the column boundaries (`bounds`); segments are filed into `segsFor[charIndex]`. Whenever a polyline continues across a boundary (`t != curT` but both non-`None`), that is recorded in `crossings` — this is the literal, honest definition of "the pen never lifted between those two letters" used both to set `connL`/`connR` per glyph and (aggregated) to compute `connectedness` for the whole profile.
10. **Dot recovery.** A dot (i-dot, j-dot, full stop) thins under Zhang–Suen skeletonization to a single isolated pixel, which is *not* a 2-point-or-longer polyline, so `TracePolylines` silently drops it — the code's own comment states this was the bug behind a dotless stored 'i' being a bare stem. The fix scans `LabelComponents(ink)` for connected components the tracer produced **no** polyline for (`tracedComps`), and for any such component at least `minDotPx = max(2, round(DOT_MIN_AREA·xh²))` pixels in area, synthesizes a short horizontal dash spanning its own pixel extent and assigns it to whichever character's boundary window contains its centroid x.
11. **Per-character normalization.** For each non-space character with at least one polyline segment: `SimplifyPolyline` (eps=0.6) is applied to each raw segment; coordinates are shifted to the baseline/left-cut origin, scaled by `1/xh` (so 1.0 = one x-height), and **de-slanted**: `xn = xn - shear·yn`. Strokes are oriented left-to-right and sorted by their leftmost x. `_TrimLigatureTails` then removes confirmed connector tails. Finally the glyph is **re-anchored in the frame it will be drawn in**: since synthesis re-applies the shear, the stored x-origin is computed from the *re-sheared* x (`xr = px + shear·py`), not the de-slanted x — otherwise a tall slanted letter would land up to `shear × ascender` x-heights off its intended cell.
12. The output dict per glyph records `char, strokes, width, advance (the recognizer's own cell width / xh), lead` (gap from the cell's left edge to where ink starts — preserved so synthesis keeps the author's own letter-internal spacing), `top, bot` (extreme y, i.e. ascender/descender reach of this instance), `entryX/entryY, exitX/exitY` (where the pen enters/leaves, post-trim — used by synthesis to choose ligature-matching variants), `connL, connR`.
13. `stats` records `slant, xh, alignConf, strokeW = inkArea/skeletonLength` (a rough stroke-thickness estimate: total ink pixels divided by total skeleton arc length — for a stroke of uniform width `w` and length `L`, ink area ≈ `w·L`, so `inkArea/skelLen ≈ w`), and `cuts` (the literal territory each character was cut from, kept for visual debugging/QA overlays).

---

### Variant quality gates

#### `_ClassOk(ch, top, bot)`

**Purpose.** A fast, author-independent sanity check on a glyph instance's vertical extent, using the character's typographic class (ascender `bdfhklt`+uppercase+digits; descender `gjpqy`; x-height-only `acemnorsuvwxz`). Rejects e.g. a variant filed under 'l' whose ink never rises above x-height (it probably captured a neighbouring x-height letter by mistake, not the 'l' itself).

#### `_ResampleN(poly, n)`

**Purpose.** Arc-length-uniform resampling of a polyline to exactly `n` points — a prerequisite for any per-point statistics across multiple variants of different point counts (used by `_AlignStroke`, `_DenoisedPrototype`).

**Maths.** Cumulative arc length `cum[k] = Σ_{i<k} ||p[i+1]-p[i]||`; new sample positions `t = linspace(0, cum[-1], n)`; new coordinates by **linear interpolation** (`np.interp`) of x and y against arc length.

#### `_AlignStroke(s, N)`

**Purpose.** Prepares one stroke for cross-variant averaging: resamples to `N` points, canonically orients it left-to-right, and for a (near-)closed stroke (endpoints within 0.30 x-heights — i.e. a loop), rolls the point order to start at the loop's leftmost point so that loops from different variants are compared starting at a consistent location around the circle.

#### `_DenoisedPrototype(variants, minN=3, N=44)`

**Purpose.** Computes a single, robust, per-stroke-median "ideal" letterform from an author's own variants of one character — cancelling roughly-random cut noise while keeping the author's true letterform (this becomes `idealGlyphs` in the profile, used by `_CoreGlyph`).

**Inner workings.**
1. Groups variants by stroke count and keeps only the **modal group** (the most common stroke count) — mixing a 2-stroke and 3-stroke rendition of the same letter would not average sensibly.
2. Requires at least `minN=3` variants in that group and at most `nc=5` strokes (beyond that, cut noise is assumed too large to denoise reliably).
3. For each stroke position, each variant's stroke is aligned (`_AlignStroke`) to `N=44` points, stacked into an array `stacks[i]` of shape `(nVariants, N, 2)`.
4. **Two iterations of shift-then-remedian**: compute `med = median(A, axis=0)`; for each variant curve, search a small x-shift `dx ∈ linspace(-0.22, 0.22, 9)` that minimizes mean distance to `med`; replace the curve with its best-shifted version; recompute `med`. The comment explains the need for this: in heavy cursive, the CTC cut lands at a slightly different point within the letter each time, so a naive per-point median smears; a small local realignment sharpens the prototype.
5. **Outlier trimming**: per-variant deviation from the final median is computed (`sqrt(sum of squared per-point differences), mean over points`); only the best 80th percentile (lowest-deviation) variants are kept before the final `np.median`.
6. Returns a list of strokes (lists of `[x, y]` rounded floats), re-anchored so the leftmost x is 0.

**Statistical framing for the report.** This is a form of **robust averaging via outlier trimming combined with iterative curve registration** — conceptually analogous to a simplified Procrustes-style alignment (local translation only, not rotation/scale) followed by a trimmed mean/median, a standard technique when averaging noisy curve samples (e.g. in shape-statistics / landmark-curve averaging literature).

#### `SmoothStroke(pts, step=0.055, passes=2)`

**Purpose.** Removes the pixel-grid "staircase" that integer-pixel skeletonization bakes into a traced centerline (the comment gives a measured figure: median inter-segment turn angle 8.9°, 90th percentile 59.5° — an artefact of the raster, not something the writer actually did).

**Inner workings.** Resamples the polyline to even arc-length spacing `step=0.055` x-heights (via the same arc-length-interpolation idea as `_ResampleN`), then applies `passes=2` iterations of a **3-tap moving average** (`0.25·prev + 0.5·cur + 0.25·next`) over interior points, holding the two endpoints fixed (so the letter keeps its exact extent and join points). This is a simple discrete **low-pass filter / curve-smoothing** operation — standard in computer graphics for de-noising digitized/rasterized curves.

#### `_PenLength(g)` and `_LongestStroke(g)`

**Purpose.** `_PenLength` computes the **total arc length** of all of a glyph's strokes combined (sum, over every stroke, of `Σ ||p[i+1]-p[i]||`), i.e. "how much pen path, in x-heights, does this glyph instance actually have." `_LongestStroke` is the same computation but takes the single longest individual stroke instead of summing all of them — used to catch a glyph that is really just one short, isolated stroke fragment (a speck) even if some other spurious extra stroke inflates the total.

These are the two measurements central to the **relative fragment gate** described in the "Known limitations" section below.

#### `_ShapeGrid(g, n=12)`

**Purpose.** A cheap, comparable shape descriptor for one glyph instance: a smoothed, L2-normalized occupancy histogram over a fixed `n×n` grid spanning the glyph's own bounding box, used for all cross-variant similarity comparisons (`_PriorScore`, `BuildLetterPrior`, `_ConsensusFilter`, `_DenoisedPrototype`'s companion functions, and the glyph-grid web rendering indirectly via the same underlying stroke data).

**Inner workings.**
1. Every stroke is **densely resampled along its length** (not just at its polyline vertices — important because after Douglas–Peucker simplification a stroke might be reduced to only a handful of vertices, and gridding those alone would describe only the corners, not the overall shape): for each segment, `k = max(2, int(d*24))` points are linearly interpolated, where `d` is the segment's Chebyshev length.
2. Points are binned into an `n×n` grid (`gx, gy` via linear scaling of the glyph's bounding box to `[0, n-1]`), counting occupancy with `np.add.at`.
3. Occupancy counts are **square-rooted** (`sqrt`) to damp the effect of long dwells (a stroke that happens to sit still in one bin longer shouldn't dominate).
4. A fixed **3×3 blur kernel** is applied (`0.25` at the center tap, `0.09375` at each of the 8 neighbours — these sum to 1.0, a normalized box-ish blur), so a small positional shift between two writings of the same letter reads as "similar" rather than a hard mismatch.
5. The final grid is flattened and **L2-normalized** (`v/||v||`), so that two grids' dot product is directly a **cosine similarity**, and `1 - dot` is a cosine distance in `[0, 1]`.

**Literature connection.** This is a hand-designed shape descriptor in the spirit of classical "grid feature" / shape-context style template matching used in historical OCR and character-recognition pipelines, here repurposed for intra-class (same character, different writer-instances) similarity rather than inter-class classification.

#### `BuildLetterPrior(libsByAuthor, minSamples=12)`

**Purpose.** Learns a **cross-author prototype shape** for every character, pooling every author's raw (unfiltered) variants of that character together. With only ~10 pages per author, a single author's own examples cannot distinguish "a well-formed but personally unusual letter" from "a genuine fragment/mis-cut" — but pooling across all ten authors can, because whatever shape *all* of them roughly agree on for, say, 'a', is very likely to actually be an 'a'. This prior is used only to **reject** an individual author's clearly malformed instances; it never substitutes another author's letterform for theirs.

**Inner workings.** For each character, collect `_ShapeGrid` descriptors from every author's every raw variant; require at least `minSamples=12` valid grids; take the **elementwise median** across all of them (robust to the fraction of grids that are themselves fragments/mis-cuts); re-normalize to unit length.

#### `_PriorScore(g, ch, prior)` / `PriorFilter(variants, ch, prior, keepFrac=0.7, minKeep=3)`

**Purpose.** `_PriorScore` is the cosine distance (`1 - dot`, both unit-norm) between one glyph instance's shape grid and the character's cross-author prior — "how atypical is this instance of the letter, by consensus of ten hands." `PriorFilter` sorts an author's variants of one character by this distance and keeps only the best `keepFrac=70%` (never fewer than `minKeep=3`), dropping the stored `priorD` value on every surviving instance for later use by synthesis (`_SwapProb`, `_CoreGlyph`'s `IDEAL_ANCHOR_MAX_PRIORD` gate).

#### `_ConsensusFilter(variants, keepFrac=0.75)`

**Purpose.** A second, *within-author* filtering pass (applied after `PriorFilter`): computes pairwise **L1 distances** between every pair of this author's own shape grids, finds the **medoid** (the variant with the smallest total distance to all others — i.e. the most "typical" shape by the author's own internal consensus, not the cross-author prior), and keeps the `keepFrac=75%` of variants closest to that medoid. This is what removes a glyph instance that got contaminated by bleeding into a neighbouring letter even when the cross-author prior wasn't decisive.

---

### Author-profile aggregation

#### `RenderRefStats(gray)`

**Purpose.** Measures how the author's **real** line looks after being put through the exact same preprocessing the writer-ID classifier sees (`resize_line_image_fixed` to the fixed input size), so that synthesis can later calibrate to match — not just letter shapes, but stroke weight / ink density, which the stroke-normalized writer-ID model is sensitive to.

**Inner workings.** Loads the resized grayscale array `a`. Computes `inkFrac = mean(a < 128)` (fraction of dark pixels — i.e. ink density at the classifier's own threshold), `aspect = width/height`, `paperLevel = median of pixels ≥ 200` (light/paper tone), `inkLevel = median of pixels < 128` (dark/ink tone), `midFrac` (fraction in the anti-aliased mid-grey band `[128,200)`) — the comment explicitly notes that real scans are neither pure-white paper nor pure-black ink with hard edges, and rendering pure black-on-white is a domain shift the writer-ID model would react to, so these soft levels are reproduced at render time. It also separately re-derives `wordGap` from the *real ink gaps* (median of column-gap runs wider than 0.5·xh), explicitly **not** from the CTC space-character span, because a CTC space gets an arbitrarily narrow territory that badly underestimates real word spacing. Finally `heightXh`, `slantMeas`, `ascMeas`, `descMeas` are measured the *same way* they'll later be measured on synthesized renders (same `EstimateSlantDeg`/band logic), specifically so the two numbers are directly comparable for calibration (this feeds `StyleCalibration` in `SynthesizeHandwriting.py`).

#### `ExtractAuthorRaw(authorId, lineItems, model)`

**Purpose.** Thin orchestration wrapper: runs `ExtractLineGlyphs` + `RenderRefStats` over every line of one author, returning the raw `(glyphs, stats)` pairs. Deliberately separated from filtering/aggregation (`BuildAuthorProfile`) so the expensive skeletonization/alignment work can be cached to disk (`NOGIT/GlyphCache10/<author>.pkl`) and **re-filtered cheaply** whenever the quality-gate constants are re-tuned, without re-running CTC alignment and skeletonization from scratch.

#### `_MedianWordGap(refs, spaceAdvs)`

**Purpose.** Picks the author's word-spacing statistic: prefers the median of the real-ink-measured `wordGap` values (from `RenderRefStats`) across all lines; falls back to the median CTC space-character advance if no real measurement is available; falls back to a hardcoded `1.2` x-heights if neither exists.

#### `BuildAuthorProfile(authorId, parsed, refs=None, prior=None, verbose=True)`

**Purpose.** The main aggregation function: turns the raw per-line extraction output into the final profile dict that gets JSON-serialized. This is the longest function in the file; its job is (1) filter out untrustworthy lines and glyph instances, (2) compute per-author global statistics, (3) select and prune each character's stored variant library, (4) build the two "ideal" per-character anchors (`idealMedoid`, `idealGlyphs`).

**Inner workings, by stage.**

1. **Line-level alignment-confidence gate.** `confCut = percentile(alignConf, ALIGN_CONF_PCT=25)` if the author has ≥8 lines, else effectively disabled (`-1e9`). Lines below this threshold are skipped entirely for glyph/statistic purposes — their character boundaries are not trusted.
2. **Per-line statistics accumulation** (for kept lines): `slants`, `xhs` (x-height in pixels), `strokeWs` (normalized by xh). Per glyph: space-character widths go to `spaceAdvs`; lowercase ascender/descender extremes beyond thresholds (`top > 1.35`, `bot < -0.25`) go to `ascRatios`/`descRatios`; every non-space glyph's `advance` is recorded per character; every glyph instance is appended to `lib[ch]`.
3. **Connectedness measurement.** For every adjacent in-word glyph pair (`glyphs[k]`, `glyphs[k+1]`, both present, neither a space), `adjPairs` is incremented, and `connPairs` if `g['connR']` is true (the pen was traced crossing continuously from this glyph into the next, per `ExtractLineGlyphs`'s crossing detection). `connectedness = connPairs / adjPairs` is the final measured cursive-joining rate — a **directly-measured empirical fraction**, not a classifier output.
4. **Per-character variant filtering (`lib2`)** — this is where the fragment/width/shape gates from the module header are applied:
   - `wMax` = `MAX_GLYPH_W_WIDE` for m/w/M/W, else `MAX_GLYPH_W`.
   - `penFloor = max(MIN_PEN_ABS, FRAGMENT_PEN_FRAC * median(_PenLength(g) for g in insts))` — **the per-character-relative fragment gate**, computed from this very character's own population of instances. See the dedicated worked example below.
   - A candidate instance `g` is kept only if: its `width` is in `(0.05, wMax)`; its vertical extent `(top - bot) > 0.25`; it has `≤ 6` strokes; `_ClassOk` passes; `lead + width ≤ 1.45·advance + 0.25` (catches ink that overflowed the recognizer's assigned cell — i.e. the cut swallowed part of a neighbour); `_PenLength(g) ≥ penFloor`; `_LongestStroke(g) ≥ MIN_LONGEST_STROKE=0.55`.
   - **Width/height tightening around a low percentile anchor**: `wRef = 40th percentile of kept widths` (a *low* percentile specifically chosen because if many variants are accidentally-doubled letters, the ordinary median would be inflated and bless them too); `hMed = median height`. A `tight` subset within `[0.6·wRef, 1.5·wRef]` width and `[0.55·hMed, 1.7·hMed]` height is preferred if it has ≥3 members.
   - **Cross-author rejection** (`PriorFilter`) then **within-author consensus rejection** (`_ConsensusFilter`) are applied in that order.
   - Surviving variants are **ranked by closeness to the (wRef, hMed) anchor** and truncated to `MAX_VARIANTS_PER_CHAR=12`.
   - Each surviving variant's strokes are smoothed (`SmoothStroke`) — since at this low-data, low-anchor regime these variants *are* the synthesis output, the raw pixel staircase would otherwise be drawn as if the writer's hand shook.
5. **`idealMedoid`**: for each character, computes pairwise L1 distances between shape grids of its surviving variants and picks the single real variant closest to all the others (the medoid) — i.e. "the single most typical thing this author actually wrote," kept intact (not averaged), so a multi-stroke letter's internal structure is never blurred.
6. **`idealGlyphs`**: for each character, `_DenoisedPrototype` gives the per-stroke robust median. It is only accepted as the profile's `idealGlyphs[ch]` entry if its own `priorD` beats the raw variants' median `priorD` by a clear margin (`pd < rawPd - 0.02 and pd < 0.32`) — i.e. only used where the averaging measurably produced a *more* letter-like shape than the noisy originals, otherwise the author's raw variants are kept as-is.
7. **Final profile dict** fields: `inkFracRef, aspectRef, paperLevel, inkLevel, midFracRef, slantMeasRef, ascMeasRef, descMeasRef` (medians/nanmedians of the per-line `RenderRefStats`), `slantDeg, xHeightPx, strokeWidthXh` (medians of the per-line measurements), `ascender, descender` (medians of the outlier ascender/descender ratios, with hardcoded fallbacks `1.7`/`-0.6` if none measured), `wordSpaceXh` (via `_MedianWordGap`), `connectedness`, `letterAdvance` (per-character median advance), `nLines`, `glyphs` (the final `lib2`), `idealGlyphs`, `idealMedoid`.

---

### Driver functions

#### `BuildAll(maxLinesPerAuthor=None, useCache=True)`

**Purpose.** The main entry point (`python BuildStyleProfile.py`) for the original 10-IAM-author pipeline. Loads `IAMLineDatasetRaw`, splits lines by author (excluding holdout pages — profiles are fitted **only** on non-holdout text, so synthesis can later be evaluated on genuinely unseen text), extracts raw glyphs per author (using the disk cache in `NOGIT/GlyphCache10/` when available), pools every author's raw variants to build the shared `BuildLetterPrior`, then runs `BuildAuthorProfile` per author and writes each profile JSON.

#### `build_personal_line_items(author)`

**Purpose.** For the two personal authors (not from the IAM dataset), segments every photographed page with `SegmentPage.ProcessPage`, reads ground-truth transcripts, discards lines marked `"MESS"`, and performs the **same train/holdout split** (shuffled with a fixed `SPLIT_SEED`, `VAL_FRACTION` held out per page) used elsewhere in the project (`TrainTextPersonal.py`'s convention), returning `(train_items, holdout_texts)`.

#### `BuildAll10Authors()`

**Purpose.** The entry point for `python BuildStyleProfile.py --10authors`, building the extended 8-dataset + 2-personal author set. Loads the 8 kept dataset authors' raw glyph libraries **straight from the existing cache** (never re-extracted here — must already exist from an earlier `BuildAll()` run), and runs fresh extraction for the 2 personal authors using the **personal fine-tuned recognizer** (`PaperCRNNNumpy(checkpoint_path=PERSONAL_WEIGHTS)`) rather than the general model — the module-level comment records the measured reason: forced-alignment cut quality for personal handwriting is far better from a recognizer actually trained on that handwriting (89.64%/90%+ vs a worse score from the general model), even though that fine-tune is worse on general text. Everything downstream (`BuildLetterPrior`, `BuildAuthorProfile`, JSON writing) is identical, unmodified code shared with `BuildAll()`.

---

## 2. ProfileIO.py -- the live-server subset

### Role in the pipeline

This module exists purely to avoid dragging heavyweight dependencies (torch, scipy, pytesseract) into the live server's import closure. It contains exactly the functions the live request path needs: loading a saved profile (LoadProfile), the CTC forced-alignment routine (CtcForcedAlign, used both offline by BuildStyleProfile.py and, indirectly, by the repair-pass weak-character detector in SynthesizeHandwriting.py's _WeakChars), and a handful of classical image-processing primitives (OtsuThresholdValue, Dilate, Erode, Close, LabelComponents) copied verbatim from SegmentPage.py rather than imported from it. BinarizeLine, Skeletonize, CoreBand, EstimateSlantDeg are reused by np_inference/author_model.py for stroke normalization at inference time, by BuildStyleProfile.py for glyph extraction, and by SynthesizeHandwriting.py's StyleCalibration for measuring synthesized output the same way real pages were measured.

### Box sums / blurs

#### `_Integral(img)`

Purpose: builds a 2-D summed-area table (integral image), with an extra zero row/column prepended, enabling O(1) rectangular-region sums regardless of rectangle size.

Maths: `ii[y,x] = sum_{y'<y, x'<x} img[y',x']`, computed via two cumulative sums (`np.cumsum` along each axis in turn): `cumsum(cumsum(img, axis=0), axis=1)`.

Literature connection: the summed-area table / integral image is the classical technique underlying fast box-filtering, famously used in the Viola-Jones face detector for rapid Haar-feature evaluation; here it is repurposed for fast box-sum morphology.

#### `BoxSum(img, ry, rx)`

Purpose: returns, for every pixel, the sum of a (2ry+1)x(2rx+1) window centred on it, with the window clamped (truncated) at image borders rather than padded with zeros.

Maths: using the integral image `ii`, the sum over rectangle `[y1,y2) x [x1,x2)` is `ii[y2,x2] - ii[y1,x2] - ii[y2,x1] + ii[y1,x1]` (the standard inclusion-exclusion integral-image box-sum formula), with `y1,y2,x1,x2` clipped per-pixel to valid bounds via `np.clip`.

#### `BoxCount(ry, rx, h, w)`

Purpose: the actual pixel count of each (possibly border-clamped, hence variable-size) window, needed to normalize/compare against BoxSum results near edges (used by Erode).

### Thresholding

#### `OtsuThresholdValue(gray)`

Purpose: computes the single best binarization threshold for a grayscale image by Otsu's method: the threshold that maximizes between-class variance of the two pixel populations (background/foreground) it would produce.

Inner workings / maths: for each candidate threshold t from 0-255: `wB` = cumulative histogram count <= t (background weight), `wF = total - wB` (foreground weight), `mB` = background mean, `mF` = foreground mean. Between-class variance `var = wB * wF * (mB - mF)^2`. The threshold maximizing `var` over all 256 candidates is returned -- the textbook Otsu formulation, via brute-force scan.

Literature connection: Otsu, N. (1979), "A threshold selection method from gray-level histograms" -- the standard reference for this exact between-class-variance-maximization binarization method.

### Binary morphology (rectangular kernels) via box sums

#### `Dilate(mask, ky, kx)` / `Erode(mask, ky, kx)` / `Close(mask, ky, kx)`

Purpose: implements the three classical binary morphological operators -- dilation, erosion, and closing (dilation then erosion) -- using box sums instead of a sliding structuring-element convolution, for speed.

Maths: for a ky x kx rectangular structuring element (anchored at its centre, half-extents `ry=ky//2, rx=kx//2`):
- Dilation: a pixel becomes foreground if ANY foreground pixel exists in its window -- `BoxSum(mask, ry, rx) > 0.5`.
- Erosion: a pixel stays foreground only if EVERY pixel in its window is foreground -- `BoxSum(mask, ry, rx) >= BoxCount(ry, rx, ...) - 0.5`.
- Closing = `Erode(Dilate(mask))` -- dilation first fills small gaps/breaks, erosion then shrinks the result back down, leaving gaps filled but overall shape preserved. This is exactly what BinarizeLine uses to rejoin a hairline break in a thin, fast-written stroke without merging separate neighbouring letters.

Literature connection: standard mathematical morphology (dilation/erosion/closing) -- the generic reference is Serra, J. (1982), Image Analysis and Mathematical Morphology; most undergraduate image-processing texts (e.g. Gonzalez & Woods) cover the same operators.

### Connected components: run-based two-pass union-find

#### `_RunUF` (class)

Purpose: a minimal union-find / disjoint-set data structure (path-compressed `find`, simple `union`) used to merge row-runs of ink into connected components.

Algorithm: `find(a)` follows parent pointers with path halving (`p[a] = p[p[a]]` on the way up) until reaching a root; `union(a,b)` attaches one root under the other. This is the classical Union-Find with path compression data structure (near-constant amortized time per operation, `O(alpha(n))`).

#### `_RowRuns(mask)`

Purpose: for every row of a binary mask, finds the contiguous horizontal runs of foreground pixels as (start, end) column-index pairs, via first-difference detection (`np.diff` of a 0/1-padded row): a run starts where the padded diff is +1 and ends where it is -1.

#### `LabelComponents(mask, connectivity=8)`

Purpose: labels all connected components of a binary mask, returning an int32 label image (0 = background) and the component count. This is the run-length-encoding + union-find variant of the classical two-pass connected-component labelling algorithm.

Inner workings: every row's ink runs get a unique global run-id. For each pair of vertically adjacent rows, runs that "touch" are merged with `union`: two runs `[s1,e1)` (row above) and `[s2,e2)` (row below) touch if `s1 < e2 + pad and s2 < e1 + pad`, where `pad = 1` for 8-connectivity (diagonal touching counts) or `0` for 4-connectivity. After scanning all adjacent row pairs, `find` resolves every run to its component root, roots are compacted into sequential labels `1..nextLabel-1`, and the label image is filled in by painting each run's assigned label across its column range.

Literature connection: a textbook connected-component labelling algorithm (two-pass, union-find-based), standard in image processing; the run-based formulation is a common performance optimization.

### `LoadProfile(authorId)`

Purpose: loads one author's style-profile JSON (`NOGIT/StyleProfiles10/<authorId>.json`) from disk. Trivial I/O wrapper -- the only function most of the live server needs from the whole offline-profile-building machinery (aside from `SynthesizeHandwriting.LoadAllProfiles`, which calls this in a loop).

### CTC forced alignment (Viterbi over the blank-interleaved label sequence)

#### `CtcForcedAlign(logProbs, text)`

Purpose: given the frozen text recognizer's per-frame class probabilities for one line image and the (already known, ground-truth) transcript `text`, finds the single most likely mapping of transcript characters onto timesteps -- i.e. where in the image, in time/x, each character actually sits. This is what makes per-character glyph extraction possible even for fully cursive, unsegmented handwriting.

Inputs: `logProbs`: (T, C) numpy array, one row per timestep, one column per CTC output class (letters, digits, punctuation, plus the CTC blank symbol, from `CHAR_TO_IDX`). `text`: the known ground-truth string.

Outputs: `(list[(char, (t0,t1) or None)], conf)` -- per transcript character (restricted to characters the model's charset knows), the inclusive-exclusive timestep span the alignment assigns it (or None if unaligned), plus `conf`, the mean per-frame log-probability along the chosen path (used as the line's trustworthiness score, `ALIGN_CONF_PCT` in BuildStyleProfile.py). Returns None if the text has no alignable characters or the sequence is too short.

CTC background and the precise recurrence, worked out: Connectionist Temporal Classification (Graves et al., 2006, "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks") defines a many-to-one mapping from a length-T sequence of per-timestep class distributions to a label sequence, via a special blank symbol that lets each output label be "stretched" over multiple consecutive timesteps, with consecutive identical labels collapsing unless separated by a blank. The standard way to build an unambiguous path through this mapping (needed both for CTC's own training loss and for forced alignment) is the blank-interleaved extended label sequence: insert a blank before, after, and between every label. For a label sequence `l_1, l_2, ..., l_N` this gives `ext = [blank, l_1, blank, l_2, blank, ..., l_N, blank]`, length `S = 2N+1`. This is exactly what the code builds:

```python
labels = [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]
ext = [0]
for l in labels:
    ext += [l, 0]
```

(`0` is the blank index.)

A valid alignment path is any sequence of positions in `ext`, one per timestep `t=1..T`, that (a) starts at `ext[0]` or `ext[1]`, (b) ends at `ext[S-1]` or `ext[S-2]`, and (c) at each step either STAYS at the same ext position, ADVANCES to the next position, or (only between two DISTINCT non-blank labels, skipping the blank between them) ADVANCES BY TWO, skipping over a blank -- this "skip" transition is explicitly disallowed when the two labels either side of the blank are identical (e.g. the double-letter case "ll"), because then the blank is mandatory to keep the two letters from collapsing into one under CTC's own collapsing rule. This three-way transition structure (stay / advance-one / skip-blank-advance-two) is exactly the standard CTC decoding/alignment lattice from Graves et al. (2006).

Dynamic programming (Viterbi) recurrence, as implemented: let `dp[t, s]` = the log-probability of the best path reaching extended-sequence position s at timestep t. The code initializes `dp[0,0] = logProbs[0, ext[0]]` and (if S>1) `dp[0,1] = logProbs[0, ext[1]]` (both valid starting positions). Then for each subsequent timestep t:

```python
stay  = dp[t-1]                                   # position s at t-1 -> s at t
prev1 = shift dp[t-1] right by 1 (stay from s-1)   # position s-1 at t-1 -> s at t
prev2 = shift dp[t-1] right by 2 (skip from s-2)   # position s-2 at t-1 -> s at t, only if allowed
# prev2 disallowed (set to -inf) when ext[s] is blank, or ext[s] == ext[s-2] (repeated label)
dp[t, s] = max(stay[s], prev1[s], prev2[s]) + logProbs[t, ext[s]]
bp[t, s] = argmax index (0=stay, 1=prev1, 2=prev2)    # backpointer
```

This is precisely the Viterbi algorithm (max-sum/max-product dynamic programming over a trellis) specialized to the CTC blank-interleaved transition topology. After the forward pass, the best end state is whichever of the last two positions (S-1 or S-2) has the higher `dp[T-1, .]`; a backward traceback through `bp` recovers the full optimal path `path[t]` for every timestep. For each original transcript character `ci`, its extended-sequence index is `sIdx = 1 + 2*ci` (non-blank slots are always odd positions in `ext`); the timesteps where `path[t] == sIdx` give that character's `(t0, t1)` span (or None if the path never visits that index -- can happen with very short/garbled lines).

`conf`, the alignment confidence, is the mean, over every timestep, of `logProbs[t, ext[path[t]]]` -- the average per-frame log-probability the chosen Viterbi path actually achieves, a direct measure of how confidently the recognizer's own output supports this alignment.

Forced alignment vs. decoding -- the crucial distinction, stated explicitly: this function performs FORCED ALIGNMENT: the transcript `text` is already known and fixed; the only question the Viterbi search answers is which timesteps correspond to which already-given characters -- a constrained search over paths that must spell out exactly this one label sequence. This is categorically different from CTC DECODING (e.g. the greedy argmax-per-timestep-then-collapse-repeats-and-blanks decoding implemented in np_inference/text_model.py, used elsewhere in this project to actually read an unknown line's text): decoding has no known target and must search/guess over all possible label sequences the network might be outputting. Both are grounded in the same underlying CTC framework from Graves et al. (2006), but solve different problems: decoding answers "what does this say?"; forced alignment answers "given what this says, where exactly is each letter?" -- the latter is what makes glyph-boundary extraction from cursive, unsegmented handwriting possible at all.

### Skeletonization (Zhang-Suen thinning, vectorized numpy)

#### `Skeletonize(mask)`

Purpose: reduces a binary ink mask to a 1-pixel-wide centerline skeleton -- the geometric backbone later traced into polylines by TracePolylines.

Algorithm -- Zhang-Suen thinning (Zhang, T.Y. & Suen, C.Y., 1984, "A fast parallel algorithm for thinning digital patterns"): a classical iterative parallel thinning algorithm. On each full iteration, two sub-iterations alternate (`step=0,1`), each removing a candidate set of foreground pixels SIMULTANEOUSLY (not one-by-one), subject to four conditions evaluated using each pixel's 8-neighbourhood P2..P9 (numbered clockwise from the pixel directly above, following the original paper's convention):
- `B` = the number of foreground neighbours (`2 <= B <= 6`) -- too few means already an endpoint (don't erode), too many means deep inside a thick blob (not yet boundary).
- `A` = the number of 0->1 transitions walking around the 8-neighbour ring in order (`A == 1`) -- the standard connectivity-preserving condition: removing a pixel with exactly one such transition cannot disconnect the shape or create a hole.
- Two more conditions distinguish the two sub-iterations, targeting the south/east boundary in one pass and the north/west boundary in the other (alternating is what makes parallel removal safe and keeps the skeleton centered):
  - Sub-iteration 0: `P2*P4*P6 = 0` and `P4*P6*P8 = 0`.
  - Sub-iteration 1: `P2*P4*P8 = 0` and `P2*P6*P8 = 0`.

A pixel is removed in a given sub-iteration only if it is currently foreground AND all of `2<=B<=6`, `A==1`, and that sub-iteration's two product conditions hold. The process repeats until no pixel qualifies for removal, converging to a stable 1-pixel-thin skeleton. The implementation vectorizes B, A and the condition products over the whole image per sub-iteration (via `np.pad` + 8 shifted views) for speed, but the algorithm itself is unmodified classical Zhang-Suen.

Literature connection: Zhang, T.Y. & Suen, C.Y. (1984), "A fast parallel algorithm for thinning digital patterns," Communications of the ACM 27(3). One of the most widely cited skeletonization algorithms in document-image analysis and OCR preprocessing.

#### `_HRun(mask)`

Purpose: for every pixel, returns the length of the horizontal run of foreground pixels it belongs to (0 if background) -- used by BinarizeLine to detect ruled-paper lines and underlines (long, thin horizontal runs) for removal.

Inner workings: each row is scanned for run starts (`a[:,1:] == 1 & a[:,:-1] == 0`, plus the first column itself if foreground); a running cumsum over these start-flags gives each foreground pixel a unique run-id; `np.bincount` counts each run-id's total length; the result is indexed back per-pixel via the run-id array. (`_HRun` on the transposed mask gives the equivalent vertical-run version, used as `vRun = _HRun(ink.T).T`.)

### `BinarizeLine(gray, stripRules=True, closeGaps=True)`

Purpose: converts a grayscale line-crop to a clean boolean ink mask, removing printed ruled-paper lines/underlines and rejoining hairline breaks in thin, fast-written strokes, then dropping remaining tiny speckle.

Inner workings:
1. Otsu-threshold (`OtsuThresholdValue`) to get an initial ink mask (`gray < thr`, since ink is dark).
2. Rule-line stripping: computes horizontal run length `hRun` and vertical run length `vRun` per pixel; flags a pixel as "rule" if its horizontal run is very long (`>= max(24, 1.1*h)`) AND its vertical run is thin (`<= max(3, 0.12*h)`, only a pen-width thick vertically). Such pixels are removed from the ink mask -- but only where no genuine vertical stroke crosses through that same column (checked again at the point of removal), so a letter's stem crossing the rule line is preserved. Leaving rule lines in would glue every letter together (falsely measuring the hand as fully cursive), drag the slant estimate toward horizontal, and paste a spurious bar across every extracted glyph.
3. Gap closing: a small morphological Close with kernel size `k = clip(round(0.045*h), 2, 4)` -- proportional to the line's own height (hence stroke pitch), not a fixed size -- rejoins a hairline scan-break in a thin stroke without merging separate neighbouring letters.
4. Speckle removal: LabelComponents labels all remaining components; any component with fewer than 6 pixels is dropped (scanner/JPEG noise, not deliberate ink).

### `CoreBand(ink)`

Purpose: finds the x-height band (baseline and topline) of a line of ink from its horizontal ink-density profile.

Inner workings: `prof[row] = ink[row,:].sum()`. Rows with density at least 45% of the row with maximum density are taken as the "core" x-height band; `top`/`base` are the first/last such row. Returns None if the band is less than 3px tall. This exploits the fact that the bulk of ordinary lowercase letter-body ink sits densely packed between baseline and x-height line, while ascenders/descenders contribute comparatively sparse ink outside that band.

### `EstimateSlantDeg(ink)`

Purpose: measures the author's handwriting slant angle by a shear-search: the correct de-slanting shear is the one that makes the author's (mostly near-vertical) strokes actually vertical, detected as the shear angle that maximizes the sharpness (peakiness) of the vertical-stroke column histogram.

Inner workings / maths: for each candidate angle `deg` in [-45 deg, 45 deg] (step 1.5 deg): compute shear factor `sh = tan(deg)`; shear every ink pixel's x-coordinate by `x + (y - yc) * (-sh)` (shearing around the vertical centroid `yc` so the shear doesn't also translate the whole mask); bin the sheared x-coordinates into a histogram (`np.bincount`); score this candidate as `sum(histogram_count^2)` -- maximized exactly when ink is concentrated into the fewest, tallest columns (vertical strokes stacked after the correct de-shear), and small when ink is smeared across many columns (strokes still slanted, pixels spread across many x-values at different heights). The angle with the highest score is kept; the function returns its negation (`-best`) per a sign convention the code comments say was "verified against a synthetically sheared test image," chosen so that `x -= tan(slant) * y` (y measured up from the baseline) is the correct de-slanting transform used throughout the rest of the codebase.

Literature connection: this shear-search / column-histogram-sharpness approach is a standard, simple handwriting-slant-estimation technique in document-image analysis (closely related to "shear-based" slant-correction methods used in historical offline-handwriting-recognition preprocessing pipelines predating deep learning).

---

## 3. SynthesizeHandwriting.py -- trajectory synthesis and G-code generation

### Role in the pipeline

This is the production module that turns `(text, authorId)` into a drawable pen trajectory, and then that trajectory into G-code, a step/direction schedule, and preview rasters. It consumes profiles built by BuildStyleProfile.py and loaded via ProfileIO.LoadProfile / LoadAllProfiles. It is called directly by server.py's `/api/generate` endpoint (via `SynthesizeJointBestOf`) and by every script in Evaluate.py. The file is organized in two halves: trajectory synthesis (SynthesizeText and its best-of-N wrappers SynthesizeLegible / SynthesizeJointBestOf), and G-code / machine-coordinate generation (merged in from a former separate WriteGCode.py module).

### Fallback alphabets: `_FB` and `_CURSIVE`

Purpose: two hand-authored, single-stroke-per-letter skeleton alphabets, in the same normalized frame the profiles use (x right, y up from baseline, 1.0 = one x-height). `_FB` is an upright print font; `_CURSIVE` is a joined-script font where every letter enters at `(_CX_IN=0.0, 0.30)` and exits at `(_CX_OUT, 0.30)` -- a fixed connection height so consecutive letters join legibly without extra logic. Both exist purely as safety nets: for a character the author never wrote at all (`_FB` lookup in `_GlyphSource`'s final fallback), or as the "legible anchor" letterform that replaces a malformed extracted glyph (`_BaselineGlyph` / `_CoreGlyph`). They are not any author's real handwriting -- they are reshaped per-author (ascender/descender scale, slant, spacing) so that even a generic substituted letter still looks like it belongs to that author's proportions.

### Uniform-pen constants

`UNIFORM_INK = True`, `UNIFORM_PEN_WIDTH_XH = 0.085`, `UNIFORM_SOFT_FRAC = 0.30`: because the physical gantry writes every author with one pen at one constant width, ink weight cannot be a style channel the robot reproduces -- so `RenderTrajectory` (when `uniformInk` is true, the default) draws every author at the same stroke width (expressed in x-heights, so resolution-independent) matched to the width the shape-only writer-ID classifier was itself trained on, ignoring the author's own measured `inkFracRef`/`strokeWidthXh`.

### Legibility-guard constants

`JOIN_PROB_CAP = 1.0` (no-op cap on cursive-join probability), `CORE_SLANT_CAP_DEG = 22.0` (the print anchor glyph is never sheared past this -- a 40-degree shear would make even a clean letterform collide and ramp off its cell), `CORE_JITTER_FRAC = 0.15` (anchor glyphs wobble far less than extracted glyphs -- re-introducing heavy jitter on a shape that exists specifically to remove ambiguity would defeat its purpose), `COLLISION_MIN_GAP_XH = 0.05` (forces this much clear x-height gap between a glyph's ink and the previous glyph's body).

### `_BaselineGlyph(ch, profile, printOnly=False)`

Purpose: produces a generic, legible letterform for character `ch`, re-proportioned into the author's own ascender/descender reach (slant and jitter are applied later by the caller).

Inner workings: picks the source alphabet -- `_CURSIVE` if the author's measured `connectedness >= 0.5` and `printOnly` is not forced, else `_FB`. Looks up `ch` (or its lowercase). For every point, y-values above 1.0 (ascender territory) are rescaled by `max(0.35, (ascender-1.0)/0.75)` and y-values below 0.0 (descender territory) by `max(0.35, abs(descender)/0.55)` -- i.e. the generic skeleton's own ascender/descender proportions (calibrated for a "typical" hand) are stretched or compressed to match this specific author's measured reach, while the x-height band (0 to 1) is left untouched. Returns `(strokes, advance)` where `advance = (max_x - min_x) + 0.12` (a small trailing gap added).

The code's design-rationale comment is important for the report: a data-driven "legible core" built from the cross-author medoid of real extracted variants was tested and measured only ~76% char / ~39% word accuracy through the frozen recognizer (no better than raw author hands, because the consensus of a messy cursive letter is still a messy cursive letter), whereas the hand-drawn single-stroke `_FB` print font reads at ~96% char / ~84% word -- which is why `_FB` (reshaped per author) is used as the legibility anchor rather than any data-driven average.

### `Trajectory` (class)

Purpose: the container for a synthesized pen path, in millimetres, y-up.

- `__init__(strokes, meta)`: stores the list of pen-down polylines and a metadata dict (author, mmPerXh, stroke count, glyph-source usage counts, etc).
- `Bounds()`: returns `(minX, minY, maxX, maxY)` over every point in every stroke.
- `PenTravelMm()`: returns `(downMm, upMm)` -- total pen-down drawing distance (sum of consecutive-point Euclidean distances within each stroke) and total pen-up travel distance (Euclidean distance between the end of one stroke and the start of the next). Used for machine-time/wear estimation.

### Glyph selection

#### `_CoreGlyph(ch, profile)`

Purpose: returns the "legible anchor" letterform for `ch`, reshaped to this specific author -- either the author's OWN cleaned-up ideal letterform (preferred, when trustworthy) or the generic cursive/print skeleton.

Inner workings: `cursive = connectedness >= CORE_CURSIVE_CONN(0.5)` decides joined vs. upright skeleton if falling back to generic. First choice: look up `ch` in the author's own `idealMedoid` or `idealGlyphs` (selected by the module constant `IDEAL_SOURCE`, default `'medoid'` -- the single most typical REAL variant the author wrote, kept intact, vs. `'median'`, the smoother but potentially structure-blurring per-stroke average). This own-ideal is used only if its stored `priorD` (cosine distance from the cross-author consensus shape, computed back in BuildStyleProfile.py) is at or below `IDEAL_ANCHOR_MAX_PRIORD = 0.30` -- i.e. only if it is itself a recognisable version of the letter (averaging a dozen malformed cuts can give a tidy shape that is still the wrong letter). If the own-ideal is unusable, falls back to `_BaselineGlyph`. Returns `(strokes, advance, lead, isCursive)`.

#### `_SwapProb(lam, priorD)`

Purpose: computes `P(replace this glyph instance with the style-aware anchor letterform)`, as a function of the global legibility blend `lam` in [0,1] and the specific instance's `priorD` (cosine distance from the cross-author consensus).

Maths/logic: if `priorD >= _GLYPH_GARBAGE(0.55)`, always swap (`1.0`) regardless of `lam` -- this is unsalvageable. If `lam <= 1e-3`, never swap (`0.0`) -- legibility blending is fully off. If `priorD <= _GLYPH_CLEAN(0.12)`, swap only rarely (`0.06 * lam`) -- a genuinely clean extraction is kept almost always. Otherwise linearly ramps the swap probability between those two regimes: `bad = clip((priorD - 0.12) / (0.55 - 0.12), 0, 1)`; `P = clip(lam * (0.55 + 0.9*bad), 0, 1)`. The design rationale (from the surrounding comments): measurement showed an author's own letterform is only reliably readable when it's a genuinely clean extraction, so anything else should swap to the anchor readily -- but the anchor still carries the author's own slant, size, spacing and joins, so the hand is not erased even when individual letters are replaced.

#### `_GlyphSource(profile, ch, rng, prevExitY=None, lam=0.0, forceCore=False, forcePrint=False)`

Purpose: the glyph-instance selector called once per character during synthesis -- decides which concrete stroke shape to draw for this occurrence of `ch`, and returns `(strokes, advance, lead, source)` where `source` is one of `'own' | 'core' | 'case' | 'similar' | 'fallback' | 'blank'` (recorded for the `glyphSources` usage statistics in the trajectory's metadata).

Inner workings, preference order:
1. `forcePrint` (set by the repair pass when a letter is still unreadable even in the cursive anchor) forces the upright `_BaselineGlyph(printOnly=True)`.
2. If the author's own glyph library has variants for `ch`: rank them by stored `priorD` (most well-formed first), then `pick(ranked)` (see below) selects one probabilistically. If `forceCore` is set or a `_SwapProb` roll fires, swap to `_CoreGlyph` instead; otherwise use the picked variant, packed via `pack(g)`.
3. If not, try the swapped case (`ch.swapcase()`) or a visually-similar character via `_SIMILAR = {'I':'l', 'O':'0', 'l':'I', '0':'O', 'o':'0', ';':':', '"':"'", '!':'l'}`. If borrowing a lowercase shape for an uppercase letter, the shape is scaled up by `k = clip(ascender / max(0.6, g['top']), 1.0, 2.0)` to reach capital height.
4. Otherwise falls back to `_CoreGlyph`, then to the static `_FB` alphabet, then to a truly blank glyph (zero strokes, just an advance) as the last resort.

`pack(g, k=1.0)`: converts a stored glyph dict into `(strokes, advance, lead)`, scaled by `k`. Crucially, `advance = max(g['advance'], g['lead'] + g['width'] + 0.04)` -- i.e. it uses the chosen VARIANT's own measured advance, never the character's population median, because a wide variant drawn at a narrow median advance would collide with the next letter (and a narrow variant at a wide median would leave an unnatural hole).

`pick(cands)`: implements weighted random sampling among only the `_ELIGIBLE_VARIANTS = 5` most typical candidates (by rank). Each candidate `i` gets base weight `w_i = 1/(1 + 1.4*i)` (geometrically favouring more typical variants while still allowing variety). If a previous-glyph exit height (`prevExitY`) is given (mid-ligature), the weight is further multiplied by a Gaussian-shaped match score `exp(-((entryY - prevExitY)/0.45)^2)` -- favouring the variant whose own entry height matches where the pen physically is, since in real cursive a letter's exact shape depends on what it is joined to; mixing a high-entry variant onto a low exit is what produces tangled joins. A roulette-wheel selection (`r = uniform(0,total); accumulate` ) then picks one candidate. The code's own comment records a measured finding motivating the eligible-variant cap: sampling uniformly over 12 variants give only 75.5% writer-ID vs 80.5% over the top 2 -- a writer forms a given letter consistently, so averaging the "signature" away across a dozen variants costs style-match accuracy, while keeping more than one (5, here) still beats a single frozen template (which would read as visibly stamped/repeated).

### Synthesis core

#### `_Jitter(strokes, rng, amp, rotAmp, scaleAmp)`

Purpose: applies small per-glyph-instance randomized position/rotation/scale noise, so repeated letters/words don't look identically stamped.

Maths: draws `dx ~ N(0, amp)`, `dy ~ N(0, 0.6*amp)` (less vertical than horizontal noise), rotation `theta ~ N(0, rotAmp)` degrees, and independent x/y scale factors `sx, sy ~ N(1, scaleAmp)`. Applies, per point, scale then a standard 2D rotation matrix `(x2*cos th - y2*sin th, x2*sin th + y2*cos th)` then translation by `(dx, dy)`.

#### `_Resample(poly, step)`

Purpose: even arc-length resampling of a polyline at fixed spacing `step` (mm, in the physical output frame) -- gives the plotter smooth, evenly-timed motion and keeps the rendered stroke width visually uniform (a `PIL.ImageDraw.line` with widely varying point spacing can look lumpy). Same arc-length-interpolation technique as `_ResampleN` in BuildStyleProfile.py.

#### `SynthesizeText(text, profile, mmPerXh=4.0, seed=None, lineWidthMm=180.0, jitter=0.15, legibility=None, perCharLam=None, _cal=None)`

Purpose: THE single-draw synthesis function -- text + profile -> one Trajectory. Every other synthesis entry point (`SynthesizeLegible`, `SynthesizeJointBestOf`, every Evaluate.py script) calls this repeatedly with different seeds/parameters and picks among the results.

Inputs: `text` (may contain `\n` for explicit line breaks), `profile` (one author's JSON dict), `mmPerXh` (physical size of one x-height in millimetres), `seed` (RNG seed for reproducible jitter/variant choices), `lineWidthMm` (word-wrap width), `jitter` (global jitter scale multiplier), `legibility` (the lambda blend toward the print/cursive anchor; `None` reads `profile['legibilityLambda']`, defaulting to 0.0 if absent), `perCharLam` (optional dict mapping a character's position-in-text index to a forced lambda, used by the repair passes), `_cal` (an internal override for the style-calibration dict, used by `StyleCalibration`'s own probe renders to avoid infinite recursion).

Outputs: a `Trajectory`.

Inner workings, in the order the code executes:

1. Resolves `legibility`, obtains the style-calibration dict `cal` (either passed in via `_cal`, or computed fresh by `StyleCalibration` if `USE_STYLE_CALIBRATION` is on).
2. Computes an effective slant cap that SOFTENS with legibility: `authorSlantCap = 45.0 - 19.0*clip(legibility,0,1)` degrees -- at `legibility=0` the author's full measured slant (up to 45 deg) is kept, at `legibility=1` it is capped down to 26 deg, since ascenders ramping at full width off their own cell is what makes a steeply-slanted hand unreadable. `slant = tan(clipped slantDeg) + cal['shearDelta']`.
3. `connEff = connectedness * (1 - JOIN_LEGIBILITY_COUPLING * clip(legibility,0,1))`. Since `JOIN_LEGIBILITY_COUPLING = 0.0` is hardcoded, this is currently always `connEff = connectedness` -- the code's own comment explains why coupling was explicitly disabled: a writer's join rate is a MEASURED property of their hand, not a quality knob, and coupling it to legibility previously converted cursive hands silently into print ones as legibility rose (measured: 94% joins down to 57% at legibility 0.65), which is exactly the feature that makes those hands recognisable.
4. `lineStep = (ascender - descender + 0.85) * mmPerXh` -- vertical spacing between wrapped/explicit lines, proportional to the author's own measured vertical reach plus a small constant margin.
5. Iterates over `_SplitWords(text)` (words plus explicit `'\n'` markers). On a newline, advances `lineIdx`, resets `penX=0`, sets `penY = -lineIdx*lineStep`. On a normal word: if adding its width (`_WordWidth`) would exceed `lineWidthMm`, wraps to a new line first.
6. For each character in the word:
   - Resolves `lam` for this character (from `perCharLam` if present, else the global `legibility`; `forceCore`/`forcePrint` flags are derived from sentinel `perCharLam` values `>= 0.999` / `>= 1.9` respectively, used by the repair passes).
   - If the hand is sufficiently connected (`connEff >= 0.2`) and this is not the word's first letter and both letters are alphabetic, computes `wantEntryY` -- the height (in x-heights, relative to the current pen baseline and drift) where the previous glyph's pen actually exited, to bias `_GlyphSource`'s variant choice toward a shape-matching entry.
   - Calls `_GlyphSource` to get the glyph's strokes/advance/lead/source.
   - If the glyph is the style-aware core anchor (`src == 'core'`), style calibration (`_ApplyCal`) is SKIPPED (the anchor is a designed-legible shape and re-stretching its carefully chosen proportions would reintroduce the exact letter-shape ambiguity it exists to remove) and jitter is applied at only `CORE_JITTER_FRAC(0.15)` of normal strength; its slant is separately capped (`30 deg` if the author is a connected/cursive hand using the joined anchor, else `CORE_SLANT_CAP_DEG=22 deg`), and its baseline drift is damped (`0.5x` if cursive-anchor, `0.25x` otherwise).
   - `place(px)` positions every stroke point at `(px + (lead + x + glyphSlant*y)*xh, penY + (y + glyphDrift)*xh)` -- i.e. the de-slanted, per-glyph-normalized coordinates are re-sheared by `glyphSlant` and scaled to millimetres via `xh = mmPerXh`.
   - A collision guard: if the newly placed glyph's leftmost ink x would land before `prevBodyMaxX + COLLISION_MIN_GAP_XH*xh` (the required clear gap after the previous glyph's widest stroke), `penX` is pushed right by exactly the shortfall and the glyph is re-placed.
   - The "body" stroke (the one with the greatest own x-span -- i.e. the main letterform stroke, as opposed to a dot, crossbar, or accent mark drawn as a separate pen-down) is identified via `argmax` over per-stroke x-extent; only the body stroke's exit point becomes the candidate join point for the NEXT letter (so a crossbar's endpoint is never mistaken for where the pen should continue to).
   - `joinable` (whether to draw a cursive ligature into this letter) requires: not the word's first letter; both letters alphabetic; and EITHER both glyphs are the author's own extracted forms (`bothOwn`) OR both are the joined cursive anchor (`bothCursiveCore`) -- a print-anchor glyph is NEVER joined through, since joining through an upright letter would look wrong. The actual decision is then a Bernoulli draw with probability `min(connEff, JOIN_PROB_CAP)` -- i.e. `connectedness` is used as a measured PROBABILITY of joining each adjacent pair, not a hard on/off threshold. The code's own comment records why this matters: using it as a threshold joined EVERY pair for any author above it, fusing whole words into oversized blobs with wrong letter pitch/component-width statistics -- exactly the geometry a nearest-author shape matcher would key on and get wrong.
   - If joining, a ligature stroke (`_Ligature`) is appended to the running pen-down `chain` followed immediately by the body stroke (same continuous pen-down polyline, since the pen never lifts); if not, the current chain is flushed (emitted as a completed stroke) and a new chain starts with this glyph's body.
   - Any non-body strokes (dots, crossbars, accents) are flushed as their own separate pen-down strokes immediately.
   - `penX` advances by `adv * xh`.
7. After each word, `penX` advances by the word-gap (`wordSpaceXh`, times mm, times a small `0.85 + 0.3*rng.random()` randomization so word spacing isn't perfectly uniform).
8. Returns a `Trajectory` with metadata including `glyphSources` (counts of each selection path used, useful for understanding how much of the output is "really" the author's hand vs. anchor/fallback).

#### `StyleCalibration(profile, mmPerXh=4.0, pxPerMm=18.0)`

Purpose: computes a small correction (`shearDelta`, `ascK`, `descK`) applied to every non-anchor glyph at synthesis time, so that a RENDERED line's measured slant and ascender/descender reach match the author's own REAL, measured values -- because glyphs are stored and composed per-instance, a line assembled from many separately-normalized glyphs does not automatically reproduce the author's own LINE-LEVEL slant and vertical reach exactly.

Inner workings: iterates 3 times: synthesizes a long fixed probe sentence with the CURRENT calibration applied (`_cal=cal`, zero jitter, seed fixed at 17), renders it, re-measures it with the EXACT same tools used to measure real pages (`ProfileIO.BinarizeLine`, `CoreBand`, `EstimateSlantDeg`), and updates: `shearDelta += tan(slantRef) - tan(gotSlant)` (an additive correction in shear-space, since shear composes additively, not angle-space); `ascK *= (ascRef - 1.0)/(gotAsc - 1.0)`, `descK *= descRef/gotDesc` (both clipped to `[0.5, 1.6]` to prevent runaway correction from a bad single probe). Cached per `(authorId, mmPerXh, pxPerMm)` in the module-level `_STYLE_CAL` dict.

The code's own comment gives the measured justification: correcting these two statistics made the ink-SENSITIVE writer-ID model slightly WORSE (86.0% -> 84.5%) but the SHAPE-ONLY model (which ignores pen weight, i.e. exactly what judges the physical, single-pen robot) measurably BETTER (59.2% -> 63.3% direct, 58.3% -> 66.7% through G-code) -- and since the shape-only judge is the one that predicts the physical robot's actual output, `USE_STYLE_CALIBRATION = True` is kept on. This is a good example for the report's "Design and Implementation" section of choosing the evaluation metric that actually matches deployment reality over the one that happens to score higher on paper.

#### `_ApplyCal(strokes, cal)`

Purpose: applies the `ascK`/`descK` scale factors from `StyleCalibration` to stroke points, but ONLY to the portion of each point's y-coordinate that lies OUTSIDE the x-height band (`y > 1.0` or `y < 0.0`) -- so ascender/descender reach is corrected without disturbing x-height itself, which is the dimension everything else (letter width, spacing) is normalized against.

#### `_Ligature(p0, p1, xh, rng)`

Purpose: synthesizes the connecting stroke drawn between two joined cursive letters, from the previous glyph's exit point `p0` to the next glyph's entry point `p1`.

Inner workings: if the horizontal gap is tiny (`< 0.06*xh`), just draws a straight 2-point line. Otherwise computes a sag amount `sag = min(0.11*xh, 0.16*|dx|) + N(0, 0.015*xh)` -- the sag is explicitly capped to be PROPORIONAL to how far the ligature travels, because the comment notes a fixed-magnitude dip would turn a SHORT join into a visible loop, which is what made earlier synthesized cursive look tangled. The ligature is a 3-point polyline (`p0`, a midpoint dipped upward by `sag`, `p1`) arc-length-resampled (`_Resample`, step 0.35mm) into a smooth curve.

#### `_SplitWords(text)` / `_WordWidth(word, profile)`

Purpose: `_SplitWords` tokenizes text into words plus explicit `'\n'` line-break markers (splitting on `'\n'` first, then on spaces within each line, appending a `'\n'` sentinel after each line except the very last). `_WordWidth` sums each character's median `letterAdvance` (falling back to `0.9` for an unseen character) to estimate a word's total width in x-heights, used for word-wrap decisions before any actual glyph is drawn.

### Rendering

#### `_Draw(traj, pxPerMm, padMm, strokePx, paper=255, ink=0, softPx=0.0)`

Purpose: rasterizes a Trajectory to a PIL grayscale image, matching the look of the real training crops (dark ink on light paper).

Inner workings: computes the image size from the trajectory's bounds plus padding; draws every pen-down stroke as a `PIL.ImageDraw.line` of width `strokePx` with rounded joints (`joint='curve'`) and explicit end-caps drawn as filled circles of radius `strokePx/2` (since `ImageDraw.line` alone leaves square-cut stroke ends, but a real pen leaves rounded ends); optionally applies a Gaussian blur (`ImageFilter.GaussianBlur`) to soften stroke edges, matching the anti-aliased look of a real scanned/photographed stroke.

#### `RenderTrajectory(traj, pxPerMm=8.0, padMm=None, strokePx=None, profile=None, size=None, matchInk=True, uniformInk=None)`

Purpose: the main render entry point used everywhere (server.py preview, Evaluate.py judging, SynthesizeLegible/JointBestOf's internal scoring).

Inner workings: if `uniformInk` (default from the module constant `UNIFORM_INK=True`) is set, every author is drawn at the same fixed width `UNIFORM_PEN_WIDTH_XH * mmPerXh * pxPerMm` pixels with a proportional soft blur -- this is the physically honest rendering path, matching both what the single-pen gantry can produce and what the shape-only writer-ID classifier was trained on. If `uniformInk=False` (legacy path), the author's own measured `strokeWidthXh` sets the width, and if the profile carries a measured `inkFracRef`, `_MatchInkDensity` is invoked to search for the exact stroke width that reproduces that measured ink density on THIS specific line.

#### `_MatchInkDensity(traj, pxPerMm, padMm, startPx, target, mm, paper=255, ink=0, soft=0.0)`

Purpose: finds the integer stroke width (in pixels) whose rendered, classifier-preprocessed ink density (`fraction of pixels < 128` after `resize_line_image_fixed`) best matches a target density, via local hill-climbing search rather than a closed-form computation.

Inner workings: `probe(px)` renders at width `px`, resizes exactly as the classifier would, and measures ink fraction, caching results. Starting from `startPx`, the search steps by +1 or -1 (whichever direction the initial error indicates) as long as each step strictly improves the error, stopping at the first non-improving step (a simple greedy 1-D local search, not exhaustive, over the bounded range `[1, hi]` where `hi = 0.55*mm*pxPerMm`). The comment explains why this has to be an empirical search rather than a formula: after the classifier's own downsampling, a thin stroke can anti-alias below the `<128` threshold entirely, making density a STEP-LIKE (non-smooth, non-invertible) function of width; and density also depends on how much text fills the line (a short line stretched to the fixed input width renders fatter letters), so a per-line search is required rather than a fixed global width.

#### `LoadAllProfiles()`

Purpose: loads every `*.json` file under `PROFILE_DIR` into a dict `{authorId: profileDict}`. Trivial but the single point every consumer (server.py, Evaluate.py, web_render_helpers.py) goes through to get the full author roster.

### Legibility-first synthesis (best-of-N + repair)

#### `_LineLogProbs(reader, img)`

Purpose: thin wrapper returning the `(T, C)` log-probability array for a rendered line image, exactly as `ExtractLineGlyphs` would obtain it, used by `_WeakChars`.

#### `_WeakChars(reader, img, visible)`

Purpose: finds which character POSITIONS in the already-rendered image the frozen recognizer is LEAST confident about, worst-first -- the diagnostic step that drives both repair passes.

Inner workings: runs `CtcForcedAlign` (from ProfileIO.py, the same forced-alignment machinery used offline) against the already-known `visible` text to get each character's assigned timestep span, then scores each character by the MEAN class log-probability the recognizer itself assigned to the correct class over its own assigned frames (`lp[sp[0]:sp[1], CHAR_TO_IDX[c]].mean()`); unaligned characters get a sentinel score of `-1e9` (worst possible, prioritised for repair first). Returns character positions sorted worst-to-best. This reuses CTC forced alignment for a second purpose beyond glyph extraction: diagnosing exactly which rendered letters are unclear, by the recognizer's own stated confidence.

#### `SynthesizeLegible(text, profile, nTries=6, mmPerXh=4.0, lineWidthMm=180.0, jitter=0.5, seed=0, reader=None, pxPerMm=18.0, legibility=None, repair=True)`

Purpose: best-of-N draws scored by TEXT ACCURACY ALONE, followed by a targeted repair pass -- an earlier, simpler alternative to `SynthesizeJointBestOf` (still used by some Evaluate.py legibility-only measurements).

Inner workings:
1. Draws `nTries` independent candidates (different seeds, same style parameters), scores each by `CharAcc(ReadText(render), visible)` (1 - normalized edit distance, see `Evaluate.py`'s `CharAcc`), keeps the best; early-exits if a candidate already scores `>= 0.99`.
2. Repair loop (up to `REPAIR_ROUNDS=3` rounds): finds the `REPAIR_WORST_K=3` worst-scoring NEW character positions via `_WeakChars` and pins them to `perChar[pos] = 1.0` (style-aware core anchor); separately, the single worst position that was ALREADY pinned once and is still weak escalates to `perChar[pos] = 2.0` (forced upright print, the harder fallback). Re-synthesizes with these pins and keeps the new candidate only if its score improved.
3. Records `legibilityScore`, `legibilityTries`, `legibilityLambda`, `repairRounds`, `repairedChars` in the returned trajectory's metadata.

#### `SynthesizeJointBestOf(author, text, profile, nTries=6, mmPerXh=4.0, lineWidthMm=180.0, jitter=0.5, seed=0, reader=None, authorModel=None, authorMapping=None, pxPerMm=18.0, legibility=None, repair=True, repairRounds=3, repairWorstK=2)`

Purpose: THE PRODUCTION synthesis method, used by server.py's `/api/generate` and by Evaluate.py's `rewrite` mode. Identical best-of-N + repair STRUCTURE to `SynthesizeLegible`, but scored by the HARMONIC MEAN of text-recognition accuracy and writer-ID confidence TOGETHER, rather than text accuracy alone.

**The harmonic mean, worked out.** For text accuracy `a` and writer-ID confidence `b` (both in `[0,1]`):

```
H = 2ab / (a + b)      if a + b > 0, else H = 0
```

implemented directly as:
```python
combined = (2 * textAcc * widConf / (textAcc + widConf)
            if (textAcc + widConf) > 1e-9 else 0.0)
```

**Why the harmonic mean rather than the arithmetic mean, precisely.** The arithmetic mean `(a+b)/2` treats a candidate that is excellent on one axis and poor on the other as EQUIVALENT to a candidate that is middling on both -- e.g. `a=1.0, b=0.2` and `a=0.6, b=0.6` both average to `0.6`. The harmonic mean does not: `H(1.0, 0.2) = 2*1.0*0.2/1.2 = 0.333`, while `H(0.6, 0.6) = 0.6` -- the harmonic mean is PULLED STRONGLY TOWARD THE SMALLER of the two values (it is always `<=` the arithmetic mean, with equality only when `a=b`), so it specifically PENALISES a candidate that is good on only one axis. This matches the actual design intent stated in the code's own comment: a plain best-of-N scored by legibility alone measurably raises text accuracy but can COST writer-ID for some authors, because the single most-legible draw among N random jitter/variant candidates has no particular reason to also be the most style-distinctive one -- for some authors the two axes pull in different directions across the N candidates, and a text-only (or any score that doesn't punish a lopsided result) score always takes the legibility side of that coin-flip. Harmonic-mean scoring instead picks a candidate that is a genuine COMPROMISE on both axes. The measured result (recorded in the comment, matching the full 10-author check reported in Evaluate.py's `rewrite` run): raised text accuracy for every author with NO writer-ID regression, and for several authors already measuring only 33-67% writer-ID confidence, raised it up to 100%.

Inner workings:
1. Best-of-N draws, each scored by `jointScore(img)` = `(combined, textAcc, widConf)` via the harmonic-mean formula above, keeping the highest `combined`; early-exits at `combined >= 0.99`.
2. A JOINT-AWARE repair pass: finds weak characters via `_WeakChars` exactly as `SynthesizeLegible` does, but critically only ACCEPTS a repair candidate if its NEW joint (harmonic-mean) score beats the current best -- a repair that fixes a letter's legibility but drags the rendering toward looking like a different author (lowering `widConf` enough to outweigh the legibility gain in the harmonic mean) is silently REJECTED rather than taken. The code's own comment contrasts this explicitly with `SynthesizeLegible`'s plain repair pass, which accepts any text-score improvement regardless of writer-ID cost, and was measured to cost writer-ID for authors whose profile doesn't already lean on the anchor.
3. Records `jointScore`, `jointTries`, `jointRepairedChars`, `legibilityLambda` in the trajectory's metadata.

### Machine configuration and G-code emission

#### `GantryConfig` (dataclass)

Purpose: single source of truth for every machine-specific constant. Key fields: `fullStepsPerRev=200` (NEMA 17 stepper), `microstepping=16` (A4988 driver 1/16-microstep setting), `mmPerRevX/Y=40.0` (GT2 2mm-pitch belt on a 20-tooth pulley -> 40mm travel per revolution), work-area bounds, draw/travel feed rates, `accelMmS2` (trapezoidal acceleration limit), `minSegmentMm` (point-decimation tolerance), and the pen-actuator timing (`penPulseMs=260.0`, `penSettleMs=90.0`) for the one-way gear-motor pen mechanism (explained under `PenController` below).

Computed properties: `microstepsPerRev = fullStepsPerRev * microstepping` (= 3200 at defaults); `stepsPerMmX/Y = microstepsPerRev / mmPerRevX/Y` (= 80 steps/mm at defaults).

#### `ToMachine(traj, cfg, fitToBounds=True, optimizeOrder=True)`

Purpose: converts a Trajectory's abstract (x,y) millimetre coordinates into actual machine-bed coordinates, placed at the configured origin, optionally scaled down to fit the available bed area, and optionally stroke-order-optimized.

Inner workings: computes a uniform `scale = min(1, availableWidth/trajWidth, availableHeight/trajHeight)` (never upscales, only downscales if the text would overflow the bed). Maps every point via `(originX + (x - x0)*scale, originY - (y1 - y)*scale)` -- both the trajectory's own frame and the machine bed use y-up convention here, with the text's top row landing at `originY` and subsequent rows descending from there. Each stroke is then decimated (`_Decimate`) to drop points closer together than `minSegmentMm`. If `optimizeOrder`, `OptimizeStrokeOrder` reorders (but does not alter the geometry of) the strokes.

#### `_Decimate(poly, minSeg)`

Purpose: drops interior polyline points that are closer than `minSeg` to the last kept point, while always preserving the first and last point exactly -- reduces G-code line count/step-schedule overhead without changing the path's endpoints or its overall shape beyond the `minSeg` tolerance.

#### `OptimizeStrokeOrder(strokes, rowTolMm=None)`

Purpose: reorders (never alters) the set of pen-down strokes with a greedy nearest-neighbour heuristic, so the pen stops criss-crossing the page between distant strokes, while still writing the page in roughly reading order (not jumping between text rows).

**Inner workings / algorithm.** This is a **greedy nearest-neighbour heuristic for a variant of the Travelling Salesman / stroke-ordering problem** (minimizing total pen-up travel distance), constrained to respect row structure:
1. Strokes are first grouped into text ROWS by their mean y-coordinate, using a tolerance `rowTolMm` (auto-derived as `1.2 * median stroke height` if not given) -- strokes within tolerance of each other's mean y are grouped into the same row, processed in top-to-bottom row order.
2. Within each row, a GREEDY nearest-neighbour walk picks, at each step, whichever remaining stroke has an endpoint (start OR end) closest to the current pen position (`math.dist`), using the first stroke's leftmost-x-first point as the initial seed. If entering from the far end of a stroke is closer than entering from its defined start, the stroke is drawn REVERSED (free for a plotter, since a stroke's drawn appearance is unaffected by direction).

The code explicitly notes the simulation-verified guarantee: since only ORDER (and occasionally direction) changes, never the strokes' actual geometry, the final drawn result is pixel-identical to the unoptimized order, while pen-up travel distance and total job time drop substantially. This is a standard, well-known heuristic (not an exact/optimal TSP solver, which would be computationally infeasible for hundreds of strokes) — greedy nearest-neighbour tours are widely used in practice for exactly this kind of "visit every stop, minimize total travel" problem when an approximate, fast answer is acceptable.

#### `_Clamp(strokes, cfg)`

Purpose: clips every point to the configured bed bounds `[boundsMin, boundsMax]`, returning the clamped strokes plus a count of how many points needed clamping (surfaced as a G-code comment warning).

#### `PenController` (class)

Purpose: tracks and emits the correct pen up/down commands for the specific, unusual hardware constraint described in the module docstring: the pen actuator is a plain one-way gear motor with NO driver and no position feedback -- it can only rotate in one direction, and every 90 degrees of rotation TOGGLES the pen state (up->down or down->up). It cannot be commanded TO a state, only PULSED to flip it. `SetPen(wantUp)` therefore must track the believed current state in software and emit a pulse ONLY on an actual state change -- emitting a pulse when no change is needed would silently invert the pen for the rest of the job (since the hardware itself can never report or be read back to confirm its actual state).

Inner workings: `isUp` starts at `cfg.penStartsUp` (an assumed, not measured, power-on state). `SetPen(wantUp)` is a no-op (returns `[]`) if `wantUp == isUp`; otherwise flips `isUp`, increments the pulse counter, and returns the two G-code lines: the pen toggle code (`M3`=down or `M5`=up per `cfg.penDownCode`/`penUpCode`) and a `G4` dwell command for exactly `(penPulseMs + penSettleMs)/1000` seconds -- long enough for the 90-degree rotation to physically complete plus a small settle margin before the next move begins.

#### `WriteGcode(traj, cfg, path, title='')`

Purpose: the trajectory-to-G-code-command translation -- emits a plain, standard G0 (rapid/pen-up move)/G1 (linear interpolated/pen-down move) G-code file plus the M3/M5 pen toggle codes, which is what server.py serves to the client and what `/api/write/<job_id>` sends to the physical gantry.

**Inner workings / the trajectory-to-G-code encoding.** Calls `ToMachine` to get machine-frame, order-optimized, clamped strokes. Emits a header comment block (author, stroke count, scale factor, steps/mm, pen mechanism description, and a clipped-points warning if any points were clamped). Sets `G21` (millimetre units) and `G90` (absolute positioning). Ensures the pen starts up (`pen.SetPen(True)`), sets the rapid-travel feed rate. For each stroke: lifts the pen (no-op if already up), issues a `G0` rapid move to the stroke's first point (pen-up travel), lowers the pen (`M3`), sets the drawing feed rate, then issues a `G1` linear move for every subsequent point in the stroke (this is purely straight-line/polyline encoding -- **there is no G2/G3 circular-arc interpolation anywhere in this emitter**; every curve, including loops like 'o', is encoded as a dense sequence of short straight G1 segments, which is why the upstream resampling steps like `SmoothStroke` and `_Resample` matter for visual smoothness). After all strokes, the pen is raised and a final `G0` parks the gantry back at the origin, followed by `M2` (program end). The function returns a summary dict (`lines`, `strokes`, `penPulses`, `scale`, `clipped`, `path`).

#### `_TrapezoidTime(dist, vMax, accel)`

Purpose: computes the time to travel distance `dist` under a standard **trapezoidal velocity profile** (accelerate from rest, optionally cruise at `vMax`, decelerate back to rest) with constant acceleration magnitude `accel` -- used only by `WriteStepSchedule`'s timing estimate, not by the G-code emitter itself (G-code feed rates leave trapezoidal profiling to the controller firmware).

**Maths.** The distance needed to accelerate from 0 to `vMax` at rate `accel` is `dAcc = vMax^2 / (2*accel)` (from `v^2 = 2*a*d`). If `dist < 2*dAcc`, the move is too short to ever reach `vMax` -- it is **triangular** instead: peak velocity `vPeak = sqrt(dist * accel)` (solving `dist = vPeak^2/accel`, i.e. accelerate for half the distance, decelerate for the other half), total time `= 2*vPeak/accel`. Otherwise it is a genuine **trapezoid**: time to accelerate plus decelerate is `2*vMax/accel`, plus the remaining cruise distance `(dist - 2*dAcc)` at constant `vMax`.

#### `WriteStepSchedule(traj, cfg, path)`

Purpose: an alternative, lower-level output format to G-code: a CSV of literal `(t_us, type, x_steps, y_steps, dir_x, dir_y, pen_state)` rows -- i.e. exactly the per-move step/direction pulse counts and timing a bare step/dir stepper-driver firmware would need to clock out directly, bypassing any G-code interpreter.

Inner workings: walks the same machine-frame, clamped strokes as `WriteGcode`. `move(x, y, drawing)` converts the millimetre delta into integer step counts (`round(dxMm * stepsPerMmX)`), computes the move's duration via `_TrapezoidTime` at the appropriate feed rate (draw vs travel), and appends a row with the elapsed time, step counts, direction signs, and current pen state. `penTo(up)` emits a `PEN_PULSE_90DEG` row whenever `PenController.SetPen` reports an actual transition, advancing the clock by the pulse+settle time. Returns `(rows, seconds, steps, penPulses, path)` summary statistics.

### G-code round-trip and simulation

#### `ParseGcode(path)`

Purpose: re-parses a previously emitted G-code file back into pen-down polylines in machine millimetre coordinates -- the inverse of `WriteGcode`. This is what makes the "what will the robot actually draw" honesty check possible: rather than trusting the trajectory object that PRODUCED the G-code, this independently re-derives geometry from the literal G-code TEXT, the same way a real controller firmware would interpret it.

Inner workings: a simple line-by-line state machine. Strips comments (everything after `;`). On `M5` (pen up), if a stroke was in progress with ≥2 points, it's finalized into `strokes`; the current polyline resets. On `M3` (pen down), starts a new current polyline seeded with the current position. On `G0`/`G1`, regex-extracts `X`/`Y` values (`re.search(r'X(-?\d+\.?\d*)', ...)`), updates the tracked current position, and if the pen is currently down, appends the new position to the current polyline. This correctly reconstructs exactly the motion a controller executing this exact text would produce.

#### `RenderGcodePreview(gcodePath, cfg, pxPerMm=6.0, strokePx=2, showTravel=True, path=None)`

Purpose: rasterizes the literal, re-parsed G-code (not the original trajectory) onto a bed-sized canvas -- the function referenced by server.py's `/api/gcode/upload` endpoint and used throughout Evaluate.py's `GcodeRoundTrip` for honest machine-path evaluation.

Inner workings: parses strokes via `ParseGcode`; optionally draws the pen-up travel moves as thin pink lines between consecutive strokes (`showTravel`) so wasted motion is visually apparent; draws every pen-down stroke in black. Note the y-flip in `P(pt)`: machine Y increases upward but image Y increases downward, so `imageY = boundsMaxY - machineY`.

#### `RasterizeStrokes(strokes, pxPerMm=8.0, strokePx=2, bounds=None)`

Purpose: a lower-level raw-stroke rasterizer (no G-code parsing involved) returning a boolean ink mask plus the bounds used, consumed by `SimulateAndCompare` to compare two different stroke sets geometrically.

#### `SimulateAndCompare(traj, cfg, gcodePath, pxPerMm=8.0, strokePx=2)`

Purpose: **the quantitative "digital -> machine" loss measurement** — this is `GcodeRoundTrip`'s sibling: rather than producing a human-viewable image, it computes numeric agreement statistics between the ORIGINAL synthesized trajectory (converted to machine coordinates) and the trajectory LITERALLY RE-PARSED from the emitted G-code file, i.e. exactly how much geometric error the trajectory->G-code encode/decode round trip itself introduces (separate from, and upstream of, any physical gantry imperfection).

**Inner workings / maths.** Rasterizes both stroke sets (`RasterizeStrokes`) over their shared bounding box and computes:
- **Intersection-over-Union (IoU)**: `|A ∩ B| / |A ∪ B|` over the two boolean ink masks — the standard set-overlap similarity measure used throughout image segmentation/detection literature.
- **Mean/max nearest-neighbour deviation in millimetres**: for a (possibly subsampled, capped at 4000/6000 points for tractability) set of points from each stroke set, computes for every point in the G-code-derived set its nearest-neighbour distance to any point in the trajectory-derived set (via brute-force pairwise Euclidean distance in chunks of 256, `sqrt(((chunk - src)**2).sum(-1)).min(axis=1)`), then reports the mean and max of these nearest-neighbour distances. This is a form of (one-directional, sampled) **Hausdorff-distance-style** geometric comparison between two point sets/curves — a standard technique for quantifying how closely two curves match when they aren't guaranteed to share exact correspondence point-for-point.

---

## 4. server.py -- Flask API layer

### Role in the pipeline

The web/API layer. Runs on the Odroid alongside the trained models and physical gantry; a separate static frontend talks to it over HTTP/JSON. Loads every model (text recognizer, ink-based writer-ID, stroke-normalised writer-ID) and every style profile ONCE at process start, not per-request, then exposes a set of REST endpoints that call into `SynthesizeHandwriting.py` (synthesis/G-code), `SegmentLean.py` (page-to-lines segmentation for the camera/classify flow), `web_render_helpers.py` (cached glyph-grid/sample-crop images), `camera_capture.py` (camera I/O), and `odroid_direct_drive.py` (the physical gantry driver, imported from a sibling top-level `GantryControl` folder). The Flask plumbing itself (routing, request parsing, file I/O) is explained only briefly here per the brief's own guidance -- the engineering substance is in the modules it calls into, already covered above.

The module has an explicit, candid SECURITY NOTE in its docstring: no user accounts, essentially no input sanitisation beyond filename character-stripping, intended for local-network use only; an optional `WEBAPP_API_KEY` environment variable gates every request behind an `X-API-Key` header check (`_check_api_key`/`_auth`, a `@app.before_request` hook) as a "basic deterrent, not real security," with an explicit warning against open-internet exposure without a VPN/TLS proxy in front of it.

### Startup / model loading

Three models are loaded once at import time: `TEXT_MODEL` (the frozen CTC text recognizer, `PaperCRNNNumpy`), `SHAPE_AUTHOR_MODEL` (the stroke-normalised writer-ID classifier -- "judges synthesis," i.e. the correct judge for machine-rendered/synthesized output, since it doesn't rely on ink density the gantry can't reproduce), and `INK_AUTHOR_MODEL` (the ink-based writer-ID classifier -- "classifies REAL photos," used only on genuine photographed pages where ink density IS a meaningful signal). `PROFILES = SY.LoadAllProfiles()` loads every author's style-profile JSON.

`_prewarm()` (run on a background `threading.Thread`, not blocking server startup) pre-renders and disk-caches every author's glyph-grid image and sample crops (`web_render_helpers.glyph_grid_path`/`sample_crops`) so the first real visitor to the stats page isn't stuck waiting -- the comment notes this first pre-warm can take minutes for personal authors (it runs full page segmentation on their real photos) and explicitly chose a background thread specifically so it never blocks the Read/Write pages during that time. `CACHE_WARM_DONE` tracks per-author completion, checked by the `/api/glyphs` and `/api/samples` routes before attempting to serve (returning a `202` "still preparing" response rather than crashing on an incomplete cache).

### `_check_api_key()` / `@app.before_request _auth()`

Purpose: the whole-server auth gate. A no-op if `WEBAPP_API_KEY` is unset; otherwise `abort(401)` on any request missing a matching `X-API-Key` header.

### `_classify_ink(pil_line_img)`

Purpose: thin wrapper running the INK-based (not stroke-normalised) writer-ID classifier on one real line image, used only by `/api/classify` (never on synthesized output -- the docstring explicitly warns this classifier is only valid for real photos). Returns `(predicted_author_id, {author: probability})`.

### `GET /api/authors`

Purpose: returns the author roster for a frontend dropdown -- `id`, `label`, `slantDeg`, `connectedness` per author, straight from each profile's own top-level fields.

### `POST /api/camera/capture`

Purpose: triggers `camera_capture.capture_image()`, saves the result under a fresh UUID, returns `{image_id, image_url}`. A camera failure returns a clean `500` JSON error rather than propagating a raw exception.

### `POST /api/camera/preview/start` / `POST /api/camera/preview/stop` / `GET /api/camera/preview/stream`

Purpose: a toggleable live camera preview, deliberately NOT always-on (the module comment explains: leaving a USB webcam's sensor running continuously is needless heat/wear). `start`/`stop` call `camera_capture.preview_start`/`preview_stop`. `stream` serves an MJPEG multipart stream (`multipart/x-mixed-replace`) by repeatedly yielding `camera_capture.preview_frame_jpeg()` frames in a generator, returning `409` if preview wasn't started first.

### `GET /api/images/<image_id>`

Purpose: serves a previously captured/uploaded image by id. Sanitises `image_id` to alphanumeric characters only before building the filesystem path (the project's only real input-sanitisation measure, applied consistently to every id-based route below).

### `POST /api/classify`

Purpose: the "read a photographed page back" endpoint -- accepts either a raw uploaded image file or a previously-captured `image_id`, plus optional `expected_text`/`expected_author` for scoring. Request format: `multipart/form-data` with either `image` (file) or `image_id` (string), plus `expected_text`, `expected_author` form fields. Response JSON: `{predicted_text, text_accuracy_pct, predicted_author, author_confidence_pct, author_correct, expected_author, lines_detected, per_author_confidence_pct}`.

Inner workings: segments the page into individual line crops via `SegmentLean.SegmentLines` (the project's production page-layout segmenter, covered elsewhere in the codebase, not in this file's scope). For each line crop: reads text via `np_inference.text_model.ReadText` (greedy CTC DECODING, not forced alignment -- this is genuinely reading unknown text) and classifies the writer via `_classify_ink`. Concatenates all lines' predicted text with spaces; averages the per-author probability vectors across all lines (not just a majority vote of per-line predictions) and takes the argmax of the AVERAGE as the page-level predicted author -- a form of simple ensemble/probability-pooling across lines. If `expected_text` was given, computes `CharAcc` (1 - CER, from `np_inference.text_model`) against it.

### `_job_response(job_id, line_count, pen_pulses, preview_img, expected_text=None, **extra)`

Purpose: the shared response-shape builder for both `/api/generate` and `/api/gcode/upload` -- reads the rendered preview image back through the text recognizer (`ReadText`) and reports `CharAcc` against `expected_text` if one was supplied, so the user immediately sees an honest "does this actually say what I asked for" signal alongside the generated job.

### `POST /api/generate`

Purpose: the main text-to-robot-drawing endpoint. Request JSON: `{text, author, nTries?}` (nTries defaults to 20). Response JSON (via `_job_response`): `{job_id, gcode_url, preview_url, gcode_line_count, pen_pulses, read_back_text, text_accuracy_pct, synth_seconds}`.

Inner workings: validates `text` is non-empty and `author` is a known profile id. Calls `SY.SynthesizeJointBestOf` (the production method) with `nTries=n_tries`, `mmPerXh=4.0`, and a `lineWidthMm` computed from the gantry's own configured bed bounds (`cfg.boundsMaxXmm - cfg.originXmm - 5`, i.e. the available drawable width minus a 5mm safety margin). Writes G-code (`SY.WriteGcode`) and a preview PNG (`SY.RenderTrajectory` at `pxPerMm=18.0`, matching the resolution used throughout Evaluate.py) into a fresh per-job directory under `NOGIT/WebJobs/<job_id>/`.

### `GET /api/gantry/status` / `POST /api/gantry/calibrate` / `POST /api/write/<job_id>`

Purpose: the physical-gantry control surface. `status` reports connection/calibration state (attempting a connection if not already connected, catching `GANTRY.GantryNotConnectedError` into a clean `200` JSON response rather than an error). `calibrate` runs the full homing/steps-per-mm measurement sequence (described in `odroid_direct_drive.py`, out of this document's scope) -- the frontend greys out Generate/Upload/Send until this has succeeded once per server session. `write/<job_id>` sends a previously generated/uploaded job's G-code file to the physical gantry via `GANTRY.write_gcode_file`; every expected failure mode (not connected, not calibrated, end-stop tripped mid-run / `estopped`) is caught and reported as an ordinary `200` JSON response with a `sent: false` flag and a human-readable message, rather than surfaced as a `500` -- the docstring frames these as "expected, everyday states, not server bugs."

### `POST /api/gcode/upload`

Purpose: lets a user test an arbitrary hand-made or downloaded G-code file through the same preview/read-back/send-to-gantry flow as a generated job -- useful for drawing pictures/shapes, not just reproduced handwriting. Request: `multipart/form-data` with `gcode` (file) and optional `expected_text`. Counts non-blank lines and `M3`/`M5` occurrences (pen pulses) directly from the raw text, then calls `SY.RenderGcodePreview` to parse and rasterize it (returning a clean `422` JSON error if the file doesn't parse as valid G-code).

### `GET /api/gcode/<job_id>` / `GET /api/images/job_<job_id>`

Purpose: serve the raw `.gcode` file / the rendered preview PNG for a previously generated or uploaded job, by sanitised job id.

### `GET /api/model_info`

Purpose: a static(ish) JSON dump of model architectures, parameter counts, file sizes, and PRE-MEASURED accuracy figures (not computed live -- these are the headline numbers from the project's own Evaluate.py runs, hardcoded as reference values, e.g. `real_ink_10authors_pct: 89.4` for the text recognizer, `real_photographed_pages_pct: 100.0` / `synthesised_output_pct: 20.0` for the ink-based writer-ID classifier (explicitly noted as only valid on real photos, collapsing on synthesized output since it relies on ink density/pressure a pen-plotter can't reproduce), `real_ink_pct: 96.7` / `synthesised_output_pct: 99.2` for the stroke-normalised judge, and the end-to-end `reproduction_pipeline.measured_on_novel_sentences` block citing `SynthesizeJointBestOf`/`Evaluate.py rewrite` as the method/measurement source). `count_params(model)` sums `arr.size` over every array in the model's state dict (`model.sd`).

### `GET /api/glyphs/<author>` / `GET /api/samples/<author>` / `GET /api/samples/<author>/<filename>`

Purpose: serve the pre-warmed (or on-demand, if the pre-warm cache check fails) glyph-grid PNG (`web_render_helpers.glyph_grid_path`) and real-handwriting sample crops (`web_render_helpers.sample_crops`) for the "what does this author's handwriting actually look like" stats/comparison page. All three check `CACHE_WARM_DONE` first and return a `202` "still preparing" response if the background pre-warm for that author hasn't finished yet, rather than letting the first slow request time out or crash. The `<filename>` route additionally validates that the requested file actually belongs to that author (`path.name.startswith(f"sample_{author}_")`) before serving it.

---

## 5. Evaluate.py -- official measurement scripts

### Role in the pipeline

This file is the project's SOURCE OF TRUTH for every headline accuracy number cited elsewhere (including the hardcoded figures in `server.py`'s `/api/model_info`). It is a straight merge of four originally-separate scripts (EvaluateStyle.py, VerifyRewrite.py, VerifyShapeStyle.py, EvaluateLegibility.py) into one file with an `argparse` subcommand dispatcher, with no logic changes from the originals. Unlike the other files in this document, Evaluate.py is allowed to import `torch` directly (it loads the original PyTorch-trained checkpoints for the TORCH-based judge models used to independently verify the numpy inference path's own internal scoring) -- this is explicitly a MEASUREMENT script, not part of the live inference path.

### `style` mode (from EvaluateStyle.py)

**What it measures.** Does synthesized handwriting actually look like the author it claims to be, on HELD-OUT text the profile was never fitted on (so the synthesizer is never scored on text it already "knows" from its own glyph library).

#### `LoadAuthorModel(device)`

Purpose: loads the correct writer-ID judge checkpoint with a three-tier fallback: prefers the stroke-normalised SHAPE-ONLY model (`author_shape_10new_weights.pt`) if present, else an older ink-aware model, else a third fallback. The docstring explains WHY the shape-only model is the correct judge for SYNTHESIZED output specifically: the plain ink-aware classifier was measured to collapse to ~20% accuracy on synthesized output, because it leans on ink density which neither the gantry's constant-width pen nor vector rendering faithfully reproduces, whereas the stroke-normalised model (no such dependency) reaches 96.7% on real held-out lines.

#### `ClassifyImage(model, pilImg, device)`

Purpose: classifies one line image's author. Critically applies `StrokeNormalize` (binarize -> skeletonize -> re-ink at fixed width, from `TrainAuthor.py`) to the image FIRST -- since the loaded model's own weights were calibrated on stroke-normalised input, feeding it a raw render/photo directly would put it off-distribution and make the prediction meaningless.

#### `MeasureLine(grayArr)`

Purpose: extracts a small feature vector (`slant`, `xh`, `wordGap`, `inkPerXh`) from one line image using the same `ProfileIO` primitives (`BinarizeLine`, `CoreBand`, `EstimateSlantDeg`) used throughout the pipeline, for the real-vs-synthesized FEATURE AGREEMENT report (not the writer-ID classifier score itself). `wordGap` is measured identically to `RenderRefStats` in BuildStyleProfile.py (median of column-gap runs wider than 0.5·xh). Returns `None` if the line has too little ink or no valid x-height band.

#### `StyleEvaluate(mmPerXh=4.0, pxPerMm=18.0, nSeeds=3, jitter=1.0, verbose=True)`

Purpose: the main measurement loop. For each author, synthesizes each of up to 12 held-out (or, if none, non-holdout) texts `nSeeds` times, classifies each render with the shape-only judge, and computes per-author writer-ID accuracy as `correct/total`. Also collects `MeasureLine` features for both the synthesized renders (first seed only) and the author's own real held-out lines, for the feature-agreement report. Returns `{perAuthor, overall, rows, features}`.

#### `FeatureReport(featRows)`

Purpose: for each author with both real and synthesized feature rows, computes the **median** (`np.nanmedian`, robust to the occasional failed measurement) of `slant`, `wordGap`, `inkPerXh` separately for real and synthesized lines, returning a `{author: {feature: (realMedian, synthMedian)}}` dict -- the real-vs-synth agreement numbers printed in the final report.

#### `BuildComparisonSheet(profiles, holdoutTexts, path, mmPerXh=4.0, pxPerMm=18.0, maxW=1500)`

Purpose: builds a single PNG sheet with one real line (top) directly above its synthesized rendering of the SAME text (bottom), per author -- a qualitative visual sanity check, laid out with PIL `ImageDraw`.

#### `MachineDistort(traj, cfg, rng, backlashMm=0.15, jitterMm=0.05)`

Purpose: synthetically injects the three dominant error sources a REAL gantry would add on top of a mathematically perfect trajectory, so writer-ID accuracy can be re-measured on a trajectory that approximates "what the physical machine would actually produce" without needing the physical machine itself.

**Inner workings / the three error models.**
1. **Microstep quantization**: every point is rounded to the nearest achievable step position, `round(x / qx) * qx` where `qx = 1/stepsPerMmX` -- the physical stepper motor literally cannot land between two adjacent microsteps.
2. **Belt backlash**: a constant lost-motion offset (`backlashMm`) is added to an axis's position whenever that axis's direction of travel REVERSES (detected by `dx * lastDx < 0`), modelling the mechanical slack in a real belt-and-pulley drivetrain that is only "taken up" again after a direction change -- a standard, well-known real-world CNC/plotter error source.
3. **Random positioning noise**: independent Gaussian jitter `N(0, jitterMm)` added to each axis at every point, modelling generic mechanical/electrical noise.

#### `EvaluatePhysical(...)`

Purpose: re-runs the same writer-ID measurement as `StyleEvaluate` but on `MachineDistort`-ed trajectories, estimating what style-match accuracy survives the (simulated) digital-to-physical transition.

#### `CollectHoldoutSamples(maxPerAuthor=1)`

Purpose: gathers one real held-out line image + its transcript per author (length-filtered to 18-62 characters, a readable single-line length) for use by `BuildComparisonSheet`.

#### `RunFullEvaluation(...)`

Purpose: the `style` subcommand's top-level driver -- runs `StyleEvaluate`, prints per-author and overall writer-ID accuracy against an explicit 85% target, builds the feature-agreement report and comparison sheet, and dumps `results.json`.

### `rewrite` mode (from VerifyRewrite.py)

**What it measures.** Does the robot's output say the RIGHT WORDS in the RIGHT HAND, on sentences the system has genuinely never seen in any form (not drawn from IAM at all) -- both writer-ID and text-recognition correctness, measured BOTH on the idealised trajectory and through the literally emitted/re-parsed G-code.

`NOVEL_SENTENCES`: six short, hand-written-for-this-test English sentences, deliberately exercising letter combinations/words absent from the IAM training/profile-fitting data.

#### `LoadTextModel(device)` / `ReadText(model, pilImg, device)`

Purpose: loads the frozen torch text recognizer checkpoint directly (not via the numpy inference path -- this is the independent measurement path) and runs greedy CTC decoding (`decode_ctc`) on one line image.

#### `CharAcc(pred, truth)`

Purpose: **Character Error Rate complement**, the project's text-accuracy metric. 

**Formula, exactly as implemented:**
```
CharAcc(pred, truth) = max(0, 1 - levenshtein(pred, truth) / len(truth))
```
i.e. `1 - CER` where `CER = levenshtein(pred, truth) / len(truth)` is the standard **Character Error Rate** used throughout ASR/OCR/handwriting-recognition literature, with `levenshtein` being the ordinary **Levenshtein edit distance** (minimum number of single-character insertions, deletions, substitutions to transform one string into the other), imported from `TrainText.py`. Floored at 0 (an edit distance larger than the truth's own length would otherwise give a negative "accuracy"). Special case: if `truth` is empty, returns `1.0` if `pred` is also empty, else `0.0`.

#### `WordAcc(pred, truth)`

Purpose: a word-level accuracy metric, computed as the fraction of the truth's words recoverable, IN ORDER, from the prediction.

**Formula.** `WordAcc = LCS(truth.split(), pred.split()) / len(truth.split())`, where `LCS` is the length of the **Longest Common Subsequence** between the two word sequences (case-insensitive comparison), computed by the standard `O(m·n)` dynamic-programming LCS recurrence:
```
dp[i+1][j+1] = dp[i][j] + 1               if truth[i].lower() == pred[j].lower()
             = max(dp[i][j+1], dp[i+1][j]) otherwise
```
This is the classical **Longest Common Subsequence** algorithm (textbook dynamic programming, same family of algorithm as the Levenshtein-distance recurrence above, specialized to the no-substitution, order-preserving "subsequence" case) applied at word granularity rather than character granularity -- it rewards getting the right words in the right relative order even if some words in between were garbled or dropped, which is a more forgiving and arguably more human-relevant metric than strict positional word accuracy.

#### `GcodeRoundTrip(traj, cfg, tmpPath, profile, pxPerMm=18.0)`

Purpose: emits G-code from a trajectory, re-parses the literal file (`SY.ParseGcode`), wraps the re-parsed strokes in a fresh `Trajectory`, and re-renders it through the SAME calibrated `RenderTrajectory` path (same author-specific ink density matching etc. as the original) used for the idealised trajectory -- i.e. this measures "what the ACTUAL EMITTED G-CODE would draw" as opposed to what the in-memory trajectory object represents before encoding, and is the mechanism behind every "(through G-code)" number reported throughout this file.

The code's own comment records a measured, counter-intuitive-at-first finding that justifies rendering with the author's own calibrated ink density here rather than a fixed stroke width: doing the latter was measured to DESTROY writer-ID accuracy (89% -> 4%) while leaving TEXT accuracy completely untouched -- because stroke weight is a style feature the writer-ID model leans on heavily (text recognition cares about shape, not weight), so the machine-path comparison has to preserve ink density or the comparison becomes meaningless for that axis specifically.

#### `RunRewrite(nSeeds=2, mmPerXh=4.0, pxPerMm=18.0, saveSamples=True)`

Purpose: the `rewrite` subcommand's top-level driver and the SOURCE of the headline end-to-end numbers quoted in `server.py`'s `/api/model_info`. For every author x every `NOVEL_SENTENCES` sentence x `nSeeds` seeds: synthesizes via `SynthesizeJointBestOf` (the production method, `nTries=30`), classifies/reads the direct render AND the G-code-round-tripped render, and accumulates `idR/idG` (writer-ID correct, direct/G-code), `charR/charG`, `wordR/wordG` (CharAcc/WordAcc, direct/G-code). Separately computes a MATCHED reference: for every real held-out IAM line, synthesizes the SAME words in that author's own style and reads both the real line and the synthesized one with the identical recognizer -- explicitly so that synthesis's score on novel sentences is never compared against the recognizer's score on unrelated real text, which the comment flags would not be a like-for-like comparison. Prints and returns per-author and aggregate summary statistics.

### `shapestyle` mode (from VerifyShapeStyle.py)

**What it measures.** Style accuracy the way the physical gantry will actually be judged: both sides of the comparison go through the SAME stroke-normalisation the shape-only judge was trained on, so the comparison is on geometry alone (slant, proportions, spacing, letterforms, connections) with ink density/pressure entirely removed as a variable, since the single-pen gantry cannot reproduce it anyway.

#### `LoadShapeModel(device)` / `Classify(model, pil, device)`

Purpose: loads the shape-only writer-ID checkpoint (`author_shape_10new_weights.pt`) directly and classifies one (already stroke-normalised) image via plain `argmax` over the model's logits -- notably simpler than `ClassifyImage` in that it does NOT apply `StrokeNormalize` itself; callers are responsible for normalising first (see `RunShapeStyle` below).

#### `RunShapeStyle(nSeeds=2, mmPerXh=4.0, pxPerMm=18.0)`

Purpose: for every author x `NOVEL_SENTENCES` x seed: synthesizes via plain `SynthesizeText` (not the best-of-N/repair methods -- this measures the BASE synthesis quality, uninflected by selection), renders, applies `TrainAuthor.StrokeNormalize` to BOTH the direct render and the G-code-round-tripped render, classifies both, and accumulates per-author direct/G-code shape-style accuracy. Also measures a CEILING: the same shape-only judge's accuracy on the authors' own REAL held-out lines (after the same stroke normalisation) -- giving the honest upper bound synthesis is being compared against, since even real handwriting doesn't classify at 100%.

### `legibility` mode (from EvaluateLegibility.py)

**What it measures.** The project's HEADLINE readability metric: how readable is the rewriting pipeline's output, on text it has genuinely never seen, when every author is written with exactly ONE pen (uniform ink, matching physical reality). Deliberately adversarial to memorisation: `NOVEL_CORPUS` (~40 sentences) is ordinary present-day English, not drawn from IAM's 1961 LOB-corpus text at all, and specifically includes visually-confusable bigrams (`rn/cl/vv/mm/ee`) that stress-test letter-shape ambiguity.

#### `_ci_char_acc` / `_ci_word_acc` / `_shape_pred`

Purpose: case-insensitive wrappers around `CharAcc`/`WordAcc` (a human reading the word doesn't care about capitalisation, so legibility scoring shouldn't either), and a convenience wrapper that stroke-normalises an image and returns the shape-only judge's predicted author id (or `None` if normalisation failed, e.g. too little ink).

#### `LegibilityEvaluate(authors=None, nSent=None, nSeeds=1, lam=None, perAuthorLam=False, gcodeEvery=6, pxPerMm=18.0, mmPerXh=4.0, saveSheets=True, legible=False, nTries=4)`

Purpose: the main `legibility` driver. For each author, for each sentence/seed: synthesizes via `SynthesizeLegible` (if `legible=True`, the best-of-N+repair path) or plain `SynthesizeText` (otherwise), ALWAYS with `uniformInk=True` rendering. Reads back char/word accuracy (case-insensitive) and the shape-only writer-ID prediction. Every `gcodeEvery`-th sentence additionally runs the G-code round trip for a cheaper, sparser machine-path check (full G-code round-tripping for every single sentence/seed would be far more expensive without materially changing the aggregate number). Saves annotated comparison sheets (`_SaveSheet`) for the first three sentences per author. Reports per-author and aggregate `char`, `word`, `shapeId` (direct and G-code), and prints the 10 worst-scoring individual renders across the whole run for qualitative inspection. Writes `results.json`.

#### `SweepLambda(lams=(0.0, 0.3, 0.5, 0.7, 1.0), authors=None, nSent=16, nSeeds=1, mmPerXh=4.0, pxPerMm=18.0, withShape=False)`

Purpose: a controlled experiment sweeping the global `legibility` blend parameter across several fixed values, with EVERYTHING ELSE held constant (same seeds, same models loaded once) across all lambda values, so the legibility-vs-style trade-off curve is directly comparable point to point. Writes `sweep.json` and prints a per-author table showing each author's char-accuracy trajectory as lambda increases -- this is the kind of ablation data that should go directly into the report's results/discussion of the legibility-style trade-off.

#### `TuneLegibility(lams=(0.35, 0.55, 0.75, 1.0), nSent=10, nSeeds=1, useLegible=True, mmPerXh=4.0, pxPerMm=18.0, write=True, nTries=3)`

Purpose: the actual CALIBRATION procedure that PRODUCES the `legibilityLambda` value stored in each author's profile JSON (consumed later by `SynthesizeText`'s default legibility resolution). For each author, tries each candidate lambda from smallest to largest and picks the SMALLEST one that clears both `LEGIBILITY_TARGET_CHAR=0.90` and `LEGIBILITY_TARGET_WORD=0.62` on the novel corpus (falling back to the largest/most-print-like lambda if no candidate clears both targets). The code's own comment explains how these specific target numbers were chosen: the frozen recognizer tops out near 96% char / 84% word even on ISOLATED CLEAN PRINT text (its own ceiling, due to residual confusions on letters like m/n/u/w, r/v that a human reads without any trouble) -- so the calibration bar is deliberately set at what is actually reachable through this specific recognizer, meaning "clears the target" means "reads as cleanly as the print font itself does," and an author whose hand genuinely cannot reach that even at full print-like synthesis is pushed to `lam=1.0` (maximum anchor use) rather than an unreachable number. If `write=True`, directly patches and re-saves each author's profile JSON with the chosen `legibilityLambda`.

#### `_SaveSheet(author, rows, path)`

Purpose: builds a small PNG sheet of (want-text, got-text, rendered-image) rows for one author, used by both `LegibilityEvaluate` and (implicitly, via the same pattern) qualitative inspection elsewhere.

### Subcommand dispatcher

`build_arg_parser()` / `main()`: a plain `argparse` subparser setup mapping `style`/`rewrite`/`shapestyle` (no options, matching their original standalone-script behaviour) and `legibility` (with its full original flag set: `--authors`, `--sentences`, `--seeds`, `--lam`, `--per-author-lam`, `--gcode-every`, `--sweep`, `--tune`, `--legible`, `--tries`) onto the corresponding driver functions above.

---

## 6. web_render_helpers.py and camera_capture.py

### web_render_helpers.py -- role in the pipeline

Pure rendering helpers (no Flask/HTTP code) consumed by `server.py`'s `/api/glyphs/<author>` and `/api/samples/<author>` routes, to populate the web app's "model/style info" stats page. Results are cached to disk under `NOGIT/WebCache/` since they are mildly expensive to render and never change unless the underlying profiles themselves change.

#### `_font(size, bold=False)`

Purpose: a tiny memoized font loader, trying to load Windows' Consolas (`consola.ttf`/`consolab.ttf`) at a given size, falling back to PIL's built-in default font if that font file isn't found (e.g. on non-Windows deployment).

#### `_render_one_glyph(g, cell=90, pad=8)`

Purpose: renders a single stored glyph dict (one variant, from a profile's `glyphs[char]` list) as a small square PIL image -- the basic tile used by the glyph-grid sheet below.

Inner workings: computes the glyph's own bounding box across all its strokes, a uniform scale factor fitting the larger dimension into `cell - 2*pad` pixels, and a coordinate transform `tx(p)` that maps the glyph's (x,y) [y-up] into image pixel space (y-down, hence the `cell - pad - ...` flip). Draws each stroke as a `PIL.ImageDraw.line` of width 2.

#### `glyph_grid_path(author)`

Purpose: builds (or returns the cached path to) a single PNG sheet showing every character an author has stored variants for, one row per character, one column per variant -- explicitly matching the manual-inspection sheets used during development (i.e. this function literally productionises what was originally an ad-hoc debugging visualization). Imports `SynthesizeHandwriting.LoadAllProfiles` to get the glyph data; raises `KeyError` for an unknown author.

#### `sample_crops(author, n=2)`

Purpose: returns up to `n` (image, transcription) pairs of REAL, non-holdout handwriting for one author, for the "what does this author's actual handwriting look like" gallery, cached to disk after first render.

Inner workings: for a dataset author, filters `IAMLineDatasetRaw` samples to that author's own page; for a personal author, calls `TrainAuthor.add_personal_samples` to populate the personal-author samples into the same dataset-sample structure, then filters similarly. Picks the first `n` non-holdout lines with transcript length in `[20, 70]` characters (a reasonably representative single-line length for display), saves them as individual PNGs plus a `meta.txt` of their transcripts for cache reuse.

#### `clear_cache(author=None)`

Purpose: deletes cached renders (glyph grids and/or sample crops), either for one specific author or everything (`author=None`), via glob-pattern deletion -- intended to be called by hand after rebuilding profiles/labels so the web stats page doesn't keep serving stale images.

### camera_capture.py -- role in the pipeline

Captures still images and MJPEG preview frames from whatever USB camera is plugged into the machine this runs on (the project's camera is a generic UVC/USB-Video-Class webcam, not a Raspberry-Pi-style CSI ribbon camera, so `cv2.VideoCapture` -- which talks to any UVC device through the OS's standard camera driver on both Windows and Linux -- is the right and only API needed; no platform-specific branching is required). Used by `server.py`'s `/api/camera/*` routes.

#### `camera_available(index=None)`

Purpose: a cheap connectivity check -- can the camera actually be opened AND produce one readable frame -- so the server can report a clean "no camera" error instead of letting a stack trace surface when testing off the physical Odroid (e.g. on a development laptop with no camera attached, or one already in use by another application).

#### `preview_start(index=None)` / `preview_stop()` / `preview_active()`

Purpose: `preview_start` opens and holds open a single global camera handle (`_preview_cap`), guarded by a lazily-created `threading.Lock` (`_get_lock`) so concurrent Flask request-handler threads can't race on it; it is a safe no-op if already open. `preview_stop` releases the handle. This start/stop toggle is deliberate, not incidental: the module docstring explains the camera should only be powered on while someone is actually watching the live feed to frame a shot and let autofocus settle, not continuously.

#### `preview_frame_jpeg()`

Purpose: grabs one frame from the already-open preview handle and JPEG-encodes it (`cv2.imencode`), for the MJPEG streaming endpoint in `server.py`. Raises `RuntimeError` if preview wasn't started, or if the frame read/encode fails.

#### `capture_image(timeout=10, index=None)`

Purpose: captures and saves one still JPEG image, returning its `Path`. This is the function behind `/api/camera/capture`.

Inner workings: if a live preview is currently open, grabs the frame from THAT existing handle instead of opening a second one -- explicitly because most webcams only allow a single process/handle on the device at a time, and because this is the intended user workflow anyway: watch the live preview until autofocus settles on the page, then capture "the good image" from what's already showing, with no need to re-open/re-warm a fresh capture handle. Otherwise, opens a fresh handle, discards the first `WARMUP_FRAMES=5` frames (a UVC webcam's very first frames after opening are often dark/unfocused while its auto-exposure algorithm settles -- a standard recommendation from essentially every OpenCV capture tutorial), and keeps the last successfully read frame within the `timeout` window. Always releases the handle it opened (but never the shared preview handle, which it doesn't own). Raises descriptive `RuntimeError`s for every distinct failure mode (camera won't open, already in use, no readable frame within timeout, write failure).

---

## Known limitations / honest caveats

### Bug 1: Douglas-Peucker collapse on closed curves (`SimplifyPolyline`, `BuildStyleProfile.py`)

**Before.** `SimplifyPolyline`'s original implementation measured every intermediate point's perpendicular distance from the straight chord connecting the polyline's FIRST and LAST points, exactly as the classical Douglas & Peucker (1973) algorithm specifies for an open curve. For a CLOSED curve -- any letterform whose pen path starts and ends at (nearly) the same point, most obviously the loop of a lowercase 'o', but also the closed portion of letters like the bowl of 'a', 'e', the loop of 'l' -- the first and last points coincide, so the "chord" has zero length and no well-defined direction. The point-to-line-segment distance formula `d = |ab.x*rel.y - ab.y*rel.x| / ||ab||` then has both numerator and denominator collapsing toward zero together as `||ab|| -> 0`, so every candidate point's measured distance reads as approximately zero REGARDLESS of how far it actually sits from the "chord." Since the algorithm's termination condition is "if the farthest point's distance is still ≤ eps, stop and return just the two endpoints," this spurious near-zero reading triggers IMMEDIATE termination on the very first recursive call, collapsing the entire closed loop down to a single repeated point. The code's own comment records the measured scale of the damage before the fix: roughly 23% of one author's (Yeukita's) stored 'o' variants were affected, against only about 2% of the open-curve 'e' (which has genuinely well-separated start/end points and so was not vulnerable to this failure mode) -- i.e. the bug selectively destroyed closed letterforms while leaving open ones intact, exactly as the geometry of the bug predicts.

**After.** Before running ordinary Douglas-Peucker, the function now checks whether the curve is (near-)closed: it computes every point's distance from the START point, and compares the final point's own distance-from-start against a quarter of the curve's own maximum extent (`fromStart[-1] <= max(1.5, 0.25 * fromStart.max())`). If so, it finds the point FARTHEST from the start (geometrically, roughly the "far side" of the loop) and splits the polyline there into two genuinely OPEN arcs, each of which is then simplified independently by the normal, correct Douglas-Peucker recursion (each arc now has well-separated, well-defined endpoints, so its own chord has real length and direction) before the two results are concatenated back together. This is a standard, documented workaround for applying open-curve polyline simplification to closed contours.

### Bug 2: fixed-constant fragment gate vs. per-character-relative gate (`BuildAuthorProfile`, `BuildStyleProfile.py`)

**Before.** The variant-quality fragment filter used a single, FIXED minimum pen-length constant (the comment names it `MIN_PEN_LEN`, at roughly 1.05 x-heights) applied uniformly across all ~74 characters in the charset, to reject glyph instances whose total traced pen path was "too short to be a real letter" (i.e. likely a scanner-speckle or isolated stroke fragment rather than a genuine letterform). This treated every character as if it should have roughly the same amount of pen-path length, which is false: a dotless lowercase 'i' is legitimately drawn with a single short stroke of roughly 0.70 x-heights of pen travel, while a looping 'o' legitimately involves roughly 2.3 x-heights of pen travel for one full circuit. Held against a single ~1.05 x-height floor, this meant essentially every honestly-extracted, genuinely-correct 'i' variant (0.70 x-heights) fell BELOW the fixed threshold and was discarded as a "fragment" -- while the only 'i' instances that SURVIVED the filter were, perversely, the CONTAMINATED ones where a cursive cut had bled into part of a neighbouring letter (inflating the measured pen length above the fixed floor by accident). The code's own comment states the measured consequence plainly: this threw away 77% of this project's legitimate dotless 'i' variants, and every stored 'i' that DID survive into the profile had effectively become "si", "ti", or "fi" -- i.e. a neighbour-contaminated shape, not a clean 'i'.

**After.** The gate is now computed RELATIVE to each character's own population: `penFloor = max(MIN_PEN_ABS, FRAGMENT_PEN_FRAC * median(_PenLength(g) for g in insts))`, with `FRAGMENT_PEN_FRAC = 0.5` and an absolute floor `MIN_PEN_ABS = 0.30` (still present to catch true specks regardless of character). `_PenLength(g)` is the straightforward sum, over every stroke in the glyph, of the stroke's own traced arc length (`sum of consecutive-point Euclidean distances`) -- the actual total distance the pen's centerline travels while drawing this one glyph instance. Because the threshold for 'i' is now set relative to the MEDIAN pen length actually measured across this author's own 'i' instances (rather than a fixed, letter-agnostic number), a short-but-genuine dotless 'i' at ~0.70 x-heights comfortably clears its own character-specific floor of `0.5 * 0.70 ≈ 0.35` x-heights, while an 'o' is compared only against other 'o's' own typical pen length. A second, complementary check, `_LongestStroke(g) >= MIN_LONGEST_STROKE (0.55)`, additionally guards against a glyph whose TOTAL pen length looks acceptable only because of a spurious EXTRA short stroke stacked on top of one genuinely tiny fragment stroke -- i.e. it is not enough for the sum to clear the bar; at least one individual stroke must itself be substantial.

### Other noted trade-offs and TODO-style findings in the code comments

- **Cutting along the slant vs. vertically** (`ExtractLineGlyphs`, `BuildStyleProfile.py`): cutting glyph boundaries in the de-slanted coordinate frame was tried and is geometrically "more correct," and visibly improved tall-letter rendering for steeply-slanted hands -- but it measured WORSE end-to-end (writer-ID 81.8% -> 68.6% uncorrected / 66.6% corrected), apparently because glyphs cut on strictly vertical boundaries carry a consistent slice of their slant context and so retile more naturally at synthesis time. Vertical cutting was kept on this measured evidence despite being the geometrically less "pure" choice -- a good example of a design decision the report should present as empirically, not theoretically, justified.
- **Loop-crossing cut-cost penalty** (same function): penalising cut columns that cross a letter's loop (on the theory that this would discourage slicing a round letter through its open middle) was tried and measured to change NOTHING (Yeukita's 'o'-sliver rate stayed at 27%) -- the actual root cause turned out to be the `SimplifyPolyline` closed-curve bug described above, not cut placement at all. This is recorded directly in the code as a rejected hypothesis.
- **Snapping cut boundaries to distant true ink gaps** (same function): searching further than the local snap radius for a genuine ink gap was tried and measured WORSE (Yeukita's dropped-glyph rate 12% -> 18%), because internal word-interior boundaries have no gap to find and were instead dragged onto a distant word-boundary gap, collapsing the glyph cells between them.
- **Awarding a whole unjoined ink component to whichever character holds most of it**, instead of slicing it at the vertical boundary (same function): tried, measured no better on print hands and slightly WORSE on one specific author (writer 551); not adopted.
- **`CORE_CURSIVE_CONN` threshold tuning** (`SynthesizeHandwriting.py`): lowering this from 0.5 to 0.3 fixed one specific author's (150's) persistent page-level misclassification as a different author (384) on two independently-built sentence pools, but then REGRESSED writer-ID/text/word accuracy on the project's own official `NOVEL_SENTENCES` validation pool (writer-ID 79.2% -> 77.5%, text 77.7% -> 77.5%, word 43.1% -> 42.1%) -- i.e. the change had overfit specifically to the sentence pools used to validate it. Reverted to 0.5; author 150's confusion with author 384 is explicitly left as an OPEN, UNSOLVED problem in the code's own comments rather than something this change actually fixed.
- **`JOIN_LEGIBILITY_COUPLING` set to 0.0, deliberately**: an earlier version coupled the cursive-join probability to the legibility blend, which silently converted cursive hands into print-like ones as legibility rose (a measured 94%-joined hand dropped to only 57% joined at legibility 0.65) -- since a hand's join rate is a MEASURED property of the writer, not a quality dial, this coupling was removed entirely (hardcoded to zero) so legibility now affects letter SHAPE/slant/spacing but never silently erases the joining behaviour that makes a cursive hand recognisably cursive.
- **Style calibration's trade-off** (`StyleCalibration`, `SynthesizeHandwriting.py`): correcting rendered slant/ascender drift toward the author's real measured values was found to slightly WORSEN the ink-sensitive writer-ID judge (86.0% -> 84.5%) while clearly IMPROVING the shape-only judge that actually predicts the physical single-pen robot's output (59.2% -> 63.3% direct, 58.3% -> 66.7% through G-code). The calibration is kept on, on the explicit stated reasoning that the shape-only number is the one that matters for the real deployment target.
- **Rendering the G-code round trip at a fixed stroke width instead of the author's calibrated ink density** (`GcodeRoundTrip`, `Evaluate.py`): tried and measured to destroy writer-ID accuracy (89% -> 4%) while leaving text accuracy completely unaffected, confirming that stroke weight specifically (not shape) is what the writer-ID judge is sensitive to; the calibrated-ink-density rendering path was kept.
- **Personal-author glyph cuts and recognizer choice** (`BuildStyleProfile.py` header comment): the personal fine-tuned recognizer was previously assumed to be the best reader of the personal authors' own handwriting for cutting purposes, but was measured (on the newer SegmentLean crops used everywhere now) to read one personal author (Thiya) at only 67.7% char / 20.0% word accuracy, against 96.5%/85.3% for the newer joint (Teklia+personal, trained together) model -- meaning the old personal fine-tune was placing that author's glyph cuts "nearly blind." The code now prefers the joint recognizer whenever its weights file is present, falling back to the personal fine-tune only if it is not.
- No explicit `TODO`/`FIXME`-tagged comments were found verbatim in any of the six files covered by this document; every open issue noted above is instead documented as a dated, measured "tried X, reverted/kept because Y" comment directly in the source, which is the convention this codebase consistently uses in place of bare TODO markers.



