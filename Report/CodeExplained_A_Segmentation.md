# Code Explained A: Segmentation (Image Processing & Line Segmentation)

This document walks through, function by function, the two line-segmentation modules in `Software/CNN/`:

1. **`SegmentLean.py`** — the production segmenter. Documented in full depth.
2. **`SegmentPage.py`** — the older, more elaborate segmenter. No longer the main server/training path, but still reachable through one caller. Documented per-function but more concisely, with the still-live call path flagged explicitly.

Both modules sit in **Block 1 (Image processing & Line Segmentation)** of the 4-block pipeline architecture: raw photographed page in, ordered line-crop images out, with nothing downstream (text recognition, author classification, style synthesis) touching a pixel of the original photo again.

---

## Part 1 — `SegmentLean.py` (production segmenter)

### Role in the pipeline

`SegmentLean.SegmentLines(path)` is the single entry point called by the server for every live segmentation request, and by essentially all current training code (`TrainText.py`, `TrainAuthor.py`, and — via `BuildStyleProfile.py` in `ReproductionImprovement/` — the author-style pipeline). It takes a file path to a photographed page and returns `(lines, overlay, meta)`:

- `lines`: an ordered list of dicts, one per detected handwritten line, each holding a native-resolution crop (`image`), a fixed-size classifier-ready crop (`clf`, 640×64), the crop's bounding box in original-image pixel coordinates, and its line order.
- `overlay`: an RGB debug image showing the ink/paper ratio with the paper-boundary and all internal line-cut curves drawn on top.
- `meta`: a dict of diagnostic scalars (skew angle, line pitch, paper coverage, counts of removed pixels, timing).

Downstream, `lines[i]["clf"]` feeds directly into the CNN text/author classifiers (`TrainText.py`/`TrainAuthor.py`-trained models expect exactly 64×640 grayscale), and `lines[i]["image"]` is what a human or a later OCR/handwriting-synthesis stage sees.

The module is a **self-contained, from-scratch numpy/PIL** image pipeline — no OpenCV, no scipy. Every low-level operation (box blur, dilation, connected components, DP shortest path) is reimplemented with vectorised numpy primitives for speed, which is itself worth noting in a literature/implementation discussion: the whole file is built from three families of classical algorithms — (1) **integral-image / summed-area-table** convolution tricks, (2) **union-find connected-component labelling**, and (3) **dynamic-programming shortest-path ("seam") search** — combined with **Otsu thresholding** and a long tail of hand-tuned heuristic rules for real-world photographed-page artifacts (shadows, binder clips, ruled paper, i-dots).

The six pipeline stages (from the module docstring) are:
1. Normalise lighting (divide by a blurred local background), Otsu-threshold ink, deskew; keep only ink actually on the paper.
2. Discard long, thin, perfectly straight strokes (underlines, ruled lines, binder edge) and page-boundary debris.
3. Estimate line pitch `P` from the row-ink profile; line centres = its peaks.
4. Between neighbouring centres, find the least-ink path via dynamic programming: a rough boundary.
5. Assign each connected ink fragment to a line, with special-case fixes for cut-off tails, stray marks, and i-dots.
6. Render each line's own ink into a clean, contrast-enhanced crop; white out everything else.

---

### `_toClassifierFormat(img, height=CLF_HEIGHT, width=CLF_WIDTH, pad_value=255)`

**Purpose.** Produces the fixed-size (64×640) grayscale image that `TrainText.py`/`TrainAuthor.py` were trained to expect, from a line crop of arbitrary aspect ratio, without ever distorting letter shapes.

**Inputs.**
- `img`: a PIL `"L"` (8-bit grayscale) image, the native-resolution crop of one handwritten line (ink dark, background near-white).
- `height, width`: target canvas size, defaulting to the module constants `CLF_HEIGHT=64, CLF_WIDTH=640`.
- `pad_value`: the grey level used to fill unused canvas (255 = white/paper).

**Outputs.** A PIL `"L"` image of exactly `(width, height)` pixels: the input resized by one uniform scale factor to fit inside the canvas, then padded with `pad_value`.

**Inner workings.** `s = min(height/h, width/w)` — a single isotropic scale (never stretches x and y differently, preserving letter proportions). New size `= round(w*s), round(h*s)`. The resized image is pasted onto a blank `pad_value` canvas at `x=0` and vertically centred: `y = (height - new_h)//2`. This is a standard "letterbox" resize, identical in spirit to how YOLO-style object detectors prepare inputs of mismatched aspect ratio for a fixed-size network input.

**Literature connection.** Aspect-ratio-preserving letterbox resizing is a standard CNN input-normalisation technique; no deeper named algorithm applies here.

---

### `_box(a, r)`

**Purpose.** Computes a fast local mean (box blur) over a `(2r+1)×(2r+1)` window centred at every pixel — the core primitive used throughout the file for local-background estimation, dilation/erosion-like operations, and smoothing.

**Inputs.** `a`: 2D numpy array (float or castable), any real-valued image. `r`: integer radius of the square window.

**Outputs.** A 2D float64 array, same shape as `a`, where each pixel holds the mean of the `(2r+1)×(2r+1)` neighbourhood around it in the *edge-padded* input.

**Inner workings and maths.** This is the classical **summed-area table** (integral image) trick (Crow, 1984, used extensively in Viola–Jones-style vision). `a` is first edge-padded by `r` on all sides. A 2D prefix sum (cumulative sum along both axes, with an extra leading zero row/column) is built: `ii[y,x] = Σ_{y'<y, x'<x} p[y',x']`. The sum over any axis-aligned rectangle `[y1,y2)×[x1,x2)` is then obtained in O(1) via the inclusion–exclusion identity
`S = ii[y2,x2] - ii[y1,x2] - ii[y2,x1] + ii[y1,x1]`,
and dividing by the window area `k² = (2r+1)²` gives the mean. The function evaluates this formula for every pixel's window simultaneously via array slicing, so the whole box-blur is O(HW) regardless of `r`.

**Literature connection.** Summed-area tables (Crow, 1984); the same O(1)-per-pixel box-filter trick underlies Viola & Jones (2001) feature computation and is a textbook image-processing primitive (Gonzalez & Woods).

---

### `_any(m, r, axis)`

**Purpose.** A fast 1D binary "is there any set pixel within `r` positions along `axis`" test — functionally a **binary dilation** by a line-segment structuring element of length `2r+1` along one axis, used repeatedly to grow/dilate masks cheaply.

**Inputs.** `m`: boolean (or 0/1) array of any dimensionality. `r`: radius (half-width) of the 1D neighbourhood. `axis`: which axis to dilate along.

**Outputs.** Boolean array, same shape as `m`: `True` at position `i` along `axis` if any of the `2r+1` positions `[i-r, i+r]` (clipped/zero-padded at the ends) is `True` in `m`.

**Inner workings and maths.** The target axis is moved to position 0. A prefix sum `c` of the (zero-padded by `r` on both ends) array along that axis is taken. Then for each output index `n` in `[0, L)`, the windowed sum is `c[n + 2r+1] - c[n]` (using the padded indexing offset), and the result is thresholded `>0`. This is exactly the summed-area-table trick in 1D: a windowed OR-over-a-binary-array computed as "windowed sum greater than zero."

**Literature connection.** Equivalent to binary morphological dilation with a 1D line structuring element (Serra, 1982, mathematical morphology), implemented via a prefix-sum shortcut instead of an explicit max-filter.

---

### `_long_runs(m, n, axis)`

**Purpose.** Flags every pixel that lies on a run of **at least `n` consecutive `True`** values along `axis` — used to detect long straight strokes (underlines, ruled lines, binder edges) that are far longer than any genuine pen stroke.

**Inputs.** `m`: boolean array. `n`: minimum run length to qualify (an integer length in pixels). `axis`: direction to scan runs along.

**Outputs.** Boolean array, same shape as `m`, `True` exactly on pixels belonging to *some* run of ≥`n` consecutive `True` values.

**Inner workings and maths.** Internally this performs a **binary opening** with a line structuring element of length `n`, computed via prefix sums rather than an explicit erode+dilate pass:
1. A prefix sum `c` of `m` (axis moved to front, zero-padded by one at the start) is built.
2. `hit[i] = 1` iff `c[i+n] - c[i] == n`, i.e. the window `[i, i+n)` is *entirely* `True` — this is exactly a 1D **erosion** by a length-`n` structuring element, anchored at the window's start.
3. A second prefix sum `ch` of `hit` is taken, and for each position `j`, the valid window-start range that could cover `j` is `lo = max(0, j-n+1)`, `hi = min(j, L-n)`; position `j` is marked covered if `ch[hi+1] - ch[lo] > 0` (i.e. at least one qualifying run-start exists in range) — this is the **dilation** of the eroded hit-map back out by the same footprint, reconstituting the full extent of every qualifying run.

Together, steps 2–3 implement **binary opening** (erode-then-dilate) with a 1D structuring element of length `n`, which is exactly the standard way to isolate "runs at least this long" while discarding shorter ones.

**Literature connection.** Morphological opening (Serra, 1982) with a line structuring element — the standard tool for detecting/removing long thin linear structures (used here for straight-line artifact removal, analogous to its use in document de-ruling literature, e.g. removing form/table lines before OCR).

---

### `_otsu(v)`

**Purpose.** Finds the grey-level (or here, ink/background ratio-level) threshold that best separates two populations (ink vs. paper) in a 1D distribution, by maximising between-class variance. Used both for the main ink/paper split and for several secondary thresholding decisions (darkness-based artifact filtering, per-line local re-thresholding).

**Inputs.** `v`: any real-valued numpy array (here, a "ink/background ratio" image, values expected roughly in `[0, 1.2]`).

**Outputs.** A single float: the threshold value, in the same units as `v`, at the centre of the best-scoring histogram bin.

**Inner workings and maths.** A 256-bin histogram `h` of `v` clipped to `[0, 1.2]` is built, normalised to a probability mass function `p = h / Σh`. Define the cumulative weight `w(t) = Σ_{i≤t} p(i)` (the background-class fraction below bin `t`) and cumulative first moment `m(t) = Σ_{i≤t} i·p(i)`. Writing `μ_T = m(255)` (the global mean bin index), Otsu's between-class variance at threshold `t` simplifies algebraically to:

```
σ_B²(t) = (μ_T·w(t) − m(t))² / [ w(t)·(1 − w(t)) ]
```

which is exactly the quantity the code computes as `s = (m[-1]*w - m)**2 / max(w*(1-w), 1e-12)`. The optimal threshold is `t* = argmax_t σ_B²(t)`, and the returned value converts the winning bin index back to the original units: `(t* + 0.5) * 1.2 / 256` (bin centre, rescaled from the 256-bin, `[0,1.2]`-range histogram back to real ratio units).

**Literature connection.** This is **Otsu's method** (Otsu, 1979, *"A threshold selection method from gray-level histograms"*, IEEE Trans. SMC) in its standard closed-form, cumulative-moments implementation — the most widely used automatic global-thresholding algorithm in document/image binarisation.

---

### `_ink(gray)`

**Purpose.** Converts a raw grayscale photo into a clean binary ink mask, correcting for uneven lighting (vignetting, shadows, desk lamps) and excluding large solid dark regions (binder clips, deep shadow pools) that are not pen strokes.

**Inputs.** `gray`: 2D float64 array, grayscale pixel values (0–255-ish), the photographed page (or a downsampled copy of it).

**Outputs.** A tuple `(mask, ratio)`:
- `mask`: boolean array, same shape, `True` = ink.
- `ratio`: float array, same shape — the illumination-normalised "darkness ratio" used both here and reused later (for Otsu-based artifact scoring and for the debug overlay).

**Inner workings and maths.** First, a **local background** is estimated via `_box(gray, r)` with radius `r = clip(round(H/41), 8, 40)` — i.e. the blur radius scales with image resolution (so a small crop isn't swallowed by an oversized blur). The illumination-normalised ratio is

```
ratio(y,x) = gray(y,x) / max(background(y,x), 1)
```

— pixels much darker than their local surroundings (ink) have `ratio ≪ 1`; paper has `ratio ≈ 1`. This is a classic **flat-field / shading correction** (dividing out a low-frequency illumination estimate), equivalent in spirit to adaptive thresholding methods (e.g. Niblack/Sauvola-style local normalisation) but using a *multiplicative* ratio rather than a local mean/standard-deviation band.

`mask = ratio < Otsu(ratio)` applies Otsu's method (see `_otsu` above) directly to the ratio image, picking a single global cutoff on the normalised distribution — effectively a "locally-adaptive, globally-thresholded" binarisation.

A second pass removes large **solid dark blobs**: `solid = _box(mask, 4) > 0.999` marks pixels whose entire 9×9 neighbourhood is ink (i.e. not individual pen strokes, which always have paper gaps between letters, but a uniformly dark region like a binder clip's shadow or interior). The final mask drops not just those solid pixels but a 7×7 ring around them too (`_box(solid, 3) > 0`, i.e. dilated by 3), so the solid region's blurred edge doesn't survive as a ragged halo.

**Literature connection.** Shading/illumination correction via division by a blurred background is a standard document-image preprocessing step (closely related to "flat-fielding" in photography and to local/adaptive binarisation families like Sauvola, 2000); the thresholding step itself is Otsu's method (Otsu, 1979).

---

### `_pitch(mask)`

**Purpose.** Estimates `P`, the line spacing (vertical distance between consecutive handwritten lines) — the single most important scale parameter in the whole pipeline, since almost every later geometric threshold ("long enough to be an underline," "close enough to belong to this line," etc.) is expressed as a fraction/multiple of `P`.

**Inputs.** `mask`: boolean ink mask, 2D array, shape `(H, W)`.

**Outputs.** Tuple `(P, sm)`:
- `P`: integer, estimated line pitch in pixels.
- `sm`: the smoothed row-ink profile (float array of length `H`), reused by the caller to find line centres (peaks) directly without recomputation.

**Inner workings and maths.** A **row-ink profile** is built by summing ink pixels per row over the central 92% of the page width (trimming 5%/3% margins to avoid edge artifacts): `prof(y) = Σ_{x∈[0.05W,0.97W]} mask(y,x)`. This profile is smoothed with a **Gaussian kernel** of σ=4 (support `±12`, i.e. a 25-tap kernel `k(i) = exp(-0.5(i/4)²)`, normalised), via `np.convolve(..., "same")`.

Three fallback strategies are tried, in order of preference:
1. **Direct peak-to-peak measurement.** Local maxima of `sm` (points where `sm[i]≥sm[i-1]` and `sm[i]>sm[i+1]`, with a noise floor `sm[i] > 0.06·max(sm)`) are found; if at least two exist, `P = median(diff(sorted(peak positions)))`. This directly measures spacing, so it correctly handles a page with *irregular* spacing (e.g. mixed single- and double-spaced paragraphs) — the chosen value is the median gap, not an assumed uniform grid.
2. **Autocorrelation.** If fewer than two peaks are resolvable (lines packed so tight the profile between them never dips), the **autocorrelation function** of the mean-centred profile is computed: `ac(τ) = Σ_y a(y)·a(y+τ)` via `np.correlate(a, a, "full")`, keeping only non-negative lags. The function looks for the first local maximum in the lag range `[15, min(200, L/3)]` whose value is within 70% of the segment's own maximum — i.e. the first strong periodic peak, which corresponds to the fundamental line-spacing period. Autocorrelation at lag 15+ is used only as a fallback because on a page with uneven paragraph gaps it can lock onto a harmonic (an integer multiple of the true spacing) rather than the true spacing itself, which direct peak measurement avoids.
3. **Letter-size heuristic.** If there's effectively only one line (nothing to compare periodicity against), connected components of `mask` are measured (via `_label`, below) and a pitch is extrapolated from the median height of "letter-sized" components (`P ≈ 2.2 × median(height)`), an empirically calibrated ratio between typical line pitch and typical x-height-to-ascender letter height.
4. **Hard default.** `P = 48` if there's no usable ink at all (an arbitrary but harmless fallback that avoids a crash).

**Literature connection.** Row-ink (horizontal) **projection profiles** are the classical technique for text-line segmentation in document image analysis (going back to Ohta et al. 1988 / Wong, Casey & Wahl 1982's X-Y cut and profile-based layout analysis); the autocorrelation fallback for estimating a fundamental spatial period is a standard signal-processing technique (Wiener–Khinchin-style periodicity detection).

---

### `_seam(cost, step=3)`

**Purpose.** Finds the minimum-total-cost path from the left edge of a cost map to the right edge, allowing the path's row index to change by up to `step` per column — this is the core geometric primitive used to carve a rough boundary between two adjacent handwritten lines that threads through blank paper (low cost) and avoids ink (high cost).

**Inputs.** `cost`: 2D float array, shape `(h, W)` — a per-pixel cost (built by the caller so that ink pixels are expensive and paper is cheap, plus a mild positional regularisation term). `step`: maximum allowed vertical displacement (in rows) between consecutive columns.

**Outputs.** A 1D integer array `y` of length `W`: for every column `x`, the row index of the minimum-cost path.

**Inner workings and maths.** This is a textbook **dynamic-programming shortest path**, structurally identical to Avidan & Shamir's seam carving (2007), but run horizontally (left→right) instead of vertically, and with a *bounded* step set rather than the usual `{-1,0,+1}`. The recurrence is:

```
acc(y, x) = cost(y, x) + min_{d ∈ [-step, step]} [ acc(y+d, x−1) + 0.02·|d| ]
```

i.e. the cheapest way to reach pixel `(y,x)` is the cheapest way to reach any of the `2·step+1` candidate predecessors in the previous column, plus that column's own cost, plus a small linear "steering penalty" `0.02·|d|` proportional to how far the path jumps — this discourages (but does not forbid) sharp vertical moves, so the path still snaps around tall ascenders/descenders when ink forces it to, without being needlessly jagged in blank regions. `back(y,x)` records which `d` was chosen (as an `int8`, offset to the 0-based shift index) for backtracking. The path is initialised at the column `W-1` by taking the global minimum of `acc[:, -1]`, then traced backwards: `y[x-1] = y[x] + back[y[x], x]`.

**Literature connection.** This is the **seam-carving** dynamic-programming shortest-path algorithm (Avidan & Shamir, 2007, *"Seam carving for content-aware image resizing"*), which is itself a direct descendant of classical DP shortest-path/edit-distance recurrences; using a minimum-cost seam to separate adjacent text lines through ink-sparse "valleys" is a documented line-segmentation technique in the handwritten-document-analysis literature (e.g. Likforman-Sulem et al.'s surveys on text line extraction use exactly this "valley path" / seam idea for touching/overlapping lines).

---

### `_label(mask)`

**Purpose.** Labels 8-connected groups of ink pixels (connected-component analysis) — the fundamental grouping primitive used throughout the file to treat "a blob of ink" as a single object with measurable area/bounding box, rather than a cloud of independent pixels.

**Inputs.** `mask`: 2D boolean array.

**Outputs.** `(labels, count)`: `labels` is an `int32` array, same shape, `0` = background and `1..count` identify each connected component; `count` is the number of components found.

**Inner workings and maths.** This is a **run-length-encoded, two-pass union-find connected-component labelling** algorithm:
1. **Run extraction (per row).** For each row `y`, the boundaries of contiguous ink runs are found via a first difference of the (zero-padded) row (`diff` of `[0, mask[y], 0]`): `+1` marks a run start, `-1` marks a run end.
2. **Union across rows.** Each run on row `y` is compared against the runs of row `y-1` (`prev`), and unioned via a union-find `parent[]` array whenever their column ranges overlap (or touch) — this is the standard "merge the labels of touching runs in the row above" step of run-based connected-component labelling.
3. **Path-compressed find.** `find(a)` walks up the `parent` chain with path halving (`parent[a] = parent[parent[a]]`) for near-O(1) amortised lookups — the classical **union-find / disjoint-set** data structure (with path compression, no explicit union-by-rank, but the compression alone keeps it efficient at this scale).
4. **Label compaction.** After all rows are processed, every run's root is resolved via `find`, and `np.unique` on the roots produces compact, contiguous output labels `1..count`.

**Literature connection.** This is a vectorised variant of the classical **two-pass connected-component labelling algorithm** (Rosenfeld & Pfaltz, 1966) using **union-find / disjoint-set forests** (Tarjan, 1975) for the equivalence-class resolution, specialised to row-run representation for speed — the same family of algorithm underlies `scipy.ndimage.label`/`cv2.connectedComponents`.

---

### `_paper(g, P)`

**Purpose.** Finds the region of the photo that is actually the paper, so that anything outside it (desk, clamp, binder, page edge, cast shadow) is excluded before ink detection runs — otherwise a dark tabletop or clamp could be mistaken for a huge block of "ink."

**Inputs.** `g`: 2D float array, grayscale page image. `P`: estimated line pitch (used to scale the morphological radii).

**Outputs.** Boolean array, same shape as `g`: `True` = paper.

**Inner workings and maths.** The image is block-averaged by a factor `s=4` (reshape-and-mean downsampling) for speed, then smoothed with `_box` at a radius proportional to the line pitch (`~0.15·P/s`). A pixel is flagged `dark` if it is less than 45% of the image's **median** brightness: `dark = c < 0.45·median(c)`. If almost nothing is "dark" (`<0.2%` of pixels), the function assumes there's no surrounding clutter and returns an all-`True` mask (a photo that's already cropped tight to the page). Otherwise, the **largest connected bright (non-dark) region** is taken as the paper (`_label` on `~dark`, picking the component with the largest pixel count via `np.bincount`). That low-resolution blob is upsampled back to full resolution with nearest-neighbour block expansion (`np.kron`). Finally, the paper boundary is pulled inward by eroding it with radius `r=0.4·P` (implemented as `~_any(_any(~full, r, 0), r, 1)`, i.e. dilating the *inverse* mask and inverting — mathematically equivalent to eroding the paper mask), which trims a margin a bit less than half a line-spacing wide, keeping shadowed/ambiguous edge pixels out of the detected text region.

**Literature connection.** A brightness threshold followed by largest-connected-component selection is a standard "foreground region of interest" extraction technique in document scanning/photo-rectification pipelines; the final erosion is classical binary morphology (Serra, 1982).

---

### `SegmentLines(path)` — the main pipeline

This function is long and does the real work of all six pipeline stages described in the module docstring. It is broken down stage-by-stage below; line numbers refer to the source file.

#### Inputs / outputs
- **Input:** `path`, a filesystem path (string or `Path`) to a photographed page image (any format PIL can open).
- **Output:** `(lines, ov, info)` as described in "Role in the pipeline" above.

#### Setup and deskew (stage 1)

The image is loaded as 8-bit grayscale (`full`) and a half-resolution copy (`half`, via PIL's `BOX` resampling — an area-averaging downsample) is made for cheap geometric analysis. `_ink` computes an ink mask on the half-res image; that mask is resized down by another factor of 2 (quarter-res) and, for candidate rotation angles `a ∈ {-4, -3.5, ..., +4}` degrees, the image is rotated and scored by

```
score(a) = Σ_y [ rowInkCount_a(y) ]²
```

i.e. the sum of squared per-row ink counts of the rotated mask — maximised exactly when text rows are horizontal and their ink is concentrated into sharp, high-contrast horizontal bands (any rotation that tilts rows "smears" ink across more rows, lowering each row's count and hence the sum of squares). The angle `best` that maximises this score is applied (bilinear rotation, white fill) to both `full` and `half`, and `_ink` is recomputed on the now-deskewed `half` image to get the working `mask`/`ratio`.

**Literature connection.** Maximising the energy (sum of squared profile values) of a horizontal projection profile over a small angle search is a classical **projection-profile skew-estimation method** (Postl, 1986; Baird, 1987) — one of the oldest and most widely used document-deskewing techniques.

#### Straight-line and boundary artifact removal (stage 2)

`P` (pitch) and the `paper` mask are computed; ink outside `paper` is dropped (`paperRemoved` logged as a diagnostic). Then:

```
flat    = pixels on a (row-tolerant) horizontal run ≥ 1.25·P long
upright = pixels on a (column-tolerant) vertical run ≥ 1.4·P long
```

built with `_any(mask, 1, 0)` / `_any(mask, 3, 1)` pre-dilation (to tolerate a slight residual tilt or wobble) followed by `_long_runs` (morphological opening, see above) and a matching post-dilation to restore the flagged region's full extent. These catch **underlines, ruled-paper lines, and the binder/page edge** — structures far straighter and longer than any real pen stroke can sustain. Both are subtracted from `mask`.

**Stage 2b** then does component-level artifact filtering. A *closed* version of the mask (`closed`: dilate then erode by `rc≈0.1P`, i.e. a **morphological closing** that bridges small pen-lift gaps within one stroke/word without materially growing its outline) is labelled with `_label`. For each resulting component, a battery of geometric/photometric features is computed from vectorised `bincount`/`minimum.at`/`maximum.at` reductions — area, bounding box, mean ink-ratio darkness — and a set of hand-tuned, pitch-relative heuristic rules (documented inline in the source, lines 251–274) flags components as artifacts: too tall and too thin (page-edge/binder strips), a narrow vertical bar taller than any letter (`vbar`), a faint thin-tall sliver that only survived straight-line removal because it has small gaps (`faintVbar`, a printed margin rule), a short/sparse/faint band right at the page edge (`edgeShadow`, a crease or shadow), anything touching the page's left border, or anything stranded far to the left of the measured left text margin (`xL`, the 5th percentile of left edges among "letter-sized" components). All flagged components are removed from `mask`.

This stage is best understood as a small **rule-based classifier** over per-component shape/darkness features — not a named classical algorithm, but structurally the same idea as heuristic document-layout-analysis rules (c.f. Nagy's surveys of layout analysis), expressed relative to the measured pitch `P` so it self-calibrates to each page's handwriting scale.

#### Line centres (stage 3)

Pitch and the smoothed row profile are recomputed on the cleaned mask (`_pitch`). Local maxima of the profile above a noise floor are candidate centres; they are then **greedily non-max-suppressed**: sorted by descending peak strength, and a peak is accepted only if it is at least `0.6·P` away from every already-accepted centre. This is a standard 1D peak-picking technique (closely related to the non-maximum suppression used in blob/keypoint/object detectors) that prevents two adjacent noisy sub-peaks within one real line from being counted as two separate lines.

#### Rough boundaries by DP seam (stage 4)

The outermost boundaries are flat lines offset `0.75·P` above the first centre and below the last. Between each pair of adjacent centres, a **cost map** is built from a **closed** version of the mask:

```
mdil    = dilate(mask, tailR)         where tailR = max(2, 0.07·P)
mClosed = erode(mdil, tailR)          (= mask closed by radius tailR)
```

This closing (not a plain dilation) is the fix for a documented bug: narrow internal necks in handwriting — e.g. beside a descender's stem, or the narrow neck of a "y" tail, or the gap between the strokes of letters like "x"/"p" in a word such as **"express"** — can otherwise leave a thin channel of pure background (cost-0) running right past or through a letter. A plain dilation would widen the *whole* silhouette (hiding real ink boundaries under a bloated blob), while closing only fills gaps *smaller than* `2·tailR` and leaves the stroke's true outer boundary essentially unchanged. Before this fix, the DP seam (which simply minimises total ink crossed) could find that zero-cost channel and thread straight through the middle of a word instead of around it; after closing, that channel is filled with "ink" cost, forcing the seam out and around the letter's real silhouette. **This is the "closing-based ink-cost map" fix referenced in the project's known-issues notes.**

The actual per-column cost fed to `_seam` is

```
cost(y,x) = 4.0·mClosed(y,x) + 0.012·((y − mid)/half_w)²,   mid = c0 + (c1−c0)/2,  half_w = max(1, (c1−c0)/2)
```

i.e. ink cost dominates, plus a weak quadratic penalty that pulls the seam back toward the midline between the two centres when ink cost is tied (so ambiguous blank regions get a sensible default boundary rather than an arbitrary jagged one). `_seam` is invoked with `step=7` — a larger-than-default per-column jump allowance — specifically so the seam can follow a tall, narrow ascender/descender tail quickly enough to hug it rather than cutting through it.

#### Piece ownership (stage 5)

This is the densest part of the file; it resolves, for every ink pixel, which of the `L` lines "owns" it, starting from the rough DP boundaries and then correcting several predictable failure modes.

- `region(y,x)` = count of inner boundary curves (`cuts[1..L-1]`) at or above `(y,x)` — i.e. the pixel's *rough* line index directly from the DP seams.
- Per closed-component (`pl`, from stage 2b's `lab`) statistics are built: centroid `(cx, cy)`, and a histogram `cnt[piece, region]` of how many of a piece's pixels (within the overall ink band) fall in each rough region.
- **`substantial`**: a region holds ≥15% of a piece's total pixel count. **`major`** = the *topmost* substantial region if the piece substantially spans more than one rough region (reasoning: when a letter's tail or a word's rough cut genuinely straddles two lines, it is the line *above* that loses the piece's crop content if it isn't reserved for it — not whichever side happens to have more raw pixels), else the region with the most raw pixels.
- **Case "minor"** (cut-off tails): a small (`<0.12P²`), narrow (`≤0.55P` wide — a tail, not a whole word) sliver of a piece sitting in a region adjacent to its `major` region is folded wholesale into `major`.
- **Case B** (stray tiny pieces, lines 345–357): very small components are either dropped (if far from any of their nominal line's substantial ink, or if "speck"-sized and not even near a close substantial component) or shifted to the next/previous line if their centroid sits clearly (`>0.5P`/`>0.7P`) above/below the nearby ink of their current line.
- **Case D** (medium pieces — brackets, quote marks, isolated tails, lines 358–364): assigned to whichever of the three candidate lines (`major-1, major, major+1`) has a substantial component whose centroid `y` is nearest.
- **Case C / "dot"** (lines 365–386): closed pieces small and roughly square are classed `dot` (candidate i-dots/full-stops); larger pieces are `stem`. For each `dot`, the code scans directly below it (within `0.4P`) for a `stem` belonging to some line `n2`; if the vertical offset between the dot and that line's ink sits in the empirically-typical i-dot range `[0.25P, 0.7P]` above the line, the dot is reassigned to `n2`.
- `own[y,x]` is finally built: `major[piece]` for any piece flagged tiny/minor, otherwise the raw `region(y,x)` value — i.e. ownership defaults to the geometric DP split except where a small, clearly-misattributed fragment has been explicitly reassigned.

**Case E** (lines 390–418) and **Case F** (lines 420–467) correct a side effect specifically introduced by the *closing* used to build the stage-4 cost map (and stage-2b's artifact-filtering closing): closing can physically weld a genuine floating i-dot onto the descender of the line above it, or weld it into a ring-like blob at the bottom of a stroke. Both cases re-examine the **raw** (un-closed) components:
- **E** looks for a small raw `dot`-shaped component sitting just above a raw `stem` component; if the stem's line differs from where `own` currently (post-DP) puts the dot, and the apparent "whole-word" area test rules out the dot being an artifact of closing a real word shut, the dot's pixels are moved onto the stem's line.
- **F** looks for a *ring*-shaped blob at the bottom of a stroke (walking up column-wise from the bottom of a component while it stays wide, stopping where it narrows sharply to roughly half its widest span — the "neck") and verifies the blob is genuinely enclosed (its background, after padding, has ≥2 connected background regions — i.e. a literal hole — rather than 1, which would mean an open hook/tail end, not a ring) before reassigning it the same way as E.

**Case H** (lines 469–503) is a final geometric sanity pass: for every line, pixels currently `own`ed by it are re-labelled into connected fragments, and any fragment with no ink at all inside that line's expected x-height band (`|y − centre| ≤ 0.28P`) — i.e. a fragment floating entirely in the gap between lines — is examined. Dot-sized fragments explicitly placed by E/F are left alone (the `locked` flag tracks which pixels E/F touched). Otherwise, the fragment's immediate neighbourhood (a small dilation) is checked for which *other* line's ink dominates nearby — if a clear majority neighbour exists and is adjacent and geometrically consistent, the fragment moves there; failing that, it falls back to whichever adjacent line's centre the fragment's mean `y` is closest to, if within `0.2P`.

#### Crop rendering (stage 6)

Each line's final crop is rendered from the **full-resolution** image, not the half-res working copy, for maximum print quality:
- A full-resolution lighting-normalisation background is built by upsampling the half-res `_box` blur back to full size (bilinear), and `norm = clip(full/bg, 0, 1)·255` reproduces the same flat-field correction idea as `_ink`, but kept as continuous grayscale (not thresholded) so pen-pressure shading is preserved.
- Lines with too little ink (`< max(60, 0.12·median(per-line counts))`) or too little vertical extent are dropped as noise.
- **Paragraph-gap trimming:** if a line's row-ink profile has a large enough blank gap (bigger than the vertical extent minus `0.8P`) splitting it into separate runs, only the largest contiguous ink run (by total ink area) is kept for the crop — this prevents a stray mark from a different paragraph (picked up by the generous DP band) from dragging the crop's bounding box out.
- The crop box is padded by `0.15P` and mapped from half-res to full-res coordinates by the scale factor `f = full.width / W`.
- **Alpha mask construction:** the binary line-ownership mask is upsampled (bilinear) to full resolution as a base alpha; a **halo** (the ink mask dilated by `0.2P`) is also upsampled, and a **local Otsu re-threshold** is computed *within this crop only* (not the whole-page Otsu) to recover full-resolution detail the half-res global threshold may have missed. The union `alpha = max(upsampledMask, localMask ∧ haloMask ∧ notOwnedByAnotherLine)` is the final per-pixel opacity — gated so recovery can never "steal" a neighbouring line's confirmed ink.
- **Contrast enhancement:** the typical ink darkness (`black`, 40th percentile of region values under `alpha>0.5`) and typical paper brightness (`white`, 75th percentile over the whole crop) define a **linear contrast stretch**:

```
linear(x) = clip( (region(x) − black) · 255/(white − black), 0, 255 )
```

followed by a **gamma correction** (γ=0.55 < 1) applied to the normalised darkness:

```
darkness(x) = ( (255 − linear(x)) / 255 )^0.55
enhanced(x) = 255·(1 − darkness(x))
```

Because γ<1 expands low-darkness (faint, lightly-pressed) values disproportionately while leaving already-saturated ink near-black, this pulls faint strokes much darker while leaving solid ink unchanged — a perceptual non-linear remap standard in image processing (the same mathematical form as monitor/print gamma correction, just applied to "inkiness" instead of display luminance).
- **Compositing:** `crop = alpha·enhanced + (1−alpha)·255` is the standard alpha-blend ("over") equation (Porter & Duff, 1984) compositing the enhanced ink over a pure-white background.
- **Left-margin normalisation:** the crop is shifted horizontally so the first column with meaningful alpha lines up with the same left margin every crop starts with, regardless of exactly where the ink happened to sit inside its padded bounding box — keeping training/visual outputs visually consistent.
- The final `(order, image, clf, bbox)` dict is appended to `lines`.

Finally, a debug overlay (`ov`) is built from the illumination ratio image (tinted red outside the detected paper region) with every DP cut curve drawn as an alternating red/blue polyline, and `info` collects the run's diagnostic scalars.

---

## Part 2 — `SegmentPage.py` (legacy segmenter)

### Role in the pipeline and current status

`SegmentPage.py` was the **original** first-principles page segmenter — more elaborate than `SegmentLean.py`, structured around MESS-vs-TEXT component classification and a chain-based line-grouping algorithm, with ten additional "override" stages layered on top of a baseline algorithm to cope with real camera-photo artifacts (shadow, ruled paper, facing-page bleed, bleed-through, coloured ink). Its entry point is `ProcessPage(imgPath) -> (results, preview, meta)`.

**It is no longer the production path.** `SegmentLean.py` (Part 1) replaced it for the server and for all current `TrainText.py`/`TrainAuthor.py` training. The **one place it is still reachable** is `Software/CNN/AuthorReproductionStuff/BuildStyleProfile.py`'s `build_personal_line_items(author)`, which calls `F.ProcessPage(str(img_path))` (where `F` is `SegmentPage`, imported at the top of that file) to segment each personal author's photographed pages before forced-aligning them for style-profile extraction. (Note: the parallel file `Software/CNN/ReproductionImprovement/BuildStyleProfile.py` has since been updated to call `SegmentLean.SegmentLines` instead for the same purpose — only the `AuthorReproductionStuff` copy still depends on `SegmentPage`.) Because `ProcessPage` is the one live call, and its body exercises essentially every function in the file's "baseline primitives" and "override stages" sections (per the module's own docstring, every stage listed was kept specifically because it was shown to still change real output), the *entire file* is, in practice, on that one live code path — nothing below is dead code, it is simply no longer the main pipeline.

Given the brief for this file ("more briefly... but do not skip the still-live parts"), each function below gets a compact purpose/mechanism entry; the overall architecture is explained once, up front, rather than re-derived per function.

### Architectural overview

`SegmentPage.py` is organised in three sections (per its own header comments):
1. **Low-level numpy image primitives** (`RawImageOps`-equivalent): box blur via integral images, Gaussian blur via iterated box blur, binary morphology via box sums, run-based union-find connected components, hole-filling, and inverse-mapped bilinear rotation/resize.
2. **Baseline segmentation primitives** (`SegmentPageCore`-equivalent): load/normalise, Otsu/adaptive thresholding, deskew-by-profile-variance, rule-line removal by chaining colinear components, a MESS-vs-TEXT component classifier based on enclosed-hole area, and `GroupLines`, a chain-then-merge line-grouping algorithm.
3. **Override stages** that `SegmentPage.py` itself adds on top: a more careful page-mask detector, facing-page-bleed removal, sparse-rule-network stripping, a post-grouping refinement pass (`RefineItems`), weak-ink hysteresis recovery for faint/coloured ink, and ownership-aware rendering that forbids one line's crop from stealing another's pixels.

`ProcessPage` wires all of this together; its stage order is documented inline in its own write-up below.

---

### Section A — low-level image primitives

#### `_Integral(img)`
One-line helper: builds a padded 2D prefix-sum ("summed-area table") of `img`, with an extra zero row/column so rectangle sums can be read off directly by index arithmetic. Same mathematical idea as `_box`'s integral image in `SegmentLean.py` (Crow, 1984).

#### `BoxSum(img, ry, rx)` / `BoxCount(ry, rx, h, w)` / `BoxMean(img, ry, rx)`
`BoxSum` returns the sum over a `(2ry+1)×(2rx+1)` window per pixel, using the integral image and the same inclusion–exclusion rectangle-sum formula as `_box` in Part 1, but with edge-*clamping* (the window truncates at the border) rather than edge-padding. `BoxCount` returns how many real pixels contributed to that window at each position (needed because of the clamping). `BoxMean = BoxSum / BoxCount` gives the true local mean even near borders. Same literature connection as `_box` (summed-area tables).

#### `GaussianBlur(img, sigma)` / `GaussianBlur1D(arr, sigma)`
Approximates a Gaussian blur of standard deviation `sigma` by applying a box blur **three times in succession**. The box radius is chosen so the variance after 3 passes matches `sigma²`: for a box of full width `W`, a single pass has variance `(W²−1)/12`; three passes combine (by independence) to variance `3·(W²−1)/12 = (W²−1)/4`, so solving `(W²−1)/4 = σ²` gives `W ≈ √(4σ²+1)`, which is exactly the radius formula in the code (`r = round(√(4σ²/3 + 1)/2)` — matching the `GaussianBlur1D` comment's derivation precisely). This is justified by the **Central Limit Theorem**: repeated convolution of a box (uniform) kernel converges rapidly to a Gaussian shape. `GaussianBlur1D` is the identical idea along a single axis, via `np.convolve`.

**Literature connection.** Iterated box blur as a fast Gaussian approximation is a well-known image-processing trick (e.g. used in real-time graphics; theoretically grounded in the CLT, as the code's own comment states).

#### `RgbToGray(rgb)`
Standard **luma** conversion: `Y = 0.299R + 0.587G + 0.114B` (the ITU-R BT.601 luma coefficients), clipped to `[0,255]`.

#### `Saturation(rgb)`
HSV **S** channel: `S = (max(R,G,B) − min(R,G,B)) / max(R,G,B)`, scaled to 0–255 to match OpenCV's convention. Used to distinguish neutral paper/pencil from saturated coloured ink or background clutter.

#### `OtsuThresholdValue(gray)`
The same **Otsu's method** as `_otsu` in Part 1 (Otsu, 1979), but implemented as an explicit loop over 256 integer bins accumulating `wB` (background weight), `sumB` (background first moment), and tracking the `t` that maximises `wB·wF·(mB−mF)²` — an unvectorised but mathematically identical formulation of between-class variance maximisation.

#### `AdaptiveThresholdInv(gray, blockSize, C)`
Mirrors OpenCV's `ADAPTIVE_THRESH_GAUSSIAN_C`: a pixel is "ink" if it is darker than its own **locally Gaussian-weighted mean** minus a constant margin `C`:
```
ink(y,x) = gray(y,x) < GaussianBlur(gray, σ)(y,x) − C
```
with `σ` derived from `blockSize` via OpenCV's own formula `σ = 0.3·((blockSize−1)·0.5 − 1) + 0.8`. This is the classical **adaptive/local thresholding** family (Niblack 1986 and successors use local mean ± offset; this is the Gaussian-weighted variant), useful where illumination varies across the page.

#### `Dilate(mask, ky, kx)` / `Erode(mask, ky, kx)` / `Open` / `Close`
Binary morphology implemented via `BoxSum`: `Dilate` is "any pixel set in the window" (`sum > 0.5`), `Erode` is "all pixels set in the window" (`sum ≥ count − 0.5`). `Open = Dilate(Erode(...))`, `Close = Erode(Dilate(...))` — the textbook definitions of morphological opening/closing (Serra, 1982) with a rectangular structuring element, built on the same box-sum trick used everywhere else in the file.

#### `_RunUF` (class)
A minimal **union-find / disjoint-set** structure (`parent` array, `find` with path halving, `union` by arbitrary root pointer reassignment — no union-by-rank). Textbook Tarjan-style disjoint-set forest.

#### `_RowRuns(mask)`
Per row, extracts the `(start, end)` column index pairs of contiguous ink runs via a first-difference of the zero-padded row — identical technique to `_label`'s row-run extraction in Part 1.

#### `LabelComponents(mask, connectivity=8)`
The connected-component labeller used throughout `SegmentPage.py` (and, notably, still the function called by `BuildStyleProfile.py`'s own import, `from SegmentPage import LabelComponents`, for other purposes). Runs are extracted per row (`_RowRuns`), then runs in adjacent rows are unioned whenever their column ranges touch (with an extra 1-pixel tolerance when `connectivity=8`, allowing diagonal touches) using `_RunUF`. Final labels are compacted via a dictionary mapping each resolved root to a sequential output label. This is the same family of algorithm as `_label` in Part 1 (Rosenfeld & Pfaltz, 1966, run-based union-find connected components), implemented with an explicit Python-level union/compaction loop rather than Part 1's fully-vectorised version.

#### `ComponentStatsFromLabels(labels, count)`
For each label `1..count`, computes bounding box (`x,y,w,h`), pixel `area`, and centroid (`cx,cy`) by sorting pixel coordinates by label and slicing contiguous blocks (`np.searchsorted` on the sorted label array) — an efficient batched alternative to `cv2.connectedComponentsWithStats`.

#### `FillHoles(mask)`
Fills every background region that is **not connected to the image border** — the classical **hole-filling via background flood-fill** technique: label the background (`~mask`) with 4-connectivity, mark every background component touching any of the four image edges as "outer," and treat every other background component as a "hole" to be filled in. Equivalent to `cv2.floodFill` from a corner, or to morphological reconstruction by dilation from the border mask (Vincent, 1993) restricted to background components.

#### `ResizeBilinear(img, newH, newW)`
Inverse-mapped **bilinear interpolation** resize: for each output pixel, the corresponding fractional source coordinate is computed (`(i+0.5)·h/newH − 0.5`), its four surrounding integer neighbours are read, and the result is their weighted average by fractional distances (`fy, fx`) — the standard bilinear resampling formula used in essentially all image-resizing libraries.

#### `DownsampleArea(mask, factor)`
Block-mean downsampling: reshapes the (factor-cropped) array into `(h/f, f, w/f, f)` blocks and averages over the two factor axes — equivalent to OpenCV's `INTER_AREA` for integer downsampling factors.

#### `RotateRaw(arr, angleDeg, nearest=False, fill=0)`
Rotation about the image centre via **inverse mapping**: for each output pixel, the source coordinate under the inverse rotation is computed (`srcX = cosθ·dx − sinθ·dy + cx`, `srcY = sinθ·dx + cosθ·dy + cy`), then either nearest-neighbour or bilinear sampling is applied, with out-of-bounds pixels filled with `fill`. This is the standard affine-warp-by-inverse-mapping technique (identical in spirit to `cv2.warpAffine`). The function's docstring flags an important historical gotcha: `Rotate()` below (the pipeline-facing wrapper) flips the angle's sign relative to this raw primitive, and a direction mismatch between the two was a real, since-fixed bug (see `ProcessPage` below).

---

### Section B — baseline segmentation primitives

#### `LoadImage(imgPath, targetLongSide=TARGET_LONG_SIDE)`
Loads an image file, applies EXIF orientation correction (`ImageOps.exif_transpose`), converts to RGB, and resizes (via `ResizeBilinear`) so its longest side is `TARGET_LONG_SIDE=2400` px — a fixed working resolution so all the pitch-relative heuristics downstream behave consistently regardless of the camera's native resolution.

#### `_LargestGoodBlock(good)`
Given a 1D boolean array, finds the single longest contiguous run of `True` and returns a mask that is `True` only over that run — a simple longest-run-of-ones scan, used to pick the dominant "paper" band along one axis while discarding shorter disconnected good-looking bands (shadow islands, etc.).

#### `CorrectIllumination(gray)`
The same **flat-field correction** as `_ink` in Part 1: divides the grayscale image by a wide Gaussian blur of itself (`σ = width/30`) and rescales to 0–255, removing slow lighting gradients while preserving local contrast (ink vs. paper).

#### `RedInkMask(rgb)`
A colour-based rule classifying pixels as "red ink" if the red channel sufficiently dominates both green and blue by a multiplicative margin (`r > 1.30g` and `r > 1.30b`, with an absolute gap `> 22`) for bright red, or a looser version (`1.18×`, gap `>14`, `r<190`) for darker red — used to exclude red-pen annotations/corrections from the main ink mask (since they're not the handwriting sample of interest and would otherwise bias e.g. deskew/line-grouping statistics).

#### `BinarizeInk(illum, pageMask)`
Applies `AdaptiveThresholdInv` (Gaussian-local adaptive threshold, margin `C=12`, block size scaled to image width) to the illumination-corrected image, restricted to the page mask — the main ink/paper binarisation used as the pipeline's "ground truth" ink before any recovery/cleanup stages run.

#### `RemoveSpeckles(ink, minSize=10)`
Labels connected components and drops any smaller than `minSize` pixels — a standard **area-opening** noise filter (remove small connected components, c.f. morphological area opening, Vincent 1993) to kill binarisation speckle.

#### `EstimateSkew(ink, searchRange=5.0)`
A coarse-to-fine angle search: downsample the ink mask by 4, and for a grid of candidate angles (first coarse, `0.5°` steps over `±searchRange`; then fine, `0.1°` steps over `±0.5°` around the coarse best), rotate (nearest-neighbour) and score by `var(rowInkProfile)` — maximising the **variance of the horizontal projection profile**, exactly the same projection-profile skew-estimation principle as `SegmentLean`'s deskew step (Postl, 1986), just using variance instead of sum-of-squares (closely related quantities; variance subtracts the mean first, sum-of-squares does not, but both are maximised by the same sharply-peaked-profile condition).

#### `Rotate(arr, angle, isMask=False, fill=0)`
A thin wrapper: `Rotate(arr, angle) = RotateRaw(arr, −angle, ...)`. The docstring block at the top of the file explains this sign flip was historically a source of a real bug: `EstimateSkew` scores candidates using `RotateRaw(ink, a)` directly, so correcting the page must use that *same* direction; an earlier version of the pipeline corrected with `Rotate(rgb, a) == RotateRaw(rgb, −a)` — the wrong sign — which silently **doubled** every tilted page's effective skew (since the per-line deskew in `RenderLine`, applied later, partially masked the damage rather than exposing it). `ProcessPage` (documented below) now explicitly rotates using the same convention `EstimateSkew` scored with.

#### `_HorizontalRunLengths(ink)` / `_VerticalRunLengths(ink)`
For every ink pixel, computes the length of the contiguous horizontal (resp. vertical, via transpose) run of ink it belongs to, using a cumulative run-id trick: assign each run a unique increasing integer id via `cumsum` of run-start flags, then `bincount` run ids to get each run's length and broadcast that length back to every pixel of the run. Used to distinguish thin, long structures (rules, underlines) from the blockier shapes of ordinary letters.

#### `UnderlineLike(c, textH, labels=None)`
A component is "underline-like" if its width is **not** disproportionately larger than `4×textH`, or if its area-to-width ratio exceeds 6.5 (would indicate it's not a thin line)... more precisely, the function returns `False` (not underline-like) when `c['w'] ≤ 4·textH` AND the component is NOT abnormally "thick for its width" (`area/w > 6.5`); when `labels` is given it refines further by checking the **median vertical span per column** is small relative to `textH` (`≤ max(8, 0.4·textH)`), i.e. the component is thin everywhere along its length, not just overall wide.

#### `RemoveRuleLines(ink, textH, illum=None)`
Detects printed/hand-ruled lines: candidate horizontal pixels are those on a long-enough horizontal run (`≥max(20,0.8·textH)`) that are also thin vertically (`vRun ≤ thkCut`, where `thkCut` is derived from the median thickness of genuinely long (`≥3.5·textH`) horizontal runs); candidate vertical pixels are the transposed equivalent. Both candidate sets are passed to `_ChainLongStructures` (below) to merge nearly-colinear fragments into full rule lines (handling rules broken by crossing pen strokes), and the detected `ruleMask` (slightly dilated, then re-restricted to genuinely thin pixels) is subtracted from `ink`.

#### `_ChainLongStructures(cand, minSpan, axis, textH, thk, lo=None, hi=None, illum=None)`
Given candidate rule-line pixels, labels them into components, optionally records each component's **darkness** (`255 − median illum`), then **chains** components along `axis` using a union-find: two components are joined if the gap between them is within `2.5·textH`, their centreline deviation is within a slope-tolerant allowance (`slopeAllow = 2·thk + 0.08·gap + 0.04·(widths)`, i.e. a generous allowance that grows with distance and width, approximating how much a *slightly sloped* straight rule line can legitimately drift), and (if darkness is tracked) the two components are similarly dark (`|Δdark| ≤ 55`, since real ruled lines are consistently printed/drawn, unlike handwriting touching them). Chained groups spanning at least `minSpan` (and, if given, starting before `lo`/ending after `hi` — i.e. running most of the way across the page/column) are kept as "this is a rule line" and their original component ids are returned as a boolean keep-mask.

#### `_Otsu1D(values, bins=64)`
Otsu's method (same derivation as `_otsu`/`OtsuThresholdValue`) applied to an arbitrary 1D array of continuous `values` via a 64-bin histogram — used later to threshold component **darkness** values rather than pixel grey levels (an interesting reuse: Otsu's method is a general bimodal-separation technique, not inherently tied to pixel intensity).

#### `ComponentStats(ink)`
Thin wrapper combining `LabelComponents` + `ComponentStatsFromLabels` into one call returning `(labels, comps)` as a list of per-component dicts.

#### `EstimateTextHeight(comps)`
Estimates the typical glyph height `textH`: takes components with area ≥30 and aspect ratio not extremely wide (`w < 12h`, excluding rule-like shapes), takes the **median** height, then refines by taking the median again restricted to heights within `[0.3, 3.0]×` that first median (trimming outliers) — a two-pass robust median, defaulting to `30.0` if no components qualify.

#### `ScoreComponentMess(comp, labels, textH)`
Produces a 0–1 "diagram-ness" (MESS, i.e. non-text) score for one component, combining two signals: (1) **enclosed-hole area** relative to glyph scale — `FillHoles` on the padded component finds background "holes," the largest hole's area is compared to `textH²`, and a hole bigger than `0.7×` that reference area starts contributing score, saturating at `1.5×`; letters like "o"/"a"/"e" have small holes relative to a whole diagram's enclosed area, so this distinguishes genuine line-art/diagram loops from ordinary letterforms; and (2) a **"wide and sparse" fallback** — a component much wider and taller than a glyph (`w>6·textH`, `h>1.8·textH`) with low fill fraction (`area/(wh) < 0.05`) scores `0.7`, catching open line-art that has no enclosed area at all. The final score is the max of the two signals.

#### `_UnionFind` (class)
A second, near-identical union-find implementation (used by the baseline `GroupLines`/`_ChainLongStructures` code, separate from `_RunUF` used by `LabelComponents`) — same path-halving `find`, arbitrary-direction `union`.

#### `_CompMask(labels, c)`
Returns the exact boolean pixel mask of one component: either its pre-computed `pixmask` (for components that were split pixel-wise during grouping, see `GroupLines` stage 4) or, failing that, `labels[bbox] == c['id']`.

#### `_LineCurve(comps)` / `_CurveY(curve, x)`
`_LineCurve` builds a smoothed **baseline curve** for a set of components: sorts them by x-centre, then for each, takes a **width-5 sliding window** (`i−2..i+2`) weighted average of `cy` by component `area` — a simple local weighted-moving-average smoother robust to any one component's noisy vertical position. `_CurveY` then linearly interpolates (`np.interp`) that sparse curve to any query x. Together these implement a piecewise-linear approximation of each line's local baseline, used throughout the grouping/refinement code to test "does this point sit on this line's curve."

#### `GroupLines(ink, labels, comps, textH, imgH)`
The baseline line-grouping algorithm, in six numbered sub-stages (inline comments `# 1.`–`# 6.` in the source):
1. **MESS candidate detection**: components scoring ≥0.5 on `ScoreComponentMess` and large/tall enough are MESS candidates, but a `_NeighborVeto` check demotes a candidate back to TEXT if it has ≥3 plausible nearby glyph-shaped neighbours no taller than `1.9×` their 75th-percentile height (catching large heading letters that would otherwise look "sparse enough" to be mistaken for a diagram). Surviving MESS components are grouped into spatial `messBlocks` by greedy nearest-block gap-merging, and each block's "core" bounding box is refined by keeping only columns genuinely spanning more than `textH` vertically (trimming thin connecting wisps from the headline bbox).
2. **Core-component chaining**: "core" text components (near-glyph height, not underline-shaped) are **sorted left-to-right** and unioned via a greedy nearest-neighbour **minimum-cost chain**: for each component, the best candidate to its right is the one minimising `cost = max(0,gap) + 3·|Δcy|` within a horizontal search window (`gap ≤ 6·textH`) and vertical tolerance (`|Δcy| ≤ 0.75·textH`) — this is a greedy 1D chaining heuristic (not a true shortest-path/MST, but structurally similar to word/line segmentation "nearest component" chaining used in classical OCR layout analysis).
3. **Chain merging by shared baseline** (two passes): chains whose `x`-spans overlap are merged if their baseline curves (`_LineCurve`/`_CurveY`) agree within `0.55·textH` (or, for non-overlapping chains, if their curves extrapolate to meet within `0.6·textH` at the midpoint of the gap); a **second, pitch-aware** pass repeats this with a tolerance scaled to the just-estimated line pitch (`0.4–0.45×` pitch) rather than `textH`, catching lines whose spacing differs from the typical glyph size.
3c. **Weak-chain demotion**: chains with fewer than 3 components and not wide/large enough are only kept if no other chain's centre line is within `0.8×` pitch of them (otherwise they're too likely a fragment of a real line rather than their own line).
4. **Leftover assignment**: non-core components not yet in any chain are handled by shape — underline-shaped ones attach to whichever chain's curve they sit a short, consistent distance below (interpreted as "this chain's underline"), or fall back to a MESS block; moderate-height components join the `nearestChain` (a penalised distance metric that adds a penalty for falling outside the chain's own x-span, discouraging over-reaching matches); and components taller than `1.8·textH` (likely interline components spanning two text rows) are split **pixel-by-pixel**, assigning each individual ink pixel to its own nearest chain and creating a new partial-component (`pixmask`-carrying) entry for each chain that claims a substantial (`≥8px`) share.
5. **Finalisation**: chains are converted into `textLines` if they have enough total area (`≥0.35·textH²`) and at least one component tall enough (`≥0.35·textH`); very small, very dense (`fill>0.45`), narrow (`<3.5·textH`) chains of ≤3 components are dropped as noise rather than real lines.
6. **MESS block consolidation**: overlapping-in-x MESS blocks whose vertical gap doesn't straddle a real text line's centre and isn't wider than either block's own height are merged (treating them as fragments of one larger diagram), and any "text line" whose bounding box sits mostly (`>70%` overlap) inside a MESS block's core is absorbed into that block instead (text-looking squiggles that are actually part of a drawing).

**Literature connection.** This whole function is a bespoke **connected-component chaining and merging** approach to text-line segmentation — conceptually related to the classical "nearest-neighbour component chaining" and "smearing" families of document layout analysis (e.g. RLSA — Run-Length Smoothing Algorithm, Wong, Casey & Wahl 1982 — achieves a similar grouping effect by morphological smearing rather than explicit graph chaining), combined with a bespoke diagram/MESS classifier based on topological hole area (loosely related to Euler-number / genus-based shape descriptors used in document image analysis for distinguishing text from graphics).

---

### Section C — override stages (`SegmentPage.py`'s own improvements)

#### `_TrimRaggedGentle(mask, factor=0.30)`
Trims a page mask's ragged edges along each axis by keeping only the largest contiguous block of rows/columns whose ink-density profile exceeds `0.30×` the profile's own (high-percentile-filtered) median — a gentler version of an earlier, stricter (`0.55`) cutoff, chosen so a page that runs off the photo frame doesn't lose its first/last real line of text to over-aggressive trimming.

#### `_ConvexFill(mask)`
Fills the **convex hull** of a mask: extracts the left/right boundary points per row, computes the 2D convex hull via the standard **Andrew's monotone chain algorithm** (sorting points, building upper and lower hull chains with a cross-product turn test `cross(o,a,b) = (a−o)×(b−o)` to discard non-left-turns), then fills it in by scanline rasterisation (per-row min/max x along each hull edge). Justified because a photographed page is geometrically a convex quadrilateral, so any concave notch in the detected mask (e.g. a shadowed corner that failed the brightness threshold) is necessarily a detection error, not a real page shape, and can safely be filled back in.

**Literature connection.** Convex hull via Andrew's monotone chain (a variant of the Graham scan family; Andrew, 1979) — standard computational geometry.

#### `_TrimDarkEdgesAdaptive(mask, blur)`
Trims mask rows/columns whose median brightness falls below a cutoff set **relative to the page's own median brightness and the true off-page background's median** (`cut = min(pageMed−45, max(pageMed−85, 0.5(pageMed+bgMed)))`) rather than an absolute darkness value — so a page edge in soft shadow (which can still be ~70% as bright as the rest of the page) is correctly kept, while genuinely different (table/desk) background is trimmed.

#### `DetectPageMask(rgb)`
The full page-boundary detector: Otsu-threshold a blurred grayscale+low-saturation image to find "paperish" regions, take the largest connected component (twice — a second pass re-thresholds relative to the first pass's own median brightness, correcting for the first pass's threshold being biased by non-paper pixels), fill holes and close small gaps, then apply `_TrimRaggedGentle` and `_TrimDarkEdgesAdaptive`. A further repair step handles shadowed bands **inside** the page: it forces the mask to be contiguous along both column and row axes (closing off any internal hole/notch), applies `_ConvexFill`, and only re-admits newly-filled pixels that are still at least `0.55×` the page's own median brightness (so true off-page background — a desk strip, a spine gap — is never wrongly pulled in), then runs `FillHoles` once more so ink strokes inside a newly-recovered shadow region are preserved rather than treated as background holes. A final light erosion (scaled to image size) trims the boundary by a few pixels; if the detected mask covers implausibly little of the frame (`<15%`), the function gives up and returns "the whole image is paper."

#### `RemoveEdgeComponentsWide(ink, pageMask, textH)`
Drops ink components that are implausibly close to the detected page-mask's boundary — either narrow components touching the left/right mask edge, or components touching both a horizontal and a vertical mask edge simultaneously (a corner rule, catching junk binarised in a hull-filled page corner, which can never be real handwriting since text doesn't originate exactly at a page corner).

#### `RemoveOffPageColumns(ink, textH, alsoClip=None)`
Detects a **facing notebook page** peeking in at the image border: the smoothed column-ink-density profile is thresholded, broken into contiguous "on" blocks, and the single densest block is assumed to be the real page's text columns; any other block that **touches the image border** (not an interior block, which would be legitimate marginalia like paragraph numbering) is treated as bleed from a facing page and its components are dropped (and, if given, `alsoClip` — the recovery mask — is also clipped to the same column range).

#### `_StripThinRuns(ink, labels, stats, sel, textH)`
For each selected (rule-suspect) component, finds, **per column**, the vertical span of ink (`ymax−ymin+1`); columns where that span is below a threshold (`thinSpan = max(6, 0.35·textH)`) are "thin" (rule-like, not glyph-like); contiguous thin-column runs of at least `2·textH` are stripped from the mask. Operating **per-column rather than globally** is what lets this catch a **sloped or wavy rule** — a whole-component run-length test would see a sloped line as many short horizontal segments and miss it, but the per-column vertical-span test doesn't care about slope at all.

#### `_MaxHole(labels, st, i)`
Same enclosed-hole measurement as used inside `ScoreComponentMess`: pads and fills a component's mask, labels the resulting holes, and returns the largest hole's pixel area (or 0 if none) — reused here to distinguish rule-glue networks (no real enclosed area) from genuine diagrams.

#### `_CompDarkness(labels, n, illum)`
Per labelled component, computes "darkness" as `255 − 25th-percentile(illum values)` — the 25th percentile (rather than the mean) is a robust proxy for the component's *darkest, most ink-like* pixels, insensitive to anti-aliased/faded edge pixels pulling the average up.

#### `HysteresisRecoverInkWide(illum, pageMask, strong, textH)`
Classic **hysteresis thresholding** (as in Canny edge detection): a looser adaptive threshold (`C=3`, more permissive than the main `C=12` binarisation) produces a `weak` ink mask; any weak component touching the `strong` (confirmed) ink mask within a near-range dilation is promoted. The "wide" variant here additionally promotes weak components reachable within a much larger **horizontal-only** dilation (`reach = 2·2·textH+1`, i.e. roughly a word-gap), but only if the weak component is substantial (`area ≥ 40`) — so a lightly-pressed trailing word one word-gap away from its row's strong ink gets recovered for crop rendering, while isolated speckle noise far away does not, and nothing bridges vertically between rows (which would wrongly merge lines).

**Literature connection.** Hysteresis thresholding (strong seeds + weak-but-connected growth) is the same two-threshold linking principle used in Canny's edge detector (Canny, 1986), applied here to ink recovery instead of edges.

#### `CleanRecoveredInk(inkRaw, ink0, illum, textH)`
Cleans the hysteresis-recovered "weak-only" pixels (`inkRaw & ~ink0`) of three artifact types that the permissive weak threshold otherwise lets through: (1) long thin near-horizontal runs within wide weak components (rule-line remnants glued to real ink, stripped via the same per-column `_StripThinRuns` technique used elsewhere), (2) components far **fainter** than the page's own established "real ink" darkness (`strongDark`, from `ink0`) — a component below `0.40×` that reference darkness is dropped outright as a bleed-through ghost, one between `0.40×` and `0.55×` is dropped only if it is also *shaped* like a rule or margin stub (flat-wide, tall-thin, or simply tiny), while anything ≥`0.55×` as dark as real ink is kept regardless of shape (it's probably genuinely faded ink, not a ghost).

#### `FaintFilterRuleAware(ink, illum, textH)`
Replaces a hypothetical naive "drop all faint components" filter with a **rule-aware** version: first detects wide, low-fill, hole-free components (candidate residual rule+word glue) and strips their thin per-column runs via `_StripThinRuns` (separating the actual rule pixels from any real letters glued to it), *then* measures the darkness distribution of the resulting (now rule-stripped) components, and applies **Otsu's method to the darkness values themselves** (`_Otsu1D`) to split "genuinely faint" from "ink-dark" components — but only if the darkness distribution is actually bimodal enough (`90th percentile − 10th percentile ≥ 60`, and the computed threshold must fall strictly between those percentiles) to trust a single global cut; otherwise it conservatively keeps everything. This two-step detach-then-threshold design is what lets faint printed rules be discarded while real (possibly lightly pressed) words glued to them survive — a single whole-component darkness test would have thrown both away together.

#### `StripSparseRuleNetworks(ink, textH)`
A simpler, earlier-stage relative of `FaintFilterRuleAware`: finds wide, tall-ish, low-fill, hole-free components (candidate glue-networks — rules that have merged multiple words/rows together into one blob) and strips their thin per-column runs via `_StripThinRuns`, without any darkness reasoning — this runs before illumination-based darkness is trustworthy (right after the main binarisation), so it relies purely on shape (thin-ness, lack of enclosed holes) to identify rule material.

#### `_LineSpan(l)` / `_Recenter(l)` / `_LinePitch(textLines, textH)` / `_LineYRange(l)` / `_SharedColumnsFrac(a, b)`
Small geometry helpers used by the refinement pass below: `_LineSpan`/`_LineYRange` give a line's x/y extents from its components; `_Recenter` recomputes a line's area-weighted mean `cy` after its component list changes; `_LinePitch` estimates line spacing the same way `_pitch`'s direct-peak-difference method does in Part 1 (median gap between sorted line centres, restricted to gaps `>0.5·textH` to avoid counting near-duplicate centres); `_SharedColumnsFrac` computes what fraction of the smaller line's occupied x-columns are also occupied by the other line (`|A∩B| / min(|A|,|B|)`) — a proxy for whether two "lines" are stacked (would share almost all columns) versus interleaved (side-by-side words on one true row, sharing almost none).

#### `MergeSameRow(textLines, textH)`
Detects and merges two detected "lines" that are really one physical row that chaining (in `GroupLines`) split apart — e.g. a dense page, an unusually large word gap, or a short word written slightly higher after a question mark. Four independent signals, any one of which triggers a merge (tiered, from strictest to most permissive):
1. **Near-identical centres** (`|Δyc| < 0.5·textH`).
2. **Containment**: the shorter line's y-range sits almost entirely (`≥70%`) inside the taller line's y-range, and both lines are individually small (total area `<4·textH²`) — i.e. a small fragment plausibly nested inside a bigger line's vertical extent.
3. **Curve-meeting at the seam**: if the two lines barely overlap in x (`≤55%` of the narrower's width), their baseline curves (`_LineCurve`/`_CurveY`) are compared at the midpoint of their gap; if they'd meet within a pitch-scaled tolerance, they're treated as one row continuing past a gap.
4. **Low column overlap** (`_SharedColumnsFrac < 0.3`) combined with close centres (`<0.6·pitch`) — interleaved words at the same height, which stacked separate rows would never exhibit.

This runs in an outer fixed-point loop (`while changed`) since one merge can change the pitch estimate and enable further merges.

#### `_UnderlineShaped(c, textH, labels)`
A geometric test for "is this component shaped like an underline (flat and wide)": too narrow (`w < 1.5·textH`) disqualifies it outright; very flat (`h ≤ max(4, 0.45·textH)`) qualifies immediately; otherwise (for moderate heights, `≤1.2·textH`) it checks the **median per-column stroke span**, qualifying only if that's thin (`≤ max(3, 0.2·textH)`) — this is what lets a double/triple underline (tall overall bounding box, but each individual stroke thin) still register as underline-shaped, which a simple height check would miss.

#### `ReassignUnderlines(textLines, textH, labels=None)`
Fixes a specific chaining failure: an underline sitting **between** the text it underlines and the row below can get chained into the row *below* instead (because it's geometrically between them). For each line, the "solid" (non-underline-shaped) components define that line's true baseline curve; any remaining (underline-shaped) component sitting noticeably **above** that curve (`curveY(x) − cy ≥ 0.5·textH`, i.e. well above where this line's own text sits — exactly where an underline *of the line above* would be) is reassigned to whichever *other* line's curve it sits the right small distance (`0.15–2.2·textH`) below.

#### `AttachFaintToLines(preInk, postInk, textH, textLines)`
Recovers components that `FaintFilterRuleAware` legitimately removed as "too faint" but that actually belong to a real text line (e.g. a lightly-pressed trailing word at the end of a row). Computes the diff mask `preInk & ~postInk` (exactly what was removed), and — in **two passes** (so a word attached in pass 1 extends a line's known span/curve for pass 2 to catch an even-further word) — re-attaches diff components that sit close (`< 0.75·textH`) to some line's baseline curve, within or a bit beyond (`8·textH`) that line's known x-span.

#### `RefineItems(textLines, messBlocks, textH, labels=None)`
A thin orchestration wrapper: `MergeSameRow` → `ReassignUnderlines` → `MergeSameRow` again (since reassigning underlines can change what should be merged), then refreshes each MESS block's vertical centre.

#### `BuildOwnerMap(shape, items, labels)`
Builds a per-pixel "owner" map (`int16`, `-1`=unowned) by painting each item's (line's or MESS block's) component pixels with that item's index, in item order — the authoritative record of "this pixel belongs to this specific crop" used to prevent later rendering/recovery stages from leaking pixels between adjacent crops.

#### `ExtendOwnerToWeak(owner, inkRawLabels, nRaw)`
Extends ownership to an entire hysteresis-recovered weak component if **all** of the strong (owned) ink underneath it belongs to a single item — reasoning that a weak component is the faint halo/fade of whichever strong ink it's physically touching, so if that's unambiguous, the whole halo should belong to the same owner (implemented via an encoding trick: pack `(componentLabel, owner)` pairs into single integers via `label*(maxOwner+2)+owner`, then use `np.unique`/`bincount` to detect, per weak component, whether exactly one distinct owner touches it). Components touching **multiple** owners, or none, are deliberately left unresolved (`-2`/`-1`) rather than guessed at.

#### `AttachWeakTrailing(items, owner, sole, inkRawLabels, nRaw, textH, weakLabels=None, nWeak=0, illumRef=None, labels=None)`
Handles pale trailing words that are too far from any strong ink anchor for even the wide hysteresis reach to connect them. For each raw recovered component (and, separately, for an even-weaker "anchor-free" pool `weakLabels`), a nested `place()` helper rejects implausible shapes (too small, too wide, not glyph-height, or flat-and-wide like a rule dash), then finds the best-matching text line by proximity to its baseline curve (within a generous rightward reach of `12·textH`, since a trailing word can be well past a line's last confirmed component) and — when illumination data is available — rejects the match if the candidate is far fainter than that line's own established ink darkness (`< 0.5× rowDark`, filtering out bleed-through ghosts that happen to sit on the same curve). Matches are appended directly as new pixmask-carrying components of the chosen line, and the owner map is updated in place.

#### `_CompDark(c, labels, illum)`
One-line helper: 25th-percentile-based component darkness (same formula as `_CompDarkness`, but for a single component rather than a batch).

#### `FilterFaintFlatComps(comps, labels, illum, textH)`
A final per-line cleanup: computes an **area-weighted median darkness** across a line's "glyph" components (excluding underline-/stub-shaped ones) as the line's own reference darkness (`ref`), then drops any flat (underline-shaped) or thin-tall "stub" component that is markedly fainter (`<0.72×ref`) than that reference — catching residual printed-rule fragments that ride along a row as flat faint comps, while leaving genuine ink-dark pen underlines (which are as dark as the line's own writing) untouched.

#### `RenderLine(gray, labels, comps, pad=6, inkRaw=None, inkRawLabels=None, gapPx=6, deskew=False, forbid=None)`
Renders one item's final crop. Builds a tight mask from exactly its components' pixels, optionally dilates it slightly and pulls in nearby raw/recovered ink within a column-reach of `gapPx·3` (so faint recoverable pixels just outside the strict component boundary still make it into the crop) — but **never** across a `forbid` mask (pixels already confirmed to belong to a *different* item), which is the key fix preventing a neighbouring line's descender/ascender from leaking into this crop even when it geometrically dips into this row's band. The crop is cropped tightly back down to its actual non-white extent after recovery. If `deskew=True` and the crop is wide and has enough components, a **per-line residual deskew** is applied: a weighted linear fit (`np.polyfit`, weighted by `√area`) of component centres gives a local slope, converted to an angle, and if it's a plausible small residual tilt (`0.4°–12°`) the crop is padded and rotated (`RotateRaw`) to level it — catching any small per-line tilt the single global deskew didn't fully correct.

#### `_InkGray(rgb)`
A colour-aware grayscale variant: `0.5·luma + 0.5·min(R,G,B)` — blue (or other saturated) ink can be nearly invisible in pure luma in a dim photo corner (since luma weights green heavily) yet is obviously dark in at least one colour channel; blending in the channel-minimum recovers that contrast for *recovery* purposes specifically (the main *detection* binarisation stays luma-pure, to avoid false ink from colour noise).

#### `ProcessPage(imgPath)` — the entry point

Ties every stage above together in this order:
1. `LoadImage` → `DetectPageMask` → `RgbToGray` → `CorrectIllumination` → `BinarizeInk` (minus `RedInkMask`) → `RemoveSpeckles`: the initial strong ink mask `ink0`.
2. **Deskew, with the historical sign-bug explicitly avoided**: `EstimateSkew(ink0, searchRange=8.0)` finds the angle, then the page is rotated with `Rotate(rgb, -a, ...)` in a loop that re-estimates and re-applies a residual correction (`searchRange=3.0`) a second time to mop up any remainder — explicitly matching the direction `EstimateSkew` itself scored with, which the file's docstring flags as the fix for the old (silently doubling) sign-mismatch bug. `ink0` is fully recomputed (illumination, binarisation, speckle removal) after each rotation since geometry changed.
3. Estimate a rough text height (`EstimateTextHeight`), then build a **colour-aware recovery mask**: `_InkGray`/`CorrectIllumination` on the colour-blended grayscale, and `HysteresisRecoverInkWide` run **twice** — once from pure-luma illumination, once from colour-aware illumination — unioned together, so faint luma ink and faint-but-colourful (blue pen) ink are both recovered; long thin runs are stripped back out of the recovered mask immediately (catching rule lines the weak threshold reintroduced).
4. `FaintFilterRuleAware` then `RemoveEdgeComponentsWide` clean up `ink0`; `RemoveOffPageColumns` removes any facing-page bleed (also clipping the recovery mask to the same bounds); `RemoveRuleLines` is run **twice** (a second pass catches rule fragments only exposed after the first pass's removals) followed by `RemoveSpeckles` and `StripSparseRuleNetworks` (another speckle pass after).
5. The recovery mask has all now-detected rule pixels dilated out (`inkRaw &= ~Dilate(rmAll,3,3)`), then is cleaned of residual rule/ghost material via `CleanRecoveredInk`.
6. `ComponentStats` + `EstimateTextHeight` on the final cleaned `ink` feed `GroupLines` (the baseline chaining algorithm) to get `textLines`/`messBlocks`, which are then run through `RefineItems` (the `MergeSameRow`/`ReassignUnderlines` fixups) and `AttachFaintToLines` (recovering legitimately-removed faint trailing words).
7. `TEXT` and `MESS` items are merged into one ordered `items` list (sorted top-to-bottom by vertical centre), with one explicit reordering rule: a `MESS` block whose top edge is just above (within `40%` of its own height) a `TEXT` item's centre is swapped to come *after* that text item (handling the case of a drawing whose top slightly overlaps the row above it).
8. Each `TEXT` item's components get a final `FilterFaintFlatComps` pass; `BuildOwnerMap` establishes authoritative per-pixel ownership; `ExtendOwnerToWeak` extends it onto unambiguous recovered-ink halos; a second, even-weaker "anchor-free" recovery pool (`weakAll`, rule-stripped via `CleanRecoveredInk` again) is built specifically to catch pale words with literally no strong-ink anchor; `AttachWeakTrailing` resolves both recovery pools onto the nearest plausible line.
9. Final rendering uses `renderGray = min(illum, illumC)` (the darker, hence more conservative/contrast-preserving, of the pure-luma and colour-aware illumination corrections) so backgrounds render uniformly white while dim-corner coloured ink keeps visible contrast; `RenderLine` is called per item with the `forbid` mask set to "every pixel owned by a different item" so crops never bleed into each other; a debug `preview` image is drawn with coloured bounding boxes (orange = MESS, green = TEXT) and order/tag labels.
10. Returns `(results, preview, meta)` where each `results[i]` carries `order`, `tag` (`"TEXT"`/`"MESS"`), `bbox`, the raw crop array, and its component count, and `meta` carries the applied skew angle, estimated text height, and TEXT/MESS counts.

---

## Known limitations / honest caveats

- **The documented DP-seam-through-thin-strokes bug and its fix (`SegmentLean.py`).** The design history of `SegmentLean.py` includes a real bug: on words with narrow internal strokes very close together — the worked example in the code's own comments is **"express"** (letters like "x" and "p" whose strokes come close to touching, and the narrow neck of a descender like "y") — a naive ink-cost map for the line-separating DP seam (built straight from the raw ink mask) could contain a thin channel of zero-cost background running right past or through the letter's interior. Because `_seam`'s DP objective simply minimises total ink crossed, it would happily route the boundary seam through that channel, slicing through the middle of a word instead of around it. The fix, implemented in `SegmentLines` stage 4 (source lines 289–298), is to build the seam's cost map from a **morphologically closed** version of the mask (`dilate` by `tailR=max(2,0.07·P)` then `erode` by the same radius) rather than the raw mask: closing fills any gap *smaller than* `2·tailR` (exactly the kind of narrow internal neck that caused the bug) while leaving the stroke's true outer silhouette essentially unchanged (unlike a plain dilation, which would also bloat the outer boundary and hide real separating whitespace). This is a textbook application of morphological closing to "heal" small defects without altering overall shape (Serra, 1982) — but it is worth being explicit in the report that this was arrived at empirically (ablating against real failure cases), not derived from a formal model of handwriting geometry.
- **`SegmentLean.py`'s stage-5 piece-ownership logic (cases A–H) is a long, hand-tuned heuristic cascade**, not a single clean algorithm — it exists because the simple "DP boundary splits pixels into lines" model breaks down for several predictable real handwriting shapes (i-dots, cut-off tails, stray marks, components the closing operation itself accidentally welds together). Each case is independently justified and testable, but the overall design is closer to an expert system / decision list than to a textbook segmentation algorithm, and a report writer should present it as such rather than implying it follows from first principles.
- **`SegmentPage.py` is legacy/superseded** for the server and for current training (`TrainText.py`/`TrainAuthor.py`), both of which now use `SegmentLean.SegmentLines`. Its **one remaining live caller** is `build_personal_line_items` in `Software/CNN/AuthorReproductionStuff/BuildStyleProfile.py`, which still imports `SegmentPage` as `F` and calls `F.ProcessPage(...)` to segment personal authors' photographed pages before forced-alignment for style-profile extraction. The parallel, more recently touched copy of that build script (`Software/CNN/ReproductionImprovement/BuildStyleProfile.py`) has already been switched over to `SegmentLean.SegmentLines` for the same purpose, which should be read as a signal that `SegmentPage.py`'s remaining usage is itself likely to be migrated away eventually — it is retained today only because that particular script path hasn't been updated yet.
- **`SegmentPage.py`'s own module docstring claims every remaining stage was validated by ablation** against real camera photos ("functions that turned out to have no effect... have been removed"); this document takes that claim at face value since no ablation logs were available to re-verify it directly, but it means essentially the whole file should be treated as "live," not merely "legacy code nobody reads."
- Several thresholds throughout both files (e.g. `0.85×otsuT` for "faint," `2.2×` for pitch-from-letter-height, the `0.15P`/`0.35P` ownership-share cutoffs, `MergeSameRow`'s four signal tiers) are empirically tuned constants rather than values derived from a model — accurate to report as engineering calibration against real scanned/photographed pages, not as theoretically optimal constants.
