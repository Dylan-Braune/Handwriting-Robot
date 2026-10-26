# Block A — Segmentation and image primitives

How a page of handwriting becomes a set of single-line images.

| File | Lines | Role |
|---|---|---|
| `SegmentLean.py` | 575 | **Production** page-to-line segmenter. The only segmenter a live web request touches. |
| `ProfileIO.py` (primitives half) | 348 | The reusable binary-image toolbox: integral images, Otsu, morphology, connected components, Zhang–Suen thinning, slant estimation, stroke normalisation. |
| `ExtractIAMLines.py` | 448 | **Offline only.** Cuts scanned IAM database pages, and personal ruled-paper pages, into labelled line crops to build the training set. |

Two segmenters exist because they solve different problems. `ExtractIAMLines.py` exploits the IAM form layout — a printed prompt box fenced by horizontal rule lines — which is a flat scan of a known template. `SegmentLean.py` assumes nothing but a photographed sheet of paper under uneven light, which is what the robot's camera actually produces.

---

## A.0 Conventions that hold throughout the block

**Ink is `True`.** Every mask here is a boolean array where `True` means "this pixel is ink". Grey images are `float64` in 0–255, not 0–1, because the code divides by a local mean and guards it with `np.maximum(..., 1.0)`; on a 0–1 scale that guard would clamp everything.

**`P` is the unit of length.** `P` is the line pitch in pixels — the centre-to-centre spacing of consecutive text lines — and almost every threshold in `SegmentLean.py` is a fraction of it (`0.5*P`, `2.2*P`, `0.065*P*P` for areas). This is the most important single decision in the file: it makes the segmenter independent of both image resolution and handwriting size with no calibration step. A 12 MP phone photo and a 2 MP one make the same decisions, because each measures its own `P` from its own row profile first. Where a bare pixel constant appears instead (`max(3, ...)`, `max(8, min(40, ...))`) it is a floor or ceiling guarding a degenerate case, never the operating value.

**Analysis runs at half resolution; crops are cut from full resolution.** `SegmentLines` immediately builds `half = full.resize((w//2, h//2), Image.BOX)` and does *all* analysis on `half`. Only the final crop stage returns to `full`, scaling coordinates by `f = full.width/W`. `Image.BOX` is a 2×2 box average — exact 2:1 decimation with no resampling ringing — so the halved image's ink statistics are the true means of the full image's. The analysis makes several O(pixels) passes, half resolution is four times less work, and `P` is still tens of pixels there, which is ample for every threshold.

---

## A.1 `SegmentLean.py` — photographed page to line crops

Six stages, implemented in order by `SegmentLines(path)`.

### A.1.1 The primitives

**`BoxMean(a, r)` — mean over a `(2r+1)²` window in O(pixels), independent of `r`.**

```python
def BoxMean(a,r):
    p = np.pad(a,r,mode="edge").astype(np.float64)
    ii = np.pad(p.cumsum(0).cumsum(1),((1,0),(1,0)))   #integral image
    k = 2*r+1
    return (ii[k:,k:]-ii[:-k,k:]-ii[k:,:-k]+ii[:-k,:-k])/(k*k)
```

This is the summed-area table of Crow (1984). With `ii[y,x] = Σ_{y'<y, x'<x} a[y',x']`, the sum over any axis-aligned rectangle is four lookups:

$$\sum_{y_0 \le y < y_1,\ x_0 \le x < x_1} a = ii[y_1,x_1] - ii[y_0,x_1] - ii[y_1,x_0] + ii[y_0,x_0]$$

The cost does not depend on window size, which matters because `r` is around 25 px on a typical page — a naive window mean would be ~2,600 operations per pixel. `mode="edge"` pads with the border value rather than zero, so the estimated background near the page edge is not dragged towards black.

**`AnyWithinRadius(m, r, axis)` — binary dilation by a 1-D structuring element of length `2r+1` along one axis.** The same prefix-sum trick on a boolean array: a pixel is set if the count of set pixels in its window exceeds zero. Two calls, `axis=0` then `axis=1`, give a separable rectangular dilation — which is what every "dilate" in this file is. Erosion is the same operation on the complement, which is why closings read as `~AnyWithinRadius(AnyWithinRadius(~dil, r, 0), r, 1)`.

**`LongRunMask(m, n, axis)` — marks every pixel lying *on* a run of at least `n` consecutive set pixels.** Two stacked prefix sums. The first finds windows of length `n` that are entirely ink (`(c[n:]-c[:-n])==n`), giving run *start* positions. The second converts "a run starts here" into "a run covers here" by asking whether any start lies in `[j-n+1, j]`. The coverage form is what the rule-line remover needs: a 400 px underline must have all 400 of its pixels removed, not just the first.

**`OtsuThreshold(v)` — Otsu (1979), vectorised.**

```python
h,_ = np.histogram(np.clip(v,0,1.2),bins=256,range=(0,1.2))
p = h/h.sum()
w,m = np.cumsum(p),np.cumsum(p*np.arange(256))
s = (m[-1]*w-m)**2/np.maximum(w*(1-w),1e-12)
return (int(np.argmax(s))+0.5)*1.2/256
```

Otsu picks the threshold maximising between-class variance $\sigma_B^2 = \omega_0\omega_1(\mu_0-\mu_1)^2$. Writing $\omega(t)$ for cumulative mass and $m(t)$ for the cumulative first moment, the standard rearrangement is

$$\sigma_B^2(t) = \frac{\big(\mu_T\,\omega(t) - m(t)\big)^2}{\omega(t)\big(1-\omega(t)\big)}$$

which is exactly what is coded, with `m[-1]` as the total mean $\mu_T$. In this form the whole 256-bin search is three cumulative sums and one `argmax`.

The range is `(0, 1.2)` rather than `(0, 255)` because this Otsu is applied to a **ratio** image, not a grey one. `v = gray / BoxMean(gray, r)` sits near 1.0 on paper and near 0 on ink, and can exceed 1 slightly where a pixel is brighter than its own neighbourhood; 1.2 is the clip ceiling. The `+0.5` returns the bin centre rather than its edge.

**`InkMask(gray)` — the binarisation that makes everything after it possible.**

```python
r = max(8,min(40,round(gray.shape[0]/41)))
ratio = gray/np.maximum(BoxMean(gray,r),1.0)
mask = ratio<OtsuThreshold(ratio)
solid = BoxMean(mask.astype(np.float64),4)>0.999
return mask&~(BoxMean(solid.astype(np.float64),3)>0),ratio
```

This is a *relative* threshold, and that is the whole point. A global Otsu on a phone photo fails because one corner of the sheet is in shadow and darker than the ink in the bright corner. Dividing by the local background mean removes the illumination field first, after which one global threshold on the ratio is valid everywhere. It is the division form of adaptive thresholding in the Sauvola/Wellner family.

- `r = height/41` — the blur radius must be large compared with a pen stroke, or the stroke becomes its own background and vanishes, and small compared with the illumination gradient. One forty-first of the page height is roughly one line pitch, which satisfies both. Clamped to `[8, 40]` so a thumbnail or a very large scan still behaves.
- `np.maximum(..., 1.0)` guards a zero local mean in a genuinely black region.
- The `solid` step removes large solid black areas. A 9×9 window (`r=4`) that is 99.9% ink is not handwriting, because handwriting is thin — it is the dark surround outside the sheet, the binder's interior, or a hard shadow. The second `BoxMean(solid, 3) > 0` grows that detection by 7×7 so the whole blob including its soft edge goes, not just the solid core.

`ratio` is returned alongside the mask and reused twice later: for the `faint` test in stage 2b, and as the debug overlay's background.

### A.1.2 `LinePitch(mask)` — measuring `P`

```python
prof = mask[:,int(0.05*W):int(0.97*W)].sum(1).astype(float)
k = np.exp(-0.5*(np.arange(-12,13)/4.0)**2)
sm = np.convolve(prof,k/k.sum(),"same")
```

`prof` is the **row ink profile** — ink pixel count per image row. The 5%/97% column crop discards the left and right margins, where a binder edge or shadow sliver would add a constant to every row and flatten the profile. It is smoothed with a Gaussian of σ = 4 px truncated at ±12 px (±3σ), which suppresses the within-line ripple from individual letters while preserving line-to-line structure.

Three strategies, in strict precedence order:

1. **Direct peak gap (preferred).** Local maxima of `sm` above 6% of the profile peak are line centres, and `P` is the *median* of consecutive differences. Checked against 16 real pages. Autocorrelation is deliberately demoted below this because it can lock onto a multiple of the true spacing on a page with uneven paragraph gaps. Using the median rather than the mean makes one paragraph break harmless, and the 6% floor rejects noise bumps between lines.
2. **Autocorrelation**, used only when fewer than two peaks resolve. The first local maximum of the autocorrelation at lag ≥ 15 that reaches 70% of its own peak. Searching `ac[15 : min(200, len//3)]` sets a 15 px minimum pitch — below that the "lines" would be thinner than a letter — and a one-third-of-page maximum, beyond which a lag cannot be measured from a handful of periods.
3. **Letter size**, for a page with a single text line, where there is no spacing to measure. Components with `area ≥ 8` and height ≥ 3 are kept, and `P = 2.2 × median height`. A text line occupies roughly an x-height while the pitch also contains ascender, descender and leading, so an x-height-dominated median letter height times 2.2 lands close to the real pitch.
4. `return 48, sm` when there is no usable ink at all — arbitrary, and it only has to be harmless, since the line loop will find nothing anyway.

### A.1.3 Stage 1 — deskew

```python
best = max(np.arange(-4,4.01,0.5),
           key=lambda a: float((np.asarray(q.rotate(a)).sum(1).astype(float)**2).sum()))
```

The rotation angle is the one that **maximises the sum of squares of the row ink profile**. That criterion is exactly right: for a fixed total ink $\sum_y p(y)$, the quantity $\sum_y p(y)^2$ is maximised when the ink is concentrated into as few rows as possible, which happens when the text lines are horizontal. The same sharpness functional reappears in `EstimateSlantDeg` for shear.

The range is ±4° in 0.5° steps, since a hand-held photograph of a sheet on a desk is not out by more than a few degrees. The search runs on `q`, a *quarter*-resolution mask, because the criterion is a coarse 1-D statistic and seventeen rotations are being tried. Both `full` and `half` are then rotated by the winning angle with `fillcolor=255`, and the mask is **recomputed** on the rotated image rather than rotated itself, so no interpolation artefacts enter the binary mask.

### A.1.4 Stage 2 — removing what is not handwriting

**`PaperRegionMask(g, P)`** finds the sheet. It decimates by `s=4` (mean over 4×4 blocks), blurs by `0.15*P/s`, and calls anything below `0.45 × median` brightness "dark surround". If less than 0.2% of the page is dark it concludes there is no surround and returns all-ones — the common case of a scan, or a photo that is all paper. Otherwise it labels the *complement*, keeps the largest component as the sheet, and shrinks it by `0.4*P` on both axes so the paper's own edge shadow falls outside. Taking the largest bright component rather than a bounding box is what lets it handle a sheet photographed at an angle on a dark desk.

**Straight-stroke removal** does the most work in the file:

```python
flat     = AnyWithinRadius(LongRunMask(AnyWithinRadius(mask,1,0),int(1.25*P),1),1,0)&mask
upright  = AnyWithinRadius(LongRunMask(AnyWithinRadius(mask,3,1),int(1.4*P),0),3,1)&mask
mask = mask&~flat&~upright
```

Read `flat` inside-out: dilate vertically by 1 px so a *slightly tilted* horizontal line still contains an unbroken horizontal run; find horizontal runs of at least `1.25*P`; dilate the result vertically by 1 to recover the line's full thickness; intersect with the original mask so only real ink is removed. A horizontal ink run longer than 1.25 line pitches cannot be a letter — the longest plausible letter is a fraction of a pitch — so it is an underline, a ruled line, or the sheet's edge. `upright` is the same construction transposed, with a 3 px tolerance because a page edge or margin rule in a hand-held photo wobbles more vertically than a printed horizontal rule does, and a threshold of `1.4*P` because a tall letter plus its descender can approach one pitch.

**Stage 2b — page-boundary artefacts.** First the mask is *closed* with radius `rc = 0.1*P` and that closed version labelled, so a letter broken into pieces by threshold noise counts as one component for ownership purposes. Then per component:

| Test | Definition | What it is |
|---|---|---|
| `tall` | height > `2.2*P` | spans more than two lines, so not a letter |
| `thin` | width ≤ `0.5*P` | narrower than half a pitch |
| `touchL/R/T/B` | within `edgeTol = max(3, 0.5*P)` of a page border | candidate page furniture |
| `vbar` | width ≤ `0.12*P` **and** height ≥ `1.2*P` | thinner than any letter and taller than any letter: a margin rule |
| `faint` | mean `ratio` > `0.85 × otsuT` | only just crossed the ink threshold; real pen ink is far darker |
| `faintVbar` | `faint`, width ≤ `0.2*P`, height ≥ `0.6*P` | a printed margin rule or a fold crease |
| `edgeShadow` | touches top/bottom, height ≤ `1.2*P`, `density < 0.15`, `faint`, width > `P` | a fold or crease: wide, sparse, pale, at the edge |
| `xmax < xL - 1.2*P` | ends more than a pitch left of the text block's left margin | a page-left artefact outside the writing |

`density = area / (bbox height × bbox width)` is what separates a crease — a thin line through a large box, so sparse — from a letter, which is dense in its own box. `xL = np.percentile(xmin[letters], 5)` establishes the text block's left margin from the letters themselves, using the 5th percentile rather than the minimum so one stray mark cannot move it. `drop[0] = False` protects label 0, the background.

The drop rule is deliberately asymmetric: `touchL` drops unconditionally, while `touchR` drops only together with `thin`. The binder and edge artefacts in this rig are consistently on the left, while real writing frequently runs right up to the right margin.

### A.1.5 Stages 3–4 — line centres and seams

`P` and the smoothed profile are recomputed on the *cleaned* mask, which matters because rule lines and page edges would otherwise contribute spurious profile peaks. Centres are picked greedily from the strongest peak downwards with a `0.6*P` exclusion radius, so one line's ascender peak and baseline peak cannot both become centres.

**`LeastCostSeam(cost, step)` — dynamic programming for the minimum-cost left-to-right path.** With $C(y,x)$ the cost image, the recurrence is

$$A(y,x) = C(y,x) + \min_{|d| \le \text{step}} \Big( A(y+d,\,x-1) + 0.02\,|d| \Big)$$

and the path is recovered by backtracking from `argmin A(·, W-1)`. This is the seam-carving formulation of Avidan & Shamir (2007) used as a text-line separator: the cheapest path through an ink-cost image threads *between* lines rather than through letters. The `0.02·|d|` term penalises vertical movement, so the seam runs straight unless deviating genuinely saves ink crossings — without it the seam wanders through equal-cost whitespace and produces ragged boundaries.

The cost image handed to it is the most carefully reasoned expression in the file:

```python
mdil    = AnyWithinRadius(AnyWithinRadius(mask,tailR,0),tailR,1)
mClosed = ~AnyWithinRadius(AnyWithinRadius(~mdil,tailR,0),tailR,1)
...
cuts.append(c0+LeastCostSeam(4.0*m[c0:c1+1]+0.012*((yy-mid)/half_w)**2, step=7))
```

- **Closing, not dilation** (`tailR = max(2, 0.07*P)`). Closing fills the thin gap a seam could otherwise thread through beside a letter stem, while leaving the stroke's outer shape close to real, so the cut traces the letter instead of a bloated blob around it. Plain dilation would also fill the gap but would inflate every stroke, and the seam would then detour around a fat halo and clip the neighbouring line.
- `4.0 * m` — ink costs 4 units per pixel against the positional term's maximum of 0.012. Crossing ink is strongly discouraged but not forbidden, which is necessary because on a dense page some crossing is unavoidable and an infinite cost would make the problem infeasible.
- `0.012*((yy-mid)/half_w)²` — a weak quadratic pull towards the midline between the two centres, normalised to equal 0.012 at either centre. It keeps the seam tracing real ink instead of collapsing to a generic V shape: with no prior a seam in a blank region is unconstrained, and with too strong a prior it ignores ink and cuts straight through. At 1:333 against the ink term, the prior only decides between near-tied paths.
- `step=7`, against the function's own default of 3. A bigger per-column step lets the cut react fast enough to get around a tall narrow descender; at half resolution with `P` ≈ 40–60 px, three rows per column cannot.

The first and last cuts are flat lines `0.75*P` above the first centre and below the last.

### A.1.6 Stage 5 — deciding which line owns which ink

`region` is the integer line index per pixel, built by summing step functions: `region = Σ_n (Y >= cuts[n])`, which evaluates to `n` between cut `n` and cut `n+1`. Ownership starts as "whichever band the pixel is in" and is then corrected by a series of rules, each handling one specific class of mistake. This is where most of the file's accumulated empirical knowledge lives.

**Majority band, biased upward.** `cnt[c, n]` is component `c`'s pixel count in band `n`. A band is `substantial` if it holds ≥ 15% of the component's ink. For a component substantial in more than one band, `major = regionIdx.min(1)` — the *topmost* such band, not the largest. When a piece's ink genuinely spans two lines it is reserved for the topmost line it touches, because that is the crop that goes wrong if it loses the piece: a descender belongs to the line above, and giving it to the line below both corrupts the lower crop and robs the upper one.

**`minor` fragments rejoin their parent.** A piece of a component sitting in a neighbouring band (`|n - major| == 1`), smaller than `0.12*P*P`, holding under 35% of the component's ink, and no wider than `0.55*P`, is a cut-off tail. The width test is the discriminating one: a cut-off tail is narrow, while a word-wide part is its own line.

**Strays.** Components with total ink below `0.06*P*P` are examined individually. `near` is the set of substantial components (`≥ 0.05*P*P`) assigned to the same line within `6*P` horizontally. With no such writing nearby, the stray is dropped (`major = -1`). Otherwise its y-centre is compared against the local median y-centre of that line's real writing, and if it sits more than `0.5*P` below or `0.7*P` above it moves to the adjacent line — or is dropped outright if it is a `speck` under `0.01*P*P`. The asymmetry between 0.5 and 0.7 reflects that descenders legitimately hang further below a line than anything rises above it. A second pass handles the `[0.06, 0.15]*P*P` band more gently: always reassign to the nearest of the three candidate lines by median y, never drop.

**Dots follow their stems.** A component is a `dot` when its area is under `0.1*P*P`, its larger dimension is at most `0.45*P`, and it is roughly round (`max(h,w) ≤ 2.2 × min(h,w)`). For each dot, walk downward up to `0.4*P` looking for a `stem` component in the columns beneath it; if one is found, adopt that stem's line, but only when the geometry agrees — an *i* dot floats 0.25 to 0.7 pitches above its line's centre. Without this, a dotted *i* on the first line of a paragraph is frequently given to the line above, because the dot sits above the seam.

**The two connected-dot cases.** These handle a dot that has become *physically joined* to the line above, and they run on `raw` (the un-closed mask) alongside `lab` (the closed one), which is why both labellings are kept.

- An *i* dot touching a descender from the line above is a single component in the closed mask, so the dot rule above cannot see it. This step finds the dot in the *raw* mask, finds the stem below it, and moves only those pixels. `wholeWord = area[closedPiece] >= 0.2*P*P` prevents it firing on a genuinely large component, `(rt+rb)/2 >= centres[n1]+0.45*P` requires the dot to sit in the lower part of its current band, and `keep = ov == n1` moves only the pixels currently owned by the line above.
- The same situation where the dot has merged into a *ring-shaped* blob at the bottom of a descender. The code walks up from each column's lowest pixel while the blob stays wide, finds where it narrows (`wd < 0.45*widest`) to locate the stem junction, requires roundness (`0.2*P ≤ widest ≤ 0.45*P`, `0.6 ≤ h/widest ≤ 1.6`), and crucially requires the blob to **enclose a hole**:

  ```python
  if LabelConnectedComponents(~np.pad(blob[...],1))[1]<2:
      continue   #a dot ring encloses a hole, a hook does not
  ```

  A dot ring has an interior background component; a descender hook does not. That topological test is what separates the two cases.

**Fragments with no ink in their own line's x-height band.** For each line, its assigned pixels are re-labelled and components found that have *zero* pixels within `0.28*P` of the line centre and size ≥ `0.006*P*P`. Such a component cannot be one of this line's letters. It moves to the line whose ink it actually touches — found by dilating by 2 px and taking the majority neighbour, requiring at least 3 touching pixels — with a direction-consistency check `(m>n)==(cy_>centres[n])`; failing that, to whichever neighbour's x-height band it sits in, within `0.2*P`. `locked` records pixels already placed by the two dot rules so this step cannot undo them, guarded further by a size and extent test so only a small dot is protected, not a tail loop.

### A.1.7 Stage 6 — building the crops

A line's crop is not a plain rectangle of the original image; it is composited:

1. **Illumination-normalised page.** `BoxMean(g, r)` (the half-res background) is upsampled to full size and the full image divided by it — the same model as `InkMask`, applied to grey values instead of to a threshold. `r` uses the identical `max(8, min(40, H/41))` rule so the two agree.
2. **Contiguity trim.** Row runs of this line's ink are found and only the tallest-ink contiguous run is kept — but only when the discarded part exceeds `0.8*P`, so a legitimate gap between an ascender and the body is never trimmed, while a stray mark a paragraph away is.
3. **Alpha matte.** `alpha` is the line's own ink, resized bilinearly to full resolution, then *extended* by `alpha = max(alpha, localMask * haloAlpha * notOtherFull)`:
   - `localMask` re-thresholds the crop region with its own local Otsu at radius `max(3, min(20, h/8))` — this line's own contrast rather than the whole page's — which recovers a faint letter the page-level threshold missed;
   - `haloAlpha` restricts that recovery to within `0.20*P` of ink this line already owns, so it cannot pull in a neighbour's stroke;
   - `notOtherFull` (resized with `NEAREST`, so ownership labels are never interpolated) forbids recovering any pixel the page already assigned to another line.

   Together: be greedy about faint ink, but only locally and only where unclaimed.
4. **Contrast stretch and gamma.** `black = percentile(inkPix, 40)` is typical ink darkness, robust against the single darkest pixel; `white = percentile(region, 75)` is paper. After the linear stretch, `darkness**0.55` applies a gamma below 1, compressing the bright end and expanding the dark end so a 20%-dark stroke comes out around 40% — which pulls faint strokes much darker.
5. **Left-margin normalisation.** Every crop is shifted so its first ink column sits exactly `pad*f` from the left edge. This matters downstream: the classifier sees a fixed 64×640 canvas, and a varying left offset would be a nuisance variable it had to learn to ignore.

`ToClassifierFormat` then fits the crop into exactly 64×640 using a **single** scale factor `s = min(64/h, 640/w)` for both axes — never an anisotropic stretch, because aspect ratio is a style feature the writer-ID classifier depends on — pasted at the left edge, vertically centred, padded white.

The returned `info` dict (`angle`, `pitch`, `paperCoverage`, `inkOutsidePaper`, `removedStraightPixels`, `artifactPieces`, `nLines`, `seconds`) is the per-page diagnostic record, and `_cuts_overlay.jpg` draws the seams in alternating red and blue over the ratio image with the non-paper region tinted. On a real personal page this reports, for example, `angle -0.5°, pitch 48, paperCoverage 0.976, removedStraightPixels 6376, artifactPieces 34, nLines 26` in 6.5 s.

---

## A.2 `ProfileIO.py` — the shared binary-image toolbox

This is the subset of the style-profile machinery that a *live request* needs, kept separate from `BuildStyleProfile.py`'s offline glyph-library code. `Charset.py` exists for the same reason one level down: `CHARSET` is needed by both `ProfileIO.CtcForcedAlign` and `Classify`, and putting it in either would create an import cycle.

### A.2.1 Integral-image morphology

`BoxIntegral`/`BoxSum`/`BoxCount` are the summed-area table again, but with **edge-clamped** windows instead of padding: `BoxSum` clips the window corners into range, and `BoxCount` returns the *actual* number of pixels in each clamped window. That pair is what makes erosion correct at the border:

```python
def Dilate(mask,ky,kx):  return BoxSum(mask,ry,rx) > 0.5                     # any pixel set
def Erode(mask,ky,kx):   return BoxSum(mask,ry,rx) >= BoxCount(...) - 0.5    # all set
def Close(mask,ky,kx):   return Erode(Dilate(mask,ky,kx),ky,kx)
```

Erosion as "the window sum equals the window's own pixel count" is exact for a clamped window, where comparing against a fixed `k*k` would erode the entire border away. The `±0.5` are float-comparison slack on integer-valued sums.

### A.2.2 `LabelComponents` — run-based union–find

`RowRuns` reduces each row to `(starts, ends)` column index arrays by differencing the padded mask. Then for each adjacent row pair the two sorted run lists are merged in one linear sweep — the two-pointer walk, advancing whichever run ends first — and overlapping runs unioned:

```python
if s1[i] < e2[j]+pad and s2[j] < e1[i]+pad:
    uf.union(runIdStart[r-1]+i, runIdStart[r]+j)
```

`pad = 1` for 8-connectivity and `0` for 4-connectivity: the one-pixel slack is exactly what diagonal adjacency means in run terms. This is the Rosenfeld–Pfaltz two-pass scheme with runs instead of pixels, so the union–find holds one node per *run* rather than per pixel — typically two orders of magnitude fewer — and `RunUnionFind.find` does path compression by halving (`p[a] = p[p[a]]`), giving near-constant amortised lookups.

`SegmentLean.LabelConnectedComponents` is the same algorithm with the union–find inlined as a `parent` list and a `find` closure, additionally re-indexing labels densely via `np.unique(roots, return_inverse=True)` so the returned count is exact. The two files are deliberately decoupled, so each carries its own copy.

### A.2.3 `Skeletonize` — Zhang & Suen (1984), vectorised

The classic two-subiteration thinning. `neighbours(P)` returns the eight shifted views $P_2 \ldots P_9$ clockwise from north, obtained by padding once and slicing, so all eight neighbour arrays come with no Python loop over pixels. Per subiteration a pixel is deleted when all of:

- $2 \le B(p) \le 6$, where $B = \sum_{k=2}^{9} P_k$ counts set neighbours. The lower bound protects line endpoints; the upper bound protects interior pixels.
- $A(p) = 1$, where $A$ counts 0→1 transitions around the ring — hence `ring = [...P9, P2]`, closing the circle. This is the **connectivity-preservation** condition: deleting a pixel with exactly one transition cannot disconnect its neighbourhood. It is the heart of the algorithm.
- step 0: $P_2P_4P_6 = 0$ and $P_4P_6P_8 = 0$. Step 1: $P_2P_4P_8 = 0$ and $P_2P_6P_8 = 0$. The two subiterations strip from opposite sides, and alternating them keeps the skeleton centred instead of eroding it towards one corner.

Iterating until no pixel changes leaves a one-pixel-wide centreline with the same topology as the input.

### A.2.4 `BinarizeLine`, `CoreBand`, `StrokeNormalize`

`HorizontalRunLength` gives every ink pixel the length of its horizontal run, by labelling runs with a cumulative sum of run-starts and then a `bincount` lookup — O(pixels), no loop. Applied to the transpose it gives vertical run length.

`BinarizeLine` thresholds with Otsu and then removes rules:

```python
rule = (hRun >= max(24, int(1.1*h))) & (vRun <= max(3, int(0.12*h)))
```

Long horizontally (≥ 1.1 × the crop height) **and** thin vertically (≤ 12% of it) — the same "a rule is long and thin, a letter is not" logic as `LongRunMask`, but expressed per pixel and scaled to the crop rather than to `P`. Leaving rules in would glue every letter together and drag the slant estimate flat. The intersection with `vRun <= ...` is what preserves a real stroke *crossing* a rule: at the crossing the vertical run is long, so those pixels survive.

The closing kernel `k = clip(round(0.045*h), 2, 4)` repairs the hairline breaks a fast hand leaves in a scan, capped at 4 px so neighbouring letters do not merge. Components under 6 pixels are then dropped as speckle.

`CoreBand(ink)` returns the x-height band as the first and last rows whose ink count is ≥ 45% of the peak row's. The row profile of a text line has a characteristic plateau between baseline and x-height line, where most strokes are, with thin tails for ascenders and descenders; 45% of peak cuts in the tails' shoulder. It returns `None` when the band is under 3 px, i.e. unusable.

`StrokeNormalize` is the function that makes writer identification about *shape* rather than *pen*:

```python
ink = BinarizeLine(gray)
xh  = band height (or a fallback: max(8, h/3))
w   = clip(round(penWidthXh*xh), 2, 9)
sk  = Skeletonize(ink)
out = Dilate(sk, 2*(w//2)+1, 2*(w//2)+1)
```

Real ink → centreline → re-inked at **one constant width** → inverted back to a white-paper image. `PEN_WIDTH_XH = 0.14` makes the re-inked stroke 14% of the x-height, clamped to 2–9 px, and `k = 2*(w//2)+1` forces an odd kernel so the dilation is centred.

This is the same transform the gantry performs when it redraws a hand with its own pen. The classifier has to identify a writer from geometry that survives being redrawn by one 0.5 mm pen at constant pressure, so training and inference both see stroke-normalised images and the author's own pen width and pressure variation are removed as nuisance variables.

### A.2.5 `EstimateSlantDeg` — shear search

```python
for deg in np.arange(-45,45.5,1.5):
    xsh = (xs + (ys-yc)*-tan(radians(deg)))
    score = (bincount(xsh - xsh.min())**2).sum()
return -best
```

The same sharpness functional as the deskew, applied to the *column* projection under a shear about the ink's own vertical centroid `yc`. A sheared column histogram is sharpest when the shear has made the dominant strokes vertical, which is the definition of the slant. The search runs over the ink coordinate lists rather than the image, so each of the 61 candidates is one `bincount`. The sign convention matters downstream: positive is a normal forward lean, so deslanting is `x -= tan(slant)*y` with y measured up from the baseline — which is why the function returns `-best`.

### A.2.6 `CtcForcedAlign` — Viterbi over the CTC lattice

This is the bridge from recognition to style extraction. Given per-frame log-probabilities and the *known* text, it returns the timestep span of each character.

The label sequence is blank-interleaved into `ext = [0, l₁, 0, l₂, 0, ..., 0]` of length `S = 2L+1`, the standard CTC extension. Then a **max**-product recursion (Viterbi, 1967) runs where CTC training uses a sum:

$$\delta_t(s) = \log p_t(\text{ext}[s]) + \max\begin{cases} \delta_{t-1}(s) & \text{stay} \\ \delta_{t-1}(s-1) & \text{advance one} \\ \delta_{t-1}(s-2) & \text{skip a blank} \end{cases}$$

The skip is forbidden when `ext[s] == 0` — you cannot skip *into* a blank — and when `ext[s] == ext[s-2]`, because two identical labels must be separated by a blank or they collapse under CTC's repeat-merging rule. Those two conditions *are* the CTC topology, and they are the same two conditions as in `NumpyNN.CtcLoss`: one file takes the max over the predecessor set, the other the log-sum-exp. The forward–backward and the forced alignment are the same lattice under different semirings.

Backtracking `s -= bp[t,s]` recovers the best path, and each character's span is the first and last frame at which the path occupies its odd position `1+2i`. A character the path never occupies gets `None`, which happens legitimately when the recogniser compressed two characters into one frame.

`conf` is the mean per-frame log-probability along the chosen path. That is what `BuildStyleProfile`'s `ALIGN_CONF_PCT = 25` thresholds against, discarding the worst-aligned quarter of an author's lines before harvesting glyphs.

Guards: characters outside the charset are dropped (`if c in CHAR_TO_IDX`), and `T < len(labels)` returns `None` because a path cannot emit more labels than there are frames. `NEG = -1e30` is a finite sentinel rather than `-np.inf`, since this is a max recursion where a finite value keeps `argmax` well-defined.

---

## A.3 `ExtractIAMLines.py` — offline dataset construction

Run only to build training data. Its single entry point `ExtractLinePatches` has two modes, and the difference between them is the point of the file.

**`isDataset=True` (IAM pages).** An IAM form has a printed prompt box fenced top and bottom by horizontal rules. `FindRuleLineClusters` finds rows whose longest ink run exceeds 25% of the page width *and* whose total ink exceeds 10% of it — the conjunction matters, since one long run with little other ink is a crease, while much ink with no long run is text. `ClusterRows(maxGap=8)` collapses the several rows of one thick printed rule into a single centre. `DetectPageBounds` then searches restricted bands — the top rule within 12–40% of page height, the bottom within 55–92% — taking the lowest top candidate and the highest bottom one, and falling back to fixed 12%/80% guesses if a band holds no rule.

Lines are found by a blank-gap detector: `hasInk = rowInkCount > max(10, 0.006*w)`, dilated by 7 rows (the `for dy in range(7)` OR-loop) to bridge within-line gaps, then differenced to get run boundaries. Runs shorter than 8 px are dropped, and consecutive runs closer than 18 px are merged to reattach descender fragments from *y*, *p* and *g*.

**`isDataset=False` (personal ruled paper).** Header and footer detection is skipped entirely, because it actively misfires on ordinary ruled paper, and line splitting switches to `SplitLinesByPeriodicity`. That smooths the row ink profile with a 9-tap box, estimates the pitch by autocorrelation (`EstimateLinePitch`, lag search 40–400), and takes

```python
find_peaks(-smoothed, distance=max(30, 0.55*pitch), prominence=max(1, 0.03*max))
```

— the *minima* of the ink profile, spaced at least 55% of a pitch apart. The switch is necessary because a densely written cursive page has no blank row between lines, since ascenders and descenders interleave, so a blank-gap detector finds one enormous line. `ruleLineHints` optionally snaps a detected valley to a nearby printed rule within `0.12*pitch`. This is the only place in the pipeline that calls SciPy, and only for peak finding.

**Labels.** Per-line labels come from `ReadLabelLines`, a hand-verified transcript file, matched to crops by index. `ReadLabelLines` strips a leading `^\d+\t` index so a line number can never leak into a training target as a literal digit. The transcript is returned alongside the crops only when the two agree on line count, so a caller can never pair *n* crops with a different number of label lines.

Two count-reconciliation helpers run before that comparison:

- `MergeUndersizedFragments` merges any detected "line" shorter than 40% of the median height into its closer neighbour — a descender tail or stray mark, distinguishable from a real line by height alone. Neighbour choice is by the smaller gap.
- `MergeOrSplitToExpectedLineCount` forces the detected count to match the transcript's: merge the pair with the smallest `gap + mean gap ink`, which is the cheapest boundary to erase, or split the tallest line (minimum 42 px) at its thinnest-ink row, staying 18 px clear of either end.

Output geometry: crops are resized to `targetHeight=32` preserving aspect ratio, capped at `maxWidth=1024`, and pasted left-aligned onto a white 1024×32 canvas. These are archival line crops; `TrainText.ResizeLineImageFixed` does the final 64×640 fit at load time. The preview image draws the detected bounds in blue, the handwriting start line in cyan, and a numbered red box per returned line.

On a real IAM page this yields 8 crops against 8 transcript lines, the first reading *"The film version of Miss Shelagh Delaney"*.
