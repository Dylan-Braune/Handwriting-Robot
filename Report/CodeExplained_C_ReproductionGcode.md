# Block C — Style extraction, synthesis and G-code

This block answers two questions: *what does this person's handwriting look like, as data?* and *given that data and a new sentence, what pen path should the robot draw?*

| File | Lines | Role |
|---|---|---|
| `BuildStyleProfile.py` | 1168 | **Offline.** Turns an author's real pages into a style profile: a glyph library plus measured style parameters. Run by hand. |
| `SynthesizeHandwriting.py` | 1391 | Text + profile → pen trajectory → G-code. Both halves of the output path. |
| `Evaluate.py` | 945 | The measurement harness. |

The data flowing between them is one JSON file per author in `NOGIT/StyleProfiles10/<author>.json`.

---

## C.1 The idea

The recogniser and writer-ID classifier of Block B *read* handwriting. To *write* it, the system needs a generative model of a hand. The route taken here is explicit and geometric rather than learned end-to-end:

1. Take the author's real line images and their known transcripts.
2. Use the frozen recogniser plus **CTC forced alignment** to find where each character sits horizontally — which works even in cursive, where there is no visible gap between letters.
3. Skeletonise the ink, trace the centrelines into polylines, cut them at the character boundaries, and normalise each piece into a slant-removed baseline/x-height frame. That is a **glyph variant**.
4. Keep several variants per character, so repeated letters differ the way a real hand's do.
5. Measure the global parameters a hand has beyond its letterforms: slant, x-height, word spacing, stroke width, ascender and descender reach, and connectedness.
6. To synthesise, place variants along a baseline by the author's own measured advances, join them with ligatures at the author's own measured join rate, jitter each instance, and shear the whole thing by the author's slant.

The output is a list of pen-down polylines in millimetres, which is already the right representation for a plotter.

---

## C.2 `BuildStyleProfile.py` — from pages to a profile

### C.2.1 Tracing a skeleton into strokes

`Skeletonize` (Block A) returns a one-pixel-wide centreline as a bitmap. `TracePolylines` turns that bitmap into *ordered* polylines, which is what a pen path needs.

It builds a pixel graph over the 8-neighbourhood, with one refinement:

```python
#a diagonal step is redundant when an orthogonal 2-step path exists
if dy and dx and ((y+dy,x) in pts or (y,x+dx) in pts):
    continue
```

Suppressing redundant diagonals prevents a staircase in the skeleton from registering as a dense web of short edges, which would fragment the walk.

Then `deg[p]` is each pixel's neighbour count, and the walk starts from every pixel with `deg != 2` — endpoints and junctions — following `deg == 2` pixels until it reaches another endpoint or junction. That yields one chain per maximal simple path. Pure loops, which have no `deg != 2` pixel anywhere, are picked up in a second pass over whatever is left uncovered.

Two cleanups follow:

- **`_PruneSpurs(chains, minLen=2.5)`** drops chains shorter than 2.5 px that dangle off a junction (one of their endpoints is used by no other chain). These are thinning artefacts — the small barbs Zhang–Suen leaves at a stroke's shoulder — not pen strokes. `or chains` at the end guarantees it never returns nothing.
- **`_JoinAtJunctions(chains, maxAngleDeg=75)`** repeatedly merges the best-aligned pair of chain ends that are within 1.5 px of each other:

  ```python
  di = _Dir(chains[i],iEnd); dj = _Dir(chains[j],jEnd)
  cosang = float(-np.dot(di,dj))        # continuing straight through means di ≈ -dj
  ```

  `_Dir` is the unit direction of a chain's last 4 points, pointing outward. Two chains that continue in nearly the same direction through a junction are one pen stroke crossing itself — the loop of an *e*, *l* or *o*, or a *t* crossbar — so joining them keeps one pen stroke as one polyline. The greedy loop always takes the best-aligned pair first, so the clearest continuation is resolved before any ambiguous one.

**`SimplifyPolyline(poly, eps=0.6)`** is Douglas–Peucker, with one addition:

```python
fromStart = np.hypot(*(p-p[0]).T)
if fromStart[-1] <= max(1.5, 0.25*fromStart.max()):
    far = int(np.argmax(fromStart))
    # split at the farthest point, simplify each half
```

On a **closed** curve the chord from first to last point has no length, so every point's perpendicular distance to it is near zero and plain Douglas–Peucker collapses the whole loop to a single point. Detecting that the endpoints nearly coincide and splitting at the point farthest from the start gives two open halves, each of which simplifies correctly. Without this, every *o*, *e* and *a* loop would vanish.

### C.2.2 `ExtractLineGlyphs` — the core of the file

Given one line image, its transcript, and the frozen recogniser, this returns one glyph per character plus the line's style statistics.

**Step 1 — align.** The line is fitted to the 64×640 canvas, passed through the recogniser, and `CtcForcedAlign(logProbs, text)` (Block A) returns a `(t0,t1)` frame span per character plus an alignment confidence.

**Step 2 — measure the frame.** `BinarizeLine` → `CoreBand` gives `top, base` and hence `xh = base - top`; `EstimateSlantDeg` gives the slant and `shear = tan(slant)`. Every glyph will be stored in this normalised frame.

**Step 3 — convert frame indices to pixel columns.** `FrameXToPixel(sp/T*INPUT_WIDTH, rawW, origH)` undoes the fixed-canvas fit, so a CTC frame index becomes a column of the real crop.

**Step 4 — snap the boundaries to honest cut columns.** This is the part that makes cursive work. A raw CTC span boundary can fall in the middle of a letter, so instead of cutting there the code builds a per-column cut cost and takes the cheapest column nearby:

```python
colSm   = smoothed column ink profile, kernel ~0.12*xh
thin    = colSm / 60th-percentile(colSm>0)        # how thick the ink is here
topAbove= clip((base - highest ink row)/xh, 0, 3) # how high the ink reaches here
cutCost = thin + 1.15*topAbove
cutCost[colProf==0] = 0.0                         # a true gap is always a fine cut
```

A good cut is **thin** (one stroke wide, not through a letter body) and **low** (near the baseline, not through an ascender). In cursive, that describes exactly the ligature crossing between two letters. A genuine blank column scores zero and always wins. `_Snap` searches `±0.5*xh` around the proposed boundary and adds `0.6 * |offset|/snapR`, a mild preference for staying near the recogniser's own estimate so the snap cannot wander to a better-looking column belonging to a different letter.

Boundaries are then built as **one shared, strictly increasing array**, with each cut at least `minCell = max(1, 0.08*xh)` past the previous one and leaving room for the remaining cells. Snapping each boundary independently could put two neighbours on the same column and produce an empty cell.

**Step 5 — skeletonise once, then cut.**

```python
skelAll  = Skeletonize(ink)
polysAll = TracePolylines(skelAll)
```

The whole line is skeletonised once and the *traced centrelines* are then cut at the boundaries. Thinning each column slice separately would bake a spurious vertical spine along every cut, because the slice's own edge looks like a stroke edge to the thinning algorithm.

Each traced point is assigned to whichever character cell contains its x, and a polyline is split wherever that assignment changes. Each split is also recorded in `crossings`, which is the honest definition of connectedness: **a centreline crossing a character boundary means the pen never lifted there.**

**Step 6 — recover dots.** A dot thins to a single skeleton pixel, which `TracePolylines` cannot make a polyline from, so it would disappear. Any connected component the tracer produced nothing for, with at least `DOT_MIN_AREA*xh*xh` pixels, is re-added as a short horizontal dash across its own width at its own centre.

**Step 7 — normalise each glyph.**

```python
yn = (base-y)/xh          # y up from the baseline, in x-heights
xn = (x-left)/xh          # x from this character's own left cut, in x-heights
xn = xn - shear*yn        # de-slant
```

So every glyph is stored upright, in x-height units, with its origin at its own cell's left edge. Strokes are then oriented left-to-right and sorted by starting x, giving a canonical order.

**Step 8 — trim ligature tails.** `_TrimLigatureTails` removes the connector that belongs to the *neighbouring* letter, and only on a side the `crossings` set has **confirmed** is joined — never guessed from shape. It walks inward from the stroke's end while the path keeps moving outward below a height ceiling:

```python
def runLen(s,connected,maxHeight,maxSlope):
    while i+1<len(s):
        if s[i][1]>=maxHeight or dx<0 or abs(dy)>maxSlope*max(dx,1e-6): break
```

Two ceilings are tried: a low, near-horizontal carrier (`lowY=0.42`, slope 0.7) and a steeper one for a connector climbing into a tall letter (`climbY=1.6`, slope 2.2). A trim is only accepted if what remains still spans more than `minKeep=0.18` of the glyph's width, so the test can never eat the letter. The exit connector is handled by mirroring x and reusing the same test. The function returns the trimmed strokes plus the **entry and exit points**, which synthesis later uses to draw its own connector.

**Step 9 — re-align and store.** Since the glyph will be drawn *re-slanted*, the x-origin is recomputed in the re-slanted frame (`xr = px + shear*py`) so placement is correct after the shear is reapplied. The stored record is:

| Field | Meaning |
|---|---|
| `strokes` | the polylines, normalised frame |
| `width` | ink extent in x-heights |
| `advance` | how far the pen moves on to the next character, from the cell width |
| `lead` | gap from the cell's left edge to where ink starts — preserves the author's own letter gaps instead of butting glyphs together |
| `top`, `bot` | highest and lowest y, in x-heights |
| `entryX/Y`, `exitX/Y` | where the pen entered and left |
| `connL`, `connR` | whether the centreline actually crossed into the neighbour |

The line's statistics are `slant`, `xh`, `alignConf`, and `strokeW = inkArea / totalSkeletonLength` — mean stroke width derived from the ratio of area to centreline length, which is the natural estimator.

### C.2.3 Filtering variants

A cut can go wrong, so each character's pool of variants is pruned hard. Several independent tests run in sequence.

**Alignment confidence.** Lines below the `ALIGN_CONF_PCT = 25`th percentile of that author's own alignment confidences are discarded whole. If the recogniser could not read the line, its character boundaries are not trustworthy. The percentile is relative to the author, so a hand that reads poorly overall is not thrown away entirely; a floor of 8 lines applies before the percentile is used at all.

**Geometric gates**, per variant:

```python
0.05 < width < wMax                                    # wMax 2.1, or 2.9 for m/w/M/W
(top-bot) > 0.25                                       # has some vertical extent
len(strokes) <= 6
_ClassOk(ch, top, bot)                                 # typographic class
lead+width <= 1.45*advance + 0.25                      # ink not far wider than its cell
_PenLength(g) >= penFloor                              # enough pen path
_LongestStroke(g) >= MIN_LONGEST_STROKE                # 0.55
```

- `_ClassOk` applies a typographic height class: an x-only letter (`acemnorsuvwxz`) must stay within `top ≤ 1.42` and `bot ≥ -0.42`; an ascender (`bdfhklt` and all capitals and digits) must reach `top ≥ 1.25`; a descender (`gjpqy`) must reach `bot ≤ -0.20`. An "l" that is only x-height tall is a bad cut, not an *l*.
- `lead+width <= 1.45*advance + 0.25` catches a cut that swallowed part of a neighbour: ink much wider than the recogniser's own cell means the boundary was wrong.
- `penFloor = max(MIN_PEN_ABS, FRAGMENT_PEN_FRAC * median pen length of this character)` is relative **to the character**, because a dotless *i* has far less pen path than an *o*. A fixed floor either kept neighbour-bleed or threw away real short letters.

**Width and height consistency.** `wRef` is the **40th percentile** of the surviving widths — a low percentile rather than the median, so a doubled letter cannot inflate the reference. Variants within `[0.6, 1.5]×wRef` and `[0.55, 1.7]×hMed` are kept, but only if at least 3 survive, otherwise the wider set is retained.

**The cross-author letter prior.** This is the strongest filter and the reasoning behind it is worth stating: one author's handful of pages cannot separate a well-formed letter from a bad cut, but thirteen authors pooled together can.

`_ShapeGrid(g, n=12)` is a cheap shape descriptor: the glyph's strokes are densely sampled into a 12×12 occupancy grid over the glyph's *own* bounding box, `sqrt`-damped so a long dwell does not dominate, blurred with a 3×3 kernel (centre weight 0.25, others 0.09375) so small shifts between two writings of a letter are tolerated, then L2-normalised. Two descriptors are compared by cosine distance, so `1 - dot(a,b)` is a shape distance in `[0,2]`.

`BuildLetterPrior` pools **every** author's raw variants per character and takes the element-wise **median** descriptor (robust to the fragments in the pool), requiring at least `minSamples=12` samples. `_PriorScore(g, ch, prior)` is then that cosine distance, stored on each kept variant as `priorD`.

`PriorFilter` sorts by `priorD` and keeps the best 70%, with a floor of 3. The prior is used **only to reject an author's malformed variants, never to replace their letterforms** — it decides which of their own writings to trust, not what the letter should look like.

**Internal consensus.** `_ConsensusFilter` computes the pairwise L1 distance matrix between the surviving descriptors, finds the **medoid** (the variant minimising total distance to all others), and keeps the closest 75%. This removes variants contaminated by a neighbouring letter that the prior happened to tolerate.

Finally the survivors are sorted by combined width and height deviation from the reference, truncated to `MAX_VARIANTS_PER_CHAR = 12`, and each is passed through `SmoothStroke`.

**`SmoothStroke(pts, step=0.055, passes=2)`** takes the pixel-grid staircase out of a traced centreline: resample to even arc-length spacing, then apply `0.25, 0.5, 0.25` binomial smoothing twice with the endpoints held fixed. At a low anchor level these variants *are* the drawn output, so the de-staircasing matters directly.

### C.2.4 Two summary letterforms per character

Beyond the variant pool, each character gets two single representatives, used by synthesis as legibility anchors.

**`idealMedoid`** is the single **real** variant most typical of the author's own set — the medoid under the same L1 descriptor distance. It is an intact piece of real writing.

**`idealGlyphs`** is `_DenoisedPrototype`, a robust per-stroke median:

1. Group variants by **stroke count** and take the modal group — averaging a 2-stroke *t* with a 1-stroke *t* is meaningless. Requires at least `minN=3` in the group and at most 5 strokes.
2. Sort each variant's strokes by starting x, and resample each to `N=44` points with `_AlignStroke`, which orients left-to-right and, for a closed loop, rolls the start to the leftmost point so corresponding points line up.
3. Two rounds of **shift-then-median**: each variant is shifted in x over `linspace(-0.22, 0.22, 9)` to whichever offset minimises its mean distance to the current median, then the median is recomputed. This removes cut-position noise, which in a cursive hand is the dominant error.
4. Drop the worst 20% by mean deviation and take the final median.

The prototype is accepted as an extra variant, prepended to the pool, only when it **measurably beats** the raw variants: `pd < rawPd - 0.02 and pd < 0.32`. A median can blur a letter whose strokes are genuinely inconsistent, so it has to earn its place.

### C.2.5 `RenderRefStats` — what the line looks like to the classifier

Letterforms alone are not enough: a synthesised line must also match the author's real lines in the statistics the classifier is sensitive to. `RenderRefStats` measures, through the classifier's own preprocessing:

| Statistic | Why |
|---|---|
| `inkFrac` | fraction of pixels below 128 — overall ink density |
| `aspect` | line width over height |
| `paperLevel`, `inkLevel` | median of bright (≥200) and dark (<128) pixels. Paper is not pure white and ink is not pure black on a real scan; rendering at those extremes is a domain shift the model reacts to |
| `midFrac` | fraction in 128–200, the soft-edge population |
| `wordGap` | median run of blank columns wider than `0.5*xh`, in x-heights, measured from **real ink gaps** rather than the recogniser's cramped space territory |
| `heightXh` | total ink height in x-heights |
| `slantMeas`, `ascMeas`, `descMeas` | measured the same way they will be measured on the synthesised render, so the two are directly comparable |

That last row is the important one: these are measured by the same functions that will later measure the output, which is what makes the calibration loop of C.3.4 valid.

### C.2.6 `BuildAuthorProfile` — the global parameters

Aggregated over the author's accepted lines:

| Parameter | How |
|---|---|
| `slantDeg` | median of per-line `EstimateSlantDeg` |
| `xHeightPx` | median per-line x-height |
| `strokeWidthXh` | median of `strokeW/xh` |
| `ascender` | median `top` over lowercase glyphs with `top > 1.35`; default 1.7 |
| `descender` | median `bot` over lowercase glyphs with `bot < -0.25`; default −0.6 |
| `wordSpaceXh` | median of the real-ink `wordGap` measurements, falling back to the median space cell width |
| `connectedness` | `connPairs / adjPairs` — the measured **fraction of adjacent in-word letter pairs this author actually joins**, from the `connR` flags |
| `letterAdvance` | per-character median advance |

`connectedness` is the single most consequential number in the profile, because synthesis uses it as a **probability per pair** rather than a threshold.

### C.2.7 Caching and the two build paths

Extraction is expensive — skeletonising and tracing every line — so `ExtractAuthorRaw`'s output is pickled per author under `NOGIT/GlyphCache10/`. Re-running then only re-*filters*, which means the filtering constants can be re-tuned without re-skeletonising. `RenderRefStats` is recomputed outside the cache because it needs no model and is cheap.

Profiles are fitted **only on non-holdout pages** (`if s['is_holdout']: continue`), so a profile never sees the text it will later be evaluated on.

`BuildAll()` builds from the IAM page set. `BuildAll10Authors()` builds the mixed set: the eight dataset authors load from their existing raw cache, while the personal authors are segmented with `SegmentLean.SegmentLines` and force-aligned with the personal fine-tuned recogniser (`PERSONAL_WEIGHTS`, preferring the joint checkpoint), since the general model reads those hands less well. Lines labelled `"MESS"` are dropped. Either way the cross-author prior is learned from **all** authors pooled, and `BuildAuthorProfile` and the JSON output are shared code.

---

## C.3 `SynthesizeHandwriting.py` — trajectory and G-code

### C.3.1 The fallback fonts

Two built-in single-stroke skeleton fonts are defined in the normalised glyph frame: `_FB`, an upright print font, and `_CURSIVE`, a joined one. They serve two purposes — a character the author never wrote, and a **legibility anchor**.

The print font `_FB` reads at roughly 96% character and 84% word accuracy through the frozen recogniser, which is far clearer than any data-driven consensus glyph. That makes it the reference point for legibility.

`_BaselineGlyph(ch, profile, printOnly)` reproportions a fallback letter into the author's own style before use:

```python
if y>1.0:   y = 1.0+(y-1.0)*max(0.35,(asc-1.0)/0.75)
elif y<0.0: y = y*max(0.35,abs(desc)/0.55)
```

Only the parts **outside** the x-height band are scaled, so the ascender and descender reach become the author's while the x-height stays fixed. A connected hand (`connectedness ≥ 0.5`) gets the cursive skeleton so a substituted letter can still join its neighbours; a disconnected hand gets print.

### C.3.2 Choosing a glyph for each character instance

`_GlyphSource` decides, per character *instance*, what to draw. The preference order is:

1. **One of the author's own stored variants.** Candidates are ranked by `priorD` (most well-formed first) and only the top `_ELIGIBLE_VARIANTS = 5` are eligible. Sampling uniformly over a dozen variants averages the author's signature away, while using only the top one or two lets a single unlucky prototype dominate a letter everywhere it appears.

   Within those five, selection is weighted:

   ```python
   wi = 1.0/(1.0+1.4*i)                        # favour the more typical
   if prevExitY is not None:
       wi *= math.exp(-(dy/0.45)**2)           # favour a matching entry height
   ```

   The first term is a decaying rank weight. The second applies when joining onto a previous letter: a variant whose own entry height matches where the pen actually is produces a cleaner join. The weights become a discrete distribution sampled by the usual cumulative walk.

2. **A case or similar-letter neighbour**, via `swapcase()` or `_SIMILAR` (`I↔l`, `O↔0`, `;→:`, `"→'`, `!→l`). Borrowing a lowercase shape for a capital scales it up to the author's ascender height, clamped to `[1.0, 2.0]`.

3. **The style-aware anchor** `_CoreGlyph`, which is the author's own `idealMedoid` when it is recognisable (`priorD ≤ IDEAL_ANCHOR_MAX_PRIORD = 0.30`), otherwise the reproportioned cursive or print skeleton.

4. **The raw `_FB` font**, then an empty glyph that only advances the pen.

`pack()` enforces one rule worth noting: the advance used is **the variant's own**, never the character median —

```python
a = max(float(g.get('advance',0.9)), float(g.get('lead',0.0))+float(g.get('width',0.9))+0.04)
```

A wide variant advanced by a narrow median would collide with the next letter.

**The legibility blend.** `legibility` (`lam`) in `[0,1]` is the probability that a glyph is drawn from the anchor instead of the author's own hand. `_SwapProb` makes that probability depend on how clean the author's own letter is:

```python
if d >= _GLYPH_GARBAGE (0.55):  return 1.0      # replace regardless
if lam <= 1e-3:                 return 0.0      # pure author
if d <= _GLYPH_CLEAN  (0.12):   return 0.06*lam # keep a clean letter
bad = clip((d-0.12)/(0.55-0.12), 0, 1)
return clip(lam*(0.55+0.9*bad), 0, 1)
```

So a genuinely clean extraction is almost always kept; a borderline one swaps readily in proportion to `lam`; a garbage one always swaps. Because the anchor still carries the author's slant, size, spacing and joins, swapping a letterform does not erase the hand.

### C.3.3 `SynthesizeText` — assembling a line

The main loop walks words, then characters within words.

**Line layout.** `lineStep = (ascender - descender + 0.85) * xh` — the author's own vertical extent plus 0.85 x-heights of leading. A word is wrapped when `penX + wordWidth > lineWidthMm`, with the width predicted from `letterAdvance` before drawing.

**Per-instance jitter.** `_Jitter` applies an independent random translation, rotation and anisotropic scale to each glyph instance:

```python
ampP = 0.022*jitter   # position, x-heights
ampR = 2.2*jitter     # rotation, degrees
ampS = 0.05*jitter    # scale
driftAmp = 0.05*jitter
```

Plus a slow baseline drift, modelled as an AR(1) random walk clamped to ±0.25 x-heights:

```python
driftV = 0.85*driftV + rng.gauss(0.0, driftAmp*0.35)
drift  = clip(drift+driftV, -0.25, 0.25)
```

The 0.85 coefficient makes the drift *correlated* between successive letters, so the baseline wanders smoothly the way a real hand's does rather than jittering independently per letter. Anchor glyphs get only `CORE_JITTER_FRAC = 0.15` of this, and half or a quarter of the drift, because they are designed-legible shapes and there is no reason to wobble them as hard as an extracted one.

**Placement and shear.** A glyph's points are placed by

```python
(px + (lead + x + glyphSlant*y)*xh,  penY + (y + glyphDrift)*xh)
```

The shear `glyphSlant*y` is applied **at placement**, which is why glyphs are stored upright: all the spacing arithmetic stays slant-independent. The author's slant is capped on a soft curve by legibility, `authorSlantCap = 45 - 19*lam`, and an anchor glyph is capped harder at `CORE_SLANT_CAP_DEG = 22°` (30° if cursive) so its ascenders do not ramp off.

**Collision guard.**

```python
curMinX = min(p[0] for s in placed for p in s)
need = prevBodyMaxX + COLLISION_MIN_GAP_XH*xh - curMinX
if need>0: penX += need; placed = place(penX)
```

If a glyph's ink would land on top of the previous glyph's body, the pen is pushed right to guarantee `0.05` x-heights of clear space, and the glyph re-placed. This catches the case where jitter and a wide variant combine badly.

**The body stroke.** A glyph can be several pen-downs — a letter plus its dot, crossbar or accent. The **body** is the stroke with the widest x extent:

```python
bodyI = int(np.argmax([max(p[0] for p in s)-min(p[0] for p in s) for s in placed]))
```

Only the body carries the cursive connection and becomes the exit point for the next letter; dots and crossbars are flushed as separate pen-downs. Letting a dot become the exit point would produce a ligature running from an *i* dot into the next letter.

**Joining.** Connectedness is used as a per-pair probability, not an on/off threshold:

```python
bothOwn         = not isCore and prevSrc!='core'
bothCursiveCore = coreCursive and prevCoreCursive
joinable = (ci>0 and prevExit is not None
            and ch.isalpha() and word[ci-1].isalpha()
            and (bothOwn or bothCursiveCore)
            and rng.random() < min(connEff, JOIN_PROB_CAP))
```

A threshold would fuse whole words into blobs; a probability reproduces the author's measured *rate* of joining. Both glyphs must be the author's own extracted forms, or both the cursive anchor — a **print** anchor glyph is always read as a separate letter, so joining into one would create a connector to nothing.

When a pair joins, the ligature and both glyphs are emitted as **one pen-down polyline**, which is what the plotter needs: one polyline is one pen-down move.

**`_Ligature`** sags in proportion to the distance it travels:

```python
sag = min(0.11*xh, 0.16*abs(dx)) + rng.gauss(0.0, 0.015*xh)
mid = ((p0[0]+p1[0])*0.5, min(p0[1],p1[1]) - sag)
```

A fixed dip would turn a short join into a loop, which is exactly what makes synthesised cursive look tangled. The cap at `0.11*xh` stops a long join from diving.

Every emitted polyline is passed through `_Resample(chain, step=0.35)`, giving even arc-length spacing — smooth motion for the plotter, and a uniform rendered stroke width.

The returned `Trajectory` carries `meta` with the author, `mmPerXh`, stroke count, a `glyphSources` histogram (`own`/`core`/`case`/`similar`/`fallback`/`blank`), slant, connectedness and line count.

### C.3.4 `StyleCalibration` — correcting line-level drift

Glyphs are stored per instance, so a line assembled from them does not automatically reproduce the author's *line-level* slant and ascender reach. `StyleCalibration` closes that loop empirically:

1. Synthesise a fixed probe sentence with `jitter=0`.
2. Render it and measure it with **the same functions** that measured the real pages — `BinarizeLine`, `CoreBand`, `EstimateSlantDeg`.
3. Compare against the profile's `slantMeasRef`, `ascMeasRef`, `descMeasRef` and update:

   ```python
   shearDelta += tan(slantRef) - tan(gotSlant)
   ascK  = clip(ascK*(ascRef-1.0)/(gotAsc-1.0), 0.5, 1.6)
   descK = clip(descK*descRef/gotDesc, 0.5, 1.6)
   ```

4. Repeat 3 times, and cache per `(author, mmPerXh, pxPerMm)`.

The shear correction is additive in `tan` space, which is the space the shear is actually applied in, and the reach corrections are multiplicative ratios clamped to `[0.5, 1.6]` so one bad probe cannot distort the author. `_ApplyCal` scales only what lies outside the x-height band, so reach changes without disturbing x-height. Anchor glyphs are exempt, since they are already designed shapes.

The code records the measured trade-off: calibration scores slightly *worse* against the ink-sensitive writer-ID model (86.0% → 84.5%) but clearly *better* against the shape-only model (59.2% → 63.3%, and 58.3% → 66.7% through G-code). The shape-only judge is the one that predicts the physical robot, which writes every author with the same pen, so calibration stays on.

### C.3.5 Rendering

`_Draw` rasterises the trajectory as black ink on white paper, matching the training crops, with round stroke ends (`d.ellipse` at each endpoint, since a pen leaves no square corners) and an optional Gaussian blur for a soft edge.

`RenderTrajectory` has two modes, and the default is the physically honest one:

```python
UNIFORM_INK = True
UNIFORM_PEN_WIDTH_XH = 0.085
UNIFORM_SOFT_FRAC = 0.30
```

The gantry writes every author with **one** physical pen at a constant stroke width, so ink weight is not a style channel it can reproduce. With `uniformInk` on, every author renders at the same width in x-heights, which is also the domain the shape-only writer-ID model is trained on, and per-author ink fields are ignored. `0.085` is thinner than an earlier `0.11` because a heavy stroke fills small loops.

Padding is `0.18*mmPerXh` — proportional to the writing rather than a fixed slab, since a fixed pad would inflate the crop height and skew the line's aspect ratio, which the classifier sees.

The legacy mode (`uniformInk=False`) instead searches for the stroke width that reproduces the author's measured ink density. `_MatchInkDensity` has to be a **search** rather than a formula, for a stated reason: after the classifier's downsampling a thin stroke anti-aliases to light grey and drops out of the `<128` ink test, so density is a step-like function of width. It also has to run on **this** line rather than a fixed probe, since density depends on how much text the line holds. The search is a cached hill-climb on `|measured − target|`.

### C.3.6 Best-of-N selection

Two selection strategies sit on top of `SynthesizeText`.

**`SynthesizeLegible`** draws the line `nTries` times — same library and parameters, different variant and jitter draws — and keeps whichever the frozen recogniser reads best by `CharAcc`. Seeds are `base + 977*k`, and the loop exits early at ≥ 0.99.

Then a **repair** pass, up to `REPAIR_ROUNDS = 3` times:

- `_WeakChars` runs CTC forced alignment of the rendered image against the intended text and scores each character by its own class log-probability over its assigned frames, worst first. A character the alignment skipped entirely gets `-1e9`.
- The worst `REPAIR_WORST_K = 3` not-yet-repaired positions are pinned to the anchor (`perCharLam[p] = 1.0`) and the line redrawn.
- The single worst character that is *still* weak after already being pinned escalates to `2.0`, which forces upright print as a last resort.
- A candidate is kept only if it scores better.

Because every candidate is a real rendering in this hand, repair only cleans up unreadable letters rather than restyling the line.

**`SynthesizeJointBestOf`** is what the live server uses, and it scores by the **harmonic mean** of text accuracy and writer-ID confidence:

```python
combined = 2*textAcc*widConf/(textAcc+widConf)
```

The reasoning, recorded in the source, is the interesting part. Plain best-of-N scored on legibility alone raises text accuracy but can *cost* writer ID, because the single most legible draw out of N random draws has no reason to also be the most distinctive one — for some authors the two axes pull in different directions across the candidates, and a text-only score always takes the legibility side of that coin flip. The harmonic mean is dominated by whichever of the two is **smaller**, so a candidate that is brilliant on one axis and poor on the other gets a low combined score, where the arithmetic mean would let 0.95/0.30 look as good as a balanced 0.65/0.65. Best-of-N under the harmonic mean is therefore forced to pick a genuine compromise on both axes.

Its repair pass is joint-aware for the same reason: a proposed repair is accepted only if the **joint** score improves, so a repair that fixes a letter but makes the line look like someone else is rejected rather than silently taking the writer-ID hit. Repaired positions are never retried, accepted or not.

### C.3.7 Machine configuration and coordinate transfer

`GantryConfig` is a dataclass holding the whole machine calibration in one block:

| Group | Values |
|---|---|
| Transmission | 200 full steps/rev (NEMA 17), 1/16 microstepping, 40 mm/rev (GT2 2 mm pitch × 20 teeth) |
| Work area | 0–200 mm both axes, text origin at (10, 180) |
| Speeds | draw 900 mm/min, travel 2400 mm/min, accel 400 mm/s², minimum segment 0.12 mm |
| Pen | one-way gear motor, 260 ms per 90° pulse, 90 ms settle, `M3` down / `M5` up, starts up |

Derived properties give `microstepsPerRev = 3200` and `stepsPerMm = 3200/40 = 80` per axis.

**`ToMachine`** places the trajectory on the bed and optionally scales it to fit:

```python
scale = min(1.0, availW/w, availH/h)
q = [(cfg.originXmm+(x-x0)*scale, cfg.originYmm-(y1-y)*scale) for (x,y) in s]
```

Both frames have y up, the standard plotter convention, so the text's top row lands at `originY` and later rows descend from there. `scale` only ever shrinks, never enlarges. `_Decimate` then drops points closer than `minSegmentMm = 0.12` to the previous kept point, which removes moves shorter than the machine can meaningfully execute.

**`OptimizeStrokeOrder`** reorders pen-down strokes to cut pen-up travel. It is careful to preserve reading order:

1. Group strokes into text **rows** by mean y, with `rowTolMm = max(1, 1.2 × median stroke height)`.
2. Within each row, greedy nearest-neighbour from the current pen position, starting at the leftmost stroke.
3. Enter each stroke from whichever end is closer, reversing it if needed — free for a plotter, since a polyline draws the same either way.

Only the order changes, so the drawn result is identical while pen-up travel and job time drop. Rows first means the machine still writes the page top to bottom, left to right.

`_Clamp` pulls every point into the work area and counts how many were moved, so the G-code header can warn.

### C.3.8 The pen state machine

The pen is a one-way gear motor with no driver and no position feedback: every 90° of rotation **toggles** the pen. It is never commanded to a state, only pulsed.

```python
class PenController:
    def SetPen(self,wantUp):
        if wantUp==self.isUp:
            return []          # emitting a pulse here would invert the pen for the rest of the job
        self.isUp = wantUp
        self.pulses+=1
        ...
        return [code, f"G4 P{dwell:.3f}"]
```

So state is tracked in software, each state *change* emits exactly one pulse, and a no-op change emits nothing. The dwell `G4 P(penPulseMs+penSettleMs)/1000` holds the machine still while the 90° rotation completes. Every transition in the file goes through this class, which is what keeps the software's idea of pen state and the mechanism's actual state in step.

### C.3.9 `WriteGcode`

```
; header: author, stroke count, scale, steps/mm, pen convention, clamp warning
G21            ; millimetres
G90            ; absolute
M5 + G4        ; pen up
G0 F2400       ; travel feed
  per stroke:
    M5 + G4            (no-op if already up)
    G0 X.. Y..         rapid to the stroke's start, pen up
    M3 + G4            pen down
    G1 F900            draw feed
    G1 X.. Y..  ...    the stroke
M5 + G4        ; pen up
G0 X10 Y180    ; park
M2             ; end
```

The returned dict reports `lines`, `strokes`, `penPulses`, `scale`, `clipped` and the path — `penPulses` being the count the web UI displays, and a direct measure of mechanical wear.

### C.3.10 Reading the G-code back

`ParseGcode` replays an emitted file and returns the pen-down polylines in machine millimetres, tracking `M3`/`M5` for pen state and `X`/`Y` on `G0`/`G1` for position. `RenderGcodePreview` draws those strokes on a bed-sized canvas with the work-area outline, and optionally the pen-up travel moves in pink, converting machine y-up to image y-down.

This closes the loop: the previewed image is rasterised from **the file that will be sent to the machine**, not from the trajectory in memory, so anything lost between trajectory and G-code is visible in the preview. The same function also serves `/api/gcode/upload`, which is how an arbitrary hand-written G-code file can be put through the identical preview and send path.

---

## C.4 `Evaluate.py` — the measurement harness

A CLI (`BuildArgParser`/`Main`) over several measurement routines. It loads its models through the same `Classify.py` code and the same checkpoint-precedence rule as everything else, so a number it reports is a number the live pipeline would produce.

**`MeasureLine`** extracts the comparable style features from any line image — real or synthesised — using Block A's primitives: slant, x-height, ink density, aspect, ascender and descender reach, word gap, and the ink fraction inside the core band. Because real and synthetic lines go through the *same* measurement code, the feature-by-feature comparison is meaningful.

**`StyleEvaluate`** synthesises held-out text per author over several seeds and compares measured features against the profile's reference values, with `FeatureReport` aggregating per feature.

**`RunRewrite`** is the central experiment: take an author's real held-out line, synthesise the *same words* in that author's style, and read both with the recogniser. The real line's accuracy is the upper bound the synthesis is trying to reach.

**`GcodeRoundTrip`** writes the trajectory to G-code and reads it back through `ParseGcode`, so the digital-to-machine step is measured rather than assumed.

**`MachineDistort`** adds backlash (0.15 mm) and jitter (0.05 mm) to a trajectory to estimate the mechanical contribution, and `EvaluatePhysical` runs that path.

**`RunShapeStyle`** evaluates writer ID with the shape-only (stroke-normalised) classifier, which is the judge that predicts the physical robot.

**`LegibilityEvaluate`**, **`SweepLambda`** and **`TuneLegibility`** sweep the legibility blend `lam` against `LEGIBILITY_TARGET_CHAR = 0.90` and `LEGIBILITY_TARGET_WORD = 0.62`, per author or aggregate, which is where `DEFAULT_LEGIBILITY = 0.65` came from. `_CiCharAcc`/`_CiWordAcc` are case-insensitive variants, and `NOVEL_SENTENCES`/`NOVEL_CORPUS` are fixed held-out text pools so a tuning run is comparable across invocations.

`BuildComparisonSheet` and `_SaveSheet` write the side-by-side real-versus-synthesised image sheets.
