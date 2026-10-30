# To do

## Tomorrow, needs the Odroid in hand

### 1. Decide whether to parallelise reading and naming
The classify page runs at about **63 s**, twice the timing spec. Reading
(43%) and naming (27%) are 70% of it, and every line crop is processed
independently in a sequential loop, so the work is trivially parallel.

Two things were ruled out already, measured on the laptop:

- **batching the forward passes** gives only 1.08-1.16x. Not worth doing.
  Confirmed numerically safe (`identical=True` for reading, 3e-8 max
  probability difference for naming) if it is ever wanted.
- the LSTM Python time loop is **not** the bottleneck. Batching 20 lines
  would have cut that loop 20-fold and barely moved the clock, so the
  convolutions dominate.

That last point is why this needs the Odroid. numpy here is built on
threaded OpenBLAS with no thread limit set, so the convolutions may already
be using every core, in which case worker processes would just fight each
other. Run on the Odroid:

    nproc
    python -c "import numpy; numpy.show_config()"
    # then time one classify each way
    OPENBLAS_NUM_THREADS=1 <the classify run>
    <the same run with the variable unset>

- **much slower with `=1`** -> BLAS already uses the cores. Parallelism
  will not pay; go after segmentation instead.
- **no real difference** -> the cores are idle and a worker pool should
  scale close to `nproc`. That is the 2x.

Implementation if it is worth it: `multiprocessing.Pool` over the crops,
`pool.map` to keep order. On Linux the workers inherit the loaded models
copy-on-write, so the 13.4 MB reader and 3.9 MB namer are not pickled per
worker. This cannot be tested representatively on the Windows laptop, which
has no `os.fork` and would reload the models in every worker.

### 2. Confirm the pulled fixes actually took on the Odroid
The 500 on `/api/classify` was `segment.py:824`, `np.percentile` of an empty
selection; the guard is in but the Odroid was running pre-fix code. Watch
for a stale server process still bound to the port, which has bitten this
project before: confirm the old one is dead rather than just starting a new
one.

## Not blocked on hardware

### 3. Rebuild the style profiles
Two cut-time fixes are verified but reach nothing until the profiles are
rebuilt, because every profile on disk predates them:

- the crossbar fix in `style._BarOwner` (`CROSSBAR_TO_LEFT`), which took
  abhinav's contaminated `h` from 60% to 5%
- it would also clean `BuildLetterPrior`, which is contaminated, so
  `PriorFilter`/`_ConsensusFilter` currently *prefer* the dirty glyphs

### 4. The amputated up-leg
`style._TrimLigatureTails` deletes a cursive letter's approach stroke as if
it belonged to the previous letter, leaving half a letter hanging in mid
air. thiya's `l` is stored starting at 2.28 x-heights with `entryY` 2.143.
This is what the strange diagonals on the page actually are -- not
connectors; rendering with every ligature disabled leaves them all in place.
Fix is to keep or regrow the up-leg when `connL` is set and the body starts
above x-height. Also needs a profile rebuild, so do it with item 3.

### 5. Thin evidence worth thickening
- the join-anatomy change is +2 pages of writer-ID (22/26 -> 24/26) on **one
  seed**, and -0.4 points of character accuracy. A 5-seed repeat would say
  whether the writer-ID gain is real before it goes in the report.
- ~~`NAME_FROM_LINES = 6`~~ **withdrawn.** It was measured on 30 pages the ink
  model was already 96-99% confident on, so it never saw the case that breaks
  it: the six were picked with `linspace`, which always includes the LAST
  index, so a junk crop at the foot of a page sat in every pool and at two
  lines was half of it. On the camera page of 2026-10-30 all 16 real lines
  named dylan on their own and the pooled answer was still 78.4%, because the
  17th crop was a blot. Every line votes now, weighted by ink.
  The caution still stands for the **shape** model on synthesised output,
  where confidence runs 37-78%: do not sample there without measuring.

### 6. Geometry is unmeasured for accuracy
None of today's ruling work has been through an accuracy run. `make.py`
renders on its own fixed A4 scale, so the 86.7% figure does not exercise the
ruling fit at all. If smaller letters hurt readback, that number does not
exist yet.
