# accel/ -- optional C acceleration layer for SegmentPage.py

This directory is **purely additive**. Nothing in `SegmentPage.py`,
`SegmentPageCore.py`, or `RawImageOps.py` was changed, and nothing here is
imported automatically by the production pipeline. It exists so a caller
can *opt in* to C-accelerated versions of two numpy hot spots, without
forking or editing the existing files.

## What's here

- `accel.c` -- portable C99 (no platform intrinsics, no SIMD, no
  non-standard headers: only `<stdint.h> <stdlib.h> <string.h> <math.h>`).
  Implements:
  - `label_components` -- two-pass union-find connected-component
    labeling on a flat `uint8_t` binary mask.
  - `box_sum` -- box-sum over a `(2*ry+1) x (2*rx+1)` window per pixel
    with edge clamping, via a summed-area table, matching
    `RawImageOps.BoxSum` bit-for-bit (to float64 rounding noise).
- `accel_py.py` -- ctypes wrapper exposing `label_components(mask,
  connectivity=8) -> (labels, count)` and `box_sum(img, ry, rx) ->
  ndarray`, with the same numeric I/O contract as the numpy originals in
  `RawImageOps.py`. Loads `libaccel.dll` on Windows (`os.name == 'nt'`)
  and `libaccel.so` elsewhere, from this same directory. Exposes
  `accel.accel_py.AVAILABLE = {'label_components': bool, 'box_sum':
  bool}` so callers can check before opting in.
- `build_windows.bat` / `build_linux.sh` -- compile the **same**
  `accel.c` into `libaccel.dll` / `libaccel.so` respectively.
- `test_accel.py` -- correctness + isolated-benchmark script (synthetic
  data and real data pulled from an actual pipeline run).
- `pipeline_sanity.py`, `pipeline_sanity2.py`, `profile_calls.py` --
  full-pipeline, drop-in-replacement sanity scripts (see "Verification
  results" below). Not needed for normal use; kept for reference, safe to
  delete.

## How to actually use this (opt-in, NOT applied to SegmentPage.py)

In your own script (never in `SegmentPage.py` itself unless you decide to
make this permanent later), before calling `SegmentPage.ProcessPage`:

```python
import RawImageOps as F
import accel.accel_py as accel

if accel.AVAILABLE['label_components']:
    F.LabelComponents = accel.label_components
if accel.AVAILABLE['box_sum']:
    F.BoxSum = accel.box_sum

import SegmentPage  # or import it first; patch applies at call time
```

This works because `SegmentPage.py` and `SegmentPageCore.py` both do
`import RawImageOps as F` -- `F` is the *same module object* everywhere,
so patching `F.LabelComponents` / `F.BoxSum` once affects every call site
in every module, with no source edits. `RawImageOps.BoxMean` internally
calls `BoxSum(...)` as a module-global lookup too, so patching `F.BoxSum`
also speeds up every `BoxMean`/`GaussianBlur`/`AdaptiveThresholdInv` call
that goes through it.

`label_components`'s C implementation does **not** reproduce the Python
version's exact label *numbering* (which depends on its row-run scan
order) -- only the same **partition** of foreground pixels into
components and the same count. Every call site in the existing code only
uses labels to test equality / look up per-component stats, never
compares label values across runs, so this is a safe substitution. This
was explicitly verified (see below), not assumed.

## ODROID N2+ deployment build

Copy `accel.c` (and `accel_py.py` if you want the wrapper) to the ODROID,
then on the ODROID itself run:

```sh
gcc -O3 -shared -fPIC -o libaccel.so accel.c -lm
```

(this is exactly what `build_linux.sh` runs). No cross-compilation is
used or needed -- `accel.c` is strict C99 with no x86-specific or
Windows-specific code, so the ODROID's own native `gcc` (ARM64) builds it
directly.

## Verification results (actual numbers, 2026-10-02 run)

Test machine: Windows x86_64 laptop, MinGW-w64 gcc building
`libaccel.dll`. Real test image:
`Software/CNN/NOGIT/dylan/preprocessing_deskew_ruleline_border_cleanup.jpg`
(loads to 2400x1800 after `LoadImage`'s long-side resize).

### label_components

**Correctness** (`test_accel.py`), partition-equivalence (not raw label
value equality -- see note above):
- Synthetic 2000x3000 mask, ~400 scattered components, connectivity 4:
  Python count=6146, C count=6146, 0 inconsistent label mappings.
- Synthetic, connectivity 8: Python count=706, C count=706, 0 mismatches.
- **Real** ink mask (`BinarizeInk` output on the real photo),
  connectivity 4: Python count=4820, C count=4820, 0 mismatches.
- **Real** ink mask, connectivity 8: Python count=4578, C count=4578, 0
  mismatches.

**Benchmark** (real ink mask, 2400x1800, connectivity=8, 5 reps):
- Python mean: 224.1 ms (reps: 224.6/221.1/221.5/221.6/231.8 ms)
- C mean: 27.7 ms (reps: 27.3/27.9/26.9/28.2/28.4 ms)
- **Speedup: 87.6%**

### box_sum

**Correctness** (`test_accel.py`), numeric comparison with combined
abs+rel tolerance (atol=1e-6, rtol=1e-6 -- well inside the ~1e-9 relative
precision float64 summation order allows):
- Synthetic 2000x3000 float64, ry=rx in {58, 9, 2, 0}: max abs diff
  3.3e-6 (at ry=rx=58; this is ordinary float64 summation-order noise on
  sums of ~255*117^2 ~ 3.5M -- relative error 2.6e-12), 0 mismatches at
  every radius tested.
- **Real** gray image from the pipeline (2400x1800), ry=rx=35 (the
  actual radius `CorrectIllumination`'s `GaussianBlur(gray,
  gray.shape[1]/30)` resolves to at this image's width): max abs diff =
  0.0, max rel diff = 0.0, 0 mismatches -- exact bit-for-bit match on
  this input.

**Benchmark** (real gray image, 2400x1800, 5 reps):
- ry=rx=35 (actual illumination-correction radius): Python mean 174.2 ms,
  C mean 44.8 ms -- **speedup 74.3%**.
- ry=rx=9 (`AdaptiveThresholdInv`-sized radius, called repeatedly
  throughout the pipeline): Python mean 159.5 ms, C mean 58.6 ms --
  **speedup 63.2%**.

### Full-pipeline sanity check (`pipeline_sanity.py` / `pipeline_sanity2.py`)

Both ran the real, unmodified `SegmentPage.ProcessPage()` on the real
photo with `F.LabelComponents` / `F.BoxSum` monkey-patched to the C
versions **only inside the test script's own process** -- `SegmentPage.py`
on disk is untouched.

- **Output identical**: every run (baseline and accelerated, 6 runs
  total across both sanity scripts) produced exactly **27 TEXT lines, 0
  MESS blocks, 27 total items**, same skew estimate (-0.30 deg). The
  drop-in replacement changes nothing about pipeline *output*.
- **Isolated function time inside the real pipeline**
  (`pipeline_sanity2.py`, which times only the code inside
  `LabelComponents`/`BoxSum` regardless of which implementation is
  active, separately from total wall time):
  - `LabelComponents`: 1396 calls/run. Python impl: 5.26s and 6.93s
    across two runs. C impl: 0.87s and 1.49s across two runs. Consistent
    with the isolated benchmark's ~88% speedup.
  - `BoxSum`: 123 calls/run. Python impl: 10.41s and 10.73s. C impl:
    1.15s and 1.99s. Consistent with the isolated benchmark's ~63-74%
    speedup.
- **Call-size distribution** (`profile_calls.py`): of the 1396
  `LabelComponents` calls, 97.4% are tiny (<10,000 px, e.g. per-component
  hole tests) and together cost only 0.5s total -- the other 32 calls
  (full-page-sized, >=1,000,000 px) cost 6.9s. All 123 `BoxSum` calls are
  large (>=34,640 px; these are whole-image `GaussianBlur`/
  `CorrectIllumination`/`AdaptiveThresholdInv` passes). This means the C
  acceleration's benefit on this pipeline comes almost entirely from the
  ~32 large `LabelComponents` calls and essentially all `BoxSum` calls,
  not from the thousands of tiny per-component calls (where ctypes/malloc
  call overhead and numpy's already-fast small-array path roughly wash
  out).
- **Total wall-clock time, known limitation**: this dev machine showed
  large, erratic variance in *total* `ProcessPage()` wall time across
  repeated identical-code runs during this session (ranging from ~29s to
  ~142s for the SAME accelerated code, and ~44-48s for the same baseline
  code), almost entirely attributable to the **unaccelerated** rest of
  the pipeline (`GroupLines` and friends, ~70% of total time, see below)
  rather than to `LabelComponents`/`BoxSum` -- the per-function
  instrumented times above stayed consistent (sub-2s variance) across
  every run regardless of the machine's overall load. Do not trust a
  single total-wall-clock number from this session as the "real" speedup
  on quieter hardware (e.g. the ODROID in isolation); trust the
  per-function numbers, which are robust and were measured directly.
  Taking the cleanest run as representative: baseline ProcessPage ~44-48s
  total, with ~5.3-6.9s in `LabelComponents` and ~10.4-10.7s in `BoxSum`
  (~16-18s of ~45s, i.e. roughly the same fraction the task's cProfile
  breakdown reported); replacing both with the C versions cuts that
  portion to ~2-3.5s, i.e. an expected ~13-15s absolute reduction
  end-to-end from these two functions alone, independent of system noise.

## What was skipped, and why

The stretch-goal third target -- a C port of a piece of the
`GroupLines`/`chainSpan`/`nearestChain` hot path (~72% of total pipeline
time, per the task's own profiling) -- was **not attempted as a shipped
implementation**. Reasons, after reading the full algorithm in
`SegmentPageCore.py` (`GroupLines`, lines ~543-920+, including
`chainSpan`, `nearestChain`, `_LineCurve`, `_CurveY`):

- The algorithm is not a fixed-shape batch computation: `chainList` is a
  Python list of dicts that is mutated (merged, items deleted, `break`d
  out of the outer loop, rescanned from scratch) across *three* separate
  merge passes (initial gap/dy chaining, baseline-curve span-overlap
  merge, pitch-aware second merge), each with different comparison
  thresholds and early-exit control flow, and later leftover-component
  assignment uses Python object identity (`id(c)`) to track which comps
  are already claimed -- none of which maps cleanly onto a "take arrays
  in, get arrays out" C function callable once per outer-loop iteration
  the way the task's stretch-goal description envisions.
- `_LineCurve` is itself a per-chain, locally-smoothed curve fit (a
  5-point weighted moving average over each chain's own sorted
  components) that gets recomputed after every merge, so a faithful port
  would need to replicate that local-smoothing shape, not just bounding
  boxes.
- Given the explicit instruction to skip rather than ship anything
  uncertain, and that correctness here (every partition decision feeding
  downstream TEXT/MESS scoring) matters far more than raw speed, this was
  judged too high-risk to attempt confidently in the time available. A
  real fix for this 72%-of-total-time hot spot likely needs an algorithm
  redesign (e.g. a spatial index for the O(n^2)-ish nearest/overlap scans
  the profiling shows 36 million inner-generator calls for), not a
  mechanical C port of the current Python control flow, which was out of
  scope for this additive, non-modifying task.

## Files in this directory

```
accel/
  accel.c              - the two implemented kernels, portable C99
  accel_py.py           - ctypes wrapper + AVAILABLE registry
  build_windows.bat     - builds libaccel.dll with the MinGW-w64 gcc
  build_linux.sh        - builds libaccel.so (run this ON the ODROID)
  libaccel.dll           - built Windows test artifact (gitignore this if
                           committing; it's a local build output)
  test_accel.py          - correctness + isolated benchmark script
  pipeline_sanity.py      - full-pipeline before/after timing + output check
  pipeline_sanity2.py     - same, with per-function time instrumentation
  profile_calls.py        - call-count/size profiling instrumentation
  README.md              - this file
```
