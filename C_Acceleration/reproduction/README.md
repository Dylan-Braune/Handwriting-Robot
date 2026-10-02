# accel -- optional C acceleration for the numpy inference path

This directory is **additive and opt-in**. Nothing in the existing
codebase imports it, and no existing file (`np_inference/layers.py`,
`np_inference/text_model.py`, `np_inference/author_model.py`,
`SynthesizeHandwriting.py`) has been modified to use it.

## What it is

`SynthesizeJointBestOf()` scores every candidate rendering by running two
from-scratch numpy CNNs (text recognizer + writer classifier), `nTries`
times per request. Profiling showed `conv2d()` in
`np_inference/layers.py` is the single biggest cost (~39% of total time).
This directory reimplements that function (plus, as a stretch goal,
`lstm_forward()`) in portable C, with a ctypes wrapper exposing the exact
same signature/return shape so it's a true drop-in replacement.

- `accel.c` -- C99 `conv2d_forward` (direct nested-loop convolution,
  bias add included) and `lstm_forward` (one direction/layer of the
  PyTorch-convention LSTM). No platform-specific headers or intrinsics
  -- only `<stdint.h> <stdlib.h> <string.h> <math.h>` -- so the same
  source builds on x86_64 Windows and ARM64 Linux (ODROID N2+) with a
  plain `gcc -O3`.
- `accel_py.py` -- ctypes wrapper: `conv2d(x, weight, bias, stride,
  padding)` and `lstm_forward(x, weight_ih, weight_hh, bias_ih,
  bias_hh, hidden_size)`, matching `np_inference/layers.py`'s functions
  of the same names. `AVAILABLE["conv2d"]` / `AVAILABLE["lstm"]` report
  whether the compiled library loaded.
- `build_windows.bat` -- builds `libaccel.dll` with the MinGW-w64 gcc
  already installed on this laptop.
- `build_linux.sh` -- **this is the script that runs on the ODROID
  N2+ itself**, native ARM64 gcc, no cross-compilation.

## Building

Windows (dev machine), from this directory or anywhere:

```
accel\build_windows.bat
```

which runs, verbatim:

```
"C:\Users\braun\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin\gcc.exe" -O3 -shared -o libaccel.dll accel.c -lm
```

**ODROID N2+ (ARM64 Linux), the deployment target:**

```sh
gcc -O3 -shared -fPIC -o libaccel.so accel.c -lm
```

(equivalently `sh build_linux.sh` from this directory once the directory
is copied/pulled onto the ODROID). This is a native build on the ODROID's
own gcc -- the exact same `accel.c` source file, no edits needed.

`accel_py.py` picks `libaccel.dll` on Windows and `libaccel.so` on Linux
automatically at import time, so the same Python wrapper works
unmodified on both.

## How one COULD opt in (not applied anywhere)

`SynthesizeHandwriting.py` and `np_inference/*.py` are untouched. If/when
the student wants to actually use this, the smallest opt-in is a
monkey-patch at process startup, before any model object is constructed
(the model classes call `L.conv2d` as a module-level lookup each time,
so patching the module attribute is sufficient and does not require
touching `text_model.py`/`author_model.py`):

```python
import accel.accel_py as accel
import np_inference.layers as L

if accel.AVAILABLE["conv2d"]:
    L.conv2d = accel.conv2d
# lstm is a stretch-goal implementation; same pattern if desired:
# if accel.AVAILABLE["lstm"]:
#     L.lstm_forward = accel.lstm_forward
```

This line could go in `server.py` near where the models are first
loaded, guarded by the `AVAILABLE` check so it silently no-ops on any
machine where `libaccel.{dll,so}` hasn't been built yet. It has
deliberately NOT been added to any existing file as part of this task.

## Verification

See `test_correctness.py`, `test_benchmark.py`, and
`test_full_pipeline.py` in this directory for the actual verification
scripts (real checkpoints, real activations, real images -- no
reinvented model code). Run them with the project's normal Python
environment from the `AuthorReproductionStuff` directory, e.g.:

```
python -m accel.test_correctness
python -m accel.test_benchmark
python -m accel.test_full_pipeline
```

Results from the run performed during development are reported in the
task's final report (max abs/relative diff, timings, decoded text
comparison, and full-pipeline timing); re-run the scripts above to
reproduce them.
