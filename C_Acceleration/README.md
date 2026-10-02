# C acceleration experiments (opt-in, NOT used by the pipeline)

Nothing in `Software/` or `FinalPipeline/` imports this folder. The pipeline runs
pure Python/numpy. These are optional C99 replacements for individual hot
functions, built so each can be swapped in and timed on its own.

- `segmentation/` - `label_components`, `box_sum` for `SegmentPage.py` (measured 84-88% faster each; ProcessPage ~25-32% faster end to end). Check: `python C_Acceleration/segmentation/pipeline_sanity.py`
- `reproduction/` - `lstm_forward` (2.8-4.4x faster, correct to 1e-15) and `conv2d_forward` (correct but SLOWER than numpy BLAS on x86; do not use without re-benchmarking on the ODROID). Check: `python C_Acceleration/reproduction/test_lstm.py`

Build on the ODROID (inside either folder): `sh build_linux.sh`
(= `gcc -O3 -shared -fPIC -o libaccel.so accel.c -lm`). Windows: `build_windows.bat`.

Not ported: `GroupLines`/`chainSpan`/`nearestChain` (~1/3 of ProcessPage time; sequential
merge algorithm). Estimated ceiling for converting everything: ~12-16s, not 2s.
