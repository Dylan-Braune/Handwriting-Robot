"""Benchmark: Python conv2d() vs C accel.conv2d() on the SAME real inputs
captured from a real forward pass. Run from AuthorReproductionStuff/:
    python -m accel.test_benchmark
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "Software" / "CNN" / "AuthorReproductionStuff"))

from np_inference import layers as L  # noqa: E402
from np_inference.text_model import (  # noqa: E402
    PaperCRNNNumpy, resize_line_image_fixed, tensor_from_resized,
)
from np_inference.author_model import AuthorClassifierCNNNumpy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent)); import accel_py as accel  # noqa: E402

IMG_PATH = HERE.parent.parent / "Software" / "CNN" / "NOGIT" / "dylan" / "preprocessing_deskew_ruleline_border_cleanup.jpg"
N_REPS = 20


class ConvCapture:
    def __init__(self):
        self.calls = []
        self._orig = L.conv2d

    def __enter__(self):
        def spy(x, weight, bias, stride=1, padding=1):
            out = self._orig(x, weight, bias, stride=stride, padding=padding)
            self.calls.append((x.copy(), weight.copy(), None if bias is None else bias.copy(), stride, padding))
            return out
        L.conv2d = spy
        return self

    def __exit__(self, *a):
        L.conv2d = self._orig


def main():
    assert accel.AVAILABLE["conv2d"], "libaccel did not load -- build it first"

    img = Image.open(IMG_PATH)
    w, h = img.size
    crop = img.crop((0, 0, min(w, 1280), min(h, 128)))
    tensor = tensor_from_resized(resize_line_image_fixed(crop)).astype(np.float64)

    text_model = PaperCRNNNumpy()
    author_model = AuthorClassifierCNNNumpy()

    with ConvCapture() as cap:
        text_model.forward(tensor)
    with ConvCapture() as cap2:
        author_model.forward(tensor)
    calls = cap.calls + cap2.calls

    seen = {}
    for c in calls:
        x, wt, b, s, p = c
        seen.setdefault((x.shape, wt.shape, s, p), c)
    calls = list(seen.values())
    print(f"Benchmarking {len(calls)} distinct real conv2d shapes, {N_REPS} reps each.\n")

    total_py, total_c = 0.0, 0.0
    for i, (x, wt, b, s, p) in enumerate(calls):
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            L.conv2d(x, wt, b, stride=s, padding=p)
        py_time = (time.perf_counter() - t0) / N_REPS

        t0 = time.perf_counter()
        for _ in range(N_REPS):
            accel.conv2d(x, wt, b, stride=s, padding=p)
        c_time = (time.perf_counter() - t0) / N_REPS

        speedup = (py_time / c_time) if c_time > 0 else float("inf")
        total_py += py_time
        total_c += c_time
        print(f"  shape x={x.shape} w={wt.shape}: numpy={py_time*1000:.3f}ms  "
              f"C={c_time*1000:.3f}ms  speedup={speedup:.2f}x")

    print(f"\nTOTAL (sum of per-call means): numpy={total_py*1000:.3f}ms  C={total_c*1000:.3f}ms  "
          f"overall speedup={total_py/total_c:.2f}x  "
          f"({(1 - total_c/total_py)*100:.1f}% time reduction)")


if __name__ == "__main__":
    main()
