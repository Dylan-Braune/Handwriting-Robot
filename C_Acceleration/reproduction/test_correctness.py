"""Correctness check: accel.conv2d_forward (C) vs np_inference.layers.conv2d
(pure numpy) on REAL weights and REAL intermediate activations pulled out
of an actual forward pass through PaperCRNNNumpy and AuthorClassifierCNNNumpy.

Does not modify any existing file. Run from AuthorReproductionStuff/:
    python -m accel.test_correctness
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "Software" / "CNN" / "AuthorReproductionStuff"))  # AuthorReproductionStuff/

from np_inference import layers as L  # noqa: E402
from np_inference.text_model import (  # noqa: E402
    PaperCRNNNumpy, resize_line_image_fixed, tensor_from_resized,
)
from np_inference.author_model import AuthorClassifierCNNNumpy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent)); import accel_py as accel  # noqa: E402

IMG_PATH = HERE.parent.parent / "Software" / "CNN" / "NOGIT" / "dylan" / "preprocessing_deskew_ruleline_border_cleanup.jpg"


def compare(name, py_out, c_out):
    diff = np.abs(py_out - c_out)
    max_abs = float(diff.max())
    denom = np.maximum(np.abs(py_out), 1e-12)
    max_rel = float((diff / denom).max())
    print(f"  [{name}] shape={py_out.shape} max_abs_diff={max_abs:.3e} max_rel_diff={max_rel:.3e}")
    return max_abs, max_rel


class ConvCapture:
    """Wraps L.conv2d to record (x, weight, bias, stride, padding) for every
    real call made during a forward pass, without modifying layers.py."""

    def __init__(self):
        self.calls = []
        self._orig = L.conv2d

    def __enter__(self):
        def spy(x, weight, bias, stride=1, padding=1):
            out = self._orig(x, weight, bias, stride=stride, padding=padding)
            # Keep a handful of representative calls (varied channel counts).
            self.calls.append((x.copy(), weight.copy(),
                                None if bias is None else bias.copy(),
                                stride, padding, out.copy()))
            return out
        L.conv2d = spy
        return self

    def __exit__(self, *a):
        L.conv2d = self._orig


def main():
    print(f"accel.AVAILABLE = {accel.AVAILABLE}")
    assert accel.AVAILABLE["conv2d"], "libaccel did not load -- build it first"

    print(f"Loading real image: {IMG_PATH}")
    assert IMG_PATH.exists(), f"missing test image {IMG_PATH}"
    img = Image.open(IMG_PATH)
    # Crop a line-sized region so this behaves like a real recognizer input.
    w, h = img.size
    crop = img.crop((0, 0, min(w, 1280), min(h, 128)))
    tensor = tensor_from_resized(resize_line_image_fixed(crop))
    print(f"  input tensor shape: {tensor.shape}, dtype: {tensor.dtype}")

    print("\nLoading real checkpoints...")
    text_model = PaperCRNNNumpy()
    author_model = AuthorClassifierCNNNumpy()
    print(f"  text checkpoint: {text_model.checkpoint_path}")
    print(f"  author checkpoint: {author_model.checkpoint_path}")

    print("\nRunning real forward passes, capturing every conv2d() call...")
    with ConvCapture() as cap:
        _ = text_model.forward(tensor.astype(np.float64))
    with ConvCapture() as cap2:
        _ = author_model.forward(tensor.astype(np.float64))

    all_calls = cap.calls + cap2.calls
    print(f"  captured {len(all_calls)} total conv2d calls "
          f"({len(cap.calls)} from text model, {len(cap2.calls)} from author model)")

    # Pick a spread of distinct shapes (channel counts / spatial sizes) to
    # actually exercise different code paths, not just the first N calls.
    seen_shapes = {}
    for call in all_calls:
        x, w, b, stride, padding, py_out = call
        key = (x.shape, w.shape, stride, padding)
        seen_shapes.setdefault(key, call)

    picks = list(seen_shapes.values())
    # Ensure at least 4, spread across the range (first, last, and middle ones).
    if len(picks) > 6:
        idxs = sorted(set([0, len(picks) // 4, len(picks) // 2, 3 * len(picks) // 4, len(picks) - 1]))
        picks = [picks[i] for i in idxs]

    print(f"\nComparing {len(picks)} distinct real conv2d calls (C vs numpy):")
    worst_abs, worst_rel = 0.0, 0.0
    for i, (x, w, b, stride, padding, py_out) in enumerate(picks):
        print(f" call {i}: x.shape={x.shape} weight.shape={w.shape} stride={stride} padding={padding}")
        c_out = accel.conv2d(x, w, b, stride=stride, padding=padding)
        a, r = compare(f"call {i}", py_out.astype(np.float64), c_out)
        worst_abs = max(worst_abs, a)
        worst_rel = max(worst_rel, r)

    print(f"\nWORST across all compared calls: max_abs_diff={worst_abs:.3e} max_rel_diff={worst_rel:.3e}")
    if worst_rel < 1e-9:
        print("PASS: C conv2d_forward matches numpy conv2d to float64 precision.")
    else:
        print("WARNING: relative diff larger than expected -- investigate before using.")
        sys.exit(1)


if __name__ == "__main__":
    main()
