"""Correctness + benchmark for accel.lstm_forward (C) vs
np_inference.layers.lstm_forward (pure numpy), using REAL LSTM weights
from the real text-recognition checkpoint (PaperCRNNNumpy's `sequence`
BiLSTM) and a realistic-length real input sequence (one from an actual
forward pass through the conv stack).

Run from AuthorReproductionStuff/:
    python -m accel.test_lstm
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

sys.path.insert(0, str(Path(__file__).resolve().parent)); import accel_py as accel  # noqa: E402

IMG_PATH = HERE.parent.parent / "Software" / "CNN" / "NOGIT" / "dylan" / "preprocessing_deskew_ruleline_border_cleanup.jpg"
N_REPS = 10


def main():
    assert accel.AVAILABLE["lstm"], "libaccel lstm_forward did not load"

    model = PaperCRNNNumpy()
    print(f"text checkpoint: {model.checkpoint_path}, hidden_size={model.hidden_size}")

    img = Image.open(IMG_PATH)
    w, h = img.size
    crop = img.crop((0, 0, min(w, 1280), min(h, 128)))
    tensor = tensor_from_resized(resize_line_image_fixed(crop)).astype(np.float64)

    # Reproduce the conv stack up to the point the real LSTM input sequence
    # is built, using the model's own forward() internals (not duplicating
    # logic -- just stop before the LSTM by calling the same helper it uses).
    from np_inference.text_model import _conv_block_forward
    sd = model.sd
    x = _conv_block_forward(tensor, sd, "stage1", 2)
    x = L.maxpool2d(x, 2, 2)
    x = _conv_block_forward(x, sd, "stage2", 4)
    x = L.maxpool2d(x, 2, 2)
    x = _conv_block_forward(x, sd, "stage3", 6)
    x = L.adaptive_max_pool_height(x)
    x = x[:, :, 0, :]
    seq_input = np.transpose(x, (2, 0, 1)).astype(np.float64)  # (W,B,128)
    print(f"real LSTM layer-0 input shape (T,B,input_size): {seq_input.shape}")

    layer0 = model.lstm_layer_params[0]
    fwd = layer0["fwd"]
    H = model.hidden_size

    print("\n--- Correctness (layer 0, forward direction) ---")
    py_out, (py_h, py_c) = L.lstm_forward(seq_input, hidden_size=H, **fwd)
    c_out, (c_h, c_c) = accel.lstm_forward(
        seq_input, fwd["weight_ih"], fwd["weight_hh"], fwd["bias_ih"], fwd["bias_hh"], H
    )
    diff_out = np.abs(py_out - c_out)
    diff_h = np.abs(py_h - c_h)
    diff_c = np.abs(py_c - c_c)
    rel_out = (diff_out / np.maximum(np.abs(py_out), 1e-12)).max()
    print(f"outputs: shape={py_out.shape} max_abs={diff_out.max():.3e} max_rel={rel_out:.3e}")
    print(f"final h: max_abs={diff_h.max():.3e}   final c: max_abs={diff_c.max():.3e}")
    ok = rel_out < 1e-9 and diff_h.max() < 1e-9 and diff_c.max() < 1e-9
    print("PASS" if ok else "FAIL -- investigate before use")

    print("\n--- Benchmark (layer 0, forward direction), "
          f"{N_REPS} reps, T={seq_input.shape[0]} ---")
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        L.lstm_forward(seq_input, hidden_size=H, **fwd)
    py_time = (time.perf_counter() - t0) / N_REPS

    t0 = time.perf_counter()
    for _ in range(N_REPS):
        accel.lstm_forward(seq_input, fwd["weight_ih"], fwd["weight_hh"], fwd["bias_ih"], fwd["bias_hh"], H)
    c_time = (time.perf_counter() - t0) / N_REPS

    print(f"numpy: {py_time*1000:.3f}ms   C: {c_time*1000:.3f}ms   "
          f"speedup={py_time/c_time:.2f}x  ({(1-c_time/py_time)*100:.1f}% reduction)")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
