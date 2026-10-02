"""Full-pipeline sanity check. Does NOT modify any production file --
monkey-patches np_inference.layers.lstm_forward (and optionally conv2d)
to the C accel version IN THIS SCRIPT'S OWN PROCESS ONLY, then:

  1. Runs PaperCRNNNumpy().forward() and AuthorClassifierCNNNumpy().forward()
     on a real test image, confirming the decoded text / classification
     result is IDENTICAL with vs without the patch.
  2. Times SynthesizeHandwriting.SynthesizeJointBestOf('dylan', ..., nTries=3)
     with the patch active vs not, on the real profile/model stack.

Run from AuthorReproductionStuff/:
    python -m accel.test_full_pipeline
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "Software" / "CNN" / "AuthorReproductionStuff"))

IMG_PATH = HERE.parent.parent / "Software" / "CNN" / "NOGIT" / "dylan" / "preprocessing_deskew_ruleline_border_cleanup.jpg"


def run_patched(use_lstm_patch, use_conv_patch):
    """Each call re-imports fresh so the patch/unpatch is clean and the
    two runs don't share any module-level cached state."""
    import importlib
    import np_inference.layers as L
    import np_inference.text_model as TM
    import np_inference.author_model as AM
    importlib.reload(L)
    importlib.reload(TM)
    importlib.reload(AM)

    sys.path.insert(0, str(Path(__file__).resolve().parent)); import accel_py as accel
    orig_lstm = L.lstm_forward
    orig_conv = L.conv2d
    if use_lstm_patch:
        L.lstm_forward = accel.lstm_forward
    if use_conv_patch:
        L.conv2d = accel.conv2d

    try:
        img = Image.open(IMG_PATH)
        w, h = img.size
        crop = img.crop((0, 0, min(w, 1280), min(h, 128)))

        text_model = TM.PaperCRNNNumpy()
        author_model = AM.AuthorClassifierCNNNumpy()

        decoded = TM.ReadText(crop, text_model)
        pred_idx, probs = AM.ClassifyImage(crop, author_model, apply_stroke_normalize=False)

        return decoded, pred_idx, probs
    finally:
        L.lstm_forward = orig_lstm
        L.conv2d = orig_conv


def main():
    print("=== Step 1: decoded text / classification identity check ===")
    decoded_base, idx_base, probs_base = run_patched(use_lstm_patch=False, use_conv_patch=False)
    print(f"baseline (pure numpy):      decoded={decoded_base!r} predIdx={idx_base}")

    decoded_lstm, idx_lstm, probs_lstm = run_patched(use_lstm_patch=True, use_conv_patch=False)
    print(f"with C lstm_forward patch:  decoded={decoded_lstm!r} predIdx={idx_lstm}")

    assert decoded_base == decoded_lstm, "DECODED TEXT MISMATCH with lstm patch!"
    assert idx_base == idx_lstm, "CLASSIFICATION MISMATCH with lstm patch!"
    assert np.allclose(probs_base, probs_lstm, atol=1e-6), "probs mismatch with lstm patch!"
    print("PASS: identical decoded text and classification with C lstm_forward active.\n")

    print("=== Step 2: SynthesizeJointBestOf timing, baseline vs C lstm_forward ===")
    import np_inference.layers as L
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import accel_py as accel
    import SynthesizeHandwriting as SH

    profile = SH.LoadProfile("dylan")
    text = "speed test"

    print("Running baseline (pure numpy lstm_forward)...")
    t0 = time.perf_counter()
    SH.SynthesizeJointBestOf("dylan", text, profile, nTries=3, repair=False)
    baseline_time = time.perf_counter() - t0
    print(f"  baseline: {baseline_time:.2f}s")

    print("Running with C lstm_forward patched in...")
    orig_lstm = L.lstm_forward
    L.lstm_forward = accel.lstm_forward
    try:
        t0 = time.perf_counter()
        SH.SynthesizeJointBestOf("dylan", text, profile, nTries=3, repair=False)
        patched_time = time.perf_counter() - t0
    finally:
        L.lstm_forward = orig_lstm
    print(f"  with C lstm_forward: {patched_time:.2f}s")

    speedup = baseline_time / patched_time if patched_time > 0 else float("inf")
    print(f"\nSynthesizeJointBestOf(nTries=3): {baseline_time:.2f}s baseline -> "
          f"{patched_time:.2f}s with C lstm_forward ({speedup:.2f}x, "
          f"{(1 - patched_time/baseline_time)*100:.1f}% reduction)")


if __name__ == "__main__":
    main()
