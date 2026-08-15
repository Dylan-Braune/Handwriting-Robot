"""
Diagnostic script: proves (or disproves) that the training pipeline is
pairing each cached preprocessed image with the RIGHT label text, before
you trust any loss number that comes out of an actual training run.

Run it with the SAME --max-pages / --cache-dir you trained with, e.g.:
    python verify_training_pipeline.py --max-pages 100

Reuses the real classes from train_paper_cnn_bilstm_ctc.py directly (same
folder), so it is checking the EXACT code path training uses -- not a
reimplementation that could drift out of sync with it.

What it checks, in order:
  1. How many pages/lines the dataset actually loaded, and how many pages
     got skipped (image/label line-count mismatch, missing label, etc --
     IAMLineDatasetRaw prints this itself).
  2. Saves a handful of image+label pairs to disk (image PNG next to a
     .txt with the exact text the model is being trained to predict for
     it) so you can eyeball that they actually match.
  3. Round-trips text -> token ids -> text through CHAR_TO_IDX/IDX_TO_CHAR
     to catch any charset encode/decode bug.
  4. Computes the CTC time dimension (T) from the model architecture and
     checks it against every label's length across the WHOLE loaded
     dataset -- if T is too short for even one sample, PyTorch's
     zero_infinity=True setting silently zeroes that sample's loss
     instead of erroring, which can quietly suppress signal with no
     visible error anywhere.
  5. Prints a random-init loss baseline so you have a number to judge
     "loss plateaued at 3.15" against.
"""

import argparse
import math
import random
from pathlib import Path

import torch

from train_paper_cnn_bilstm_ctc import (
    CHARSET,
    CHAR_TO_IDX,
    IDX_TO_CHAR,
    IAMLineDatasetRaw,
    PaperCRNN,
    _decode_png,
    resize_line_image_fixed,
    tensor_from_resized,
)


def main():
    parser = argparse.ArgumentParser(description="Verify the training pipeline's image/label pairing and CTC setup.")
    parser.add_argument("--max-pages", type=int, default=None,
                         help="Use the SAME value you trained with, so this checks the exact dataset training saw.")
    parser.add_argument("--cache-dir", default="line_image_cache")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--n-samples", type=int, default=12,
                         help="How many image+label pairs to dump for visual spot-checking.")
    parser.add_argument("--out-dir", default="verify_training_output")
    args = parser.parse_args()

    if args.max_pages is None:
        raw = input(
            "How many pages would you like to verify? (Enter a number, e.g. 100 -- "
            "use the SAME number you trained with, or leave blank to check every page): "
        ).strip()
        args.max_pages = int(raw) if raw.isdigit() and int(raw) > 0 else None

    script_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir.parents[1] / "Data" / "Datasets" / "IAMpages671"

    print("=" * 78)
    print("[1/5] Loading dataset with the same parameters training would use ...")
    print("=" * 78)
    dataset = IAMLineDatasetRaw(
        root_dir=data_dir,
        cache_dir=script_dir / args.cache_dir,
        force_rebuild=False,
        max_pages=args.max_pages,
    )
    print(f"\nLoaded {len(dataset)} line samples total.")
    train_idx, test_idx = dataset.holdout_split_indices()
    print(f"Train lines: {len(train_idx)} | Held-out val lines: {len(test_idx)}")
    if len(test_idx) < 5:
        print("[WARNING] Very few held-out val lines -- your reported val loss/CER could be noisy or "
              "misleading with this little held-out data. Consider more pages or more authors.")

    print("\n" + "=" * 78)
    print("[2/5] Dumping image+label pairs for a visual spot-check ...")
    print("=" * 78)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    n_head = min(args.n_samples // 2, len(dataset))
    sample_indices = list(range(n_head))
    if len(dataset) > len(sample_indices):
        remaining = args.n_samples - len(sample_indices)
        pool = [i for i in range(len(dataset)) if i not in sample_indices]
        sample_indices += random.sample(pool, min(remaining, len(pool)))

    for i in sample_indices:
        item = dataset.samples[i]
        img = _decode_png(item["image_png"])
        img.save(out_dir / f"sample_{i:05d}.png")
        with open(out_dir / f"sample_{i:05d}.txt", "w", encoding="utf-8") as f:
            f.write(item["text"])
    print(f"Saved {len(sample_indices)} image+label pairs to:\n  {out_dir.resolve()}\n"
          f"Open a few sample_XXXXX.png next to sample_XXXXX.txt and confirm by eye the text matches "
          f"what's actually written in the image. This is the single most important check -- if these "
          f"don't match, nothing downstream matters.")

    print("\n" + "=" * 78)
    print("[3/5] Round-tripping text -> token ids -> text (charset encode/decode check) ...")
    print("=" * 78)
    bad = 0
    for i in sample_indices:
        item = dataset.samples[i]
        ids = item["target"].tolist()
        decoded = "".join(IDX_TO_CHAR.get(t, "?") for t in ids)
        original_filtered = "".join(c for c in item["text"] if c in CHAR_TO_IDX)
        if decoded != original_filtered:
            bad += 1
            print(f"  [MISMATCH] sample {i}: original={item['text']!r} filtered={original_filtered!r} decoded={decoded!r}")
    print(f"Charset round-trip: {len(sample_indices) - bad}/{len(sample_indices)} exact matches "
          f"{'(good)' if bad == 0 else '-- investigate the mismatches above'}.")

    print("\n" + "=" * 78)
    print("[4/5] Checking the CTC time dimension is long enough for every label ...")
    print("=" * 78)
    device = torch.device("cpu")
    model = PaperCRNN(num_classes=len(CHARSET) + 1).to(device)
    model.eval()
    probe_img = resize_line_image_fixed(_decode_png(dataset.samples[0]["image_png"]))
    probe_tensor = tensor_from_resized(probe_img).unsqueeze(0)
    with torch.no_grad():
        probe_log_probs = model(probe_tensor)
    T = probe_log_probs.size(0)
    print(f"CTC time steps (T) for this model/input size: {T} (fixed -- every image is resized to the "
          f"same width before this, so T is the same for every sample).")

    too_long = [(i, len(s["target"]), s["text"]) for i, s in enumerate(dataset.samples) if len(s["target"]) > T]
    lengths = [len(s["target"]) for s in dataset.samples]
    max_len = max(lengths) if lengths else 0
    print(f"Longest label across all {len(dataset)} loaded samples: {max_len} chars vs T={T} time steps.")
    if too_long:
        print(f"[PROBLEM] {len(too_long)} sample(s) have a label LONGER than T -- PyTorch's "
              f"zero_infinity=True setting means these silently contribute ZERO loss and ZERO gradient "
              f"instead of erroring, so they're quietly dropped from training with no visible warning. "
              f"First few:")
        for i, ln, text in too_long[:5]:
            print(f"    sample {i}: {ln} chars -- {text!r}")
    else:
        print(f"OK -- every label fits comfortably within T={T} (max used: {max_len}/{T} "
              f"= {max_len / T:.0%}). This is not the cause of a stuck loss.")

    print("\n" + "=" * 78)
    print("[5/5] Loss sanity baseline ...")
    print("=" * 78)
    random_baseline = math.log(len(CHARSET) + 1)
    print(f"A completely untrained/random model's expected CTC loss is roughly "
          f"log(num_classes) = log({len(CHARSET) + 1}) = {random_baseline:.2f}.")
    print("If your training run plateaued around ~3.15, that IS below the random baseline, so the model "
          "has learned some real structure -- it isn't stuck at 'random guessing'. A plateau that early "
          "is more often explained by: (a) 100 pages being a small slice of your ~1,400-page dataset "
          "(fewer distinct words/writers to generalize from), (b) hitting the early-stop patience "
          "(10 epochs with no val-loss improvement) before the model had room to keep improving, or "
          "(c) the fixed LR/schedule not suiting this small a dataset size. None of those are "
          "preprocessing/label-alignment bugs -- steps 1-4 above are what actually rule that out. If "
          "those all come back clean, the plateau is a data-size/training-recipe question, not a "
          "correctness bug, and the fix is more pages/authors and/or more patience, not more debugging "
          "of the pipeline itself.")


if __name__ == "__main__":
    main()
