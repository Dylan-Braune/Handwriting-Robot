"""
One-time migration: converts the old monolithic `iam_lines_cache_raw.pt`
cache (built by an earlier version of train_paper_cnn_bilstm_ctc.py) into
the new sharded cache format that script now expects:
    line_cache_raw/manifest.pt
    line_cache_raw/shards/shard_000000.pt, shard_000001.pt, ...

Why this exists: the old cache stored every line sample in one growing
Python list inside a single .pt file, and rewrote that *entire* file on
every checkpoint. Once the dataset grew toward ~10,000 samples, each
checkpoint became a multi-GB disk write -- which is what was hammering your
machine every couple of minutes. The new format writes small per-checkpoint
shard files plus a tiny manifest instead, so checkpoints stay cheap no
matter how big the dataset gets.

This script does a one-time, in-memory conversion of your *existing* cache
into that new layout -- no re-OCR, no re-segmentation, just a re-save. It
also PNG-compresses each line image while it's at it (mostly-white
handwriting scans compress well), which should noticeably shrink the
on-disk size versus the old raw-array storage.

Usage:
    python migrate_line_cache.py
    python migrate_line_cache.py --old iam_lines_cache_raw.pt --new line_cache_raw
"""

import argparse
import io
import os
from pathlib import Path

import torch

SHARD_SIZE = 300  # samples per shard file -- arbitrary, just keeps files small


def encode_png(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Migrate the old monolithic line cache to the new sharded format.")
    parser.add_argument("--old", default="iam_lines_cache_raw.pt", help="Path to the existing monolithic cache.")
    parser.add_argument("--new", default="line_cache_raw", help="Directory for the new sharded cache.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    old_path = Path(args.old)
    if not old_path.is_absolute():
        old_path = script_dir / old_path
    new_dir = Path(args.new)
    if not new_dir.is_absolute():
        new_dir = script_dir / new_dir

    if not old_path.exists():
        raise FileNotFoundError(
            f"Old cache not found at {old_path}. If it's named differently, pass --old <path>."
        )

    print(f"[Migrate] Loading old cache from {old_path} -- this is the slow/heavy part, "
          f"reads the whole file once, then we never touch it this way again.")
    old_cache = torch.load(old_path, weights_only=False)
    old_samples = old_cache.get("samples", [])
    author_folders = old_cache.get("author_folders", [])
    author_to_idx = old_cache.get("author_to_idx", {})
    processed_pages = old_cache.get("processed_pages", [])
    print(f"[Migrate] Loaded {len(old_samples)} samples across {len(processed_pages)} pages.")

    if not old_samples:
        print("[Migrate] Old cache has no samples -- nothing to migrate.")
        return

    shards_dir = new_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = new_dir / "manifest.pt"

    shard_files = []
    n_shards = (len(old_samples) + SHARD_SIZE - 1) // SHARD_SIZE
    for shard_idx in range(n_shards):
        chunk = old_samples[shard_idx * SHARD_SIZE : (shard_idx + 1) * SHARD_SIZE]
        new_chunk = []
        for sample in chunk:
            # Old samples stored a live PIL image under "pil_image"; new
            # samples store PNG-encoded bytes under "image_png". Handle
            # either, in case this is re-run against an already-converted
            # or partially-converted source.
            if "image_png" in sample:
                image_png = sample["image_png"]
            else:
                image_png = encode_png(sample["pil_image"])
            new_chunk.append(
                {
                    "image_png": image_png,
                    "target": sample["target"],
                    "text": sample["text"],
                    "is_holdout": sample["is_holdout"],
                }
            )

        shard_name = f"shard_{shard_idx:06d}.pt"
        shard_path = shards_dir / shard_name
        tmp_path = shard_path.with_suffix(".tmp")
        torch.save(new_chunk, tmp_path)
        os.replace(tmp_path, shard_path)
        shard_files.append(shard_name)
        print(f"\r[Migrate] Wrote {shard_idx + 1}/{n_shards} shards", end="", flush=True)
    print()

    tmp_manifest = manifest_path.with_suffix(".tmp")
    torch.save(
        {
            "author_folders": author_folders,
            "author_to_idx": author_to_idx,
            "processed_pages": processed_pages,
            "shard_files": shard_files,
        },
        tmp_manifest,
    )
    os.replace(tmp_manifest, manifest_path)

    old_size_mb = old_path.stat().st_size / (1024 * 1024)
    new_size_mb = sum(f.stat().st_size for f in shards_dir.glob("*.pt")) / (1024 * 1024)
    new_size_mb += manifest_path.stat().st_size / (1024 * 1024)

    print(f"\n[Migrate] Done: {len(old_samples)} samples -> {len(shard_files)} shard file(s) in {new_dir}")
    print(f"[Migrate] Old cache: {old_size_mb:.1f} MB   New cache: {new_size_mb:.1f} MB")
    print(f"[Migrate] train_paper_cnn_bilstm_ctc.py will now pick up {new_dir} automatically and skip "
          f"straight to training on the {len(processed_pages)} pages already processed.")
    print(f"[Migrate] Once you've confirmed a training run picks this up correctly, "
          f"it's safe to delete the old file at {old_path}.")


if __name__ == "__main__":
    main()
