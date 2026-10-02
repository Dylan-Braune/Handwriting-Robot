"""
CNN-BiLSTM-CTC line recognizer, reproducing the architecture and training
recipe from:

    Kizilirmak & Yanikoglu, "CNN-BiLSTM model for English Handwriting
    Recognition: Comprehensive Evaluation on the IAM Dataset" (2023)
    https://arxiv.org/html/2307.00664

WHAT'S REPLICATED FROM THE PAPER
---------------------------------
Backbone (Section 3.1 + Table 1 -- the best-scoring config, 12 conv layers /
2 max-pools, CER 5.01% on their validation split):
  - 12 conv layers, all 3x3 kernels, BatchNorm + ReLU after every conv.
  - Filters go 32 -> 64 -> 128 across three stages of 2, 4, and 6 conv
    layers respectively (2 + 4 + 6 = 12).
  - A 2x2 max-pool is applied exactly twice: once after the 32-filter stage,
    once after the 64-filter stage. The 128-filter stage has no further
    pooling (this is what "2 max pooling operations" means in Table 1).
  - The remaining height dimension is then collapsed with a single
    adaptive max-pool, and the tensor is permuted into a (W, D) sequence
    of feature vectors -- exactly as described in Section 3.1.

Sequence encoder (Section 3.2 + Table 2 -- the best-scoring config):
  - Two-layer bidirectional LSTM, hidden size 256.
  - No dropout in the LSTM (the paper is explicit about this: "We used two
    BiLSTM layers having 256 hidden nodes, without any dropout applied").

Decoder: a linear layer + log_softmax -> CTC loss / greedy CTC decoding
(Section 3.3). The paper's best numbers use word-beam-search decoding with
an external lexicon (Brown + WikiText2 + IAM train vocab) -- that requires
extra infra (a beam-search decoder library + corpora) not present in this
repo, so this script decodes greedily instead, same as your other scripts.
Swapping in a lexicon-constrained beam search later (e.g. pyctcdecode) is a
reasonable follow-up if you want to chase the paper's exact CER/WER.

Preprocessing (Section 6.1 "Experimental Setup"):
  - Images resized to a FIXED 100 (H) x 960 (W), ignoring aspect ratio. The
    paper explicitly says they tried aspect-ratio-preserving resize+pad
    first and got *worse* results with it, which is the opposite of what
    your other scripts do -- worth keeping in mind if you compare results.
  - As a side benefit, fixing the size (rather than padding a variable
    amount of blank canvas per sample) removes the per-sample contrast
    normalization instability that variable-width padding causes -- see the
    resize_line_image_fixed() docstring below.

Training recipe (Section 6.1):
  - RMSprop optimizer, lr=1e-3, weight_decay=1e-5, batch_size=16.
  - Up to 200 epochs, early stopping after 10 epochs with no validation
    LOSS improvement (not val accuracy -- the paper is specific about this).
  - No explicit LR schedule beyond the fixed initial rate ("tailored in
    some cases" but nothing further specified), so none is added here.

Data augmentation (Section 4.2 + Table 3 -- their best "combined" recipe):
  - At most ONE augmentation is applied per training image, each chosen
    with independent probability p=0.5 from: shear (k in [-0.6, 0.6]),
    rotation (angle in [-2.5, 2.5] degrees), elastic distortion
    (sigma in {3, 4}, alpha in {15, 20}), and a geometric/perspective
    transform standing in for the paper's Luo-et-al. "geometric
    transformations" category (their exact method needs an external repo
    this project doesn't have -- RandomPerspective is a reasonable
    approximation of the same spirit: mild non-affine geometric jitter).

WHAT REUSES YOUR EXISTING PIPELINE
------------------------------------
Line segmentation and label alignment are NOT reimplemented here. This
script calls the exact same ExtractIAMLines.ExtractLinePatches /
ReadLabelLines functions your other scripts use, over the same
IAMpages671 page images + hand/OCR-verified *_labels.txt files. The only
difference is which crop it keeps: ExtractLinePatches already returns each
line twice -- an aspect-ratio-preserving "processed_patch" (what your other
scripts train on) and a "raw_crop" (the untouched line region before any
resize/pad). This script uses raw_crop and does its own fixed-size stretch,
to match the paper's stated preprocessing.

NOT REPLICATED (out of scope / needs infra this repo doesn't have)
------------------------------------
  - Pretraining on ~2.5M synthetic handwriting images (Section 4.1).
  - Word-beam-search decoding with an external lexicon (Section 6.5).
  - Test-time augmentation (Section 5).

Usage:
    python TrainText.py
"""

import argparse
import io
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pytesseract
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ExtractIAMLines import ExtractLinePatches, ReadLabelLines

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -----------------------------------------------------------------------------
# Merged in from TrainTextJoint.py / TrainTextPersonal.py: both need to import
# SegmentPage (the personal-page pipeline) and TrainTextJoint.py additionally
# put AuthorReproductionStuff on sys.path -- replicated here unchanged so the
# merged file behaves exactly like running either original script directly,
# regardless of the caller's working directory.
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "AuthorReproductionStuff"))

import SegmentPage as PS
from authors_config import PERSONAL_AUTHORS, VAL_FRACTION, SPLIT_SEED


# -----------------------------------------------------------------------------
# Dual logging (console + file), matching the pattern in your other scripts.
# -----------------------------------------------------------------------------
class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


# 0 is reserved for the CTC blank token. Same charset as the rest of your
# codebase so label files / cached datasets stay compatible.
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:'\"!?()-"
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARSET)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}


def encode_text(text):
    return torch.tensor([CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX], dtype=torch.long)


def decode_ctc(log_probs):
    """Greedy CTC decode. The paper's best numbers use word-beam-search with
    a lexicon instead -- see the module docstring."""
    best_path = torch.argmax(log_probs, dim=2).detach().cpu().numpy()
    decoded = []
    for batch_idx in range(best_path.shape[1]):
        prev, chars = None, []
        for token in best_path[:, batch_idx]:
            token = int(token)
            if token != 0 and token != prev:
                chars.append(IDX_TO_CHAR.get(token, ""))
            prev = token
        decoded.append("".join(chars))
    return decoded


def levenshtein(a, b):
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    previous = list(range(len(b) + 1))
    for ca in a:
        current = [previous[0] + 1]
        for j, cb in enumerate(b):
            current.append(min(current[j] + 1, previous[j + 1] + 1, previous[j] + (ca != cb)))
        previous = current
    return previous[-1]


# -----------------------------------------------------------------------------
# Preprocessing: fixed stretch (Section 6.1 used 100x960), not aspect-
# preserving pad-to-canvas. Because every sample ends up exactly the same
# size with no blank padding region, normalization stats aren't skewed by
# how much of the canvas happens to be padding for a given line -- unlike a
# padded-canvas scheme where per-sample mean/std are dominated by background
# for short lines and by ink for long ones. We still use a *fixed*
# normalization constant (not per-sample mean/std) to keep this consistent.
#
# NOTE: this deviates from the paper's exact 100x960 -- deliberately halved
# to 64x640 (same ~1:10 aspect ratio, ~2.3x fewer pixels) since 100x960 on
# CPU was taking ~1hr/epoch, which isn't workable. This does NOT require
# rebuilding line_cache_raw/ -- the cache stores raw unresized crops and
# this resize happens live in __getitem__, so changing these two constants
# takes effect immediately on your existing cache. If you have a real GPU
# available later, bumping these back toward the paper's 100x960 (or higher,
# given your 8MP camera has plenty of native detail to support it) is a
# reasonable thing to revisit.
# -----------------------------------------------------------------------------
INPUT_HEIGHT = 64
INPUT_WIDTH = 640

# The CTC time dimension T is fixed by the model architecture: two 2x2
# max-pools halve the width twice, so T = INPUT_WIDTH // 4 = 160 for the
# current settings, regardless of what's actually in a given image. Any
# label longer than T is mathematically impossible for CTC to align (and
# with zero_infinity=True, PyTorch just silently zeroes that sample's loss
# instead of erroring) -- and in practice a "line" label anywhere near that
# long is a sign of collapsed/merged line segmentation, not a real single
# handwritten line. Cap well under T so genuinely-long-but-legitimate lines
# still get through while collapsed-multi-line labels get filtered out.
MAX_SAFE_LABEL_CHARS = int((INPUT_WIDTH // 4) * 0.75)  # 120 at current settings


def resize_line_image_fixed(pil_img, height=INPUT_HEIGHT, width=INPUT_WIDTH,
                            pad_value=255):
    """Fit the line into the fixed training canvas by UNIFORM scaling plus
    white padding -- never by squashing width and height independently to
    hit (640, 64), which is what a plain .resize() to a fixed size does.

    Measured on real line crops: they sit around 14:1 width:height, not the
    canvas's 10:1, so the old approach scaled height MORE than width (0.48x
    vs 0.35x on a typical crop) -- every letter came out narrower and taller
    than the writer actually made it, which is a real distortion of the
    exact stroke proportions the recognizer (and, via forced alignment,
    every downstream glyph cut) has to judge.

    Content is scaled by s = min(height/origH, width/origW) -- whichever
    axis is the binding constraint -- and left-aligned at x=0, vertically
    centered. Left-alignment matters beyond cosmetics: it's what makes
    `frame_x_to_pixel` below a correct inverse of this placement, which the
    forced-alignment cut boundaries depend on."""
    img = pil_img.convert("L")
    w, h = img.size
    if w <= 0 or h <= 0:
        return Image.new("L", (width, height), pad_value)
    s = min(height / h, width / w)
    new_w = max(1, round(w * s))
    new_h = max(1, round(h * s))
    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (width, height), pad_value)
    canvas.paste(resized, (0, max(0, (height - new_h) // 2)))
    return canvas


def frame_x_to_pixel(frameX, origW, origH, width=INPUT_WIDTH, height=INPUT_HEIGHT):
    """Inverse of resize_line_image_fixed's placement, for x only.

    A CTC frame index t maps to canvas x-coordinate frameX = t/T*width
    (the model's width-only downsampling is uniform). This converts that
    canvas x back to a pixel x-coordinate in the ORIGINAL, un-resized line
    image, which is what every forced-alignment cut boundary is expressed
    in. Must use the exact same scale factor resize_line_image_fixed used,
    or cuts silently drift once the resize stops being a plain stretch."""
    s = min(height / max(1, origH), width / max(1, origW))
    return frameX / max(1e-6, s)


def tensor_from_resized(pil_img):
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = 1.0 - arr  # ink -> high value, background -> 0, matching your other scripts
    arr = (arr - 0.5) / 0.5  # fixed normalization, roughly [-1, 1]
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)


# -----------------------------------------------------------------------------
# Data augmentation (Section 4.2 / Table 3 "combined" recipe): pick at most
# ONE of these per image, each gated by its own p=0.5 coin flip, so most
# images get either nothing or exactly one mild deformation.
# -----------------------------------------------------------------------------
def _augment_shear(img):
    k = random.uniform(-0.6, 0.6)
    shear_deg = math.degrees(math.atan(k))
    return TF.affine(img, angle=0.0, translate=(0, 0), scale=1.0, shear=[shear_deg, 0.0], fill=255)


def _augment_rotate(img):
    angle = random.uniform(-2.5, 2.5)
    return TF.affine(img, angle=angle, translate=(0, 0), scale=1.0, shear=[0.0, 0.0], fill=255)


def _augment_perspective(img):
    # Stand-in for the paper's Luo et al. "geometric transformations" (their
    # exact implementation lives in an external repo not vendored here).
    w, h = img.size
    distortion_scale = random.uniform(0.03, 0.08)
    startpoints, endpoints = torchvision_perspective_points(w, h, distortion_scale)
    return TF.perspective(img, startpoints, endpoints, fill=255)


def torchvision_perspective_points(width, height, distortion_scale):
    half_w, half_h = width // 2, height // 2
    topleft = (0, 0)
    topright = (width - 1, 0)
    botright = (width - 1, height - 1)
    botleft = (0, height - 1)
    startpoints = [topleft, topright, botright, botleft]

    def jitter(x, y, max_dx, max_dy):
        return (
            int(x + random.uniform(-max_dx, max_dx)),
            int(y + random.uniform(-max_dy, max_dy)),
        )

    max_dx = distortion_scale * half_w
    max_dy = distortion_scale * half_h
    endpoints = [
        jitter(*topleft, max_dx, max_dy),
        jitter(*topright, max_dx, max_dy),
        jitter(*botright, max_dx, max_dy),
        jitter(*botleft, max_dx, max_dy),
    ]
    return startpoints, endpoints


def _augment_elastic(img):
    # Simard et al. elastic distortion, as described in Section 4.2: sample
    # a displacement field, smooth it with a Gaussian of std sigma, scale it
    # by alpha, then warp the image by that field.
    sigma = random.choice([3, 4])
    alpha = random.choice([15, 20])
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape

    dx = (np.random.rand(h, w) * 2 - 1)
    dy = (np.random.rand(h, w) * 2 - 1)
    dx = gaussian_filter(dx, sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(dy, sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    indices = (np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)))
    warped = map_coordinates(arr, indices, order=1, mode="constant", cval=255.0).reshape(h, w)
    return Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8))


_AUGMENTATIONS = [_augment_shear, _augment_rotate, _augment_elastic, _augment_perspective]


def augment_line_image(pil_img):
    """At most one augmentation, each independently gated at p=0.5, matching
    the paper's Table 3 "combined" recipe (their best-performing setting).

    IMPORTANT: call this on an already-resized (INPUT_HEIGHT x INPUT_WIDTH)
    image, not the raw scan-resolution crop. _augment_elastic in particular
    allocates several full-resolution float64 arrays proportional to the
    image's pixel count -- on a raw crop from a high-DPI scan/photo that can
    be a very large, unbounded allocation (this is what caused the
    MemoryError crash: a single oversized raw crop, multiplied across
    several parallel DataLoader workers). Resizing first bounds every
    augmentation's cost to a small, fixed, known size."""
    candidates = [fn for fn in _AUGMENTATIONS if random.random() < 0.5]
    if not candidates:
        return pil_img
    return random.choice(candidates)(pil_img)


# -----------------------------------------------------------------------------
# Image storage: PNG-encoded bytes rather than raw pixel arrays or live PIL
# objects. Handwriting scans are mostly white background, which PNG (a
# lossless, run/entropy-coded format) compresses very well -- this is what
# keeps the on-disk cache from ballooning the way raw arrays did.
# -----------------------------------------------------------------------------
def _encode_png(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _decode_png(png_bytes):
    return Image.open(io.BytesIO(png_bytes)).convert("L")


# -----------------------------------------------------------------------------
# LineImageCache: stores ONLY the preprocessed line-crop PNG bytes, keyed by
# page and line index -- completely independent of label TEXT. Labels are
# read fresh from each page's _labels.txt every run (cheap: a text-file
# parse, no OCR, no cropping) and zipped against these cached images by
# line index in IAMLineDatasetRaw below. This is the whole point of
# splitting it out: regenerating/correcting label text NEVER requires
# re-running line segmentation or re-encoding images. Only rerun/rebuild
# THIS cache if the segmentation code itself changes (e.g. a change to
# ExtractIAMLines's line-detection logic) -- not when labels change.
#
# Built with expectedLineCount=None / labelLines=None on purpose, so the
# detected line boundaries never depend on any particular label file's
# line count -- that's what keeps images and labels genuinely independent.
# If a page's CURRENT label file ends up with a different number of lines
# than this cache detected (e.g. an older, not-yet-regenerated label using
# a different line count), IAMLineDatasetRaw skips just that one page with
# a warning rather than silently misaligning images with the wrong text.
#
# Same incremental shard+manifest checkpointing as before: `manifest.pt`
# tracks which pages are cached and which shard files exist; each
# checkpoint writes only newly-collected pages into a fresh shard file and
# never rewrites previously-saved data.
# -----------------------------------------------------------------------------
class LineImageCache:
    def __init__(self, root_dir, cache_dir="line_image_cache", force_rebuild=False,
                 save_every_n_pages=10, max_pages=None):
        self.cache_dir = Path(cache_dir)
        self.shards_dir = self.cache_dir / "shards"
        self.manifest_path = self.cache_dir / "manifest.pt"
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        # page_key ("author/page.png") -> list of PNG bytes, one per
        # detected line, in top-to-bottom order.
        self.pages = {}
        processed_pages = set()
        shard_files = []

        if self.manifest_path.exists() and not force_rebuild:
            print(f"[ImageCache] Loading manifest from: {self.manifest_path}")
            manifest = torch.load(self.manifest_path, weights_only=False)
            processed_pages = set(manifest.get("processed_pages", []))
            shard_files = manifest.get("shard_files", [])
            for shard_name in shard_files:
                shard_path = self.shards_dir / shard_name
                if shard_path.exists():
                    self.pages.update(torch.load(shard_path, weights_only=False))
            print(f"[ImageCache] Loaded images for {len(self.pages)} page(s) from {len(shard_files)} shard file(s).")

        def save_manifest():
            tmp_path = self.manifest_path.with_suffix(".tmp")
            torch.save(
                {"processed_pages": list(processed_pages), "shard_files": shard_files},
                tmp_path,
            )
            os.replace(tmp_path, self.manifest_path)

        def flush_pending(pending_pages):
            """Write only the newly-collected pages' images into their own
            shard file, then update the manifest. Previously-written
            shards are never touched again."""
            if not pending_pages:
                return
            shard_name = f"shard_{len(shard_files):06d}.pt"
            shard_path = self.shards_dir / shard_name
            tmp_path = shard_path.with_suffix(".tmp")
            torch.save(pending_pages, tmp_path)
            os.replace(tmp_path, shard_path)
            shard_files.append(shard_name)
            save_manifest()
            print(f"\n[ImageCache Checkpoint] Flushed {len(pending_pages)} new page(s) to {shard_name} "
                  f"({len(self.pages)} pages cached total).")

        author_folders = sorted(p.name for p in root_dir.iterdir() if p.is_dir())
        all_pages = []
        for author_id in author_folders:
            author_dir = root_dir / author_id
            for img_path in sorted(author_dir.glob("*.png")):
                all_pages.append((author_id, img_path))

        if max_pages is not None and max_pages > 0:
            all_pages = all_pages[:max_pages]

        total_files = len(all_pages)
        page_counter = 0
        pending_pages = {}
        try:
            for idx, (author_id, img_path) in enumerate(all_pages, 1):
                page_key = f"{author_id}/{img_path.name}"
                if page_key in processed_pages:
                    continue

                pct = (idx / total_files) * 100.0 if total_files > 0 else 0.0
                print(f"\r[ImageCache Preprocessing] Progress: {pct:5.1f}% ({idx}/{total_files} pages)", end="", flush=True)

                try:
                    # expectedLineCount=None / labelLines=None: segmentation
                    # stands entirely on its own, never shaped by whatever a
                    # label file happens to say -- see comment above.
                    line_samples, _, _ = ExtractLinePatches(
                        str(img_path),
                        targetHeight=32,
                        maxWidth=1024,
                        expectedLineCount=None,
                        labelLines=None,
                    )
                except Exception as e:
                    print(f"\n[ImageCache] Skipping {page_key} (segmentation error: {e})")
                    line_samples = []

                png_list = []
                for sample in line_samples:
                    pil_line = Image.fromarray(sample["raw_crop"]).convert("L")
                    png_list.append(_encode_png(pil_line))

                self.pages[page_key] = png_list
                pending_pages[page_key] = png_list
                processed_pages.add(page_key)
                page_counter += 1
                if page_counter % save_every_n_pages == 0:
                    flush_pending(pending_pages)
                    pending_pages = {}

            if page_counter > 0:
                print()
        except KeyboardInterrupt:
            print("\n[Paused] Interrupted! Flushing pending page images before exiting.")
            flush_pending(pending_pages)
            raise

        # Catches whatever's left over from the last incomplete batch of
        # save_every_n_pages (e.g. the run ended mid-batch).
        if pending_pages:
            flush_pending(pending_pages)

        print(f"[ImageCache] Total pages with cached images: {len(self.pages)}")

    def get(self, page_key):
        return self.pages.get(page_key)


# -----------------------------------------------------------------------------
# Dataset: combines LineImageCache's label-independent cached images with
# each page's CURRENT label text (always re-read fresh from _labels.txt)
# into the final list of training samples. Because that combining step is
# cheap (dict lookups + text encoding, no OCR, no cropping, no PNG
# encoding), it's simply redone every run instead of cached -- which is
# what makes label regeneration free: correct a page's _labels.txt and
# rerun training, and only this cheap zip-together step reruns. You only
# need to delete/rebuild line_image_cache/ if the segmentation code
# itself changes, never when only label text changes.
# -----------------------------------------------------------------------------
class IAMLineDatasetRaw(Dataset):
    def __init__(
        self,
        root_dir,
        min_pages_for_holdout=3,
        cache_dir="line_image_cache",
        force_rebuild=False,
        save_every_n_pages=10,
        max_pages=None,
        is_train=True,
    ):
        self.samples = []
        self.is_train = is_train

        root_dir = self.resolve_dataset_root(Path(root_dir))

        image_cache = LineImageCache(
            root_dir,
            cache_dir=cache_dir,
            force_rebuild=force_rebuild,
            save_every_n_pages=save_every_n_pages,
            max_pages=max_pages,
        )

        self.author_folders = sorted(p.name for p in root_dir.iterdir() if p.is_dir())
        self.author_to_idx = {author: idx for idx, author in enumerate(self.author_folders)}

        mismatched_pages = 0
        oversized_labels = 0
        for author_id in self.author_folders:
            author_dir = root_dir / author_id
            author_pages = sorted(author_dir.glob("*.png"))
            if not author_pages:
                continue
            labeled_pages = [p for p in author_pages if ReadLabelLines(str(p))]
            holdout_page_name = labeled_pages[-1].name if len(labeled_pages) >= min_pages_for_holdout else None

            for img_path in author_pages:
                page_key = f"{author_id}/{img_path.name}"
                label_lines = ReadLabelLines(str(img_path))
                if not label_lines:
                    continue

                png_list = image_cache.get(page_key)
                if png_list is None:
                    continue  # not in the image cache (e.g. max_pages truncation)

                if len(png_list) != len(label_lines):
                    mismatched_pages += 1
                    print(f"[Dataset] Skipping {page_key}: cached image has {len(png_list)} line(s) but "
                          f"the current label has {len(label_lines)} line(s) -- segmentation and label "
                          f"disagree on line count for this page. This page's images need re-cropping "
                          f"(delete/rebuild line_image_cache/ for it) before it can be used.")
                    continue

                is_holdout = img_path.name == holdout_page_name
                for line_idx in range(len(label_lines)):
                    text = label_lines[line_idx]
                    target = encode_text(text)
                    if len(target) == 0:
                        continue
                    # A "line" label this long almost certainly isn't one physical
                    # handwritten line -- it's a sign that line segmentation
                    # collapsed several real lines into one detected region (e.g.
                    # tight line spacing) and the alignment step then dumped that
                    # whole run of printed words into a single bucket. A single
                    # handwritten line crop has no realistic way to legibly fit
                    # this much text, and the CTC time dimension (T, fixed by
                    # model architecture -- see PaperCRNN) can't fit it either:
                    # anything >= T silently contributes ZERO loss/gradient under
                    # zero_infinity=True, so letting it into training either does
                    # nothing (best case) or wastes a sample slot for nothing.
                    # Skip loudly instead of letting it fail silently.
                    if len(target) > MAX_SAFE_LABEL_CHARS:
                        oversized_labels += 1
                        print(f"[Dataset] Skipping oversized line label ({len(target)} chars, cap is "
                              f"{MAX_SAFE_LABEL_CHARS}) at {page_key} line {line_idx} -- likely merged/"
                              f"collapsed line segmentation on this page, not a real single line: "
                              f"{text[:80]!r}...")
                        continue
                    self.samples.append({
                        "page_key": page_key,
                        "line_idx": line_idx,
                        "image_png": png_list[line_idx],
                        "target": target,
                        "text": text,
                        "is_holdout": is_holdout,
                    })

        if mismatched_pages:
            print(f"[Dataset] {mismatched_pages} page(s) skipped due to image/label line-count mismatch.")
        if oversized_labels:
            print(f"[Dataset] {oversized_labels} line(s) skipped as oversized (likely collapsed/merged "
                  f"line segmentation -- see warnings above for which pages).")
        print(f"[Dataset] Total line samples: {len(self.samples)}")

    @staticmethod
    def resolve_dataset_root(root_dir):
        data_dir = root_dir / "data"
        return data_dir if data_dir.exists() and any(p.is_dir() for p in data_dir.iterdir()) else root_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        pil_img = resize_line_image_fixed(_decode_png(item["image_png"]))
        if self.is_train:
            pil_img = augment_line_image(pil_img)
        img_tensor = tensor_from_resized(pil_img)
        return img_tensor, item["target"], item["text"]

    def holdout_split_indices(self):
        train_idx = [i for i, s in enumerate(self.samples) if not s["is_holdout"]]
        test_idx = [i for i, s in enumerate(self.samples) if s["is_holdout"]]
        return train_idx, test_idx


class TrainEvalView(Dataset):
    """Thin wrapper so train/val subsets can independently toggle
    augmentation without deep-copying the whole underlying dataset (avoids
    the memory/time cost of copy.deepcopy over thousands of samples)."""

    def __init__(self, base_dataset, indices, is_train):
        self.base_dataset = base_dataset
        self.indices = indices
        self.is_train = is_train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        item = self.base_dataset.samples[self.indices[i]]
        pil_img = resize_line_image_fixed(_decode_png(item["image_png"]))
        if self.is_train:
            pil_img = augment_line_image(pil_img)
        img_tensor = tensor_from_resized(pil_img)
        return img_tensor, item["target"], item["text"]


def collate_fn(batch):
    images, targets, texts = zip(*batch)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flat_targets = torch.cat(targets)
    return torch.stack(images, 0), flat_targets, target_lengths, list(texts)


# -----------------------------------------------------------------------------
# Model: 12 conv layers / 2 max-pools (32 -> 64 -> 128 filters), height
# collapsed to 1 via adaptive max-pool, then a 2-layer BiLSTM (hidden=256,
# no dropout) and a linear + log_softmax CTC head. See Section 3 / Table 1 /
# Table 2 of the paper.
# -----------------------------------------------------------------------------
def conv_block(in_ch, out_ch, n_layers):
    layers = []
    for i in range(n_layers):
        layers += [
            nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


class PaperCRNN(nn.Module):
    def __init__(self, num_classes, lstm_hidden=256, lstm_layers=2,
                 conv_width_mult=1.0, dropout=0.0):
        """conv_width_mult and dropout are NEW, both default to exactly the
        original paper's architecture (1x width, no dropout) so every
        existing caller (BuildStyleProfile, ClassifyText, VerifyRewrite,
        ...) is unaffected. Added for TrainTextHF_Sweep.py's architecture
        search -- conv_width_mult scales all three conv stages' channel
        counts together (32/64/128 * mult, rounded), dropout applies
        between BiLSTM layers (only has an effect when lstm_layers > 1,
        same as PyTorch's own nn.LSTM dropout semantics) and after each
        conv stage."""
        super().__init__()
        w1 = max(4, round(32 * conv_width_mult))
        w2 = max(8, round(64 * conv_width_mult))
        w3 = max(16, round(128 * conv_width_mult))

        # Stage 1: 2 conv layers @ 32 filters, then 2x2 max-pool.
        self.stage1 = conv_block(1, w1, n_layers=2)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Stage 2: 4 conv layers @ 64 filters, then 2x2 max-pool.
        self.stage2 = conv_block(w1, w2, n_layers=4)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Stage 3: 6 conv layers @ 128 filters, no further pooling.
        self.stage3 = conv_block(w2, w3, n_layers=6)
        self.conv_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Collapse whatever height remains into a single row.
        self.height_pool = nn.AdaptiveMaxPool2d((1, None))

        self.sequence = nn.LSTM(
            input_size=w3,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,  # paper default: no dropout
            batch_first=False,
        )
        self.text_head = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.conv_dropout(self.stage3(x))
        x = self.height_pool(x)          # (B, W3, 1, W)
        x = x.squeeze(2).permute(2, 0, 1)  # (W, B, W3)
        seq, _ = self.sequence(x)
        logits = self.text_head(seq)
        return nn.functional.log_softmax(logits, dim=2)


# -----------------------------------------------------------------------------
# Evaluation: char accuracy (1 - CER) and average CTC loss, since the paper
# early-stops on validation LOSS specifically (not accuracy).
# -----------------------------------------------------------------------------
def evaluate(model, dataloader, device, ctc_loss_fn):
    model.eval()
    total_chars, total_char_errors = 0, 0
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for images, targets, target_lengths, texts in dataloader:
            images = images.to(device)
            targets_dev = targets.to(device)
            target_lengths_dev = target_lengths.to(device)

            log_probs = model(images)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long, device=device)
            loss = ctc_loss_fn(log_probs, targets_dev, input_lengths, target_lengths_dev)
            total_loss += float(loss.item())
            n_batches += 1

            predictions = decode_ctc(log_probs)
            for pred, truth in zip(predictions, texts):
                total_chars += len(truth)
                total_char_errors += levenshtein(pred, truth)

    cer = total_char_errors / max(1, total_chars)
    char_acc = 1.0 - cer
    avg_loss = total_loss / max(1, n_batches)
    return avg_loss, char_acc, cer


def train(config, device):
    data_dir = config["data_dir"]
    script_dir = Path(__file__).resolve().parent

    dataset = IAMLineDatasetRaw(
        root_dir=data_dir,
        cache_dir=script_dir / config["cache_dir"],
        force_rebuild=config.get("force_rebuild", False),
        max_pages=config.get("max_pages"),
    )
    if len(dataset) < 4:
        raise RuntimeError("Not enough line samples to train.")

    train_idx, test_idx = dataset.holdout_split_indices()
    if not test_idx:
        raise RuntimeError("No held-out lines found -- check min_pages_for_holdout / your label coverage.")

    train_set = TrainEvalView(dataset, train_idx, is_train=True)
    val_set = TrainEvalView(dataset, test_idx, is_train=False)
    print(f"[Split] Train lines: {len(train_set)} | Held-out val lines: {len(val_set)}")

    is_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn,
        num_workers=4 if is_cuda else 0, pin_memory=is_cuda,
    )
    val_loader = DataLoader(
        val_set, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn,
        num_workers=4 if is_cuda else 0, pin_memory=is_cuda,
    )

    model = PaperCRNN(num_classes=len(CHARSET) + 1).to(device)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    # Section 6.1: RMSprop, lr=1e-3, weight_decay=1e-5, batch_size=16.
    optimizer = optim.RMSprop(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # Deviation from the paper's literal fixed-LR recipe: on their much
    # larger IAM training split, validation loss moves smoothly enough that
    # a fixed LR + patience=10 early stop works. On your ~10k-line dataset,
    # validation loss is noisier epoch-to-epoch (see the epoch 26-35 log --
    # a tight, non-monotonic band around the epoch-25 best, not a real
    # ceiling), so a fixed high LR keeps it bouncing around a basin instead
    # of settling into it. Decaying the LR when validation loss plateaus
    # lets it actually settle; the paper's own text even notes they
    # "tailored" the LR in some experiments, so this isn't out of spirit.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)

    weights_dir = script_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = weights_dir / f"{config['name']}_checkpoint.pt"
    best_path = weights_dir / f"{config['name']}_best.pt"

    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    restart_count = 0

    # Resume full training state (model + optimizer + counters), not just
    # weights -- a partial resume that only restores weights silently resets
    # RMSprop's running averages and the early-stopping counter, which can
    # visibly disrupt an in-progress training run.
    if ckpt_path.exists() and config.get("resume", True):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        patience_counter = ckpt["patience_counter"]
        restart_count = ckpt.get("restart_count", 0)
        print(f"[Resume] Loaded checkpoint from epoch {ckpt['epoch']}, resuming at epoch {start_epoch} "
              f"(best_val_loss={best_val_loss:.4f}, patience_counter={patience_counter}, "
              f"restart_count={restart_count}, lr={optimizer.param_groups[0]['lr']:.6f}).")

    max_epochs = config["epochs"]
    early_stop_patience = config["early_stop_patience"]
    eval_every = max(1, config.get("eval_every", 1))

    max_restarts = config.get("max_restarts", 8)


    n_train_batches = len(train_loader)

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        epoch_start = time.time()

        for images, targets, target_lengths, _texts in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            log_probs = model(images)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long, device=device)
            loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

            # Heartbeat so a slow/CPU epoch doesn't look like a hang -- see
            # running loss, batch progress, and time-per-batch every few
            # batches (not every batch, to keep the print overhead trivial).
            if n_batches == 1 or n_batches % 5 == 0 or n_batches == n_train_batches:
                elapsed = time.time() - epoch_start
                sec_per_batch = elapsed / n_batches
                eta_sec = sec_per_batch * (n_train_batches - n_batches)
                print(
                    f"\r[{config['name']}] Epoch {epoch:03d} | "
                    f"Batch {n_batches:04d}/{n_train_batches} | "
                    f"Running loss: {total_loss / n_batches:.3f} | "
                    f"{sec_per_batch:.2f}s/batch | ETA {eta_sec / 60:.1f} min",
                    end="",
                    flush=True,
                )

        print()  # move off the \r line before the epoch summary
        train_loss = total_loss / max(1, n_batches)
        print(f"[{config['name']}] Epoch {epoch:03d} training pass done in {(time.time() - epoch_start) / 60:.1f} min.")

        # Validation is skipped on epochs that aren't a multiple of eval_every
        # (always run on the very last epoch so you get a final read). This
        # matters because CER computation in evaluate() runs Levenshtein
        # distance in a pure-Python double loop per sample -- on CPU that can
        # genuinely take longer than the (vectorized, tensor-op) training
        # pass itself, especially with longer line labels. patience_counter
        # only advances on epochs where validation actually ran, so
        # early_stop_patience means "N validation checks with no improvement",
        # not strictly "N epochs" -- with eval_every=5 that's up to 5x more
        # wall-clock epochs between checks, which is the tradeoff for the
        # speedup.
        run_validation = (epoch % eval_every == 0) or (epoch == max_epochs)
        if run_validation:
            val_start = time.time()
            val_loss, val_char_acc, val_cer = evaluate(model, val_loader, device, ctc_loss_fn)
            val_elapsed = time.time() - val_start

            # NOTE: this was previously missing entirely -- the scheduler was
            # constructed but never stepped, so the LR silently stayed fixed
            # at its initial value for the whole run regardless of how long
            # val loss plateaued. ReduceLROnPlateau needs the val loss fed to
            # it explicitly on every validation check (not every epoch, if
            # eval_every > 1 -- its own internal patience counts in units of
            # "times .step() was called", so skipped epochs correctly don't
            # count against it either).
            lr_before = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            lr_after = optimizer.param_groups[0]["lr"]
            lr_msg = f" | LR reduced {lr_before:.6f} -> {lr_after:.6f}" if lr_after < lr_before else ""

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_path)
                saved_msg = " [BEST MODEL SAVED]"
            else:
                patience_counter += 1
                saved_msg = ""

            print(
                f"[{config['name']}] Epoch {epoch:03d}/{max_epochs} | "
                f"Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f} (Best: {best_val_loss:.3f}) | "
                f"Val char-acc: {val_char_acc:.2%} | Val CER: {val_cer:.2%} | "
                f"Val took {val_elapsed / 60:.1f} min{lr_msg}{saved_msg}"
            )
        else:
            print(f"[{config['name']}] Epoch {epoch:03d}/{max_epochs} | "
                  f"Train loss: {train_loss:.3f} | (validation skipped this epoch -- runs every "
                  f"{eval_every} epochs; last val loss: {best_val_loss:.3f} best-so-far)")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "restart_count": restart_count,
                "charset": CHARSET,
                "input_height": INPUT_HEIGHT,
                "input_width": INPUT_WIDTH,
            },
            ckpt_path,
        )

        # Section 6.1: early stop after 10 VALIDATION CHECKS with no val LOSS
        # improvement (not raw epochs, when eval_every > 1 -- see note above).
        #
        # WARM RESTART instead of stopping outright: rather than ending the
        # run the first time patience runs out, reload the best weights seen
        # so far, throw away the optimizer's momentum and the LR-decay
        # schedule's state (both reset to their fresh initial values), and
        # keep training. The idea: the LR may have decayed down to something
        # tiny by the time patience triggers (via the scheduler above), which
        # can itself be *why* progress stalled -- a fresh, larger LR applied
        # to the best-known weights sometimes finds a further improvement
        # that the decayed-LR run couldn't reach. best_val_loss/best_path
        # are NOT reset -- a restart only ever needs to beat the existing
        # best to matter, same as normal training. Capped at max_restarts
        # (config["max_restarts"], default 3) so a run that's genuinely done
        # improving still stops for good eventually rather than restarting
        # forever within the max_epochs budget.
        if run_validation and patience_counter >= early_stop_patience:
            if restart_count < max_restarts:
                restart_count += 1
                print(f"[{config['name']}] No val loss improvement for {early_stop_patience} validation "
                      f"check(s) -- WARM RESTART #{restart_count}/{max_restarts}: reloading best weights "
                      f"(val_loss={best_val_loss:.4f}) and resetting LR to {config['lr']:.6f}.")
                if best_path.exists():
                    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
                optimizer = optim.RMSprop(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-5)
                patience_counter = 0
                continue
            else:
                print(f"[{config['name']}] Early stopping for good: no val loss improvement for "
                      f"{early_stop_patience} validation check(s) after {max_restarts} warm restart(s) "
                      f"already used (eval_every={eval_every}).")
                break

    return best_val_loss


def main_base(args):
    """Original TrainText.py direct-run behavior: train on your own segmented
    IAMpages671 page scans. Unchanged except that `args` is now handed in by
    the subcommand dispatcher instead of being parsed here."""
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parents[1] / "Data" / "Datasets" / "IAMpages671"

    max_pages = args.max_pages
    if max_pages is None:
        raw = input(
            "How many pages would you like to train on? (Enter a number, e.g. 100, "
            "or leave blank to use every page available): "
        ).strip()
        max_pages = int(raw) if raw.isdigit() and int(raw) > 0 else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Training on: {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    # Redirect BOTH stdout and stderr through the same logger. Without this,
    # a crash's traceback only goes to stderr, which was never being written
    # to the .log file -- so a run could die with no visible explanation
    # anywhere except a terminal window you may not still have open.
    logger = DualLogger(str(script_dir / "TrainText.log"))
    sys.stdout = logger
    sys.stderr = logger

    config = {
        "name": "paper_cnn_bilstm_ctc",
        "data_dir": data_dir,
        "cache_dir": "NOGIT/line_image_cache",  # label-independent image cache; see LineImageCache
        "force_rebuild": args.force_rebuild,
        "max_pages": max_pages,
        "batch_size": 16,        # Section 6.1
        "lr": 1e-3,               # Section 6.1
        "weight_decay": 1e-5,     # Section 6.1
        "epochs": 200,            # Section 6.1

        "early_stop_patience": 15,  # Section 6.1

        "eval_every": args.eval_every,
        "max_restarts": args.max_restarts,
        "resume": True,
    }

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    pages_desc = f"first {config['max_pages']} page(s)" if config['max_pages'] else "all available pages"
    print(f"\n--- Training {config['name']} "
          f"(lr={config['lr']}, batch_size={config['batch_size']}, max_epochs={config['epochs']}, "
          f"data={pages_desc}) ---")
    try:
        best_val_loss = train(config, device)
        print(f"\nDone. Best validation loss: {best_val_loss:.4f}")
    except Exception:
        import traceback
        print("\n[FATAL] Training crashed. Full traceback below (also saved to the .log file):")
        traceback.print_exc()
        raise


# =============================================================================
# Merged in from TrainTextHF.py -- train on the pre-segmented HuggingFace
# Teklia/IAM-line dataset instead of our own page scans. Everything model-side
# (PaperCRNN, preprocessing, CHARSET, decode, augmentation, evaluate()) is the
# shared code above; only the data layer below is new.
# =============================================================================
WEIGHTS_DIR = SCRIPT_DIR / "weights"
HF_NAME = "paper_cnn_bilstm_ctc_hf"
HF_DATASET = "Teklia/IAM-line"
DEFAULT_HF_CACHE = SCRIPT_DIR / "NOGIT" / "hf_cache"


class HFLineDataset(Dataset):
    def __init__(self, hf_split, is_train, max_samples=None):
        self.hf = hf_split
        self.is_train = is_train

        # One pass over the text column only (no image decode) to find the
        # rows we can actually train on.
        texts = self.hf["text"]
        self.rows = []          # (hf_index, cleaned_label)
        dropped_empty = dropped_long = 0
        for i, t in enumerate(texts):
            kept = "".join(c for c in t.strip() if c in CHAR_TO_IDX)
            if len(kept) == 0:
                dropped_empty += 1
                continue
            if len(kept) > MAX_SAFE_LABEL_CHARS:
                dropped_long += 1
                continue
            self.rows.append((i, kept))

        if max_samples is not None and max_samples > 0:
            self.rows = self.rows[:max_samples]

        split_name = "train" if is_train else "eval"
        print(f"[HFData/{split_name}] {len(self.rows)} usable line(s) "
              f"(dropped {dropped_empty} empty-after-filter, {dropped_long} "
              f"over {MAX_SAFE_LABEL_CHARS} chars, from {len(texts)} total).")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        hf_idx, label = self.rows[i]
        pil_img = self.hf[hf_idx]["image"]                 # PIL, RGB, 128px
        pil_img = resize_line_image_fixed(pil_img)         # -> L, 640x64
        if self.is_train:
            pil_img = augment_line_image(pil_img)
        img_tensor = tensor_from_resized(pil_img)
        return img_tensor, encode_text(label), label


def load_hf(cache_dir):
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "The 'datasets' library is required. Install it with:\n"
            "    pip install datasets"
        )
    print(f"[HF] load_dataset('{HF_DATASET}')  (cache: {cache_dir})")
    return load_dataset(HF_DATASET, cache_dir=str(cache_dir))


# -----------------------------------------------------------------------------
# Training loop -- same recipe as the base train(): RMSprop lr=1e-3
# wd=1e-5, ReduceLROnPlateau on val loss, best-val-loss checkpointing, warm
# restarts, early stop after N validation checks with no improvement.
# -----------------------------------------------------------------------------
def train_hf(config, device):
    ds = load_hf(config["cache_dir"])
    train_set = HFLineDataset(ds["train"], is_train=True,
                              max_samples=config.get("max_samples"))
    val_set = HFLineDataset(ds["validation"], is_train=False,
                            max_samples=config.get("max_samples"))
    if len(train_set) < 4 or len(val_set) < 1:
        raise RuntimeError("Not enough usable samples to train.")

    is_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=4 if is_cuda else 0,
        pin_memory=is_cuda,
    )
    val_loader = DataLoader(
        val_set, batch_size=config["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=4 if is_cuda else 0,
        pin_memory=is_cuda,
    )

    model = PaperCRNN(num_classes=len(CHARSET) + 1).to(device)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.RMSprop(model.parameters(), lr=config["lr"],
                              weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = WEIGHTS_DIR / f"{HF_NAME}_checkpoint.pt"
    best_path = WEIGHTS_DIR / f"{HF_NAME}_best.pt"

    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    restart_count = 0

    if ckpt_path.exists() and config.get("resume", True):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        patience_counter = ckpt["patience_counter"]
        restart_count = ckpt.get("restart_count", 0)
        print(f"[Resume] from epoch {ckpt['epoch']}, resuming at {start_epoch} "
              f"(best_val_loss={best_val_loss:.4f}, patience={patience_counter}, "
              f"restarts={restart_count}, lr={optimizer.param_groups[0]['lr']:.6f}).")
    elif config.get("init_from"):
        # Warm start from an already-trained recogniser (same architecture and
        # charset) instead of random init. Nothing about the data changes --
        # this only saves the epochs that would be spent relearning generic
        # stroke features, which matters a lot without a GPU.
        src = Path(config["init_from"])
        if src.exists():
            sd = torch.load(src, map_location=device, weights_only=False)
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            model.load_state_dict(sd)
            print(f"[Init] warm-started from {src.name}")
        else:
            print(f"[Init] {src} not found -- training from scratch")

    max_epochs = config["epochs"]
    early_stop_patience = config["early_stop_patience"]
    eval_every = max(1, config.get("eval_every", 1))
    max_restarts = config.get("max_restarts", 8)
    n_train_batches = len(train_loader)

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        epoch_start = time.time()

        for images, targets, target_lengths, _texts in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            log_probs = model(images)
            input_lengths = torch.full((images.size(0),), log_probs.size(0),
                                       dtype=torch.long, device=device)
            loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            if n_batches == 1 or n_batches % 5 == 0 or n_batches == n_train_batches:
                elapsed = time.time() - epoch_start
                spb = elapsed / n_batches
                eta = spb * (n_train_batches - n_batches)
                print(f"\r[{HF_NAME}] Epoch {epoch:03d} | Batch {n_batches:04d}/"
                      f"{n_train_batches} | Running loss {total_loss / n_batches:.3f} "
                      f"| {spb:.2f}s/batch | ETA {eta / 60:.1f} min",
                      end="", flush=True)

        print()
        train_loss = total_loss / max(1, n_batches)
        print(f"[{HF_NAME}] Epoch {epoch:03d} training pass done in "
              f"{(time.time() - epoch_start) / 60:.1f} min.")

        run_validation = (epoch % eval_every == 0) or (epoch == max_epochs)
        if run_validation:
            vs = time.time()
            val_loss, val_char_acc, val_cer = evaluate(
                model, val_loader, device, ctc_loss_fn)
            lr_before = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            lr_after = optimizer.param_groups[0]["lr"]
            lr_msg = (f" | LR {lr_before:.6f} -> {lr_after:.6f}"
                      if lr_after < lr_before else "")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_path)
                saved_msg = " [BEST SAVED]"
            else:
                patience_counter += 1
                saved_msg = ""

            print(f"[{HF_NAME}] Epoch {epoch:03d}/{max_epochs} | Train {train_loss:.3f} "
                  f"| Val loss {val_loss:.3f} (best {best_val_loss:.3f}) | "
                  f"Val char-acc {val_char_acc:.2%} | Val CER {val_cer:.2%} | "
                  f"val {((time.time() - vs) / 60):.1f} min{lr_msg}{saved_msg}")
        else:
            print(f"[{HF_NAME}] Epoch {epoch:03d}/{max_epochs} | Train {train_loss:.3f} "
                  f"| (validation skipped -- runs every {eval_every} epochs)")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "restart_count": restart_count,
            "charset": CHARSET,
            "input_height": INPUT_HEIGHT,
            "input_width": INPUT_WIDTH,
            "source": HF_DATASET,
        }, ckpt_path)

        if run_validation and patience_counter >= early_stop_patience:
            if restart_count < max_restarts:
                restart_count += 1
                print(f"[{HF_NAME}] No val improvement for {early_stop_patience} "
                      f"check(s) -- WARM RESTART #{restart_count}/{max_restarts}: "
                      f"reload best, reset LR to {config['lr']:.6f}.")
                if best_path.exists():
                    model.load_state_dict(torch.load(
                        best_path, map_location=device, weights_only=False))
                optimizer = optim.RMSprop(model.parameters(), lr=config["lr"],
                                          weight_decay=config["weight_decay"])
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-5)
                patience_counter = 0
                continue
            print(f"[{HF_NAME}] Early stopping for good after {max_restarts} warm "
                  f"restart(s).")
            break

    return best_val_loss


def eval_only_hf(config, device):
    """Load the best HF-trained weights and report CER on validation + test.
    Falls back to _checkpoint.pt if _best.pt isn't there yet."""
    best_path = WEIGHTS_DIR / f"{HF_NAME}_best.pt"
    ckpt_path = WEIGHTS_DIR / f"{HF_NAME}_checkpoint.pt"
    model = PaperCRNN(num_classes=len(CHARSET) + 1).to(device)
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device,
                                         weights_only=False))
        print(f"[Eval] loaded {best_path.name}")
    elif ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        print(f"[Eval] loaded {ckpt_path.name} (epoch {ck['epoch']})")
    else:
        raise SystemExit(f"No trained weights in {WEIGHTS_DIR} -- train first.")

    ds = load_hf(config["cache_dir"])
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    for split in ("validation", "test"):
        loader = DataLoader(
            HFLineDataset(ds[split], is_train=False,
                          max_samples=config.get("max_samples")),
            batch_size=config["batch_size"], shuffle=False,
            collate_fn=collate_fn, num_workers=0)
        loss, char_acc, cer = evaluate(model, loader, device, ctc_loss_fn)
        print(f"[Eval] {split:10s} | loss {loss:.3f} | "
              f"char-acc {char_acc:.2%} | CER {cer:.2%} | "
              f"({len(loader.dataset)} lines)")


def main_hf(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    SCRIPT_DIR.joinpath("NOGIT").mkdir(exist_ok=True)
    sys.stdout = DualLogger(str(SCRIPT_DIR / f"{HF_NAME}.log"))
    sys.stderr = sys.stdout

    config = {
        "cache_dir": args.cache_dir,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": 1e-5,
        "epochs": args.epochs,
        "early_stop_patience": args.early_stop_patience,
        "eval_every": args.eval_every,
        "max_restarts": args.max_restarts,
        "resume": not args.no_resume,
        "init_from": args.init_from,
    }

    if args.eval_only:
        eval_only_hf(config, device)
        return

    print(f"\n--- Training {HF_NAME} on {HF_DATASET} "
          f"(lr={config['lr']}, batch_size={config['batch_size']}, "
          f"max_epochs={config['epochs']}) ---")
    try:
        best = train_hf(config, device)
        print(f"\nDone. Best validation loss: {best:.4f}")
    except Exception:
        import traceback
        print("\n[FATAL] Training crashed:")
        traceback.print_exc()
        raise


# =============================================================================
# Merged in from TrainTextJoint.py -- trains ONE recogniser on BOTH the full
# Teklia/IAM-line dataset AND personal pages (Software/CNN/NOGIT/yeukita +
# dylan) together every epoch via a WeightedRandomSampler, so the model never
# "sequentially forgets" either domain the way naive fine-tuning did. See the
# original TrainTextJoint.py module docstring (preserved in git history) for
# the measured numbers that motivated this approach.
# =============================================================================
NOGIT_DIR = SCRIPT_DIR / "NOGIT"
JOINT_NAME = "paper_cnn_bilstm_ctc_joint"


def collect_personal_lines_joint():
    """Same personal-page collection as the `personal` subcommand: segments
    every yeukita/dylan photo with SegmentPage, pairs TEXT crops with
    their _labels.txt lines (MESS dropped), filters chars/length exactly
    like HFLineDataset does, and splits per-page into train/val."""
    all_rows = []
    for folder in PERSONAL_AUTHORS:
        for img_path in sorted((NOGIT_DIR / folder).glob("*.jpg")):
            results, _preview, _meta = PS.ProcessPage(str(img_path))
            crops = [r["raw_crop"] for r in results if r["tag"] == "TEXT"]
            gt = [g for g in ReadLabelLines(str(img_path)) if g.strip() != "MESS"]
            n = min(len(crops), len(gt))
            for crop, text in zip(crops[:n], gt[:n]):
                kept = "".join(c for c in text.strip() if c in CHAR_TO_IDX)
                if 0 < len(kept) <= MAX_SAFE_LABEL_CHARS:
                    all_rows.append((f"{folder}/{img_path.name}", crop, kept))

    rng = random.Random(SPLIT_SEED)
    train_rows, val_rows = [], []
    by_page = {}
    for page, crop, text in all_rows:
        by_page.setdefault(page, []).append((crop, text))
    for page, rows in by_page.items():
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * VAL_FRACTION))
        val_rows.extend(rows[:n_val])
        train_rows.extend(rows[n_val:])
    print(f"[PersonalData] {len(train_rows)} train / {len(val_rows)} val lines "
          f"from {len(by_page)} pages")
    return train_rows, val_rows


class PersonalLineDataset(Dataset):
    """Shared by the `joint` and `personal` subcommands -- identical in both
    original source files, so defined once here."""

    def __init__(self, rows, is_train):
        self.rows = rows
        self.is_train = is_train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        crop, label = self.rows[i]
        pil_img = Image.fromarray(crop).convert("L")
        pil_img = resize_line_image_fixed(pil_img)
        if self.is_train:
            pil_img = augment_line_image(pil_img)
        return tensor_from_resized(pil_img), encode_text(label), label


def build_train_loader(teklia_train, personal_train, batch_size, personal_fraction):
    combined = torch.utils.data.ConcatDataset([teklia_train, personal_train])
    n_tek, n_per = len(teklia_train), len(personal_train)
    # Per-sample weight so that, in expectation, personal_fraction of every
    # sampled epoch comes from the personal set -- no literal duplication.
    w_tek = (1.0 - personal_fraction) / max(1, n_tek)
    w_per = personal_fraction / max(1, n_per)
    weights = [w_tek] * n_tek + [w_per] * n_per
    sampler = WeightedRandomSampler(weights, num_samples=n_tek, replacement=True)
    return DataLoader(combined, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)


def main_joint(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    ds = load_hf(Path(args.cache_dir))
    teklia_train = HFLineDataset(ds["train"], is_train=True, max_samples=args.max_samples)
    teklia_val = HFLineDataset(ds["validation"], is_train=False, max_samples=args.max_samples)
    personal_train_rows, personal_val_rows = collect_personal_lines_joint()
    personal_train = PersonalLineDataset(personal_train_rows, is_train=True)
    personal_val = PersonalLineDataset(personal_val_rows, is_train=False)

    train_loader = build_train_loader(teklia_train, personal_train, args.batch_size, args.personal_fraction)
    teklia_val_loader = DataLoader(teklia_val, batch_size=32, shuffle=False, collate_fn=collate_fn)
    personal_val_loader = DataLoader(personal_val, batch_size=8, shuffle=False, collate_fn=collate_fn)
    print(f"[Mix] {len(teklia_train)} Teklia + {len(personal_train)} personal train lines, "
          f"sampled at {args.personal_fraction:.0%} personal per epoch "
          f"({len(train_loader)} batches/epoch)")

    model = PaperCRNN(num_classes=len(CHARSET) + 1).to(device)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = WEIGHTS_DIR / f"{JOINT_NAME}_checkpoint.pt"
    best_path = WEIGHTS_DIR / f"{JOINT_NAME}_best.pt"

    start_epoch, best_min_acc = 1, -1.0
    if ckpt_path.exists() and not args.no_resume:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_min_acc = ck["best_min_acc"]
        print(f"[Resume] epoch {ck['epoch']} -> {start_epoch}, best_min_acc={best_min_acc:.4f}")
    else:
        sd = torch.load(args.init_from, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd)
        print(f"[Init] warm-started from {args.init_from}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        t0 = time.time()
        for images, targets, target_lengths, _texts in train_loader:
            images, targets, target_lengths = images.to(device), targets.to(device), target_lengths.to(device)
            optimizer.zero_grad()
            log_probs = model(images)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long, device=device)
            loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(1, n_batches)

        tek_loss, tek_acc, tek_cer = evaluate(model, teklia_val_loader, device, ctc_loss_fn)
        per_loss, per_acc, per_cer = evaluate(model, personal_val_loader, device, ctc_loss_fn)
        scheduler.step(tek_loss + per_loss)

        min_acc = min(tek_acc, per_acc)
        saved = ""
        torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(), epoch=epoch,
                        best_min_acc=max(best_min_acc, min_acc)), ckpt_path)
        if min_acc > best_min_acc:
            best_min_acc = min_acc
            torch.save(model.state_dict(), best_path)
            saved = " [BEST SAVED]"
        print(f"[joint] Epoch {epoch:03d}/{args.epochs} | Train {train_loss:.3f} "
              f"| Teklia CharAcc {tek_acc:.2%} CER {tek_cer:.2%} "
              f"| Personal CharAcc {per_acc:.2%} CER {per_cer:.2%} "
              f"| {time.time()-t0:.1f}s{saved}", flush=True)

    print(f"\nDone. Best checkpoint (by min(Teklia acc, Personal acc) = {best_min_acc:.2%}) "
          f"saved to {best_path}")


# =============================================================================
# Merged in from TrainTextPersonal.py -- fine-tunes PaperCRNN on YOUR OWN
# handwriting (NOGIT/yeukita + NOGIT/dylan), warm-started from an existing
# checkpoint. Kept because BuildStyleProfile10Authors.py depends on the
# checkpoint this produces.
# =============================================================================
PERSONAL_NAME = "paper_cnn_bilstm_ctc_personal"
PERSONAL_DIRS = ["yeukita", "dylan"]
DEFAULT_INIT_FROM = WEIGHTS_DIR / "paper_cnn_bilstm_ctc_hf_best.pt"
PERSONAL_VAL_FRACTION = 0.15
PERSONAL_SPLIT_SEED = 0


def collect_personal_lines_personal():
    """Segments every photo in yeukita/dylan with SegmentPage.ProcessPage
    (the same personal-page pipeline ClassifyText.py uses), pairs each
    TEXT-tagged crop with its ground-truth line (MESS lines dropped, same
    convention as ClassifyText.process_image), and filters characters/
    length exactly like HFLineDataset does. Returns (train_rows, val_rows),
    each a list of (raw_crop_ndarray, cleaned_label) tuples."""
    all_rows = []   # (page_name, raw_crop, cleaned_label)
    for folder in PERSONAL_DIRS:
        for img_path in sorted((NOGIT_DIR / folder).glob("*.jpg")):
            results, _preview, _meta = PS.ProcessPage(str(img_path))
            crops = [r["raw_crop"] for r in results if r["tag"] == "TEXT"]
            gt = ReadLabelLines(str(img_path))
            gt = [g for g in gt if g.strip() != "MESS"]
            if len(crops) != len(gt):
                print(f"[WARN] {folder}/{img_path.name}: {len(crops)} crops vs "
                      f"{len(gt)} label lines -- skipping this page's mismatched tail")
            n = min(len(crops), len(gt))
            for crop, text in zip(crops[:n], gt[:n]):
                kept = "".join(c for c in text.strip() if c in CHAR_TO_IDX)
                if len(kept) == 0 or len(kept) > MAX_SAFE_LABEL_CHARS:
                    continue
                all_rows.append((f"{folder}/{img_path.name}", crop, kept))

    rng = random.Random(PERSONAL_SPLIT_SEED)
    train_rows, val_rows = [], []
    # Per-page shuffle+split so both splits see every page's vocabulary,
    # rather than holding out whole pages (which would starve val/train of
    # whatever technical notation is unique to the held-out pages).
    by_page = {}
    for page, crop, text in all_rows:
        by_page.setdefault(page, []).append((crop, text))
    for page, rows in by_page.items():
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * PERSONAL_VAL_FRACTION))
        val_rows.extend(rows[:n_val])
        train_rows.extend(rows[n_val:])

    print(f"[PersonalData] {len(train_rows)} train / {len(val_rows)} val lines "
          f"from {len(by_page)} pages ({len(all_rows)} total usable lines)")
    return train_rows, val_rows


def train_personal(config, device):
    train_rows, val_rows = collect_personal_lines_personal()
    if len(train_rows) < 4 or len(val_rows) < 1:
        raise RuntimeError("Not enough usable personal lines to train.")

    train_set = PersonalLineDataset(train_rows, is_train=True)
    val_set = PersonalLineDataset(val_rows, is_train=False)
    train_loader = DataLoader(train_set, batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"],
                            shuffle=False, collate_fn=collate_fn)

    model = PaperCRNN(num_classes=len(CHARSET) + 1).to(device)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    if config.get("freeze_conv"):
        # Freeze the CNN feature extractor (generic stroke/edge detectors,
        # shouldn't need to change for a new handwriting style) and only
        # let the LSTM + output layer adapt. Constrains how far the model
        # can drift from the general-purpose weights, trading some
        # personal-domain accuracy for much less forgetting of everything
        # else -- confirmed the naive full-fine-tune lost ~10-11 char-acc
        # points on clean Teklia val/test despite only 16 epochs.
        for module in (model.stage1, model.stage2, model.stage3):
            for p in module.parameters():
                p.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"[Freeze] conv stages frozen -- training {sum(p.numel() for p in trainable)} "
              f"of {sum(p.numel() for p in model.parameters())} params")
    else:
        trainable = model.parameters()

    optimizer = optim.RMSprop(trainable, lr=config["lr"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    name = PERSONAL_NAME + ("_frozen" if config.get("freeze_conv") else "")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = WEIGHTS_DIR / f"{name}_checkpoint.pt"
    best_path = WEIGHTS_DIR / f"{name}_best.pt"

    start_epoch, best_val_loss = 1, float("inf")
    if ckpt_path.exists() and not config.get("no_resume"):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_val_loss = ck["best_val_loss"]
        print(f"[Resume] epoch {ck['epoch']} -> {start_epoch}, best_val_loss={best_val_loss:.4f}")
    else:
        src = Path(config["init_from"])
        if src.exists():
            sd = torch.load(src, map_location=device, weights_only=False)
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            model.load_state_dict(sd)
            print(f"[Init] warm-started from {src}")
        else:
            raise SystemExit(f"[Init] {src} not found -- refusing to train this small "
                              f"a dataset from scratch. Pass --init-from a real checkpoint.")

    for epoch in range(start_epoch, config["epochs"] + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        t0 = time.time()
        for images, targets, target_lengths, _texts in train_loader:
            images, targets, target_lengths = images.to(device), targets.to(device), target_lengths.to(device)
            optimizer.zero_grad()
            log_probs = model(images)
            input_lengths = torch.full((images.size(0),), log_probs.size(0), dtype=torch.long, device=device)
            loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(1, n_batches)

        val_loss, val_char_acc, val_cer = evaluate(model, val_loader, device, ctc_loss_fn)
        scheduler.step(val_loss)

        saved = ""
        torch.save(dict(model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(),
                        epoch=epoch, best_val_loss=min(best_val_loss, val_loss)), ckpt_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            saved = " [BEST SAVED]"
        print(f"[personal] Epoch {epoch:03d}/{config['epochs']} | Train {train_loss:.3f} "
              f"| Val {val_loss:.3f} | CharAcc {val_char_acc:.2%} | CER {val_cer:.2%} "
              f"| {time.time()-t0:.1f}s{saved}", flush=True)


def main_personal(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    train_personal(dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
              init_from=args.init_from, no_resume=args.no_resume,
              freeze_conv=args.freeze_conv), device)


# =============================================================================
# Subcommand dispatcher: `python TrainText.py <base|joint|hf|personal> [opts]`
# Each subcommand's options are exactly the original standalone script's
# argparse options (same flags, defaults, and help text), just nested here.
# Running with no arguments at all preserves the original TrainText.py
# behavior of defaulting straight into the `base` mode.
# =============================================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train the paper CNN-BiLSTM-CTC text recognizer (multiple data sources).")
    sub = parser.add_subparsers(dest="command")

    # --- base: original TrainText.py behavior (own page scans, IAMpages671) ---
    p_base = sub.add_parser(
        "base", help="Train on your own segmented IAMpages671 page scans (original TrainText.py behavior).")
    p_base.add_argument("--max-pages", type=int, default=None,
                         help="Only train on the first N pages (in author-folder-sorted order) instead of "
                              "the whole dataset -- e.g. --max-pages 100 to test on just the pages you've "
                              "regenerated labels for so far. Omit (or pass 0) to use every page available.")
    p_base.add_argument("--force-rebuild", action="store_true",
                         help="Ignore any existing line_image_cache/ and re-segment every page from scratch.")
    p_base.add_argument("--eval-every", type=int, default=1,
                         help="Run validation every N epochs instead of every epoch (default 1). Validation's "
                              "CER computation is pure-Python and can be slow, so bumping this to e.g. 5 skips "
                              "most of that cost. Early stopping then waits for N validation CHECKS (not raw "
                              "epochs) with no improvement -- see the in-code comment in train().")
    p_base.add_argument("--max-restarts", type=int, default=3,
                         help="When early-stop patience runs out, warm-restart from the best checkpoint with "
                              "a fresh LR/optimizer instead of stopping outright, up to this many times "
                              "(default 3). After the last restart also plateaus, training stops for good.")

    # --- hf: original TrainTextHF.py behavior ---
    p_hf = sub.add_parser(
        "hf", help="Train on the HuggingFace Teklia/IAM-line dataset (original TrainTextHF.py behavior).")
    p_hf.add_argument("--eval-only", action="store_true",
                       help="Load the best HF weights and print val + test CER, no training.")
    p_hf.add_argument("--max-samples", type=int, default=None,
                       help="Cap each split to the first N usable lines (quick smoke test).")
    p_hf.add_argument("--epochs", type=int, default=200)
    p_hf.add_argument("--batch-size", type=int, default=16)
    p_hf.add_argument("--lr", type=float, default=1e-3)
    p_hf.add_argument("--eval-every", type=int, default=1)
    p_hf.add_argument("--early-stop-patience", type=int, default=15)
    p_hf.add_argument("--max-restarts", type=int, default=8)
    p_hf.add_argument("--no-resume", action="store_true",
                       help="Ignore any existing checkpoint and start fresh.")
    p_hf.add_argument("--cache-dir", default=str(DEFAULT_HF_CACHE),
                       help="Where HuggingFace caches the downloaded dataset.")
    p_hf.add_argument("--init-from", default=None,
                       help="Warm-start model weights from an existing checkpoint "
                            "of the same architecture (e.g. the page-trained "
                            "paper_cnn_bilstm_ctc_best.pt). Only used when there "
                            "is no HF checkpoint to resume from.")

    # --- joint: original TrainTextJoint.py behavior ---
    p_joint = sub.add_parser(
        "joint", help="Train jointly on Teklia/IAM-line + personal pages (original TrainTextJoint.py behavior).")
    p_joint.add_argument("--init-from", required=True,
                          help="warm-start checkpoint, e.g. weights/paper_cnn_bilstm_ctc_hf_best.pt "
                               "-- REQUIRED, this script refuses to train such a mixed objective from scratch")
    p_joint.add_argument("--epochs", type=int, default=30)
    p_joint.add_argument("--batch-size", type=int, default=16)
    p_joint.add_argument("--lr", type=float, default=5e-4)
    p_joint.add_argument("--personal-fraction", type=float, default=0.20,
                          help="fraction of each training epoch drawn from personal lines (default 20%%)")
    p_joint.add_argument("--max-samples", type=int, default=None,
                          help="cap Teklia train/val size for a quick smoke test")
    p_joint.add_argument("--cache-dir", default=str(DEFAULT_HF_CACHE))
    p_joint.add_argument("--no-resume", action="store_true")

    # --- personal: original TrainTextPersonal.py behavior ---
    p_personal = sub.add_parser(
        "personal", help="Fine-tune on personal pages only (original TrainTextPersonal.py behavior).")
    p_personal.add_argument("--epochs", type=int, default=60)
    p_personal.add_argument("--batch-size", type=int, default=8)
    p_personal.add_argument("--lr", type=float, default=3e-4)
    p_personal.add_argument("--init-from", default=str(DEFAULT_INIT_FROM))
    p_personal.add_argument("--no-resume", action="store_true")
    p_personal.add_argument("--freeze-conv", action="store_true",
                             help="freeze the CNN feature extractor, only train LSTM+output -- "
                                  "saves to a separate _frozen checkpoint, see train_personal()'s comment")

    return parser


def main():
    parser = build_arg_parser()
    argv = sys.argv[1:]
    # No arguments at all: preserve the original TrainText.py behavior of
    # training the base (own page scans) model by default.
    args = parser.parse_args(argv if argv else ["base"])
    command = args.command or "base"

    if command == "base":
        main_base(args)
    elif command == "hf":
        main_hf(args)
    elif command == "joint":
        main_joint(args)
    elif command == "personal":
        main_personal(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
