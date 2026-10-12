"""
TrainAuthor.py

Writer identification. This file now covers THREE related, independently
runnable trainers that all share the same `AuthorClassifierCNN` architecture
and were previously three separate scripts (merged here for consolidation --
no logic, constants, or behavior were changed in the merge):

  1. `original`  -- the ORIGINAL standalone TrainAuthor.py training script.
     Trains on the old 10-author IAM set (150/151/152/153/154/155/384/551/
     552/588). SUPERSEDED: its own checkpoint output is no longer the live
     one (154/155 were later dropped as redundant with 150/151/152's style
     cluster), but it is kept reachable since `AuthorClassifierCNN` is still
     the shared architecture everything below uses.
  2. `10authors` -- the former TrainAuthor10.py. Trains AuthorClassifierCNN
     on the CURRENT 10-author set (8 kept IAM authors + yeukita/dylan).
     THIS PRODUCES THE LIVE CHECKPOINT (author_classifier_10new_weights.pt)
     that server.py loads to classify real photos.
  3. `shape`     -- the former TrainAuthorShape.py. Trains a SEPARATE
     shape-normalized classifier (same AuthorClassifierCNN class, different
     preprocessing -- stroke-normalized lines) producing
     author_shape_10new_weights.pt, which feeds synthesis-quality judging
     at inference time.

WHY A SEPARATE MODEL (not bolted onto the text recognizer as a second head):
the version of this that existed before (train_handwriting_robot_v1_
baseline.py -- deleted in commit 63ba8d9 "FolderCleanupNOGIT", recoverable
from git history) trained a single MultiTaskLineCRNN with both an
author_head and a text CTC head sharing one backbone. On the same 10 IAM
authors the `original` trainer below targets (150/151/152/153/154/155/384/
551/552/588 -- Data/Datasets/IAMpages10, built by Software/DatasetSplitting/
DatasetSplit10Authors.py), see logs/train_10author.log: author-ID accuracy
reached 98.9-100% by epoch 20, while text accuracy stalled around 28-29%
char-accuracy (70% CER) for the whole 200-epoch run. That's not a
coincidence of hyperparameters -- writer-ID over 10 known classes is just a
much easier task than open-vocabulary text recognition, so joint training
keeps spending gradient budget on a problem that's already solved while the
harder problem stays undertrained. Splitting them into two focused models
(this file + train_paper_cnn_bilstm_ctc.py) matches how the writer-ID
literature actually does it too (e.g. arXiv:2009.04877, a dedicated
single-task writer-ID CNN, not a joint model).

WHY THIS SHOULD BE MORE ACCURATE THAN THE OLD RUN even before considering
the split: it reuses TrainText.py's IAMLineDatasetRaw, i.e. the CURRENT line
segmentation + label alignment (regenerate_labels_with_alignment.py's
output), not the old pytesseract-labelled, periodicity-segmented data the
2024 joint run trained on. Better inputs, easier task, dedicated model.

TRANSFER LEARNING OPTION: this file's backbone (stage1/pool1/stage2/pool2/
stage3/height_pool) is architecturally IDENTICAL to PaperCRNN's, on purpose
-- so a trained paper_cnn_bilstm_ctc_best.pt's backbone weights can be
loaded straight in as a starting point (load_backbone_from_text_model
below) before training the author head. That backbone already learned to
notice pen-stroke shape as a side effect of learning to read handwriting
(the same logic as "Encoding CNN Activations for Writer Recognition",
arXiv:1712.07923) so it's a reasonable head start rather than training a
CNN from nothing on a fairly small 10-author dataset.

WHY `shape` EXISTS (former TrainAuthorShape.py): the 10-author model
trained by `10authors` reaches 100% on real held-out pages, but it turns
out to lean almost entirely on ink weight: redraw the same real lines with
every author's ink reduced to a centreline and re-inked at one constant
width, and that model collapses to 11-16% (chance is 10%). It is reading
the pen, not the hand. That matters because the gantry writes every author
with the SAME pen at a constant stroke width. Ink density is not a style
channel the machine can reproduce, so a score that depends on it does not
predict what the physical robot will achieve -- and it is not what a person
judges by eye either. So the `shape` model is trained on stroke-normalized
lines: binarize, skeletonize to the centreline, and re-ink at a fixed
width, identically for every author. It can only succeed by learning
slant, proportions, spacing, letterforms and connection habits -- the
things the robot actually reproduces. It is the honest yardstick for
physical style accuracy. Same weight format as the other author models, so
EvaluateStyle.py and VerifyRewrite.py can load it interchangeably. Covers
the SAME 10-author set as `10authors`: the 8 kept dataset authors
(150,151,152,153,384,551,552,588) plus the 2 personal authors (yeukita,
dylan) -- personal samples are added via this file's own
add_personal_samples(), so the same PNG-dict format, and the same
train/holdout split (seed 0, 15%), is shared everywhere in this project.

Does NOT modify TrainText.py -- only imports from it.

Run:
    python TrainAuthor.py original     # old 10-author set incl. 154/155
    python TrainAuthor.py 10authors    # current 10-author set (live checkpoint)
    python TrainAuthor.py shape        # shape-normalized classifier
    python TrainAuthor.py              # defaults to `original` (unchanged
                                        # behavior from the pre-merge file)
"""

import argparse
import io
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "AuthorReproductionStuff"))

from TrainText import (
    IAMLineDatasetRaw,
    PaperCRNN,
    _decode_png,
    augment_line_image,
    conv_block,
    encode_text,
    resize_line_image_fixed,
    tensor_from_resized,
)
import SegmentLean as SL
from ExtractIAMLines import ReadLabelLines
from authors_config import (
    DATASET_AUTHORS, PERSONAL_AUTHORS, VAL_FRACTION, SPLIT_SEED,
    AUTHOR_CLASSIFIER_RUN_NAME, personal_author_pages,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[1] / "Data" / "Datasets" / "IAMpages10"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "NOGIT" / "line_cache_authors10"
DEFAULT_TEXT_WEIGHTS = SCRIPT_DIR / "weights" / "paper_cnn_bilstm_ctc_best.pt"
WEIGHTS_DIR = SCRIPT_DIR / "weights"
NOGIT_DIR = SCRIPT_DIR / "NOGIT"


# -----------------------------------------------------------------------------
# Dataset: wraps IAMLineDatasetRaw (imported, untouched) to additionally
# return an author index per sample -- the base class already tracks
# author_folders/author_to_idx and a page_key of "author_id/filename.png"
# per sample, it just never surfaced the index through __getitem__ because
# TrainText.py never needed it.
# -----------------------------------------------------------------------------
class AuthorLabeledView(Dataset):
    def __init__(self, base: IAMLineDatasetRaw, indices, is_train):
        self.base = base
        self.indices = indices
        self.is_train = is_train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        item = self.base.samples[self.indices[i]]
        pil_img = resize_line_image_fixed(_decode_png(item["image_png"]))
        if self.is_train:
            pil_img = augment_line_image(pil_img)
        img_tensor = tensor_from_resized(pil_img)
        authorId = item["page_key"].split("/")[0]
        authorIdx = self.base.author_to_idx[authorId]
        return img_tensor, authorIdx, item["page_key"]


def collate_fn(batch):
    images, authorIdxs, pageKeys = zip(*batch)
    return torch.stack(images, 0), torch.tensor(authorIdxs, dtype=torch.long), list(pageKeys)


# -----------------------------------------------------------------------------
# Model: same backbone shape as PaperCRNN (see module docstring for why),
# global-average-pooled over width, small MLP head -> author logits.
# -----------------------------------------------------------------------------
class AuthorClassifierCNN(nn.Module):
    def __init__(self, num_authors):
        super().__init__()
        self.stage1 = conv_block(1, 32, n_layers=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.stage2 = conv_block(32, 64, n_layers=4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.stage3 = conv_block(64, 128, n_layers=6)
        self.height_pool = nn.AdaptiveMaxPool2d((1, None))
        self.width_pool = nn.AdaptiveAvgPool1d(1)
        self.author_head = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(96, num_authors),
        )

    def forward(self, x):
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.stage3(x)
        x = self.height_pool(x)        # (B, 128, 1, W)
        x = x.squeeze(2)                # (B, 128, W)
        pooled = self.width_pool(x).squeeze(2)  # (B, 128)
        return self.author_head(pooled)

    def load_backbone_from_text_model(self, textModelStateDict):
        """Copies over every backbone weight whose name+shape matches a
        PaperCRNN checkpoint -- silently skips author_head (PaperCRNN has no
        such thing) and the LSTM/text_head (this model has neither)."""
        own = self.state_dict()
        matched = {k: v for k, v in textModelStateDict.items() if k in own and own[k].shape == v.shape}
        own.update(matched)
        self.load_state_dict(own)
        return len(matched)

    def freeze_backbone(self):
        for module in (self.stage1, self.pool1, self.stage2, self.pool2, self.stage3):
            for p in module.parameters():
                p.requires_grad = False


def evaluate(model, dataloader, device, authorFolders):
    model.eval()
    correct, total = 0, 0
    perAuthorCorrect = {a: 0 for a in authorFolders}
    perAuthorTotal = {a: 0 for a in authorFolders}

    with torch.no_grad():
        for images, authorIdxs, pageKeys in dataloader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu()
            for pred, truth, pageKey in zip(preds, authorIdxs, pageKeys):
                authorId = pageKey.split("/")[0]
                total += 1
                perAuthorTotal[authorId] += 1
                if int(pred) == int(truth):
                    correct += 1
                    perAuthorCorrect[authorId] += 1

    acc = correct / max(1, total)
    perAuthorAcc = {a: perAuthorCorrect[a] / perAuthorTotal[a] for a in authorFolders if perAuthorTotal[a] > 0}
    return acc, perAuthorAcc


# =============================================================================
# `original` subcommand -- the ORIGINAL standalone TrainAuthor.py behavior
# (old 10-author set including 154/155, interactive input() prompts).
# =============================================================================
def train(dataDir, cacheDir, numEpochs, batchSize, learningRate, initFromTextWeights, freezeBackbone, runName):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Training on: {device}")

    base = IAMLineDatasetRaw(root_dir=dataDir, cache_dir=str(cacheDir))
    if len(base) < 4:
        raise RuntimeError("Not enough line samples to train -- check dataDir points at IAMpages10.")

    trainIdx, valIdx = base.holdout_split_indices()
    if not valIdx:
        raise RuntimeError("No holdout lines found -- every author needs >= min_pages_for_holdout labeled pages.")

    trainSet = AuthorLabeledView(base, trainIdx, is_train=True)
    valSet = AuthorLabeledView(base, valIdx, is_train=False)
    trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True, collate_fn=collate_fn)
    valLoader = DataLoader(valSet, batch_size=batchSize, shuffle=False, collate_fn=collate_fn)

    print(f"[Split] Train lines: {len(trainIdx)} | Held-out val lines: {len(valIdx)} | "
          f"Authors: {len(base.author_folders)} ({', '.join(base.author_folders)})")

    model = AuthorClassifierCNN(num_authors=len(base.author_folders)).to(device)

    if initFromTextWeights is not None:
        stateDict = torch.load(initFromTextWeights, map_location=device, weights_only=False)
        if "model_state_dict" in stateDict:
            stateDict = stateDict["model_state_dict"]
        matchedCount = model.load_backbone_from_text_model(stateDict)
        print(f"[Init] Loaded {matchedCount} matching backbone tensor(s) from {initFromTextWeights}")

    if freezeBackbone:
        model.freeze_backbone()
        print("[Init] Backbone frozen -- only author_head will train (fast, low overfitting risk on a small dataset).")

    lossFn = nn.CrossEntropyLoss()
    trainableParams = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainableParams, lr=learningRate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numEpochs)

    weightsPath = WEIGHTS_DIR / f"{runName}_weights.pt"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    bestAcc = 0.0
    bestState = None

    print("\nStarting author classifier training...")
    print("=" * 88)
    for epoch in range(1, numEpochs + 1):
        model.train()
        totalLoss = 0.0
        for images, authorIdxs, _ in trainLoader:
            images, authorIdxs = images.to(device), authorIdxs.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = lossFn(logits, authorIdxs)
            loss.backward()
            nn.utils.clip_grad_norm_(trainableParams, max_norm=5.0)
            optimizer.step()
            totalLoss += float(loss.item())
        scheduler.step()

        valAcc, perAuthorAcc = evaluate(model, valLoader, device, base.author_folders)
        if valAcc > bestAcc:
            bestAcc = valAcc
            bestState = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        avgLoss = totalLoss / max(1, len(trainLoader))
        print(f"Epoch {epoch:03d}/{numEpochs} | loss {avgLoss:.3f} | val author-acc {valAcc:.3f} "
              f"(best {bestAcc:.3f})")

    if bestState is not None:
        model.load_state_dict(bestState)

    finalAcc, perAuthorAcc = evaluate(model, valLoader, device, base.author_folders)
    print("=" * 88)
    print(f"Best val author-accuracy: {bestAcc:.3f}  (proposal Requirement 2 target: 0.95)")
    print("Per-author accuracy on held-out pages:")
    for authorId, acc in sorted(perAuthorAcc.items()):
        print(f"  {authorId}: {acc:.3f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "author_mapping": base.author_to_idx,
        "best_val_author_acc": bestAcc,
    }, weightsPath)
    print(f"\nWeights saved to: {weightsPath}")


def main_original():
    print("=== TrainAuthor.py ===")
    dataDir = input(f"IAM 10-author dataset path (blank = {DEFAULT_DATA_DIR}): ").strip() or str(DEFAULT_DATA_DIR)
    cacheDir = input(f"Line-image cache dir (blank = {DEFAULT_CACHE_DIR}): ").strip() or str(DEFAULT_CACHE_DIR)

    initChoice = input(f"Initialize backbone from the trained text model's weights? "
                        f"[Y/n, blank = yes, from {DEFAULT_TEXT_WEIGHTS.name}]: ").strip().lower()
    initFromTextWeights = None
    if initChoice not in ("n", "no"):
        weightsInput = input(f"Text model weights path (blank = {DEFAULT_TEXT_WEIGHTS}): ").strip() or str(DEFAULT_TEXT_WEIGHTS)
        if os.path.exists(weightsInput):
            initFromTextWeights = weightsInput
        else:
            print(f"[Warning] {weightsInput} not found -- training backbone from scratch instead.")

    freezeChoice = input("Freeze the backbone and only train the author head? "
                          "[Y/n, blank = yes -- recommended with only 10 authors' worth of data]: ").strip().lower()
    freezeBackbone = freezeChoice not in ("n", "no")

    epochsInput = input("Epochs (blank = 40): ").strip()
    numEpochs = int(epochsInput) if epochsInput else 40

    runName = input("Run name for saved weights (blank = 'author_classifier_10'): ").strip() or "author_classifier_10"

    train(
        dataDir=dataDir,
        cacheDir=cacheDir,
        numEpochs=numEpochs,
        batchSize=16,
        learningRate=0.001,
        initFromTextWeights=initFromTextWeights,
        freezeBackbone=freezeBackbone,
        runName=runName,
    )


# =============================================================================
# `10authors` subcommand -- former TrainAuthor10.py. Trains
# AuthorClassifierCNN (unchanged) on the NEW 10-author set: 8 kept dataset
# authors (150,151,152,153,384,551,552,588 -- dropped 154/155 for being
# redundant with 150/151/152's style cluster) + 2 personal authors
# (yeukita, dylan). THIS PRODUCES THE LIVE CHECKPOINT server.py loads.
#
# HOW THE TWO DATA SOURCES ARE UNIFIED
#     IAMLineDatasetRaw (TrainText.py) stores each line sample as a dict with
#     PNG-encoded image bytes + a page_key "authorId/filename.png" + an
#     is_holdout flag. That's a plain, source-agnostic format -- so instead
#     of writing a parallel Dataset class for the personal authors, this
#     just PNG-encodes their SegmentPage-derived line crops into the exact
#     same dict shape and appends them into base.samples directly. Every
#     downstream piece (author_folders, author_to_idx, holdout_split_indices,
#     AuthorLabeledView, evaluate(), train()) then works unmodified across
#     both data sources.
#
#     The personal train/holdout split uses the SAME seed and fraction as
#     BuildStyleProfile10Authors.py and TrainTextPersonal.py, so it's the
#     same lines held out everywhere in this project, not three different
#     "holdout" definitions for the same person's handwriting.
# =============================================================================

# TrainAuthor.py's own default (DEFAULT_TEXT_WEIGHTS above) points at the
# OLDER, weaker (non-HF) text checkpoint. Prefer the joint Teklia+personal
# one for the backbone warm-start -- it reads handwriting better in
# general, which the writer-ID backbone inherits for free (same reasoning
# as BuildStyleProfile.py's own "prefer the better checkpoint" pattern).
for _name in ("paper_cnn_bilstm_ctc_joint_best.pt", "paper_cnn_bilstm_ctc_hf_best.pt"):
    _candidate = NOGIT_DIR.parent / "weights" / _name
    if _candidate.exists():
        DEFAULT_TEXT_WEIGHTS = _candidate
        break

RUN_NAME = AUTHOR_CLASSIFIER_RUN_NAME


def _encode_png(gray_crop):
    buf = io.BytesIO()
    Image.fromarray(gray_crop).convert("L").save(buf, format="PNG")
    return buf.getvalue()


def add_personal_samples(base):
    """Segments every personal author's photos, splits per-page into
    train/holdout (same convention as elsewhere in this project), and
    appends PNG-encoded samples directly into base.samples so they're
    indistinguishable from IAM-derived samples to everything downstream."""
    rng = random.Random(SPLIT_SEED)
    n_added = 0
    for author in PERSONAL_AUTHORS:
        for img_path, label_path in personal_author_pages(author):
            lean_lines, _ov, _info = SL.SegmentLines(str(img_path))
            crops = [np.asarray(ln["image"]) for ln in lean_lines]
            gt = ReadLabelLines(str(img_path), str(label_path))
            gt = [g for g in gt if g.strip() != "MESS"]
            n = min(len(crops), len(gt))
            rows = list(zip(range(n), crops[:n], gt[:n]))
            rng.shuffle(rows)
            n_val = max(1, round(len(rows) * VAL_FRACTION))
            for rank, (orig_idx, crop, text) in enumerate(rows):
                target = encode_text(text)
                if len(target) == 0:
                    continue
                is_holdout = rank < n_val
                page_key = f"{author}/{img_path.stem}_{orig_idx:02d}.png"
                base.samples.append({
                    "page_key": page_key,
                    "line_idx": orig_idx,
                    "image_png": _encode_png(crop),
                    "target": target,
                    "text": text,
                    "is_holdout": is_holdout,
                })
                n_added += 1
    print(f"[Personal] added {n_added} line samples for {PERSONAL_AUTHORS}")


def main_10authors(epochs=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    base = IAMLineDatasetRaw(root_dir=str(DEFAULT_DATA_DIR), cache_dir=str(DEFAULT_CACHE_DIR))
    before = len(base.samples)
    base.samples = [s for s in base.samples if s["page_key"].split("/")[0] in DATASET_AUTHORS]
    print(f"[Filter] kept {len(base.samples)}/{before} IAM samples for {DATASET_AUTHORS}")

    add_personal_samples(base)

    base.author_folders = sorted(DATASET_AUTHORS + PERSONAL_AUTHORS)
    base.author_to_idx = {a: i for i, a in enumerate(base.author_folders)}
    print(f"[Authors] {len(base.author_folders)}: {base.author_folders}")

    trainIdx, valIdx = base.holdout_split_indices()
    if not valIdx:
        raise RuntimeError("No holdout lines found.")

    trainSet = AuthorLabeledView(base, trainIdx, is_train=True)
    valSet = AuthorLabeledView(base, valIdx, is_train=False)
    trainLoader = DataLoader(trainSet, batch_size=16, shuffle=True, collate_fn=collate_fn)
    valLoader = DataLoader(valSet, batch_size=16, shuffle=False, collate_fn=collate_fn)
    print(f"[Split] Train lines: {len(trainIdx)} | Held-out val lines: {len(valIdx)}")

    model = AuthorClassifierCNN(num_authors=len(base.author_folders)).to(device)
    if DEFAULT_TEXT_WEIGHTS.exists():
        sd = torch.load(DEFAULT_TEXT_WEIGHTS, map_location=device, weights_only=False)
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        matched = model.load_backbone_from_text_model(sd)
        print(f"[Init] loaded {matched} matching backbone tensor(s) from {DEFAULT_TEXT_WEIGHTS.name}")

    lossFn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    num_epochs = epochs if epochs is not None else 20
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    weights_path = WEIGHTS_DIR / f"{RUN_NAME}_weights.pt"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    print("\n" + "=" * 88)
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for images, authorIdxs, _ in trainLoader:
            images, authorIdxs = images.to(device), authorIdxs.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = lossFn(logits, authorIdxs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item())
        scheduler.step()

        val_acc, per_author_acc = evaluate(model, valLoader, device, base.author_folders)
        saved = ""
        # Save to disk THE MOMENT a new best is found, not just once at the
        # very end -- this run oscillates a lot epoch to epoch (a small
        # dataset + a fairly high LR), so the best epoch is rarely the
        # last one, and stopping early (or a crash) must not lose it.
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(),
                       "author_mapping": base.author_to_idx,
                       "best_val_author_acc": best_acc}, weights_path)
            saved = " [SAVED]"

        print(f"Epoch {epoch:03d}/{num_epochs} | loss {total_loss/max(1,len(trainLoader)):.3f} "
              f"| val author-acc {val_acc:.3f} (best {best_acc:.3f}){saved}", flush=True)

    print("=" * 88)
    print(f"Best val author-accuracy: {best_acc:.3f}")
    if weights_path.exists():
        ck = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        _, per_author_acc = evaluate(model, valLoader, device, base.author_folders)
        print("Per-author accuracy (best saved checkpoint):")
        for a, acc in sorted(per_author_acc.items()):
            print(f"  {a}: {acc:.3f}")
    print(f"Weights saved to: {weights_path}")


# =============================================================================
# `shape` subcommand -- former TrainAuthorShape.py: writer identification
# from SHAPE ALONE (see module docstring's "WHY `shape` EXISTS" section).
# =============================================================================
# Former TrainAuthorShape.py lived in Software/CNN/AuthorReproductionStuff/,
# so its SCRIPT_DIR.parents[2]/SCRIPT_DIR.parent paths resolved to the same
# repo-root/Data and Software/CNN/NOGIT locations that DEFAULT_DATA_DIR,
# DEFAULT_CACHE_DIR and DEFAULT_TEXT_WEIGHTS above already point at (this
# file's own SCRIPT_DIR is Software/CNN) -- reused directly rather than
# duplicated under new names.
SHAPE_DATA_DIR = DEFAULT_DATA_DIR
SHAPE_CACHE_DIR = DEFAULT_CACHE_DIR
SHAPE_TEXT_WEIGHTS = SCRIPT_DIR / "weights" / "paper_cnn_bilstm_ctc_best.pt"
# Prefer joint (Teklia+personal, no forgetting) > HF-only > original --
# see BuildStyleProfile.py's matching comment for the measured numbers.
for _name in ("paper_cnn_bilstm_ctc_joint_best.pt", "paper_cnn_bilstm_ctc_hf_best.pt"):
    _candidate = SCRIPT_DIR / "weights" / _name
    if _candidate.exists():
        SHAPE_TEXT_WEIGHTS = _candidate
        break
# Renamed (not reusing the old shape_norm_cache.pkl / author_shape_10_weights.pt)
# because the author set changed -- 154/155 dropped, yeukita/dylan added -- and
# BuildNormCache() below trusts an existing cache file blindly, so a stale one
# would silently keep training on the wrong 10 authors.
SHAPE_NORM_CACHE = SCRIPT_DIR / "NOGIT" / "shape_norm_cache_10new.pkl"
SHAPE_OUT_WEIGHTS = SCRIPT_DIR / "weights" / "author_shape_10new_weights.pt"

# One pen for everybody. Expressed relative to the line's own x-height so
# the normalization is resolution-independent.
PEN_WIDTH_XH = 0.14


def StrokeNormalize(gray, penWidthXh=PEN_WIDTH_XH):
    """Real ink -> centreline -> re-inked at one constant width.

    Removes the author's pen/pressure entirely while keeping every
    geometric property of the writing, which is exactly the transformation
    the gantry performs when it redraws a hand with its own pen."""
    import BuildStyleProfile as SP
    import SegmentPage as F

    ink = SP.BinarizeLine(gray)
    if not ink.any():
        return None
    band = SP.CoreBand(ink)
    xh = float(band[1] - band[0]) if band else 0.0
    if xh < 4:
        xh = max(8.0, ink.shape[0] / 3.0)
    w = int(np.clip(round(penWidthXh * xh), 2, 9))
    sk = SP.Skeletonize(ink)
    if not sk.any():
        return None
    k = 2 * (w // 2) + 1
    out = F.Dilate(sk, k, k)
    return Image.fromarray(((~out) * 255).astype(np.uint8))


def BuildNormCache(force=False):
    if SHAPE_NORM_CACHE.exists() and not force:
        with open(SHAPE_NORM_CACHE, "rb") as f:
            return pickle.load(f)
    base = IAMLineDatasetRaw(root_dir=str(SHAPE_DATA_DIR), cache_dir=str(SHAPE_CACHE_DIR))
    before = len(base.samples)
    base.samples = [s for s in base.samples if s["page_key"].split("/")[0] in DATASET_AUTHORS]
    print("[Filter] kept %d/%d IAM samples for %s" % (len(base.samples), before, DATASET_AUTHORS))
    add_personal_samples(base)
    rows = []
    for i, s in enumerate(base.samples):
        g = np.array(_decode_png(s["image_png"]).convert("L"))
        im = StrokeNormalize(g)
        if im is None:
            continue
        small = resize_line_image_fixed(im)
        rows.append(dict(author=s["page_key"].split("/")[0],
                         holdout=s["is_holdout"],
                         png=np.array(small, dtype=np.uint8)))
        if (i + 1) % 100 == 0:
            print("  normalized %d/%d lines" % (i + 1, len(base.samples)))
    SHAPE_NORM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(SHAPE_NORM_CACHE, "wb") as f:
        pickle.dump(rows, f)
    return rows


def Features(model, arr, device, aug=False):
    im = Image.fromarray(arr)
    if aug:
        im = augment_line_image(im)
    t = tensor_from_resized(im).unsqueeze(0).to(device)
    with torch.no_grad():
        x = model.pool1(model.stage1(t))
        x = model.pool2(model.stage2(x))
        x = model.stage3(x)
        x = model.height_pool(x).squeeze(2)
        return model.width_pool(x).squeeze(2)[0].cpu().numpy()


def Train(nAug=4, epochs=500, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = BuildNormCache()
    authors = sorted({r["author"] for r in rows})
    a2i = {a: i for i, a in enumerate(authors)}
    print("[Data] %d stroke-normalized lines, %d authors" % (len(rows), len(authors)))

    model = AuthorClassifierCNN(num_authors=len(authors)).to(device)
    sd = torch.load(SHAPE_TEXT_WEIGHTS, map_location=device, weights_only=False)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    print("[Init] loaded %d backbone tensors" % model.load_backbone_from_text_model(sd))
    model.eval()

    tr = [r for r in rows if not r["holdout"]]
    va = [r for r in rows if r["holdout"]]
    print("[Split] train %d / val %d" % (len(tr), len(va)))

    def build(rs, aug):
        X, Y = [], []
        for r in rs:
            for k in range(aug if aug else 1):
                X.append(Features(model, r["png"], device, aug=bool(k)))
                Y.append(a2i[r["author"]])
        return (torch.tensor(np.stack(X), dtype=torch.float32),
                torch.tensor(Y, dtype=torch.long))

    print("[Features] extracting...")
    Xtr, Ytr = build(tr, nAug)
    Xva, Yva = build(va, 0)
    print("[Features] train %s val %s" % (tuple(Xtr.shape), tuple(Xva.shape)))

    head = model.author_head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossFn = nn.CrossEntropyLoss()
    best, bestState = 0.0, None
    for ep in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            lossFn(head(Xtr[b].to(device)), Ytr[b].to(device)).backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            acc = (head(Xva.to(device)).argmax(1).cpu() == Yva).float().mean().item()
        if acc > best:
            best = acc
            bestState = {k: v.detach().cpu().clone()
                         for k, v in model.state_dict().items()}
        if ep % 100 == 0 or ep == 1:
            print("  epoch %4d  val shape-only author-acc %.3f (best %.3f)" % (ep, acc, best))

    if bestState:
        model.load_state_dict(bestState)
    with torch.no_grad():
        pred = model.author_head(Xva.to(device)).argmax(1).cpu().numpy()
    print("\nBest val author-accuracy (SHAPE ONLY): %.3f" % best)
    per = {}
    for p, y in zip(pred, Yva.numpy()):
        d = per.setdefault(authors[y], [0, 0])
        d[1] += 1
        d[0] += int(p == y)
    for a in sorted(per):
        c, t = per[a]
        print("  %s: %.3f  (%d/%d)" % (a, c / max(1, t), c, t))

    SHAPE_OUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "author_mapping": a2i,
                "best_val_author_acc": best}, SHAPE_OUT_WEIGHTS)
    print("\nWeights saved to: %s" % SHAPE_OUT_WEIGHTS)
    return best


def main_shape():
    Train()


# =============================================================================
# Subcommand dispatcher: `python TrainAuthor.py <original|10authors|shape>`
# Running with no arguments at all preserves the original standalone
# TrainAuthor.py behavior of defaulting straight into `original` (the
# interactive input()-prompt trainer).
# =============================================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train writer-ID classifiers (AuthorClassifierCNN, multiple data sources/preprocessing).")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("original", help="Original TrainAuthor.py behavior -- old 10-author set incl. 154/155, "
                                     "interactive prompts.")
    p_10 = sub.add_parser("10authors", help="Former TrainAuthor10.py behavior -- current 10-author set "
                                      "(8 kept IAM + yeukita/dylan); produces the live server.py checkpoint.")
    p_10.add_argument("epochs", type=int, nargs="?", default=None,
                       help="Number of epochs (blank = 20, matching the original "
                            "`python TrainAuthor10.py <epochs>` positional argument).")
    sub.add_parser("shape", help="Former TrainAuthorShape.py behavior -- shape-normalized (ink-removed) "
                                  "classifier used for synthesis-quality judging.")
    return parser


def main():
    parser = build_arg_parser()
    argv = sys.argv[1:]
    # No arguments at all: preserve the original TrainAuthor.py behavior of
    # running the `original` interactive trainer by default.
    args = parser.parse_args(argv if argv else ["original"])
    command = args.command or "original"

    if command == "original":
        main_original()
    elif command == "10authors":
        main_10authors(epochs=args.epochs)
    elif command == "shape":
        main_shape()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
