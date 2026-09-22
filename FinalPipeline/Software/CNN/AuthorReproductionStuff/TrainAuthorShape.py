"""
TrainAuthorShape.py -- writer identification from SHAPE ALONE.

WHY THIS EXISTS. The 10-author model in TrainAuthor.py reaches
100% on real held-out pages, but it turns out to lean almost entirely on
ink weight: redraw the same real lines with every author's ink reduced to a
centreline and re-inked at one constant width, and that model collapses to
11-16% (chance is 10%). It is reading the pen, not the hand.

That matters because the gantry writes every author with the SAME pen at a
constant stroke width. Ink density is not a style channel the machine can
reproduce, so a score that depends on it does not predict what the physical
robot will achieve -- and it is not what a person judges by eye either.

So this model is trained on stroke-normalized lines: binarize, skeletonize
to the centreline, and re-ink at a fixed width, identically for every
author. It can only succeed by learning slant, proportions, spacing,
letterforms and connection habits -- the things the robot actually
reproduces. It is the honest yardstick for physical style accuracy.

Same weight format as the other two author models, so evaluate_style.py and
VerifyRewrite.py can load it interchangeably.

Covers the SAME 10-author set as TrainAuthor10.py: the 8 kept dataset
authors (150,151,152,153,384,551,552,588) plus the 2 personal authors
(yeukita, dylan) -- personal samples are added via TrainAuthor10's
add_personal_samples(), so the same PNG-dict format, and the same
train/holdout split (seed 0, 15%), is shared everywhere in this project.

Does NOT modify TrainText.py or TrainAuthor.py.

Run:
    python TrainAuthorShape.py
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import RawImageOps as F
import BuildStyleProfile as SP
from TrainAuthor import AuthorClassifierCNN
from TrainAuthor10 import add_personal_samples, DATASET_AUTHORS
from TrainText import (
    IAMLineDatasetRaw, _decode_png, augment_line_image,
    resize_line_image_fixed, tensor_from_resized,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[2] / "Data" / "Datasets" / "IAMpages10"
CACHE_DIR = SCRIPT_DIR.parent / "NOGIT" / "line_cache_authors10"
TEXT_WEIGHTS = SCRIPT_DIR.parent / "NOGIT" / "weights" / "paper_cnn_bilstm_ctc_best.pt"
# Prefer joint (Teklia+personal, no forgetting) > HF-only > original --
# see BuildStyleProfile.py's matching comment for the measured numbers.
for _name in ("paper_cnn_bilstm_ctc_joint_best.pt", "paper_cnn_bilstm_ctc_hf_best.pt"):
    _candidate = SCRIPT_DIR.parent / "NOGIT" / "weights" / _name
    if _candidate.exists():
        TEXT_WEIGHTS = _candidate
        break
# Renamed (not reusing the old shape_norm_cache.pkl / author_shape_10_weights.pt)
# because the author set changed -- 154/155 dropped, yeukita/dylan added -- and
# BuildNormCache() below trusts an existing cache file blindly, so a stale one
# would silently keep training on the wrong 10 authors.
NORM_CACHE = SCRIPT_DIR.parent / "NOGIT" / "shape_norm_cache_10new.pkl"
OUT_WEIGHTS = SCRIPT_DIR.parent / "NOGIT" / "weights" / "author_shape_10new_weights.pt"

# One pen for everybody. Expressed relative to the line's own x-height so
# the normalization is resolution-independent.
PEN_WIDTH_XH = 0.14


def StrokeNormalize(gray, penWidthXh=PEN_WIDTH_XH):
    """Real ink -> centreline -> re-inked at one constant width.

    Removes the author's pen/pressure entirely while keeping every
    geometric property of the writing, which is exactly the transformation
    the gantry performs when it redraws a hand with its own pen."""
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
    if NORM_CACHE.exists() and not force:
        with open(NORM_CACHE, "rb") as f:
            return pickle.load(f)
    base = IAMLineDatasetRaw(root_dir=str(DATA_DIR), cache_dir=str(CACHE_DIR))
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
    NORM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(NORM_CACHE, "wb") as f:
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
    sd = torch.load(TEXT_WEIGHTS, map_location=device, weights_only=False)
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

    OUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "author_mapping": a2i,
                "best_val_author_acc": best}, OUT_WEIGHTS)
    print("\nWeights saved to: %s" % OUT_WEIGHTS)
    return best


if __name__ == "__main__":
    Train()
