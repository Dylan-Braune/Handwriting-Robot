"""
train_author_fast.py -- a quick writer-ID model over the same 10 authors,
used as a SECOND opinion / fallback yardstick for evaluate_style.py.

It is the frozen-backbone variant that train_author_classifier.py already
offers, but made fast by precomputing the backbone features ONCE per line
(the backbone never changes when frozen, so re-running it every epoch is
pure waste) and then fitting only the small head. Trains in seconds on CPU.

It writes weights in exactly the same format as train_author_classifier.py
(`model_state_dict` + `author_mapping`) so evaluate_style.py can load either
file interchangeably.

Does NOT modify train_paper_cnn_bilstm_ctc.py or train_author_classifier.py.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_author_classifier import AuthorClassifierCNN
from train_paper_cnn_bilstm_ctc import (
    IAMLineDatasetRaw, _decode_png, augment_line_image,
    resize_line_image_fixed, tensor_from_resized,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "Data" / "Datasets" / "IAMpages10"
CACHE_DIR = SCRIPT_DIR / "NOGIT" / "line_cache_authors10"
TEXT_WEIGHTS = SCRIPT_DIR / "NOGIT" / "weights" / "paper_cnn_bilstm_ctc_best.pt"
OUT_WEIGHTS = SCRIPT_DIR / "NOGIT" / "weights" / "author_fast_10_weights.pt"


def BackboneFeatures(model, pil, device):
    t = tensor_from_resized(resize_line_image_fixed(pil)).unsqueeze(0).to(device)
    with torch.no_grad():
        x = model.pool1(model.stage1(t))
        x = model.pool2(model.stage2(x))
        x = model.stage3(x)
        x = model.height_pool(x).squeeze(2)
        return model.width_pool(x).squeeze(2)[0].cpu().numpy()


def Train(nAug=3, epochs=400, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = IAMLineDatasetRaw(root_dir=str(DATA_DIR), cache_dir=str(CACHE_DIR))
    model = AuthorClassifierCNN(num_authors=len(base.author_folders)).to(device)
    sd = torch.load(TEXT_WEIGHTS, map_location=device, weights_only=False)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    n = model.load_backbone_from_text_model(sd)
    print(f"[Init] loaded {n} backbone tensors from the text model")
    model.eval()

    trainIdx, valIdx = base.holdout_split_indices()
    print(f"[Split] train {len(trainIdx)} / val {len(valIdx)} lines, "
          f"{len(base.author_folders)} authors")

    def build(indices, aug):
        X, Y, K = [], [], []
        for i in indices:
            item = base.samples[i]
            pil = _decode_png(item["image_png"])
            reps = aug if aug else 1
            for r in range(reps):
                p = augment_line_image(resize_line_image_fixed(pil)) if r else pil
                X.append(BackboneFeatures(model, p, device))
                Y.append(base.author_to_idx[item["page_key"].split("/")[0]])
                K.append(item["page_key"])
        return (torch.tensor(np.stack(X), dtype=torch.float32),
                torch.tensor(Y, dtype=torch.long), K)

    print("[Features] extracting (backbone is frozen, so this happens once)...")
    Xtr, Ytr, _ = build(trainIdx, nAug)
    Xva, Yva, Kva = build(valIdx, 0)
    print(f"[Features] train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}")

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
            loss = lossFn(head(Xtr[b].to(device)), Ytr[b].to(device))
            loss.backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            acc = (head(Xva.to(device)).argmax(1).cpu() == Yva).float().mean().item()
        if acc > best:
            best = acc
            bestState = {k: v.detach().cpu().clone()
                         for k, v in model.state_dict().items()}
        if ep % 50 == 0 or ep == 1:
            print(f"  epoch {ep:4d}  val author-acc {acc:.3f}  (best {best:.3f})")

    if bestState:
        model.load_state_dict(bestState)
    head.eval()
    with torch.no_grad():
        pred = model.author_head(Xva.to(device)).argmax(1).cpu().numpy()
    perAuthor = {}
    for p, y, k in zip(pred, Yva.numpy(), Kva):
        a = k.split("/")[0]
        d = perAuthor.setdefault(a, [0, 0])
        d[1] += 1
        d[0] += int(p == y)
    print(f"\nBest val author-accuracy: {best:.3f}")
    for a in sorted(perAuthor):
        c, t = perAuthor[a]
        print(f"  {a}: {c / max(1, t):.3f}  ({c}/{t})")

    OUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "author_mapping": base.author_to_idx,
                "best_val_author_acc": best}, OUT_WEIGHTS)
    print(f"\nWeights saved to: {OUT_WEIGHTS}")
    return best


if __name__ == "__main__":
    Train()
