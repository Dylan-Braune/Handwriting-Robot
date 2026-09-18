"""
TrainAuthor10.py -- writer-ID classifier (AuthorClassifierCNN, unchanged
from TrainAuthor.py) retrained on the NEW 10-author set: 8 kept dataset
authors (150,151,152,153,384,551,552,588 -- dropped 154/155 for being
redundant with 150/151/152's style cluster) + 2 personal authors
(yeukita, dylan).

HOW THE TWO DATA SOURCES ARE UNIFIED
    IAMLineDatasetRaw (TrainText.py) stores each line sample as a dict with
    PNG-encoded image bytes + a page_key "authorId/filename.png" + an
    is_holdout flag. That's a plain, source-agnostic format -- so instead
    of writing a parallel Dataset class for the personal authors, this
    just PNG-encodes their SegmentPage-derived line crops into the exact
    same dict shape and appends them into base.samples directly. Every
    downstream piece (author_folders, author_to_idx, holdout_split_indices,
    AuthorLabeledView, evaluate(), train()) then works unmodified across
    both data sources.

    The personal train/holdout split uses the SAME seed and fraction as
    BuildStyleProfile10Authors.py and TrainTextPersonal.py, so it's the
    same lines held out everywhere in this project, not three different
    "holdout" definitions for the same person's handwriting.

Run:
    python TrainAuthor10.py
"""
import io
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "AuthorReproductionStuff"))

import SegmentPage as PS
from ExtractIAMLines import ReadLabelLines
from TrainText import IAMLineDatasetRaw, encode_text
from TrainAuthor import (
    AuthorLabeledView, AuthorClassifierCNN, collate_fn, evaluate,
    DEFAULT_DATA_DIR, DEFAULT_CACHE_DIR, DEFAULT_TEXT_WEIGHTS, WEIGHTS_DIR,
)

NOGIT_DIR = SCRIPT_DIR / "NOGIT"
DATASET_AUTHORS = ["150", "151", "152", "153", "384", "551", "552", "588"]
PERSONAL_AUTHORS = ["yeukita", "dylan"]
VAL_FRACTION = 0.15
SPLIT_SEED = 0
RUN_NAME = "author_classifier_10new"


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
        for img_path in sorted((NOGIT_DIR / author).glob("*.jpg")):
            results, _preview, _meta = PS.ProcessPage(str(img_path))
            crops = [r["raw_crop"] for r in results if r["tag"] == "TEXT"]
            gt = ReadLabelLines(str(img_path))
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


def main():
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
    num_epochs = 60
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc, best_state = 0.0, None
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
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:03d}/{num_epochs} | loss {total_loss/max(1,len(trainLoader)):.3f} "
              f"| val author-acc {val_acc:.3f} (best {best_acc:.3f})", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    final_acc, per_author_acc = evaluate(model, valLoader, device, base.author_folders)
    print("=" * 88)
    print(f"Best val author-accuracy: {best_acc:.3f}")
    print("Per-author accuracy on held-out lines:")
    for a, acc in sorted(per_author_acc.items()):
        print(f"  {a}: {acc:.3f}")

    weights_path = WEIGHTS_DIR / f"{RUN_NAME}_weights.pt"
    torch.save({"model_state_dict": model.state_dict(),
               "author_mapping": base.author_to_idx,
               "best_val_author_acc": best_acc}, weights_path)
    print(f"\nWeights saved to: {weights_path}")


if __name__ == "__main__":
    main()
