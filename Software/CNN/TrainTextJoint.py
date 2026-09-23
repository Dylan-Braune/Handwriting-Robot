"""
TrainTextJoint.py -- trains ONE recogniser on BOTH the full Teklia/IAM-line
dataset (6480 train lines, general handwriting) AND your own personal pages
(Software/CNN/NOGIT/yeukita + dylan, ~300 lines, your handwriting +
technical/math notation), to fix what sequential fine-tuning couldn't:

    Measured on clean, human-verified line crops (Teklia val/test):
        original (Teklia-only)         93.87% / 91.60% char-acc
        naive personal fine-tune       84.26% / 80.56% char-acc  (real regression)
        frozen-conv personal fine-tune 85.87% / 82.62% char-acc  (barely better)
    Naive fine-tuning always drifts the model away from Teklia's
    distribution because it stops seeing Teklia examples entirely. Freezing
    the conv layers barely helped -- most of the drift turned out to be in
    the LSTM/output layers, not the conv feature extractor. The actual fix
    is to keep training on BOTH datasets together, every epoch, so nothing
    is ever "sequentially forgotten."

HOW THE TWO DATASETS ARE MIXED
    Teklia has 6480 train lines, your personal pages have ~300 -- if you
    just concatenated them, personal lines would be ~4.5% of every epoch
    and get drowned out (this is exactly the imbalance that makes "just
    combine the data" not enough on its own). Instead this uses a
    WeightedRandomSampler so personal lines are oversampled to make up
    --personal-fraction (default 20%) of every epoch, without literally
    duplicating any data on disk or in memory.

    Validation is reported SEPARATELY for each domain every epoch --
    Teklia val (976 lines) and personal val (~52 lines, same held-out
    lines TrainTextPersonal.py and BuildStyleProfile10Authors.py use) --
    so you can watch directly whether either one is regressing. The "best"
    checkpoint is chosen by the LOWER of the two char-accuracies (i.e. it
    won't call a checkpoint "best" for being great at one domain while
    quietly bad at the other).

USAGE (on Colab, after `git clone` + `pip install -q datasets`):
    python TrainTextJoint.py --init-from NOGIT/weights/paper_cnn_bilstm_ctc_hf_best.pt

    Upload paper_cnn_bilstm_ctc_hf_best.pt to the Colab session yourself
    (or Google Drive) first -- weights aren't committed to git, only code
    and the small personal-page data are.

WEIGHTS (separate name, nothing existing is overwritten):
    NOGIT/weights/paper_cnn_bilstm_ctc_joint_checkpoint.pt
    NOGIT/weights/paper_cnn_bilstm_ctc_joint_best.pt
"""
import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "AuthorReproductionStuff"))

import SegmentPage as PS
from ExtractIAMLines import ReadLabelLines
from TrainText import (
    CHAR_TO_IDX, CHARSET, MAX_SAFE_LABEL_CHARS, PaperCRNN,
    augment_line_image, collate_fn, encode_text, evaluate,
    resize_line_image_fixed, tensor_from_resized,
)
from TrainTextHF import HFLineDataset, load_hf, DEFAULT_HF_CACHE

NOGIT_DIR = SCRIPT_DIR / "NOGIT"
WEIGHTS_DIR = NOGIT_DIR / "weights"
NAME = "paper_cnn_bilstm_ctc_joint"
PERSONAL_AUTHORS = ["yeukita", "dylan", "robert", "owen", "aston", "abhinav"]
VAL_FRACTION = 0.15   # same fraction AND seed as TrainTextPersonal.py /
SPLIT_SEED = 0        # BuildStyleProfile10Authors.py -- same held-out lines everywhere


def collect_personal_lines():
    """Same personal-page collection as TrainTextPersonal.py: segments
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", required=True,
                    help="warm-start checkpoint, e.g. NOGIT/weights/paper_cnn_bilstm_ctc_hf_best.pt "
                         "-- REQUIRED, this script refuses to train such a mixed objective from scratch")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--personal-fraction", type=float, default=0.20,
                    help="fraction of each training epoch drawn from personal lines (default 20%%)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap Teklia train/val size for a quick smoke test")
    ap.add_argument("--cache-dir", default=str(DEFAULT_HF_CACHE))
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    ds = load_hf(Path(args.cache_dir))
    teklia_train = HFLineDataset(ds["train"], is_train=True, max_samples=args.max_samples)
    teklia_val = HFLineDataset(ds["validation"], is_train=False, max_samples=args.max_samples)
    personal_train_rows, personal_val_rows = collect_personal_lines()
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
    ckpt_path = WEIGHTS_DIR / f"{NAME}_checkpoint.pt"
    best_path = WEIGHTS_DIR / f"{NAME}_best.pt"

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


if __name__ == "__main__":
    main()
