"""
TrainTextHF.py -- train the SAME text recogniser as TrainText.py, but on the
pre-segmented Teklia/IAM-line dataset from HuggingFace instead of our own
page scans + home-grown line segmentation.

WHY
    Teklia/IAM-line ships ~10.4k human-segmented IAM line images with clean
    transcriptions (train 6480 / validation 976 / test 2920). Training on it
    removes every source of noise our own pipeline adds -- page-bounds
    detection, periodicity line splitting, CTC forced-alignment labels -- so
    it's a cleaner signal for the recogniser and comparable to published IAM
    line-HTR numbers.

WHAT IS SHARED WITH TrainText.py
    Everything model-side is imported verbatim: the PaperCRNN architecture,
    the 64x640 fixed-stretch preprocessing, the CHARSET, the greedy CTC
    decode, the augmentation recipe, and evaluate(). So a checkpoint trained
    here is a drop-in weights file for ClassifyText.py -- point it at
    NOGIT/weights/paper_cnn_bilstm_ctc_hf_best.pt.

    Only the DATA layer is different: a thin torch Dataset over the HF split
    replaces IAMLineDatasetRaw / LineImageCache / ExtractLinePatches.

WEIGHTS
    NOGIT/weights/paper_cnn_bilstm_ctc_hf_checkpoint.pt  (full resume state)
    NOGIT/weights/paper_cnn_bilstm_ctc_hf_best.pt        (best val-loss weights)
    Separate names from your existing paper_cnn_bilstm_ctc_best.pt --
    nothing you already have is overwritten.

SETUP
    pip install datasets

USAGE
    python TrainTextHF.py                     # train (auto-resumes)
    python TrainTextHF.py --eval-only         # load best weights, print val + test CER
    python TrainTextHF.py --max-samples 500   # quick smoke test on a tiny slice
    python TrainTextHF.py --epochs 60 --eval-every 2
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Reuse the model + preprocessing + eval from the existing trainer unchanged.
# (Importing TrainText also pulls in ExtractIAMLines / pytesseract / scipy the
#  same way ClassifyText.py does -- nothing new is required beyond `datasets`.)
from TrainText import (
    CHARSET,
    CHAR_TO_IDX,
    DualLogger,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    MAX_SAFE_LABEL_CHARS,
    PaperCRNN,
    augment_line_image,
    collate_fn,
    encode_text,
    evaluate,
    resize_line_image_fixed,
    tensor_from_resized,
)

SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SCRIPT_DIR / "NOGIT" / "weights"
NAME = "paper_cnn_bilstm_ctc_hf"
HF_DATASET = "Teklia/IAM-line"
DEFAULT_HF_CACHE = SCRIPT_DIR / "NOGIT" / "hf_cache"


# -----------------------------------------------------------------------------
# Data: a thin torch Dataset over one HuggingFace split.
#
# Teklia/IAM-line columns:  image -> PIL.Image (RGB, 128px tall)
#                           text  -> line transcription (mixed case)
#
# The label filter mirrors encode_text()'s "drop unknown chars silently"
# behaviour, plus the same MAX_SAFE_LABEL_CHARS cap TrainText uses so a label
# CTC can't possibly align (T = INPUT_WIDTH // 4 = 160) never poisons a batch.
# Rows whose transcription is empty after filtering, or too long, are dropped
# up front and counted so you can see how much of the split was usable.
# -----------------------------------------------------------------------------
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
# Training loop -- same recipe as TrainText.train(): RMSprop lr=1e-3
# wd=1e-5, ReduceLROnPlateau on val loss, best-val-loss checkpointing, warm
# restarts, early stop after N validation checks with no improvement.
# -----------------------------------------------------------------------------
def train(config, device):
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
    ckpt_path = WEIGHTS_DIR / f"{NAME}_checkpoint.pt"
    best_path = WEIGHTS_DIR / f"{NAME}_best.pt"

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
                print(f"\r[{NAME}] Epoch {epoch:03d} | Batch {n_batches:04d}/"
                      f"{n_train_batches} | Running loss {total_loss / n_batches:.3f} "
                      f"| {spb:.2f}s/batch | ETA {eta / 60:.1f} min",
                      end="", flush=True)

        print()
        train_loss = total_loss / max(1, n_batches)
        print(f"[{NAME}] Epoch {epoch:03d} training pass done in "
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

            print(f"[{NAME}] Epoch {epoch:03d}/{max_epochs} | Train {train_loss:.3f} "
                  f"| Val loss {val_loss:.3f} (best {best_val_loss:.3f}) | "
                  f"Val char-acc {val_char_acc:.2%} | Val CER {val_cer:.2%} | "
                  f"val {((time.time() - vs) / 60):.1f} min{lr_msg}{saved_msg}")
        else:
            print(f"[{NAME}] Epoch {epoch:03d}/{max_epochs} | Train {train_loss:.3f} "
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
                print(f"[{NAME}] No val improvement for {early_stop_patience} "
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
            print(f"[{NAME}] Early stopping for good after {max_restarts} warm "
                  f"restart(s).")
            break

    return best_val_loss


def eval_only(config, device):
    """Load the best HF-trained weights and report CER on validation + test.
    Falls back to _checkpoint.pt if _best.pt isn't there yet."""
    best_path = WEIGHTS_DIR / f"{NAME}_best.pt"
    ckpt_path = WEIGHTS_DIR / f"{NAME}_checkpoint.pt"
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Train the PaperCRNN text recogniser on Teklia/IAM-line.")
    p.add_argument("--eval-only", action="store_true",
                   help="Load the best HF weights and print val + test CER, no training.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap each split to the first N usable lines (quick smoke test).")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--max-restarts", type=int, default=8)
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore any existing checkpoint and start fresh.")
    p.add_argument("--cache-dir", default=str(DEFAULT_HF_CACHE),
                   help="Where HuggingFace caches the downloaded dataset.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    SCRIPT_DIR.joinpath("NOGIT").mkdir(exist_ok=True)
    sys.stdout = DualLogger(str(SCRIPT_DIR / f"{NAME}.log"))
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
    }

    if args.eval_only:
        eval_only(config, device)
        return

    print(f"\n--- Training {NAME} on {HF_DATASET} "
          f"(lr={config['lr']}, batch_size={config['batch_size']}, "
          f"max_epochs={config['epochs']}) ---")
    try:
        best = train(config, device)
        print(f"\nDone. Best validation loss: {best:.4f}")
    except Exception:
        import traceback
        print("\n[FATAL] Training crashed:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
