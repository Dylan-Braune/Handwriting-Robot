"""
TrainTextPersonal.py -- fine-tunes the SAME text recogniser (PaperCRNN) on
YOUR OWN handwriting: the 13 personal lab-book photos in NOGIT/yeukita and
NOGIT/dylan, segmented with SegmentPage.ProcessPage (the personal-page
pipeline, same one ClassifyText.py uses for non-IAM pages) and paired with
their hand-verified _labels.txt transcriptions.

WHY A SEPARATE WEIGHTS FILE
    This model is also used for writer-ID/style-profile work on 10 OTHER
    authors' handwriting (BuildStyleProfile.py etc.) and for the IAM/Teklia
    holdout benchmark. Fine-tuning on ~330 lines of just YOUR handwriting
    could easily make it worse at reading everyone else's, even as it gets
    better at yours. So this NEVER overwrites paper_cnn_bilstm_ctc_best.pt
    or paper_cnn_bilstm_ctc_hf_best.pt -- it warm-starts FROM one of them
    and saves to its own name:
        NOGIT/weights/paper_cnn_bilstm_ctc_personal_checkpoint.pt
        NOGIT/weights/paper_cnn_bilstm_ctc_personal_best.pt

DATA
    ~330 lines total across 13 pages is small for a CTC recognizer, so:
    - warm start is mandatory (never trains from scratch)
    - low learning rate, few epochs
    - a genuine held-out validation split (per-page random split, not whole
      pages held out, so both splits see the same technical vocabulary --
      math notation, Greek-letter names, etc. -- that's unevenly
      distributed across pages)
    - characters outside the model's existing CHARSET (e.g. ^ _ = [ ] which
      show up in equations) are silently dropped from the training target,
      exactly the same convention TrainTextHF.py's HFLineDataset already
      uses for its own out-of-vocabulary characters -- not a new policy
      invented here.

USAGE
    python TrainTextPersonal.py                       # fine-tune (auto-resumes)
    python TrainTextPersonal.py --epochs 60 --lr 3e-4
    python TrainTextPersonal.py --init-from NOGIT/weights/paper_cnn_bilstm_ctc_hf_best.pt
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import SegmentPage as PS
from ExtractIAMLines import ReadLabelLines
from TrainText import (
    CHAR_TO_IDX,
    CHARSET,
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
NOGIT_DIR = SCRIPT_DIR / "NOGIT"
WEIGHTS_DIR = NOGIT_DIR / "weights"
NAME = "paper_cnn_bilstm_ctc_personal"
PERSONAL_DIRS = ["yeukita", "dylan", "robert", "owen", "aston", "abhinav"]
DEFAULT_INIT_FROM = WEIGHTS_DIR / "paper_cnn_bilstm_ctc_hf_best.pt"
VAL_FRACTION = 0.15
SPLIT_SEED = 0


def collect_personal_lines():
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

    rng = random.Random(SPLIT_SEED)
    train_rows, val_rows = [], []
    # Per-page shuffle+split so both splits see every page's vocabulary,
    # rather than holding out whole pages (which would starve val/train of
    # whatever technical notation is unique to the held-out pages).
    by_page = {}
    for page, crop, text in all_rows:
        by_page.setdefault(page, []).append((crop, text))
    for page, rows in by_page.items():
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * VAL_FRACTION))
        val_rows.extend(rows[:n_val])
        train_rows.extend(rows[n_val:])

    print(f"[PersonalData] {len(train_rows)} train / {len(val_rows)} val lines "
          f"from {len(by_page)} pages ({len(all_rows)} total usable lines)")
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
        img_tensor = tensor_from_resized(pil_img)
        return img_tensor, encode_text(label), label


def train(config, device):
    train_rows, val_rows = collect_personal_lines()
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

    name = NAME + ("_frozen" if config.get("freeze_conv") else "")
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init-from", default=str(DEFAULT_INIT_FROM))
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--freeze-conv", action="store_true",
                    help="freeze the CNN feature extractor, only train LSTM+output -- "
                         "saves to a separate _frozen checkpoint, see train()'s comment")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    train(dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
              init_from=args.init_from, no_resume=args.no_resume,
              freeze_conv=args.freeze_conv), device)
