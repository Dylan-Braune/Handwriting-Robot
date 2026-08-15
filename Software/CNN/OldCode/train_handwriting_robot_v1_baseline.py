import argparse
import hashlib
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytesseract
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from FullLineBoxMaker import ExtractLinePatches, ReadLabelLines

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# 0 is reserved for the CTC blank token.
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:'\"!?()-"
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARSET)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}


def encode_text(text):
    encoded = [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]
    return torch.tensor(encoded, dtype=torch.long)


def decode_ctc(log_probs):
    best_path = torch.argmax(log_probs, dim=2).detach().cpu().numpy()
    decoded = []

    for batch_idx in range(best_path.shape[1]):
        prev = None
        chars = []
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
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def normalize_line_image(pil_img, input_height, input_width):
    img = pil_img.convert("L")
    arr = np.array(img, dtype=np.float32)

    # Keep the original aspect ratio, then pad to a consistent CRNN input.
    h, w = arr.shape
    scale = input_height / float(max(1, h))
    new_w = min(input_width, max(1, int(w * scale)))
    resized = img.resize((new_w, input_height), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (input_width, input_height), color=255)
    canvas.paste(resized, (0, 0))

    arr = np.array(canvas, dtype=np.float32) / 255.0
    arr = 1.0 - arr
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)


class IAMLineDataset(Dataset):
    def __init__(self, root_dir, target_author=None, input_height=32, input_width=512,
                 max_pages=None, min_pages_for_holdout=3, use_cache=True, cache_dir=None):
        self.samples = []
        self.input_height = input_height
        self.input_width = input_width

        root_dir = self.resolve_dataset_root(Path(root_dir))

        # The page-loading + line-extraction + OCR pass below is identical every time
        # for a given (root_dir, target_author, height, width, max_pages, min_pages_for_holdout)
        # combination -- e.g. every trial in a hyperparameter sweep. Cache its result so
        # repeat runs (different --lr/--batch-size/etc, or just restarting) skip straight
        # to training instead of re-paying that ~10-20 min cost every time.
        cache_path = None
        if use_cache:
            cache_root = Path(cache_dir) if cache_dir else Path(__file__).resolve().parent / "dataset_cache"
            cache_root.mkdir(exist_ok=True)
            key = f"{root_dir}|{target_author}|{input_height}|{input_width}|{max_pages}|{min_pages_for_holdout}"
            key_hash = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
            cache_path = cache_root / f"dataset_{key_hash}.pt"

            if cache_path.exists():
                print(f"[Dataset] Loading cached line extraction: {cache_path.name}")
                try:
                    cached = torch.load(cache_path, weights_only=False)
                    self.samples = cached["samples"]
                    self.author_folders = cached["author_folders"]
                    self.author_to_idx = cached["author_to_idx"]
                    print(f"[Dataset] Loaded {len(self.samples)} cached line samples "
                          f"({len(self.author_folders)} authors) -- skipped re-extraction.")
                    return
                except Exception as e:
                    # A crash mid-write (before the atomic-rename fix below existed, or from
                    # an interrupted disk write) can leave a corrupt cache file. Don't die on
                    # a bad cache -- just rebuild it from scratch like there was no cache at all.
                    print(f"[Dataset] Cache file was unreadable ({e}); rebuilding from scratch.")
                    self.samples = []

        if target_author and target_author.lower() not in {"all", "none"}:
            self.author_folders = [target_author]
        else:
            self.author_folders = sorted(p.name for p in root_dir.iterdir() if p.is_dir())

        self.author_to_idx = {author: idx for idx, author in enumerate(self.author_folders)}
        print(f"[Dataset] Root: {root_dir}")
        print(f"[Dataset] Authors: {len(self.author_folders)} folder(s)")

        page_count = 0
        skipped_count = 0
        for author_id in self.author_folders:
            author_dir = root_dir / author_id
            if not author_dir.exists():
                raise FileNotFoundError(f"Could not find author folder: {author_dir}")

            author_idx = self.author_to_idx[author_id]
            author_pages = sorted(author_dir.glob("*.png"))

            # Only reserve a holdout page for authors with enough pages to spare one.
            # With ~2.15 pages/author on average across this dataset, unconditionally
            # holding out "the last page" per author (as an earlier version did) throws
            # away ~42% of all training lines and leaves many authors with zero training
            # examples at all -- authors with too few pages keep everything for training.
            labeled_pages = [p for p in author_pages if ReadLabelLines(str(p))]
            holdout_page_name = (
                labeled_pages[-1].name if len(labeled_pages) >= min_pages_for_holdout else None
            )

            for img_path in author_pages:
                if max_pages is not None and page_count >= max_pages:
                    break

                label_lines = ReadLabelLines(str(img_path))

                if not label_lines:
                    # Pages without a quality-gated _labels.txt (see generate_all_labels.py,
                    # which OCR-confidence-filters the printed header before writing labels)
                    # are skipped entirely rather than falling back to an ungated re-OCR here,
                    # since that previously produced garbage training targets (see the stale
                    # *_generated_labels.txt files this script used to leave behind).
                    skipped_count += 1
                    page_count += 1
                    print(f"  [Skip] {author_id}/{img_path.name}: no quality-gated label file")
                    continue

                expected_count = len(label_lines)
                line_samples, _generated_text_lines, _ = ExtractLinePatches(
                    str(img_path),
                    targetHeight=input_height,
                    maxWidth=input_width,
                    expectedLineCount=expected_count,
                    labelLines=label_lines
                )
                page_count += 1

                if not line_samples:
                    skipped_count += 1
                    print(f"  [Skip] {author_id}/{img_path.name}: no usable line/label pairs")
                    continue

                matched = min(len(line_samples), len(label_lines))
                status = "OK" if matched == len(label_lines) else "MISMATCH"
                is_holdout = img_path.name == holdout_page_name
                holdout_tag = " [HOLDOUT]" if is_holdout else ""
                print(f"  [{status}] {author_id}/{img_path.name}: {matched}/{len(label_lines)} aligned line samples{holdout_tag}")

                for idx in range(matched):
                    text = label_lines[idx]
                    target = encode_text(text)
                    if len(target) > 0:
                        self.samples.append({
                            "image": line_samples[idx]["processed_patch"],
                            "author": author_idx,
                            "target": target,
                            "text": text,
                            "page": img_path.name,
                            "line_idx": idx,
                            "is_holdout": is_holdout,
                        })

            if max_pages is not None and page_count >= max_pages:
                break

        print(f"[Dataset] Processed pages: {page_count} | Skipped pages: {skipped_count}")
        print(f"[Dataset] Total line samples: {len(self.samples)}")

        if cache_path is not None:
            # Write to a temp file, fsync it so the bytes are physically on disk (not just
            # handed to the OS's write-back cache -- a hard crash can lose those before they
            # land), THEN rename into place. Either the rename happens (fully valid, durable
            # file) or it doesn't (no file at all) -- never a half-written/corrupt one.
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with open(tmp_path, "wb") as f:
                torch.save({
                    "samples": self.samples,
                    "author_folders": self.author_folders,
                    "author_to_idx": self.author_to_idx,
                }, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, cache_path)
            print(f"[Dataset] Cached this extraction to {cache_path.name} -- future runs with the same "
                  f"data-dir/author/height/width/max-pages/min-pages-for-holdout will load instantly.")

    @staticmethod
    def resolve_dataset_root(root_dir):
        data_dir = root_dir / "data"
        if data_dir.exists() and any(p.is_dir() for p in data_dir.iterdir()):
            return data_dir
        return root_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_tensor = normalize_line_image(item["image"], self.input_height, self.input_width)
        author_tensor = torch.tensor(item["author"], dtype=torch.long)
        return image_tensor, author_tensor, item["target"], item["text"], item["page"], item["line_idx"]

    def holdout_split_indices(self):
        train_idx = [i for i, s in enumerate(self.samples) if not s["is_holdout"]]
        test_idx = [i for i, s in enumerate(self.samples) if s["is_holdout"]]
        return train_idx, test_idx


def collate_fn(batch):
    images, authors, targets, texts, pages, line_idxs = zip(*batch)
    images = torch.stack(images, 0)
    authors = torch.stack(authors, 0)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flat_targets = torch.cat(targets)
    return images, authors, flat_targets, target_lengths, list(texts), list(pages), list(line_idxs)


class MultiTaskLineCRNN(nn.Module):
    def __init__(self, num_authors, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 16 x 256

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),        # 8 x 128

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),        # 4 x 128

            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),        # 2 x 128

            nn.Conv2d(192, 256, kernel_size=(2, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),       # 1 x 128
        )

        self.author_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.author_head = nn.Sequential(
            nn.Linear(256, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(96, num_authors),
        )

        self.sequence = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=False,
        )
        self.text_head = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.cnn(x)

        pooled = self.author_pool(features).flatten(1)
        author_logits = self.author_head(pooled)

        seq = features.squeeze(2).permute(2, 0, 1)
        seq, _ = self.sequence(seq)
        text_logits = self.text_head(seq)
        text_log_probs = nn.functional.log_softmax(text_logits, dim=2)
        return author_logits, text_log_probs


def evaluate(model, dataloader, device):
    model.eval()
    total_chars = 0
    total_char_errors = 0
    exact = 0
    total = 0
    author_correct = 0
    examples = []

    with torch.no_grad():
        for images, authors, targets, target_lengths, texts, pages, line_idxs in dataloader:
            images = images.to(device)
            author_logits, text_log_probs = model(images)
            predictions = decode_ctc(text_log_probs)
            author_preds = torch.argmax(author_logits, dim=1).cpu()

            for pred, truth, page, line_idx, author_pred, author_true in zip(
                predictions, texts, pages, line_idxs, author_preds, authors
            ):
                total += 1
                total_chars += len(truth)
                total_char_errors += levenshtein(pred, truth)
                exact += int(pred == truth)
                author_correct += int(int(author_pred) == int(author_true))
                if len(examples) < 5:
                    examples.append((page, line_idx, truth, pred))

    cer = total_char_errors / max(1, total_chars)
    char_acc = 1.0 - cer
    exact_acc = exact / max(1, total)
    author_acc = author_correct / max(1, total)
    return cer, char_acc, exact_acc, author_acc, examples


def train_model(data_dir, target_author=None, num_epochs=30, batch_size=8, learning_rate=0.001,
                input_height=32, input_width=512, val_ratio=0.10, eval_every=5,
                max_pages=None, seed=7, resume=False, checkpoint_every=1,
                target_char_acc=0.95, target_patience=3, held_out_last_page=False,
                min_pages_for_holdout=3, author_loss_weight=1.0, run_name=None, use_cache=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Training on: {device}")
    print(f"[Config] author_loss_weight={author_loss_weight} "
          f"({'author-ID training paused' if author_loss_weight == 0 else 'author-ID training active'})")

    dataset = IAMLineDataset(
        root_dir=data_dir,
        target_author=target_author,
        input_height=input_height,
        input_width=input_width,
        max_pages=max_pages,
        use_cache=use_cache,
        min_pages_for_holdout=min_pages_for_holdout,
    )
    if len(dataset) < 4:
        raise RuntimeError("Not enough line samples to train.")

    if held_out_last_page:
        train_idx, test_idx = dataset.holdout_split_indices()
        if not test_idx:
            raise RuntimeError("held_out_last_page requested but no holdout lines were found.")
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, test_idx)
        train_count, val_count = len(train_idx), len(test_idx)
        print(f"[Mode] Page-level holdout: {train_count} train lines, {val_count} held-out lines "
              f"(last page per author, {len(dataset.author_folders)} authors).")
    elif val_ratio <= 0:
        train_count = len(dataset)
        val_count = 0
        train_set = dataset
        val_set = dataset
        print("[Mode] Overfit/sanity mode: training and evaluating on all available lines.")
    else:
        val_count = max(8, int(len(dataset) * val_ratio))
        train_count = len(dataset) - val_count
        train_set, val_set = random_split(
            dataset,
            [train_count, val_count],
            generator=torch.Generator().manual_seed(seed),
        )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = MultiTaskLineCRNN(
        num_authors=len(dataset.author_folders),
        num_classes=len(CHARSET) + 1,
    ).to(device)

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    author_loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    cnn_dir = Path(__file__).resolve().parent
    # Every version of this script (train_handwriting_robot_v1_baseline.py,
    # a future _v2_..., etc.) saves under its own name in weights/, so newer
    # iterations never overwrite older ones' results. Pass run_name explicitly
    # (e.g. for a hyperparameter sweep) to keep multiple trials of the *same*
    # script from clobbering each other's checkpoint/weights files.
    run_name = run_name or Path(__file__).stem
    weights_dir = cnn_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    weights_path = weights_dir / f"{run_name}_weights.pth"
    checkpoint_path = weights_dir / f"{run_name}_checkpoint.pth"

    start_epoch = 1
    best_val_cer = float("inf")
    best_state = None
    target_streak = 0

    if resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_cer = ckpt["best_val_cer"]
        best_state = ckpt.get("best_state")
        target_streak = ckpt.get("target_streak", 0)
        print(f"[Resume] Loaded checkpoint from epoch {ckpt['epoch']}, resuming at epoch {start_epoch}")

    print("\nStarting multi-task CRNN training (text recognition + author identification)...")
    print(f"Train lines: {train_count} | Val lines: {val_count if val_ratio > 0 else 'same as train'} | "
          f"Authors: {len(dataset.author_folders)} | Input: {input_height}x{input_width}")
    print("=" * 88)

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_text_loss = 0.0
        total_author_loss = 0.0

        for images, authors, targets, target_lengths, texts, pages, line_idxs in train_loader:
            images = images.to(device)
            authors = authors.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            author_logits, text_log_probs = model(images)

            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=text_log_probs.size(0),
                dtype=torch.long,
                device=device,
            )

            loss_text = ctc_loss_fn(text_log_probs, targets, input_lengths, target_lengths)
            loss_author = author_loss_fn(author_logits, authors)
            loss = loss_text + author_loss_weight * loss_author
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_text_loss += float(loss_text.item())
            total_author_loss += float(loss_author.item())

        scheduler.step()
        should_eval = epoch == 1 or epoch % eval_every == 0 or epoch == num_epochs
        if should_eval:
            val_cer, val_char_acc, val_exact, val_author_acc, _ = evaluate(model, val_loader, device)
            if val_cer < best_val_cer:
                best_val_cer = val_cer
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if val_char_acc >= target_char_acc:
                target_streak += 1
            else:
                target_streak = 0
        else:
            val_cer = best_val_cer
            val_char_acc, val_exact, val_author_acc = 1.0 - best_val_cer, 0.0, 0.0

        if should_eval:
            batches = max(1, len(train_loader))
            print(
                f"Epoch {epoch:03d}/{num_epochs} | "
                f"loss {total_loss / batches:.3f} (text {total_text_loss / batches:.3f} / "
                f"author {total_author_loss / batches:.3f}) | "
                f"val char-acc {val_char_acc:.4f} | val CER {val_cer:.4f} | "
                f"val exact-line {val_exact:.3f} | val author-acc {val_author_acc:.3f}"
            )

        if epoch % checkpoint_every == 0 or epoch == num_epochs:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_cer": best_val_cer,
                "best_state": best_state,
                "target_streak": target_streak,
                "author_mapping": dataset.author_to_idx,
                "charset": CHARSET,
                "input_height": input_height,
                "input_width": input_width,
            }, checkpoint_path)

        if should_eval and target_streak >= target_patience:
            print(f"\n[Early stop] val char-accuracy >= {target_char_acc:.2%} for {target_streak} consecutive evals.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_cer, train_char_acc, train_exact, train_author_acc, _ = evaluate(model, train_loader, device)
    val_cer, val_char_acc, val_exact, val_author_acc, val_examples = evaluate(model, val_loader, device)
    all_cer, all_char_acc, all_exact, all_author_acc, all_examples = evaluate(model, all_loader, device)

    torch.save({
        "model_state_dict": model.state_dict(),
        "author_mapping": dataset.author_to_idx,
        "charset": CHARSET,
        "input_height": input_height,
        "input_width": input_width,
        "target_author": target_author,
        "author_training_enabled": True,
    }, weights_path)

    print("=" * 88)
    print(f"Best validation CER: {best_val_cer:.4f} (char-acc {1.0 - best_val_cer:.4f})")
    print(f"Train char-acc: {train_char_acc:.4f} | Train exact-line: {train_exact:.3f} | Train author-acc: {train_author_acc:.3f}")
    print(f"Val   char-acc: {val_char_acc:.4f} | Val   exact-line: {val_exact:.3f} | Val   author-acc: {val_author_acc:.3f}")
    print(f"All   char-acc: {all_char_acc:.4f} | All   exact-line: {all_exact:.3f} | All   author-acc: {all_author_acc:.3f}")
    print("\nSample validation predictions:")
    for page, line_idx, truth, pred in val_examples:
        print(f"  {page} line {line_idx:02d}")
        print(f"    truth: {truth}")
        print(f"    pred : {pred}")
    print(f"\nWeights saved to:\n  {weights_path}")


def main():
    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir.parents[1] / "Data" / "Datasets" / "IAMpages671"

    parser = argparse.ArgumentParser(description="Train a line-level CRNN for handwriting text recognition.")
    parser.add_argument("--data-dir", default=str(default_data_dir))
    parser.add_argument("--author", default=None, help="Optional author folder to train on. Default trains all authors.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-pages", type=int, default=None, help="Optional smoke-test limit. Default uses all pages.")
    parser.add_argument("--resume", action="store_true", help="Resume from handwriting_robot_line_crnn_checkpoint.pth if present.")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--target-char-acc", type=float, default=0.95)
    parser.add_argument("--target-patience", type=int, default=3, help="Consecutive evals at/above target-char-acc before early stopping.")
    parser.add_argument("--held-out-last-page", action="store_true",
                        help="Reserve each author's alphabetically-last page as the test set, instead of a random line-level val split.")
    parser.add_argument("--min-pages-for-holdout", type=int, default=3,
                        help="Only reserve a holdout page for authors with at least this many labeled pages; "
                             "authors with fewer keep all their pages in training.")
    parser.add_argument("--author-loss-weight", type=float, default=1.0,
                        help="Multiplier on the author-ID loss term. Set to 0 to pause author-ID training "
                             "(text-recognition still trains normally; author-acc is still reported, just not optimized).")
    parser.add_argument("--run-name", default=None,
                        help="Override the weights/checkpoint filename tag (default: this script's own filename). "
                             "Use a distinct value per trial when running a hyperparameter sweep so trials don't overwrite each other.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the dataset_cache/ line-extraction cache and always re-extract from scratch.")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        target_author=args.author,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        input_height=args.height,
        input_width=args.width,
        val_ratio=args.val_ratio,
        eval_every=args.eval_every,
        max_pages=args.max_pages,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        target_char_acc=args.target_char_acc,
        target_patience=args.target_patience,
        held_out_last_page=args.held_out_last_page,
        min_pages_for_holdout=args.min_pages_for_holdout,
        author_loss_weight=args.author_loss_weight,
        run_name=args.run_name,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
