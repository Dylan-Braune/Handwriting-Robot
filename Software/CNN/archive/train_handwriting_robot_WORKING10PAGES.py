import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from FullLineBoxMaker import ExtractLinePatches, ReadLabelLines


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
    def __init__(self, root_dir, target_author="150", input_height=32, input_width=512):
        self.samples = []
        self.input_height = input_height
        self.input_width = input_width

        root_dir = Path(root_dir)
        if target_author:
            self.author_folders = [target_author]
        else:
            self.author_folders = sorted(p.name for p in root_dir.iterdir() if p.is_dir())

        self.author_to_idx = {author: idx for idx, author in enumerate(self.author_folders)}
        print(f"[Dataset] Authors: {self.author_folders}")

        for author_id in self.author_folders:
            author_dir = root_dir / author_id
            if not author_dir.exists():
                raise FileNotFoundError(f"Could not find author folder: {author_dir}")

            author_idx = self.author_to_idx[author_id]
            for img_path in sorted(author_dir.glob("*.png")):
                label_lines = ReadLabelLines(str(img_path))
                if not label_lines:
                    print(f"  [Skip] No labels for {img_path.name}")
                    continue

                line_samples, _, _ = ExtractLinePatches(
                    str(img_path),
                    targetHeight=input_height,
                    maxWidth=input_width,
                    expectedLineCount=len(label_lines),
                    labelLines=label_lines
                )

                matched = min(len(line_samples), len(label_lines))
                status = "OK" if matched == len(label_lines) else "MISMATCH"
                print(f"  [{status}] {img_path.name}: {matched}/{len(label_lines)} aligned line samples")

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
                        })

        print(f"[Dataset] Total line samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_tensor = normalize_line_image(item["image"], self.input_height, self.input_width)
        author_tensor = torch.tensor(item["author"], dtype=torch.long)
        return image_tensor, author_tensor, item["target"], item["text"], item["page"], item["line_idx"]


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
            authors = authors.to(device)
            author_logits, text_log_probs = model(images)
            predictions = decode_ctc(text_log_probs)
            author_pred = torch.argmax(author_logits, dim=1)
            author_correct += int((author_pred == authors).sum().item())

            for pred, truth, page, line_idx in zip(predictions, texts, pages, line_idxs):
                total += 1
                total_chars += len(truth)
                total_char_errors += levenshtein(pred, truth)
                exact += int(pred == truth)
                if len(examples) < 5:
                    examples.append((page, line_idx, truth, pred))

    cer = total_char_errors / max(1, total_chars)
    exact_acc = exact / max(1, total)
    author_acc = author_correct / max(1, total)
    return cer, exact_acc, author_acc, examples


def train_model(data_dir, target_author="150", num_epochs=200, batch_size=8, learning_rate=0.001,
                input_height=32, input_width=512, val_ratio=0.18, eval_every=5, seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Training on: {device}")

    dataset = IAMLineDataset(
        root_dir=data_dir,
        target_author=target_author,
        input_height=input_height,
        input_width=input_width,
    )
    if len(dataset) < 4:
        raise RuntimeError("Not enough line samples to train.")

    if val_ratio <= 0:
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

    author_loss_fn = nn.CrossEntropyLoss()
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print("\nStarting CRNN line-text training...")
    print(f"Train lines: {train_count} | Val lines: {val_count if val_ratio > 0 else 'same as train'} | Input: {input_height}x{input_width}")
    print("=" * 88)

    best_val_cer = float("inf")
    best_state = None
    for epoch in range(1, num_epochs + 1):
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
            loss = loss_text + 0.15 * loss_author
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_text_loss += float(loss_text.item())
            total_author_loss += float(loss_author.item())

        scheduler.step()
        should_eval = epoch == 1 or epoch % eval_every == 0 or epoch == num_epochs
        if should_eval:
            val_cer, val_exact, val_author_acc, _ = evaluate(model, val_loader, device)
            if val_cer < best_val_cer:
                best_val_cer = val_cer
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            val_cer, val_exact, val_author_acc = best_val_cer, 0.0, 0.0

        if should_eval:
            batches = max(1, len(train_loader))
            print(
                f"Epoch {epoch:03d}/{num_epochs} | "
                f"loss {total_loss / batches:.3f} | "
                f"text {total_text_loss / batches:.3f} | "
                f"author {total_author_loss / batches:.3f} | "
                f"val CER {val_cer:.3f} | "
                f"val exact {val_exact:.3f} | "
                f"val author {val_author_acc:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_cer, train_exact, train_author_acc, _ = evaluate(model, train_loader, device)
    val_cer, val_exact, val_author_acc, val_examples = evaluate(model, val_loader, device)
    all_cer, all_exact, all_author_acc, all_examples = evaluate(model, all_loader, device)

    cnn_dir = Path(__file__).resolve().parent
    weights_path = cnn_dir / "handwriting_robot_line_crnn_weights.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "author_mapping": dataset.author_to_idx,
        "charset": CHARSET,
        "input_height": input_height,
        "input_width": input_width,
        "target_author": target_author,
    }, weights_path)

    print("=" * 88)
    print(f"Best validation CER: {best_val_cer:.3f}")
    print(f"Train CER: {train_cer:.3f} | Train exact: {train_exact:.3f} | Train author: {train_author_acc:.3f}")
    print(f"Val   CER: {val_cer:.3f} | Val exact: {val_exact:.3f} | Val author: {val_author_acc:.3f}")
    print(f"All   CER: {all_cer:.3f} | All exact: {all_exact:.3f} | All author: {all_author_acc:.3f}")
    print("\nSample validation predictions:")
    for page, line_idx, truth, pred in val_examples:
        print(f"  {page} line {line_idx:02d}")
        print(f"    truth: {truth}")
        print(f"    pred : {pred}")
    print(f"\nWeights saved to:\n  {weights_path}")


def main():
    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir.parents[1] / "Data" / "Datasets" / "IAMpages10"

    parser = argparse.ArgumentParser(description="Train a line-level multi-task CRNN for handwriting text and author ID.")
    parser.add_argument("--data-dir", default=str(default_data_dir))
    parser.add_argument("--author", default="150")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.18)
    parser.add_argument("--eval-every", type=int, default=5)
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
    )


if __name__ == "__main__":
    main()
