"""
TrainAuthor.py

Writer identification, trained as its OWN focused model rather than bolted
back onto the text recognizer as a second output head.

WHY A SEPARATE MODEL, NOT THE OLD JOINT ONE: the version of this that
existed before (train_handwriting_robot_v1_baseline.py -- deleted in commit
63ba8d9 "FolderCleanupNOGIT", recoverable from git history) trained a single
MultiTaskLineCRNN with both an author_head and a text CTC head sharing one
backbone. On the same 10 IAM authors this script targets (150/151/152/153/
154/155/384/551/552/588 -- Data/Datasets/IAMpages10, built by
Software/DatasetSplitting/DatasetSplit10Authors.py), see
logs/train_10author.log: author-ID accuracy reached 98.9-100% by epoch 20,
while text accuracy stalled around 28-29% char-accuracy (70% CER) for the
whole 200-epoch run. That's not a coincidence of hyperparameters -- writer-
ID over 10 known classes is just a much easier task than open-vocabulary
text recognition, so joint training keeps spending gradient budget on a
problem that's already solved while the harder problem stays undertrained.
Splitting them into two focused models (this file + train_paper_cnn_bilstm_
ctc.py) matches how the writer-ID literature actually does it too (e.g.
arXiv:2009.04877, a dedicated single-task writer-ID CNN, not a joint model).

WHY THIS SHOULD BE MORE ACCURATE THAN THE OLD RUN even before considering
the split: it reuses TrainText.py's IAMLineDatasetRaw,
i.e. the CURRENT line segmentation + label alignment (regenerate_labels_
with_alignment.py's output), not the old pytesseract-labelled, periodicity-
segmented data the 2024 joint run trained on. Better inputs, easier task,
dedicated model.

TRANSFER LEARNING OPTION: this file's backbone (stage1/pool1/stage2/pool2/
stage3/height_pool) is architecturally IDENTICAL to PaperCRNN's, on purpose
-- so a trained paper_cnn_bilstm_ctc_best.pt's backbone weights can be
loaded straight in as a starting point (load_backbone_from_text_model
below) before training the author head. That backbone already learned to
notice pen-stroke shape as a side effect of learning to read handwriting
(the same logic as "Encoding CNN Activations for Writer Recognition",
arXiv:1712.07923) so it's a reasonable head start rather than training a
CNN from nothing on a fairly small 10-author dataset.

Does NOT modify TrainText.py -- only imports from it.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from TrainText import (
    IAMLineDatasetRaw,
    PaperCRNN,
    _decode_png,
    augment_line_image,
    conv_block,
    resize_line_image_fixed,
    tensor_from_resized,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[1] / "Data" / "Datasets" / "IAMpages10"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "NOGIT" / "line_cache_authors10"
DEFAULT_TEXT_WEIGHTS = SCRIPT_DIR / "NOGIT" / "weights" / "paper_cnn_bilstm_ctc_best.pt"
WEIGHTS_DIR = SCRIPT_DIR / "NOGIT" / "weights"


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


if __name__ == "__main__":
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
