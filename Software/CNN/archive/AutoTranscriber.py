import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# 1. FIRST-PRINCIPLES NUMPY LINE EXTRACTION PIPELINE
# =====================================================================

def OtsuThreshold(grayImg):
    hist, _ = np.histogram(grayImg, bins=256, range=(0, 256))
    total = grayImg.size
    currentMax, bestThresh, sumB, sum1, wB = 0, 0, 0, np.dot(np.arange(256), hist), 0

    for t in range(256):
        wB += hist[t]
        if wB == 0: continue
        wF = total - wB
        if wF == 0: break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        varBetween = wB * wF * ((mB - mF) ** 2)
        if varBetween > currentMax:
            currentMax = varBetween
            bestThresh = t

    return (grayImg < bestThresh).astype(np.uint8) * 255


def DetectPageBounds(binaryImg):
    h, w = binaryImg.shape
    ruleRows = []
    for r in range(h):
        row = binaryImg[r, :] > 0
        if not np.any(row): continue
        diff = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
        if len(starts) > 0 and np.max(ends - starts) > w * 0.25:
            ruleRows.append(r)

    if not ruleRows:
        return int(h * 0.12), int(h * 0.80)

    clusters, currGroup = [], [ruleRows[0]]
    for i in range(1, len(ruleRows)):
        if ruleRows[i] <= ruleRows[i - 1] + 3:
            currGroup.append(ruleRows[i])
        else:
            clusters.append(int(np.mean(currGroup)))
            currGroup = [ruleRows[i]]
    clusters.append(int(np.mean(currGroup)))

    topCandidates = [c for c in clusters if c < h * 0.35]
    botCandidates = [c for c in clusters if c > h * 0.65]

    topY = topCandidates[-1] + 2 if topCandidates else int(h * 0.12)
    botY = botCandidates[0] - 3 if botCandidates else int(h * 0.80)

    return topY, botY


def ExtractLinePatchesFromScan(imgPath, targetHeight=32, maxWidth=1024):
    """
    Extracts normalized 32x1024 grayscale line patches directly from a full page scan.
    """
    rawImage = Image.open(imgPath)
    RawGrayscaleArray = np.array(rawImage.convert('L'))
    h, w = RawGrayscaleArray.shape

    BinaryInvertedImage = OtsuThreshold(RawGrayscaleArray)
    topY, botY = DetectPageBounds(BinaryInvertedImage)
    hwRegion = BinaryInvertedImage[topY:botY, :].copy()
    regionH, regionW = hwRegion.shape

    # Mask vertical rule lines & left margin borders
    colInkHeights = np.sum(hwRegion > 0, axis=0)
    vertLineCols = np.where(colInkHeights > regionH * 0.40)[0]
    hwRegion[:, vertLineCols] = 0
    hwRegion[:, :int(w * 0.07)] = 0

    # 1D Horizontal Projection
    rowInkCount = np.sum(hwRegion > 0, axis=1)
    hasInk = rowInkCount > 10

    pad1D = np.pad(hasInk, (2, 2), mode='constant')
    smoothedInk = np.zeros_like(hasInk)
    for dy in range(5):
        smoothedInk = np.logical_or(smoothedInk, pad1D[dy:dy + len(hasInk)])

    diff = np.diff(np.concatenate(([0], smoothedInk.astype(np.int8), [0])))
    lineStarts, lineEnds = np.where(diff == 1)[0], np.where(diff == -1)[0]

    rawLines = [(s, e) for s, e in zip(lineStarts, lineEnds) if (e - s) >= 8]

    # Merge descenders ('y', 'p', 'g')
    mergedLines = []
    for line in rawLines:
        if not mergedLines:
            mergedLines.append(line)
        else:
            prevS, prevE = mergedLines[-1]
            currS, currE = line
            if (currS - prevE) < 16:
                mergedLines[-1] = (prevS, currE)
            else:
                mergedLines.append(line)

    LinePatches = []
    for s, e in enumerate(mergedLines):
        sRow, eRow = e
        absY1 = topY + max(0, sRow - 3)
        absY2 = topY + min(regionH, eRow + 3)
        cropH = absY2 - absY1

        lineBinary = hwRegion[sRow:eRow, :]
        colSum = np.sum(lineBinary > 0, axis=0)
        inkCols = np.where(colSum > 0)[0]
        if len(inkCols) == 0: continue

        absX1 = max(0, inkCols[0] - 6)
        absX2 = min(w, inkCols[-1] + 6)
        cropW = absX2 - absX1

        rawLineCrop = RawGrayscaleArray[absY1:absY2, absX1:absX2]
        scaleRatio = targetHeight / float(cropH)
        newWidth = min(maxWidth, int(cropW * scaleRatio))

        cropPIL = Image.fromarray(rawLineCrop)
        resizedPIL = cropPIL.resize((newWidth, targetHeight), Image.Resampling.LANCZOS)

        paddedCanvas = Image.new('L', (maxWidth, targetHeight), color=255)
        paddedCanvas.paste(resizedPIL, (0, 0))

        # Z-Score Normalization
        arr = np.array(paddedCanvas, dtype=np.float32)
        normArr = (arr - np.mean(arr)) / (np.std(arr) + 1e-8)
        LinePatches.append(normArr)

    return LinePatches


# =====================================================================
# 2. PYTORCH DATASET AND COLLATOR
# =====================================================================

CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,-' \"();:!"
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARSET)}  # 0 = CTC Blank
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}

class IAMAuthorLineDataset(Dataset):
    def __init__(self, root_dir, target_author="150"):
        self.samples = []
        
        # --- FILTER FOR TARGET AUTHOR FOLDER ("150") ---
        if target_author is not None:
            target_path = os.path.join(root_dir, target_author)
            if os.path.exists(target_path):
                self.author_folders = [target_author]
            else:
                raise ValueError(f"Could not find target author folder '{target_author}' inside '{root_dir}'")
        else:
            self.author_folders = sorted([
                f for f in os.listdir(root_dir) 
                if os.path.isdir(os.path.join(root_dir, f))
            ])
        
        self.author_to_idx = {author_id: idx for idx, author_id in enumerate(self.author_folders)}
        print(f"[Dataset] Loading data for Author(s): {self.author_folders}")

        total_pages = 0
        total_lines = 0

        for author_id in self.author_folders:
            author_path = os.path.join(root_dir, author_id)
            image_files = sorted(glob.glob(os.path.join(author_path, "*.png")))
            author_label = self.author_to_idx[author_id]
            
            for img_path in image_files:
                base_name = os.path.splitext(img_path)[0]
                
                # Check for either filename_labels.txt or filename.txt
                label_path = base_name + "_labels.txt"
                if not os.path.exists(label_path):
                    label_path = base_name + ".txt"

                if not os.path.exists(label_path):
                    print(f"  [Warning] Skipping '{os.path.basename(img_path)}' - No matching .txt label file found.")
                    continue

                with open(label_path, 'r', encoding='utf-8') as f:
                    text_lines = [line.strip() for line in f.readlines() if line.strip()]

                # Extract 32x1024 line patches using 1D pipeline
                line_patches = ExtractLinePatchesFromScan(img_path)
                total_pages += 1

                # Align image crops 1-to-1 with label file lines
                min_len = min(len(line_patches), len(text_lines))
                filename = os.path.basename(img_path)
                print(f"  -> Page {filename}: Extracted {len(line_patches)} line crops | Matched {min_len} text labels")

                for i in range(min_len):
                    patch = line_patches[i]
                    target_str = text_lines[i]
                    text_encoded = [CHAR_TO_IDX[c] for c in target_str if c in CHAR_TO_IDX]

                    if len(text_encoded) > 0:
                        self.samples.append((patch, author_label, text_encoded))
                        total_lines += 1

        print(f"\n[Dataset Summary] Processed {total_pages} pages -> Created {total_lines} 32x1024 line training samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch, author_label, text_encoded = self.samples[idx]
        
        patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)  # (1, 32, 1024)
        author_tensor = torch.tensor(author_label, dtype=torch.long)
        text_tensor = torch.tensor(text_encoded, dtype=torch.long)
        
        return patch_tensor, author_tensor, text_tensor


def collate_fn(batch):
    patches, authors, texts = zip(*batch)
    
    patches = torch.stack(patches, 0)
    authors = torch.stack(authors, 0)
    
    target_lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    targets = torch.cat(texts)
    
    return patches, authors, targets, target_lengths


# =====================================================================
# 3. MULTI-TASK NEURAL NETWORK ARCHITECTURE
# =====================================================================

class MultiTaskHandwritingCRNN(nn.Module):
    def __init__(self, num_authors, num_classes):
        super(MultiTaskHandwritingCRNN, self).__init__()
        
        # Shared CNN Feature Extractor (Input: 32x1024 -> Output width: 256 time steps)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),        # (32, 16, 512)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),        # (64, 8, 256)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),      # (128, 4, 256)
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),      # (256, 2, 256)
            
            nn.Conv2d(256, 256, kernel_size=(2, 1)), # (256, 1, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Branch 1: Author Identification Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.author_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_authors)
        )
        
        # Branch 2: Text Recognition CRNN Head
        self.bilstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, bidirectional=True, batch_first=False)
        self.text_classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        
        # Author Branch
        author_pooled = self.global_pool(features).squeeze(-1).squeeze(-1)
        author_logits = self.author_classifier(author_pooled)
        
        # Text Branch
        text_seq = features.squeeze(2).permute(2, 0, 1)
        lstm_out, _ = self.bilstm(text_seq)
        text_logits = self.text_classifier(lstm_out)
        text_log_probs = nn.functional.log_softmax(text_logits, dim=2)
        
        return author_logits, text_log_probs


# =====================================================================
# 4. TRAINING LOOP
# =====================================================================

def train_model(data_dir, target_author="150", num_epochs=30, batch_size=8, learning_rate=0.0005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Training on: {device}")

    dataset = IAMAuthorLineDataset(root_dir=data_dir, target_author=target_author)
    if len(dataset) == 0:
        print("Error: Dataset empty. Make sure image and .txt files are present.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    num_authors = len(dataset.author_folders)
    num_classes = len(CHARSET) + 1  # +1 for CTC Blank Token (index 0)

    model = MultiTaskHandwritingCRNN(num_authors=num_authors, num_classes=num_classes).to(device)

    criterion_author = nn.CrossEntropyLoss()
    criterion_text = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\nStarting Training on 32x1024 Handwritten Line Crops...")
    print("=" * 75)

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_auth_loss, total_text_loss = 0.0, 0.0, 0.0

        for patches, authors, targets, target_lengths in dataloader:
            patches, authors, targets = patches.to(device), authors.to(device), targets.to(device)

            optimizer.zero_grad()
            author_logits, text_log_probs = model(patches)

            # --- SINGLE AUTHOR SAFEGUARD ---
            if num_authors > 1:
                loss_author = criterion_author(author_logits, authors)
            else:
                loss_author = torch.tensor(0.0, device=device)

            # Loss 2: Text Recognition (Time dimension = 256)
            input_lengths = torch.full(
                size=(patches.size(0),), 
                fill_value=text_log_probs.size(0), 
                dtype=torch.long
            )
            loss_text = criterion_text(text_log_probs, targets, input_lengths, target_lengths)

            joint_loss = loss_author + loss_text
            joint_loss.backward()
            optimizer.step()

            total_loss += joint_loss.item()
            total_auth_loss += loss_author.item()
            total_text_loss += loss_text.item()

        avg_loss = total_loss / len(dataloader)
        avg_auth = total_auth_loss / len(dataloader)
        avg_text = total_text_loss / len(dataloader)

        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] | "
              f"Total Loss: {avg_loss:.4f} | "
              f"Author Loss: {avg_auth:.4f} | "
              f"Text CTC Loss: {avg_text:.4f}")

    # Save weights
    cnn_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(cnn_dir, "handwriting_line_multitask_weights.pth")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'author_mapping': dataset.author_to_idx,
        'charset': CHARSET
    }, weights_path)
    
    print("=" * 75)
    print(f"Training complete! Weights saved to:\n  {weights_path}")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Data", "Datasets", "IAMpages10"))

    if os.path.exists(DATASET_DIR):
        train_model(data_dir=DATASET_DIR, target_author="150", num_epochs=30, batch_size=8, learning_rate=0.0005)
    else:
        print(f"Error: Dataset path '{DATASET_DIR}' does not exist.")