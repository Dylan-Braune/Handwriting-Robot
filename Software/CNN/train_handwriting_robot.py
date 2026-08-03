import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytesseract

# Point PyTesseract to your installed executable path on Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# =====================================================================
# 1. ADVANCED WORD EXTRACTION & ALIGNMENT PIPELINE
# =====================================================================

def extract_printed_ground_truth(img):
    """
    Crops the top printed box (~10% to 24% height) and uses PyTesseract
    to extract the expected ground truth words.
    """
    h, w = img.shape
    printed_region = img[int(h * 0.10):int(h * 0.24), :]
    
    # Run OCR on printed text box using PSM 6 (Assume single uniform block of text)
    raw_text = pytesseract.image_to_string(printed_region, config='--psm 6')
    
    # Clean up string into list of valid words
    clean_text = raw_text.replace('\n', ' ').strip()
    words = [w.strip() for w in clean_text.split() if w.strip()]
    return words

def extract_handwritten_word_blocks(img, target_shape=(32, 128)):
    """
    1. Crops middle handwritten region (24% to 80% height), ignoring bottom signature.
    2. Applies horizontal Morphological Dilation to merge characters into whole words.
    3. Clusters bounding boxes into true top-to-bottom, left-to-right reading order.
    """
    h, w = img.shape
    handwritten_region = img[int(h * 0.24):int(h * 0.80), :]

    # Binarize (Otsu Thresholding)
    _, binary = cv2.threshold(handwritten_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological Horizontal Dilation: Merge letters into words!
    # A 15x3 rectangular kernel connects character gaps within words
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Find word contours on dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Filter noise and tiny artifacts
        if bw > 12 and bh > 12:
            raw_boxes.append((x, y, bw, bh))

    if not raw_boxes:
        return []

    # Sort boxes into Line-by-Line Reading Order
    raw_boxes.sort(key=lambda b: b[1]) # Sort vertically first
    
    sorted_boxes = []
    current_line = [raw_boxes[0]]
    
    for box in raw_boxes[1:]:
        prev_y = current_line[-1][1]
        prev_h = current_line[-1][3]
        
        # If Y-coordinate is within line height threshold, add to current line
        if abs(box[1] - prev_y) < max(20, prev_h * 0.5):
            current_line.append(box)
        else:
            # Sort current line left-to-right (X-axis) and commit
            current_line.sort(key=lambda b: b[0])
            sorted_boxes.extend(current_line)
            current_line = [box]
            
    if current_line:
        current_line.sort(key=lambda b: b[0])
        sorted_boxes.extend(current_line)

    # Extract cropped & normalized 32x128 word patches from ORIGINAL handwritten region
    blocks = []
    for (x, y, bw, bh) in sorted_boxes:
        crop = handwritten_region[y:y+bh, x:x+bw]
        resized = cv2.resize(crop, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
        
        # Z-score normalization
        normalized = (resized.astype(np.float32) - np.mean(resized)) / (np.std(resized) + 1e-8)
        blocks.append(normalized)

    return blocks

# =====================================================================
# 2. PYTORCH DATASET AND DATA LOADER
# =====================================================================

CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,-' "
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARSET)}  # 0 = CTC Blank
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}

class IAMAuthorTextDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.author_folders = sorted([
            f for f in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, f))
        ])
        
        self.author_to_idx = {author_id: idx for idx, author_id in enumerate(self.author_folders)}
        print(f"[Dataset] Found {len(self.author_folders)} Authors: {self.author_folders}")

        total_pages = 0
        aligned_pairs = 0

        for author_id in self.author_folders:
            author_path = os.path.join(root_dir, author_id)
            image_files = glob.glob(os.path.join(author_path, "*.png"))
            author_label = self.author_to_idx[author_id]
            
            for img_path in image_files:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                total_pages += 1
                filename = os.path.basename(img_path)

                # 1. Read ground truth word sequence from top printed block
                expected_words = extract_printed_ground_truth(img)
                
                # 2. Extract morphologically dilated word blocks in reading order
                word_blocks = extract_handwritten_word_blocks(img)
                
                # Align blocks 1-to-1 with expected words
                min_len = min(len(word_blocks), len(expected_words))
                
                print(f"  -> Page {filename}: Extracted {len(word_blocks)} boxes | Found {len(expected_words)} OCR words (Aligned {min_len} pairs)")

                for i in range(min_len):
                    block = word_blocks[i]
                    target_word = expected_words[i]
                    
                    text_encoded = [CHAR_TO_IDX[c] for c in target_word if c in CHAR_TO_IDX]
                    if len(text_encoded) > 0:
                        self.samples.append((block, author_label, text_encoded))
                        aligned_pairs += 1

        print(f"\n[Dataset Summary] Processed {total_pages} pages -> Successfully created {aligned_pairs} matched word patches.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        block, author_label, text_encoded = self.samples[idx]
        
        block_tensor = torch.tensor(block, dtype=torch.float32).unsqueeze(0)  # (1, 32, 128)
        author_tensor = torch.tensor(author_label, dtype=torch.long)
        text_tensor = torch.tensor(text_encoded, dtype=torch.long)
        
        return block_tensor, author_tensor, text_tensor

def collate_fn(batch):
    blocks, authors, texts = zip(*batch)
    
    blocks = torch.stack(blocks, 0)
    authors = torch.stack(authors, 0)
    
    target_lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    targets = torch.cat(texts)
    
    return blocks, authors, targets, target_lengths

# =====================================================================
# 3. MULTI-TASK NEURAL NETWORK ARCHITECTURE
# =====================================================================

class MultiTaskHandwritingCRNN(nn.Module):
    def __init__(self, num_authors, num_classes):
        super(MultiTaskHandwritingCRNN, self).__init__()
        
        # Shared CNN Feature Extractor
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (32, 16, 64)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (64, 8, 32)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)), # (128, 4, 32)
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)), # (256, 2, 32)
            
            nn.Conv2d(256, 256, kernel_size=(2, 1)), # (256, 1, 32)
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

def train_model(data_dir, num_epochs=25, batch_size=16, learning_rate=0.0005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Training on: {device}")

    dataset = IAMAuthorTextDataset(root_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    num_authors = len(dataset.author_folders)
    num_classes = len(CHARSET) + 1

    model = MultiTaskHandwritingCRNN(num_authors=num_authors, num_classes=num_classes).to(device)

    criterion_author = nn.CrossEntropyLoss()
    criterion_text = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\nStarting Model Training with Aligned Word Blocks...")
    print("=" * 70)

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_auth_loss, total_text_loss = 0.0, 0.0, 0.0

        for blocks, authors, targets, target_lengths in dataloader:
            blocks, authors, targets = blocks.to(device), authors.to(device), targets.to(device)

            optimizer.zero_grad()
            author_logits, text_log_probs = model(blocks)

            loss_author = criterion_author(author_logits, authors)

            input_lengths = torch.full(
                size=(blocks.size(0),), 
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
    weights_path = os.path.join(cnn_dir, "handwriting_robot_multitask_weights.pth")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'author_mapping': dataset.author_to_idx,
        'charset': CHARSET
    }, weights_path)
    
    print("=" * 70)
    print(f"Training complete! Weights saved to:\n  {weights_path}")

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Data", "Datasets", "IAMpages10"))

    if os.path.exists(DATASET_DIR):
        train_model(data_dir=DATASET_DIR, num_epochs=25, batch_size=16, learning_rate=0.0005)
    else:
        print(f"Error: Path '{DATASET_DIR}' does not exist.")