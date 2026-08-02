import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# 1. PREPROCESSING & TEXT SEPARATION PIPELINE
# Concepts derived from the literature study[cite: 1]
# =====================================================================

def preprocess_and_extract_blocks(image_path, target_shape=(32, 128)):
    """
    1. Loads the full IAM page scan in grayscale[cite: 1].
    2. Strips off the upper printed text block (~35% of page height).
    3. Converts lower handwritten region to binary[cite: 1].
    4. Extracts bounding-box blocks/words and resizes them to target_shape[cite: 1].
    5. Normalizes image intensity using Z-score scaling[cite: 1].
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    h, w = img.shape
    # Strip top ~35% containing printed text; focus on bottom handwritten section
    handwritten_region = img[int(h * 0.35):, :]

    # Binarization and Inversion (using Otsu thresholding)
    _, binary = cv2.threshold(handwritten_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find word/block contours using bounding box techniques[cite: 1]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blocks = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Filter noise and tiny artifact contours
        if bw > 15 and bh > 10:
            crop = handwritten_region[y:y+bh, x:x+bw]
            
            # Resize block to standard dimensions (32 x 128)
            resized = cv2.resize(crop, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
            
            # Z-Score Normalization to reduce unwanted bias[cite: 1]
            normalized = (resized.astype(np.float32) - np.mean(resized)) / (np.std(resized) + 1e-8)
            blocks.append(normalized)

    return blocks

# =====================================================================
# 2. PYTORCH DATASET AND DATA LOADER
# =====================================================================

# Character vocabulary dictionary for CRNN CTC Decoding
CHARSET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,-' "
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARSET)}  # 0 is reserved for CTC Blank token
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

        for author_id in self.author_folders:
            author_path = os.path.join(root_dir, author_id)
            image_files = glob.glob(os.path.join(author_path, "*.png"))
            
            for img_path in image_files:
                # Extract preprocessed blocks per page scan
                blocks = preprocess_and_extract_blocks(img_path)
                author_label = self.author_to_idx[author_id]
                
                for block in blocks:
                    # Generic dummy text target string for demo; replace with real transcripts if ascii files are used
                    text_target = "handwriting"  
                    text_encoded = [CHAR_TO_IDX[c] for c in text_target if c in CHAR_TO_IDX]
                    
                    self.samples.append((block, author_label, text_encoded))

        print(f"[Dataset] Successfully extracted {len(self.samples)} handwriting block patches.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        block, author_label, text_encoded = self.samples[idx]
        
        # Format image block for PyTorch input tensor: (Channel, Height, Width) -> (1, 32, 128)
        block_tensor = torch.tensor(block, dtype=torch.float32).unsqueeze(0)
        author_tensor = torch.tensor(author_label, dtype=torch.long)
        text_tensor = torch.tensor(text_encoded, dtype=torch.long)
        
        return block_tensor, author_tensor, text_tensor

def collate_fn(batch):
    """Custom collate function to handle variable sequence lengths for CTC Loss."""
    blocks, authors, texts = zip(*batch)
    
    blocks = torch.stack(blocks, 0)
    authors = torch.stack(authors, 0)
    
    target_lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    targets = torch.cat(texts)
    
    return blocks, authors, targets, target_lengths

# =====================================================================
# 3. JOINT MULTI-TASK NEURAL NETWORK ARCHITECTURE
# Incorporating CNN and RNN techniques from literature study[cite: 1]
# =====================================================================

class MultiTaskHandwritingCRNN(nn.Module):
    def __init__(self, num_authors, num_classes):
        super(MultiTaskHandwritingCRNN, self).__init__()
        
        # --- Shared CNN Feature Extractor ---
        # CNN layers used for feature extraction by sliding a kernel over the image[cite: 1]
        # Max-pooling is used to downsample feature maps and prevent overfitting[cite: 1]
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Shape: (32, 16, 64)
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Shape: (64, 8, 32)
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)), # Shape: (128, 4, 32)
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)), # Shape: (256, 2, 32)
            
            nn.Conv2d(256, 256, kernel_size=(2, 1)), # Valid Conv -> Shape: (256, 1, 32)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # --- Branch 1: Author Identification Head ---
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.author_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_authors)
        )
        
        # --- Branch 2: Text Recognition CRNN Head ---
        # Recurrent Neural Network sequence modeling to consider sequential data[cite: 1]
        self.bilstm = nn.LSTM(
            input_size=256, 
            hidden_size=128, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=False
        )
        self.text_classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # Feature Extraction
        features = self.backbone(x)  # Tensor: (B, 256, 1, W')
        
        # 1. Author Identification Head
        author_pooled = self.global_pool(features).squeeze(-1).squeeze(-1) # (B, 256)
        author_logits = self.author_classifier(author_pooled)             # (B, num_authors)
        
        # 2. Text Recognition Head
        text_seq = features.squeeze(2)          # (B, 256, W')
        text_seq = text_seq.permute(2, 0, 1)    # (W', B, 256) -> Sequence length first
        
        lstm_out, _ = self.bilstm(text_seq)     # (W', B, 256)
        text_logits = self.text_classifier(lstm_out) # (W', B, num_classes)
        text_log_probs = nn.functional.log_softmax(text_logits, dim=2)
        
        return author_logits, text_log_probs

# =====================================================================
# 4. TRAINING & VALIDATION PIPELINE
# =====================================================================

def train_model(data_dir, num_epochs=15, batch_size=16, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Training on: {device}")

    # Initialize Dataset and DataLoader
    dataset = IAMAuthorTextDataset(root_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    num_authors = len(dataset.author_folders)
    num_classes = len(CHARSET) + 1  # Includes CTC blank token at Index 0

    # Model instantiation
    model = MultiTaskHandwritingCRNN(num_authors=num_authors, num_classes=num_classes).to(device)

    # Loss Functions & Optimizer
    criterion_author = nn.CrossEntropyLoss()
    criterion_text = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\nStarting Model Training...")
    print("=" * 60)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_author_loss = 0.0
        total_text_loss = 0.0

        for batch_idx, (blocks, authors, targets, target_lengths) in enumerate(dataloader):
            blocks = blocks.to(device)
            authors = authors.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward Pass
            author_logits, text_log_probs = model(blocks)

            # Calculate Author Identification Loss
            loss_author = criterion_author(author_logits, authors)

            # Calculate Text Recognition CTC Loss
            input_lengths = torch.full(
                size=(blocks.size(0),), 
                fill_value=text_log_probs.size(0), 
                dtype=torch.long
            )
            loss_text = criterion_text(text_log_probs, targets, input_lengths, target_lengths)

            # Total Joint Multi-task Loss
            joint_loss = loss_author + loss_text

            joint_loss.backward()
            optimizer.step()

            total_loss += joint_loss.item()
            total_author_loss += loss_author.item()
            total_text_loss += loss_text.item()

        avg_loss = total_loss / len(dataloader)
        avg_auth = total_author_loss / len(dataloader)
        avg_text = total_text_loss / len(dataloader)

        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] | "
              f"Total Loss: {avg_loss:.4f} | "
              f"Author Loss: {avg_auth:.4f} | "
              f"Text CTC Loss: {avg_text:.4f}")

    # =====================================================================
    # 5. SAVE WEIGHTS & METADATA
    # =====================================================================
    # Automatically get the CNN folder path and save it exactly there
    cnn_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(cnn_dir, "handwriting_robot_multitask_weights.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'author_mapping': dataset.author_to_idx,
        'charset': CHARSET
    }, weights_path)
    
    print("=" * 60)
    print(f"Training completed successfully!")
    print(f"Model weights and training configuration saved to: {weights_path}")

if __name__ == "__main__":
    # 1. Get the directory of this current script (.../Handwriting-Robot/Software/CNN)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # 2. Go up 2 levels (..) to Handwriting-Robot, then down into Data/Datasets/IAMpages10
    DATASET_DIR = os.path.abspath(
        os.path.join(SCRIPT_DIR, "..", "..", "Data", "Datasets", "IAMpages10")
    )

    print(f"[Path Manager] Script location: {SCRIPT_DIR}")
    print(f"[Path Manager] Target dataset:  {DATASET_DIR}\n")

    # 3. Run training
    if os.path.exists(DATASET_DIR):
        train_model(
            data_dir=DATASET_DIR,
            num_epochs=15,
            batch_size=8,
            learning_rate=0.0005,
        )
    else:
        print(
            f"Error: Could not resolve path '{DATASET_DIR}'. Check your directory structure!"
        )