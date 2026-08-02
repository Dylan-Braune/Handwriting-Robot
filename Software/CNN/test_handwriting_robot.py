import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import pytesseract
from collections import Counter

# --- WINDOWS ONLY: Point PyTesseract to your installed executable ---
# If you installed Tesseract to the default path, leave this. Otherwise, update it.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# =====================================================================
# 1. NEURAL NETWORK ARCHITECTURE (Must match the training script exactly)
# =====================================================================
class MultiTaskHandwritingCRNN(nn.Module):
    def __init__(self, num_authors, num_classes):
        super(MultiTaskHandwritingCRNN, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            
            nn.Conv2d(256, 256, kernel_size=(2, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.author_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_authors)
        )
        
        self.bilstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, bidirectional=True, batch_first=False)
        self.text_classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        author_pooled = self.global_pool(features).squeeze(-1).squeeze(-1)
        author_logits = self.author_classifier(author_pooled)
        
        text_seq = features.squeeze(2).permute(2, 0, 1)
        lstm_out, _ = self.bilstm(text_seq)
        text_logits = self.text_classifier(lstm_out)
        text_log_probs = nn.functional.log_softmax(text_logits, dim=2)
        
        return author_logits, text_log_probs

# =====================================================================
# 2. IMAGE PREPROCESSING & INFERENCE LOGIC
# =====================================================================

def process_page_for_inference(image_path, target_shape=(32, 128)):
    """
    Splits the page into the printed expected text (Top 35%) 
    and the handwritten extracted blocks (Bottom 65%).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, []

    h, w = img.shape
    split_point = int(h * 0.35)
    
    # 1. Get printed text using PyTesseract
    printed_region = img[:split_point, :]
    expected_text = pytesseract.image_to_string(printed_region).strip().replace('\n', ' ')

    # 2. Extract handwritten blocks
    handwritten_region = img[split_point:, :]
    _, binary = cv2.threshold(handwritten_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours top-to-bottom, left-to-right to read sequentially
    # (Simple bounding box sort for basic sequential reading)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = [bb for bb in bounding_boxes if bb[2] > 15 and bb[3] > 10]
    bounding_boxes.sort(key=lambda b: (b[1] // 50, b[0])) # Sort by Y (grouped by 50px lines), then X

    blocks = []
    for (x, y, bw, bh) in bounding_boxes:
        crop = handwritten_region[y:y+bh, x:x+bw]
        resized = cv2.resize(crop, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
        normalized = (resized.astype(np.float32) - np.mean(resized)) / (np.std(resized) + 1e-8)
        
        # Convert to tensor immediately
        block_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        blocks.append(block_tensor)

    return expected_text, blocks

def decode_ctc_predictions(text_log_probs, charset):
    """Greedy CTC Decoder: Removes duplicates and blank tokens."""
    # text_log_probs shape: (Sequence Length, Batch Size, Num Classes)
    pred_indices = torch.argmax(text_log_probs, dim=2).squeeze(1)
    
    decoded_text = []
    prev_idx = -1
    for idx in pred_indices:
        idx = idx.item()
        # 0 is the CTC blank token
        if idx != 0 and idx != prev_idx:
            # Map index back to character (1-based index to 0-based charset index)
            decoded_text.append(charset[idx - 1])
        prev_idx = idx
        
    return "".join(decoded_text)

# =====================================================================
# 3. EVALUATION LOOP
# =====================================================================

def evaluate_model(dataset_dir, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")

    # Load Weights and Metadata
    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' not found!")
        return

    checkpoint = torch.load(weights_path, map_location=device)
    author_mapping = checkpoint['author_mapping']
    charset = checkpoint['charset']
    
    # Reverse mapping to get author ID strings (e.g., 0 -> "150")
    idx_to_author = {v: k for k, v in author_mapping.items()}
    num_authors = len(author_mapping)
    num_classes = len(charset) + 1

    # Initialize model
    model = MultiTaskHandwritingCRNN(num_authors, num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set to evaluation mode!

    # Find all author folders
    author_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]

    print("=" * 70)
    print("STARTING FULL PAGE EVALUATION")
    print("=" * 70)

    with torch.no_grad(): # No gradients needed for testing
        for author_id in author_folders:
            author_path = os.path.join(dataset_dir, author_id)
            image_files = glob.glob(os.path.join(author_path, "*.png"))
            
            for img_path in image_files:
                filename = os.path.basename(img_path)
                
                # 1. Process Page
                expected_text, blocks = process_page_for_inference(img_path)
                
                if not blocks:
                    continue

                page_predicted_authors = []
                page_decoded_text = []

                # 2. Run Inference on each block/word
                for block_tensor in blocks:
                    block_tensor = block_tensor.to(device)
                    
                    author_logits, text_log_probs = model(block_tensor)
                    
                    # Author Prediction (get the index with the highest probability)
                    pred_author_idx = torch.argmax(author_logits, dim=1).item()
                    page_predicted_authors.append(idx_to_author[pred_author_idx])
                    
                    # Text Decoding
                    decoded_word = decode_ctc_predictions(text_log_probs, charset)
                    page_decoded_text.append(decoded_word)

                # 3. Aggregate Page Results
                # Final author prediction is the majority vote across all words on the page
                most_common_author = Counter(page_predicted_authors).most_common(1)[0][0]
                
                # Combine decoded words into a full sentence
                full_predicted_text = " ".join(page_decoded_text)

                # 4. Print Output
                print(f"\n--- File: {filename} ---")
                print(f"True Author ID     : {author_id}")
                print(f"Predicted Author ID: {most_common_author} " + 
                      ("✅" if author_id == most_common_author else "❌"))
                print(f"Expected Text (OCR): {expected_text[:80]}...") # Truncated for terminal readability
                print(f"Decoded CRNN Text  : {full_predicted_text[:80]}...")

if __name__ == "__main__":
    # Setup Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Data", "Datasets", "IAMpages10"))
    WEIGHTS_PATH = os.path.join(SCRIPT_DIR, "handwriting_robot_multitask_weights.pth")

    evaluate_model(DATASET_DIR, WEIGHTS_PATH)