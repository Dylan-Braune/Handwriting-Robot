import os
import cv2
import numpy as np
import pytesseract
import re

# Point PyTesseract to your installed executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# =====================================================================
# 1. SYMMETRIC OCR TOKENIZATION
# =====================================================================

def tokenize_ocr_text(raw_text, punctuation_mode="attached"):
    """
    Tokenizes raw text into discrete tokens with strict rules for punctuation.
    
    Modes:
      - 'attached': Trailing punctuation belongs to the word ("hello," -> ['hello,'])
      - 'isolated': Punctuation gets its own token ("hello," -> ['hello', ','])
    """
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    tokenized_lines = []

    for line in lines:
        tokens = []
        if punctuation_mode == "attached":
            # Whitespace split preserves attached punctuation
            tokens = [t.strip() for t in line.split() if t.strip()]
        elif punctuation_mode == "isolated":
            # Regex splits words and individual punctuation marks into separate tokens
            tokens = re.findall(r"\w+|[^\w\s]", line)
        
        if tokens:
            tokenized_lines.append(tokens)
            
    return tokenized_lines

# =====================================================================
# 2. ADAPTIVE BOUNDING BOX EXTRACTION & FORCE MATCHING
# =====================================================================

def merge_two_boxes(b1, b2):
    """Merges two bounding boxes (x, y, w, h) into a single bounding box."""
    x = min(b1[0], b2[0])
    y = min(b1[1], b2[1])
    w = max(b1[0] + b1[2], b2[0] + b2[2]) - x
    h = max(b1[1] + b1[3], b2[1] + b2[3]) - y
    return (x, y, w, h)

def force_match_boxes_to_tokens(raw_boxes, target_count):
    """
    Adjusts visual bounding boxes so that len(boxes) == target_count.
    
    - If raw_boxes > target_count: Merges adjacent boxes with smallest horizontal gap.
    - If raw_boxes < target_count: Splits the widest box horizontally into sub-boxes.
    """
    boxes = list(raw_boxes)
    if not boxes or target_count <= 0:
        return boxes
    
    # 1. Reduce count if we have too many boxes (e.g., dots on i's, split strokes)
    while len(boxes) > target_count:
        if len(boxes) <= 1:
            break
            
        min_gap = float('inf')
        merge_idx = 0
        
        for i in range(len(boxes) - 1):
            # Calculate horizontal gap between box i and box i+1
            b1, b2 = boxes[i], boxes[i+1]
            gap = b2[0] - (b1[0] + b1[2])
            if gap < min_gap:
                min_gap = gap
                merge_idx = i
                
        # Merge closest adjacent pair
        merged = merge_two_boxes(boxes[merge_idx], boxes[merge_idx + 1])
        boxes[merge_idx] = merged
        boxes.pop(merge_idx + 1)

    # 2. Increase count if we have too few boxes (e.g., merged cursive words)
    while len(boxes) < target_count:
        if len(boxes) == 0:
            break
            
        # Find index of the widest box
        widest_idx = max(range(len(boxes)), key=lambda i: boxes[i][2])
        x, y, w, h = boxes[widest_idx]
        
        if w < 2:  # Prevent infinite loop on 1-pixel boxes
            break
            
        # Split in half horizontally
        w1 = w // 2
        w2 = w - w1
        b1 = (x, y, w1, h)
        b2 = (x + w1, y, w2, h)
        
        boxes[widest_idx] = b1
        boxes.insert(widest_idx + 1, b2)

    return boxes

def extract_and_align_page(img, punctuation_mode="attached", target_patch_size=(32, 128)):
    """
    Dynamically segments handwriting into word/punctuation patches that match
    the exact OCR token sequence.
    """
    if img is None:
        print("[Error] Input image is None.")
        return [], None

    h_img, w_img = img.shape[:2]

    # --- Step A: Adaptive Thresholding & Scale Calculation ---
    # Crop middle handwritten region dynamically
    offset_y = int(h_img * 0.22)
    hw_region = img[offset_y:int(h_img * 0.82), :]

    # Binarize
    _, binary = cv2.threshold(hw_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Calculate median character height to make thresholds scale-invariant
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(c)[3] for c in contours if cv2.boundingRect(c)[3] > 5]
    
    if not heights:
        print("[Warning] No contours found in handwritten region.")
        return [], None
    
    median_char_h = int(np.median(heights))
    
    # --- Step B: OCR Ground Truth Extraction ---
    printed_region = img[int(h_img * 0.08):int(h_img * 0.22), :]
    raw_ocr_text = pytesseract.image_to_string(printed_region, config='--psm 6')
    ocr_lines = tokenize_ocr_text(raw_ocr_text, punctuation_mode=punctuation_mode)

    if not ocr_lines:
        print("[Warning] OCR could not extract target text.")
        return [], None

    # --- Step C: Line Segmentation (Adaptive Horizontal Dilation) ---
    line_kernel_w = max(40, median_char_h * 3)
    line_kernel_h = max(3, median_char_h // 4)
    
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (line_kernel_w, line_kernel_h))
    dilated_lines = cv2.dilate(binary, kernel_line, iterations=2)
    
    line_cnts, _ = cv2.findContours(dilated_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    visual_line_boxes = []
    for cnt in line_cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > median_char_h * 2 and h > median_char_h * 0.5:
            visual_line_boxes.append((x, y, w, h))

    # Sort lines top-to-bottom
    visual_line_boxes.sort(key=lambda b: b[1])

    # --- Step D: Token Extraction & Force Matching per Line ---
    min_lines = min(len(visual_line_boxes), len(ocr_lines))
    extracted_samples = []
    
    # Visualization Canvas
    if len(img.shape) == 2:
        debug_canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        debug_canvas = img.copy()

    for line_idx in range(min_lines):
        lx, ly, lw, lh = visual_line_boxes[line_idx]
        target_tokens = ocr_lines[line_idx]
        
        # Crop binary line mask
        line_binary = binary[ly:ly+lh, lx:lx+lw]
        
        # Word/Punctuation dilation kernel
        if punctuation_mode == "attached":
            word_kernel_w = max(8, int(median_char_h * 0.5))
        else:
            word_kernel_w = max(4, int(median_char_h * 0.25))

        kernel_word = cv2.getStructuringElement(cv2.MORPH_RECT, (word_kernel_w, 3))
        dilated_words = cv2.dilate(line_binary, kernel_word, iterations=1)

        word_cnts, _ = cv2.findContours(dilated_words, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_word_boxes = []
        for cnt in word_cnts:
            wx, wy, ww, wh = cv2.boundingRect(cnt)
            if ww > 4 and wh > 4:
                # Convert back to full image coordinates
                raw_word_boxes.append((lx + wx, offset_y + ly + wy, ww, wh))

        if not raw_word_boxes:
            continue

        # Sort raw boxes left-to-right
        raw_word_boxes.sort(key=lambda b: b[0])

        # FORCE MATCHING: Guarantee len(matched_boxes) == len(target_tokens)
        matched_boxes = force_match_boxes_to_tokens(raw_word_boxes, target_count=len(target_tokens))

        # --- Step E: Extract, Resize Patches & Draw Debug Boxes ---
        for i, box in enumerate(matched_boxes):
            bx, by, bw, bh = box
            token_text = target_tokens[i]

            # Safely clamp crop boundaries to image dimensions
            by_start = max(0, by)
            by_end = min(h_img, by + bh)
            bx_start = max(0, bx)
            bx_end = min(w_img, bx + bw)

            crop = img[by_start:by_end, bx_start:bx_end]
            
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue

            # Scale to neural network feature extractor size (32, 128)
            resized_patch = cv2.resize(crop, (target_patch_size[1], target_patch_size[0]), interpolation=cv2.INTER_AREA)

            extracted_samples.append({
                'patch': resized_patch,
                'token': token_text,
                'box': box
            })

            # Draw Green Bounding Box on Debug Canvas
            cv2.rectangle(debug_canvas, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(debug_canvas, token_text, (bx, max(15, by - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return extracted_samples, debug_canvas

# =====================================================================
# 3. DIAGNOSTIC EXECUTION & VISUALIZATION
# =====================================================================

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAMPLE_IMAGE_PATH = os.path.abspath(os.path.join(
        SCRIPT_DIR, "..", "..", "Data", "Datasets", "IAMpages10", "150", "c03-000a.png"
    ))

    output_dir = os.path.join(SCRIPT_DIR, "alignment_debug_output")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(SAMPLE_IMAGE_PATH):
        print(f"Error: Could not find test image at '{SAMPLE_IMAGE_PATH}'")
    else:
        print(f"Testing extraction pipeline on: {os.path.basename(SAMPLE_IMAGE_PATH)}")
        img = cv2.imread(SAMPLE_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

        samples, debug_img = extract_and_align_page(img, punctuation_mode="attached")

        if debug_img is not None and len(samples) > 0:
            print(f"\n[Success] Extracted {len(samples)} aligned word patches!")
            
            # Save labeled debug image showing bounding boxes overlaid with OCR tokens
            debug_path = os.path.join(output_dir, "aligned_page_preview.png")
            cv2.imwrite(debug_path, debug_img)
            print(f" Saved visual bounding box verification to:\n  {debug_path}")

            # Save first 15 cropped patches to inspect individual word/punctuation boxes
            patches_dir = os.path.join(output_dir, "sample_patches")
            os.makedirs(patches_dir, exist_ok=True)
            
            for idx, sample in enumerate(samples[:15]):
                patch_filename = f"patch_{idx:02d}_{sample['token']}.png"
                clean_filename = re.sub(r'[^\w\-_\. ]', '_', patch_filename)
                cv2.imwrite(os.path.join(patches_dir, clean_filename), sample['patch'])

            print(f" Saved 15 individual resized cropped word patches to:\n  {patches_dir}\n")
            
            print("Sample align check:")
            for idx, sample in enumerate(samples[:10]):
                print(f"  Box {idx+1:02d}: Ground Truth Token = '{sample['token']}' | Tensor Shape = {sample['patch'].shape}")
        else:
            print("[Failed] Could not extract samples. Check warnings above.")