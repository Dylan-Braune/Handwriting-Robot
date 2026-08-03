import os
import cv2
import numpy as np
import pytesseract
import re

# Point PyTesseract to your installed executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# =====================================================================
# 1. FORM REGION ISOLATION VIA HORIZONTAL RULE DETECTION
# =====================================================================

def detect_handwriting_bounds(img):
    """
    Finds the two horizontal black rule lines on the IAM page to dynamically
    isolate the middle handwritten region, completely ignoring the top prompt
    box and the bottom signature block.
    """
    h, w = img.shape
    
    # Inverted binary for line detection
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal morphological kernel (looking for wide horizontal lines)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    # Find line contours
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    y_lines = []
    for cnt in contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
        if w_c > w * 0.4:  # Must span at least 40% of page width
            y_lines.append(y_c + h_c // 2)

    y_lines.sort()

    # Fallback to defaults (~22% top, ~82% bottom) if lines aren't clearly detected
    y_top = y_lines[0] if len(y_lines) >= 1 else int(h * 0.22)
    y_bottom = y_lines[-1] if len(y_lines) >= 2 else int(h * 0.82)

    # Ensure valid vertical order
    if y_bottom <= y_top + 100:
        y_top, y_bottom = int(h * 0.22), int(h * 0.82)

    return y_top, y_bottom

# =====================================================================
# 2. OCR GROUND TRUTH TOKENIZATION
# =====================================================================

def tokenize_printed_text(printed_crop, punctuation_mode="attached"):
    """
    Reads the digital text from the top box and splits it into line-by-line tokens.
    """
    raw_text = pytesseract.image_to_string(printed_crop, config='--psm 6')
    raw_lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    
    tokenized_lines = []
    for line in raw_lines:
        if punctuation_mode == "attached":
            tokens = [t.strip() for t in line.split() if t.strip()]
        else:
            tokens = re.findall(r"\w+|[^\w\s]", line)
        if tokens:
            tokenized_lines.append(tokens)
            
    return tokenized_lines

# =====================================================================
# 3. BOUNDING BOX FORCE MATCHING (1-TO-1 TOKEN ALIGNMENT)
# =====================================================================

def force_match_boxes(raw_boxes, target_count):
    """
    Iteratively merges or splits visual bounding boxes so that
    len(boxes) == target_count on every line.
    """
    boxes = list(raw_boxes)
    if not boxes or target_count <= 0:
        return boxes

    # Merging extra boxes (e.g., dots on 'i', unjoined strokes)
    while len(boxes) > target_count:
        if len(boxes) <= 1:
            break
        min_gap = float('inf')
        merge_idx = 0
        for i in range(len(boxes) - 1):
            gap = boxes[i+1][0] - (boxes[i][0] + boxes[i][2])
            if gap < min_gap:
                min_gap = gap
                merge_idx = i
        
        # Merge closest pair
        b1, b2 = boxes[merge_idx], boxes[merge_idx + 1]
        x = min(b1[0], b2[0])
        y = min(b1[1], b2[1])
        w = max(b1[0] + b1[2], b2[0] + b2[2]) - x
        h = max(b1[1] + b1[3], b2[1] + b2[3]) - y
        
        boxes[merge_idx] = (x, y, w, h)
        boxes.pop(merge_idx + 1)

    # Splitting merged cursive words
    while len(boxes) < target_count:
        if len(boxes) == 0:
            break
        widest_idx = max(range(len(boxes)), key=lambda i: boxes[i][2])
        x, y, w, h = boxes[widest_idx]
        if w < 2:
            break
        w1, w2 = w // 2, w - (w // 2)
        boxes[widest_idx] = (x, y, w1, h)
        boxes.insert(widest_idx + 1, (x + w1, y, w2, h))

    return boxes

# =====================================================================
# 4. PREPROCESSING & RAW CROP EXTRACTION
# =====================================================================

def process_and_segment_handwriting(img):
    """
    Full pipeline performing region crop, binary thresholding, and word separation.
    Saves raw unscaled image patches exactly as they appear on the page.
    """
    h_img, w_img = img.shape[:2]

    # --- Step 1: Detect Horizontal Separator Lines ---
    y_top, y_bottom = detect_handwriting_bounds(img)
    
    # --- Step 2: Extract Ground Truth Tokens from Top Box ---
    printed_region = img[int(h_img * 0.08):y_top, :]
    ocr_lines = tokenize_printed_text(printed_region, punctuation_mode="attached")

    if not ocr_lines:
        print("[Warning] Could not read printed text from top box.")
        return [], None

    # --- Step 3: Preprocess Middle Handwritten Region ---
    hw_region = img[y_top:y_bottom, :]

    # Inverted Binary Thresholding
    _, binary = cv2.threshold(hw_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Compute median character height for scale invariance
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(c)[3] for c in cnts if cv2.boundingRect(c)[3] > 5]
    if not heights:
        return [], None
    median_h = int(np.median(heights))

    # --- Step 4: Line Separation ---
    line_k_w = max(40, median_h * 3)
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (line_k_w, max(3, median_h // 4)))
    dilated_lines = cv2.dilate(binary, kernel_line, iterations=2)
    
    line_cnts, _ = cv2.findContours(dilated_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_boxes = [cv2.boundingRect(c) for c in line_cnts if cv2.boundingRect(c)[2] > median_h * 2]
    line_boxes.sort(key=lambda b: b[1])  # Top-to-bottom

    # --- Step 5: Word Separation & Token Alignment ---
    min_lines = min(len(line_boxes), len(ocr_lines))
    extracted_samples = []

    # Visual Debug Canvas
    debug_canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    
    # Draw gray boundary lines showing ignored regions
    cv2.line(debug_canvas, (0, y_top), (w_img, y_top), (255, 0, 0), 2)
    cv2.line(debug_canvas, (0, y_bottom), (w_img, y_bottom), (255, 0, 0), 2)

    for l_idx in range(min_lines):
        lx, ly, lw, lh = line_boxes[l_idx]
        target_tokens = ocr_lines[l_idx]

        line_binary = binary[ly:ly+lh, lx:lx+lw]
        word_k_w = max(6, int(median_h * 0.45))
        kernel_word = cv2.getStructuringElement(cv2.MORPH_RECT, (word_k_w, 3))
        dilated_words = cv2.dilate(line_binary, kernel_word, iterations=1)

        w_cnts, _ = cv2.findContours(dilated_words, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_w_boxes = []
        for c in w_cnts:
            wx, wy, ww, wh = cv2.boundingRect(c)
            if ww > 4 and wh > 4:
                raw_w_boxes.append((lx + wx, y_top + ly + wy, ww, wh))

        if not raw_w_boxes:
            continue

        raw_w_boxes.sort(key=lambda b: b[0])  # Left-to-right

        # Guarantee exact 1-to-1 match for tokens
        matched_boxes = force_match_boxes(raw_w_boxes, target_count=len(target_tokens))

        for i, box in enumerate(matched_boxes):
            bx, by, bw, bh = box
            token = target_tokens[i]

            # Crop raw grayscale image patch directly from the original image
            crop = img[max(0, by):min(h_img, by+bh), max(0, bx):min(w_img, bx+bw)]
            
            # Skip if the crop is empty
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue

            # Store only the raw unedited patch
            extracted_samples.append({
                'raw_patch': crop,
                'token': token,
                'box': box
            })

            # Draw Green Bounding Box ONLY over middle handwritten region
            cv2.rectangle(debug_canvas, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(debug_canvas, token, (bx, max(15, by - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    return extracted_samples, debug_canvas

# =====================================================================
# 5. EXECUTION
# =====================================================================

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAMPLE_IMAGE_PATH = os.path.abspath(os.path.join(
        SCRIPT_DIR, "..", "..", "Data", "Datasets", "IAMpages10", "150", "c03-000a.png"
    ))

    output_dir = os.path.join(SCRIPT_DIR, "handwriting_extraction_output")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(SAMPLE_IMAGE_PATH):
        print(f"Error: Test image not found at '{SAMPLE_IMAGE_PATH}'")
    else:
        print(f"Processing handwritten page: {os.path.basename(SAMPLE_IMAGE_PATH)}")
        img = cv2.imread(SAMPLE_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

        samples, debug_img = process_and_segment_handwriting(img)

        if debug_img is not None and len(samples) > 0:
            # Save labeled debug image
            debug_path = os.path.join(output_dir, "isolated_handwriting_preview.png")
            cv2.imwrite(debug_path, debug_img)
            
            # Save ALL raw word patches
            patches_dir = os.path.join(output_dir, "sample_patches")
            os.makedirs(patches_dir, exist_ok=True)
            
            # Removed the [:10] limit, this now iterates through every extracted patch
            for idx, s in enumerate(samples):
                clean_name = re.sub(r'[^\w\-_\. ]', '_', f"patch_{idx:02d}_{s['token']}.png")
                cv2.imwrite(os.path.join(patches_dir, clean_name), s['raw_patch'])

            print(f"\n[Success] Extracted {len(samples)} handwritten word patches!")
            print(f" Verified preview saved to:\n  {debug_path}")
            print(f" All {len(samples)} sample patches saved to:\n  {patches_dir}")
        else:
            print("[Failed] Extraction returned 0 samples. Check region detection.")