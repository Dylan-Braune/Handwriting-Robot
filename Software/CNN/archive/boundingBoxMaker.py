import os
import cv2
import numpy as np

def detect_handwriting_bounds(img):
    """Finds the top and bottom bounds of the handwritten section."""
    h, w = img.shape
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    y_lines = [cv2.boundingRect(cnt)[1] + cv2.boundingRect(cnt)[3] // 2 
               for cnt in contours if cv2.boundingRect(cnt)[2] > w * 0.4]
    y_lines.sort()

    y_top = y_lines[0] if len(y_lines) >= 1 else int(h * 0.22)
    y_bottom = y_lines[-1] if len(y_lines) >= 2 else int(h * 0.82)
    if y_bottom <= y_top + 100:
         y_top, y_bottom = int(h * 0.22), int(h * 0.82)

    return y_top, y_bottom

def preview_smart_segmentation(img):
    """
    Isolates lines, then intelligently merges characters while keeping 
    punctuation separate based on vertical alignment and baseline logic.
    """
    h_img, w_img = img.shape[:2]
    y_top, y_bottom = detect_handwriting_bounds(img)
    hw_region = img[y_top:y_bottom, :]

    _, binary = cv2.threshold(hw_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(c)[3] for c in cnts if cv2.boundingRect(c)[3] > 5]
    if not heights: return None
    median_h = int(np.median(heights))

    # --- Phase 3: Line Separation ---
    line_k_w = max(40, median_h * 3)
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (line_k_w, max(3, median_h // 4)))
    dilated_lines = cv2.dilate(binary, kernel_line, iterations=2)
    
    line_cnts, _ = cv2.findContours(dilated_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_boxes = [cv2.boundingRect(c) for c in line_cnts if cv2.boundingRect(c)[2] > median_h * 2]
    line_boxes.sort(key=lambda b: b[1])  

    debug_canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
    cv2.line(debug_canvas, (0, y_top), (w_img, y_top), (0, 0, 255), 2)
    cv2.line(debug_canvas, (0, y_bottom), (w_img, y_bottom), (0, 0, 255), 2)

    for lx, ly, lw, lh in line_boxes:
        cv2.rectangle(debug_canvas, (lx, y_top + ly), (lx + lw, y_top + ly + lh), (255, 0, 0), 2)

        # --- Phase 4: Smart Character Merging ---
        line_binary = binary[ly:ly+lh, lx:lx+lw]
        
        # TALLER KERNEL (2x5): Re-connects vertical broken letters (like 'p') 
        # without smearing sideways into commas/other words.
        kernel_stroke = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
        dilated_strokes = cv2.dilate(line_binary, kernel_stroke, iterations=1)

        w_cnts, _ = cv2.findContours(dilated_strokes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_boxes = []
        for c in w_cnts:
            wx, wy, ww, wh = cv2.boundingRect(c)
            
            # LOWER NOISE FLOOR: >=2px captures tiny pen dots like "Mr."
            if ww >= 2 and wh >= 2 and (ww * wh) >= 4: 
                # EDGE FILTER: Ignore left-side page scanning artifacts
                if wx > 15:  
                    raw_boxes.append([wx, wy, ww, wh])

        raw_boxes.sort(key=lambda b: b[0]) # Left-to-right

        merged_blocks = []
        
        # STRICTER GAP THRESHOLD: Drops from 50% to 35% of letter height to separate closer words
        word_gap_thresh = int(median_h * 0.35) 
        
        for box in raw_boxes:
            wx, wy, ww, wh = box
            
            is_bottom_only = wy > (lh * 0.55) 
            is_punct_size = ww < (median_h * 0.8) and wh < (median_h * 1.2) 
            is_punctuation = is_punct_size and is_bottom_only

            is_dot_size = ww < (median_h * 0.5) and wh < (median_h * 0.5)
            is_top_half = (wy + wh) < (lh * 0.5)
            is_i_dot = is_dot_size and is_top_half

            if not merged_blocks:
                merged_blocks.append({'box': [wx, wy, ww, wh], 'is_punct': is_punctuation})
                continue

            last_block = merged_blocks[-1]
            lwx, lwy, lww, lwh = last_block['box']
            gap = wx - (lwx + lww)
            
            if is_i_dot and gap < word_gap_thresh:
                merged_blocks[-1]['box'] = [
                    min(lwx, wx), min(lwy, wy),
                    max(lwx + lww, wx + ww) - min(lwx, wx),
                    max(lwy + lwh, wy + wh) - min(lwy, wy)
                ]
                continue
                
            if is_punctuation:
                merged_blocks.append({'box': [wx, wy, ww, wh], 'is_punct': True})
                continue
                
            if gap < word_gap_thresh and not last_block['is_punct']:
                merged_blocks[-1]['box'] = [
                    min(lwx, wx), min(lwy, wy),
                    max(lwx + lww, wx + ww) - min(lwx, wx),
                    max(lwy + lwh, wy + wh) - min(lwy, wy)
                ]
            else:
                merged_blocks.append({'box': [wx, wy, ww, wh], 'is_punct': False})

        for block in merged_blocks:
            bx, by, bw, bh = block['box']
            color = (0, 0, 255) if block['is_punct'] else (0, 255, 0)
            cv2.rectangle(debug_canvas, (lx + bx, y_top + ly + by), 
                          (lx + bx + bw, y_top + ly + by + bh), color, 2)

    return debug_canvas

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAMPLE_IMAGE_PATH = os.path.abspath(os.path.join(
        SCRIPT_DIR, "..", "..", "Data", "Datasets", "IAMpages10", "150", "c03-000a.png"
    ))

    output_dir = os.path.join(SCRIPT_DIR, "handwriting_extraction_output")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(SAMPLE_IMAGE_PATH):
        img = cv2.imread(SAMPLE_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
        debug_img = preview_smart_segmentation(img)

        if debug_img is not None:
            debug_path = os.path.join(output_dir, "smart_boxes_preview.png")
            cv2.imwrite(debug_path, debug_img)
            print(f"[Success] Saved smart bounding box preview to:\n  {debug_path}")
    else:
         print(f"Error: Could not find image at {SAMPLE_IMAGE_PATH}")