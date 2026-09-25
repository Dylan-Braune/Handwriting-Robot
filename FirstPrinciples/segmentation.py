"""
segmentation.py -- separates a conditioned page into individual
characters (FU 2.2), using a from-scratch connected-component search
(flood fill) instead of any image-processing library's labelling
function, and a hand-written nearest-neighbour resize.
"""
import numpy as np


def _flood_fill(binary_img, visited, start_y, start_x):
    """Explores one connected blob of ink pixels (4-connected: up,
    down, left, right) using a stack, and returns its bounding box."""
    h, w = binary_img.shape
    stack = [(start_y, start_x)]
    visited[start_y, start_x] = True
    min_y = max_y = start_y
    min_x = max_x = start_x
    while stack:
        y, x = stack.pop()
        min_y, max_y = min(min_y, y), max(max_y, y)
        min_x, max_x = min(min_x, x), max(max_x, x)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and binary_img[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                stack.append((ny, nx))
    return min_y, max_y, min_x, max_x


def find_characters(binary_img, min_area=15, line_height_px=20):
    """Returns one bounding box (min_y, max_y, min_x, max_x) per
    connected ink blob -- each blob is one character, or a small group
    of touching/cursive characters. Boxes are ordered into reading
    order: top-to-bottom by text line, then left-to-right."""
    visited = np.zeros_like(binary_img, dtype=bool)
    boxes = []
    ys, xs = np.nonzero(binary_img)
    for y, x in zip(ys, xs):
        if binary_img[y, x] and not visited[y, x]:
            box = _flood_fill(binary_img, visited, y, x)
            area = (box[1] - box[0] + 1) * (box[3] - box[2] + 1)
            if area >= min_area:
                boxes.append(box)
    boxes.sort(key=lambda b: (b[0] // line_height_px, b[2]))
    return boxes


def crop_and_resize(binary_img, box, size=20):
    """Crops one character's bounding box and resizes it to a fixed
    size x size square (the neural network needs a constant input
    size), using nearest-neighbour resampling: pick, for each output
    pixel, the closest pixel in the original crop."""
    min_y, max_y, min_x, max_x = box
    crop = binary_img[min_y:max_y + 1, min_x:max_x + 1]
    h, w = crop.shape
    row_idx = (np.arange(size) * h / size).astype(int).clip(0, h - 1)
    col_idx = (np.arange(size) * w / size).astype(int).clip(0, w - 1)
    return crop[row_idx][:, col_idx]
