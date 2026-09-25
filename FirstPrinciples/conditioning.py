"""
conditioning.py -- fixes noise and skew in the scanned page (FU 2.1),
by hand, using only numpy arithmetic. Two steps: binarize (separate
ink from paper), then deskew (straighten a crooked scan).
"""
import numpy as np


def binarize(gray):
    """Turns a grayscale page into ink=1 / background=0 using Otsu's
    method: try every possible split point (0-255) and keep the one
    that best separates the two pixel groups, i.e. maximises the
    variance BETWEEN the ink group and the background group."""
    hist = np.bincount(gray.flatten(), minlength=256).astype(np.float64)
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)

    best_t, best_score = 0, -1.0
    weight_bg, sum_bg = 0.0, 0.0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        score = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        # >= (not >): when several thresholds tie for best (e.g. a flat
        # gap between two clean clusters), keep the LAST one, not the
        # first -- otherwise the threshold lands exactly on the ink
        # value itself and "gray < best_t" then matches nothing.
        if score >= best_score:
            best_score, best_t = score, t

    return (gray < best_t).astype(np.uint8)  # ink is darker than paper


def _rotate(binary_img, angle_deg):
    """Rotates a binary image about its centre by angle_deg. Just the
    standard 2D rotation matrix applied to every ink pixel's (x, y)
    coordinate, then nearest-neighbour placement -- coordinate maths,
    not a call into an image-processing library."""
    h, w = binary_img.shape
    cy, cx = h / 2.0, w / 2.0
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    out = np.zeros_like(binary_img)
    ys, xs = np.nonzero(binary_img)
    y_shift, x_shift = ys - cy, xs - cx
    new_y = np.round(x_shift * sin_t + y_shift * cos_t + cy).astype(int)
    new_x = np.round(x_shift * cos_t - y_shift * sin_t + cx).astype(int)
    valid = (new_y >= 0) & (new_y < h) & (new_x >= 0) & (new_x < w)
    out[new_y[valid], new_x[valid]] = 1
    return out


def deskew(binary_img, search_range_deg=10, step_deg=0.5):
    """Finds and corrects page skew: tries a range of small rotation
    angles and keeps whichever one gives the sharpest horizontal ink
    profile (text lines pack tightly into rows when the page is
    straight, giving high variance in row-by-row ink counts)."""
    best_angle, best_variance = 0.0, -1.0
    for angle in np.arange(-search_range_deg, search_range_deg + step_deg, step_deg):
        rotated = _rotate(binary_img, angle)
        row_ink_counts = rotated.sum(axis=1)
        variance = row_ink_counts.var()
        if variance > best_variance:
            best_variance, best_angle = variance, angle
    return _rotate(binary_img, best_angle)
