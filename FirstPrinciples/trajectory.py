"""
trajectory.py -- converts a segmented character's binary image into a
pen trajectory: an ordered list of (x, y) points the gantry moves
through to draw it (FU 2.5). Two hand-written steps:
1) thin the character down to a 1-pixel-wide centreline (Zhang-Suen
   thinning algorithm),
2) walk along that centreline in order to build a path.
"""
import numpy as np


def _neighbours(img, y, x):
    """The 8 pixels around (y, x), clockwise starting north -- used by
    the two Zhang-Suen removal conditions below."""
    return [img[y-1, x], img[y-1, x+1], img[y, x+1], img[y+1, x+1],
            img[y+1, x], img[y+1, x-1], img[y, x-1], img[y-1, x-1]]


def skeletonize(binary_img, max_iterations=50):
    """Zhang-Suen thinning: repeatedly strips ink pixels off the edge
    of a stroke that aren't needed to keep it connected, leaving only
    the centreline. Runs until nothing changes or max_iterations."""
    img = binary_img.copy().astype(np.uint8)
    h, w = img.shape
    for _ in range(max_iterations):
        changed = False
        for sub_step in (0, 1):
            to_remove = []
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if img[y, x] != 1:
                        continue
                    p = _neighbours(img, y, x)
                    black_neighbours = sum(p)
                    transitions = sum(
                        (p[i] == 0 and p[(i + 1) % 8] == 1) for i in range(8)
                    )
                    if not (2 <= black_neighbours <= 6 and transitions == 1):
                        continue
                    if sub_step == 0:
                        if p[0] * p[2] * p[4] == 0 and p[2] * p[4] * p[6] == 0:
                            to_remove.append((y, x))
                    else:
                        if p[0] * p[2] * p[6] == 0 and p[0] * p[4] * p[6] == 0:
                            to_remove.append((y, x))
            for y, x in to_remove:
                img[y, x] = 0
                changed = True
        if not changed:
            break
    return img


def trace_path(skeleton):
    """Orders the skeleton's pixels into a walkable path: start
    anywhere, then repeatedly hop to the nearest not-yet-visited
    skeleton pixel. Simple, and good enough for single thin strokes."""
    ys, xs = np.nonzero(skeleton)
    points = list(zip(xs.tolist(), ys.tolist()))
    if not points:
        return []
    path = [points.pop(0)]
    while points:
        last_x, last_y = path[-1]
        distances = [(px - last_x) ** 2 + (py - last_y) ** 2 for px, py in points]
        nearest = int(np.argmin(distances))
        path.append(points.pop(nearest))
    return path


def image_to_trajectory(binary_char_img):
    skeleton = skeletonize(binary_char_img)
    return trace_path(skeleton)
