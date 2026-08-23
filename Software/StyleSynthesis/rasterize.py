"""
rasterize.py

Turns a stroke list (the same format motion_planner.py consumes) into a
grayscale image -- used two ways:
  1. Quick visual sanity check while developing (just save + look at it).
  2. Feeding the synthesized word back through the EXISTING trained CRNN
     classifier (train_paper_cnn_bilstm_ctc.py's PaperCRNN) as an accuracy
     check, before any of this touches a motor. That round-trip is the
     "confirm these mappings work accurately" step -- see validate_synthesis.py.
"""

from PIL import Image, ImageDraw


def rasterize_strokes(strokes, lineWidthPx=3, marginPx=20):
    if not strokes:
        return Image.new("L", (marginPx * 2, marginPx * 2), 255)

    allX = [x for stroke in strokes for x, y in stroke]
    allY = [y for stroke in strokes for x, y in stroke]
    minX, maxX = min(allX), max(allX)
    minY, maxY = min(allY), max(allY)

    w = int(maxX - minX) + 2 * marginPx
    h = int(maxY - minY) + 2 * marginPx
    img = Image.new("L", (max(1, w), max(1, h)), 255)
    draw = ImageDraw.Draw(img)

    for stroke in strokes:
        pts = [(x - minX + marginPx, y - minY + marginPx) for x, y in stroke]
        if len(pts) >= 2:
            draw.line(pts, fill=0, width=lineWidthPx, joint="curve")
        elif len(pts) == 1:
            r = lineWidthPx / 2
            x, y = pts[0]
            draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    return img
