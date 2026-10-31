#draw the combined test sheet from the G-code ITSELF, not from the geometry
#the maker used.  the point of the picture is to show what the file will make
#the pen do, so the path is parsed back out of the file: a bug in the writer
#shows up here, and a plot re-using the same constants would not.
#
#   python plot.py
#
#two pictures come out.  the plain one is the drawn line alone, which is what
#the paper should look like.  the path one adds the pen-up moves, so the whole
#toolpath is visible: where the gantry travels between the two figures and out
#to the park position.
import math
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import reproduce as REP

SRC = HERE/"square_triangle_100mm_80mm.gcode"

SCALE = 4           #pixels per millimetre
MARGIN = 40         #pixels of border around the sheet

usableW = REP.X_SWITCH_TRAVEL_MM - 2*REP.EDGE_TOLERANCE_MM
usableH = REP.Y_SWITCH_TRAVEL_MM - 2*REP.EDGE_TOLERANCE_MM

INK = (15, 15, 15)
TRAVEL = (215, 110, 110)
SHEET = (215, 215, 215)

MOVE = re.compile(r"^(G0|G1)\b")
AXIS = re.compile(r"([XY])(-?\d+(?:\.\d+)?)")


#---Parse: the pen path, straight out of the file---
def Parse(path):
    segs, x, y, pen = [], 0.0, 0.0, False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(";")[0].strip()
        if not line:
            continue
        if line.startswith("M3"):
            pen = True
            continue
        if line.startswith("M5"):
            pen = False
            continue
        if not MOVE.match(line):
            continue
        nx, ny = x, y
        for ax, val in AXIS.findall(line):
            if ax == "X":
                nx = float(val)
            else:
                ny = float(val)
        segs.append(((x, y), (nx, ny), pen))
        x, y = nx, ny
    return segs


def P(px, py, H):
    #machine Y runs up the page, image Y runs down it
    return (MARGIN+px*SCALE, H-MARGIN-py*SCALE)


def Dashed(d, a, b, H, colour, dash=8, gap=6, width=2):
    ax, ay = P(*a, H=H)
    bx, by = P(*b, H=H)
    total = math.hypot(bx-ax, by-ay)
    if total <= 0:
        return
    t = 0.0
    while t < total:
        t2 = min(total, t+dash)
        d.line([(ax+(bx-ax)*t/total, ay+(by-ay)*t/total),
                (ax+(bx-ax)*t2/total, ay+(by-ay)*t2/total)],
               fill=colour, width=width)
        t = t2+gap


def Render(segs, withTravel):
    H = int(usableH*SCALE)+2*MARGIN
    W = int(usableW*SCALE)+2*MARGIN
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([P(0, usableH, H), P(usableW, 0, H)], outline=SHEET, width=2)

    for a, b, down in segs:
        if down:
            d.line([P(*a, H=H), P(*b, H=H)], fill=INK, width=3)
        elif withTravel:
            Dashed(d, a, b, H, TRAVEL)
    return im


segs = Parse(SRC)
nDraw = sum(1 for _a, _b, p in segs if p)
for travel, name in ((False, "square_triangle_preview.png"),
                     (True, "square_triangle_path.png")):
    im = Render(segs, travel)
    im.save(HERE/name)
    print("wrote %-34s (%d x %d px)" % (name, im.width, im.height))
print("  %d drawn moves, %d travel moves, %d total"
      % (nDraw, len(segs)-nDraw, len(segs)))
