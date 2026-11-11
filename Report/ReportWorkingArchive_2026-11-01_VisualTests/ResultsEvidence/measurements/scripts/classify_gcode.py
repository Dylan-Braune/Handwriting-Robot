#identify the writer of a .gcode job by rendering it and classifying it.
#
#the four .gcode files in Downloads are jobs the robot actually ran, but the
#log only names a writer for a handful of sessions.  the G-code itself carries
#the answer: it is the writer's hand expressed as pen moves, so rendering it
#to a page and putting that page through the ordinary classify path names the
#writer without needing the log at all.
#
#this is also the gcode-versus-reproduction comparison in its own right: the
#same job can be classified as G-code here and as a photographed sheet
#elsewhere, and the two answers should agree.
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
warnings.filterwarnings("ignore")

CP3 = Path(r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline3")
sys.path.insert(0, str(CP3)); sys.path.insert(0, str(CP3/"web"))
import segment as SEG
import recognise as REC
import server as SRV

DL = Path(r"C:\Users\braun\Downloads")
OUT = Path(__file__).resolve().parent/"gcodeid"
OUT.mkdir(exist_ok=True)

PX_PER_MM = 11.8          #about 300 dpi, so the strokes land at a sane width
PEN_PX = 3

MOVE = re.compile(r"^(G0|G1)\b")
AXIS = re.compile(r"([XY])(-?\d+(?:\.\d+)?)")


def RenderGcode(path):
    """pen-down moves only, drawn on white, the way the paper ends up"""
    segs, x, y, pen = [], 0.0, 0.0, False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
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
        if pen:
            segs.append(((x, y), (nx, ny)))
        x, y = nx, ny
    if not segs:
        return None
    xs = [p[0] for s in segs for p in s]
    ys = [p[1] for s in segs for p in s]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = 8.0
    W = int((x1-x0+2*pad)*PX_PER_MM)
    H = int((y1-y0+2*pad)*PX_PER_MM)
    im = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(im)
    for a, b in segs:
        d.line([((a[0]-x0+pad)*PX_PER_MM, H-(a[1]-y0+pad)*PX_PER_MM),
                ((b[0]-x0+pad)*PX_PER_MM, H-(b[1]-y0+pad)*PX_PER_MM)],
               fill=0, width=PEN_PX)
    return im


#the sweep namer, and the crops redrawn at ITS pen width before it sees them
CKPT = DL/"author_shape_11_pw170_weights.npz"
PW = 0.17
model = REC.AuthorClassifierCNNNumpy(checkpointPath=str(CKPT))
byIdx = {v: k for k, v in model.authorMapping.items()}
NAMES = [byIdx[i] for i in range(len(byIdx))]
print("namer: %s  (%d writers, pen width %.2f)\n" % (CKPT.name, len(NAMES), PW))

rows = []
for g in sorted(DL.glob("*.gcode")):
    im = RenderGcode(g)
    if im is None:
        print("%-36s no pen-down moves" % g.name)
        continue
    im.save(OUT/(g.stem+"_render.png"))
    gray = np.asarray(im, dtype=np.uint8)
    try:
        lines, _d, info = SEG.SegmentLines(gray)
    except Exception as e:
        print("%-36s segmentation failed: %s" % (g.name, e))
        continue
    crops = [np.asarray(l["image"]) for l in lines]
    if not crops:
        print("%-36s no lines segmented" % g.name)
        continue
    per, w = [], []
    for c in crops:
        n = REC.StrokeNormalize(c, penWidthXh=PW)
        _k, p = REC.ClassifyImage(Image.fromarray(n if n is not None else c).convert("L"),
                                  model, strokeNormalise=False)
        per.append(np.asarray(p, dtype=np.float64))
        w.append(float((c < 128).sum()))
    w = np.array(w)
    if w.sum() <= 0:
        w = np.ones(len(crops))
    avg = (np.vstack(per)*w[:, None]).sum(axis=0)/w.sum()
    named = [NAMES[int(np.argmax(p))] for p in per]
    best = NAMES[int(np.argmax(avg))]
    agree = sum(1 for n in named if n == best)
    top = sorted(zip(NAMES, avg), key=lambda t: -t[1])[:3]
    rows.append({"file": g.name, "gcodeLines": len(g.read_text(errors="replace").splitlines()),
                 "imagePx": list(im.size), "lines": len(crops), "named": best,
                 "conf": round(100*float(avg.max()), 1),
                 "agree": "%d/%d" % (agree, len(crops)),
                 "top3": [[a, round(100*float(v), 1)] for a, v in top]})
    print("%-36s %5d gcode lines -> %2d text lines -> %-8s %5.1f%%  (lines agreeing %d/%d)"
          % (g.name, rows[-1]["gcodeLines"], len(crops), best,
             100*avg.max(), agree, len(crops)), flush=True)
    print("     top three: %s" % ", ".join("%s %.1f%%" % (a, 100*v) for a, v in top))

(OUT/"results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
print("\nrenders and results in", OUT)
