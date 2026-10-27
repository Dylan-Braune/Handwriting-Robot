"""Run the segmenter on each rendered sentence and draw what it found on top.

red boxes   the line crops the segmenter would hand to the recogniser
blue curves the seam cuts it chose between lines
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import numpy as np
from PIL import Image, ImageDraw
import paths, segment as SEG, train as TR

rows = []
for i, w in enumerate(paths.PersonalWriters, 1):
    src = HERE/("%d_%s.png" % (i, w))
    if not src.exists():
        continue
    gray = TR.ReadImageFile(src)
    lines, dbg, info = SEG.SegmentLines(gray)

    im = Image.fromarray(gray).convert("RGB")
    d = ImageDraw.Draw(im)
    H, W = gray.shape
    # the seam cuts come back at half resolution, so they double
    for cut in dbg.get("cuts", []):
        arr = np.asarray(cut)
        if arr.ndim != 1 or len(arr) < 2:
            continue
        pts = [(2.0*x, 2.0*float(arr[x])) for x in range(0, len(arr), 4)]
        d.line(pts, fill=(60, 110, 255), width=2)
    for k, ln in enumerate(lines):
        x0, y0, x1, y1 = ln["bbox"]
        d.rectangle([x0, y0, x1, y1], outline=(230, 30, 30), width=3)
        d.text((x0+4, max(0, y0-14)), "line %d  %dx%d" % (k, x1-x0, y1-y0), fill=(230, 30, 30))
    out = HERE/("%d_%s_segmented.png" % (i, w))
    im.save(out)

    covered = sum((ln["bbox"][2]-ln["bbox"][0]) for ln in lines)
    rows.append((w, gray.shape, len(lines), info["pitch"], covered, out.name))
    print("%-9s image %5dx%-5d  lines %d  pitch %3d  widest crop %4d px of %4d (%.0f%%)  %s"
          % (w, W, H, len(lines), info["pitch"],
             max((ln["bbox"][2]-ln["bbox"][0]) for ln in lines) if lines else 0,
             W, 100*(max((ln["bbox"][2]-ln["bbox"][0]) for ln in lines)/W if lines else 0),
             out.name), flush=True)

with open(HERE/"SEGMENTATION.txt", "w", encoding="utf-8") as f:
    f.write("the segmenter run on each rendered sentence\n")
    f.write("red = the crop handed to the recogniser, blue = seam cuts\n\n")
    for w, shp, n, pitch, cov, name in rows:
        f.write("%-10s %5dx%-5d  %d line(s), pitch %d px  ->  %s\n"
                % (w, shp[1], shp[0], n, pitch, name))
print("wrote", HERE/"SEGMENTATION.txt")
