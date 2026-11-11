"""Render a subsystem illustration using existing crops and live image primitives.

No models are loaded, no training is run and no hardware interface is invoked.
This is a processing illustration, not an accuracy or qualification experiment.
"""
from pathlib import Path
import re
import sys
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import pypdfium2 as pdfium

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "Software" / "CleanPipeline3"))
import segment as SEG
import recognise as REC

sources = [
    ("Owen", ROOT / "Software/CleanPipeline2/testing/results/images/crops/owen_1.png"),
    ("Yeukita", ROOT / "Software/CleanPipeline2/testing/results/images/crops/yeukita_1.png"),
]
pdf_path = HERE / "Figures/design-stroke-normalisation.pdf"
# 455.24 pt is the report's text column. The page is authored at exactly that
# width and included at width=\linewidth, so LaTeX does not rescale it and a
# point here is a point on the page. The type is 12 pt, as in the other
# figures, rather than the 10 pt this used before.
COLUMN = 455.24
# The crops are prepared first so the page can be made exactly as tall as the
# panels need. Each writer's crop has its own proportions, so a fixed box
# height either letterboxed the writing in white or would have stretched it.
prepared = []
for name, path in sources:
    gray = np.asarray(Image.open(path).convert("L"))
    # Select the first short segment for readable stroke detail. Compute the
    # transform on the full crop before taking the same segment in each panel.
    ink = SEG.BinarizeLine(gray)
    skeleton = SEG.Skeletonize(ink)
    normalized = REC.StrokeNormalize(gray, penWidthXh=0.17)
    if normalized is None:
        raise RuntimeError(f"No ink in {path}")
    yy, xx = np.nonzero(ink)
    left = max(0, int(xx.min()) - 3)
    right = min(gray.shape[1], left + max(110, int(gray.shape[0] * 2.8)))
    # Drop the blank paper above and below the writing. Without this the
    # panel is mostly margin and the letters print a couple of millimetres
    # high, which is the whole complaint about these crops.
    rws = np.nonzero(ink.any(1))[0]
    pad = max(2, int(0.10 * (rws[-1] - rws[0] + 1)))
    top = max(0, int(rws[0]) - pad)
    bot = min(gray.shape[0], int(rws[-1]) + 1 + pad)
    # The thinned centreline is one pixel wide. At this printed size that is
    # a third of a point and vanishes, so the DISPLAY copy is thickened by
    # one pixel. The thinning itself is unchanged; the caption says so.
    thick = skeleton.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            thick |= np.roll(np.roll(skeleton, dy, 0), dx, 1)
    # Contrast-stretch the grey photograph only: a pencil stroke printed at
    # this size is otherwise a grey smudge. The mask, skeleton and normalised
    # panels are already black on white and are the code's own output.
    g = np.asarray(gray, dtype=np.float64)
    dark = float(np.percentile(g[g < 200], 1.0)) if (g < 200).any() else g.min()
    page = float(np.percentile(g, 97.0))
    stretched = (255.0 * np.clip((g - dark) / max(page - dark, 1.0), 0, 1) ** 1.55)
    arrays = [stretched, 255 * (~ink), 255 * (~thick), normalized]
    prepared.append((name, [np.asarray(a[top:bot, left:right], dtype=np.uint8)
                            for a in arrays], (bot - top) / float(right - left)))

# The box is sized to the crop's own shape, so preserveAspectRatio has nothing
# left to letterbox and the panel is filled rather than floated in white.
# the five photographed writers appear as a letter, never as a name
AUTHOR={"owen":"O","yeukita":"Y","dylan":"D","abhinav":"A","thiya":"T"}
_NAMEPAT=re.compile(r"(?i)\b(owen|yeukita|dylan|abhinav|thiya)\s*(\d+)?")
def _sub(m):
    out="Author "+AUTHOR[m.group(1).lower()]
    if m.group(2): out+=" page %d"%int(m.group(2))
    return out
def Anon(s):
    return _NAMEPAT.sub(_sub,str(s))
def AuthorLabel(w):
    k=str(w).strip().lower()
    return "Author "+AUTHOR[k] if k in AUTHOR else ("IAM "+str(w) if str(w).isdigit() else str(w))


COLTITLES = ["original", "binary-mask", "skeleton", "fixed-width"]
_KEPT = [0]


# each panel also written on its own, when FIG_PANELS names a folder
def _Keep(name, col, arr):
    import os, re
    root = os.environ.get("FIG_PANELS")
    if not root:
        return
    try:
        from panelexport import _WritePng
        d = Path(root)/pdf_path.stem
        d.mkdir(parents=True, exist_ok=True)
        _KEPT[0] += 1
        who = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
        _WritePng(d/("%02d_%s-%s.png" % (_KEPT[0], who, COLTITLES[col])), arr)
    except Exception as e:
        print("    (panel not written: %s)" % e)


LEFTLAB, BOXW, GAPX, GAPY, TOPSTRIP, BOTSTRIP = 52.0, 93.0, 10.0, 26.0, 26.0, 10.0
heights = [BOXW * ar for _, _, ar in prepared]
pageH = sum(heights) + (len(heights) - 1) * GAPY + TOPSTRIP + BOTSTRIP
c = canvas.Canvas(str(pdf_path), pagesize=(COLUMN, pageH))
c.setFont("Times-Roman", 12)
for col, title in enumerate(["Original", "Binary mask", "Skeleton",
                             "Fixed width"]):
    c.drawCentredString(LEFTLAB + col * (BOXW + GAPX) + BOXW / 2,
                        pageH - TOPSTRIP + 8, title)
y = pageH - TOPSTRIP
for row, (name, arrays, ar) in enumerate(prepared):
    boxH = heights[row]
    y -= boxH
    for col, arr in enumerate(arrays):
        _Keep(name, col, arr)
        c.drawImage(ImageReader(Image.fromarray(arr)),
                    LEFTLAB + col * (BOXW + GAPX), y,
                    width=BOXW, height=boxH, preserveAspectRatio=True,
                    anchor="c", mask="auto")
    c.drawString(2, y + boxH / 2 - 4, Anon(name))
    # one hairline between columns: without it the four versions of the same
    # words read as one continuous line of writing
    c.setStrokeColorRGB(.66, .70, .73);c.setLineWidth(.5)
    for col in range(1, 4):
        x = LEFTLAB + col * (BOXW + GAPX) - GAPX / 2
        c.line(x, y - 2, x, y + boxH + 2)
    y -= GAPY
c.showPage()
c.save()
doc = pdfium.PdfDocument(str(pdf_path))
doc[0].render(scale=2.5).to_pil().save(HERE / "Figures/design-stroke-normalisation.png")
doc.close()
print("Generated stroke-normalisation illustration from existing crops.")
