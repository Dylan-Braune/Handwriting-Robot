"""Render a subsystem illustration using existing crops and live image primitives.

No models are loaded, no training is run and no hardware interface is invoked.
This is a processing illustration, not an accuracy or qualification experiment.
"""
from pathlib import Path
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
c = canvas.Canvas(str(pdf_path), pagesize=(460, 180))
c.setFont("Times-Roman", 10)
for col, title in enumerate(["Original", "Binary mask", "Skeleton", "Fixed width"]):
    c.drawCentredString(45 + col * 104 + 51, 163, title)
for row, (name, path) in enumerate(sources):
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
    arrays = [gray, 255 * (~ink), 255 * (~skeleton), normalized]
    for col, arr in enumerate(arrays):
        panel = Image.fromarray(np.asarray(arr[:, left:right], dtype=np.uint8))
        c.drawImage(ImageReader(panel), 45 + col * 104, 93 - row * 75,
                    width=102, height=58, preserveAspectRatio=True,
                    anchor="c", mask="auto")
    c.drawString(2, 118 - row * 75, name)
c.showPage()
c.save()
doc = pdfium.PdfDocument(str(pdf_path))
doc[0].render(scale=2.5).to_pil().save(HERE / "Figures/design-stroke-normalisation.png")
doc.close()
print("Generated stroke-normalisation illustration from existing crops.")
