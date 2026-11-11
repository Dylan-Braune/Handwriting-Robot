from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

out = Path(__file__).parent
items = [
    ('APPENDIX4_2026.pdf', [12, 13, 14, 15]),
    ('aston_not_finished.pdf', [32, 36, 43, 46]),
    ('23531003_Report_v2.pdf', [42, 47, 48, 52]),
]
for n, (name, ids) in enumerate(items):
    doc = pdfium.PdfDocument(str(Path('C:/Users/braun/Downloads') / name))
    sheet = Image.new('RGB', (1800, 2540), '#cccccc')
    draw = ImageDraw.Draw(sheet)
    for j, i in enumerate(ids):
        im = doc[i].render(scale=1.5).to_pil().convert('RGB')
        im.thumbnail((880, 1215))
        x, y = (j % 2)*900+10, (j//2)*1270+35
        sheet.paste(im, (x, y))
        draw.text((x, y-22), f'{name}: PDF page {i+1}', fill='black')
    sheet.save(out / f'audit-reference-{n+1}.jpg')
