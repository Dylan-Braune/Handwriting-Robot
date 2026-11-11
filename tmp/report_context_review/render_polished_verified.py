from pathlib import Path
import json
import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from pypdf import PdfReader

root = Path(__file__).resolve().parents[2]
source = root / 'Report/DesignPolished_2026-11-01/design_preview.pdf'
out = root / 'tmp/report_context_review/polished_verified_render'
out.mkdir(parents=True, exist_ok=True)
doc = pdfium.PdfDocument(str(source))
pages = []
for i in range(len(doc)):
    im = doc[i].render(scale=1.6).to_pil().convert('RGB')
    im.save(out / f'page-{i+1:02d}.png')
    pages.append(im)
for start in range(0, len(pages), 4):
    sheet = Image.new('RGB', (2000, 2860), '#cccccc')
    draw = ImageDraw.Draw(sheet)
    for j, page in enumerate(pages[start:start+4]):
        im = page.copy()
        im.thumbnail((980, 1370))
        x, y = (j % 2) * 1000 + 10, (j // 2) * 1430 + 35
        draw.text((x, y - 25), f'Page {start+j+1}', fill='black')
        sheet.paste(im, (x, y))
    sheet.save(out / f'sheet-{start//4+1:02d}.jpg', quality=90)
reader = PdfReader(source)
texts = [p.extract_text() or '' for p in reader.pages]
(out/'extracted.txt').write_text('\n\n'.join(texts), encoding='utf-8')
print(json.dumps({'pages': len(pages), 'sheets': (len(pages)+3)//4,
                  'unresolved_question_marks': any('??' in t for t in texts),
                  'out': str(out)}, indent=2))
