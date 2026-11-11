from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[2]
doc=pdfium.PdfDocument(str(R/'Report/IntegratedReport_2026-11-01/report_preview.pdf'))
out=R/'tmp/report_context_review/full_report_render';out.mkdir(exist_ok=True)
for i in range(len(doc)):
 im=doc[i].render(scale=1.5).to_pil().convert('RGB');im.save(out/f'page-{i+1:03d}.png')
 if (i+1)%20==0:print('Rendered',i+1,'of',len(doc),flush=True)
for start in range(0,len(doc),4):
 sheet=Image.new('RGB',(1800,2560),'#ccc');draw=ImageDraw.Draw(sheet)
 for j in range(min(4,len(doc)-start)):
  im=Image.open(out/f'page-{start+j+1:03d}.png');im.thumbnail((880,1220));x=(j%2)*900+10;y=(j//2)*1280+30
  draw.text((x,y-22),f'Page {start+j+1}',fill='black');sheet.paste(im,(x,y))
 sheet.save(out/f'sheet-{start//4+1:02d}.jpg',quality=90)
print('Completed',len(doc),'rendered pages.')
