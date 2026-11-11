from pathlib import Path
import pypdfium2 as p
from PIL import Image,ImageDraw
b=Path('Report/IntegratedReport_2026-11-01/Figures/VisualTests');fs=sorted(b.glob('*.pdf'));o=Path('tmp/report_context_review/visualtests')
for st in range(0,len(fs),4):
 im=Image.new('RGB',(1600,1600),'#ddd');d=ImageDraw.Draw(im)
 for j,f in enumerate(fs[st:st+4]):
  doc=p.PdfDocument(str(f));a=doc[0].render(scale=1.5).to_pil().convert('RGB');a.thumbnail((790,760));x=j%2*800;y=j//2*800;im.paste(a,(x,y+30));d.text((x+4,y+4),f.stem,fill='black')
 im.save(o/f'included-{st//4+1}.jpg')
print('Rendered all 12 included plot assets.')
