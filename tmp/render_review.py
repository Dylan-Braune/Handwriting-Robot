from pathlib import Path
from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
P=Path('Report/SubmissionRevision_2026-11-02');out=Path('tmp/final_report_review/render');out.mkdir(exist_ok=True)
pdf=P/'.build/full_report.pdf';r=PdfReader(pdf);print('Physical pages',len(r.pages));labels=r.page_labels;print('Main indices',[(i,l) for i,l in enumerate(labels) if l in ['1','9','24','30','36','46','58','63','64','67']])
doc=pdfium.PdfDocument(str(pdf));indices=[i for i,l in enumerate(labels) if l.isdigit() and int(l)<=70]
for start in range(0,len(indices),12):
 sel=indices[start:start+12];sheet=Image.new('RGB',(1000,1416),'#d7d7d7');d=ImageDraw.Draw(sheet)
 for j,i in enumerate(sel):
  im=doc[i].render(scale=.4).to_pil();im.thumbnail((242,330));x=(j%4)*250;y=(j//4)*472;sheet.paste(im,(x,y+24));d.text((x+10,y+5),f'PDF {i+1}, report {labels[i]}',fill='black')
 sheet.save(out/f'contact_{start//12+1}.jpg')
for name in ['design-seam-two-panels','design-owen-first-features','results_reproduced_confusion']:
 d=pdfium.PdfDocument(str(P/'Figures'/f'{name}.pdf'));d[0].render(scale=1.6).to_pil().save(out/f'{name}.png')
# find remaining warnings
log=(P/'.build/full_report.log').read_text(errors='replace');print('\n'.join(x for x in log.splitlines() if any(k in x for k in ['undefined','multiply defined','Overfull'])))
