from pathlib import Path
from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
import re,json
b=Path('Report/IntegratedReport_2026-11-01');r=PdfReader(b/'full_report.pdf');d=pdfium.PdfDocument(str(b/'full_report.pdf'));o=Path('tmp/report_context_review/visualtests/report');o.mkdir(exist_ok=True)
needles=['Integer stepping and command-space','Seam paths and local cost fields','Initial seam-band ownership','Feature maps and height pooling','Forced alignment, greedy decoding','Real ink, its thinned centreline','Variant clouds, actual medoids','Nominal drawing, travel and dwell','Analytic and central-difference gradients','Real-crop normalisation kernels','The same glyph in image coordinates','Eligible character boundaries']
found={};pages=[p.extract_text() or '' for p in r.pages]
for needle in needles:
 inds=[i for i,t in enumerate(pages) if needle in t];assert len(inds)==1,(needle,inds);found[needle]=inds[0]+1
inds=sorted({i for n in found.values() for i in [n-2,n-1] if i>=0})
for i in inds:d[i].render(scale=1.2).to_pil().save(o/f'page-{i+1}.png')
for st in range(0,len(inds),4):
 im=Image.new('RGB',(1600,2260),'#ddd');dr=ImageDraw.Draw(im)
 for j,i in enumerate(inds[st:st+4]):
  a=Image.open(o/f'page-{i+1}.png');a.thumbnail((790,1090));x=j%2*800;y=j//2*1130;im.paste(a,(x,y+28));dr.text((x+4,y+5),f'PDF page {i+1}',fill='black')
 im.save(o/f'sheet-{st//4+1}.jpg')
assert not any('??' in t for t in pages)
print(json.dumps({'pdf_pages':len(pages),'figures':found,'contact_sheets':(len(inds)+3)//4},indent=2))
(b/'VisualTestEvidence/INTEGRATION_CHECK.json').write_text(json.dumps({'pdf_pages':len(pages),'figure_pdf_pages':found,'unresolved_markers':False},indent=2),encoding='utf-8')
