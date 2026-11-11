from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
OUT=Path(__file__).parent
S=OUT/'sources'
checks=[('a4988ds',5),('a4988ds',8),('bresenham1965',3),('zhang1984',1),
 ('kizilirmak2023',3),('kizilirmak2023',4),('ioffe2015',3),('ioffe2015',4),
 ('graves2006',1),('graves2006',3),('graves2006',4),('graves2006',5),
 ('srivastava2014',2),('srivastava2014',5),('rosenfeld1966',5),('avidan2007',2),
 ('graves2013',4),('graves2013',5),('kingma2015',1),('loshchilov2019',2),
 ('tieleman2012',28),('levenshtein1966',0),('levenshtein1966',1)]
for start in range(0,len(checks),4):
 sheet=Image.new('RGB',(1800,2540),'#cccccc');draw=ImageDraw.Draw(sheet)
 for j,(key,idx) in enumerate(checks[start:start+4]):
  doc=pdfium.PdfDocument(str(S/f'{key}.pdf'));im=doc[idx].render(scale=1.6).to_pil().convert('RGB');im.thumbnail((880,1215))
  x,y=(j%2)*900+10,(j//2)*1270+35;sheet.paste(im,(x,y));draw.text((x,y-22),f'{key}, PDF page {idx+1}',fill='black')
 sheet.save(OUT/f'source-check-{start//4+1:02d}.jpg')
