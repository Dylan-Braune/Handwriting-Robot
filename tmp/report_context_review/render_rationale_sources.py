from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
HERE=Path(__file__).parent
items=[('haines2016',3),('graves2013',17),('dai2024',8)]
sheet=Image.new('RGB',(2400,1160),'#cccccc');d=ImageDraw.Draw(sheet)
for j,(name,page) in enumerate(items):
    doc=pdfium.PdfDocument(str(HERE/'sources'/(name+'.pdf')))
    im=doc[page].render(scale=1.4).to_pil().convert('RGB');im.thumbnail((780,1110))
    sheet.paste(im,(j*800+10,35));d.text((j*800+10,10),f'{name}, PDF page {page+1}',fill='black');doc.close()
sheet.save(HERE/'rationale-source-pages.jpg',quality=92)
