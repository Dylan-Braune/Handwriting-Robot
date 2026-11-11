from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import subprocess, json, re
from pypdf import PdfReader
BASE=Path(__file__).resolve().parent
PDF=BASE/'build/main.pdf'
QA=BASE/'visual_qa';QA.mkdir(exist_ok=True)
reader=PdfReader(PDF)
print('Pages:',len(reader.pages),flush=True)
texts=[p.extract_text() or '' for p in reader.pages]
(QA/'page_text.json').write_text(json.dumps(texts),encoding='utf-8')
eqs=[]
for i,t in enumerate(texts,1):
    for n in re.findall(r'Report equation\s+(\d+)',t):eqs.append((int(n),i))
print('Equation homes:',eqs,flush=True)
assert sorted(n for n,p in eqs)==list(range(1,36))
subprocess.run(['pdftoppm','-scale-to','340','-png',str(PDF),str(QA/'thumb')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
files=sorted(QA.glob('thumb-*.png'))
for start in range(0,len(files),64):
    sheet=Image.new('RGB',(8*350,8*505),'#d6dce1');draw=ImageDraw.Draw(sheet)
    for j,p in enumerate(files[start:start+64]):
        im=Image.open(p).convert('RGB');im.thumbnail((340,475))
        x=(j%8)*350+5;y=(j//8)*505+22
        sheet.paste(im,(x,y));draw.text((x,y-17),p.stem.replace('thumb-','Page '),fill='black')
    sheet.save(QA/f'contact_{start//64+1:02d}.jpg',quality=90)
pages={1,2,3,4,5,6,14,15,16,20,len(reader.pages)}
pages|={p for n,p in eqs if n in [1,12,16,19,21,23,25,29,31,35]}
for i,t in enumerate(texts,1):
    if 'Implementation speed and accuracy' in t and ('Integral' in t or 'CTC' in t):pages.add(i)
for p in sorted(pages):
    subprocess.run(['pdftoppm','-f',str(p),'-l',str(p),'-scale-to','1400','-singlefile','-png',str(PDF),str(QA/f'page_{p:03d}')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
(QA/'qa_summary.json').write_text(json.dumps({'pages':len(texts),'report_equations':eqs,'fullsize_pages':sorted(pages)},indent=2),encoding='utf-8')
print('Contact sheets and full-size samples rendered.',flush=True)
