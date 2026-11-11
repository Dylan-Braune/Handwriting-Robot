from pathlib import Path
from PIL import Image,ImageDraw
fs=sorted(Path('Report/DesignVisualTests_2026-11-01/figures').glob('*.png'))
o=Path('tmp/report_context_review/visualtests');o.mkdir(exist_ok=True)
for st in range(0,len(fs),6):
 im=Image.new('RGB',(1800,1800),'#ddd');d=ImageDraw.Draw(im)
 for j,p in enumerate(fs[st:st+6]):
  x=j%2*900;y=j//2*600;a=Image.open(p).convert('RGB');a.thumbnail((890,565));im.paste(a,(x,y+30));d.text((x+5,y+5),p.stem,fill='black')
 im.save(o/f'sheet-{st//6+1}.jpg')
print(len(fs),'images', (len(fs)+5)//6,'sheets')
