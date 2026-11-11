from PIL import Image,ImageDraw
from pathlib import Path
b=Path('Report/SubmissionRevision_2026-11-02/Figures');im=Image.new('RGB',(1400,1200),'#ddd');d=ImageDraw.Draw(im)
for j in range(8):
 p=b/f'cad-component-{j+1:02d}.png';a=Image.open(p).convert('RGB');a.thumbnail((340,540));x=j%4*350;y=j//4*600;im.paste(a,(x,y+35));d.text((x+4,y+4),f'{j+1:02d}',fill='black')
im.save('tmp/final_report_review/cad.jpg')
