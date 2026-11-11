from pathlib import Path
from PIL import Image,ImageDraw
P=Path('Report/SubmissionRevision_2026-11-02/Figures/PhysicalPages');files=sorted(P.glob('*.jpg'));sheet=Image.new('RGB',(1400,700),'#ddd');d=ImageDraw.Draw(sheet)
for j,f in enumerate(files):
 im=Image.open(f);im.thumbnail((190,300));x=j%7*200;y=j//7*350;sheet.paste(im,(x,y+25));d.text((x+5,y+5),f.stem,fill='black')
sheet.save('tmp/final_report_review/render/page-orientations.jpg')
