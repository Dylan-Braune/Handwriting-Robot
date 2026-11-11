from pathlib import Path
from PIL import Image,ImageDraw,ImageOps
folder=Path('C:/Users/braun/OneDrive/Pictures/Camera Roll')
items=sorted(folder.glob('WIN_202610*.jpg'))
out=Path(__file__).parent/'camera_review';out.mkdir(exist_ok=True)
for start in range(0,len(items),16):
    sheet=Image.new('RGB',(1600,1240),'white');d=ImageDraw.Draw(sheet)
    for j,p in enumerate(items[start:start+16]):
        im=ImageOps.exif_transpose(Image.open(p));im.thumbnail((390,275))
        x=(j%4)*400;y=(j//4)*310
        sheet.paste(im,(x+(390-im.width)//2,y));d.text((x+3,y+278),p.name,fill='black')
    sheet.save(out/f'sheet-{start//16+1:02}.jpg')
print(len(items),'images', (len(items)+15)//16,'sheets')
