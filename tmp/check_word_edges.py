from pathlib import Path
import cv2,numpy as np,json
from PIL import Image
B=Path('Report/GeometricFidelity_2026-11-02');rr=json.loads((B/'word_boxes.json').read_text())
for key,lo in [('original_owen_09',1530),('original_yeukita_04',2140)]:
 row=next(r for r in rr if r['key']==key);a=np.asarray(Image.open(row['source_file']).convert('L'));_,m=cv2.threshold(a,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU);n,lab,st,ct=cv2.connectedComponentsWithStats(m,8)
 print(key,[list(map(int,z)) for z in st[1:] if z[0]>lo and z[4]>=6])
