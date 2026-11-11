from pathlib import Path
import json,sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path('Report/GeometricFidelity_2026-11-02');rr=json.loads((B/'word_boxes.json').read_text())
keys=['original_thiya_00','original_owen_09','original_abhinav_04','original_abhinav_03','original_yeukita_04']
fig,axes=plt.subplots(len(keys),1,figsize=(18,9),layout='constrained')
for ax,key in zip(axes,keys):
 rows=[r for r in rr if r['key']==key];a=np.asarray(Image.open(rows[0]['source_file']).convert('L'));ax.imshow(a,cmap='gray',vmin=0,vmax=255);ax.set_aspect('auto');ax.set_title(key);ax.set_xticks(range(0,a.shape[1],100));ax.set_yticks([0,a.shape[0]//2,a.shape[0]-1]);ax.grid(axis='x',alpha=.3)
 for r in rows:ax.axvline(r['x0'],color='red',lw=.7);ax.text((r['x0']+r['x1'])/2,10,f"{r['word_index']} {r['token']}",color='blue',fontsize=8,ha='center')
fig.savefig(B/'word_boundary_review.png',dpi=110)
print([(r['key'],r['word_index'],r['token'],r['x0'],r['x1'],r['y0'],r['y1']) for r in rr if r['key'] in keys])
