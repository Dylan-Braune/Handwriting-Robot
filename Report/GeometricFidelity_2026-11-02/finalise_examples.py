from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path('Report/GeometricFidelity_2026-11-02');env={'__file__':str((B/'measure_geometry.py').resolve()),'__name__':'functions'};exec((B/'measure_geometry.py').read_text().split('results={};start=time.time()')[0],env)
r=json.loads((B/'raw_results.json').read_text())['character'];samples=[next(q for q in r if q['writer']=='owen' and q['char']==c) for c in ['T','n','V']]
fig,ax=plt.subplots(3,3,figsize=(7.3,6.0),layout='constrained')
for j,row in enumerate(samples):
 d=np.load(B/'masks'/f"{row['id']}.npz");_,m=env['compare'](d['original'],d['physical'],True);yy,xx=np.where(m['original']|m['physical']);sl=np.s_[max(0,yy.min()-5):yy.max()+6,max(0,xx.min()-5):xx.max()+6]
 for i,key in enumerate(['original','physical']):ax[i,j].imshow(m[key][sl],cmap='gray_r',interpolation='nearest');ax[i,j].axis('off')
 a=m['original_skeleton'];b=m['physical_skeleton'];rgb=np.ones((*a.shape,3));rgb[a]=[.85,.3,.15];rgb[b]=[.1,.4,.75];rgb[a&b]=[.35,.15,.55];ax[2,j].imshow(rgb[sl],interpolation='nearest');ax[2,j].axis('off');ax[2,j].set_title(f"F-score {100*row['f1_0.10']:.2f}%",fontsize=10,y=-.19)
 ax[0,j].set_title(f"{row['char']} in {row['token']}",fontsize=11)
for i,l in enumerate(['Original mask','Robot mask','Centreline overlay']):ax[i,0].text(-.15,.5,l,transform=ax[i,0].transAxes,rotation=90,ha='center',va='center',fontsize=9)
fig.suptitle('Owen examples at the fixed 0.10 x-height tolerance\nOrange original, blue robot, purple exact overlap',fontsize=11);fig.savefig(B/'figures/report_character_examples.png',dpi=240)
# Make the limits explicit and retain the exact measurement code that generated the hashes.
p=B/'FINDINGS.md';s=p.read_text().replace('The observed point estimates exceed 85% in the sampled Owen case.','The observed point estimates exceed 85% in the sampled Owen case. Seventeen of the 25 individual Owen characters have F-scores at least 85%. Thus a mean score of 87.09% must not be described as 87.09% of characters passing a threshold.');p.write_text(s)
raw=json.loads((B/'raw_results.json').read_text());old="sys.path.insert(0,str(ROOT/'Software/CleanPipeline3'))";new="sys.path.insert(0,str(ROOT/'Software/CleanPipeline3' if (ROOT/'Software/CleanPipeline3/segment.py').exists() else B/'source'))"
import hashlib
for fn,expected in raw['code_sha256'].items():
 text=(B/fn).read_text().replace(new,old)
 candidates=[text.encode(),text.replace('\n','\r\n').encode()]
 for by in candidates:
  if hashlib.sha256(by).hexdigest()==expected:(B/'source'/(Path(fn).stem+'_as_run.py')).write_bytes(by);break
 else:raise AssertionError(('original code hash not recovered',fn))
print('Example figure and exact as-run source records saved')
