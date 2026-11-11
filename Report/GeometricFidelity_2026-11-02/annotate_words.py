from pathlib import Path
import sys,json,re,hashlib
import numpy as np
from PIL import Image,ImageDraw
import cv2
ROOT=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'Software/CleanPipeline3' if (ROOT/'Software/CleanPipeline3/segment.py').exists() else B/'source'))
import segment as SEG
M=json.loads((B/'input_manifest.json').read_text());ann={};out=[]
if (B/'word_annotations.json').exists():ann=json.loads((B/'word_annotations.json').read_text())
def clean(a):
 m=SEG.BinarizeLine(a,stripRules=True,closeGaps=False);band=SEG.CoreBand(m)
 if band:
  t,b=band;xh=b-t;nn,lab,stats,cents=cv2.connectedComponentsWithStats(m.astype('uint8'),8)
  for i in range(1,nn):
   x,y,w,h,area=stats[i];cy=cents[i,1]
   if cy<t-.75*xh or cy>b+.75*xh or (h>3*xh and w<.12*xh):m[lab==i]=False
 return m
def bounds(m):
 ys,xs=np.where(m);return (int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)) if len(xs) else (0,0,m.shape[1],m.shape[0])
def gaps(mask):
 p=mask.any(0);v=np.diff(np.r_[True,p,True].astype(int));starts=np.where(v==-1)[0];ends=np.where(v==1)[0];return [(int(s),int(e)) for s,e in zip(starts,ends) if s>0 and e<len(p)]
for kind in ['original','physical']:
 for r in M[kind]:
  p=B/r['file'] if kind=='original' else ROOT/r['file'];a=np.asarray(Image.open(p).convert('L'));mask=clean(a);n=len(r['text'].split());key=f"{kind}_{r['writer']}_{r['line']:02}"
  x0,y0,x1,y1=bounds(mask);gg=gaps(mask)
  # Transcript-length priors select word gaps, without looking at the matching image.
  from scipy.ndimage import gaussian_filter1d
  tokens=r['text'].split();prof=gaussian_filter1d(mask.sum(0).astype(float),1.0)
  cand={x0:0.,x1:0.}
  for st,en in gg:cand[(st+en)//2]=-(2*np.log1p((en-st)/max(3,(y1-y0)/2)))
  for xx in range(x0+2,x1-2):
   if prof[xx]<=prof[xx-1] and prof[xx]<prof[xx+1]:cand.setdefault(xx,5*prof[xx]/max(1,y1-y0))
  xx=np.array(sorted(cand));cost=np.array([cand[q] for q in xx]);xx=xx[(xx>=x0)&(xx<=x1)];cost=np.array([cand[q] for q in xx]);N=len(xx)
  def letters(t):return sum(.45 if c in 'ilI1.,:;' else .65 if c in 'ftjr' else 1.35 if c in 'mwMW' else 1.0 for c in t)
  lens=np.array([letters(t) for t in tokens]);unit=(x1-x0)/(lens.sum()+1.3*(n-1));want=(lens+1.3)*unit;want[[0,-1]]-=.65*unit
  dp=np.full((n+1,N),np.inf);prev=np.full((n+1,N),-1,int);dp[0,0]=0
  for j in range(1,n+1):
   for q in range(1,N):
    widths=xx[q]-xx[:q];sigma=.35*want[j-1]+.6*unit
    losses=dp[j-1,:q]+.5*((widths-want[j-1])/sigma)**2+cost[q]
    losses[widths<.2*unit]=np.inf
    k=int(np.argmin(losses));dp[j,q]=losses[k];prev[j,q]=k
  at=N-1;cuts=[int(xx[at])]
  for j in range(n,0,-1):at=prev[j,at];cuts.append(int(xx[at]))
  cuts=cuts[::-1]
  if key in ann and 'cuts' in ann[key]:cuts=ann[key]['cuts']
  assert len(cuts)==n+1,(key,n,cuts)
  cb=SEG.CoreBand(mask);xh=int(cb[1]-cb[0]) if cb else max(3,y1-y0)
  if key in ann:
   for l,t,rt,bt in ann[key].get('mask_exclude_rects',[]):mask[t:bt,l:rt]=False
  rows=[]
  for j,(token,l,h) in enumerate(zip(r['text'].split(),cuts[:-1],cuts[1:])):
   m=mask[:,l:h];box=bounds(m);xx,yy,xx1,yy1=box
   rows.append({'writer':r['writer'],'kind':kind,'line':r['line'],'key':key,'word_index':j,'token':token,'x0':l+xx,'x1':l+xx1,'y0':yy,'y1':yy1,'xh':xh,'source_file':str(p.relative_to(ROOT)),'mask_box':box})
  for row in rows:
   overrides=ann.get(key,{}).get('word_overrides',{}).get(str(row['word_index']),{});row.update(overrides)
   reason=ann.get(key,{}).get('exclude_words',{}).get(str(row['word_index']))
   if reason:row['excluded']=reason
  out.extend(rows)
(B/'word_boxes.json').write_text(json.dumps(out,indent=2))
# Word crops make boundary errors visible without calculating any similarity.
for w in ['owen','yeukita','dylan','thiya','abhinav']:
 rows=[r for r in out if r['writer']==w];sheet=Image.new('RGB',(1600,((len(rows)+7)//8)*130),'white');draw=ImageDraw.Draw(sheet)
 for j,r in enumerate(rows):
  a=np.asarray(Image.open(ROOT/r['source_file']).convert('L'));im=Image.fromarray(a[r['y0']:r['y1'],r['x0']:r['x1']]).convert('RGB');im.thumbnail((190,85));x=j%8*200;y=j//8*130;sheet.paste(im,(x+3,y+38));draw.text((x+3,y+2),f"{r['kind'][:1]}{r['line']}.{r['word_index']} {r['token']}",fill='black')
 sheet.save(B/f'words_{w}.jpg')
print('Words',len(out),'lines',len(M['original'])+len(M['physical']))
print('x-heights',[(r['key'],r['xh']) for r in out if r['word_index']==0])
