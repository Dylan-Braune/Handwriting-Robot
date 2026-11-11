from pathlib import Path
import sys,json,io,zlib,pickle,re,hashlib,collections
import numpy as np,cv2
from scipy.ndimage import gaussian_filter1d
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;ROOT=B.parents[1];OLD=ROOT/'Report/GeometricFidelity_2026-11-02';sys.path.insert(0,str(ROOT/'Software/CleanPipeline3'));import segment as SEG
for n in ['glyphs','line_snapshots','review']: (B/n).mkdir(exist_ok=True)
WRITERS=['dylan','yeukita','owen','abhinav','150'];raw_cache={};source_rows=[];evals=[];refs=[];audit=[]
# Import only the old preprocessing definitions, without running its scoring loop.
env={'__file__':str(OLD/'prepare_pairs.py'),'__name__':'old_preparation_functions'};exec((OLD/'prepare_pairs.py').read_text().split('words=[];chars=[];')[0],env)
mask_old=env['mask_for']
def bounds(m):
 y,x=np.where(m);return int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)
def clean(a):
 m=SEG.BinarizeLine(a,stripRules=True,closeGaps=False);band=SEG.CoreBand(m)
 if band:
  t,b=band;xh=b-t;nn,lab,stats,cents=cv2.connectedComponentsWithStats(m.astype('uint8'),8)
  for i in range(1,nn):
   x,y,w,h,ar=stats[i];cy=cents[i,1]
   if cy<t-.75*xh or cy>b+.75*xh or (h>3*xh and w<.12*xh):m[lab==i]=False
 return m

def word_cuts(m,tokens):
 x0,y0,x1,y1=bounds(m);n=len(tokens)
 if n==1:return [x0,x1]
 p=m.any(0);v=np.diff(np.r_[True,p,True].astype(int));gs=np.where(v==-1)[0];ge=np.where(v==1)[0];cand={x0:0.,x1:0.}
 for s,e in zip(gs,ge):
  if s>0 and e<len(p):cand[(int(s)+int(e))//2]=-2*np.log1p((e-s)/max(3,(y1-y0)/2))
 prof=gaussian_filter1d(m.sum(0).astype(float),1.)
 for x in range(x0+2,x1-2):
  if prof[x]<=prof[x-1] and prof[x]<prof[x+1]:cand.setdefault(x,5*prof[x]/max(1,y1-y0))
 xs=np.array(sorted(q for q in cand if x0<=q<=x1));cost=np.array([cand[q] for q in xs]);N=len(xs)
 weight=lambda t:sum(.45 if c in 'ilI1.,:;' else .65 if c in 'ftjr' else 1.35 if c in 'mwMW' else 1. for c in t)
 lens=np.array([weight(t) for t in tokens]);unit=(x1-x0)/(lens.sum()+1.3*(n-1));want=(lens+1.3)*unit;want[[0,-1]]-=.65*unit
 dp=np.full((n+1,N),np.inf);pre=np.full((n+1,N),-1,int);dp[0,0]=0
 for j in range(1,n+1):
  for q in range(1,N):
   wid=xs[q]-xs[:q];loss=dp[j-1,:q]+.5*((wid-want[j-1])/(.35*want[j-1]+.6*unit))**2+cost[q];loss[wid<.2*unit]=np.inf;k=int(loss.argmin());dp[j,q]=loss[k];pre[j,q]=k
 if not np.isfinite(dp[n,-1]):return None
 at=N-1;cuts=[int(xs[at])]
 for j in range(n,0,-1):at=pre[j,at];cuts.append(int(xs[at]))
 return cuts[::-1]

def spans(m):
 d=np.diff(np.r_[False,m.any(0),False].astype(int));return list(zip(np.where(d==1)[0],np.where(d==-1)[0]))

def add_glyphs(w,kind,token,m,gray,xh,meta,edges=False):
 if not token.isalpha() or not m.any():return
 ss=spans(m);items=[]
 if len(ss)==len(token):items=[(j,s,'count_agreement') for j,s in enumerate(ss)]
 elif edges and len(ss)>1:items=[(0,ss[0],'edge_proposal'),(len(token)-1,ss[-1],'edge_proposal')]
 target=evals if kind=='physical' else refs
 for j,(l,r),method in items:
  a=m[:,l:r];yy=np.where(a.any(1))[0]
  if not len(yy):continue
  y0=int(yy.min());y1=int(yy.max()+1);a=a[y0:y1];g=gray[y0:y1,l:r]
  if a.sum()<6:continue
  uid=f'{"E" if kind=="physical" else "R"}{len(target):05}';scale=48/xh;nm=cv2.resize(a.astype('uint8'),(max(1,round(a.shape[1]*scale)),max(1,round(a.shape[0]*scale))),interpolation=cv2.INTER_NEAREST).astype(bool)
  np.savez_compressed(B/'glyphs'/f'{uid}.npz',mask=nm,native=a,gray=g)
  target.append(dict(id=uid,writer=w,char=token[j],word=token,char_index=j,method=method,xh=xh,**meta))

def from_old(r,kind,edges):
 if r.get('excluded') or r['writer'] not in WRITERS:return
 a=np.asarray(Image.open(ROOT/r['source_file']).convert('L'));m=mask_old(r);g=a[r['y0']:r['y1'],r['x0']:r['x1']]
 add_glyphs(r['writer'],kind,r['token'],m,g,r['xh'],dict(page=r['key'],line=r['line'],word_index=r['word_index'],source=r['source_file'],word_box=[r['x0'],r['y0'],r['x1'],r['y1']]),edges)
# Physical observations remain the source-text blocks, to preserve the prior test population.
old_rows=json.loads((OLD/'word_boxes.json').read_text())
for r in old_rows:
 if r['kind']=='physical':from_old(r,'physical',True)
# IAM 150 source-text physical lines.
paired=json.loads((ROOT/'Report/ReportWorkingArchive_2026-11-01_VisualTests/ResultsEvidence/measurements/outputs/blocks/lines.json').read_text())
for r in paired:
 if r['writer']!='150' or r['block']!='C':continue
 p=ROOT/f"Software/CleanPipeline3/ReportResults/SegmentedReproducedPageCrops/{r['page']}/line_{r['line']:02}.png";a=np.asarray(Image.open(p).convert('L'));m=clean(a);tokens=r['want'].split();cuts=word_cuts(m,tokens);band=SEG.CoreBand(m);xh=band[1]-band[0]
 for j,(t,l,rt) in enumerate(zip(tokens,cuts[:-1],cuts[1:])):add_glyphs('150','physical',t,m[:,l:rt],a[:,l:rt],xh,dict(page=r['page'],line=r['line'],word_index=j,source=str(p.relative_to(ROOT)),word_box=[l,0,rt,a.shape[0]]),True)
target_letters={r['char'] for r in evals};print('Physical proposals',len(evals),collections.Counter(r['writer'] for r in evals),flush=True)
# Supplement the bank with edge proposals from the already reviewed original source passages.
for r in old_rows:
 if r['kind']=='original' and r['writer'] in WRITERS:from_old(r,'reference',True)

def decode(raw):
 try:return np.load(io.BytesIO(zlib.decompress(raw)),allow_pickle=False)
 except Exception:return np.asarray(Image.open(io.BytesIO(raw)).convert('L'))
# All available pages contribute candidate references. Retain only completely separable short words.
cache=pickle.loads((ROOT/'Software/CNN/NOGIT/personal_lines_cache.pkl').read_bytes())
for r in cache['samples']:
 w=r['page_key'].split('/')[0]
 if w not in WRITERS:continue
 page=r['page_key'].split('/')[1].rsplit('_',1)[0];lab=ROOT/f'Software/CNN/PersonalDataset/{w.capitalize()}/Labels/{page}_labels.txt';labels=lab.read_text(encoding='utf-8-sig').splitlines();i=int(r['line_idx'])
 if i>=len(labels) or labels[i]!=r['text']:audit.append(dict(writer=w,page=page,line=i,reason='cache label differs from current label'));continue
 if not any(t.isalpha() and len(t)<=4 for t in r['text'].split()):continue
 source_rows.append(dict(writer=w,page=page,line=i,text=r['text'],image=decode(r['imageBytes']),source=str(lab.relative_to(ROOT))))
# IAM cache contains native line images and labels. Match to the current page transcript.
iam=pickle.loads((ROOT/'Software/CNN/NOGIT/qtp_cache/iam_lines.pkl').read_bytes())
for r in iam:
 if r['author']!='150':continue
 page=Path(r['page_key']).stem;lab=ROOT/f'Data/Datasets/IAMpages10/150/{page}_labels.txt';labels=[t.strip() for t in lab.read_text().splitlines() if t.strip()];i=int(r['line_idx'])
 if i>=len(labels) or labels[i]!=r['text'].strip():audit.append(dict(writer='150',page=page,line=i,reason='IAM cache label differs from current label'));continue
 source_rows.append(dict(writer='150',page=page,line=i,text=r['text'],image=np.asarray(Image.open(io.BytesIO(r['png'])).convert('L')),source=str(lab.relative_to(ROOT))))
for ii,r in enumerate(source_rows):
 a=r['image'];m=clean(a)
 if not m.any():continue
 tokens=r['text'].split();cuts=word_cuts(m,tokens);band=SEG.CoreBand(m)
 if cuts is None or band is None:continue
 xh=band[1]-band[0]
 snap=B/'line_snapshots'/f"{r['writer']}_{r['page']}_{r['line']:02}.png";Image.fromarray(a).save(snap)
 for j,(t,l,rt) in enumerate(zip(tokens,cuts[:-1],cuts[1:])):
  if not(t.isalpha() and len(t)<=4):continue
  add_glyphs(r['writer'],'reference',t,m[:,l:rt],a[:,l:rt],xh,dict(page=r['page'],line=r['line'],word_index=j,source=str(snap.relative_to(ROOT)),word_box=[l,0,rt,a.shape[0]]),False)
 if ii%50==0:print('Reference lines',ii,'/',len(source_rows),'candidate glyphs',len(refs),flush=True)
# Equal maximum bank size, page-diverse order. No image score is used for bank selection.
bank=[]
for w in WRITERS:
 for c in sorted(target_letters):
  rr=[r for r in refs if r['writer']==w and r['char']==c];rr.sort(key=lambda r:(r['page'],r['line'],r['word_index'],r['char_index']));chosen=[];pages=set()
  for r in rr:
   if r['page'] not in pages:chosen.append(r);pages.add(r['page'])
   if len(chosen)==8:break
  for r in rr:
   if len(chosen)>=8:break
   if r not in chosen:chosen.append(r)
  bank.extend(chosen)
(B/'candidates.json').write_text(json.dumps(dict(physical=evals,references=bank,reference_candidate_count=len(refs),source_line_count=len(source_rows),exclusions=audit),indent=2))
# Contact sheets are for identity/separability review before scores are inspected.
for kind,rows in [('physical',evals),('reference',bank)]:
 for w in WRITERS:
  rr=[r for r in rows if r['writer']==w]
  for chunk in range(0,len(rr),100):
   part=rr[chunk:chunk+100];sheet=Image.new('RGB',(1600,math.ceil(len(part)/16)*130),'white') if False else Image.new('RGB',(1600,((len(part)+15)//16)*130),'white');draw=ImageDraw.Draw(sheet)
   for k,r in enumerate(part):
    d=np.load(B/'glyphs'/f"{r['id']}.npz");im=Image.fromarray(np.where(d['native'],0,255).astype('uint8')).convert('RGB');im.thumbnail((88,80));x=k%16*100;y=k//16*130;sheet.paste(im,(x+4,y+38));draw.text((x+2,y+2),f"{r['id']} {r['char']}",fill='black');draw.text((x+2,y+17),r['word'][:11],fill='black')
   sheet.save(B/'review'/f'{kind}_{w}_{chunk//100:02}.jpg')
print('Selected bank',len(bank),'from',len(refs),'glyphs. Source lines',len(source_rows));print('Counts',collections.Counter(r['writer'] for r in bank))
