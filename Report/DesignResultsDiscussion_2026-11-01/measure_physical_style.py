"""New retrospective style study on physical output and original validation crops.
No neural classifier, training, synthesis or hardware execution. Output goes only
to this report's ResultsEvidence. Reuses the archived descriptor definitions with
current segment.py primitives, explicitly recording this measurement revision.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys,json,types,importlib.util,pickle,io,hashlib,time,zlib
import numpy as np
from PIL import Image
from collections import defaultdict
B=Path(__file__).resolve().parent;ROOT=B.parents[1];CP=ROOT/'Software/CleanPipeline3';sys.path.insert(0,str(CP))
import segment as SEG
import extractiam as EX
import paths
E=B/'ResultsEvidence';OUT=E/'physical_style';OUT.mkdir(exist_ok=True)
stub=types.ModuleType('qtp_common');sys.modules['qtp_common']=stub
pio=types.ModuleType('ProfileIO')
for n in ['BinarizeLine','CoreBand','EstimateSlantDeg','Skeletonize']:setattr(pio,n,getattr(SEG,n))
sys.modules['ProfileIO']=pio
sg=types.ModuleType('SegmentPage');sg.LabelComponents=lambda mask,connectivity=8:SEG.LabelComponents(mask)
sys.modules['SegmentPage']=sg
spec=importlib.util.spec_from_file_location('frozen_features',E/'archive/qtp_features.py');Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
LABEL={'abhinav':'A','dylan':'B','owen':'O','thiya':'T','yeukita':'Y'}
writers=['150','152','153','384','552','588',*LABEL]
manifest={'measurement':'New offline physical-output style analysis, 1 November 2026','host':sys.platform,'feature_names':Q.FEATURES,'normalised_xheight_px':28,'pairing':'Original validation material and robot output grouped by writer; different text content, not matched sentences.','inputs':{},'code':{},'original':[],'physical':[],'exclusions':[]}
def hashfile(p):manifest['inputs'][str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
for p in [Path(__file__),CP/'segment.py',CP/'extractiam.py',E/'archive/qtp_features.py']:manifest['code'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
def measure(kind,w,arr,text,source,idx):
 f=Q.line_features(arr,text)
 if f is None:manifest['exclusions'].append(dict(kind=kind,writer=w,source=source,line=idx,reason='feature extraction returned no usable band'));return
 manifest[kind].append(dict(writer=w,source=source,line=idx,text=text,**{k:float(v) if np.isfinite(v) else None for k,v in f.items()}))
for w in writers[:6]:
 d=ROOT/'Data/Datasets/IAMpages10'/w
 pages=[p for p in sorted(d.glob('*.png')) if p.with_name(p.stem+'_labels.txt').exists()];p=pages[-1];hashfile(p);lp=p.with_name(p.stem+'_labels.txt');hashfile(lp)
 crops,_=EX.ExtractPage(p,verbose=False);texts=lp.read_text(encoding='utf-8').strip().splitlines();texts=[t.split('\t',1)[-1].strip() for t in texts if t.strip()]
 if len(crops)!=len(texts):manifest['exclusions'].append(dict(kind='original',writer=w,reason='crop/transcript count mismatch',crops=len(crops),labels=len(texts)));continue
 for i,(crop,text) in enumerate(zip(crops,texts)):measure('original',w,crop,text,str(p.relative_to(ROOT)),i)
 print('original',w,len(crops),flush=True)
cache=paths.Cache/'personal_lines_cache.pkl';hashfile(cache)
with cache.open('rb') as f:data=pickle.load(f)
def decode(raw):
 try:return np.load(io.BytesIO(zlib.decompress(raw)),allow_pickle=False)
 except Exception:return np.asarray(Image.open(io.BytesIO(raw)).convert('L'))
for w in LABEL:
 rr=[s for s in data['samples'] if s['page_key'].split('/')[0]==w and s.get('is_holdout')]
 for s in rr:measure('original',w,decode(s['imageBytes']),s['text'],s['page_key'],int(s['line_idx']))
 print('original',w,len(rr),flush=True)
paired=json.loads((E/'measurements/outputs/blocks/lines.json').read_text());pages=defaultdict(list)
for r in paired:pages[r['page']].append(r)
for page,rr in pages.items():
 p=CP/'ReportResults/FullReproducedPages'/(page+'.jpg');hashfile(p)
 gray=np.ascontiguousarray(np.rot90(np.asarray(Image.open(p).convert('L'))))
 crops,_,_=SEG.SegmentLines(gray)
 rr=sorted(rr,key=lambda r:r['line'])
 if len(crops)!=len(rr):
  manifest['exclusions'].append(dict(kind='physical',page=page,reason='crop/transcript count mismatch; no forced correspondence',crops=len(crops),labels=len(rr)));print('exclude physical',page,len(crops),len(rr),flush=True);continue
 for i,(crop,r) in enumerate(zip(crops,rr)):measure('physical',r['writer'],np.asarray(crop['image']),r['want'],str(p.relative_to(ROOT)),i)
 print('physical',page,len(crops),flush=True)
means={}
for kind in ['original','physical']:
 means[kind]={}
 for w in writers:
  rs=[r for r in manifest[kind] if r['writer']==w]
  means[kind][w]=[float(np.nanmean([r[f] if r[f] is not None else np.nan for r in rs])) if rs else None for f in Q.FEATURES]
use=[w for w in writers if all(v is not None for v in means['original'][w]+means['physical'][w])]
R=np.asarray([means['original'][w] for w in use],float);S=np.asarray([means['physical'][w] for w in use],float)
shape=[i for i,f in enumerate(Q.FEATURES) if f not in Q.PEN_FEATURES]
def pooled(cols,mode,rr=R,ss=S):
 a=rr[:,cols];b=ss[:,cols];mu=a.mean(0);sd=a.std(0);sd=np.where(sd<1e-12,1,sd);a=(a-mu)/sd
 if mode=='within':b=(b-b.mean(0))/np.where(b.std(0)<1e-12,1,b.std(0))
 else:b=(b-mu)/sd
 return float(np.corrcoef(a.ravel(),b.ravel())[0,1])
manifest['writer_means']=means;manifest['writers_analysed']=use
manifest['per_feature']={f:float(np.corrcoef(R[:,i],S[:,i])[0,1]) for i,f in enumerate(Q.FEATURES)}
manifest['correlations']={f'{name}_{mode}':pooled(cols,mode) for name,cols in [('all_nine',list(range(9))),('shape_seven',shape)] for mode in ['within','absolute']}
rng=np.random.default_rng(23539195);bs=[]
for _ in range(10000):
 ix=rng.integers(0,len(use),len(use));bs.append(pooled(shape,'within',R[ix],S[ix]))
manifest['shape_within_writer_resampling_95']=np.nanpercentile(bs,[2.5,97.5]).tolist()
manifest['counts']={kind:dict(__total__=len(manifest[kind]),**{w:sum(r['writer']==w for r in manifest[kind]) for w in writers}) for kind in ['original','physical']}
(OUT/'results.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:manifest[k] for k in ['counts','writers_analysed','correlations','per_feature','shape_within_writer_resampling_95','exclusions']},indent=2))
