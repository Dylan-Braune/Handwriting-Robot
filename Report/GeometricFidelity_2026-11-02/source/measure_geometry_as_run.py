from pathlib import Path
import sys,json,hashlib,time,csv
import numpy as np
from scipy.ndimage import distance_transform_edt,shift
ROOT=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'Software/CleanPipeline3'));import segment as SEG
manifest=json.loads((B/'pair_manifest.json').read_text());TOLS=[.05,.10,.15,.20]
def canvas(a,b):
 h=max(a.shape[0],b.shape[0])+50;w=max(a.shape[1],b.shape[1])+50;out=[]
 for m in [a,b]:
  z=np.zeros((h,w),bool);y=(h-m.shape[0])//2;x=(w-m.shape[1])//2;z[y:y+m.shape[0],x:x+m.shape[1]]=m;out.append(z)
 return out

def compare(a,b,return_masks=False):
 a,b=canvas(a,b);sa=SEG.Skeletonize(a);sb=SEG.Skeletonize(b)
 ya,xa=np.where(sa);yb,xb=np.where(sb)
 if not(len(xa) and len(xb)):return {'error':'empty skeleton'}
 da=distance_transform_edt(~sa);db=distance_transform_edt(~sb)
 best=(np.inf,0,0)
 for dy in range(-5,6):
  for dx in range(-5,6):
   dist=.5*(da[yb+dy,xb+dx].mean()+db[ya-dy,xa-dx].mean())
   if (dist,dy*dy+dx*dx,dy,dx)<(best[0],best[1]*best[1]+best[2]*best[2],best[1],best[2]):best=(float(dist),dy,dx)
 _,dy,dx=best;bb=np.roll(b,(dy,dx),axis=(0,1));bs=np.roll(sb,(dy,dx),axis=(0,1));dbs=distance_transform_edt(~bs)
 def overlap(x,y):
  inter=int((x&y).sum());union=int((x|y).sum());return inter/union if union else 0.,2*inter/(x.sum()+y.sum()) if x.sum()+y.sum() else 0.
 iou,dice=overlap(a,bb);an=da<=.085*48;bn=dbs<=.085*48;ni,nd=overlap(an,bn)
 dba=da[bs];dab=dbs[sa];out={'iou':float(iou),'dice':float(dice),'normalised_iou':float(ni),'normalised_dice':float(nd),'mean_distance_xh':best[0]/48,'dx':dx,'dy':dy,'original_points':len(xa),'physical_points':len(xb)}
 for t in TOLS:
  p=float((dba<=48*t).mean());r=float((dab<=48*t).mean());out[f'precision_{t:.2f}']=p;out[f'recall_{t:.2f}']=r;out[f'f1_{t:.2f}']=2*p*r/(p+r) if p+r else 0.
 vals=[overlap(a,dbs<=rad)[0] for rad in range(13)];k=int(np.argmax(vals));out.update(adaptive_iou=float(vals[k]),adaptive_radius=k)
 if return_masks:return out,dict(original=a,physical=bb,original_skeleton=sa,physical_skeleton=bs)
 return out

results={};start=time.time()
for kind,key in [('word','word_pairs'),('character','character_pairs')]:
 rows=[]
 for i,r in enumerate(manifest[key]):
  d=np.load(B/'masks'/f"{r['id']}.npz");res=compare(d['original'],d['physical']);rr={k:v for k,v in r.items() if k not in ['original','physical']};rr.update(res);rows.append(rr)
 results[kind]=rows;print(kind,len(rows),'in',round(time.time()-start,1),'s',flush=True)
# Each same-character wrong-writer comparison is included, not just the least similar.
controls=[];chars=manifest['character_pairs']
for i,r in enumerate(chars):
 src=np.load(B/'masks'/f"{r['id']}.npz")
 same=[q for q in chars if q['writer']!=r['writer'] and q['char']==r['char']]
 # One deterministic source occurrence per other writer, not the best or worst glyph.
 by={}
 for q in same:by.setdefault(q['writer'],q)
 wrong=next((chars[(i+j)%len(chars)] for j in range(1,len(chars)) if chars[(i+j)%len(chars)]['writer']==r['writer'] and chars[(i+j)%len(chars)]['char']!=r['char']),None)
 for typ,qq in [('other_writer_same_character',list(by.values())),('same_writer_wrong_character',[wrong] if wrong else [])]:
  for q in qq:
   dd=np.load(B/'masks'/f"{q['id']}.npz");c=compare(dd['original'],src['physical']);controls.append(dict(id=r['id'],writer=r['writer'],char=r['char'],type=typ,control_id=q['id'],control_writer=q['writer'],control_char=q['char'],**c))
results['controls']=controls
# Sanity check the identical-mask case.
r=manifest['character_pairs'][0];d=np.load(B/'masks'/f"{r['id']}.npz");identity=compare(d['original'],d['original']);assert identity['iou']==1 and identity['f1_0.10']==1 and identity['mean_distance_xh']==0
results['checks']={'identity':identity,'no_neural_modules_loaded':not any(k in sys.modules for k in ['recognise','train','trainlibrary','torch'])}
assert results['checks']['no_neural_modules_loaded']
results['inputs_sha256']={p:hashlib.sha256((B/p).read_bytes()).hexdigest() for p in ['PROTOCOL.md','input_manifest.json','word_annotations.json','word_boxes.json','pair_manifest.json']}
results['code_sha256']={p:hashlib.sha256((B/p).read_bytes()).hexdigest() for p in ['prepare_inputs.py','annotate_words.py','prepare_pairs.py','measure_geometry.py']}
(B/'raw_results.json').write_text(json.dumps(results,indent=2))
keys=['iou','normalised_iou','normalised_dice','adaptive_iou','f1_0.05','f1_0.10','f1_0.15','f1_0.20','mean_distance_xh']
summ={}
for kind in ['word','character']:
 ss={}
 for w in ['owen','yeukita','dylan','thiya','abhinav','pooled']:
  rs=[r for r in results[kind] if w=='pooled' or r['writer']==w]
  ss[w]={'n':len(rs),**{k:float(np.mean([r[k] for r in rs])) if rs else None for k in keys},'f1_010_at_least_85_count':sum(r['f1_0.10']>=.85 for r in rs)}
 summ[kind]=ss
for typ in ['other_writer_same_character','same_writer_wrong_character']:
 rr=[r for r in controls if r['type']==typ];summ[typ]={'n':len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in keys}}
(B/'summary.json').write_text(json.dumps(summ,indent=2));print(json.dumps(summ,indent=2));print('Seconds',time.time()-start)
