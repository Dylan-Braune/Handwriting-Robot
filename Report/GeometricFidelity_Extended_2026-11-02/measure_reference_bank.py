from pathlib import Path
import json,time,hashlib,csv,sys,collections
import numpy as np
from fast_geometry import FastGeometry
from geometry_core import compare
B=Path(__file__).resolve().parent;x=json.loads((B/'accepted_candidates.json').read_text());rows=x['physical']+x['references'];arrays={r['id']:np.load(B/'glyphs'/f"{r['id']}.npz")['mask'] for r in rows};print('Masks',len(arrays),'largest',max(a.shape[0] for a in arrays.values()),max(a.shape[1] for a in arrays.values()),flush=True)
t0=time.time();fast=FastGeometry(arrays);print('Cached',round(time.time()-t0,2),'s',flush=True)
banks={}
for r in x['references']:banks.setdefault((r['writer'],r['char']),[]).append(r)
# Independent comparison against the original complete raster scorer on diverse real pairs.
checks=[]
for i,e in enumerate(x['physical'][::7]):
 rr=banks.get((e['writer'],e['char']),[])
 if not rr:continue
 r=rr[i%len(rr)];a=fast.score(r['id'],e['id']);b=compare(arrays[r['id']],arrays[e['id']]);assert abs(a['f1']-b['f1_0.10'])<1e-12,(e['id'],r['id'],a,b);assert a['dx']==b['dx'] and a['dy']==b['dy'];checks.append(abs(a['f1']-b['f1_0.10']))
print('Full raster checks',len(checks),'max diff',max(checks),flush=True)
results=[];controls=[];all_scores=[];unsupported=[]
for i,e in enumerate(x['physical']):
 own=banks.get((e['writer'],e['char']),[])
 if not own:unsupported.append(e);continue
 own_scores=[]
 for r in own:
  score=fast.score(r['id'],e['id']);own_scores.append((r,score));all_scores.append(dict(physical=e['id'],reference=r['id'],writer=e['writer'],reference_writer=r['writer'],char=e['char'],**score))
 best=max(own_scores,key=lambda z:z[1]['f1']);r,score=best;results.append(dict(**e,selected_reference=r['id'],reference_count=len(own),**score))
 for w in ['dylan','yeukita','owen','abhinav','150']:
  if w==e['writer']:continue
  other=banks.get((w,e['char']),[]);k=min(len(own),len(other))
  if not k:continue
  ds=[]
  for q in other[:k]:
   v=fast.score(q['id'],e['id']);ds.append((q,v));all_scores.append(dict(physical=e['id'],reference=q['id'],writer=e['writer'],reference_writer=w,char=e['char'],**v))
  qb,sb=max(ds,key=lambda z:z[1]['f1']);oa,os=max(own_scores[:k],key=lambda z:z[1]['f1']);controls.append(dict(physical=e['id'],writer=e['writer'],char=e['char'],other_writer=w,bank_size=k,own_reference=oa['id'],other_reference=qb['id'],own_f1=os['f1'],other_f1=sb['f1'],difference=os['f1']-sb['f1']))
 if i%20==0:print('Scored',i,'/',len(x['physical']),round(time.time()-t0,1),'s',flush=True)
# Numerical metrics for each selected pair use the original full implementation.
ref_by_id={r['id']:r for r in x['references']}
for r in results:
 q=compare(arrays[r['selected_reference']],arrays[r['id']]);assert abs(r['f1']-q['f1_0.10'])<1e-12;r.update({k:q[k] for k in ['iou','normalised_iou','normalised_dice','adaptive_iou','f1_0.05','f1_0.15','f1_0.20']})
summary={};W=['dylan','yeukita','owen','abhinav','150']
for w in W:
 rr=[r for r in results if r['writer']==w];counts=collections.Counter(r['char'] for r in rr);means={c:float(np.mean([r['f1'] for r in rr if r['char']==c])) for c in counts};eligible=[c for c in means if counts[c]>=2];bestletter=max(eligible or list(means),key=lambda c:means[c]);best=max(rr,key=lambda r:r['f1']);cc=[q for q in controls if q['writer']==w];bycontrol={}
 for q in cc:bycontrol.setdefault(q['physical'],[]).append(q)
 summary[w]={'proposed':sum(r['writer']==w for r in x['physical']),'scored':len(rr),'letters':len(counts),'mean_f1':float(np.mean([r['f1'] for r in rr])),'mean_iou':float(np.mean([r['iou'] for r in rr])),'mean_normalised_iou':float(np.mean([r['normalised_iou'] for r in rr])),'best_pair':{'id':best['id'],'reference':best['selected_reference'],'char':best['char'],'f1':best['f1']},'best_letter_with_two_or_more_examples':{'char':bestletter,'n':counts[bestletter],'mean_f1':means[bestletter]},'letter_counts':dict(counts),'letter_means':means,'other_writer_controls':{'physical_characters':len(bycontrol),'own_mean':float(np.mean([np.mean([v['own_f1'] for v in q]) for q in bycontrol.values()])) if bycontrol else None,'other_mean':float(np.mean([np.mean([v['other_f1'] for v in q]) for q in bycontrol.values()])) if bycontrol else None}}
summary['aggregates']={'equally_weighted_writer_mean':float(np.mean([summary[w]['mean_f1'] for w in W])),'character_weighted_mean':float(np.mean([r['f1'] for r in results])),'best_pair_per_writer_selected_mean':float(np.mean([summary[w]['best_pair']['f1'] for w in W])),'best_letter_per_writer_selected_mean':float(np.mean([summary[w]['best_letter_with_two_or_more_examples']['mean_f1'] for w in W]))}
out={'results':results,'controls':controls,'unsupported':unsupported,'summary':summary,'validation':{'raster_comparison_cases':len(checks),'max_f1_difference':max(checks),'no_neural_modules_loaded':not any(k in sys.modules for k in ['recognise','train','torch'])},'inputs_sha256':{f:hashlib.sha256((B/f).read_bytes()).hexdigest() for f in ['PROTOCOL.md','candidates.json','quality_review.json','accepted_candidates.json']},'elapsed_seconds':time.time()-t0}
(B/'results.json').write_text(json.dumps(out,indent=2));(B/'summary.json').write_text(json.dumps(summary,indent=2))
for name,rr in [('all_reference_comparisons',all_scores),('per_character',results),('controls',controls)]:
 with (B/f'{name}.csv').open('w',newline='',encoding='utf8') as f:
  writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr)
print(json.dumps(summary,indent=2));print('Seconds',time.time()-t0)
