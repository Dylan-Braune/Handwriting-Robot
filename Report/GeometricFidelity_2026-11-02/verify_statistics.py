from pathlib import Path
import sys,json,hashlib,csv
import numpy as np
from scipy.spatial import cKDTree
B=Path(__file__).resolve().parent;env={'__file__':str(B/'measure_geometry.py'),'__name__':'geometry_functions'};exec((B/'measure_geometry.py').read_text().split('results={};start=time.time()')[0],env);compare=env['compare']
raw=json.loads((B/'raw_results.json').read_text());rng=np.random.default_rng(23539195);check=[]
for r in raw['character']:
 d=np.load(B/'masks'/f"{r['id']}.npz");rr,m=compare(d['original'],d['physical'],True);a=np.argwhere(m['original_skeleton']);b=np.argwhere(m['physical_skeleton']);ab=cKDTree(b).query(a)[0];ba=cKDTree(a).query(b)[0]
 p=float((ba<=4.8).mean());rec=float((ab<=4.8).mean());f=2*p*rec/(p+rec) if p+rec else 0.
 assert abs(f-r['f1_0.10'])<1e-12,r['id'];check.append(abs(f-r['f1_0.10']))
stats={}
for kind in ['word','character']:
 ss={}
 for w in ['owen','yeukita','dylan','thiya','abhinav','pooled']:
  rows=[r for r in raw[kind] if w=='pooled' or r['writer']==w]
  if not rows:ss[w]={'n':0};continue
  # Whole words are the sampling groups for the character subset.
  groups={}
  for r in rows:groups.setdefault(r['word_id'] if kind=='character' else r['id'],[]).append(r)
  gg=list(groups.values());boot=[]
  for _ in range(10000):
   sample=[q for i in rng.integers(0,len(gg),len(gg)) for q in gg[i]];boot.append(np.mean([q['f1_0.10'] for q in sample]))
  p=sum(r['precision_0.10']*r['physical_points'] for r in rows)/sum(r['physical_points'] for r in rows);rec=sum(r['recall_0.10']*r['original_points'] for r in rows)/sum(r['original_points'] for r in rows)
  st={'n':len(rows),'word_groups':len(groups),'mean_character_or_word_f1':float(np.mean([r['f1_0.10'] for r in rows])),'group_bootstrap_95':np.percentile(boot,[2.5,97.5]).tolist(),'point_pooled_precision':p,'point_pooled_recall':rec,'point_pooled_f1':2*p*rec/(p+rec)}
  for typ in (['same_writer_wrong_character','other_writer_same_character'] if kind=='character' else []):
   controls=[q for q in raw['controls'] if q['type']==typ and (w=='pooled' or q['writer']==w)];by={}
   for q in controls:by.setdefault(q['id'],[]).append(q['f1_0.10'])
   tr=[q['f1_0.10'] for q in rows if q['id'] in by]
   if by:st[typ]={'n_characters':len(by),'correct_matched_mean':float(np.mean(tr)),'control_mean':float(np.mean([np.mean(x) for x in by.values()]))}
  ss[w]=st
 stats[kind]=ss
stats['independent_reference_check']={'pairs':len(check),'method':'cKDTree nearest-neighbour distances compared with raster Euclidean distance transforms','max_f1_difference':max(check)}
(B/'statistics.json').write_text(json.dumps(stats,indent=2))
for kind in ['character','word','controls']:
 rows=raw[kind];keys=list(rows[0]);
 with (B/(kind+'_scores.csv')).open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
print(json.dumps(stats['character'],indent=2));print(stats['independent_reference_check'])
