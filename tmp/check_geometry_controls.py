import json
from pathlib import Path
x=json.loads(Path('Report/GeometricFidelity_2026-11-02/raw_results.json').read_text());c=x['character'];q=x['controls'];out={}
for w in ['owen','yeukita','dylan','abhinav']:
 a=[r for r in c if r['writer']==w];groups={}
 for r in a:groups.setdefault(r['word_id'],[]).append(r['f1_0.10'])
 out[w]={'n':len(a),'words':len(groups),'tokens':list(groups),'scores':[round(100*r['f1_0.10'],2) for r in a]}
 for typ in ['same_writer_wrong_character','other_writer_same_character']:
  qq=[r for r in q if r['writer']==w and r['type']==typ];by={}
  for r in qq:by.setdefault(r['id'],[]).append(r['f1_0.10'])
  vals=[sum(v)/len(v) for v in by.values()];true=[r['f1_0.10'] for r in a if r['id'] in by]
  out[w][typ]={'n_characters':len(vals),'control_mean':100*sum(vals)/len(vals) if vals else None,'matched_true_mean':100*sum(true)/len(true) if true else None}
print(json.dumps(out,indent=2))
