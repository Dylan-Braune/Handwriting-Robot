from pathlib import Path
import collections, csv, hashlib, json, sys, time
import cv2
import numpy as np
from scipy.spatial import cKDTree
from geometry_core import compare
from fast_geometry import FastGeometry

B=Path(__file__).resolve().parent
t0=time.time()
x=json.loads((B/'accepted_candidates.json').read_text())
W=x['writers']

def heightnorm(a):
    y,z=np.where(a)
    a=a[y.min():y.max()+1,z.min():z.max()+1]
    return cv2.resize(a.astype('uint8'),(max(1,round(a.shape[1]*48/a.shape[0])),48),interpolation=cv2.INTER_NEAREST).astype(bool)

def load(uid):
    return np.load(B/'inputs'/f'{uid}.npz')['mask']

rows=[];pairs=[];unsupported=[];max_fast_diff=0.;max_tree_diff=0.;checked=0
for w in W:
    physical=[r for r in x['physical'] if r['writer']==w]
    for token in sorted({r['token'] for r in physical}):
        bank=[r for r in x['references'] if r['writer']==w and r['token']==token]
        tests=[r for r in physical if r['token']==token]
        if not bank:
            unsupported.extend(dict(**r,reason='No accepted exact same-writer reference word') for r in tests)
            continue
        arrays={r['id']:load(r['id']) for r in bank+tests}
        diags={uid:heightnorm(a) for uid,a in arrays.items()}
        primary=FastGeometry(arrays);secondary=FastGeometry(diags)
        for r in tests:
            values=[]
            for q in bank:
                a=primary.score(q['id'],r['id']);b=secondary.score(q['id'],r['id'])
                values.append((q,a,b))
                pairs.append(dict(physical=r['id'],reference=q['id'],writer=w,token=token,line_F=a['f1'],word_F=b['f1']))
            row=dict(id=r['id'],writer=w,token=token,page=r['page'],line=r['line'],word_index=r['word_index'],references=len(bank))
            for prefix,idx,masks in [('line',1,arrays),('word',2,diags)]:
                winning=max(values,key=lambda z:z[idx]['f1']);q,s=winning[0],winning[idx]
                full,m=compare(masks[q['id']],masks[r['id']],True)
                delta=abs(s['f1']-full['f1_0.10']);max_fast_diff=max(max_fast_diff,delta)
                assert delta<1e-12,(r['id'],prefix,s,full)
                assert (s['dx'],s['dy'])==(full['dx'],full['dy'])
                # Independent nearest-neighbour calculation on the registered point sets.
                aa=np.argwhere(m['original_skeleton']);bb=np.argwhere(m['physical_skeleton'])
                p=float((cKDTree(aa).query(bb)[0]<=4.8).mean());re=float((cKDTree(bb).query(aa)[0]<=4.8).mean())
                f=2*p*re/(p+re) if p+re else 0.
                delta=abs(f-s['f1']);max_tree_diff=max(max_tree_diff,delta)
                assert delta<1e-12,(r['id'],prefix,f,s)
                checked+=1
                row.update({f'{prefix}_reference':q['id'],f'{prefix}_F':s['f1'],f'{prefix}_P':s['precision'],f'{prefix}_R':s['recall'],f'{prefix}_dx':s['dx'],f'{prefix}_dy':s['dy'],f'{prefix}_IoU':full['iou'],f'{prefix}_F05':full['f1_0.05'],f'{prefix}_F15':full['f1_0.15'],f'{prefix}_original_points':s['original_points'],f'{prefix}_physical_points':s['physical_points']})
            rows.append(row)
    print(w,'scored',sum(r['writer']==w for r in rows),'unsupported',sum(r['writer']==w for r in unsupported),'seconds',round(time.time()-t0,1),flush=True)
summary={}
for w in W:
    rr=[r for r in rows if r['writer']==w]
    summary[w]={'eligible_proposals':sum(r['writer']==w for r in x['all_physical_proposals']),
                'accepted_physical':sum(r['writer']==w for r in x['physical']),
                'scored_words':len(rr),'unsupported_words':sum(r['writer']==w for r in unsupported)}
    for prefix in ['line','word']:
        vals=[r[prefix+'_F'] for r in rr]
        summary[w][prefix]={'mean_F':float(np.mean(vals)) if vals else None,'median_F':float(np.median(vals)) if vals else None,'at_least_85':sum(v>=.85 for v in vals),'mean_IoU':float(np.mean([r[prefix+'_IoU'] for r in rr])) if rr else None,'best':max(rr,key=lambda r:r[prefix+'_F'])['id'] if rr else None,'worst':min(rr,key=lambda r:r[prefix+'_F'])['id'] if rr else None}
aggregates={p:{'equal_writer_mean_F':float(np.mean([summary[w][p]['mean_F'] for w in W if summary[w][p]['mean_F'] is not None])), 'word_weighted_mean_F':float(np.mean([r[p+'_F'] for r in rows])),'word_count_at_least_85':sum(r[p+'_F']>=.85 for r in rows)} for p in ['line','word']}
out={'summary':summary,'aggregates':aggregates,'results':rows,'unsupported':unsupported,'all_reference_pairs':pairs,
     'validation':{'selected_pairs_checked_both_methods':checked,'max_cached_vs_raster_difference':max_fast_diff,'max_independent_tree_difference':max_tree_diff,'neural_modules_loaded':[k for k in ['torch','recognise','train'] if k in sys.modules]},
     'provenance':{'accepted_candidates_sha256':hashlib.sha256((B/'accepted_candidates.json').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((B/'PROTOCOL.md').read_bytes()).hexdigest(),'runtime_seconds':time.time()-t0}}
(B/'results.json').write_text(json.dumps(out,indent=2))
(B/'summary.json').write_text(json.dumps({'writers':summary,'aggregates':aggregates},indent=2))
for name,rr in [('word_scores',rows),('all_reference_comparisons',pairs),('unsupported_words',unsupported)]:
    if not rr:continue
    with (B/f'{name}.csv').open('w',newline='',encoding='utf8') as f:
        out=csv.DictWriter(f,fieldnames=list(rr[0]));out.writeheader();out.writerows(rr)
print(json.dumps({'scored':len(rows),'unsupported':len(unsupported),'aggregates':aggregates,'validation':out if False else 'all assertions passed'},indent=2))
