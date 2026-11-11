from pathlib import Path
import collections, csv, hashlib, json, shutil, sys, time
import cv2
import numpy as np
from scipy.spatial import cKDTree

B=Path(__file__).resolve().parent
ROOT=B.parents[1]
OLD=ROOT/'Report/GeometricFidelity_AllAuthors_2026-11-02'
for d in ['inputs','source','figures']:(B/d).mkdir(exist_ok=True)
if not (B/'accepted_candidates.json').exists():
    for f in ['geometry_core.py','fast_geometry.py','accepted_candidates.json']:
        shutil.copy2(OLD/f,B/f)
    shutil.copy2(OLD/'source/segment.py',B/'source/segment.py')
    shutil.copy2(OLD/'results.json',B/'prior_robot_results.json')
    c=json.loads((B/'accepted_candidates.json').read_text())
    for r in c['references']+c['physical']:
        shutil.copy2(OLD/'inputs'/f"{r['id']}.npz",B/'inputs'/f"{r['id']}.npz")
from geometry_core import compare
from fast_geometry import FastGeometry
x=json.loads((B/'accepted_candidates.json').read_text());W=x['writers']
prior=json.loads((B/'prior_robot_results.json').read_text())['results']
tested={r['id'] for r in prior}
physical=[r for r in x['physical'] if r['id'] in tested]

def load(uid):return np.load(B/'inputs'/f'{uid}.npz')['mask']
def norm(a):
    y,z=np.where(a);a=a[y.min():y.max()+1,z.min():z.max()+1]
    return cv2.resize(a.astype('uint8'),(max(1,round(a.shape[1]*48/a.shape[0])),48),interpolation=cv2.INTER_NEAREST).astype(bool)

# Exact grayscale duplicates cannot serve as independent natural repeats.
seen={};duplicates=[]
for r in x['references']:
    g=np.load(B/'inputs'/f"{r['id']}.npz")['gray'];key=(r['writer'],r['token'],g.shape,hashlib.sha256(g.tobytes()).hexdigest())
    if key in seen:duplicates.append((seen[key],r['id']))
    seen[key]=r['id']
assert not duplicates,duplicates

natural_pairs=[];robot_pairs=[];folds=[];word_results=[];excluded=[];checks=[];start=time.time()
for w in W:
    for token in sorted({r['token'] for r in physical if r['writer']==w}):
        refs=[r for r in x['references'] if r['writer']==w and r['token']==token]
        ps=[r for r in physical if r['writer']==w and r['token']==token]
        if len(refs)<2:
            excluded.extend(dict(physical=r['id'],writer=w,token=token,reason='Only one genuine reference occurrence',reference_pages=1) for r in ps)
            continue
        arrays={r['id']:load(r['id']) for r in refs+ps}
        diags={uid:norm(a) for uid,a in arrays.items()}
        scorers={'line':FastGeometry(arrays),'word':FastGeometry(diags)}
        scores={}
        # Every distinct ordered original pair, including same-page indicators.
        for probe in refs:
            for ref in refs:
                if ref['id']==probe['id']:continue
                for method,scorer in scorers.items():
                    v=scorer.score(ref['id'],probe['id']);scores[method,probe['id'],ref['id']]=v['f1']
                    natural_pairs.append(dict(writer=w,token=token,method=method,probe=probe['id'],reference=ref['id'],probe_page=probe['page'],reference_page=ref['page'],cross_page=probe['page']!=ref['page'],F=v['f1']))
        for p in ps:
            for ref in refs:
                for method,scorer in scorers.items():
                    v=scorer.score(ref['id'],p['id']);scores[method,p['id'],ref['id']]=v['f1']
                    robot_pairs.append(dict(physical=p['id'],writer=w,token=token,method=method,reference=ref['id'],F=v['f1']))
        for mode in ['cross_page','leave_one_occurrence_out']:
            for p in ps:
                local=[]
                for probe in refs:
                    bank=[q for q in refs if q['id']!=probe['id'] and (mode!='cross_page' or q['page']!=probe['page'])]
                    if not bank:continue
                    for method in scorers:
                        nref=max(bank,key=lambda q:scores[method,probe['id'],q['id']])
                        rref=max(bank,key=lambda q:scores[method,p['id'],q['id']])
                        f=dict(mode=mode,method=method,writer=w,token=token,physical=p['id'],genuine_probe=probe['id'],bank_ids=[q['id'] for q in bank],bank_size=len(bank),natural_selected_reference=nref['id'],robot_selected_reference=rref['id'],natural_best_F=scores[method,probe['id'],nref['id']],robot_best_F=scores[method,p['id'],rref['id']],natural_pair_mean_F=float(np.mean([scores[method,probe['id'],q['id']] for q in bank])),robot_pair_mean_F=float(np.mean([scores[method,p['id'],q['id']] for q in bank])))
                        folds.append(f);local.append(f)
                if not local:
                    if mode=='cross_page':excluded.append(dict(physical=p['id'],writer=w,token=token,reason='Genuine repeats are only on one source page',reference_pages=len(set(q['page'] for q in refs))))
                    continue
                for method in scorers:
                    ff=[f for f in local if f['method']==method]
                    q=dict(mode=mode,method=method,writer=w,token=token,physical=p['id'],genuine_probes=len(ff),reference_pages=len(set(q['page'] for q in refs)))
                    for field in ['natural_best_F','robot_best_F','natural_pair_mean_F','robot_pair_mean_F']:
                        q[field]=float(np.mean([f[field] for f in ff]))
                    q['best_score_ratio']=q['robot_best_F']/q['natural_best_F'] if q['natural_best_F'] else None
                    q['pair_mean_score_ratio']=q['robot_pair_mean_F']/q['natural_pair_mean_F'] if q['natural_pair_mean_F'] else None
                    word_results.append(q)
        # Full raster + independent point-distance checks on genuine pairs, selected
        # independently of score. Every token bank supplies its first cross-page pair
        # when available, otherwise its first distinct occurrence pair.
        probe=refs[0];ref=next((q for q in refs if q['page']!=probe['page']),refs[1])
        for method,masks in [('line',arrays),('word',diags)]:
            s,m=compare(masks[ref['id']],masks[probe['id']],True)
            cached=scores[method,probe['id'],ref['id']];assert abs(cached-s['f1_0.10'])<1e-12
            a=np.argwhere(m['original_skeleton']);b=np.argwhere(m['physical_skeleton'])
            prec=float((cKDTree(a).query(b)[0]<=4.8).mean());rec=float((cKDTree(b).query(a)[0]<=4.8).mean());f=2*prec*rec/(prec+rec) if prec+rec else 0.
            assert abs(cached-f)<1e-12
            checks.append(dict(method=method,probe=probe['id'],reference=ref['id'],max_difference=max(abs(cached-f),abs(cached-s['f1_0.10']))))
    print(w,'words',sum(r['writer']==w and r['mode']=='cross_page' and r['method']=='word' for r in word_results),'seconds',round(time.time()-start,1),flush=True)

summary={};aggregates={}
for mode in ['cross_page','leave_one_occurrence_out']:
    summary[mode]={};aggregates[mode]={}
    for method in ['line','word']:
        summary[mode][method]={}
        for w in W:
            rr=[r for r in word_results if r['mode']==mode and r['method']==method and r['writer']==w]
            if not rr:continue
            z=dict(physical_words=len(rr),word_types=len(set(r['token'] for r in rr)))
            for field in ['natural_best_F','robot_best_F','natural_pair_mean_F','robot_pair_mean_F']:
                z[field]=float(np.mean([r[field] for r in rr]))
            z['best_score_ratio']=z['robot_best_F']/z['natural_best_F']
            z['pair_mean_score_ratio']=z['robot_pair_mean_F']/z['natural_pair_mean_F']
            summary[mode][method][w]=z
        ss=list(summary[mode][method].values());agg={field:float(np.mean([z[field] for z in ss])) for field in ['natural_best_F','robot_best_F','natural_pair_mean_F','robot_pair_mean_F']}
        agg.update(writers=len(ss),physical_words=sum(z['physical_words'] for z in ss),word_types=sum(z['word_types'] for z in ss))
        agg['ratio_of_writer_balanced_best_means']=agg['robot_best_F']/agg['natural_best_F']
        agg['ratio_of_writer_balanced_pair_means']=agg['robot_pair_mean_F']/agg['natural_pair_mean_F']
        agg['mean_of_writer_best_ratios']=float(np.mean([z['best_score_ratio'] for z in ss]))
        aggregates[mode][method]=agg
out=dict(summary=summary,aggregates=aggregates,words=word_results,folds=folds,original_original_pairs=natural_pairs,robot_original_pairs=robot_pairs,excluded=excluded,validation=dict(genuine_pair_checks=len(checks),max_difference=max(r['max_difference'] for r in checks),identical_gray_duplicates=duplicates,neural_modules_loaded=[k for k in ['torch','recognise','train'] if k in sys.modules]),input_sha256={f:hashlib.sha256((B/f).read_bytes()).hexdigest() for f in ['accepted_candidates.json','prior_robot_results.json','PROTOCOL.md']},runtime_seconds=time.time()-start)
(B/'results.json').write_text(json.dumps(out,indent=2));(B/'summary.json').write_text(json.dumps(dict(summary=summary,aggregates=aggregates),indent=2))
for name,rr in [('word_baselines',word_results),('matched_folds',folds),('original_original_pairs',natural_pairs),('robot_original_pairs',robot_pairs),('excluded_words',excluded)]:
    with (B/f'{name}.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader()
        for r in rr:writer.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()})
print(json.dumps(aggregates,indent=2))
