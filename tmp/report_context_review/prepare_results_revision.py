from pathlib import Path
import shutil,json,re,hashlib,math
import numpy as np
from collections import Counter,defaultdict
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'Report/DesignResultsDiscussion_2026-11-01'
if not B.exists():
    shutil.copytree(ROOT/'Report/DesignPolished_2026-11-01',B,ignore=shutil.ignore_patterns('design_preview.*','__pycache__'))
E=B/'ResultsEvidence';E.mkdir(exist_ok=True)
shutil.copytree(ROOT/'Report/FinalReport_v6/measurements',E/'measurements',dirs_exist_ok=True)
archive=E/'archive';archive.mkdir(exist_ok=True)
for n in ['qtp1a_results.json','qtp1b_results.json','qtp1a_real_features.json','qtp1a_synth_features.json']:
    shutil.copy2(ROOT/'Report/_dev/qtp_data'/n,archive/n)
for n in ['qtp1a_analyse.py','qtp1b_overlay.py','qtp_features.py','qtp_common.py']:
    shutil.copy2(ROOT/'Report/_dev/qtp_scripts'/n,archive/n)
for n in ['res_blocks.png','res_per_author.png','res_confusion_reproduced.png','res_connectedness.png','res_style_signature.png','res_confusion_holdout.png','res_gantry_geometry.png','qtp1_feature_scatter.png','qtp1_overlay_dists.png']:
    p=ROOT/'Report/FinalReport_v6/Figures'/n
    if p.exists():shutil.copy2(p,B/'Figures'/n)
proposal=ROOT/'ProjectProposal2026DylanFinalRev0 - Dylan Braune-1.pdf'
fields=PdfReader(proposal).get_fields()
E.joinpath('proposal_form_values.json').write_text(json.dumps({k:str(v.get('/V','')) for k,v in fields.items() if v.get('/V')},indent=2),encoding='utf-8')

def load(n):return json.loads((E/'measurements/outputs'/n).read_text())
rows=load('blocks/lines.json');hold=load('holdout_namer/lines.json')
def metrics(rr):
    chars=sum(r['refChars'] for r in rr);d=sum(r['dist'] for r in rr)
    words=sum(r['refWords'] for r in rr);wr=sum(r['wordsRight'] for r in rr)
    ok=sum(r['writer']==r['named'] for r in rr)
    return dict(n=len(rr),characters=chars,edits=d,char_accuracy=100*(1-d/chars),word_match=100*wr/words,named=ok,naming=100*ok/len(rr))
m=dict(reproduction=metrics(rows),blocks={k:metrics([r for r in rows if r['block']==k]) for k in 'ABC'},writers={k:metrics([r for r in rows if r['writer']==k]) for k in sorted({r['writer'] for r in rows})})
def wilson(k,n):
    z=1.96;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [100*(c-h),100*(c+h)]
m['holdout']=dict(n=len(hold),correct=sum(r['writer']==r['named'] for r in hold))
m['holdout']['wilson']=wilson(m['holdout']['correct'],len(hold))
# Writer-cluster resampling describes sensitivity to this eleven-writer set.
rng=np.random.default_rng(23539195)
wm=list(m['writers'].values());a=np.array([[r['characters'],r['edits']] for r in wm])
idx=rng.integers(0,len(wm),(10000,len(wm)));draw=a[idx].sum(axis=1)
m['reproduction']['writer_cluster_bootstrap_95']=np.percentile(100*(1-draw[:,1]/draw[:,0]),[2.5,97.5]).tolist()
recs=load('reader_paper_cnn_bilstm_ctc_scratch_best/results.json')
m['reader']=dict(lines=sum(r['lines'] for r in recs),characters=sum(r['n'] for r in recs),edits=sum(r['d'] for r in recs),word_match=100*sum(r['wr'] for r in recs)/sum(r['wn'] for r in recs))
m['reader']['char_accuracy']=100*(1-m['reader']['edits']/m['reader']['characters'])
# Check the archival correlation against the actual matrices, without rerunning synthesis.
st=json.loads((archive/'qtp1a_results.json').read_text());rr=np.array(st['real_held_means']);ss=np.array(st['synth_means'])
def corr(cols):
    x=rr[:,cols];y=ss[:,cols];x=(x-x.mean(0))/x.std(0);y=(y-y.mean(0))/y.std(0)
    return float(np.corrcoef(x.ravel(),y.ravel())[0,1])
shape=[i for i,f in enumerate(st['features']) if f not in ('stroke_w_xh','ink_density')]
m['archive_style']=dict(writers=st['writers'],synthetic_lines=sum(st['n_synth']),held_lines=sum(st['n_real_held']),all_nine=corr(list(range(9))),shape_seven=corr(shape),recorded=st['combined_correlation']['real_held'])
# Explicit terminal-status evidence, rather than an inference from elapsed time.
raw=(E/'measurements/raw/odroid_server_log.txt').read_text(encoding='utf-8',errors='replace').splitlines()
head=re.compile(r'^\s*timing for reproduce: ([\d.]+) s total, (\d+) characters,')
records=[];last=-1
for i,ln in enumerate(raw):
    mt=head.match(ln)
    if not mt:continue
    body='\n'.join(raw[i:i+10]);dm=re.search(r'drawing\s+([\d.]+) s',body)
    if not dm:continue
    prior='\n'.join(raw[last+1:i]);status='complete' if 'Program end (M2)' in prior else 'stopped' if 'G-code job STOPPED' in prior else 'unconfirmed'
    sy=re.search(r'synthesis\s+([\d.]+) s',body)
    records.append(dict(log_line=i+1,characters=int(mt[2]),total=float(mt[1]),drawing=float(dm[1]),synthesis=float(sy[1]) if sy else 0,status=status))
    last=i
confirmed=[r for r in records if r['status']=='complete' and r['characters']>0]
groups=defaultdict(list)
for r in confirmed:groups[r['characters']].append(r)
m['reproduction_timing']=dict(selection='Console records with explicit Program end (M2) and positive character count; no speed-based completion inference.',n=len(confirmed),passed=sum(r['total']<=4*r['characters'] for r in confirmed),groups={c:dict(n=len(g),worst=max(r['total'] for r in g),best=min(r['total'] for r in g)) for c,g in sorted(groups.items())},records=records)
m['hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [proposal,*sorted((E/'measurements/outputs').rglob('*.json')),archive/'qtp1a_results.json',archive/'qtp1b_results.json',E/'measurements/raw/odroid_server_log.txt']}
E.joinpath('REVIEWED_METRICS.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in m.items() if k not in ['hashes','writers','blocks','archive_style','reproduction_timing']},indent=2))
print('timings',m['reproduction_timing']['n'],m['reproduction_timing']['groups'])
print('style',m['archive_style']['all_nine'],m['archive_style']['shape_seven'])
