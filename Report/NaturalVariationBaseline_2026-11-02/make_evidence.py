from pathlib import Path
import csv, json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry_core import compare

B=Path(__file__).resolve().parent
x=json.loads((B/'results.json').read_text())
c=json.loads((B/'accepted_candidates.json').read_text())
W=c['writers'];labels={w:w if w.isdigit() else w.capitalize() for w in W}
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
F=B/'figures'

def load(uid):return np.load(B/'inputs'/f'{uid}.npz')
def norm(a):
    yy,xx=np.where(a);a=a[yy.min():yy.max()+1,xx.min():xx.max()+1]
    return cv2.resize(a.astype('uint8'),(max(1,round(a.shape[1]*48/a.shape[0])),48),interpolation=cv2.INTER_NEAREST).astype(bool)
def overlay(a,b):
    _,m=compare(norm(a),norm(b),True);aa=m['original_skeleton'];bb=m['physical_skeleton']
    rgb=np.ones((*aa.shape,3));rgb[aa]=[.87,.31,.15];rgb[bb]=[.08,.38,.75];rgb[aa&bb]=[.35,.15,.55]
    yy,xx=np.where(aa|bb)
    return rgb[yy.min()-4:yy.max()+5,xx.min()-4:xx.max()+5]

s=x['summary']['cross_page']['word']
fig,ax=plt.subplots(figsize=(11,5.3),layout='constrained');xx=np.arange(len(W));bw=.36
for i,(field,color,title) in enumerate([('natural_best_F','#677f66','Genuine to genuine'),('robot_best_F','#496f8a','Robot to genuine')]):
    vals=[100*s[w][field] for w in W];ax.bar(xx+(i-.5)*bw,vals,bw,color=color,label=title)
    for j,v in enumerate(vals):ax.text(xx[j]+(i-.5)*bw,v+1.2,f'{v:.1f}',ha='center',fontsize=9)
ax.set(xticks=xx,xticklabels=[labels[w]+f"\n(n={s[w]['physical_words']})" for w in W],ylim=(0,110),ylabel='Matched-bank mean centreline F-score / %')
ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True);ax.legend(ncol=2,loc='upper right')
fig.suptitle('Cross-page natural-writing baseline and physical robot output\nIdentical reference choices available to both probes in every matched comparison',fontsize=13)
for ext in ['png','pdf','svg']:fig.savefig(F/f'natural_vs_robot.{ext}',dpi=220)
plt.close(fig)

fig,ax=plt.subplots(figsize=(7,6.3),layout='constrained')
colors=plt.get_cmap('tab20')
for i,w in enumerate(W):
    rr=[r for r in x['words'] if r['mode']=='cross_page' and r['method']=='word' and r['writer']==w]
    ax.scatter([100*r['natural_best_F'] for r in rr],[100*r['robot_best_F'] for r in rr],label=labels[w],s=32,alpha=.8,color=colors(i))
ax.plot([0,100],[0,100],color='#555555',lw=1,label='Equal agreement')
ax.plot([0,100],[0,85],color='#a34f36',ls='--',lw=1,label='85% of natural baseline')
ax.set(xlim=(0,102),ylim=(0,102),xlabel='Genuine-to-genuine mean F / %',ylabel='Robot-to-genuine mean F / %')
ax.legend(loc='upper left',fontsize=8,ncol=2);ax.grid(alpha=.12)
fig.suptitle('Every physical word in the cross-page baseline comparison',fontsize=12)
for ext in ['png','pdf']:fig.savefig(F/f'word_scatter.{ext}',dpi=220)
plt.close(fig)

# One illustrative fold per writer, selected for closeness to the writer's two
# average scores, rather than to maximise the visual score or baseline ratio.
selected=[]
for w in W:
    ff=[f for f in x['folds'] if f['mode']=='cross_page' and f['method']=='word' and f['writer']==w]
    target=s[w]
    r=min(ff,key=lambda f:(f['natural_best_F']-target['natural_best_F'])**2+(f['robot_best_F']-target['robot_best_F'])**2)
    selected.append(r)
    fig,axes=plt.subplots(3,2,figsize=(8,6),layout='constrained')
    for j,(ref,probe,label,score) in enumerate([(r['natural_selected_reference'],r['genuine_probe'],'Genuine probe',r['natural_best_F']),(r['robot_selected_reference'],r['physical'],'Physical robot probe',r['robot_best_F'])]):
        a=load(ref);b=load(probe)
        for i,d in enumerate([a,b]):axes[i,j].imshow(d['gray'],cmap='gray',vmin=0,vmax=255);axes[i,j].axis('off')
        axes[0,j].set_title(f'Selected reference {ref}',fontsize=10)
        axes[1,j].set_title(label+' '+probe,fontsize=10)
        axes[2,j].imshow(overlay(a['mask'],b['mask']));axes[2,j].axis('off');axes[2,j].set_title(f'F {100*score:.2f}%',fontsize=12)
    fig.suptitle(f"{labels[w]}: {r['token']}\nBoth probes choose from the same {r['bank_size']}-reference bank",fontsize=13)
    for ext in ['png','pdf']:fig.savefig(F/f'comparison_{w}.{ext}',dpi=200)
    plt.close(fig)
(B/'illustrated_folds.json').write_text(json.dumps(selected,indent=2))

# Paired word-type cluster resampling, conditional on the fixed banks and writers.
rng=np.random.default_rng(23539195);boot={};word_boot=[];stats={}
for mode in ['cross_page','leave_one_occurrence_out']:
    boot[mode]={};stats[mode]={}
    for method in ['line','word']:
        samples={}
        for w in W:
            rr=[r for r in x['words'] if r['mode']==mode and r['method']==method and r['writer']==w]
            groups={}
            for r in rr:groups.setdefault(r['token'],[]).append(r)
            gg=list(groups.values());counts=np.array([len(g) for g in gg]);draw=rng.integers(0,len(gg),(10000,len(gg)));den=counts[draw].sum(1)
            fields={}
            for field in ['natural_best_F','robot_best_F','natural_pair_mean_F','robot_pair_mean_F']:
                sums=np.array([sum(r[field] for r in g) for g in gg]);fields[field]=sums[draw].sum(1)/den
            samples[w]=fields
        nat=np.mean([samples[w]['natural_best_F'] for w in W],axis=0);rob=np.mean([samples[w]['robot_best_F'] for w in W],axis=0)
        npair=np.mean([samples[w]['natural_pair_mean_F'] for w in W],axis=0);rpair=np.mean([samples[w]['robot_pair_mean_F'] for w in W],axis=0)
        stats[mode][method]={'natural_mean_95':np.percentile(nat,[2.5,97.5]).tolist(),'robot_mean_95':np.percentile(rob,[2.5,97.5]).tolist(),'relative_best_ratio_95':np.percentile(rob/nat,[2.5,97.5]).tolist(),'relative_pair_mean_ratio_95':np.percentile(rpair/npair,[2.5,97.5]).tolist()}
stats['scope']='Descriptive paired word-type cluster resampling, 10000 draws, fixed 11 writers and original reference banks. Excludes uncertainty from new pages, reference-bank selection and image segmentation.'
(B/'statistics.json').write_text(json.dumps(stats,indent=2))

with (B/'per_writer_summary.csv').open('w',newline='',encoding='utf8') as f:
    out=csv.writer(f);out.writerow(['mode','normalisation','writer','physical_words','word_types','natural_best_F_percent','robot_best_F_percent','relative_best_ratio_percent','natural_pair_mean_F_percent','robot_pair_mean_F_percent','relative_pair_mean_ratio_percent'])
    for mode,mm in x['summary'].items():
        for method,ww in mm.items():
            for w,z in ww.items():out.writerow([mode,method,w,z['physical_words'],z['word_types'],100*z['natural_best_F'],100*z['robot_best_F'],100*z['best_score_ratio'],100*z['natural_pair_mean_F'],100*z['robot_pair_mean_F'],100*z['pair_mean_score_ratio']])
print('Figures and matched baseline tables saved')
print('Cross-page word-height descriptive ranges',stats['cross_page']['word'])
