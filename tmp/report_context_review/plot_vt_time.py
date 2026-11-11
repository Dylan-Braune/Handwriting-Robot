from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
b=Path('Report/IntegratedReport_2026-11-01');o=b/'Figures/VisualTests';o.mkdir(exist_ok=True)
a=json.loads(Path('Report/DesignVisualTests_2026-11-01/data/motion_results.json').read_text())['T56']
fig,axs=plt.subplots(1,2,figsize=(8.2,3),gridspec_kw={'width_ratios':[1.3,1]},layout='constrained')
x=list(range(len(a)));d=[r['drawS'] for r in a];t=[r['travelS'] for r in a];w=[r['dwellS'] for r in a];names=[r['writer'].capitalize() for r in a]
axs[0].bar(x,d,color='#123b60',label='Drawing');axs[0].bar(x,t,bottom=d,color='#91b7d0',label='Pen-up travel');axs[0].bar(x,w,bottom=[u+v for u,v in zip(d,t)],color='#a6adb3',label='Pen dwell')
for j in x:axs[0].text(j,d[j]+t[j]+w[j]+.4,f'{d[j]+t[j]+w[j]:.2f} s',ha='center',fontsize=8)
axs[0].set(xticks=x,xticklabels=names,ylabel='Nominal budget / s',ylim=(0,40));axs[0].legend(loc='upper center',ncol=3,fontsize=7,frameon=False)
f=[100*w[j]/(d[j]+t[j]+w[j]) for j in x];axs[1].barh(names,f,color='#123b60');axs[1].set(xlim=(0,75),xlabel='Dwell share of nominal budget / %')
for j,z in enumerate(f):axs[1].text(z+1,j,f'{z:.1f}%',va='center',fontsize=8)
for ax in axs:ax.spines[['top','right']].set_visible(False)
fig.savefig(o/'T56_time_budget.pdf');plt.close(fig)
