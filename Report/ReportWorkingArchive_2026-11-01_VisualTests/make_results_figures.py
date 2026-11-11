"""Portable report figures and tables from preserved measurements; no model runs."""
from pathlib import Path
import json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

B=Path(__file__).resolve().parent; ROOT=B.parents[1];E=B/'ResultsEvidence';F=B/'Figures'
def load(n):return json.loads((E/'measurements/outputs'/n).read_text())
m=json.loads((E/'REVIEWED_METRICS.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.labelsize':9.5,'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42})
BLUE='#254e70';GREY='#a6adb4';LIGHT='#d7e1e8'
def save(fig,n):
    fig.savefig(F/(n+'.pdf'),bbox_inches='tight');plt.close(fig)
def clean(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',color='#dddddd',linewidth=.5);ax.set_axisbelow(True)
writers=load('blocks/per_author.json');writers=sorted(writers,key=lambda r:r['char'])
fig,ax=plt.subplots(figsize=(6.1,3.7),layout='constrained');y=np.arange(len(writers))
ax.barh(y+.16,[r['char'] for r in writers],height=.3,color=BLUE,label='Character accuracy')
ax.barh(y-.16,[r['id'] for r in writers],height=.3,color=GREY,label='Writer identification')
ax.axvline(85,color='black',ls='--',lw=.9,label='85% reference')
ax.set(yticks=y,yticklabels=[r['label'] for r in writers],xlim=(0,104),xlabel='Accuracy / %',ylabel='Writer');clean(ax)
ax.legend(loc='lower right',fontsize=8.5);save(fig,'results_writer_performance')

hold=load('holdout_namer/lines.json');names=['150','152','153','384','552','588','abhinav','dylan','owen','thiya','yeukita'];labels=['150','152','153','384','552','588','A','B','O','T','Y']
# Stored holdout predictions use both short labels and full writer identifiers.
alias=dict(zip(names,labels));alias.update(dict(zip(labels,labels)));cm=np.zeros((11,11),int)
for r in hold:cm[labels.index(alias[r['writer']]),labels.index(alias[r['named']])]+=1
fig,ax=plt.subplots(figsize=(5.1,4.5),layout='constrained');ax.imshow(cm,cmap='Blues',vmin=0,vmax=31)
for i in range(11):
 for j in range(11):
  if cm[i,j]:ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>15 else 'black',fontsize=9)
ax.set(xticks=range(11),xticklabels=labels,yticks=range(11),yticklabels=labels,xlabel='Predicted writer',ylabel='Reference writer');save(fig,'results_holdout_confusion')

tim=m['reproduction_timing']['groups'];xx=[int(c) for c in tim];worst=[tim[str(c)]['worst']/c for c in xx]
fig,ax=plt.subplots(figsize=(6.1,2.8),layout='constrained');ax.bar(range(len(xx)),worst,color=BLUE)
ax.axhline(4,color='black',ls='--',label='R3 limit: 4 s/character')
for i,c in enumerate(xx):ax.text(i,worst[i]+.05,'n='+str(tim[str(c)]['n']),ha='center',fontsize=9)
ax.set(xticks=range(len(xx)),xticklabels=xx,xlabel='Reported characters per completed job',ylabel='Worst total / s per character',ylim=(0,4.5));ax.legend(fontsize=9);ax.spines[['top','right']].set_visible(False);save(fig,'results_r3_confirmed')

fig,ax=plt.subplots(figsize=(6.1,2.8),layout='constrained');im=[16.14,16.81];net=[30.12,38.67];yy=np.arange(2)
ax.barh(yy,im,color=GREY,label='Image processing');ax.barh(yy,net,left=im,color=BLUE,label='Reader + writer classifier')
for i,(n,allow,total) in enumerate([(500,14.5,46.26),(982,26.55,55.49)]):
 ax.plot([allow,allow],[i-.42,i+.42],color='black',ls='--');ax.text(total+.4,i,f'{total:.2f} s',va='center',fontsize=9)
ax.set(yticks=yy,yticklabels=['500 characters','982 characters'],xlabel='Recorded classification time / s',xlim=(0,64));clean(ax);ax.legend(loc='lower right',fontsize=9)
save(fig,'results_r4_stages')

fig,ax=plt.subplots(figsize=(6.1,2.8),layout='constrained');errs=[.30,.20,.10,-.10,.30,56.2-math.hypot(40,40),56.9-math.hypot(40,40)]
ax.bar(range(7),errs,color=[BLUE]*4+[GREY]*3);ax.axhline(.5,color='black',ls='--');ax.axhline(-.5,color='black',ls='--');ax.axhline(0,color='black',lw=.5)
ax.set(xticks=range(7),xticklabels=['Sq. bottom','Sq. top','Sq. left','Sq. right','Tri. base','Tri. left','Tri. right'],ylabel='Measured length - nominal / mm',ylim=(-.6,.6));ax.tick_params(axis='x',labelrotation=22);ax.spines[['top','right']].set_visible(False);save(fig,'results_geometry_errors')

fig,axs=plt.subplots(4,1,figsize=(6.1,3.3),layout='constrained')
for ax,(p,label) in zip(axs,[('AuthorY','Y: 97.01%'),('AuthorO','O: 94.25%'),('AuthorD','B: 93.11%'),('AuthorT','T: 60.97%')]):
 src=ROOT/'Software/CleanPipeline3/ReportResults/SegmentedReproducedPageCrops'/p/'line_00.png'
 arr=np.asarray(Image.open(src).convert('L'));ax.imshow(arr,cmap='gray',vmin=0,vmax=255,aspect='equal');ax.axis('off');ax.text(-.02,.5,label,transform=ax.transAxes,ha='right',va='center',fontsize=9)
save(fig,'results_physical_examples')

# LaTeX tables are rebuilt from the same preserved JSON, avoiding manual transcription.
T=B/'3_Main_Report/results_tables';T.mkdir(exist_ok=True)
def table(name,cols,heads,rows,cap,label):
 s='\\begin{table}[!htbp]\\centering\\small\n\\begin{tabular}{'+cols+'}\\toprule\n'+' & '.join('\\textbf{'+h+'}' for h in heads)+'\\\\\\midrule\n'
 s+='\n'.join(' & '.join(str(v) for v in r)+'\\\\' for r in rows)+'\n\\bottomrule\\end{tabular}\n\\caption{'+cap+'}\\label{'+label+'}\n\\end{table}\n'
 (T/(name+'.tex')).write_text(s,encoding='utf-8')
table('r1_blocks','lrrrr',['Text block','Lines','Characters','Character / \\%','Writer / \\%'],[[{'A':'A: paragraph','B':'B: pangram','C':'C: source text'}[k],v['n'],v['characters'],f"{v['char_accuracy']:.2f}",f"{v['naming']:.2f}"] for k,v in m['blocks'].items()]+[['All',184,5471,'84.41','86.41']], 'Read-back character accuracy and writer identification on the photographed robot output. These are different measures; writer identification is supplementary to the correlation target.','tab:res-r1-blocks')
table('r1_writers','lrrrr',['Writer','Lines','Character / \\%','Writer / \\%','Word match / \\%'],[[r['label'],r['n'],f"{r['char']:.2f}",f"{r['id']:.2f}",f"{r['word']:.2f}"] for r in sorted(writers,key=lambda r:-r['char'])], 'Full per-writer reproduction results on 184 labelled lines. Word match uses equal word positions and is not one minus word edit error rate.','tab:app-r1-writers')
recs=load('reader_paper_cnn_bilstm_ctc_joint_best/results.json');alt=load('reader_paper_cnn_bilstm_ctc_scratch_best/results.json');tr=[]
for i in [5,0,1,2,3,4]:
 r=recs[i];a=alt[i];tr.append(['Public IAM validation' if r['set']=='teklia' else 'Photographed '+r['writer'],r['lines'],r['n'],f"{100*(1-r['d']/r['n']):.2f}",f"{100*(1-a['d']/a['n']):.2f}"])
tr+=[['All',1076,45747,'94.06','94.39']]
table('r2_reader','lrrrr',['Evaluation set','Lines','Characters','Joint / \\%','NumPy / \\%'],tr,'Character accuracy of the selected library-trained joint reader and the evaluated NumPy-trained reader, using greedy CTC decoding. The public set is the distributor\'s validation partition, not an independent final test set.','tab:res-r2-reader')
table('r3','rrrrr',['Characters','Records','Allowance / s','Worst total / s','Worst / s per char'],[[c,tim[str(c)]['n'],f'{4*c:.0f}',f"{tim[str(c)]['worst']:.2f}",f"{tim[str(c)]['worst']/c:.3f}"] for c in xx],'ODROID reproduction records with explicit program-end confirmation. Generation and drawing are included in the recorded total. All 25 retained records meet their allowances.','tab:res-r3-confirmed')
table('r4','lrr',['Recorded quantity','500 characters','982 characters'],[['Loading / s','0.63','0.53'],['Conditioning / s','6.61','6.04'],['Segmentation / s','4.29','5.59'],['Cropping / s','4.61','4.65'],['Image stages / s','16.14','16.81'],['Reading / s','19.02','24.66'],['Writer classification / s','11.09','14.01'],['Network stages / s','30.12','38.67'],['Total / s','46.26','55.49'],['R4 allowance / s','14.50','26.55'],['Overrun / s','31.76','28.94']], 'Two recorded ODROID classification requests. Totals are taken from the timer; rounded stage values may differ by 0.01 s when summed.','tab:res-r4-times')
table('r5','llrrr',['Shape','Feature','Nominal / mm','Measured / mm','Error / mm'],[['Square',k,'100.00',f'{v:.2f}',f'{v-100:+.2f}'] for k,v in [('Bottom X',100.3),('Top X',100.2),('Left Y',100.1),('Right Y',99.9)]]+[['Triangle',k,f'{n:.2f}',f'{v:.2f}',f'{v-n:+.2f}'] for k,n,v in [('Base',80,80.3),('Left slope',math.hypot(40,40),56.2),('Right slope',math.hypot(40,40),56.9)]], 'Author-recorded dimensions from one calibrated square and one triangle. These are feature-length errors, rather than an absolute coordinate-error map.','tab:res-r5-lengths')
print('Generated six figures and six tables from preserved evidence.')

ps=E/'physical_style/results.json'
if ps.exists():
 d=json.loads(ps.read_text());names=d['writers_analysed'];R=np.asarray([d['writer_means']['original'][w] for w in names]);S=np.asarray([d['writer_means']['physical'][w] for w in names])
 desc=['Slant / degrees','Stroke width / x-height','Ink density','Word gap / x-height','Components per word','Character width / x-height','Ascender / x-height','Descender / x-height','Ascender extent ratio']
 fig,axes=plt.subplots(3,3,figsize=(6.2,6.0),layout='constrained')
 for i,ax in enumerate(axes.flat):
  ax.scatter(R[:,i],S[:,i],s=17,color=BLUE);lo=min(R[:,i].min(),S[:,i].min());hi=max(R[:,i].max(),S[:,i].max());ax.plot([lo,hi],[lo,hi],color=GREY,lw=.8,ls='--')
  ax.text(.04,.96,f"r = {d['per_feature'][d['feature_names'][i]]:.3f}",transform=ax.transAxes,va='top',fontsize=9)
  ax.set_xlabel(desc[i],fontsize=9);ax.tick_params(labelsize=8.5);ax.spines[['top','right']].set_visible(False)
 fig.supxlabel('Original writer means on horizontal axes; physical output means on vertical axes',fontsize=9)
 save(fig,'results_physical_style_scatter')
 table('style_physical','lrrr',['Feature set','Features','Separate domains','Absolute variant'],[['All geometric descriptors',9,'0.713','0.633'],['Shape descriptors (ink omitted)',7,'0.697','0.611']], 'New physical-to-original comparison across eleven writers. The first correlation standardises features separately in each domain; the absolute variant uses original-domain statistics for both. The target reference is 0.85.','tab:res-physical-style')
