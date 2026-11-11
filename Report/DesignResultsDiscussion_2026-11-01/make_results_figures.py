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
# Times New Roman throughout, matching the report body (epr400.cls loads
# mathptmx), with the maths set in Times as well rather than in a sans or a
# Computer Modern substitute. Every size is 2 pt larger than the sizes this
# script used before, so the smallest text in any figure is 10.5 pt.
TNR='Times New Roman'
plt.rcParams.update({'font.family':'serif','font.serif':[TNR,'STIXGeneral','DejaVu Serif'],
 'mathtext.fontset':'custom','mathtext.rm':TNR,'mathtext.it':TNR+':italic',
 'mathtext.bf':TNR+':bold','mathtext.sf':TNR,'mathtext.default':'it',
 'font.size':11.5,'axes.labelsize':11.5,'xtick.labelsize':11,'ytick.labelsize':11,
 'legend.fontsize':11,'axes.titlesize':11.5,'pdf.fonttype':42})
BLUE='#254e70';GREY='#a6adb4';LIGHT='#d7e1e8'
# The report's \textwidth is 455.24 pt = 6.30 in. A figure authored at that
# width and included at width=\linewidth is printed at scale 1, so a point in
# the figure is a point on the page and the type sizes above survive. Anything
# narrower is rescaled up or down and the sizes stop meaning what they say.
WIDTH=6.30
# no writer name reaches a figure: every drawn label is rewritten as a letter
import re as _re
_NAMEPAT=_re.compile(r"(?i)\b(owen|yeukita|dylan|abhinav|thiya)\s*(\d+)?")
_LET={"owen":"O","yeukita":"Y","dylan":"D","abhinav":"A","thiya":"T"}
def _anonsub(m):
    out="Author "+_LET[m.group(1).lower()]
    if m.group(2): out+=" page %d"%int(m.group(2))
    return out
def AnonFigure(fig):
    import matplotlib.text as _mt
    for t in fig.findobj(_mt.Text):
        try: old=t.get_text()
        except Exception: continue
        if old and _NAMEPAT.search(old): t.set_text(_NAMEPAT.sub(_anonsub,old))


def save(fig,n,width=WIDTH):
    p=F/(n+'.pdf')
    AnonFigure(fig)
    fig.savefig(p)
    try:
        from panelexport import SavePanels
        k=SavePanels(fig,n)
        if k: print(f'  wrote {k} separate panel(s) for {n}')
    except ImportError:
        pass
    plt.close(fig)
    # every figure here uses constrained layout, which fits the content
    # inside the canvas, so the saved page is already one column wide; the
    # check below is a guard, and pads rather than lets a rescale through.
    try:
        from pypdf import PdfReader,PdfWriter
        rd=PdfReader(str(p));pg=rd.pages[0];have=float(pg.mediabox.width);want=72.0*width
        if have>want+3.0: print(f'  ! {n} is {have/72:.2f} in wide, over {width:.2f} in')
        elif want-have>3.0:
            grow=(want-have)/2.0
            x0,y0=float(pg.mediabox.left),float(pg.mediabox.bottom)
            x1,y1=float(pg.mediabox.right),float(pg.mediabox.top)
            for box in (pg.mediabox,pg.cropbox):
                box.lower_left=(x0-grow,y0);box.upper_right=(x1+grow,y1)
            w=PdfWriter();w.add_page(pg)
            with open(p,'wb') as fh: w.write(fh)
    except ImportError: pass


def crisp(gray,lo=1.0,hi=97.0,gamma=1.55):
    """One grey crop stretched so the ink reaches black and the page white.

    A photographed pencil line printed small is a grey smudge; this makes it
    read as a stroke. No pixel moves, so nothing measured is changed."""
    a=np.asarray(gray,dtype=np.float64)
    ink=a[a<200]
    dark=float(np.percentile(ink,lo)) if ink.size else float(a.min())
    page=float(np.percentile(a,hi))
    if page-dark<1.0: page=dark+1.0
    return (255.0*np.clip((a-dark)/(page-dark),0.0,1.0)**gamma).astype(np.uint8)
def clean(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',color='#dddddd',linewidth=.5);ax.set_axisbelow(True)
writers=load('blocks/per_author.json');writers=sorted(writers,key=lambda r:r['char'])
fig,ax=plt.subplots(figsize=(WIDTH,4.5),layout='constrained');y=np.arange(len(writers))
ax.barh(y+.16,[r['char'] for r in writers],height=.3,color=BLUE,label='Character accuracy')
ax.barh(y-.16,[r['id'] for r in writers],height=.3,color=GREY,label='Writer identification')
ax.axvline(85,color='black',ls='--',lw=.9,label='85% reference')
ax.set(yticks=y,yticklabels=[r['label'] for r in writers],xlim=(0,104),xlabel='Accuracy / %',ylabel='Writer');clean(ax)
# the legend sits under the axes: inside, it covered two of the bars
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=3,fontsize=10.5,frameon=False)
save(fig,'results_writer_performance')

hold=load('holdout_namer/lines.json');names=['150','152','153','384','552','588','abhinav','dylan','owen','thiya','yeukita'];labels=['150','152','153','384','552','588','A','B','O','T','Y']
# Stored holdout predictions use both short labels and full writer identifiers.
alias=dict(zip(names,labels));alias.update(dict(zip(labels,labels)));cm=np.zeros((11,11),int)
for r in hold:cm[labels.index(alias[r['writer']]),labels.index(alias[r['named']])]+=1
fig,ax=plt.subplots(figsize=(WIDTH,5.5),layout='constrained');ax.imshow(cm,cmap='Blues',vmin=0,vmax=31)
for i in range(11):
 for j in range(11):
  if cm[i,j]:ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>15 else 'black',fontsize=11)
ax.set(xticks=range(11),xticklabels=labels,yticks=range(11),yticklabels=labels,xlabel='Predicted writer',ylabel='Reference writer');save(fig,'results_holdout_confusion')

tim=m['reproduction_timing']['groups'];xx=[int(c) for c in tim];worst=[tim[str(c)]['worst']/c for c in xx]
fig,ax=plt.subplots(figsize=(WIDTH,3.2),layout='constrained');ax.bar(range(len(xx)),worst,color=BLUE)
ax.axhline(4,color='black',ls='--',label='R3 limit: 4 s/character')
for i,c in enumerate(xx):ax.text(i,worst[i]+.05,'n='+str(tim[str(c)]['n']),ha='center',fontsize=11)
ax.set(xticks=range(len(xx)),xticklabels=xx,xlabel='Reported characters per completed job',ylabel='Worst total / s per character',ylim=(0,4.5));ax.legend(fontsize=11);ax.spines[['top','right']].set_visible(False);save(fig,'results_r3_confirmed')

fig,ax=plt.subplots(figsize=(WIDTH,3.2),layout='constrained');im=[16.14,16.81];net=[30.12,38.67];yy=np.arange(2)
ax.barh(yy,im,color=GREY,label='Image processing');ax.barh(yy,net,left=im,color=BLUE,label='Reader + writer classifier')
for i,(n,allow,total) in enumerate([(500,14.5,46.26),(982,26.55,55.49)]):
 ax.plot([allow,allow],[i-.42,i+.42],color='black',ls='--');ax.text(total+.4,i,f'{total:.2f} s',va='center',fontsize=11)
ax.set(yticks=yy,yticklabels=['500 characters','982 characters'],xlabel='Recorded classification time / s',xlim=(0,70));clean(ax);ax.legend(loc='lower right',fontsize=11,framealpha=1)
save(fig,'results_r4_stages')

fig,ax=plt.subplots(figsize=(WIDTH,3.2),layout='constrained');errs=[.30,.20,.10,-.10,.30,56.2-math.hypot(40,40),56.9-math.hypot(40,40)]
ax.bar(range(7),errs,color=[BLUE]*4+[GREY]*3);ax.axhline(.5,color='black',ls='--');ax.axhline(-.5,color='black',ls='--');ax.axhline(0,color='black',lw=.5)
ax.set(xticks=range(7),xticklabels=['Sq. bottom','Sq. top','Sq. left','Sq. right','Tri. base','Tri. left','Tri. right'],ylabel='Measured length - nominal / mm',ylim=(-.6,.6));ax.tick_params(axis='x',labelrotation=22);ax.spines[['top','right']].set_visible(False);save(fig,'results_geometry_errors')

# Reproduced line crops. A whole line is about sixteen times wider than it
# is tall, so printed across one column it is four millimetres high and
# nothing in it can be read. Each crop is therefore trimmed to the rows that
# carry ink and cut to a window of fixed aspect, so every writer is shown at
# the same, much larger scale. aspect='equal' keeps the writing unstretched.
ASPECT=7.0
rowsPE=[('AuthorY','Y: 97.01%'),('AuthorO','O: 94.25%'),('AuthorD','B: 93.11%'),('AuthorT','T: 60.97%')]
cropsPE=[]
for p,label in rowsPE:
 src=ROOT/'Software/CleanPipeline3/ReportResults/SegmentedReproducedPageCrops'/p/'line_00.png'
 a=np.asarray(Image.open(src).convert('L'))
 ink=a<200;count=ink.sum(1)
 rws=np.nonzero(count>=max(1.0,0.015*count.max()))[0] if count.any() else np.arange(a.shape[0])
 mg=max(2,int(0.10*(rws[-1]-rws[0]+1)))
 y0,y1=max(0,int(rws[0])-mg),min(a.shape[0],int(rws[-1])+1+mg)
 cls=np.nonzero(ink.any(0))[0]
 x0=max(0,int(cls[0])-6) if len(cls) else 0
 x1=min(a.shape[1],x0+int(round(ASPECT*(y1-y0))))
 cropsPE.append((crisp(a[y0:y1,x0:x1]),label))
panelW=WIDTH-0.62
rowH=panelW/max(c.shape[1]/c.shape[0] for c,_ in cropsPE)
fig,axs=plt.subplots(4,1,figsize=(WIDTH,4*rowH+3*0.14+0.10),layout='constrained')
for ax,(arr,label) in zip(axs,cropsPE):
 ax.imshow(arr,cmap='gray',vmin=0,vmax=255,aspect='equal',interpolation='nearest')
 ax.set_xticks([]);ax.set_yticks([])
 for s in ax.spines.values(): s.set_color(GREY);s.set_linewidth(.6)
 ax.text(-.012,.5,label,transform=ax.transAxes,ha='right',va='center',fontsize=11)
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
 fig,axes=plt.subplots(3,3,figsize=(WIDTH,6.6),layout='constrained')
 for i,ax in enumerate(axes.flat):
  ax.scatter(R[:,i],S[:,i],s=17,color=BLUE);lo=min(R[:,i].min(),S[:,i].min());hi=max(R[:,i].max(),S[:,i].max());ax.plot([lo,hi],[lo,hi],color=GREY,lw=.8,ls='--')
  ax.text(.04,.96,f"r = {d['per_feature'][d['feature_names'][i]]:.3f}",transform=ax.transAxes,va='top',fontsize=11)
  ax.set_xlabel(desc[i],fontsize=10.5);ax.tick_params(labelsize=10.5);ax.spines[['top','right']].set_visible(False)
 fig.supxlabel('Original writer means on horizontal axes; physical output means on vertical axes',fontsize=11)
 save(fig,'results_physical_style_scatter')
 table('style_physical','lrrr',['Feature set','Features','Separate domains','Absolute variant'],[['All geometric descriptors',9,'0.713','0.633'],['Shape descriptors (ink omitted)',7,'0.697','0.611']], 'New physical-to-original comparison across eleven writers. The first correlation standardises features separately in each domain; the absolute variant uses original-domain statistics for both. The target reference is 0.85.','tab:res-physical-style')
