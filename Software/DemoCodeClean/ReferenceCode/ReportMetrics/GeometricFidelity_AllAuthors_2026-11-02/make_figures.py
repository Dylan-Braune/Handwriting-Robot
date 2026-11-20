from pathlib import Path
import csv, json, math
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from geometry_core import compare

B=Path(__file__).resolve().parent
x=json.loads((B/'results.json').read_text())
c=json.loads((B/'accepted_candidates.json').read_text())
writers=c['writers'];labels={w:w if w.isdigit() else w.capitalize() for w in writers}
F=B/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'svg.fonttype':'none','pdf.fonttype':42})

def load(uid):return np.load(B/'inputs'/f'{uid}.npz')

def heightnorm(a):
    yy,xx=np.where(a);a=a[yy.min():yy.max()+1,xx.min():xx.max()+1]
    return cv2.resize(a.astype('uint8'),(max(1,round(a.shape[1]*48/a.shape[0])),48),interpolation=cv2.INTER_NEAREST).astype(bool)

def overlay(a,b,prefix):
    if prefix=='word':a,b=heightnorm(a),heightnorm(b)
    _,m=compare(a,b,True);aa=m['original_skeleton'];bb=m['physical_skeleton']
    rgb=np.ones((*aa.shape,3));rgb[aa]=[.87,.31,.15];rgb[bb]=[.08,.38,.75];rgb[aa&bb]=[.35,.15,.55]
    yy,xx=np.where(aa|bb)
    return rgb[yy.min()-4:yy.max()+5,xx.min()-4:xx.max()+5]

fig,ax=plt.subplots(figsize=(11,5.3),layout='constrained');pos=np.arange(len(writers));bw=.36
for i,(p,color,name) in enumerate([('line','#46718b','Line x-height'),('word','#8b6849','Whole-word height')]):
    vals=[100*x['summary'][w][p]['mean_F'] for w in writers]
    ax.bar(pos+(i-.5)*bw,vals,width=bw,color=color,label=name)
    for j,v in enumerate(vals):ax.text(pos[j]+(i-.5)*bw,v+1.2,f'{v:.1f}',ha='center',fontsize=9)
ax.axhline(85,ls='--',color='#a34f36',lw=1,label='85% reference')
ax.set(xticks=pos,xticklabels=[labels[w]+'\n(n='+str(x['summary'][w]['scored_words'])+')' for w in writers],ylim=(0,108),ylabel='Mean nearest-reference whole-word F-score / %')
ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True);ax.legend(loc='upper right',ncol=3,fontsize=10)
fig.suptitle('Two normalisations on the same assessed words for all eleven writers',fontsize=14)
for ext in ['png','pdf','svg']:fig.savefig(F/f'all_author_summary.{ext}',dpi=210)
plt.close(fig)

for w in writers:
    rows=[r for r in x['results'] if r['writer']==w]
    if not rows:continue
    # Both methods are shown on identical example words. Ranks use the second method.
    order=sorted(rows,key=lambda r:r['word_F'])
    examples=[order[-1],order[len(order)//2],order[0]]
    fig,axes=plt.subplots(4,3,figsize=(10.5,8),layout='constrained')
    for j,(r,title) in enumerate(zip(examples,['Selected strongest','Middle-ranked','Lowest'])):
        a=load(r['word_reference']);b=load(r['id'])
        for i,data in enumerate([a,b]):
            axes[i,j].imshow(data['gray'],cmap='gray',vmin=0,vmax=255);axes[i,j].axis('off')
        axes[0,j].set_title(f"{title}: {r['token']}\nOriginal selected by word-height score",fontsize=10)
        axes[1,j].set_title('Physical robot photograph crop',fontsize=10)
        for i,p in [(2,'line'),(3,'word')]:
            aa=load(r[p+'_reference'])['mask'];bb=b['mask'];axes[i,j].imshow(overlay(aa,bb,p));axes[i,j].axis('off')
            name='Line x-height' if p=='line' else 'Word height'
            axes[i,j].set_title(f"{name} F {100*r[p+'_F']:.2f}%",fontsize=10,pad=6)
    fig.suptitle(f"Writer {labels[w]}\nOrange original centreline, blue physical output, purple exact overlap",fontsize=13)
    for ext in ['png','pdf']:fig.savefig(F/f'comparison_{w}.{ext}',dpi=190)
    plt.close(fig)
    # Every result remains visible in the audit sheets. Each cell has both selected
    # originals and the actual physical crop, with IDs rather than fabricated text.
    for start in range(0,len(rows),12):
        part=rows[start:start+12];sheet=Image.new('RGB',(1600,math.ceil(len(part)/4)*310),'white');dr=ImageDraw.Draw(sheet)
        for j,r in enumerate(part):
            xx=j%4*400;yy=j//4*310
            dr.rectangle([xx,yy,xx+399,yy+309],outline='#cccccc')
            dr.text((xx+6,yy+4),f"{r['id']} {r['token']}",fill='black')
            for k,(uid,label) in enumerate([(r['line_reference'],'Line reference'),(r['word_reference'],'Word reference'),(r['id'],'Physical')]):
                d=load(uid);im=Image.fromarray(d['gray']).convert('RGB');im.thumbnail((380,65));dr.text((xx+6,yy+23+k*85),label+' '+uid,fill='black');sheet.paste(im,(xx+6,yy+38+k*85))
            dr.text((xx+6,yy+294),f"Line F {100*r['line_F']:.2f}%   Word F {100*r['word_F']:.2f}%",fill='black')
        sheet.save(F/f'all_words_{w}_{start//12:02}.jpg')

with (B/'per_author_summary.csv').open('w',newline='',encoding='utf8') as f:
    out=csv.writer(f);out.writerow(['writer','eligible_words','accepted_physical','assessed_words','unsupported_words','line_mean_F_percent','word_mean_F_percent','line_median_F_percent','word_median_F_percent','line_count_F_at_least_85','word_count_F_at_least_85','line_mean_IoU_percent','word_mean_IoU_percent'])
    for w in writers:
        s=x['summary'][w]
        out.writerow([w,s['eligible_proposals'],s['accepted_physical'],s['scored_words'],s['unsupported_words'],100*s['line']['mean_F'],100*s['word']['mean_F'],100*s['line']['median_F'],100*s['word']['median_F'],s['line']['at_least_85'],s['word']['at_least_85'],100*s['line']['mean_IoU'],100*s['word']['mean_IoU']])
print('All-author summary and per-writer comparison/audit figures saved',flush=True)
