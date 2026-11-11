from pathlib import Path
import json,math,hashlib,csv,sys,zipfile,shutil,platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from geometry_core import compare
B=Path(__file__).resolve().parent;ROOT=B.parents[1];x=json.loads((B/'results.json').read_text());c=json.loads((B/'accepted_candidates.json').read_text());refs={r['id']:r for r in c['references']};es={r['id']:r for r in c['physical']};W=['dylan','yeukita','owen','abhinav','150'];labels={'dylan':'Dylan','yeukita':'Yeukita','owen':'Owen','abhinav':'Abhinav','150':'IAM 150'}
F=B/'figures';F.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'font.family':'Times New Roman','svg.fonttype':'none'})
# All selected winning reference pairs are retained as visual audit sheets.
for w in W:
 rr=[r for r in x['results'] if r['writer']==w]
 sheet=Image.new('RGB',(1500,((len(rr)+11)//12)*190),'white');dr=ImageDraw.Draw(sheet)
 for i,r in enumerate(rr):
  a=np.load(B/'glyphs'/f"{r['selected_reference']}.npz");b=np.load(B/'glyphs'/f"{r['id']}.npz");xx=i%12*125;yy=i//12*190;dr.text((xx+3,yy+1),f"{r['id']} {r['char']}",fill='black');dr.text((xx+3,yy+15),f"F {100*r['f1']:.1f}%",fill='black')
  for j,d in enumerate([a,b]):
   im=Image.fromarray(np.where(d['mask'],0,255).astype('uint8')).convert('RGB');im.thumbnail((110,67));sheet.paste(im,(xx+5,yy+35+72*j))
 sheet.save(F/f'all_pairs_{w}.png')
 # Detailed best, typical and weakest examples, chosen after scoring and labelled explicitly.
 ordered=sorted(rr,key=lambda r:r['f1']);examples=[ordered[-1],ordered[len(ordered)//2],ordered[0]];titles=['Selected strongest','Typical score','Lowest score']
 fig,axes=plt.subplots(3,3,figsize=(9,7),layout='constrained')
 for j,(r,title) in enumerate(zip(examples,titles)):
  a=np.load(B/'glyphs'/f"{r['selected_reference']}.npz");b=np.load(B/'glyphs'/f"{r['id']}.npz");out,m=compare(a['mask'],b['mask'],True)
  axes[0,j].imshow(a['gray'],cmap='gray',vmin=0,vmax=255);axes[0,j].axis('off');axes[0,j].set_title(f"{title}: {r['char']}\nOriginal in {refs[r['selected_reference']]['word']}",fontsize=10)
  axes[1,j].imshow(b['gray'],cmap='gray',vmin=0,vmax=255);axes[1,j].axis('off');axes[1,j].set_title(f"Robot in {r['word']}",fontsize=10)
  aa=m['original_skeleton'];bb=m['physical_skeleton'];rgb=np.ones((*aa.shape,3));rgb[aa]=[.87,.31,.15];rgb[bb]=[.08,.38,.75];rgb[aa&bb]=[.35,.15,.55];yy,xx=np.where(aa|bb);axes[2,j].imshow(rgb[max(0,yy.min()-6):yy.max()+7,max(0,xx.min()-6):xx.max()+7]);axes[2,j].axis('off');axes[2,j].set_title(f"F-score {100*r['f1']:.2f}%\n{r['id']} / {r['selected_reference']}",fontsize=9,y=-.20)
 axes[0,0].text(-.15,.5,'Original photograph crop',transform=axes[0,0].transAxes,rotation=90,ha='center',va='center',fontsize=9);axes[1,0].text(-.15,.5,'Physical robot photograph crop',transform=axes[1,0].transAxes,rotation=90,ha='center',va='center',fontsize=9)
 fig.suptitle(f"{labels[w]}: reference-bank comparison at 0.10 x-height\nOrange original centreline, blue robot, purple exact overlap",fontsize=12);fig.savefig(F/f'comparison_{w}.png',dpi=210);fig.savefig(F/f'comparison_{w}.pdf');plt.close(fig)
# Main summary: full usable sample means, not five best-case examples.
fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained');vals=[100*x['summary'][w]['mean_f1'] for w in W];bars=ax.bar(np.arange(5),vals,color=['#47718a','#648d77','#436b8a','#a67c52','#75866d']);ax.axhline(85,ls='--',color='#a34f36',lw=1.2,label='85% reference')
for i,w in enumerate(W):ax.text(i,vals[i]+1.5,f'{vals[i]:.2f}%',ha='center',fontsize=10)
ax.set(xticks=np.arange(5),xticklabels=[f"{labels[w]}\n(n={x['summary'][w]['scored']})" for w in W],ylim=(0,105),ylabel='Nearest-reference mean F-score / %');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(loc='lower left');fig.suptitle('All assessed characters: up to eight same-writer variants per letter\nEqually weighted writer mean 89.29%',fontsize=12);fig.savefig(F/'writer_summary.png',dpi=230);fig.savefig(F/'writer_summary.pdf');fig.savefig(F/'writer_summary.svg');plt.close(fig)
# One strongest pair from each author, explicitly selected examples.
fig,axes=plt.subplots(3,5,figsize=(12,7),layout='constrained')
for j,w in enumerate(W):
 r=next(q for q in x['results'] if q['id']==x['summary'][w]['best_pair']['id']);a=np.load(B/'glyphs'/f"{r['selected_reference']}.npz");b=np.load(B/'glyphs'/f"{r['id']}.npz");_,m=compare(a['mask'],b['mask'],True)
 for i,d in enumerate([a,b]):axes[i,j].imshow(d['gray'],cmap='gray',vmin=0,vmax=255);axes[i,j].axis('off')
 axes[0,j].set_title(f"{labels[w]}: {r['char']}\nOriginal in {refs[r['selected_reference']]['word']}",fontsize=10);axes[1,j].set_title(f"Robot in {r['word']}",fontsize=9)
 aa=m['original_skeleton'];bb=m['physical_skeleton'];rgb=np.ones((*aa.shape,3));rgb[aa]=[.87,.31,.15];rgb[bb]=[.08,.38,.75];rgb[aa&bb]=[.35,.15,.55];yy,xx=np.where(aa|bb);axes[2,j].imshow(rgb[max(0,yy.min()-6):yy.max()+7,max(0,xx.min()-6):xx.max()+7]);axes[2,j].axis('off');axes[2,j].set_title(f"F {100*r['f1']:.2f}%",fontsize=10,y=-.15)
fig.suptitle('Selected strongest valid example for each writer\nOrange original, blue physical robot output, purple exact overlap',fontsize=13);fig.savefig(F/'selected_best_five_writers.png',dpi=240);fig.savefig(F/'selected_best_five_writers.pdf');plt.close(fig)
# Word-group resampling: conditional on these five writers and this fixed reference bank.
rng=np.random.default_rng(23539195);boot={};point={};bygroup={}
for w in W:
 rr=[r for r in x['results'] if r['writer']==w];g={}
 for r in rr:g.setdefault((r['page'],r['line'],r['word_index']),[]).append(r)
 gg=list(g.values());bb=[]
 for _ in range(10000):
  sample=[q['f1'] for i in rng.integers(0,len(gg),len(gg)) for q in gg[i]];bb.append(float(np.mean(sample)))
 boot[w]=bb;p=sum(r['precision']*r['physical_points'] for r in rr)/sum(r['physical_points'] for r in rr);re=sum(r['recall']*r['original_points'] for r in rr)/sum(r['original_points'] for r in rr);point[w]=dict(precision=p,recall=re,f1=2*p*re/(p+re));bygroup[w]=len(gg)
macro=np.mean(np.array([boot[w] for w in W]),axis=0);stats={'writer_mean_word_resampling_95':{w:np.percentile(boot[w],[2.5,97.5]).tolist() for w in W},'fixed_five_writer_macro_word_resampling_95':np.percentile(macro,[2.5,97.5]).tolist(),'point_pooled_per_writer':point,'word_groups':bygroup,'scope':'Conditional descriptive resampling of physical words with the bank fixed. Not an independent generalisation interval.'};(B/'statistics.json').write_text(json.dumps(stats,indent=2))
with (B/'per_writer_summary.csv').open('w',newline='',encoding='utf8') as f:
 out=csv.writer(f);out.writerow(['writer','characters','letter_types','mean_F_percent','point_pooled_F_percent','best_selected_pair_char','best_pair_F_percent','best_selected_letter_min2','letter_n','letter_mean_F_percent','strict_IoU_percent'])
 for w in W:
  z=x['summary'][w];l=z['best_letter_with_two_or_more_examples'];out.writerow([w,z['scored'],z['letters'],100*z['mean_f1'],100*point[w]['f1'],z['best_pair']['char'],100*z['best_pair']['f1'],l['char'],l['n'],100*l['mean_f1'],100*z['mean_iou']])
print('Visual comparisons saved for all five writers.');print('Macro conditional resampling range',stats['fixed_five_writer_macro_word_resampling_95'])
