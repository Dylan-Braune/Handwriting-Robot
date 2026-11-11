from pathlib import Path
import json,sys,csv,shutil
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B));from geometry_core import compare
plt.rcParams.update({'font.family':'Times New Roman','font.size':11})
x=json.loads((B/'manifest.json').read_text());reject={'R051':'Foreign isolated mark beside the word of','R019':'Foreign isolated mark beside the word calculates'};refs=[r for r in x['references'] if r['id'] not in reject];(B/'quality_review.json').write_text(json.dumps({'excluded':reject,'review':'Selected reference contact sheets and existing physical-word contact sheets inspected before scores. Quick whole-word exploratory review.'},indent=2))
def load(uid):return np.load(B/'inputs'/f'{uid}.npz')['mask']
def heightnorm(a):
 y,x=np.where(a);a=a[y.min():y.max()+1,x.min():x.max()+1];s=48/a.shape[0];return cv2.resize(a.astype('uint8'),(max(1,round(a.shape[1]*s)),48),interpolation=cv2.INTER_NEAREST).astype(bool)
rows=[];allpairs=[]
for r in x['physical']:
 cand=[q for q in refs if q['writer']==r['writer'] and q['token']==r['token']];b=load(r['id']);scores=[];diagnostics=[]
 for q in cand:
  a=load(q['id']);s=compare(a,b);d=compare(heightnorm(a),heightnorm(b));scores.append((q,s));diagnostics.append((q,d));allpairs.append(dict(physical=r['id'],reference=q['id'],writer=r['writer'],token=r['token'],line_xheight_F=s['f1_0.10'],word_height_F=d['f1_0.10']))
 q,s=max(scores,key=lambda z:z[1]['f1_0.10']);qd,d=max(diagnostics,key=lambda z:z[1]['f1_0.10']);rows.append(dict(id=r['id'],writer=r['writer'],token=r['token'],old_id=r['old_id'],reference=q['id'],reference_count=len(cand),F=s['f1_0.10'],strict_iou=s['iou'],word_height_reference=qd['id'],word_height_F=d['f1_0.10']))
summary={}
for w in ['thiya','abhinav']:
 rr=[r for r in rows if r['writer']==w];summary[w]={'words':len(rr),'line_xheight_mean_F':float(np.mean([r['F'] for r in rr])),'word_height_mean_F':float(np.mean([r['word_height_F'] for r in rr])),'best_line_xheight':max(rr,key=lambda r:r['F']),'best_word_height':max(rr,key=lambda r:r['word_height_F']),'words_at_85_line_xheight':sum(r['F']>=.85 for r in rr),'words_at_85_word_height':sum(r['word_height_F']>=.85 for r in rr)}
 ordered=sorted(rr,key=lambda r:r['F']);examples=[ordered[-1],ordered[len(ordered)//2],ordered[0]];fig,axes=plt.subplots(4,3,figsize=(11,7.5),layout='constrained')
 for j,(r,title) in enumerate(zip(examples,['Selected strongest','Middle-ranked','Lowest'])):
  a=load(r['reference']);b=load(r['id']);_,m=compare(a,b,True)
  for i,v in [(0,a),(1,b)]:axes[i,j].imshow(~v,cmap='gray',vmin=0,vmax=1);axes[i,j].axis('off')
  axes[0,j].set_title(f"{title}: {r['token']}\nOriginal",fontsize=11);axes[1,j].set_title('Physical robot output',fontsize=10)
  for i,(aa,bb) in enumerate([(m['original_skeleton'],m['physical_skeleton'])],start=2):
   rgb=np.ones((*aa.shape,3));rgb[aa]=[.87,.31,.15];rgb[bb]=[.08,.38,.75];rgb[aa&bb]=[.35,.15,.55];yy,xx=np.where(aa|bb);axes[i,j].imshow(rgb[yy.min()-3:yy.max()+4,xx.min()-3:xx.max()+4]);axes[i,j].axis('off');axes[i,j].set_title(f"Line-height F {100*r['F']:.2f}%",fontsize=10,y=-.16)
  da=heightnorm(load(r['word_height_reference']));db=heightnorm(b);_,dm=compare(da,db,True);aa=dm['original_skeleton'];bb=dm['physical_skeleton'];rgb=np.ones((*aa.shape,3));rgb[aa]=[.87,.31,.15];rgb[bb]=[.08,.38,.75];rgb[aa&bb]=[.35,.15,.55];yy,xx=np.where(aa|bb);axes[3,j].imshow(rgb[yy.min()-3:yy.max()+4,xx.min()-3:xx.max()+4]);axes[3,j].axis('off');axes[3,j].set_title(f"Separate word-height diagnostic F {100*r['word_height_F']:.2f}%",fontsize=9,y=-.18)
 fig.suptitle(f"{w.capitalize()}: exploratory connected-word comparison\nOrange original, blue robot, purple exact overlap",fontsize=13);fig.savefig(B/'figures'/f'connected_words_{w}.png',dpi=190);fig.savefig(B/'figures'/f'connected_words_{w}.pdf');plt.close(fig)
(B/'results.json').write_text(json.dumps({'summary':summary,'words':rows,'all_reference_pairs':allpairs,'scope':'Small exploratory same-word reconstruction test, not character or author accuracy. Primary uses prior line-xheight rule. Word-height diagnostic is separately normalised and not directly comparable to earlier character results.'},indent=2))
with (B/'word_scores.csv').open('w',newline='') as f:
 o=csv.DictWriter(f,fieldnames=list(rows[0]));o.writeheader();o.writerows(rows)
print(json.dumps(summary,indent=2))
