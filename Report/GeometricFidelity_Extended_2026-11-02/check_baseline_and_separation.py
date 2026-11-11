from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry_core import compare
plt.rcParams.update({'font.family':'Times New Roman','svg.fonttype':'none'})
B=Path(__file__).resolve().parent;OLD=B/'baseline_source'
x=json.loads((B/'results.json').read_text());c=json.loads((B/'accepted_candidates.json').read_text());refs={r['id']:r for r in c['references']};W=['dylan','yeukita','owen','abhinav','150'];labels={'dylan':'Dylan','yeukita':'Yeukita','owen':'Owen','abhinav':'Abhinav','150':'IAM 150'}
# The exact earlier physical masks, with the expanded reference bank.
p=json.loads((OLD/'pair_manifest.json').read_text());old=json.loads((OLD/'raw_results.json').read_text());words={r['id']:r for r in p['word_pairs']};lookup={(r['writer'],r['line'],r['word_index'],r['char_index']):r for r in x['results']};paired=[];miss=[]
for r in old['character']:
 w=words[r['word_id']]['physical'];key=(r['writer'],w['line'],w['word_index'],r['index']);q=lookup.get(key)
 if q is None:miss.append(r['id']);continue
 a=np.load(OLD/'masks'/f"{r['id']}.npz")['physical'];b=np.load(B/'glyphs'/f"{q['id']}.npz")['mask'];assert np.array_equal(a,b),(r['id'],q['id'])
 paired.append({'old_id':r['id'],'new_id':q['id'],'writer':r['writer'],'char':r['char'],'fixed_occurrence_f':r['f1_0.10'],'nearest_reference_f':q['f1'],'difference':q['f1']-r['f1_0.10']})
by={w:{'n':sum(r['writer']==w for r in paired),'fixed_occurrence_mean':float(np.mean([r['fixed_occurrence_f'] for r in paired if r['writer']==w])),'nearest_reference_mean':float(np.mean([r['nearest_reference_f'] for r in paired if r['writer']==w]))} for w in W[:-1]}
(B/'matched_baseline.json').write_text(json.dumps({'same_physical_masks_verified':len(paired),'unmatched_old_characters':miss,'per_writer':by,'pooled':{k:float(np.mean([r[k] for r in paired])) for k in ['fixed_occurrence_f','nearest_reference_f']},'pairs':paired},indent=2))
# Exploratory selected examples with positive separation from every other writer.
selected=[];fig,axes=plt.subplots(4,5,figsize=(12,9),layout='constrained')
for j,w in enumerate(W):
 choices=[]
 for r in x['results']:
  if r['writer']!=w:continue
  cc=[q for q in x['controls'] if q['physical']==r['id']]
  if len(cc)!=4 or min(q['own_f1'] for q in cc)<.85:continue
  worst=min(cc,key=lambda q:q['difference']);choices.append((worst['difference'],r,worst))
 _,r,q=max(choices,key=lambda z:z[0]);selected.append({'physical':r,'least_favourable_matched_control':q,'all_four_controls':[v for v in x['controls'] if v['physical']==r['id']]})
 a=np.load(B/'glyphs'/f"{q['own_reference']}.npz");b=np.load(B/'glyphs'/f"{r['id']}.npz");d=np.load(B/'glyphs'/f"{q['other_reference']}.npz");_,m=compare(a['mask'],b['mask'],True)
 for i,g in [(0,a),(1,b),(3,d)]:axes[i,j].imshow(g['gray'],cmap='gray',vmin=0,vmax=255);axes[i,j].axis('off')
 axes[0,j].set_title(f"{labels[w]}: {r['char']}\nGenuine reference",fontsize=10);axes[1,j].set_title(f"Robot in {r['word']}",fontsize=9)
 aa=m['original_skeleton'];bb=m['physical_skeleton'];rgb=np.ones((*aa.shape,3));rgb[aa]=[.87,.31,.15];rgb[bb]=[.08,.38,.75];rgb[aa&bb]=[.35,.15,.55];yy,xx=np.where(aa|bb);axes[2,j].imshow(rgb[max(0,yy.min()-6):yy.max()+7,max(0,xx.min()-6):xx.max()+7]);axes[2,j].axis('off');axes[2,j].set_title(f"Own F {100*q['own_f1']:.2f}%",fontsize=10,y=-.12)
 axes[3,j].set_title(f"{labels[q['other_writer']]} control\nF {100*q['other_f1']:.2f}%, gap {100*q['difference']:.2f} pp",fontsize=9,y=-.25)
fig.suptitle('Exploratory selection of distinguishable character examples\nEqual-size banks within each comparison, selected using the smallest gap across four rival writers',fontsize=12);fig.savefig(B/'figures/selected_distinguishable_letters.png',dpi=230);fig.savefig(B/'figures/selected_distinguishable_letters.pdf');plt.close(fig)
(B/'selected_separability.json').write_text(json.dumps({'selection':'Post-selected maximum of minimum own-minus-other margin across all four other writers, requiring own F >= 0.85 in all controls. Same-letter banks matched to min cardinality separately for each rival. Descriptive examples only.','selected':selected},indent=2))
print('Same-mask baseline',len(paired),'unmatched',miss);print(json.dumps(by,indent=2));print('Selected distinguishable',[(s['physical']['writer'],s['physical']['char'],s['least_favourable_matched_control']['difference']) for s in selected])
