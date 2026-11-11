from pathlib import Path
import json,sys,math,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parent;ROOT=B.parents[1];raw=json.loads((B/'raw_results.json').read_text());summary=json.loads((B/'summary.json').read_text());stats=json.loads((B/'statistics.json').read_text());manifest=json.loads((B/'pair_manifest.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
F=B/'figures';F.mkdir(exist_ok=True)
# Show every writer, the prespecified threshold, and the small per-character populations.
names=['owen','yeukita','dylan','abhinav','thiya','pooled'];labels=['Owen','Yeukita','Dylan','Abhinav','Thiya','Pooled'];x=np.arange(len(names));means=[100*summary['character'][w]['f1_0.10'] if summary['character'][w]['n'] else np.nan for w in names];points=[100*stats['character'][w]['point_pooled_f1'] if stats['character'][w]['n'] else np.nan for w in names]
fig,ax=plt.subplots(figsize=(10,5),layout='constrained');bars=ax.bar(x-.16,means,width=.30,color='#245f82',label='Mean per-character F-score');ax.bar(x+.16,points,width=.30,color='#9ebecf',label='Pooled centreline-point F-score');ax.axhline(85,ls='--',color='#aa4a29',lw=1.2,label='85% reference')
for i,w in enumerate(names):
 st=stats['character'][w];n=st['n']
 if n:
  label_y=(100*st['group_bootstrap_95'][1]+1.5) if n>=10 else means[i]+1.5
  ax.text(i-.16,label_y,f'{means[i]:.1f}',ha='center',fontsize=9)
  if n>=10:
   lo,hi=np.array(st['group_bootstrap_95'])*100;ax.errorbar(i-.16,means[i],yerr=[[means[i]-lo],[hi-means[i]]],fmt='none',capsize=3,color='black',lw=1)
 else:ax.text(i,13,'No separable\ncharacter pairs',ha='center',fontsize=9,color='#555')
ax.set_xticks(x,[f'{l}\n(n={summary["character"][w]["n"]})' for w,l in zip(names,labels)]);ax.set_ylim(0,106);ax.set_ylabel('Stroke-agreement F-score / %');ax.spines[['top','right']].set_visible(False);ax.legend(loc='lower left',fontsize=8.5);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.savefig(F/'character_agreement.png',dpi=220);fig.savefig(F/'character_agreement.svg');plt.close(fig)
# The tolerance sweep is reported in full, not chosen retrospectively to obtain 85.
fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
for w,l in zip(names[:4],labels[:4]):ax.plot([.05,.10,.15,.20],[100*summary['character'][w][f'f1_{q:.2f}'] for q in [.05,.10,.15,.20]],marker='o',label=f'{l}, n={summary["character"][w]["n"]}')
ax.axhline(85,color='black',ls='--',lw=.9);ax.axvline(.1,color='gray',ls=':',lw=1);ax.set(xlabel='Distance tolerance / estimated x-height',ylabel='Mean character F-score / %',ylim=(0,103),xticks=[.05,.1,.15,.2]);ax.legend(fontsize=9);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);fig.savefig(F/'tolerance_sensitivity.png',dpi=220);plt.close(fig)
# Full Owen gallery, including lower-score cases, after the same bounded alignment.
env={'__file__':str(B/'measure_geometry.py'),'__name__':'geometry_functions'};exec((B/'measure_geometry.py').read_text().split('results={};start=time.time()')[0],env);compare=env['compare']
rr=[r for r in raw['character'] if r['writer']=='owen'];fig,axes=plt.subplots(5,5,figsize=(10,12),layout='constrained')
for ax,r in zip(axes.flat,rr):
 d=np.load(B/'masks'/f"{r['id']}.npz");_,m=compare(d['original'],d['physical'],True);a=m['original_skeleton'];b=m['physical_skeleton'];y,xs=np.where(a|b);lo=max(0,int(xs.min())-7);hi=int(xs.max())+8;top=max(0,int(y.min())-7);bot=int(y.max())+8
 rgb=np.ones((*a.shape,3));rgb[a]=[.85,.3,.15];rgb[b]=[.1,.4,.75];rgb[a&b]=[.35,.15,.55];ax.imshow(rgb[top:bot,lo:hi]);ax.axis('off');ax.set_title(f"{r['char']}  {100*r['f1_0.10']:.1f}%\n{r['token']}, character {r['index']+1}",fontsize=9)
fig.suptitle('All 25 Owen character pairs\nOrange original centreline, blue physical output, purple exact overlap',fontsize=12);fig.savefig(F/'owen_all_25_overlays.png',dpi=210);plt.close(fig)
# Blind read-back packet, preserving the full physical source-text blocks (not just selected pairs).
inputdata=json.loads((B/'input_manifest.json').read_text());rng=np.random.default_rng(23539195);writers=['owen','yeukita','dylan','thiya','abhinav'];rng.shuffle(writers);answer=[];lines=[]
for idx,w in enumerate(writers,1):
 rr=[r for r in inputdata['physical'] if r['writer']==w];ims=[]
 for r in rr:
  im=Image.open(ROOT/r['file']).convert('RGB');im.thumbnail((1500,140));ims.append((r,im))
 sheet=Image.new('RGB',(1580,100+len(ims)*210),'white');dr=ImageDraw.Draw(sheet);code=f'P{idx:02}';dr.text((20,15),f'{code} - Read the handwriting, preserving spelling. Use ? for unreadable characters.',fill='black')
 for j,(r,im) in enumerate(ims):
  yy=75+j*210;sheet.paste(im,(30,yy));dr.text((30,yy+155),f'{code}-L{j+1:02}: ',fill='black');answer.append({'id':f'{code}-L{j+1:02}','writer':w,'page':r['page'],'source_line':r['line'],'expected':r['text']});lines.append({'id':f'{code}-L{j+1:02}','transcription':''})
 folder=B/'blind_readback';folder.mkdir(exist_ok=True);sheet.save(folder/f'{code}.png')
with (B/'blind_readback/responses.csv').open('w',newline='',encoding='utf-8') as f:
 ww=csv.DictWriter(f,fieldnames=['id','transcription']);ww.writeheader();ww.writerows(lines)
(B/'blind_readback/answer_key.json').write_text(json.dumps(answer,indent=2))
# Record all numerical results as tables for reuse without manual transcription.
with (B/'per_writer_summary.csv').open('w',newline='',encoding='utf8') as f:
 ww=csv.writer(f);ww.writerow(['writer','word_pairs','separable_characters','character_mean_f1_010_percent','point_pooled_f1_010_percent','raw_character_iou_percent','stroke_normalised_character_iou_percent','bootstrap_lower_percent','bootstrap_upper_percent'])
 for w in names:
  n=summary['character'][w]['n'];st=stats['character'][w];ww.writerow([w,summary['word'][w]['n'],n,100*summary['character'][w]['f1_0.10'] if n else '',100*st['point_pooled_f1'] if n else '',100*summary['character'][w]['iou'] if n else '',100*summary['character'][w]['normalised_iou'] if n else '',*(np.array(st['group_bootstrap_95'])*100 if n else ['',''])])
print('Figures and blind readback material written')
