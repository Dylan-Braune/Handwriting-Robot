from pathlib import Path
import json,sys,io,zlib,pickle,ast,shutil
import numpy as np,cv2
from scipy.ndimage import gaussian_filter1d
from PIL import Image,ImageDraw
ROOT=Path.cwd();B=ROOT/'Report/GeometricFidelity_Cursive_2026-11-02';B.mkdir(exist_ok=True)
for d in ['inputs','figures','source']:(B/d).mkdir(exist_ok=True)
EXT=ROOT/'Report/GeometricFidelity_Extended_2026-11-02';OLD=ROOT/'Report/GeometricFidelity_2026-11-02'
for f in ['geometry_core.py','fast_geometry.py']:shutil.copy2(EXT/f,B/f)
shutil.copy2(EXT/'source/segment.py',B/'source/segment.py');sys.path.insert(0,str(B));from geometry_core import SEG
# Reuse only preprocessing and word-boundary functions, without running old experiments.
tree=ast.parse((EXT/'build_candidates.py').read_text());defs=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['bounds','clean','word_cuts','decode']],type_ignores=[]);exec(compile(defs,'saved_preprocessing','exec'))
old=json.loads((OLD/'pair_manifest.json').read_text());physical=[];references=[];targets={}
for r in old['word_pairs']:
 if r['writer'] not in ['thiya','abhinav'] or len(r['token'])<2:continue
 uid='E'+str(len(physical)).zfill(3);d=np.load(OLD/'masks'/f"{r['id']}.npz");np.savez_compressed(B/'inputs'/f'{uid}.npz',mask=d['physical']);rid='R'+str(len(references)).zfill(3);np.savez_compressed(B/'inputs'/f'{rid}.npz',mask=d['original']);physical.append(dict(id=uid,writer=r['writer'],token=r['token'],old_id=r['id'],physical=r['physical']));references.append(dict(id=rid,writer=r['writer'],token=r['token'],page=r['original']['key'],line=r['original']['line'],source=r['original']['source_file'],baseline=True));targets.setdefault(r['writer'],set()).add(r['token'])
cache=pickle.loads((ROOT/'Software/CNN/NOGIT/personal_lines_cache.pkl').read_bytes());count=0
for r in cache['samples']:
 w=r['page_key'].split('/')[0]
 if w not in targets:continue
 toks=r['text'].split()
 if not set(toks)&targets[w]:continue
 page=r['page_key'].split('/')[1].rsplit('_',1)[0];lab=ROOT/f'Software/CNN/PersonalDataset/{w.capitalize()}/Labels/{page}_labels.txt';labels=lab.read_text(encoding='utf-8-sig').splitlines();i=int(r['line_idx'])
 if i>=len(labels) or labels[i]!=r['text']:continue
 a=decode(r['imageBytes']);m=clean(a);cuts=word_cuts(m,toks);band=SEG.CoreBand(m)
 if cuts is None or band is None:continue
 count+=1;xh=band[1]-band[0]
 for j,(t,l,rt) in enumerate(zip(toks,cuts[:-1],cuts[1:])):
  if t not in targets[w]:continue
  mm=m[:,l:rt];yy,xx=np.where(mm)
  if not len(xx):continue
  native=mm[yy.min():yy.max()+1,xx.min():xx.max()+1];s=48/xh;nm=cv2.resize(native.astype('uint8'),(max(1,round(native.shape[1]*s)),max(1,round(native.shape[0]*s))),interpolation=cv2.INTER_NEAREST).astype(bool);uid='R'+str(len(references)).zfill(3);np.savez_compressed(B/'inputs'/f'{uid}.npz',mask=nm);references.append(dict(id=uid,writer=w,token=t,page=page,line=i,word_index=j,source=str(lab.relative_to(ROOT)),baseline=False))
# Fixed page-diverse bank, before geometry scoring.
chosen=[]
for w,ts in targets.items():
 for t in ts:
  rr=sorted([r for r in references if r['writer']==w and r['token']==t],key=lambda r:(r['page'],r['line']));sel=[];pages=set()
  for r in rr:
   if r['page'] not in pages:sel.append(r);pages.add(r['page'])
   if len(sel)==8:break
  for r in rr:
   if len(sel)==8:break
   if r not in sel:sel.append(r)
  chosen+=sel
(B/'manifest.json').write_text(json.dumps({'physical':physical,'references':chosen,'source_lines':count,'reference_candidates':len(references),'method':'Whole-word masks. Existing 48-pixel line x-height, isotropic scale, +/-5-pixel translation and F tolerance 0.10 x-height unchanged. Up to 8 page-diverse source-ordered original same-word references selected before scoring. Also record a separate word-height-normalised diagnostic to investigate layout/scale sensitivity.'},indent=2))
for w in targets:
 rr=[r for r in chosen if r['writer']==w];sheet=Image.new('RGB',(1600,((len(rr)+7)//8)*115),'white');draw=ImageDraw.Draw(sheet)
 for i,r in enumerate(rr):
  mask=np.load(B/'inputs'/f"{r['id']}.npz")['mask'];im=Image.fromarray(np.where(mask,0,255).astype('uint8')).convert('RGB');im.thumbnail((190,75));xx=i%8*200;yy=i//8*115;draw.text((xx+3,yy+2),f"{r['id']} {r['token']}",fill='black');sheet.paste(im,(xx+3,yy+25))
 sheet.save(B/'figures'/f'reference_review_{w}.png')
print('Physical',len(physical),'banks',len(chosen),'candidate refs',len(references),'source lines',count)
