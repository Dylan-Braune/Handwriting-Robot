from pathlib import Path
import sys,json,re,difflib,hashlib
import numpy as np,cv2
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'Software/CleanPipeline3'));import segment as SEG
rows=json.loads((B/'word_boxes.json').read_text());ann=json.loads((B/'word_annotations.json').read_text());cache={}
def mask_for(r):
 key=r['key']
 if key not in cache:
  a=np.asarray(Image.open(ROOT/r['source_file']).convert('L'));m=SEG.BinarizeLine(a,stripRules=True,closeGaps=False);band=SEG.CoreBand(m)
  if band:
   t,b=band;xh=b-t;nn,lab,stats,cents=cv2.connectedComponentsWithStats(m.astype('uint8'),8)
   for i in range(1,nn):
    x,y,w,h,area=stats[i];cy=cents[i,1]
    if cy<t-.75*xh or cy>b+.75*xh or (h>3*xh and w<.12*xh):m[lab==i]=False
  for l,t,rt,bt in ann.get(key,{}).get('mask_exclude_rects',[]):m[t:bt,l:rt]=False
  cache[key]=m
 return cache[key][r['y0']:r['y1'],r['x0']:r['x1']].copy()
def norm(m,xh):
 scale=48/xh;return cv2.resize(m.astype('uint8'),(max(1,round(m.shape[1]*scale)),max(1,round(m.shape[0]*scale))),interpolation=cv2.INTER_NEAREST).astype(bool)
def spans(m):
 v=np.diff(np.r_[False,m.any(0),False].astype(int));return list(zip(np.where(v==1)[0].tolist(),np.where(v==-1)[0].tolist()))
words=[];chars=[];unmatched=[];excluded=[]
(B/'masks').mkdir(exist_ok=True)
for w in ['owen','yeukita','dylan','thiya','abhinav']:
 ori=[r for r in rows if r['writer']==w and r['kind']=='original'];phy=[r for r in rows if r['writer']==w and r['kind']=='physical']
 sm=difflib.SequenceMatcher(None,[r['token'] for r in ori],[r['token'] for r in phy],autojunk=False);matchedo=set();matchedp=set()
 for oa,pa,size in sm.get_matching_blocks():
  for off in range(size):
   o=ori[oa+off];p=phy[pa+off];matchedo.add(oa+off);matchedp.add(pa+off)
   if o.get('excluded') or p.get('excluded'):excluded.append({'writer':w,'token':o['token'],'reason':o.get('excluded') or p.get('excluded')});continue
   om=mask_for(o);pm=mask_for(p);key=f'{w}_{len(words):03}';np.savez_compressed(B/'masks'/f'{key}.npz',original=norm(om,o['xh']),physical=norm(pm,p['xh']))
   rec={'id':key,'writer':w,'token':o['token'],'original':o,'physical':p};words.append(rec)
   os=spans(om);ps=spans(pm);text=o['token']
   if not text.isalpha() or len(os)!=len(text) or len(ps)!=len(text):continue
   for j,(ch,(l,r),(pl,pr)) in enumerate(zip(text,os,ps)):
    a=om[:,l:r];b=pm[:,pl:pr];ys=np.where(a.any(1))[0];pys=np.where(b.any(1))[0];a=a[ys[0]:ys[-1]+1];b=b[pys[0]:pys[-1]+1];cid=f'{key}_c{j:02}'
    np.savez_compressed(B/'masks'/f'{cid}.npz',original=norm(a,o['xh']),physical=norm(b,p['xh']))
    chars.append({'id':cid,'word_id':key,'writer':w,'token':text,'char':ch,'index':j})
 for kind,rs,matched in [('original',ori,matchedo),('physical',phy,matchedp)]:
  for i,r in enumerate(rs):
   if i not in matched:unmatched.append({'writer':w,'kind':kind,'line':r['line'],'word':r['word_index'],'token':r['token']})
manifest={'word_pairs':words,'character_pairs':chars,'unmatched':unmatched,'excluded':excluded,'annotation_sha256':hashlib.sha256((B/'word_annotations.json').read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((B/'PROTOCOL.md').read_bytes()).hexdigest()};(B/'pair_manifest.json').write_text(json.dumps(manifest,indent=2))
# Review all character pairs before measuring. Top original, bottom physical.
for w in ['owen','yeukita','dylan','thiya','abhinav']:
 rr=[r for r in chars if r['writer']==w];sheet=Image.new('RGB',(1500,max(1,(len(rr)+14)//15)*150),'white');d=ImageDraw.Draw(sheet)
 for i,r in enumerate(rr):
  ar=np.load(B/'masks'/f"{r['id']}.npz");x=i%15*100;y=i//15*150;d.text((x+3,y+2),f"{i} {r['char']} {r['token'][:6]}",fill='black')
  for j,kind in enumerate(['original','physical']):
   im=Image.fromarray(np.where(ar[kind],0,255).astype('uint8')).convert('RGB');im.thumbnail((90,57));sheet.paste(im,(x+5,y+23+j*64))
 sheet.save(B/f'characters_{w}.jpg')
print('Word pairs',len(words),'character pairs',len(chars));print('Per writer',[(w,sum(r['writer']==w for r in words),sum(r['writer']==w for r in chars)) for w in ['owen','yeukita','dylan','thiya','abhinav']]);print('Excluded',excluded);print('Unmatched',unmatched)
