from pathlib import Path
import pickle,io,zlib,json,hashlib
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path.cwd();B=ROOT/'Report/GeometricFidelity_2026-11-02';(B/'inputs').mkdir(exist_ok=True)
cache=ROOT/'Software/CNN/NOGIT/personal_lines_cache.pkl'
data=pickle.loads(cache.read_bytes());chosen={'owen':('Owen01',[8,9,10]),'yeukita':('Yeukita03',[4,5]),'dylan':('Dylan02',[0,1]),'thiya':('Thiya04',[0]),'abhinav':('Abhinav03',[2,3,4,5])}
recs=json.loads((ROOT/'Report/ReportWorkingArchive_2026-11-01_VisualTests/ResultsEvidence/measurements/outputs/blocks/lines.json').read_text());manifest={'inputs':{},'original':[],'physical':[]}
def sha(p):manifest['inputs'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
def decode(raw):
 try:return np.load(io.BytesIO(zlib.decompress(raw)),allow_pickle=False)
 except Exception:return np.asarray(Image.open(io.BytesIO(raw)).convert('L'))
sha(cache)
for w,(page,inds) in chosen.items():
 label=ROOT/f'Software/CNN/PersonalDataset/{w.capitalize()}/Labels/{page}_labels.txt';sha(label)
 labels=label.read_text(encoding='utf-8-sig').splitlines()
 for i in inds:
  ss=[s for s in data['samples'] if s['page_key'].startswith(w+'/'+page+'_') and int(s['line_idx'])==i];assert len(ss)==1,(w,i,len(ss))
  s=ss[0];arr=decode(s['imageBytes']);dst=B/'inputs'/f'original_{w}_{i:02}.png';Image.fromarray(arr).save(dst)
  manifest['original'].append({'writer':w,'page':page,'line':i,'text':s['text'],'label_text':labels[i],'cache_label_matches':s['text']==labels[i], 'file':str(dst.relative_to(B)),'shape':list(arr.shape)})
 for r in recs:
  if r['writer']!=w or r['block']!='C':continue
  src=ROOT/f"Software/CleanPipeline3/ReportResults/SegmentedReproducedPageCrops/{r['page']}/line_{r['line']:02}.png";sha(src)
  manifest['physical'].append({'writer':w,'page':r['page'],'line':r['line'],'text':r['want'],'file':str(src.relative_to(ROOT))})
(B/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
# Read-only contact sheet for visual correspondence checking.
rows=manifest['original']+manifest['physical'];sheet=Image.new('RGB',(1400,len(rows)*105),'white');d=ImageDraw.Draw(sheet)
for i,r in enumerate(rows):
 p=B/r['file'] if 'original_' in r['file'] else ROOT/r['file'];im=Image.open(p).convert('RGB');im.thumbnail((1350,68));sheet.paste(im,(20,i*105+29));d.text((20,i*105+5),f"{r['writer']} {r['page']} line {r['line']}: {r['text']}",fill='black')
sheet.save(B/'input_contact.jpg')
print(json.dumps({k:[{a:r[a] for a in ['writer','line','text']} for r in manifest[k]] for k in ['original','physical']},indent=2));print('Cache-label checks',[(r['writer'],r['line'],r['cache_label_matches']) for r in manifest['original']])
