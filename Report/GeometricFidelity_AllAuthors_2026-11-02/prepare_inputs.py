from pathlib import Path
import ast, collections, hashlib, io, json, pickle, shutil, sys, zlib
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d

B = Path(__file__).resolve().parent
ROOT = B.parents[1]
EXT = ROOT/'Report/GeometricFidelity_Extended_2026-11-02'
OLD = ROOT/'Report/GeometricFidelity_2026-11-02'
W = ['150','152','153','384','552','588','abhinav','dylan','owen','thiya','yeukita']
for d in ['source','inputs','review','figures','line_snapshots']:
    (B/d).mkdir(exist_ok=True)
for f in ['geometry_core.py','fast_geometry.py']:
    shutil.copy2(EXT/f,B/f)
shutil.copy2(EXT/'source/segment.py',B/'source/segment.py')
sys.path.insert(0,str(B))
from geometry_core import SEG

tree = ast.parse((EXT/'build_candidates.py').read_text())
defs = ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['bounds','clean','word_cuts','decode']],type_ignores=[])
exec(compile(defs,'frozen_preprocessing','exec'))
(B/'source/preprocessing.py').write_text(ast.unparse(defs)+'\n')
ann = json.loads((OLD/'word_annotations.json').read_text())
cache = {}
tree = ast.parse((OLD/'prepare_pairs.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mask_for'],type_ignores=[]),'reviewed_preprocessing','exec'))
old_rows = json.loads((OLD/'word_boxes.json').read_text())
old_inputs = json.loads((OLD/'input_manifest.json').read_text())
canonical = {(r['writer'],r['line']):r['page'] for r in old_inputs['original']}
blocks_path = ROOT/'Report/ReportWorkingArchive_2026-11-01_VisualTests/ResultsEvidence/measurements/outputs/blocks/lines.json'
blocks = [r for r in json.loads(blocks_path.read_text()) if r['block']=='C']
physical, references, exclusions, audit = [], [], [], []

def eligible(t):
    return len(t)>=2 and any(c.isalpha() for c in t)

def add(kind,w,t,mask,gray,xh,meta):
    if not mask.any():
        exclusions.append(dict(kind=kind,writer=w,token=t,reason='empty ink mask',**meta))
        return
    y,x=np.where(mask)
    top,bottom,left,right=int(y.min()),int(y.max()+1),int(x.min()),int(x.max()+1)
    native=mask[top:bottom,left:right]
    g=gray[top:bottom,left:right]
    scale=48/xh
    normal=cv2.resize(native.astype('uint8'),(max(1,round(native.shape[1]*scale)),max(1,round(native.shape[0]*scale))),interpolation=cv2.INTER_NEAREST).astype(bool)
    out=physical if kind=='physical' else references
    uid=('E' if kind=='physical' else 'R')+str(len(out)).zfill(5)
    np.savez_compressed(B/'inputs'/f'{uid}.npz',mask=normal,native=native,gray=g)
    out.append(dict(id=uid,writer=w,token=t,xheight=xh,**meta))

# Use previously reviewed physical word boundaries for the personal writers.
old_physical={(r['writer'],r['line'],r['word_index']):r for r in old_rows if r['kind']=='physical'}
for r in blocks:
    w=r['writer'];p=ROOT/f"Software/CleanPipeline3/ReportResults/SegmentedReproducedPageCrops/{r['page']}/line_{r['line']:02}.png"
    gray=np.asarray(Image.open(p).convert('L'));tokens=r['want'].split()
    snapshot=B/'line_snapshots'/f"physical_{r['page']}_{r['line']:02}.png"
    shutil.copy2(p,snapshot)
    m=clean(gray);cuts=word_cuts(m,tokens);band=SEG.CoreBand(m)
    if cuts is None or band is None:
        exclusions.append(dict(kind='physical_line',writer=w,page=r['page'],line=r['line'],reason='no usable word segmentation'))
        continue
    for j,t in enumerate(tokens):
        meta=dict(page=r['page'],line=r['line'],word_index=j,source=str(p.relative_to(ROOT)))
        if not eligible(t):
            exclusions.append(dict(kind='physical',writer=w,token=t,reason='single-character or non-word token',**meta));continue
        old=old_physical.get((w,r['line'],j))
        if old is not None and old['token']==t:
            if old.get('excluded'):
                exclusions.append(dict(kind='physical',writer=w,token=t,reason=old['excluded'],**meta));continue
            mm=mask_for(old);g=gray[old['y0']:old['y1'],old['x0']:old['x1']]
            add('physical',w,t,mm,g,old['xh'],dict(**meta,boundary_source='earlier visual review',box=[old['x0'],old['y0'],old['x1'],old['y1']]))
        else:
            l,rt=cuts[j:j+2]
            add('physical',w,t,m[:,l:rt],gray[:,l:rt],band[1]-band[0],dict(**meta,boundary_source='transcript guided word proposal',box=[l,0,rt,gray.shape[0]]))
targets={w:{r['token'] for r in physical if r['writer']==w} for w in W}
print('Physical proposals',len(physical),dict(collections.Counter(r['writer'] for r in physical)),flush=True)

# Previously reviewed original occurrences take precedence over automatic crops.
occurrences=set()
for r in old_rows:
    w=r['writer']
    if r['kind']!='original' or r['token'] not in targets.get(w,set()) or r.get('excluded'):continue
    page=canonical[(w,r['line'])];key=(w,page,r['line'],r['word_index'])
    p=ROOT/r['source_file'];gray=np.asarray(Image.open(p).convert('L'));mm=mask_for(r)
    add('reference',w,r['token'],mm,gray[r['y0']:r['y1'],r['x0']:r['x1']],r['xh'],dict(page=page,line=r['line'],word_index=r['word_index'],source=str(p.relative_to(ROOT)),boundary_source='earlier visual review',box=[r['x0'],r['y0'],r['x1'],r['y1']]))
    occurrences.add(key)

samples=[]
personal_path=ROOT/'Software/CNN/NOGIT/personal_lines_cache.pkl'
for r in pickle.loads(personal_path.read_bytes())['samples']:
    w=r['page_key'].split('/')[0]
    if w not in W or not set(r['text'].split()) & targets[w]:continue
    page=r['page_key'].split('/')[1].rsplit('_',1)[0];i=int(r['line_idx'])
    label=ROOT/f'Software/CNN/PersonalDataset/{w.capitalize()}/Labels/{page}_labels.txt'
    labels=label.read_text(encoding='utf-8-sig').splitlines()
    if i>=len(labels) or labels[i]!=r['text']:
        audit.append(dict(writer=w,page=page,line=i,reason='cache label does not match current label'));continue
    samples.append(dict(writer=w,page=page,line=i,text=r['text'],raw=r['imageBytes'],source=str(label.relative_to(ROOT)),kind='personal'))
iam_path=ROOT/'Software/CNN/NOGIT/qtp_cache/iam_lines.pkl'
for r in pickle.loads(iam_path.read_bytes()):
    w=r['author']
    if w not in W or not set(r['text'].split()) & targets[w]:continue
    page=Path(r['page_key']).stem;i=int(r['line_idx']);label=ROOT/f'Data/Datasets/IAMpages10/{w}/{page}_labels.txt'
    labels=[t.strip() for t in label.read_text().splitlines() if t.strip()]
    if i>=len(labels) or labels[i]!=r['text'].strip():
        audit.append(dict(writer=w,page=page,line=i,reason='IAM cache label does not match current label'));continue
    samples.append(dict(writer=w,page=page,line=i,text=r['text'],raw=r['png'],source=str(label.relative_to(ROOT)),kind='iam'))

for i,r in enumerate(samples):
    w=r['writer'];gray=decode(r['raw']) if r['kind']=='personal' else np.asarray(Image.open(io.BytesIO(r['raw'])).convert('L'))
    mm=clean(gray);tokens=r['text'].split();cuts=word_cuts(mm,tokens);band=SEG.CoreBand(mm)
    if cuts is None or band is None:continue
    for j,t in enumerate(tokens):
        key=(w,r['page'],r['line'],j)
        if t not in targets[w] or key in occurrences:continue
        l,rt=cuts[j:j+2]
        add('reference',w,t,mm[:,l:rt],gray[:,l:rt],band[1]-band[0],dict(page=r['page'],line=r['line'],word_index=j,source=r['source'],boundary_source='transcript guided word proposal',box=[l,0,rt,gray.shape[0]]))
        occurrences.add(key)
    if i%100==0:print('Source lines',i,'/',len(samples),'reference proposals',len(references),flush=True)

chosen=[]
for w in W:
    for t in sorted(targets[w]):
        rows=sorted([r for r in references if r['writer']==w and r['token']==t],key=lambda r:(r['page'],r['line'],r['word_index']))
        bank=[];pages=set()
        for r in rows:
            if r['page'] not in pages:bank.append(r);pages.add(r['page'])
            if len(bank)==8:break
        for r in rows:
            if len(bank)==8:break
            if r not in bank:bank.append(r)
        chosen.extend(bank)
manifest=dict(writers=W,physical=physical,references=chosen,reference_candidate_count=len(references),source_lines=len(samples),input_exclusions=exclusions,label_audit=audit,inputs_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [blocks_path,personal_path,iam_path]},protocol_sha256=hashlib.sha256((B/'PROTOCOL.md').read_bytes()).hexdigest())
(B/'candidates.json').write_text(json.dumps(manifest,indent=2))
for kind,rows in [('physical',physical),('reference',chosen)]:
    for w in W:
        rr=[r for r in rows if r['writer']==w]
        for k in range(0,len(rr),64):
            part=rr[k:k+64];sheet=Image.new('RGB',(1800,((len(part)+7)//8)*112),'white');draw=ImageDraw.Draw(sheet)
            for j,r in enumerate(part):
                d=np.load(B/'inputs'/f"{r['id']}.npz");im=Image.fromarray(d['gray']).convert('RGB');im.thumbnail((215,75));xx=j%8*225;yy=j//8*112
                draw.text((xx+3,yy+1),r['id']+' '+r['token'],fill='black');sheet.paste(im,(xx+3,yy+27))
            sheet.save(B/'review'/f'{kind}_{w}_{k//64:02}.jpg')
print('Bank entries',len(chosen),'from',len(references),'genuine occurrences',flush=True)
print('Review sheets',len(list((B/'review').glob('*.jpg'))),flush=True)
