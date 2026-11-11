"""Offline report evidence. Does not train, modify profiles, or access hardware.
Run from any directory; outputs confined to this revision/Evidence and Figures.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import sys,json,hashlib,time,platform,copy,itertools
import numpy as np
from PIL import Image,ImageDraw
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import pypdfium2 as pdfium

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CODE=ROOT/'Software/CleanPipeline3'
sys.path.insert(0,str(CODE))
import segment as seg
import recognise as rec
import style,synthesise as syn,reproduce as rep
OUT=HERE/'Evidence'; OUT.mkdir(exist_ok=True)
FIG=HERE/'Figures'; FIG.mkdir(exist_ok=True)
records={'scope':'Offline subsystem illustration and deterministic software checks; no hardware or training; not a QTP.',
         'host':platform.platform(),'python':sys.version.split()[0],'numpy':np.__version__,
         'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.glob('*.py')},'inputs':{},'segmentation':[]}
def record(p):
    try: name=str(p.relative_to(ROOT))
    except ValueError: name=str(p)
    records['inputs'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
def save(): (OUT/'bench.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
def pdf(name,w=720,h=430):
    c=canvas.Canvas(str(FIG/(name+'.pdf')),pagesize=(w,h)); c.setTitle(name); return c
def finish(c,name):
    c.showPage();c.save()
    d=pdfium.PdfDocument(str(FIG/(name+'.pdf')))
    d[0].render(scale=1.6).to_pil().save(FIG/(name+'.png'));d.close()
def text(c,x,y,s,size=11,bold=False):
    c.setFont('Helvetica-Bold' if bold else 'Helvetica',size);c.setFillColorRGB(.08,.12,.17);c.drawString(x,y,str(s))
def panel(c,a,x,y,w,h,title):
    a=Image.fromarray(np.asarray(a,dtype=np.uint8)) if not isinstance(a,Image.Image) else a
    iw,ih=a.size; k=min(w/iw,h/ih)
    c.drawImage(ImageReader(a),x+(w-iw*k)/2,y+(h-ih*k)/2,width=iw*k,height=ih*k)
    text(c,x,y+h+7,title,11,True)
def drawpaths(c,strokes,x,y,w,h,color=(.08,.25,.4)):
    pts=np.array([p for s in strokes for p in s],float)
    if not len(pts): return
    lo=pts.min(0);hi=pts.max(0);k=min(w/max(hi[0]-lo[0],1e-6),h/max(hi[1]-lo[1],1e-6))
    c.setStrokeColorRGB(*color);c.setLineWidth(.65)
    for s in strokes:
        if len(s)<2:continue
        p=c.beginPath();p.moveTo(x+(s[0][0]-lo[0])*k,y+(s[0][1]-lo[1])*k)
        for q in s[1:]:p.lineTo(x+(q[0]-lo[0])*k,y+(q[1]-lo[1])*k)
        c.drawPath(p)
def page(writer):
    p=ROOT/f'Software/CNN/PersonalDataset/{writer}/Pages/{writer}01.jpg';record(p)
    im=Image.open(p).convert('L'); im.thumbnail((1600,1600));return np.array(im),p
def stages(a):
    full,half,mask,ratio,angle=seg.Deskew(a)
    clean,paper,pitch,_,_=seg.RemovePrintedStructure(mask,half.astype(float))
    clean,lab,nc,area,na=seg.DropArtifactPieces(clean,ratio,pitch,*clean.shape)
    pitch,centres=seg.LineCentres(clean,clean.shape[0])
    cuts,ca,L=seg.SeamCuts(clean,centres,pitch,*clean.shape)
    overlay=Image.fromarray(half).convert('RGB'); d=ImageDraw.Draw(overlay)
    for n,line in enumerate(cuts):
        d.line([(x,int(v)) for x,v in enumerate(line)],fill=(190,35,45),width=2)
    for n,y in enumerate(centres):
        d.text((8,int(y)),str(n+1),fill=(20,60,200))
    return full,half,mask,ratio,clean,overlay,angle

# Three original pages with controlled, explicitly synthetic disturbances.
allruns={}
for writer in ['Owen','Dylan','Yeukita']:
    a,p=page(writer)
    yy,xx=np.indices(a.shape)
    gradient=.42+.58*(xx/max(1,a.shape[1]-1))
    cases={'original':a,'gradient':np.uint8(a*gradient),
           'rotate+3':np.array(Image.fromarray(a).rotate(3,resample=Image.Resampling.BICUBIC,expand=True,fillcolor=255)),
           'quarter-turn':np.rot90(a)}
    lp=p.parent.parent/'Labels'/f'{writer}01_labels.txt';record(lp)
    expected=len([l for l in lp.read_text(encoding='utf-8-sig').splitlines() if l.strip()])
    for name,img in cases.items():
        print('Segment',writer,name,flush=True)
        upright,turn,s0,s90=seg.UprightPage(img)
        lines,diag,stats=seg.SegmentLines(upright)
        result={'writer':writer,'case':name,'expected_label_lines':expected,'input_size':[int(img.shape[1]),int(img.shape[0])],
                'axis_turn':turn,'axis_scores':[s0,s90],**stats}
        records['segmentation'].append(result);allruns[(writer,name)]=(upright,lines,diag,stats)
        target=OUT/f'{writer}_{name.replace("+","plus")}'
        target.mkdir(exist_ok=True)
        for i,l in enumerate(lines):Image.fromarray(l['image']).save(target/f'line_{i:02}.png')
        save()

a,lines,diag,stats=allruns[('Owen','original')]
full,half,mask,ratio,clean,overlay,angle=stages(a)
c=pdf('design-segmentation-stages',720,475)
for j,(im,title) in enumerate([(a,'(a) Original Owen01'),(np.uint8(np.clip(ratio/1.2,0,1)*255),'(b) Local brightness ratio'),
                               (np.uint8(~mask)*255,'(c) Thresholded ink'),(overlay,'(d) Cleaned mask / seam locations')]):
    panel(c,im,15+j*176,130,165,310,title)
for i,l in enumerate(lines[:3]):panel(c,l['image'],15+i*235,20,222,63,f'Output line {i+1}')
text(c,15,110,f'Actual pipeline: {len(lines)} crops; pitch {stats["pitch"]} analysis pixels; deskew {angle:+.1f} deg.',10)
finish(c,'design-segmentation-stages')

c=pdf('design-segmentation-variation',720,420)
for j,name in enumerate(['original','gradient','rotate+3']):
    a0,ll,dd,ss=allruns[('Owen',name)]
    panel(c,a0,15+j*235,130,220,245,{'original':'Original photograph','gradient':'Synthetic 0.42-1.00 gradient','rotate+3':'Synthetic +3 deg rotation'}[name])
    if len(ll)>1:panel(c,ll[1]['image'],15+j*235,35,220,65,'Recovered line 2')
    text(c,15+j*235,112,f'{len(ll)} crops; deskew {ss["angle"]:+.1f} deg',10)
text(c,15,10,'Digital perturbations of one photograph; no measured illumination/lux claim.',10)
finish(c,'design-segmentation-variation')

# Real photographic robot output: separate from input handwriting.
rp=ROOT/'Software/CleanPipeline3/ReportResults/FullReproducedPages/AuthorO.jpg';record(rp)
robot=np.array(Image.open(rp).convert('L'));robot=seg.UprightPage(robot)[0]
rl,rd,rs=seg.SegmentLines(robot)
records['robot_example']={'path':str(rp.relative_to(ROOT)),**rs}
c=pdf('design-physical-page-example',720,455)
panel(c,robot,20,25,315,390,'Photographed Owen-style robot output')
for i,l in enumerate(rl[:4]):panel(c,l['image'],365,335-i*91,330,58,f'Recovered line {i+1}')
finish(c,'design-physical-page-example');save()

# Software-only generation and emitted-file reconstruction across four profiles.
profiles={}; trjs={};records['gcode']=[]
for writer in ['owen','dylan','yeukita','150']:
    p=ROOT/f'Software/CNN/NOGIT/StyleProfiles10/{writer}_curated.json';record(p)
    profile=json.loads(p.read_text(encoding='utf-8'));profiles[writer]=profile
    t=syn.SynthesizeText('Writing with style.',profile,seed=2026,mmPerXh=4,lineWidthMm=170,jitter=.15)
    cfg=rep.GantryConfig();gpath=OUT/f'{writer}_example.gcode'
    info=rep.WriteGcode(t,cfg,gpath,title='Offline report illustration; not executed')
    parsed=rep.ParseGcode(gpath)
    pts=np.array([p for s in parsed for p in s])
    inside=bool(((pts[:,0]>=cfg.boundsMinXmm-.001)&(pts[:,0]<=cfg.boundsMaxXmm+.001)&(pts[:,1]>=cfg.boundsMinYmm-.001)&(pts[:,1]<=cfg.boundsMaxYmm+.001)).all())
    records['gcode'].append({'writer':writer,'text':'Writing with style.','seed':2026,'within_configured_bounds':inside,**info,'trajectory_meta':t.meta})
    trjs[writer]=(t,parsed)
save()
c=pdf('design-style-gcode-examples',720,420)
for j,writer in enumerate(profiles):
    t,parsed=trjs[writer];y=320-j*95
    text(c,15,y+65,writer.capitalize(),12,True)
    drawpaths(c,t.strokes,110,y,260,55)
    drawpaths(c,parsed,405,y,290,55)
text(c,110,403,'Stored profile -> synthesised trajectory',11,True)
text(c,405,403,'Emitted G-code -> parsed pen paths',11,True)
text(c,15,10,'Same text and seed; independently fitted panels. These are software renders, not physical drawings.',10)
finish(c,'design-style-gcode-examples')

# Independent eight-neighbour reference labeller, compared by component partitions.
def bfs(mask):
    labels=np.zeros(mask.shape,np.int32);n=0
    for y,x in zip(*np.nonzero(mask)):
        if labels[y,x]:continue
        n+=1;labels[y,x]=n;stack=[(y,x)]
        while stack:
            u,v=stack.pop()
            for dy,dx in itertools.product([-1,0,1],repeat=2):
                q,r=u+dy,v+dx
                if 0<=q<mask.shape[0] and 0<=r<mask.shape[1] and mask[q,r] and not labels[q,r]:
                    labels[q,r]=n;stack.append((q,r))
    return labels,n
rng=np.random.default_rng(2026);match=0
for i in range(100):
    m=rng.random((9,11))<(.1+.7*(i/99));a,n=seg.LabelComponents(m);b,k=bfs(m)
    aa=a[m];bb=b[m]
    assert n==k and np.array_equal(aa[:,None]==aa[None,:],bb[:,None]==bb[None,:])
    match+=1
records['connected_components']={'random_masks':100,'shape':[9,11],'seed':2026,'partition_matches':match,'connectivity':8}

# First-principles CTC vs explicit path enumeration and finite differences.
import train
z=rng.normal(size=(5,1,4)).astype(np.float64);target=[1,1]
loss,grad=train.CtcLoss(z,[target])
ez=np.exp(z[:,0]-z[:,0].max(1,keepdims=True));pr=ez/ez.sum(1,keepdims=True)
def collapse(path):
    merged=[k for j,k in enumerate(path) if j==0 or k!=path[j-1]]
    return [k for k in merged if k!=0]
mass=sum(float(np.prod([pr[t,k] for t,k in enumerate(path)])) for path in itertools.product(range(4),repeat=5) if collapse(path)==target)
err=[];eps=1e-5
for idx in np.ndindex(z.shape):
    zz=z.copy();zz[idx]+=eps;la=train.CtcLoss(zz,[target])[0]
    zz[idx]-=2*eps;lb=train.CtcLoss(zz,[target])[0]
    err.append(abs((la-lb)/(2*eps)-grad[idx]))
records['ctc']={'seed':2026,'shape':list(z.shape),'target':target,'blank':0,'enumerated_paths':4**5,
                'loss':float(loss),'enumeration_loss':float(-np.log(mass)),
                'loss_absolute_error':float(abs(loss+np.log(mass))),'epsilon':eps,'max_gradient_absolute_error':float(max(err))}
save();print('Core checks complete',flush=True)

# Forced alignment on an actual original line. One NumPy reader forward pass.
weights=ROOT/'Software/CNN/weights/paper_cnn_bilstm_ctc_joint_best.npz';record(weights)
model=rec.PaperCRNNNumpy(weights)
truth=(ROOT/'Software/CNN/PersonalDataset/Owen/Labels/Owen01_labels.txt').read_text(encoding='utf-8-sig').splitlines()[0].strip()
crop=ROOT/'Handwriting-Robot-Pipeline/Software/CNN/Crops/Owen/Owen01/00.png';record(crop)
gray=np.array(Image.open(crop).convert('L'));print('Reader archived original-line alignment',flush=True)
logp=model.forward(rec.TensorFromResized(rec.ResizeLineImageFixed(gray)))
class Cached:
    def forward(self,x):return logp
glyphs,stats=style.ExtractLineGlyphs(gray,truth,Cached())
if stats is None:raise RuntimeError('Alignment failed for original line')
align,conf=rec.CtcForcedAlign(logp[:,0,:],truth)
records['alignment']={'writer':'owen','source_line':0,'transcript':truth,'crop_provenance':'Archived original-handwriting crop visually matched to the first labelled line; not the new reduced-page segmentation output',
                      'stats':stats,'greedy_text':rec.DecodeCtc(logp)[0]}
ov=Image.fromarray(gray).convert('RGB');d=ImageDraw.Draw(ov)
for i,ch,l,r in stats['cuts']:
    if ch!=' ':d.line((l,0,l,gray.shape[0]-1),fill=(200,30,40),width=2)
c=pdf('design-glyph-alignment',720,350)
text(c,15,330,'Known transcript: '+truth,11,True)
panel(c,gray,15,210,690,90,'Original line crop')
panel(c,ov,15,95,690,90,'Refined shared boundaries (red)')
selected=[]
for i,g in enumerate(glyphs):
    if g and truth[i]!=' ':
        selected.append((truth[i],g))
    if len(selected)==9:break
for j,(ch,g) in enumerate(selected):
    drawpaths(c,g['strokes'],20+j*77,20,55,45)
    text(c,40+j*77,7,ch,11,True)
finish(c,'design-glyph-alignment');save()

# Candidate scores are diagnostic selection evidence, explicitly normalised at .17.
shape=ROOT/'Software/CNN/weights/author_shape_11_pw170_weights.npz'
if not shape.exists():
    found=list((ROOT/'Software/CNN').rglob(shape.name));shape=found[0] if found else shape
if not shape.exists():shape=Path('C:/Users/braun/Downloads')/shape.name
if shape.exists():
    record(shape); namer=rec.AuthorClassifierCNNNumpy(shape);candidates=[]
    c=pdf('design-candidate-selection',720,430)
    sentence='Writing with style.'
    for k in range(3):
        print('Score candidate',k,flush=True)
        t=syn.SynthesizeText(sentence,profiles['owen'],seed=2026+977*k,mmPerXh=4,jitter=.5)
        im=syn.RenderTrajectory(t,pxPerMm=18,profile=profiles['owen'],uniformInk=True)
        read=rec.ReadText(im,model);A=rec.CharAcc(read,sentence)
        norm=rec.StrokeNormalize(np.array(im,dtype=np.uint8),penWidthXh=.17)
        probs=rec.Softmax(namer.forward(rec.TensorFromResized(rec.ResizeLineImageFixed(norm))),axis=1)[0]
        P=float(probs[namer.authorMapping['owen']]);J=2*A*P/(A+P) if A+P else 0
        candidates.append({'seed':2026+977*k,'read':read,'A':A,'P':P,'J':J,'meta':t.meta})
        panel(c,im,15,290-k*120,485,80,f'Candidate {k+1}')
        text(c,520,340-k*120,f'A = {A:.3f}',12)
        text(c,520,320-k*120,f'P = {P:.3f}',12)
        text(c,520,300-k*120,f'J = {J:.3f}',12,True)
    winner=int(np.argmax([p['J'] for p in candidates]))
    text(c,15,405,f'Owen; fixed text; selected candidate {winner+1} from these three trials',12,True)
    text(c,15,10,'Pre-repair diagnostic trials; shape width 0.17. Selection scores are not independent physical results.',10)
    records['candidates']={'text':sentence,'writer':'owen','pen_width_xh':.17,'selected':winner+1,'trials':candidates}
    finish(c,'design-candidate-selection')
else:records['candidates']={'unavailable_checkpoint':str(shape)}
save();print('Visual bench finished',flush=True)
