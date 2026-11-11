"""Offline report evidence. Does not train, modify profiles, or access hardware.
Run from any directory; outputs confined to this revision/Evidence and Figures.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import sys,json,hashlib,time,platform,copy,itertools,re
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
# These figures are laid out on a 720 pt wide canvas but printed across the
# report's 455.24 pt text column, so LaTeX scales every one of them by
# 455.24/720 = 0.632. A label set at 11 pt therefore printed at 7 pt, under
# the 8 pt floor Appendix 4 sets and well under the sizes the rest of the
# figures use. The layout is kept and the type is set K times larger, so a
# size given below is the size the reader actually gets on paper.
COLUMN=455.24
K=720.0/COLUMN
# Palette: black, silver, white and shades of blue only, as everywhere else
# in the report. The seam overlay used to be drawn in red.
INK=(.08,.12,.17); SEAM=(.071,.227,.420); SEAMRGB=(18,58,107); MARKRGB=(10,10,10)
_FIGNAME=['figure']; _PANELN=[0]
# the five photographed writers appear as a letter, never as a name
AUTHOR={"owen":"O","yeukita":"Y","dylan":"D","abhinav":"A","thiya":"T"}
_NAMEPAT=re.compile(r"(?i)\b(owen|yeukita|dylan|abhinav|thiya)\s*(\d+)?")
def _sub(m):
    out="Author "+AUTHOR[m.group(1).lower()]
    if m.group(2): out+=" page %d"%int(m.group(2))
    return out
def Anon(s):
    return _NAMEPAT.sub(_sub,str(s))
def AuthorLabel(w):
    k=str(w).strip().lower()
    return "Author "+AUTHOR[k] if k in AUTHOR else ("IAM "+str(w) if str(w).isdigit() else str(w))
def pdf(name,w=720,h=430):
    _FIGNAME[0]=name; _PANELN[0]=0
    c=canvas.Canvas(str(FIG/(name+'.pdf')),pagesize=(w,h)); c.setTitle(name); return c
def _keep(a,title):
    # each panel also written on its own, when FIG_PANELS names a folder
    import os,re
    root=os.environ.get('FIG_PANELS')
    if not root: return
    try:
        from panelexport import _WritePng
        _PANELN[0]+=1
        d=Path(root)/_FIGNAME[0]; d.mkdir(parents=True,exist_ok=True)
        slug=re.sub(r'[^a-z0-9]+','-',str(title).lower()).strip('-')[:52] or 'panel'
        _WritePng(d/f'{_PANELN[0]:02d}_{slug}.png',np.asarray(a.convert('RGB')))
    except Exception as e:
        print(f'    (panel not written: {e})')
def finish(c,name):
    c.showPage();c.save()
    d=pdfium.PdfDocument(str(FIG/(name+'.pdf')))
    d[0].render(scale=1.6).to_pil().save(FIG/(name+'.png'));d.close()
def text(c,x,y,s,size=11,bold=False):
    # Times, because the report body is Times (epr400.cls loads mathptmx)
    # and every other figure in the set is now set in it.
    c.setFont('Times-Bold' if bold else 'Times-Roman',size*K)
    c.setFillColorRGB(*INK);c.drawString(x,y,Anon(s))
def crisp(a,gamma=1.55):
    """One grey crop stretched so the ink reaches black and the page white.

    A photographed pencil stroke printed small is a grey smudge. No pixel
    moves, so nothing measured or segmented here is changed."""
    a=np.asarray(a,dtype=np.float64)
    if a.ndim!=2: return np.asarray(a,dtype=np.uint8)
    ink=a[a<200]
    dark=float(np.percentile(ink,1.0)) if ink.size else float(a.min())
    page=float(np.percentile(a,97.0))
    if page-dark<1.0: page=dark+1.0
    return (255.0*np.clip((a-dark)/(page-dark),0.0,1.0)**gamma).astype(np.uint8)
def linewin(a,aspect=7.0):
    """The left part of a line crop, trimmed to the rows that carry ink.

    A whole line is fifteen or more times wider than it is tall. Drawn in a
    panel a third of the page wide it is a couple of millimetres high and
    illegible, so a window of this aspect is taken instead and printed
    several times larger. Nothing is stretched: the window keeps its shape."""
    a=np.asarray(a)
    ink=a<200;cnt=ink.sum(1)
    if not cnt.any(): return a
    rws=np.nonzero(cnt>=max(1.0,0.015*cnt.max()))[0]
    m=max(2,int(0.10*(rws[-1]-rws[0]+1)))
    y0,y1=max(0,int(rws[0])-m),min(a.shape[0],int(rws[-1])+1+m)
    cls=np.nonzero(ink.any(0))[0]
    x0=max(0,int(cls[0])-6) if len(cls) else 0
    return a[y0:y1,x0:min(a.shape[1],x0+int(round(aspect*(y1-y0))))]
def panel(c,a,x,y,w,h,title,stretch=True):
    if not isinstance(a,Image.Image):
        a=np.asarray(a)
        a=Image.fromarray(crisp(a) if (stretch and a.ndim==2) else np.asarray(a,dtype=np.uint8))
    _keep(a,title)
    iw,ih=a.size; k=min(w/iw,h/ih)   # k is the same in both axes: no stretch
    c.drawImage(ImageReader(a),x+(w-iw*k)/2,y+(h-ih*k)/2,width=iw*k,height=ih*k)
    text(c,x,y+h+9*K,Anon(title),10.5,True)
def _keeppaths(strokes,title):
    # a vector panel has no image of its own, so the pen paths are drawn
    # once more into an array with the project's own round-pen rasteriser
    import os,re
    root=os.environ.get('FIG_PANELS')
    if not root or not strokes: return
    try:
        from panelexport import _WritePng
        pts=np.array([q for t in strokes for q in t],float)
        lo=pts.min(0);hi=pts.max(0);k=1200.0/max(hi[0]-lo[0],1e-6)
        W=int(round((hi[0]-lo[0])*k))+24;H=int(round((hi[1]-lo[1])*k))+24
        arr=np.full((max(H,8),max(W,8)),255,np.uint8)
        for t in strokes:
            if len(t)<2: continue
            seg.DrawPolyline(arr,[(12+(q[0]-lo[0])*k,H-12-(q[1]-lo[1])*k) for q in t],20,2.0)
        _PANELN[0]+=1
        d=Path(root)/_FIGNAME[0]; d.mkdir(parents=True,exist_ok=True)
        slug=re.sub(r'[^a-z0-9]+','-',str(title).lower()).strip('-')[:52] or 'paths'
        _WritePng(d/f'{_PANELN[0]:02d}_{slug}.png',arr)
    except Exception as e:
        print(f'    (path panel not written: {e})')
def drawpaths(c,strokes,x,y,w,h,color=(.08,.25,.4),title='pen-paths'):
    _keeppaths(strokes,title)
    pts=np.array([p for s in strokes for p in s],float)
    if not len(pts): return
    lo=pts.min(0);hi=pts.max(0);k=min(w/max(hi[0]-lo[0],1e-6),h/max(hi[1]-lo[1],1e-6))
    c.setStrokeColorRGB(*color);c.setLineWidth(1.15)
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
        d.line([(x,int(v)) for x,v in enumerate(line)],fill=SEAMRGB,width=3)
    for n,y in enumerate(centres):
        d.text((8,int(y)),str(n+1),fill=MARKRGB)
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
# Panel titles are short because each column is 176 pt wide and the type is
# now set at the size it prints at. The two output crops are drawn across the
# whole width instead of three across a third of it, where a 1500 px line was
# squeezed into 222 pt and nothing in it could be read.
# The panel box is sized to the page's own proportions, so the photographs
# fill it instead of floating in 90 pt of white. Crops 2 and 3 are shown
# because crop 1 of this page is the printed rule above the first line.
# two panels to a row rather than four, each one the top half of the page:
# at twice the width and half the height the writing is legible, and the
# figure shows the stages rather than a thumbnail of every line on the sheet
def tophalf(im):
    if isinstance(im,Image.Image):
        return im.crop((0,0,im.size[0],im.size[1]//2))
    q=np.asarray(im)
    return q[:max(1,q.shape[0]//2)]

c=pdf('design-segmentation-stages',720,780)
halfAR=(a.shape[0]//2)/a.shape[1]
PW=330.0; PH=PW*halfAR
for j,(im,title) in enumerate([(a,'(a) Original'),(np.uint8(np.clip(ratio/1.2,0,1)*255),'(b) Brightness ratio'),
                               (np.uint8(~mask)*255,'(c) Thresholded ink'),(overlay,'(d) Seams found')]):
    col,row=j%2,j//2
    panel(c,tophalf(im),15+col*350,520-row*(PH+34),PW,PH,title,stretch=(j==0))
# two short lines: one long one runs off a 720 pt canvas once the type is
# set at the size it prints at
text(c,15,245,f'Author O page 1: {len(lines)} crops; pitch {stats["pitch"]} analysis pixels; deskew {angle:+.1f} deg.',10.5)
text(c,15,223,'Each panel shows the top half of the sheet.',10.5)
for i,l in enumerate(lines[1:3]):panel(c,l['image'],15,130-i*110,690,62,f'Output line {i+2}')
finish(c,'design-segmentation-stages')

c=pdf('design-segmentation-variation',720,500)
for j,name in enumerate(['original','gradient','rotate+3']):
    a0,ll,dd,ss=allruns[('Owen',name)]
    panel(c,a0,15+j*235,225,220,245,{'original':'Original photograph','gradient':'Synthetic gradient','rotate+3':'Synthetic +3 deg rotation'}[name])
    text(c,15+j*235,200,f'{len(ll)} crops; deskew {ss["angle"]:+.1f} deg',10.5)
    # the start of the recovered line, at an aspect that fills the panel
    if len(ll)>1:panel(c,linewin(ll[1]['image'],4.5),15+j*235,100,220,50,'Start of recovered line 2')
text(c,15,25,'Digital perturbations of one photograph; no measured illumination claim.',10.5)
finish(c,'design-segmentation-variation')

# Real photographic robot output: separate from input handwriting.
rp=ROOT/'Software/CleanPipeline3/ReportResults/FullReproducedPages/AuthorO.jpg';record(rp)
robot=np.array(Image.open(rp).convert('L'));robot=seg.UprightPage(robot)[0]
rl,rd,rs=seg.SegmentLines(robot)
records['robot_example']={'path':str(rp.relative_to(ROOT)),**rs}
c=pdf('design-physical-page-example',720,495)
panel(c,robot,20,25,315,430,'Photographed robot output in the Author O style')
for i,l in enumerate(rl[:4]):panel(c,l['image'],365,370-i*108,330,62,f'Recovered line {i+1}')
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
c=pdf('design-style-gcode-examples',720,470)
for j,writer in enumerate(profiles):
    t,parsed=trjs[writer];y=350-j*100
    who=AuthorLabel(writer)
    text(c,15,y+22,who,11,True)
    drawpaths(c,t.strokes,110,y,250,58,title=f'{who}-synthesised-trajectory')
    drawpaths(c,parsed,400,y,290,58,title=f'{who}-parsed-from-gcode')
text(c,110,448,'Stored profile to synthesised trajectory',10.5,True)
text(c,400,448,'Emitted G-code to parsed pen paths',10.5,True)
text(c,15,18,'Same text and seed; independently fitted panels. These are software renders, not physical drawings.',10.5)
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
    if ch!=' ':d.line((l,0,l,gray.shape[0]-1),fill=SEAMRGB,width=3)
c=pdf('design-glyph-alignment',720,380)
text(c,15,358,'Known transcript: '+truth,10.5,True)
panel(c,gray,15,210,690,90,'Original line crop')
panel(c,ov,15,95,690,90,'Refined shared boundaries (blue)',stretch=False)
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
    c=pdf('design-candidate-selection',720,450)
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
        text(c,520,347-k*120,f'A = {A:.3f}',11)
        text(c,520,320-k*120,f'P = {P:.3f}',11)
        text(c,520,293-k*120,f'J = {J:.3f}',11,True)
    winner=int(np.argmax([p['J'] for p in candidates]))
    text(c,15,412,f'Owen; fixed text; selected candidate {winner+1} from these three trials',11,True)
    text(c,15,14,'Pre-repair diagnostic trials; shape width 0.17. Selection scores are not independent physical results.',10.5)
    records['candidates']={'text':sentence,'writer':'owen','pen_width_xh':.17,'selected':winner+1,'trials':candidates}
    finish(c,'design-candidate-selection')
else:records['candidates']={'unavailable_checkpoint':str(shape)}
save();print('Visual bench finished',flush=True)
