"""Real archived capture sequence. No invented lighting values or transcripts."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
source=(HERE/'run_visual_bench.py').read_text(encoding='utf-8')
exec(compile(source.split('# Three original pages')[0],str(HERE/'run_visual_bench.py'),'exec'))
records={'scope':'Actual archived repeated photographs, native resolution, no inferred lux values. Visual illustration only.','inputs':{},'captures':[]}
folder=Path('C:/Users/braun/OneDrive/Pictures/Camera Roll')
names=['WIN_20261028_23_23_30_Pro.jpg','WIN_20261028_23_28_34_Pro.jpg','WIN_20261028_23_32_40_Pro.jpg']
results=[]
for name in names:
    p=folder/name
    records['inputs'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    a=np.array(Image.open(p).convert('L'));print('Native capture',name,flush=True)
    upright,turn,s0,s90=seg.UprightPage(a)
    ll,diag,stats=seg.SegmentLines(upright)
    records['captures'].append({'file':name,'input_size':[a.shape[1],a.shape[0]],'axis_turn':turn,'axis_scores':[s0,s90],**stats})
    sub=OUT/'camera'/p.stem;sub.mkdir(parents=True,exist_ok=True)
    for i,l in enumerate(ll):Image.fromarray(l['image']).save(sub/f'line_{i:02}.png')
    results.append((upright,ll,stats))
c=pdf('design-segmentation-variation',720,440)
for j,(a,ll,stats) in enumerate(results):
    panel(c,a,15+j*235,150,220,245,['Capture A: diffuse shadow','Capture B: striped shadow','Capture C: changed page angle'][j])
    text(c,15+j*235,132,f'{len(ll)} crops; deskew {stats["angle"]:+.1f} deg',10)
    # Same top-of-page writing ROI rather than pretending a failed crop is correct.
    roi=a[:a.shape[0]//3,:]
    panel(c,roi,15+j*235,34,220,72,'Detail of photographed writing')
text(c,15,10,'Actual 28 October photographs; illumination and angle were not instrumented.',10)
finish(c,'design-segmentation-variation')
(OUT/'camera_bench.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

# Native-resolution original-page example, retaining every actual crop.
p=ROOT/'Software/CNN/PersonalDataset/Owen/Pages/Owen01.jpg'
a=np.array(Image.open(p).convert('L'));print('Native original Owen01',flush=True)
upright,turn,*_=seg.UprightPage(a);ll,diag,stats=seg.SegmentLines(upright)
full,half,mask,ratio,clean,overlay,angle=stages(upright)
c=pdf('design-segmentation-stages',720,475)
for j,(im,title) in enumerate([(upright,'(a) Original Owen01'),(np.uint8(np.clip(ratio/1.2,0,1)*255),'(b) Local brightness ratio'),
                               (np.uint8(~mask)*255,'(c) Thresholded ink'),(overlay,'(d) Actual seam locations')]):
    panel(c,im,15+j*176,130,165,310,title)
for i,l in enumerate(ll[:2]):panel(c,l['image'],15,58-i*48,690,32,f'Output line {i+1}')
text(c,15,110,f'Native input: {a.shape[1]} x {a.shape[0]}; {len(ll)} crops; deskew {angle:+.1f} deg.',10)
finish(c,'design-segmentation-stages')
records['native_original']={'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),**stats}
sub=OUT/'native_Owen01';sub.mkdir(exist_ok=True)
for i,l in enumerate(ll):Image.fromarray(l['image']).save(sub/f'line_{i:02}.png')
(OUT/'camera_bench.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('Camera evidence complete',flush=True)
