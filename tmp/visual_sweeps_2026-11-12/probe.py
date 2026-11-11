import sys,os,json,time
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
sys.dont_write_bytecode=True
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,str(BASE/'snapshot'))
import paths
paths.Data=ROOT/'Software/CNN';paths.Weights=paths.Data/'weights';paths.Profiles=paths.Data/'NOGIT/StyleProfiles10'
import numpy as np
from PIL import Image,ImageOps
import segment as S,recognise as R,synthesise as Y,style as T,reproduce as G
t=time.perf_counter()
p=json.load(open(paths.Profiles/'dylan_curated.json'))
tr=Y.SynthesizeText('hello world',p,seed=17,_cal={'shearDelta':0.,'ascK':1.,'descK':1.})
print('synth',time.perf_counter()-t,tr.meta,flush=True)
print('glyph example',list(p['glyphs']['a'][0].keys()),flush=True)
im=ImageOps.exif_transpose(Image.open(paths.PersonalPages/'Dylan/Pages/Dylan01.jpg')).convert('L') if False else ImageOps.exif_transpose(Image.open(ROOT/'Software/CNN/PersonalDataset/Dylan/Pages/Dylan01.jpg')).convert('L')
im.thumbnail((1100,1500));im.save(BASE/'input_page.png')
t=time.perf_counter();lines,aux,meta=S.SegmentLines(im)
print('segment',time.perf_counter()-t,meta,[(x['image'].shape,x['bbox']) for x in lines[:3]],flush=True)
(BASE/'baseline_segments').mkdir(exist_ok=True)
for i,c in enumerate(lines):Image.fromarray(c['image']).save(BASE/'baseline_segments'/f'line_{i:02d}.png')
for kind,fn,cls in [('writer','author_shape_11w_pw170_scratch_weights.npz',R.AuthorClassifierCNNNumpy),('reader','paper_cnn_bilstm_ctc_joint_best.npz',R.PaperCRNNNumpy)]:
 t=time.perf_counter();m=cls(paths.Weights/fn)
 result=R.ClassifyImage(lines[1]['image'],m,strokeNormalise=False) if kind=='writer' else R.ReadTextScored(lines[1]['image'],m)
 print(kind,time.perf_counter()-t,repr(result),getattr(m,'authorMapping',None),flush=True)
