from pathlib import Path
import json,hashlib,sys
import numpy as np
from scipy.spatial import cKDTree
from geometry_core import SEG,canvas
B=Path(__file__).resolve().parent;x=json.loads((B/'results.json').read_text());diff=[]
for r in x['results']:
 a=np.load(B/'glyphs'/f"{r['selected_reference']}.npz")['mask'];b=np.load(B/'glyphs'/f"{r['id']}.npz")['mask'];a,b=canvas(a,b);sa=SEG.Skeletonize(a);sb=np.roll(SEG.Skeletonize(b),(r['dy'],r['dx']),axis=(0,1));pa=np.argwhere(sa);pb=np.argwhere(sb);p=float((cKDTree(pa).query(pb)[0]<=4.8).mean());re=float((cKDTree(pb).query(pa)[0]<=4.8).mean());f=2*p*re/(p+re) if p+re else 0.;diff.append(abs(f-r['f1']));assert diff[-1]<1e-12
out={'independent_method':'Euclidean cKDTree nearest-neighbour queries on registered skeletons, without raster distance transforms','characters_checked':len(diff),'maximum_F_difference':max(diff),'neural_modules_loaded':[k for k in ['recognise','train','torch'] if k in sys.modules],'results_sha256':hashlib.sha256((B/'results.json').read_bytes()).hexdigest()};(B/'independent_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
