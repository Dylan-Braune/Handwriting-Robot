"""Post-design sensitivity, not historical tuning or a qualification test.
No training, learned-model execution, profile changes or hardware access.
"""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
CODE=ROOT/'Software/CleanPipeline3';sys.path.insert(0,str(CODE))
import segment as seg

def seam(cost,penalty,step=7):
    h,width=cost.shape;acc=cost.copy();back=np.zeros((h,width),np.int8)
    idx=np.arange(h);shifts=range(-step,step+1)
    for x in range(1,width):
        previous=acc[:,x-1];candidates=np.full((len(shifts),h),np.inf)
        for i,d in enumerate(shifts):
            if d<0:candidates[i,-d:]=previous[:d]+penalty*abs(d)
            elif d>0:candidates[i,:-d]=previous[d:]+penalty*d
            else:candidates[i]=previous
        j=candidates.argmin(0);acc[:,x]+=candidates[j,idx];back[:,x]=j-step
    y=np.empty(width,int);y[-1]=acc[:,-1].argmin()
    for x in range(width-1,0,-1):y[x-1]=y[x]+back[y[x],x]
    return y

out={'scope':'Post-design one-at-a-time seam sensitivity on six preselected gaps; no character/line ground-truth boundary accuracy or global optimum claimed.',
 'inputs':{},'code_sha256':{'segment.py':hashlib.sha256((CODE/'segment.py').read_bytes()).hexdigest()},'runs':[],
 'settings':{'thumbnail_max_pixels':[1600,1600],'step':7,'ink_weight':4.,'baseline_midpoint_weight':.012,'baseline_step_penalty':.02},'baseline_reference_matches':0}
settings=[(a,.02) for a in [0.,.006,.012,.024,.048]]+[(.012,b) for b in [0.,.01,.04]]
for writer in ['Owen','Yeukita']:
    path=ROOT/f'Software/CNN/PersonalDataset/{writer}/Pages/{writer}01.jpg'
    out['inputs'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    img=Image.open(path).convert('L');img.thumbnail((1600,1600));a=np.array(img)
    upright,turn,_,_=seg.UprightPage(a)
    full,half,mask,ratio,angle=seg.Deskew(upright)
    clean,paper,pitch,_,_=seg.RemovePrintedStructure(mask,half.astype(float))
    clean,lab,nc,area,na=seg.DropArtifactPieces(clean,ratio,pitch,*clean.shape)
    pitch,centres=seg.LineCentres(clean,clean.shape[0])
    radius=max(2,int(.07*pitch))
    md=seg.AnyWithinRadius(seg.AnyWithinRadius(clean,radius,0),radius,1)
    closed=~seg.AnyWithinRadius(seg.AnyWithinRadius(~md,radius,0),radius,1)
    gaps=sorted(set([0,(len(centres)-2)//2,len(centres)-2]))
    for gap in gaps:
        c0,c1=centres[gap:gap+2];yy=np.arange(c0,c1+1)[:,None]
        mid=c0+.5*(c1-c0);hg=max(1.,(c1-c0)/2)
        raw=clean[c0:c1+1];ink=closed[c0:c1+1].astype(float)
        for alpha,beta in settings:
            cost=4*ink+alpha*((yy-mid)/hg)**2
            line=seam(cost,beta);xx=np.arange(len(line))
            if alpha==.012 and beta==.02:
                assert np.array_equal(line,seg.LeastCostSeam(cost,step=7))
                out['baseline_reference_matches']+=1
            out['runs'].append({'writer':writer,'gap_after_centre':int(gap),'midpoint_weight':alpha,'step_penalty':beta,'columns':len(line),
                'raw_ink_intersections':int(raw[line,xx].sum()),'closed_ink_intersections':int(ink[line,xx].sum()),
                'vertical_step_sum':int(np.abs(np.diff(line)).sum()),'mean_midpoint_offset_pixels':float(np.abs(line+c0-mid).mean()),'axis_turn':int(turn),'deskew_angle':float(angle)})
    print(writer,'finished',flush=True)
out['summary']=[]
for alpha,beta in settings:
    runs=[r for r in out['runs'] if r['midpoint_weight']==alpha and r['step_penalty']==beta]
    out['summary'].append({'midpoint_weight':alpha,'step_penalty':beta,'seams':len(runs),
      'raw_ink_intersections':sum(r['raw_ink_intersections'] for r in runs),
      'closed_ink_intersections':sum(r['closed_ink_intersections'] for r in runs),
      'vertical_step_sum':sum(r['vertical_step_sum'] for r in runs),
      'mean_midpoint_offset_pixels':sum(r['mean_midpoint_offset_pixels'] for r in runs)/len(runs)})
(HERE/'Evidence/coefficient_sensitivity.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out['summary'],indent=2))
