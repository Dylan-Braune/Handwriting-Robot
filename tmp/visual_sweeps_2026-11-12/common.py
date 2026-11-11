import os,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
sys.dont_write_bytecode=True
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.environ['CROP_WORKERS']='1';os.environ['LINE_WORKERS']='1'
sys.path.insert(0,str(BASE/'deps'));sys.path.insert(0,str(BASE/'snapshot'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps,ImageEnhance,ImageFilter,ImageDraw,ImageFont
import json,time,hashlib,copy,inspect,contextlib,traceback,re,ast
import paths
paths.Data=ROOT/'Software/CNN';paths.Weights=paths.Data/'weights';paths.Profiles=paths.Data/'NOGIT/StyleProfiles10'
paths.PersonalPages=paths.Data/'PersonalDataset'
import segment as S,recognise as R,style as T,synthesise as Y,reproduce as G
OUT=BASE/'results';OUT.mkdir(exist_ok=True)
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
RAW=OUT/'raw';RAW.mkdir(exist_ok=True)
for forbidden in ['Connect','Calibrate','RunGcodeFile','_RunGcodeLines','EmergencyStop','RequestAbort','BresenhamMove','_StepAxis','_GcodeLinearMove','_GcodeArcMove']:
    def blocked(*a,**kw):raise RuntimeError('Hardware execution is blocked in this harness')
    setattr(G,forbidden,blocked)
IDENTITY={'shearDelta':0.,'ascK':1.,'descK':1.}
PROFILES={p.stem.replace('_curated',''):json.load(open(p,encoding='utf-8')) for p in paths.Profiles.glob('*_curated.json')}
PAGE=np.asarray(Image.open(BASE/'input_page.png').convert('L'))
LINES=[np.asarray(Image.open(p).convert('L')) for p in sorted((BASE/'baseline_segments').glob('*.png'))]
LINE=LINES[1]
TRUTH=(paths.PersonalPages/'Dylan/Labels/Dylan01_labels.txt').read_text(encoding='utf-8-sig').splitlines()[1].strip()
RESULTS=[];PATCHES=[];ASSETS={};MODELS={}

def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)

def savejson(path,obj):Path(path).write_text(json.dumps(obj,indent=2,default=plain,allow_nan=False),encoding='utf-8')

@contextlib.contextmanager
def tune(mod,**kw):
    old={k:getattr(mod,k) for k in kw}
    PATCHES.append({'kind':'in-memory binding','module':mod.__name__,'changes':{k:{'before':str(old[k]),'after':str(v)} for k,v in kw.items()}})
    try:
        for k,v in kw.items():setattr(mod,k,v)
        yield
    finally:
        for k,v in old.items():setattr(mod,k,v)

@contextlib.contextmanager
def edit(mod,name,pairs):
    orig=getattr(mod,name);src=inspect.getsource(orig)
    for a,b in pairs:
        assert a in src,(name,a)
        src=src.replace(a,b)
    PATCHES.append({'kind':'isolated function clone','module':mod.__name__,'function':name,'replacements':pairs})
    ns={};exec(compile(src,str(BASE/'snapshot'/f'{mod.__name__}.py')+' [sweep clone]','exec'),mod.__dict__,ns)
    setattr(mod,name,ns[name])
    try:yield
    finally:setattr(mod,name,orig)

def model(kind='writer',filename=None):
    filename=filename or ('author_shape_11w_pw170_scratch_weights.npz' if kind=='writer' else 'paper_cnn_bilstm_ctc_joint_best.npz')
    if filename not in MODELS:
        p=paths.Weights/filename;ASSETS[str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        MODELS[filename]=(R.AuthorClassifierCNNNumpy if kind=='writer' else R.PaperCRNNNumpy)(p)
    return MODELS[filename]

def classify(img,m=None,norm=False):
    m=m or model();idx,prob=R.ClassifyImage(img,m,strokeNormalise=norm)
    inv={v:k for k,v in m.authorMapping.items()}
    return {'writer':inv[idx],'confidence':float(prob[idx]),'p_dylan':float(prob[m.authorMapping['dylan']])},prob

def trajectory(text='hello world hello',writer='dylan',profile=None,**kw):
    p=copy.deepcopy(profile or PROFILES[writer]);p.pop('_gapRange',None)
    args=dict(mmPerXh=4.,seed=17,lineWidthMm=180.,jitter=0.,legibility=0.,_cal=IDENTITY)
    args.update(kw)
    return Y.SynthesizeText(text,p,**args)

def trajcase(tr):
    x0,y0,x1,y1=tr.Bounds();down,up=tr.PenTravelMm()
    return {'kind':'image','image':np.asarray(Y.RenderTrajectory(tr,pxPerMm=10)), 'extent_scale':.1,
            'trajectory':{'strokes':tr.strokes,'meta':tr.meta},
            'metrics':{'width_mm':round(x1-x0,3),'height_mm':round(y1-y0,3),'strokes':len(tr.strokes),
                       'draw_mm':round(down,3),'lines':tr.meta.get('lines',1),'core_glyphs':tr.meta.get('glyphSources',{}).get('core',0)},
            'detail':tr.meta}

def imagecase(a,**metrics):return {'kind':'image','image':np.asarray(a),'metrics':metrics}

def chartcase(labels,vals,**metrics):return {'kind':'chart','labels':list(labels),'values':list(vals),'metrics':metrics}

def pathcase(strokes,**metrics):return {'kind':'path','strokes':strokes,'metrics':metrics}

def glyphsheet(gs):
    strokes=[]
    for i,g in enumerate(gs):
        if g:
            for s in g['strokes']:strokes.append([(x+3*(i%12),y-4*(i//12)) for x,y in s])
    return pathcase(strokes,variants=len(gs))

def figure(exp):
    cases=exp['cases'];n=len(cases);cols=2 if n>3 else 1
    ratios=[c['image'].shape[0]/c['image'].shape[1] for c in cases if c.get('image') is not None]
    rowheight=2.5
    if ratios and min(ratios)>.6:cols=min(3,n);rowheight=5.8
    elif ratios and max(ratios)<.35:rowheight=1.4
    rows=(n+cols-1)//cols
    fig,axs=plt.subplots(rows,cols,figsize=(10.3,rowheight*rows+0.4),squeeze=False)
    fig.patch.set_facecolor('white');kind=cases[0].get('kind')
    ims=[c.get('image') for c in cases if c.get('image') is not None]
    maxw=max((im.shape[1]*c.get('extent_scale',1) for c in cases if (im:=c.get('image')) is not None),default=1)
    maxh=max((im.shape[0]*c.get('extent_scale',1) for c in cases if (im:=c.get('image')) is not None),default=1)
    pathspts=[p for c in cases for s in c.get('strokes',[]) for p in s]
    if pathspts:
        pp=np.asarray(pathspts);lo=pp.min(0);hi=pp.max(0);margin=np.maximum((hi-lo)*.04,.15)
    for ax,c in zip(axs.flat,cases):
        ax.set_title(str(c['value']),loc='left',fontweight='bold',fontsize=11,color='#17344a')
        if c['kind']=='image':
            im=c['image'];scale=c.get('extent_scale',1)
            ax.imshow(im,cmap='gray',vmin=0,vmax=255,extent=(0,im.shape[1]*scale,im.shape[0]*scale,0),interpolation='nearest')
            ax.set_xlim(0,maxw);ax.set_ylim(maxh,0);ax.set_aspect('equal');ax.set_facecolor('#f3f5f7')
            ax.set_xlabel('mm in preview' if scale!=1 else 'output pixels',fontsize=8)
        elif c['kind']=='path':
            for s in c['strokes']:
                if s:ax.plot(*np.asarray(s).T,color='#126b91',linewidth=1.3)
            for s in c.get('travel',[]):ax.plot(*np.asarray(s).T,color='#c57a24',linewidth=.7,linestyle='--')
            if pathspts:ax.set_xlim(lo[0]-margin[0],hi[0]+margin[0]);ax.set_ylim(lo[1]-margin[1],hi[1]+margin[1])
            ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2);ax.set_xlabel(c.get('units','stage coordinates'),fontsize=8)
        elif c['kind']=='chart':
            ax.bar(np.arange(len(c['values'])),c['values'],color='#126b91');ax.set_xticks(np.arange(len(c['values'])),c['labels'],rotation=25,ha='right',fontsize=7);ax.grid(axis='y',alpha=.2)
        else:ax.text(.02,.8,c.get('error','No output'),wrap=True,transform=ax.transAxes,color='#8b2e3a');ax.axis('off')
    for ax in list(axs.flat)[n:]:ax.axis('off')
    fig.tight_layout(h_pad=2.0,w_pad=2)
    target=FIG/(exp['id']+'.png');fig.savefig(target,dpi=165,bbox_inches='tight');plt.close(fig)
    return str(target)

def experiment(id,title,stage,guides,values,expected,fn,input_name,control,source,metric=None,direction=None,qualification=''):
    exp={'id':id,'title':title,'stage':stage,'guide_ids':guides,'values':values,'expected':expected,'input':input_name,
         'control':control,'source':source,'qualification':qualification,'metric':metric,'direction':direction,'cases':[]}
    # Persist the protocol before running any member of the sweep.
    savejson(RAW/(id+'_protocol.json'),exp)
    print('START',id,title,flush=True)
    for j,v in enumerate(values):
        PATCHES.clear();t=time.perf_counter()
        try:c=fn(v);c['execution']='completed'
        except Exception as e:
            c={'kind':'error','metrics':{},'error':str(e),'traceback':traceback.format_exc(),'execution':'failed'}
            print('FAIL',id,v,str(e),flush=True)
        c['value']=v;c['wall_seconds']=round(time.perf_counter()-t,5);c['instrumentation']=copy.deepcopy(PATCHES)
        if 'image' in c:
            p=RAW/f'{id}_{j:02d}.png';Image.fromarray(c['image']).save(p);c['output_image']=str(p)
        if 'trajectory' in c:savejson(RAW/f'{id}_{j:02d}_trajectory.json',c['trajectory'])
        disk={k:x for k,x in c.items() if k not in ['image','trajectory']};savejson(RAW/f'{id}_{j:02d}.json',disk)
        exp['cases'].append(c)
    exp['figure']=figure(exp)
    vals=[c['metrics'].get(metric) for c in exp['cases']] if metric else []
    if any(c['execution']=='failed' for c in exp['cases']):exp['verdict']='Execution failure recorded'
    elif metric and all(isinstance(x,(int,float)) for x in vals):
        dif=np.diff(vals)
        good=(np.all(dif>=-1e-7) if direction=='up' else np.all(dif<=1e-7) if direction=='down' else np.ptp(vals)<1e-7 if direction=='same' else True)
        exp['verdict']='Prediction supported on these inputs' if good else 'Prediction not monotone / contradicted on these inputs'
        if np.ptp(vals)<1e-7 and direction in ['up','down']:exp['verdict']='No measured change on these inputs'
        exp['observation']=f'{metric}: '+', '.join(f'{v} -> {x:.5g}' for v,x in zip(values,vals))
    else:
        exp['verdict']='Observed outputs recorded; qualitative / conditional prediction'
        exp['observation']='; '.join(str(c['value'])+': '+str(c['metrics']) for c in exp['cases'])
    RESULTS.append(exp)
    savejson(OUT/('results_'+os.environ.get('SWEEP_PART','all')+'.json'),[{**e,'cases':[{k:v for k,v in c.items() if k not in ['image','trajectory']} for c in e['cases']]} for e in RESULTS])
    print('DONE',id,exp['verdict'],flush=True)
    return exp
