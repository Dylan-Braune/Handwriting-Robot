"""Read-only processing of real project inputs for report calculation traces.
No training epochs, optimizer updates, hardware calls or model/profile writes.
Inputs and code are hashed. Outputs stay in this revision's Evidence folder.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import sys,json,hashlib,copy,inspect
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];CODE=ROOT/'Software/CleanPipeline3'
sys.path.insert(0,str(CODE))
import segment as seg,recognise as rec,style,synthesise as syn,reproduce as rep,train
OUT=HERE/'Evidence';OUT.mkdir(exist_ok=True)
R={'scope':'Offline calculation traces on existing personal pages, trained reader/writer weights and final profiles. No training, optimizer update or physical motion. Reproduction is a generated software job, not an archived physical execution.',
   'inputs':{},'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.glob('*.py')},'records':{}}
records=R['records'];capture=True
def record(p):
    R['inputs'][str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return p
def save():
    (OUT/'real_calculations.json').write_text(json.dumps(R,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def floats(d):return {k:float(v) for k,v in d.items()}
def profile(frame,event,arg):
    if event!='return' or not capture:return
    name=frame.f_code.co_name;loc=frame.f_locals;file=Path(frame.f_code.co_filename).name
    if file=='segment.py':
        if name=='InkMask' and 'brightness' not in records:
            a=loc['gray'];rad=loc['r'];ratio=loc['ratio'];mask=arg[0]
            yy,xx=np.nonzero(mask);keep=(yy>a.shape[0]*.2)&(yy<a.shape[0]*.7)&(xx>a.shape[1]*.2)&(xx<a.shape[1]*.8)
            yy,xx=yy[keep],xx[keep];j=len(yy)//2;y,x=int(yy[j]),int(xx[j])
            pad=np.pad(a,rad,mode='edge');ii=seg.Integral(pad);k=2*rad+1
            corners=[ii[y+k,x+k],ii[y,x+k],ii[y+k,x],ii[y,x]];summ=corners[0]-corners[1]-corners[2]+corners[3]
            records['brightness']={'stage':'Owen01 reduced-page pre-deskew mask','shape':list(a.shape),'pixel_yx':[y,x],'radius':rad,'G':float(a[y,x]),'prefix_corners':list(map(float,corners)),'window_sum':float(summ),'window_area':k*k,'background':float(summ/(k*k)),'ratio':float(ratio[y,x]),'threshold':float(seg.Otsu(ratio)),'ink':bool(mask[y,x])}
        elif name=='Otsu' and not loc['isInt'] and 'otsu' not in records:
            w=loc['w'];m=loc['m'];t=loc['t'];w0=w[t];w1=1-w0
            records['otsu']={'bin':t,'threshold':float(arg),'w0':float(w0),'w1':float(w1),'mu0_bin':float(m[t]/w0),'mu1_bin':float((m[-1]-m[t])/w1),'criterion_bin_units':float(loc['var'][t])}
        elif name=='SkewScore':
            records.setdefault('skew_trials',[]).append({'angle':float(loc['angleDeg']),'score':float(arg)})
        elif name=='LeastCostSeam' and ('seam' not in records or not records['seam']['has_bend']):
            path=arg;cost=loc['cost'];acc=loc['acc'];step=loc['step'];changes=np.flatnonzero(np.diff(path))
            x=int(changes[0]+1 if len(changes) else len(path)//2);y=int(path[x]);py=int(path[x-1]);d=py-y
            cand=sorted([(float(acc[y+dd,x-1]+.02*abs(dd)),dd) for dd in range(-step,step+1) if 0<=y+dd<cost.shape[0]])
            records['seam']={'local_yx':[y,x],'previous_y':py,'d':d,'local_cost':float(cost[y,x]),'predecessor_cost':float(acc[py,x-1]),'movement_cost':.02*abs(d),'accumulated':float(acc[y,x]),'best_candidates':cand[:3],'has_bend':bool(len(changes))}
            caller=frame.f_back.f_locals;c0,c1=int(caller['c0']),int(caller['c1'])
            records['seam_cost']={'centres':[c0,c1],'y':y+c0,'x':x,'midpoint':(c0+c1)/2,'half_gap':(c1-c0)/2,'closed_mask':int(cost[y,x]>=4),'cost':float(cost[y,x]),'pitch':float(caller['P'])}
        elif name=='FitFixedCanvas' and 'fit' not in records:
            records['fit']={k:int(loc[k]) for k in ['h','w','height','width','newW','newH','y0']};records['fit']['scale']=float(loc['s'])
    if file=='recognise.py':
        if name=='Conv2d' and 'conv' not in records:
            xp=loc['xp'];weight=loc['weight'];bias=loc['bias'];scores=np.abs(arg[0,0]);y,x=map(int,np.unravel_index(scores.argmax(),scores.shape))
            patch=xp[0,:,y:y+3,x:x+3];terms=patch*weight[0]
            records['conv']={'output_channel':0,'yx':[y,x],'patch':patch.flatten().astype(float).tolist(),'weights':weight[0].flatten().astype(float).tolist(),'bias':float(bias[0]),'sum_products':float(terms.sum()),'output':float(arg[0,0,y,x])}
        elif name=='BatchNorm2d' and 'bn' not in records:
            y,x=records['conv']['yx'];j=(0,0,y,x)
            records['bn']=floats({'z':loc['x'][j],'mean':loc['runningMean'][0],'variance':loc['runningVar'][0],'gamma':loc['weight'][0],'beta':loc['bias'][0],'eps':loc['eps'],'output':arg[j]})
        elif name=='StrokeNormalize' and 'stroke_width' not in records and 'k' in loc:
            records['stroke_width']={'xh':float(loc['xh']),'rho':float(loc['penWidthXh']),'w':loc['w'],'k':loc['k']}
        elif name=='CtcForcedAlign' and 'viterbi' not in records and 'path' in loc:
            ts=np.flatnonzero(np.array(loc['ext'])[loc['path']]==rec.CHAR_TO_IDX['V']);t=int(ts[0]);s=int(loc['path'][t]);dp=loc['dp'];ext=loc['ext'];pred=[s,s-1]
            if s>=2 and ext[s]!=0 and ext[s]!=ext[s-2]:pred.append(s-2)
            pred=[r for r in pred if r>=0]
            records['viterbi']={'frame':t,'state':s,'character':'V','emission_logp':float(loc['logProbs'][t,ext[s]]),'predecessor_scores':[float(dp[t-1,r]) for r in pred],'value':float(dp[t,s]),'span':arg[0][5]}
        elif name=='forward' and isinstance(loc.get('self'),rec.PaperCRNNNumpy):
            saved['raw_logits']=loc['logits'].copy()
    if file=='train.py' and name=='CtcLoss' and 'ctc' not in records:
        lp=loc['lp'];la=loc['logAlpha'];lb=loc['logBeta'];occ=loc['occ'];labels=loc['labels'];ext=loc['ext']
        v=records.get('viterbi');t=v['frame'] if v else 15;k=rec.CHAR_TO_IDX['V'];s=1+2*truth.index('V');pred=[s,s-1]
        if s>=2 and ext[s]!=ext[s-2]:pred.append(s-2)
        vals=[float(la[t-1,r]) for r in pred if np.isfinite(la[t-1,r])]
        records['ctc']={'loss':float(arg[0]),'logZ':float(loc['logP']),'frame':t,'class':k,'character':'V','p':float(loc['probs'][t,0,k]),'q':float(occ[t,k]),'gradient':float(arg[1][t,0,k]),'state':s,'predecessor_logalpha':vals,'emission_logp':float(lp[t,k]),'logalpha':float(la[t,s])}
    if file=='synthesise.py' and name=='place' and 'placement' not in records and arg and frame.f_back.f_locals.get('text')=='Writing with style.':
        u,v=loc['gStrokes'][0][0];out=arg[0][0]
        records['placement']=floats({'u':u,'v':v,'lead':loc['lead'],'shear':loc['glyphSlant'],'drift':loc['glyphDrift'],'xh_mm':loc['xh'],'penX':loc['px'],'penY':loc['penY'],'x':out[0],'y':out[1]})
    if file=='reproduce.py':
        if name=='ToMachine' and 'machine' not in records:
            cfg=loc['cfg'];p=loc['traj'].strokes[0][0]
            records['machine']=floats({'x':p[0],'y':p[1],'x0':loc['x0'],'y1':loc['y1'],'scale':loc['scale'],'ox':cfg.originXmm,'oy':cfg.originYmm,'mx':loc['mx'],'my':loc['my'],'off':loc['off']})
            records['machine']['mapped_x']=float(cfg.originXmm+loc['mx']+(p[0]-loc['x0'])*loc['scale'])
            records['machine']['mapped_y']=float(cfg.originYmm-loc['my']-loc['off']-(loc['y1']-p[1])*loc['scale'])
        elif name=='WriteGcode':
            records['timing']={**floats({k:loc[k] for k in ['drawMm','travelMm','drawS','travelS','dwellS']}),'pulses':loc['pen'][1],'returned':arg}
    return
saved={}

def lineno(fun,fragment):
    lines,start=inspect.getsourcelines(fun)
    return next(start+i for i,s in enumerate(lines) if fragment in s)
lstmline=lineno(rec.LstmForward,'c = f*c+i*g')
glyphline=lineno(style.ExtractLineGlyphs,'q.append((round(xn,3)')
offsetline=lineno(style.ExtractLineGlyphs,'norm = [[(round(px-x0r')
thinline=lineno(seg.Skeletonize,'img[go] = 0')
def trace(frame,event,arg):
    if event=='call':
        return trace if frame.f_code in {rec.LstmForward.__code__,style.ExtractLineGlyphs.__code__,seg.Skeletonize.__code__} else None
    if event!='line':return trace
    d=frame.f_locals;line=frame.f_lineno
    if line==lstmline and d['t']==10 and 'lstm' not in records:
        vals={k:float(d[k][0,0]) for k in ['c','i','f','g','o']};vals['frame']=10;vals['c_new']=vals['f']*vals['c']+vals['i']*vals['g'];vals['h_new']=vals['o']*np.tanh(vals['c_new']);records['lstm']=vals
    elif line==glyphline and d['ch']=='V' and 'glyph_point' not in records:
        records['glyph_point']={**floats({k:d[k] for k in ['x','y','left','base','xh','shear','xn','yn']}),'character':'V'}
    elif line==offsetline and d['ch']=='V' and 'glyph_point' in records and 'offset' not in records['glyph_point']:
        records['glyph_point']['offset']=float(d['x0r'])
    elif line==thinline and d['step']==0 and 'thinning' not in records:
        y,x=map(int,np.argwhere(d['go'])[0]);ns=[int(d[f'P{k}'][y,x]) for k in range(2,10)]
        records['thinning']={'stage':'first skeletonisation call in trace','yx':[y,x],'neighbours':ns,'count':int(d['B'][y,x]),'transitions':int(d['A'][y,x]),'subpass':0}
    return trace

page=record(ROOT/'Software/CNN/PersonalDataset/Owen/Pages/Owen01.jpg')
im=Image.open(page).convert('L');im.thumbnail((1600,1600));a=np.array(im)
sys.setprofile(profile)
full,half,mask,ratio,angle=seg.Deskew(a)
clean,_,pitch,_,_=seg.RemovePrintedStructure(mask,half.astype(float))
clean,lab,nc,area,na=seg.DropArtifactPieces(clean,ratio,pitch,*clean.shape)
pitch,centres=seg.LineCentres(clean,clean.shape[0]);cuts,_,_=seg.SeamCuts(clean,centres,pitch,*clean.shape)
save();print('Page calculation trace complete',flush=True)
crop=record(ROOT/'Handwriting-Robot-Pipeline/Software/CNN/Crops/Owen/Owen01/00.png');gray=np.array(Image.open(crop).convert('L'))
label=record(ROOT/'Software/CNN/PersonalDataset/Owen/Labels/Owen01_labels.txt');truth=label.read_text(encoding='utf-8-sig').splitlines()[0].strip()
weights=record(ROOT/'Software/CNN/weights/paper_cnn_bilstm_ctc_joint_best.npz');model=rec.PaperCRNNNumpy(weights)
sys.settrace(trace)
lp=model.forward(rec.TensorFromResized(rec.ResizeLineImageFixed(gray)))
alignment=rec.CtcForcedAlign(lp[:,0,:],truth)
records['reader']={'transcript':truth,'decoded':rec.DecodeCtc(lp)[0],'shape':list(lp.shape),'crop_shape':list(gray.shape),'alphabet_count':len(rec.CHAR_TO_IDX),'head_weight_shape':list(model.sd['text_head.weight'].shape)}
targets=[[rec.CHAR_TO_IDX[c] for c in truth if c in rec.CHAR_TO_IDX]]
z=saved['raw_logits'].astype(np.float64);loss,grad=train.CtcLoss(z,targets)
capture=False
idx=(records['ctc']['frame'],0,records['ctc']['class']);eps=1e-4
zp=z.copy();zp[idx]+=eps;zm=z.copy();zm[idx]-=eps
lplus=train.CtcLoss(zp,targets)[0];lminus=train.CtcLoss(zm,targets)[0]
records['finite_difference']={'index':idx,'epsilon':eps,'Lplus':float(lplus),'Lminus':float(lminus),'estimate':float((lplus-lminus)/(2*eps)),'analytic':float(grad[idx])}
capture=True
class Cached:
    def forward(self,x):return lp
glyphs,stats=style.ExtractLineGlyphs(gray,truth,Cached())
if 'glyph_point' in records:
    q=records['glyph_point'];q['stored_u']=round(round(q['xn'],3)-q['offset'],3);q['stored_v']=round(q['yn'],3);q['lead']=round(float(np.clip(q['offset'],0,1.5)),3)
    assert any((q['stored_u'],q['stored_v']) in s for s in glyphs[truth.index('V')]['strokes'])
records['cut']={}
# Extract the real column cost at the V's right boundary without altering extraction.
ink=seg.BinarizeLine(gray);top,base=seg.CoreBand(ink);xh=base-top;col=ink.sum(0).astype(float);k=max(3,int(round(xh*.12)))|1
smooth=np.convolve(col,np.ones(k)/k,mode='same');ref=float(np.percentile(smooth[smooth>0],60));highest=np.where(ink,np.arange(ink.shape[0])[:,None],ink.shape[0]).min(0)
cut=next(c for c in stats['cuts'] if c[1]=='V');x=int(round(cut[3]));height=float(np.clip((base-highest[x])/max(1,xh),0,3));cost=0 if col[x]==0 else float(smooth[x]/ref+1.15*height)
records['cut']={'character':'V','right_boundary':float(cut[3]),'column':x,'smooth_ink':float(smooth[x]),'reference_percentile':ref,'height_above':height,'raw_ink_count':float(col[x]),'cost':cost}
pairs=[g for j,g in enumerate(glyphs[:-1]) if g and g.get('char')!=' ' and glyphs[j+1] and glyphs[j+1].get('char')!=' ' and 'connR' in g]
records['line_connectedness']={'adjacent':len(pairs),'crossed':sum(bool(g.get('connR')) for g in pairs),'fraction':sum(bool(g.get('connR')) for g in pairs)/max(1,len(pairs)),'scope':'extracted Owen01 line 1 only; not a rebuilt final profile'}
rec.StrokeNormalize(gray,penWidthXh=.17)
sys.settrace(None)
save();print('Reader, alignment and glyph traces complete',flush=True)

profile_path=record(ROOT/'Software/CNN/NOGIT/StyleProfiles10/owen_curated.json');p=json.loads(profile_path.read_text(encoding='utf-8'))
med=float(np.median([g.get('priorD',.5) for vs in p['glyphs'].values() for g in vs]));records['anchoring']={'median_distance':med,'lambda':syn.AutoLegibility(p),'stored_override':p.get('legibilityLambda')}
traj=syn.SynthesizeText('Writing with style.',copy.deepcopy(p),seed=2026,mmPerXh=4,lineWidthMm=170,jitter=.15)
cfg=rep.GantryConfig();rep.WriteGcode(traj,cfg,OUT/'real_trace_owen.gcode',title='Offline calculation trace; never executed')
records['config']={k:getattr(cfg,k) for k in ['fullStepsPerRev','microstepping','mmPerRevX','mmPerRevY','stepsPerMmX','stepsPerMmY','boundsMaxXmm','boundsMaxYmm','originYmm','pageMarginXmm','pageMarginYmm','drawFeedMmMin','travelFeedMmMin','penUpDwellMs','penDownDwellMs']}
sys.setprofile(None)
# Use actual generated file segments to evaluate pulse timing at nominal conversion.
paths=rep.ParseGcode(OUT/'real_trace_owen.gcode');s=next(s for s in paths if len(s)>=2 and np.linalg.norm(np.array(s[1])-s[0])>.05 and all(abs(s[1][j]-s[0][j])>.02 for j in [0,1]))
aa,bb=s[:2];dx=int(round(bb[0]*cfg.stepsPerMmX))-int(round(aa[0]*cfg.stepsPerMmX));dy=int(round(bb[1]*cfg.stepsPerMmY))-int(round(aa[1]*cfg.stepsPerMmY));nm=max(abs(dx),abs(dy));nn=min(abs(dx),abs(dy));dist=float(np.linalg.norm(np.array(bb)-aa));tau=max(dist/((cfg.drawFeedMmMin/60)*nm),2e-6)
records['motion_segment']={'from':aa,'to':bb,'dx_steps':dx,'dy_steps':dy,'major':nm,'minor':nn,'initial_error':nm//2,'distance_mm':dist,'feed':cfg.drawFeedMmMin,'tau_s':tau,'delay_s':(nm+nn)*tau,'nominal_scale':True,'scope':'software-only pulse-delay calculation from actual generated segment; no recovered homing count or physical movement'}
shape=record(Path('C:/Users/braun/Downloads/author_shape_11_pw170_weights.npz'));namer=rec.AuthorClassifierCNNNumpy(shape)
probs=[];weights_line=[]
for i in [0,1]:
    cp=record(crop.with_name(f'{i:02}.png'));g=np.array(Image.open(cp).convert('L'));norm=rec.StrokeNormalize(g,penWidthXh=.17)
    pr=rec.Softmax(namer.forward(rec.TensorFromResized(rec.ResizeLineImageFixed(norm))),axis=1)[0]
    probs.append(float(pr[namer.authorMapping['owen']]));weights_line.append(int((g<128).sum()))
records['vote']={'scope':'first two archived original Owen01 crops only; not a whole-page classification result','ink_weights':weights_line,'owen_probabilities':probs,'pooled':float(np.dot(weights_line,probs)/sum(weights_line))}
dylan=record(ROOT/'Software/CNN/NOGIT/StyleProfiles10/dylan_curated.json');dp=json.loads(dylan.read_text(encoding='utf-8'));gs=dp['glyphs']['a'];G=np.stack([style._ShapeGrid(g) for g in gs]);D=np.abs(G[:,None]-G[None,:]).sum(-1)
records['glyph_pool']={'writer':'dylan','character':'a','variants':len(gs),'sum_L1_distances':D.sum(1).astype(float).tolist(),'medoid_recomputed_index':int(np.argmin(D.sum(1))),'priorD_first_variant':float(gs[0]['priorD']),'scope':'descriptor comparison recomputed on stored final variants; not a historical filtering log'}
save();print('All real calculation traces saved',flush=True)
