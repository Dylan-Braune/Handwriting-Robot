import os
os.environ['SWEEP_PART']='segmentation'
from common import *
full,half,mask,ratio,angle=S.Deskew(PAGE)
clean,paper,P,_,_=S.RemovePrintedStructure(mask.copy(),half.astype(float))
clean,lab,nc,area,_=S.DropArtifactPieces(clean,ratio,P,*clean.shape)
P,centres=S.LineCentres(clean,len(clean));H,W=clean.shape
cuts,ca,L=S.SeamCuts(clean,centres,P,H,W)
BASEOWN=S.AssignOwnership(clean,lab,nc,cuts,ca,L,centres,P,H,W)

def overlay(gray,cuts=None,centres=None,ownership=None):
    rgb=np.repeat(np.asarray(gray,dtype=np.uint8)[:,:,None],3,2)
    if ownership is not None:
        palette=np.array([[35,120,185],[195,82,40],[38,145,83],[135,70,157]])
        yy,xx=np.nonzero(ownership>=0);rgb[yy,xx]=palette[ownership[yy,xx]%4]
    im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
    if cuts is not None:
        for cut in cuts:d.line([(x,int(y)) for x,y in enumerate(cut)],fill='#d73b46',width=2)
    if centres is not None:
        for c in centres:d.line((0,c,W,c),fill='#1888c5',width=1)
    return np.asarray(im)

def segcase(img=PAGE):
    lines,aux,meta=S.SegmentLines(img)
    g=np.clip(aux['ratio']*235,0,255).astype('uint8')
    c=imagecase(overlay(g,aux['cuts']),line_count=len(lines),angle_deg=meta['angle'],pitch_px=meta['pitch'],removed_rule_px=meta['removedStraightPixels'])
    c['detail']=meta
    if lines:c['extra_crop']=lines[min(1,len(lines)-1)]['image'].tolist()
    return c

def run(id,title,gids,vals,expected,fn,metric=None,direction=None,input_name='Dylan01 photograph reduced to 1100 pixels maximum dimension; same pixels in every case',control=None):
    return experiment(id,title,'Segmentation',gids,vals,expected,fn,input_name,control or title,'segment.py / named stage; source-clone edits recorded per case',metric,direction,
                      'Masks and separators are actual code outputs. Red is a separator; blue is a centre. No line-content ground-truth accuracy is inferred from line count.')

def background(v):
    with edit(S,'InkMask',[('r = max(8,min(40,round(gray.shape[0]/41)))',f'r = {v}')]):
        m,r=S.InkMask(half.astype(float));c=imagecase(m.astype('uint8')*255,foreground_pixels=int(m.sum()),radius_px=v)
        c['detail']={'otsu_ratio_threshold':float(S.Otsu(r))};return c
run('P01','Local background radius',[44],[4,18,40],'Different windows change the local paper estimate and foreground. No monotone accuracy prediction follows.',background)
def ratio_threshold(v):
    with edit(S,'InkMask',[('mask = ratio<Otsu(ratio)',f'mask = ratio<(Otsu(ratio)+({v}))')]):
        m,r=S.InkMask(half.astype(float));raw=r<S.Otsu(r)+v
        c=imagecase(m.astype('uint8')*255,raw_foreground=int(raw.sum()),clean_foreground=int(m.sum()));return c
run('P02','Ink-ratio threshold offset',[45],[-.08,0,.08],'Raising the threshold must retain at least as many raw threshold pixels. Solid-region removal can undo part of this increase.',ratio_threshold,'raw_foreground','up')
def gamma(v):
    with edit(S,'_BuildLineCrop',[('**0.55',f'**{v}')]):
        lines,aux,meta=S.SegmentLines(PAGE);a=lines[1]['image'];return imagecase(a,mean_brightness=float(a.mean()),dark_pixels=int((a<128).sum()),line_count=len(lines))
run('P03','Crop darkness exponent',[46],[.3,.55,1.0],'For fixed crop ownership, larger darkness exponent makes intermediate strokes lighter; mean brightness should increase.',gamma,'mean_brightness','up')
def axis(v):
    with tune(S,AXIS_TURN_MARGIN=v):
        a,rot,s0,s90=S.UprightPage(np.rot90(PAGE));return imagecase(a,rotation_deg=rot,upright_score=s0,sideways_score=s90,decision_ratio=s90/max(s0,1e-9))
run('P04','Quarter-turn evidence margin',[47],[.5,1.35,50.0],'A sufficiently high margin suppresses a turn even when the alternate orientation scores better.',axis,input_name='The real photograph rotated 90 degrees before testing')
def deskew_range(v):
    a=S.Rotate(PAGE,8,fill=255)
    with edit(S,'Deskew',[('np.arange(-4,4.01,0.5)',f'np.arange(-{v},{v}+.01,.5)')]):
        full,_,_,_,best=S.Deskew(a);return imagecase(full,selected_deg=float(best),injected_rotation_deg=8.0)
run('P05','Deskew angular range',[48],[0,4,10],'A wider range permits a correction outside the old +/-4 degree search; it may also admit a different maximum.',deskew_range,input_name='Real photograph with a known additional +8 degree code rotation')
def deskew_step(v):
    a=S.Rotate(PAGE,2.3,fill=255)
    with edit(S,'Deskew',[('np.arange(-4,4.01,0.5)',f'np.arange(-4,4.01,{v})')]):
        f,_,_,_,best=S.Deskew(a);return imagecase(f,selected_deg=float(best),grid_step_deg=v)
run('P06','Deskew angular increment',[49],[.25,.5,2.0],'Finer steps permit more angular choices, not a guaranteed better estimate.',deskew_step)
def peak(v):
    with edit(S,'LinePitch',[('0.06*sm.max()',f'{v}*sm.max()')]),edit(S,'LineCentres',[('0.06*sm.max()',f'{v}*sm.max()')]):
        p,cc=S.LineCentres(clean,H);return imagecase(overlay(half,centres=cc),centres=len(cc),pitch=p)
run('P07','Row-peak strength floor',[50],[.02,.06,.30],'A higher floor rejects weak peaks; changed pitch estimation can also affect centre spacing.',peak)
def separation(v):
    with edit(S,'LineCentres',[('>=0.6*P',f'>={v}*P')]):
        p,cc=S.LineCentres(clean,H);return imagecase(overlay(half,centres=cc),centres=len(cc),pitch=p)
run('P08','Centre-separation multiplier',[51],[.3,.6,1.1],'With the same profile and pitch, larger minimum separation should admit fewer or equal centres.',separation,'centres','down')
def valley(v):
    p,cc=S.LineCentres(clean,H,mergeValley=v);return imagecase(overlay(half,centres=cc),centres=len(cc),pitch=p)
run('P09','Valley merge threshold',[52],[.1,.6,1.0],'Increasing the threshold makes merging less aggressive; at one the merge loop is disabled.',valley,'centres','up')
def pitchfloor(v):
    with tune(S,PEAK_MIN_PITCH_XH=v):
        p,cc=S.LineCentres(clean,H);return imagecase(overlay(half,centres=cc),centres=len(cc),pitch=p)
run('P10','Pitch floor in letter heights',[53],[.7,1.5,3.0],'A higher floor can reject a small-gap estimate and select a fallback; there can be plateaus and jumps.',pitchfloor)
for id,old,gid,vals,title in [('P11','1.25*P',54,[.5,1.25,3.0],'Horizontal printed-run length'),('P12','1.4*P',54,[.5,1.4,3.0],'Vertical printed-run length')]:
    def case(v,old=old):
        with edit(S,'RemovePrintedStructure',[(old,f'{v}*P')]):
            mm,_,pp,out,n=S.RemovePrintedStructure(mask.copy(),half.astype(float));return imagecase(mm.astype('uint8')*255,removed_pixels=n,remaining_foreground=int(mm.sum()),pitch=pp)
    run(id,title,[gid],vals,'Longer required runs should classify fewer straight structures as printed. This can preserve rules as well as handwriting.',case,'removed_pixels','down')

def seamcase():
    cc,arr,n=S.SeamCuts(clean,centres,P,H,W);s=arr[1:-1]
    cross=sum(int(clean[y,np.arange(W)].sum()) for y in s)
    move=sum(int(np.abs(np.diff(y)).sum()) for y in s)
    maxstep=max((int(np.abs(np.diff(y)).max()) for y in s),default=0)
    return imagecase(overlay(half,cc),ink_crossings=cross,total_vertical_motion=move,max_row_step=maxstep)
for id,title,gid,func,old,vals,fmt in [
('P13','Seam ink penalty',55,'SeamCuts','4.0*m',[0,4,20],lambda v:f'{v}*m'),
('P14','Seam midpoint attraction',56,'SeamCuts','0.012*((',[0,.012,1.0],lambda v:f'{v}*(('),
('P15','Seam vertical-movement penalty',57,'LeastCostSeam','0.02*np.abs',[0,.02,1.0],lambda v:f'{v}*np.abs'),
('P16','Seam step allowance',58,'SeamCuts','step=7',[1,3,7],lambda v:f'step={v}'),
('P17','Seam closing-radius fraction',59,'SeamCuts','int(0.07*P)',[0,.07,.25],lambda v:f'int({v}*P)')]:
    def case(v,func=func,old=old,fmt=fmt):
        with edit(S,func,[(old,fmt(v))]):return seamcase()
    run(id,title,[gid],vals,'Increasing the named cost favours less of the penalised behaviour, relative to other terms. Step and closing changes alter the feasible or priced paths; pixel quantisation can mask a small change.',case)
def ownership(v):
    with edit(S,'_BandOwnership',[('cnt>=0.15*',f'cnt>={v}*')]):
        own=S.AssignOwnership(clean,lab,nc,cuts,ca,L,centres,P,H,W)
        return imagecase(overlay(half,ownership=own),changed_owner_pixels=int((own!=BASEOWN).sum()),assigned_pixels=int((own>=0).sum()))
run('P18','Substantial ownership-overlap fraction',[60],[.03,.15,.45],'Larger fractions recognise fewer band overlaps as substantial; later fragment rules can compensate. Ownership colour reveals the final assignment.',ownership)
def minink(v):
    with edit(S,'SegmentLines',[('minInk = max(60,0.12*np.median(perLine[perLine>0]))',f'minInk = {v}')]):return segcase()
run('P19','Minimum crop ink count',[61],[0,60,3000],'A sufficiently high minimum discards short or sparse crops. Removed crops do not prove better classification.',minink,'line_count','down')
def solids(v,which):
    a=copy.deepcopy([{'image':x,'test_id':i} for i,x in enumerate(LINES[:4])]);blot=np.full((70,700),255,np.uint8);blot[15:55,40:620]=0;a.append({'image':blot,'test_id':4})
    kept=S.DropSolidCrops(a,**{which:v});show=np.full((5*85,700),255,np.uint8)
    ids={l['test_id'] for l in kept}
    for i,l in enumerate(a):
        im=Image.fromarray(l['image']);im.thumbnail((650,70));show[i*85:i*85+im.height,:im.width]=np.asarray(im)
        if i not in ids:show[i*85:i*85+3,:]=70
    c=imagecase(show,kept=len(kept),blot_kept=int(4 in ids));c['detail']={'kept_ids':sorted(ids),'marker':'dark top rule marks a rejected row; original candidate remains visible'};return c
run('P20','Solid-crop absolute ink limit',[62],[5,15,45], 'Higher absolute limits retain more dense candidates when the median-ratio condition also passes.',lambda v:solids(v,'maxAbsPct'),'kept','up',input_name='Four real line crops plus one deliberately constructed solid blot')
run('P21','Solid-crop median-ratio limit',[62],[1.2,3,10], 'A higher ratio is more permissive; both conditions must pass to reject a crop.',lambda v:solids(v,'maxMedianRatio'),'kept','up',input_name='Four real line crops plus one deliberately constructed solid blot')
savejson(OUT/'assets_segmentation.json',ASSETS)
