import os
os.environ['SWEEP_PART']='synthesis'
from common import *
INPUT='Real stored Dylan glyph profile; text hello world hello; seed 17; 4 mm/x-height. Identity calibration isolates local changes.'
def sweep(id,title,gids,vals,expected,key=None,metric=None,direction=None,writer='dylan',text='hello world hello',fn=None,**fixed):
    def case(v):
        p=copy.deepcopy(PROFILES[writer]);kw=dict(fixed)
        if key and key.startswith('profile:'):p[key.split(':',1)[1]]=v
        elif key:kw[key]=v
        return trajcase(trajectory(text,profile=p,**kw))
    return experiment(id,title,'Synthesis',gids,vals,expected,fn or case,INPUT,key or title,'synthesise.py / SynthesizeText and named helpers',metric,direction,
                      'Single-candidate stage isolation; no repair. Fixed seed does not keep random consumption fixed after branch changes.')

sweep('S01','Connectedness below and across its joining floor',[1,2,4],[0,.30,.44,.45,.75,1.0],
      'Below 0.45 there should be no synthesis joins. Above the floor, eligible pairs may join; stroke counts need not be monotone because branch-dependent randomness also changes.',key='profile:connectedness')
def cursive(v):
    p=copy.deepcopy(PROFILES['dylan']);p.update(connectedness=v,glyphs={},idealMedoid={},idealGlyphs={})
    return trajcase(trajectory('hello hello',profile=p))
sweep('S02','Fallback alphabet switch at connectedness 0.5',[3],[.44,.49,.50,.80],
      'With personal shapes deliberately absent, crossing 0.5 changes the designed fallback alphabet and its join eligibility.',fn=cursive)
def constant_case(name,v,**kw):
    with tune(Y,**{name:v}):return trajcase(trajectory(**kw))
sweep('S03','Maximum horizontal synthesis join reach',[6],[0,.4,.8,1.6],
      'A larger reach admits additional eligible attachments; it cannot force joins rejected by other gates.',fn=lambda v:constant_case('MAX_JOIN_GAP_XH',v,profile={**PROFILES['dylan'],'connectedness':1.0}))
for id,title,gids,key,vals,exp,metric,direction in [
('S04','Overall slant',[7],'profile:slantDeg',[-20,0,20],'Shear should move tall points horizontally; overall width may change. Caps and collision correction still apply.',None,None),
('S05','Physical x-height',[8],'mmPerXh',[2,4,6],'Increasing size should enlarge geometry before bounds fitting. Text fits one wide line here.','height_mm','up'),
('S06','Wrapping width',[9],'lineWidthMm',[60,100,180],'Narrowing width should cause at least as many line breaks; it does not compress glyphs.','lines','down'),
('S07','Word-space profile multiplier',[10],'profile:wordSpaceXh',[.3,1.79,4.0],'Larger word gaps should increase the unwrapped span.','width_mm','up'),
('S09','Combined jitter',[13],'jitter',[0,.5,1.5],'More jitter adds displacement, angle, size and slow baseline variation; quality need not improve.',None,None),
('S15','Random seed',[20],'seed',[0,17,977],'Different seeds can select different stored shapes even at zero coordinate jitter.',None,None),
('S16','Anchoring strength',[22],'legibility',[0,.5,.85,1.0],'At or above 0.85 the substitution policy commits to representative/fallback shapes. Core selection can reduce variation.','core_glyphs','up')]:
    sweep(id,title,gids,vals,exp,key=key,metric=metric,direction=direction)

def gap(v):
    with tune(Y,MIN_LETTER_GAP_XH=v,MAX_LETTER_GAP_XH=v+.02):return trajcase(trajectory('hellohellohello'))
sweep('S08','Letter gap bounds',[11],[.0,.10,.35],'Larger allowed gaps should increase the span, subject to the collision guard.',fn=gap,metric='width_mm',direction='up')
sweep('S10','Collision clearance',[12],[0,.05,.4],'Larger clearance should push colliding neighbours further apart.',fn=lambda v:constant_case('COLLISION_MIN_GAP_XH',v),metric='width_mm',direction='up')
for id,title,gid,old,values in [
('S11','Position jitter coefficient',14,'ampP = 0.022*jitter',[0,.022,.09]),
('S12','Rotation jitter coefficient',15,'ampR = 2.2*jitter',[0,2.2,12.]),
('S13','Scale jitter coefficient',16,'ampS = 0.05*jitter',[0,.05,.25])]:
    var=old.split(' = ')[0]
    def case(v,old=old,var=var):
        with edit(Y,'SynthesizeText',[(old,f'{var} = {v}*jitter')]):return trajcase(trajectory(jitter=1.0))
    sweep(id,title,[gid],values,'Increasing this multiplier changes its named perturbation at jitter=1; other jitter mechanisms remain active.',fn=case)
def drift(v):
    with edit(Y,'SynthesizeText',[('driftV = 0.85*driftV',f'driftV = {v}*driftV')]):
        return trajcase(trajectory('hello world '*6,jitter=1.0,lineWidthMm=450))
sweep('S14','Baseline drift persistence',[17],[0,.85,.98],'Higher persistence retains drift velocity longer; clamping can mask the visible effect.',fn=drift)
def bound(v):
    with edit(Y,'SynthesizeText',[('drift+driftV,-0.25,0.25',f'drift+driftV,-{v},{v}')]):
        return trajcase(trajectory('hello world '*6,jitter=1.5,lineWidthMm=450))
sweep('S17','Baseline drift bound',[18],[0,.25,.8],'A zero bound removes drift position; larger bounds permit more vertical wander.',fn=bound)
def clearance(v):
    with edit(Y,'SynthesizeText',[('ascender-descender+0.85',f'ascender-descender+{v}')]):
        c=trajcase(trajectory('hello world\nhello world\nhello world'));c['metrics']['pitch_mm']=c['detail']['lineStepMm'];return c
sweep('S18','Inter-line clearance',[19],[.1,.85,1.8],'Larger clearance should increase synthesis pitch before machine placement.',fn=clearance,metric='pitch_mm',direction='up')
def calibration(v):
    Y._STYLE_CAL.clear();kw={'_cal':IDENTITY} if not v else {'_cal':None}
    c=trajcase(trajectory(**kw));c['detail']['calibration']={str(k):v for k,v in Y._STYLE_CAL.items()};return c
sweep('S19','Automatic style calibration',[21],[False,True],'Enabling the current correction can change shear and vertical reach. The old switch is absent; the supported _cal argument supplies the neutral reference.',fn=calibration)
def autolam(v):
    good,bad=v
    with tune(Y,_AUTO_LAM_GOOD=good,_AUTO_LAM_BAD=bad):
        p=copy.deepcopy(PROFILES['dylan']);p.pop('legibilityLambda',None);lam=Y.AutoLegibility(p)
        c=trajcase(trajectory(profile=p,legibility=lam));c['metrics']['automatic_lambda']=lam;return c
sweep('S20','Automatic anchoring endpoints',[23],[[.12,.20],[.24,.32],[.35,.45]],'Moving both quality endpoints upward should reduce automatic lambda for one unchanged glyph-distance distribution.',fn=autolam,metric='automatic_lambda',direction='down')
sweep('S21','Representative quality cutoff',[25],[.05,.30,.7],'A larger cutoff accepts more personal medoids. Source metadata calls both medoid and generic fallback core; geometry is the evidence.',fn=lambda v:constant_case('IDEAL_ANCHOR_MAX_PRIORD',v,legibility=1.0))
def variants(v):
    p=copy.deepcopy(PROFILES['dylan']);p['glyphs']['a']=p['glyphs']['a'][:v]
    return trajcase(trajectory('aaaaa aaaaa',profile=p))
sweep('S22','Stored variant pool for a repeated letter',[26],[1,3,8],'A one-entry pool should repeat the same stored a when jitter and anchoring are zero; more entries can add geometric variety.',fn=variants)
def missing(v):
    p=copy.deepcopy(PROFILES['dylan'])
    if v>=1:p['glyphs'].pop('a',None)
    if v>=2:
        p['glyphs'].pop('A',None);p['idealMedoid'].pop('a',None)
    c=trajcase(trajectory('a a a',profile=p));return c
sweep('S23','Missing-character fallback',[27],[0,1,2],'Removing a first exercises alternate/case/representative routes; removing its medoid exposes the generic route. Only temporary profile copies change.',fn=missing)
def perchar(v):return trajcase(trajectory('hello world',perCharLam={1:v}))
sweep('S24','Position-specific anchoring',[28],[0,1.0,2.0],'At 1 the selected character is forced to core; at 2 the current code forces print fallback, with possible neighbouring changes.',fn=perchar)

# Extra active controls not explicitly named in the old guide.
sweep('S25','Eligible variant cap (current code)',[],[1,5,12],'Only the best N ranked stored shapes enter sampling; the current default is five.',fn=lambda v:constant_case('_ELIGIBLE_VARIANTS',v,text='aaaaa aaaaa'))
for author in ['dylan','abhinav','151']:
    experiment('S26_'+author,'Cross-writer connectedness check: '+author,'Synthesis',[1,2,4],[.3,.6,1.0],
               'Sub-threshold joining suppression should hold for each hand; above it, geometry-dependent joins can differ.',
               lambda v,a=author:trajcase(trajectory(profile={**PROFILES[a],'connectedness':v})),
               'Existing curated '+author+' profile; same hello world hello text and seed 17','connectedness','synthesise.py / SynthesizeText',qualification='Replication across three stored profiles; not a population accuracy test.')
savejson(OUT/'assets_synthesis.json',ASSETS)
