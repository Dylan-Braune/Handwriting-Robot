from pathlib import Path
import json,inspect
HERE=Path(__file__).resolve().parent
s=(HERE/'run_visual_bench.py').read_text(encoding='utf-8')
exec(compile(s.split('# Three original pages')[0],str(HERE/'run_visual_bench.py'),'exec'))
records=json.loads((OUT/'bench.json').read_text(encoding='utf-8'))

# Exhaustive seam reference, including the same movement penalty.
rng=np.random.default_rng(83);seam_errors=[]
for _ in range(20):
    cost=rng.random((4,6));path=seg.LeastCostSeam(cost,step=1)
    def score(p):return sum(cost[y,x] for x,y in enumerate(p))+.02*sum(abs(a-b) for a,b in zip(p,p[1:]))
    best=min(score(p) for p in itertools.product(range(4),repeat=6) if all(abs(a-b)<=1 for a,b in zip(p,p[1:])))
    seam_errors.append(abs(score(path)-best))
records['seam_reference']={'cases':20,'seed':83,'shape':[4,6],'step':1,'maximum_cost_error':max(seam_errors),'paths_considered_per_case_before_step_guard':4**6}

# File round-trip with joins/closures disabled so corresponding vertices are testable.
base=[[(0.,0.),(4.3271,0.),(7.6543,3.2167)],[(14.1234,2.4321),(16.0,5.6789)]]
fixtures=[]
for offset in [0.,8.4,25.2]:
    for factor in [1.,3.,30.]:
        t=syn.Trajectory([[(x*factor,y*factor) for x,y in p] for p in base],{'author':'fixture','lines':1})
        cfg=rep.GantryConfig(lineOffsetMm=offset,joinStrokesMm=-1,closeRingMm=0,minSegmentMm=0)
        expected,scale=rep.ToMachine(t,cfg)
        file=OUT/f'roundtrip_{offset}_{factor}.gcode';info=rep.WriteGcode(t,cfg,file)
        actual=rep.ParseGcode(file)
        same_shapes=[len(p) for p in actual]==[len(p) for p in expected]
        err=float(np.max(np.abs(np.array([p for q in actual for p in q])-np.array([p for q in expected for p in q])))) if same_shapes else None
        fixtures.append({'offset_mm':offset,'factor':factor,'scale':scale,'same_path_vertex_counts':same_shapes,'maximum_coordinate_error_mm':err,'clipped_points':info['clipped']})
records['roundtrip']=fixtures

# Actual connector calls made during synthesis, with pure-function instrumentation.
calls=[];orig=syn._Ligature
def capture(p0,p1,xh,rng):
    caller=inspect.currentframe().f_back.f_locals
    result=orig(p0,p1,xh,rng)
    calls.append({'pair':caller['word'][caller['ci']-1:caller['ci']+1],
                  'previous_path':list(caller['chain']), 'next_path':list(caller['body']),
                  'connector':list(result),'p0':p0,'p1':p1,'xh':xh,'gap_xh':(p1[0]-p0[0])/xh})
    return result
syn._Ligature=capture
try:
    p=ROOT/'Software/CNN/NOGIT/StyleProfiles10/150_curated.json';record(p)
    profile=json.loads(p.read_text(encoding='utf-8'))
    t=syn.SynthesizeText('minimum writing',profile,seed=2026,mmPerXh=4,jitter=.15,legibility=0)
finally:syn._Ligature=orig
records['joining']={'text':'minimum writing','writer':'150','seed':2026,'legibility':0,'calls':calls}
if calls:
    q=calls[0];c=pdf('design-glyph-joining',720,300)
    text(c,15,278,f'Actual accepted connector: writer 150, pair "{q["pair"]}", gap {q["gap_xh"]:.3f} x-heights',12,True)
    paths=[q['previous_path'],q['next_path']];pts=np.array([p for s in paths for p in s]);lo=pts.min(0);hi=pts.max(0);k=min(640/max(hi[0]-lo[0],1e-6),195/max(hi[1]-lo[1],1e-6))
    def plot(path,color,width):
        if not path:return
        c.setStrokeColorRGB(*color);c.setLineWidth(width);p=c.beginPath();p.moveTo(35+(path[0][0]-lo[0])*k,45+(path[0][1]-lo[1])*k)
        for x,y in path[1:]:p.lineTo(35+(x-lo[0])*k,45+(y-lo[1])*k)
        c.drawPath(p)
    for path in paths:plot(path,(.08,.25,.4),1.5)
    plot([q['p0']]+q['connector'],(.8,.12,.12),2)
    for key,label in [('p0','exit'),('p1','entry')]:
        x,y=q[key];px=35+(x-lo[0])*k;py=45+(y-lo[1])*k;c.circle(px,py,3,fill=0);text(c,px+5,py+8,label,10)
    text(c,15,15,'Blue: existing body paths. Red: generated connector. Detached marks are separate paths.',10)
    finish(c,'design-glyph-joining')
save()
print('Seam and round-trip checks complete;',len(calls),'actual joins',flush=True)
