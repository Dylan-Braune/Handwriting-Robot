import os
os.environ['SWEEP_PART']='classification'
from common import *
def run(id,title,gids,vals,expected,fn,metric=None,direction=None,control=None):
    return experiment(id,title,'Recognition and candidate search',gids,vals,expected,fn,
                      'Second actual line crop from Dylan01; existing saved networks; fixed pixels except named input edit',control or title,
                      'recognise.py / ClassifyImage, StrokeNormalize, ReadTextScored; web/server.py / NameWriter; synthesise.py / SynthesizeJointBestOf',metric,direction,
                      'One photographed page/line is a sensitivity example, not held-out dataset validation. Softmax confidence is not correctness.')

def named(img,m=None,norm=False):
    met,probs=classify(img,m,norm);c=imagecase(img,**met);c['detail']={'probabilities':probs,'class_mapping':(m or model()).authorMapping};return c
def checkpoints(v):
    m=model('writer',v);c=named(LINE,m);c['metrics']['classes']=len(m.authorMapping);return c
run('C01','Explicit saved writer checkpoint selection',[34,87],['author_classifier_10new_weights.npz','author_shape_10new_weights.npz','author_shape_11w_pw170_scratch_weights.npz'],
    'Different checkpoints may change writer probabilities. The current server ink and shape handles alias the same object; this test explicitly loads distinct files. Class mappings are recorded.',checkpoints)
run('C02','Raw versus stroke-normalised writer input',[35],[False,True],
    'Normalisation should change pen-width evidence and may change the named writer or confidence; improvement is not guaranteed.',
    lambda v:named(R.StrokeNormalize(LINE) if v else LINE,norm=False))
def width(v):
    a=R.StrokeNormalize(LINE,penWidthXh=v);c=named(a);c['metrics']['foreground_pixels']=int((a<128).sum());return c
run('C03','Redrawn pen width in x-heights',[36],[.06,.17,.32],'Larger discrete dilation widths should increase foreground while classifier confidence can change non-monotonically.',width,'foreground_pixels','up')
def reader(v):
    m=model('reader',v);txt,score=R.ReadTextScored(LINE,m);c=imagecase(LINE,char_score=R.CharAcc(txt,TRUTH),frame_log_score=score)
    c['detail']={'predicted_text':txt,'reference_label':TRUTH,'checkpoint':v};return c
run('C04','Text-reader checkpoint selection',[37],['paper_cnn_bilstm_ctc_scratch_best.npz','paper_cnn_bilstm_ctc_joint_best.npz','paper_cnn_bilstm_ctc_hf_best.npz','paper_cnn_bilstm_ctc_personal_best.npz'],
    'A newly loaded reader can change the transcript without training. No checkpoint-order accuracy prediction is made.',reader)

# Invoke only the isolated pure pooling function, never import or start the server.
server_src=(BASE/'snapshot/web/server.py').read_text(encoding='utf-8')
server_tree=ast.parse(server_src)
poolnode=next(n for n in server_tree.body if isinstance(n,ast.FunctionDef) and n.name=='NameWriter')
poolsrc=ast.get_source_segment(server_src,poolnode)
probcache={}
def inferprob(a):
    key=hashlib.sha256(a.tobytes()).hexdigest()
    if key not in probcache:probcache[key]=classify(a)[1]
    return probcache[key]
class FrozenLW:
    NameOne=None
    @staticmethod
    def Map(func,jobs):return [(i,inferprob(c)) for i,c in jobs]
def poolcase(crops,threshold=128,equal=False):
    src=poolsrc.replace('c < 128',f'c < {threshold}')
    if equal:src=src.replace('w = np.array([float((c < '+str(threshold)+').sum()) for c in crops], dtype=np.float64)','w = np.ones(len(crops), dtype=np.float64)')
    ns={'np':np,'LW':FrozenLW};exec(compile(src,'isolated NameWriter','exec'),ns)
    _,avg,w=ns['NameWriter'](crops);inv={v:k for k,v in model().authorMapping.items()};idx=int(avg.argmax())
    c=chartcase([inv[i] for i in range(len(avg))],avg,writer=inv[idx],confidence=float(avg[idx]),p_dylan=float(avg[model().authorMapping['dylan']]),lines=len(crops))
    c['detail']={'line_weights':w,'probabilities':avg,'cached_actual_line_probabilities':True};return c
run('C05','Equal versus ink-weighted page pooling',[38],['equal','ink'],
    'Only line weights change; every cached neural line probability stays fixed. The pooled label or confidence may change.',lambda v:poolcase(LINES,equal=v=='equal'))
run('C06','Included evidence in the page vote',[39],[1,4,len(LINES)],
    'Adding line evidence changes the pooled distribution, not any individual crop prediction. More lines need not raise confidence monotonically.',lambda v:poolcase(LINES[:v]))
run('C07','Ink-count threshold used for pooling',[40],[64,128,200],
    'Higher thresholds count more grey pixels as ink. Relative weights and page probabilities can change; input pixels and line predictions stay fixed.',lambda v:poolcase(LINES,threshold=v))
run('C08','How much of a line is supplied',[41],[.25,.5,1.0],
    'Less handwriting changes the evidence; confidence is not guaranteed to decrease smoothly.',lambda v:named(LINE[:,:max(1,int(LINE.shape[1]*v))]))
run('C09','Blur sensitivity',[42],[0,1.5,3.0],
    'Blur removes fine structure; recognition and writer confidence may change, without a monotone accuracy guarantee.',lambda v:named(S.Blur(LINE,v) if v else LINE))
run('C10','Contrast sensitivity',[42],[.4,1.,2.],
    'Changing contrast changes pixel intensities supplied to the existing model.',lambda v:named(np.asarray(ImageEnhance.Contrast(Image.fromarray(LINE)).enhance(v))))
run('C11','Brightness sensitivity',[42],[.5,1.,1.5],
    'Darkening or clipping the photograph may change identity evidence; no direction of confidence is assumed.',lambda v:named(np.asarray(ImageEnhance.Brightness(Image.fromarray(LINE)).enhance(v))))
baseline,baseline_probs=classify(LINE)
def abstain(v):
    accepted=baseline['confidence']>=v;c=named(LINE);c['metrics']['accepted']=int(accepted);c['detail']['decision']=baseline['writer'] if accepted else 'ABSTAIN';return c
run('C12','Confidence-based abstention',[43],[0,.99,1.0],
    'Raising the decision threshold can only reduce acceptance for fixed probabilities. It does not recalibrate those probabilities.',abstain,'accepted','down')
def bound_default(v):
    if v=='explicit .40':a=R.StrokeNormalize(LINE,penWidthXh=.4)
    else:
        with tune(R,PenWidthXh=.4 if v=='global .40' else .17):a=R.StrokeNormalize(LINE)
    return imagecase(a,foreground=int((a<128).sum()))
run('C13','Bound defaults versus a changed module constant',[90],['baseline','global .40','explicit .40'],
    'Changing PenWidthXh alone should not change the already-bound default argument. Passing .40 explicitly should change the output.',bound_default)

reader_model=model('reader');writer_model=model();profile=copy.deepcopy(PROFILES['dylan'])
def joint(**kw):
    args=dict(nTries=3,mmPerXh=4.,lineWidthMm=180.,jitter=.5,seed=17,
              reader=reader_model,authorModel=writer_model,pxPerMm=18.,legibility=0.,repair=False)
    args.update(kw);tr=Y.SynthesizeJointBestOf('dylan','hello world',copy.deepcopy(profile),**args)
    c=trajcase(tr);c['metrics'].update(joint_score=tr.meta['jointScore'],tries=tr.meta['jointTries'],repaired_chars=tr.meta['jointRepairedChars']);return c
run('C14','Candidate count',[29,92],[1,3,6],
    'The best internal joint score cannot decrease when the same deterministic candidate prefix is retained. This is not independent physical validation.',lambda v:joint(nTries=v),'joint_score','up')
run('C15','Repair disabled versus enabled',[30,92],[False,True],
    'Accepted repair uses a strict joint-score improvement rule, so the returned internal score should not decrease.',lambda v:joint(repair=v),'joint_score','up')
run('C16','Repair-round budget',[31],[0,1,3],
    'A larger budget permits more accepted attempts but can plateau immediately after a repeated rejected trial.',lambda v:joint(repair=True,repairRounds=v),'joint_score','up')
run('C17','Weak positions targeted per repair trial',[32],[1,2,4],
    'Changing more positions can change the outcome in either direction. Rejection preserves the previous best candidate.',lambda v:joint(repair=True,repairWorstK=v))

pool=[]
for k in range(6):
    tr=Y.SynthesizeText('hello world',copy.deepcopy(profile),seed=17+977*k,jitter=.5,legibility=0.)
    img=Y.RenderTrajectory(tr,pxPerMm=18.)
    pred=R.ReadText(img,reader_model);acc=R.CharAcc(pred,'hello world');_,pr=R.ClassifyImage(img,writer_model)
    pool.append((tr,acc,float(pr[writer_model.authorMapping['dylan']]),pred))
savejson(RAW/'C18_fixed_candidate_scores.json',[{'candidate':k,'char_score':x[1],'writer_probability':x[2],'text':x[3]} for k,x in enumerate(pool)])
def weighted(v):
    scores=[(a if v==1 else w if v==0 else 1/(v/max(a,1e-12)+(1-v)/max(w,1e-12))) for _,a,w,_ in pool]
    idx=int(np.argmax(scores));tr,a,w,pred=pool[idx];c=trajcase(tr);c['metrics'].update(selected_candidate=idx,char_score=a,writer_probability=w,objective=float(scores[idx]));c['detail']['predicted_text']=pred;return c
run('C18','Text/writer objective on one fixed candidate pool',[33],[0,.5,1.0],
    'Zero selects writer-only, one selects text-only, and .5 is the equal harmonic mean. The winner can change while every candidate remains identical.',weighted)
savejson(OUT/'assets_classification.json',ASSETS)
