#u23539195 coefficient justification: the architectural numbers, derived not asserted
#parameter counts, receptive fields, CTC column arithmetic, the held-out split and
#the mixed-batch composition, all read from the submitted code and the real
#checkpoints so the document quotes measured values
import json
import sys
from pathlib import Path

import numpy as np

import patch

OUT = Path(__file__).resolve().parent.parent/"results"/"architecture.json"


#---ConvStage: the parameter count and output size of one conv/bn/relu stage---
def ConvStage(inCh,outCh,nLayers,k=3):
    convW = 0
    c = inCh
    for _ in range(nLayers):
        convW += outCh*c*k*k + outCh      #weights + bias
        convW += 4*outCh                  #batch-norm gamma, beta, mean, var
        c = outCh
    return convW


#---ReceptiveField: how many input pixels one output of the backbone sees---
def ReceptiveField():
    #(kernel,stride) in order: stage1 x2, pool, stage2 x4, pool, stage3 x6
    layers = ([(3,1)]*2) + [(2,2)] + ([(3,1)]*4) + [(2,2)] + ([(3,1)]*6)
    rf,jump = 1,1
    for (k,s) in layers:
        rf += (k-1)*jump
        jump *= s
    return rf,jump


def Main():
    facts = {}

    #----- the shared backbone -----
    stages = [("stage1",1,32,2),("stage2",32,64,4),("stage3",64,128,6)]
    backbone = 0
    perStage = {}
    for (name,i,o,n) in stages:
        p = ConvStage(i,o,n)
        perStage[name] = dict(inCh=i,outCh=o,layers=n,params=p)
        backbone += p
    facts["backbone"] = dict(perStage=perStage,params=backbone,
                             convLayers=sum(n for _,_,_,n in stages))

    rf,jump = ReceptiveField()
    facts["receptiveField"] = dict(pixels=rf,stride=jump)

    #----- the reader head -----
    H = 256
    lstm = 0
    for inSize in (128,2*H):
        perDir = 4*H*(inSize+H) + 8*H       #W_ih, W_hh, b_ih, b_hh
        lstm += 2*perDir                     #forward and backward
    numClasses = 74+1
    textHead = numClasses*(2*H) + numClasses
    facts["reader"] = dict(hidden=H,lstmLayers=2,lstmParams=lstm,
                           headParams=textHead,
                           total=backbone+lstm+textHead,
                           numClasses=numClasses)

    #----- the writer classifier head -----
    nAuthors = 13
    head = 96*128+96 + nAuthors*96+nAuthors
    facts["namer"] = dict(hiddenUnits=96,authors=nAuthors,headParams=head,
                          total=backbone+head,dropout=0.2)

    #----- CTC column arithmetic -----
    W = 640
    T = W//4
    facts["ctc"] = dict(inputWidth=W,frames=T,
                        maxSafeLabelChars=int(T*0.75),
                        pixelsPerFrame=4,
                        note="T=W/4 because width is pooled twice by 2")

    #----- what the checkpoints on disk actually contain -----
    weights = patch.DATA/"weights"
    ck = {}
    for f in sorted(weights.glob("*.npz")):
        try:
            with np.load(f,allow_pickle=False) as z:
                n = int(sum(z[k].size for k in z.files if not k.startswith("__")))
                names = ([str(x) for x in z["__author_names__"]]
                         if "__author_names__" in z.files else None)
                ck[f.name] = dict(params=n,authorNames=names,
                                  tensors=len([k for k in z.files
                                               if not k.startswith("__")]),
                                  megabytes=round(f.stat().st_size/1e6,2))
        except Exception as e:
            ck[f.name] = dict(error=str(e))
    facts["checkpointsOnDisk"] = ck

    #the two checkpoints the submitted code names
    facts["namedBySubmittedCode"] = dict(
        text="paper_cnn_bilstm_ctc_scratch_best.npz",
        author="author_shape_11w_pw170_scratch_weights.npz",
        textPresent=(weights/"paper_cnn_bilstm_ctc_scratch_best.npz").exists(),
        authorPresent=(weights/"author_shape_11w_pw170_scratch_weights.npz").exists())

    #----- the held-out split -----
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "subpaths",patch.SUBMITTED/"paths.py")
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    facts["split"] = dict(valFraction=sp.ValFraction,seed=sp.SplitSeed,
                          databaseWriters=sp.DatabaseWriters,
                          personalWriters=sp.PersonalWriters,
                          allWriters=len(sp.AllWriters))

    #how a 15 % split lands on a given number of lines, and the Wilson
    #half-width it buys at p=0.9, which is why the fraction matters
    def Wilson(p,n,z=1.96):
        if n <= 0:
            return None
        d = 1+z*z/n
        c = (p+z*z/(2*n))/d
        h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
        return dict(lo=round(c-h,4),hi=round(c+h,4),halfWidth=round(h,4))
    rows = {}
    for nLines in (200,500,1000,2000):
        for frac in (0.05,0.10,0.15,0.25,0.35):
            nv = int(round(nLines*frac))
            rows["%d@%.2f" % (nLines,frac)] = dict(
                val=nv,train=nLines-nv,wilson90=Wilson(0.9,nv))
    facts["splitArithmetic"] = rows

    #----- mixed-batch composition at the personal fraction -----
    comp = {}
    for frac in (0.0,0.10,0.20,0.35,0.50):
        for batch in (8,):
            nExtra = int(round(batch*frac))
            comp["%.2f" % frac] = dict(batch=batch,personal=nExtra,
                                       main=batch-nExtra)
    facts["mixedBatch"] = comp

    #----- augmentation: how often each is applied -----
    #each of the four is a candidate with probability 0.5, then ONE candidate is
    #chosen, so the marginal probability of a given augmentation is not 0.5
    n = 4
    pNone = 0.5**n
    pEach = 0.0
    from math import comb
    for k in range(1,n+1):
        pEach += comb(n-1,k-1)*(0.5**n)*(1.0/k)
    facts["augmentation"] = dict(
        candidates=n,pNoAugment=round(pNone,4),
        pGivenAugmentApplied=round(pEach,4),
        note="each augmentation is a candidate at p=0.5, then exactly one "
             "candidate is applied, so the marginal rate per augmentation is "
             "lower than 0.5")

    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(facts,indent=1,default=str),encoding="utf-8")
    print(json.dumps({k:facts[k] for k in
                      ("backbone","receptiveField","reader","namer","ctc",
                       "augmentation","namedBySubmittedCode")},
                     indent=1,default=str))
    return 0


if __name__ == "__main__":
    sys.exit(Main())
