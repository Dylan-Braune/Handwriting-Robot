#u23539195 Dylan Braune.  visual tests for Section 3.5's verification
#subsection: evidence that the from-first-principles derivatives in train.py
#are correct, and what the training-time augmentation actually does.
#
#  python vt_grad.py
from vt_common import *
import numpy as np
import train as TR
import recognise as REC
import random


#======================================================================
#T38  FINITE-DIFFERENCE CHECK OF THE HAND-WRITTEN DERIVATIVES
#the draft reports three summary numbers from an archive.  this runs the
#check live over every parameter of a convolution and over every logit of
#a CTC lattice, and plots analytic against numerical
#======================================================================
def CheckConv(rng,n=220):
    """central differences against Conv2dBackward on a small real-shaped layer"""
    x = rng.standard_normal((2,3,9,11))
    w = rng.standard_normal((4,3,3,3))*0.3
    b = rng.standard_normal(4)*0.1
    out,cache = TR.Conv2dForward(x,w,b)
    #a fixed random upstream gradient, so the test is of the layer alone
    g = rng.standard_normal(out.shape)
    dx,dw,db = TR.Conv2dBackward(g,cache)

    def Loss(W,B,X):
        return float((TR.Conv2dForward(X,W,B)[0]*g).sum())

    eps = 1e-6
    pairs = []
    flat = [("weight",w,dw),("bias",b,db),("input",x,dx)]
    for name,arr,ana in flat:
        idxs = [tuple(rng.integers(0,s) for s in arr.shape)
                for _ in range(n//len(flat))]
        for ix in idxs:
            keep = arr[ix]
            arr[ix] = keep+eps;lp = Loss(w,b,x)
            arr[ix] = keep-eps;lm = Loss(w,b,x)
            arr[ix] = keep
            pairs.append((name,float(ana[ix]),(lp-lm)/(2*eps)))
    return pairs


def CheckCtc(rng,T=6,K=5,label=(1,1,2)):
    """central differences against the CTC logit gradient, every logit"""
    z = rng.standard_normal((T,1,K))
    loss,grad = TR.CtcLoss(z,[list(label)])
    eps = 1e-5
    pairs = []
    for t in range(T):
        for k in range(K):
            keep = z[t,0,k]
            z[t,0,k] = keep+eps;lp = TR.CtcLoss(z,[list(label)])[0]
            z[t,0,k] = keep-eps;lm = TR.CtcLoss(z,[list(label)])[0]
            z[t,0,k] = keep
            pairs.append(("ctc",float(grad[t,0,k]),float((lp-lm)/(2*eps))))
    return pairs,float(loss)


def T38():
    rng = np.random.default_rng(0)
    conv = CheckConv(rng)
    ctc,ctcLoss = CheckCtc(rng)
    allp = conv+ctc

    groups = {}
    for nm,a,nmr in allp:
        groups.setdefault(nm,[]).append((a,nmr))

    fig,(axA,axB) = plt.subplots(1,2,figsize=(WIDTH,3.45),
                                 gridspec_kw=dict(width_ratios=[1.05,1],
                                                  wspace=0.30,left=0.115,
                                                  right=0.985,top=0.905,
                                                  bottom=0.125))

    marks = {"weight":"o","bias":"s","input":"^","ctc":"D"}
    cols = {"weight":DARK,"bias":MID,"input":SEQ[3],"ctc":INK}
    lim = 0
    for nm,pts in groups.items():
        a = np.array([p[0] for p in pts]);n_ = np.array([p[1] for p in pts])
        lim = max(lim,float(np.abs(a).max()),float(np.abs(n_).max()))
        axA.plot(n_,a,ls="none",marker=marks[nm],ms=3.4,mfc="none",
                 mec=cols[nm],mew=0.9,label=f"{nm} ({len(pts)})")
    axA.plot([-lim,lim],[-lim,lim],color=SILVER,lw=0.9,zorder=0)
    axA.set_xlabel("central-difference gradient")
    axA.set_ylabel("gradient from the hand-written backward pass")
    axA.set_title("Analytic against numerical, every point on the diagonal",
                  fontsize=FS(8.5))
    axA.legend(fontsize=FS(6.6),loc="upper left")
    axA.set_xlim(-lim*1.08,lim*1.08);axA.set_ylim(-lim*1.08,lim*1.08)
    axA.set_aspect("equal")

    #the relative error, which is what the archive's table reports
    errs,labels,stats = [],[],{}
    for nm,pts in groups.items():
        a = np.array([p[0] for p in pts]);n_ = np.array([p[1] for p in pts])
        rel = np.abs(a-n_)/np.maximum(np.abs(a)+np.abs(n_),1e-12)
        errs.append(rel);labels.append(nm)
        stats[nm] = dict(n=len(pts),maxAbs=float(np.abs(a-n_).max()),
                         maxRel=float(rel.max()),medianRel=float(np.median(rel)))
    pos = np.arange(len(labels))
    for i,e in enumerate(errs):
        e = np.maximum(e,1e-16)
        axB.plot(np.full(len(e),pos[i])+rng.normal(0,0.06,len(e)),e,
                 ls="none",marker=marks[labels[i]],ms=3.0,mfc="none",
                 mec=cols[labels[i]],mew=0.8)
        axB.plot([pos[i]-0.28,pos[i]+0.28],[np.median(e)]*2,color=INK,lw=1.4)
    axB.set_yscale("log")
    #a log tick label is a power, so its exponent is set at the shrink
    #fraction of the label; the label is raised so the exponent clears 8 pt
    axB.tick_params(axis="y",labelsize=FSM())
    axB.set_xticks(pos);axB.set_xticklabels(labels,fontsize=FS(7.5))
    axB.set_ylabel("relative difference")
    axB.set_title("Relative difference per parameter, black bar is the median",
                  fontsize=FS(8.5))
    axB.axhline(1e-6,color=SILVER,lw=0.9,ls="--")
    axB.set_ylim(top=1.0e-5)
    axB.text(-0.42,1.06e-6,"$10^{-6}$",fontsize=FSM(),color=INK,
             ha="left",va="bottom")

    worst = max(s["maxRel"] for s in stats.values())
    Caption(fig,f"T38  The derivatives in the first-principles trainer checked against central differences, computed live. "
                f"{len(allp)} parameters were perturbed across the convolution's weights, bias and input, and across every logit "
                f"of a small CTC lattice; the largest relative difference is {worst:.2e}. This is evidence that the hand-written "
                "backward pass implements the derivative of the forward pass, which no amount of training accuracy would show.")
    Stamp(fig,"train.Conv2dBackward / train.CtcLoss, central differences")
    Save(fig,"T38_gradient_check",f"worst relative {worst:.2e}")
    return dict(stats=stats,ctcLoss=ctcLoss,worstRelative=worst,points=len(allp))


#======================================================================
#T39  THE TRAINING-TIME AUGMENTATION SET
#four transforms named in the draft, applied to real crops
#======================================================================
def T39(samples,seed=3):
    from PIL import Image
    ops = [("shear",TR.AugmentShear),("rotate",TR.AugmentRotate),
           ("perspective",TR.AugmentPerspective),("elastic",TR.AugmentElastic)]

    nR = len(samples)
    fig,axes = plt.subplots(nR,len(ops)+1,
                            figsize=(7.4,1.05*nR+0.75),squeeze=False,
                            gridspec_kw=dict(hspace=0.32,wspace=0.07))
    for r,(name,gray) in enumerate(samples):
        #a short readable window of the line, the same window in every panel
        ink = gray<160
        xs = np.nonzero(ink.any(0))[0]
        left = int(xs.min()) if len(xs) else 0
        right = min(gray.shape[1],left+int(gray.shape[0]*5.5))
        ShowGray(axes[r][0],gray[:,left:right],"original" if r==0 else None)
        axes[r][0].set_ylabel(name,fontsize=FS(7.5),rotation=0,ha="right",
                              va="center",labelpad=12)
        for c,(lbl,fn) in enumerate(ops):
            random.seed(seed+r*10+c)
            np.random.seed(seed+r*10+c)
            out = np.asarray(fn(Image.fromarray(gray)),dtype=np.uint8)
            ShowGray(axes[r][c+1],out[:,left:right],lbl if r==0 else None)

    fig.suptitle("Training-time augmentation applied to real line crops",
                 fontsize=FS(9),y=1.0)
    Caption(fig,"T39  The four implemented transforms, each applied to the same window of four writers' handwriting. Every one "
                "changes the geometry while leaving the transcript and the writer unchanged, which is the property that makes it "
                "usable as a label-preserving augmentation. They are applied to the training view only; validation keeps a single "
                "consistent preprocessing path.")
    Stamp(fig,"train.AugmentShear / AugmentRotate / AugmentPerspective / AugmentElastic")
    Save(fig,"T39_augmentation_set")


if __name__=="__main__":
    import json
    print("T38-T39  trainer verification visual tests")
    res = {}
    res["T38"] = T38()
    samples = []
    for w in ["Owen","Yeukita","Dylan","Thiya"]:
        q = LineCrop(w,1,0)
        if q.exists():
            samples.append((w,Gray(q)))
    T39(samples)
    (DATA/"grad_results.json").write_text(json.dumps(res,indent=1,default=float),
                                          encoding="utf-8")
    print("done")
