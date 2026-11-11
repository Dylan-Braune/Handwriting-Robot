#u23539195 Dylan Braune.  visual tests for Section 3.6's layout, joining and
#candidate-selection subsections.  real stored profiles, real synthesis.
#
#  python vt_synth.py
from vt_common import *
import numpy as np
import json
import synthesise as SY
import recognise as REC
import style as ST


def Profile(writer):
    p = PROFILES/f"{writer}_curated.json"
    if not p.exists():
        p = PROFILES/f"{writer}.json"
    return json.loads(p.read_text(encoding="utf-8")),p.name


def DrawTraj(ax,traj,color=DARK,lw=1.1,marks=False):
    for s in traj.strokes:
        a = np.asarray(s,dtype=float)
        if len(a)>1:
            ax.plot(a[:,0],-a[:,1],color=color,lw=lw,solid_capstyle="round")
        if marks and len(a):
            ax.plot(a[0,0],-a[0,1],"o",ms=2.0,color=INK,zorder=4)
    ax.set_aspect("equal");ax.grid(False)
    ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(SILVER);s.set_linewidth(0.5)


#======================================================================
#T48  JITTER AND BASELINE DRIFT
#controlled variation against the point where it stops being legible
#======================================================================
def T48(writer="owen",text="Writing with style."):
    prof,fn = Profile(writer)
    amps = [0.0,0.15,0.35,0.70]
    fig,axes = plt.subplots(len(amps),1,figsize=(7.2,0.95*len(amps)+1.0),
                            gridspec_kw=dict(hspace=0.5))
    out = {}
    for i,j in enumerate(amps):
        traj = SY.SynthesizeText(text,prof,mmPerXh=4.0,seed=2026,jitter=j)
        DrawTraj(axes[i],traj)
        d,u = traj.PenTravelMm()
        axes[i].set_title(f"jitter = {j:.2f}" +
                          ("   (the implemented default is 0.15)" if j==0.15 else ""),
                          fontsize=FS(8.3))
        out[j] = dict(strokes=len(traj.strokes),drawMm=round(d,2))
    Caption(fig,"T48  The same sentence from one stored profile at four jitter amplitudes. Zero reuses an identical path for every "
                "occurrence of a letter, which no hand does; too much destroys letter identity and lets neighbouring glyphs "
                "collide. The implemented default sits between the two, and a correlated drift term moves the baseline gradually "
                "rather than introducing unrelated vertical jumps.")
    Stamp(fig,f"synthesise.SynthesizeText | {fn}")
    Save(fig,"T48_jitter_ladder")
    return out


#======================================================================
#T50  LINE-LEVEL STYLE CALIBRATION
#what the three correction iterations change
#======================================================================
def T50(writers,text="Writing with style."):
    fig,axes = plt.subplots(len(writers)*2,1,
                            figsize=(7.2,0.85*len(writers)*2+1.0),
                            gridspec_kw=dict(hspace=0.55))
    out = {}
    for i,w in enumerate(writers):
        prof,fn = Profile(w)
        flat = dict(shearDelta=0.0,ascK=1.0,descK=1.0)
        cal = SY.StyleCalibration(prof,4.0)
        t0 = SY.SynthesizeText(text,prof,mmPerXh=4.0,seed=7,_cal=flat)
        t1 = SY.SynthesizeText(text,prof,mmPerXh=4.0,seed=7,_cal=cal)
        DrawTraj(axes[2*i],t0,color=SILVER)
        axes[2*i].set_title(f"{Author(w)}, before calibration",fontsize=FS(8.3))
        DrawTraj(axes[2*i+1],t1,color=DARK)
        axes[2*i+1].set_title(
            f"{Author(w)}, after: shear {cal['shearDelta']:+.4f}, "
            f"ascender x{cal['ascK']:.3f}, descender x{cal['descK']:.3f}",
            fontsize=FS(8.3))
        out[w] = {k:round(float(v),5) for k,v in cal.items()
                  if isinstance(v,(int,float))}
    Caption(fig,"T50  Line-level calibration. A glyph library is measured one letter at a time, but slant and vertical proportion "
                "are properties of an assembled line, and the two need not agree. A probe line is rendered without jitter, measured "
                "with the same style functions used on the writer's own pages, and bounded shear and extent corrections are applied "
                "over three iterations. This improves internal consistency of the representation; it is not independent evidence "
                "of physical stylistic fidelity.")
    Stamp(fig,"synthesise.StyleCalibration")
    Save(fig,"T50_style_calibration")
    return out


#======================================================================
#T20 and T51  JOINING: WHAT IS CONNECTED AND WHAT IS NOT
#======================================================================
def T20(writers,text="online writing"):
    fig = plt.figure(figsize=(7.2,1.05*len(writers)+2.5))
    gs = fig.add_gridspec(len(writers)+1,1,
                          height_ratios=[1]*len(writers)+[1.5],hspace=0.55)
    rows = []
    for i,w in enumerate(writers):
        prof,fn = Profile(w)
        conn = float(prof.get("connectedness",0.0))
        traj = SY.SynthesizeText(text,prof,mmPerXh=4.0,seed=11)
        ax = fig.add_subplot(gs[i])
        DrawTraj(ax,traj,marks=True)
        thr = float(getattr(SY,"CONNECT_MIN",0.45))
        note = ("  -- below the %.2f threshold, joining suppressed"%thr
                if conn<thr else "  -- joining enabled")
        ax.set_title(f"{Author(w)}:  measured connectedness {conn:.3f}{note};"
                     f"  {len(traj.strokes)} pen-down paths",fontsize=FS(8.3))
        rows.append((w,conn,len(traj.strokes)))

    ax = fig.add_subplot(gs[len(writers)])
    names = [Author(r[0]) for r in rows]
    cs = [r[1] for r in rows]
    xs = np.arange(len(rows))
    ax.bar(xs,cs,0.55,color=[DARK if c>=float(getattr(SY,"CONNECT_MIN",0.45)) else SILVER for c in cs])
    thr = float(getattr(SY,"CONNECT_MIN",0.45));ax.axhline(thr,color=INK,lw=1.1,ls="--")
    ax.text(len(rows)-0.5,thr+0.015,f"{thr:.2f} join threshold",fontsize=FS(6.8),
            ha="right",va="bottom",color=INK)
    ax.set_xticks(xs);ax.set_xticklabels(names,fontsize=FS(7.5))
    ax.set_ylabel("stored connectedness",fontsize=FS(7.5))
    for i,c in enumerate(cs):
        ax.text(i,c+0.012,f"{c:.3f}",ha="center",fontsize=FS(7),color=INK)
    ax.set_title("Each writer's own measured connection fraction decides whether "
                 "their text is joined",fontsize=FS(8.3))

    Caption(fig,"T20 and T51  Joining is driven by the writer, not by a global setting. The connection fraction measured from a "
                "writer's own pages is used as the probability of joining eligible neighbouring letters, and a profile below the "
                "threshold is not joined at all. A dot marks the start of each pen-down path, so a joined pair appears as one path "
                "and an unjoined pair as two. A join also requires a compatible glyph source and a gap no larger than 0.80 "
                "x-heights, which stops a generic fallback shape being presented as natural cursive.")
    Stamp(fig,"synthesise.SynthesizeText, stored profiles")
    Save(fig,"T20_joining_by_writer")
    return rows


#======================================================================
#T27b / T51b  THE CANDIDATE SEARCH, SCORED
#reader accuracy, writer posterior and the harmonic mean that selects
#======================================================================
def T51(author,writer,text="Writing with style.",nTries=5,seed=2026):
    prof,fn = Profile(writer)
    reader = REC.PaperCRNNNumpy(READER_NPZ)
    namer = REC.AuthorClassifierCNNNumpy(SHAPE_SELECTED)
    mapping = namer.authorMapping
    if author not in mapping:
        print(f"  T51 skipped: {author} is not a class of {SHAPE_SELECTED.name}")
        return None
    idx = mapping[author]

    cands = []
    for k in range(nTries):
        traj = SY.SynthesizeText(text,prof,mmPerXh=4.0,seed=seed+977*k)
        img = SY.RenderTrajectory(traj,pxPerMm=18.0,uniformInk=True)
        got = REC.ReadText(img,reader)
        A = REC.CharAcc(got,text)
        _p,probs = REC.ClassifyImage(img,namer)
        P = float(probs[idx])
        J = 2*A*P/(A+P) if (A+P)>1e-9 else 0.0
        cands.append(dict(k=k,seed=seed+977*k,img=img,read=got,A=A,P=P,J=J))
    best = max(range(len(cands)),key=lambda i:cands[i]["J"])

    fig = plt.figure(figsize=(7.2,0.95*nTries+2.4))
    gs = fig.add_gridspec(nTries+1,1,height_ratios=[1]*nTries+[1.6],hspace=0.62)
    for i,c in enumerate(cands):
        ax = fig.add_subplot(gs[i])
        ShowGray(ax,c["img"],None)
        mark = "  <= selected" if i==best else ""
        ax.set_title(f"seed {c['seed']}:  A={c['A']:.3f}  P={c['P']:.3f}  "
                     f"J={c['J']:.3f}{mark}   read as “{c['read']}”",
                     fontsize=FS(7.6))
        for s in ax.spines.values():
            s.set_color(INK if i==best else SILVER)
            s.set_linewidth(1.2 if i==best else 0.5)

    ax = fig.add_subplot(gs[nTries])
    xs = np.arange(nTries)
    ax.bar(xs-0.26,[c["A"] for c in cands],0.25,color=SEQ[3],label="text score $A$")
    ax.bar(xs,[c["P"] for c in cands],0.25,color=SILVER,label="writer posterior $P$")
    ax.bar(xs+0.26,[c["J"] for c in cands],0.25,color=DARK,label="harmonic mean $J$")
    ax.set_xticks(xs);ax.set_xticklabels([f"cand {i+1}" for i in range(nTries)],
                                         fontsize=FS(7))
    ax.set_ylim(0,1.05);ax.set_ylabel("score",fontsize=FS(7.5))
    ax.legend(fontsize=FS(6.8),ncol=3,loc="upper center",bbox_to_anchor=(0.5,1.26))

    Caption(fig,"T51  Candidate selection for one writer, one sentence and a fixed search configuration. Every candidate is "
                "rendered at a constant ink width, read back and classified, and the harmonic mean is what selects: a candidate "
                "weak on either axis is penalised rather than rescued by strength on the other. The selecting models are the same "
                "ones used elsewhere in the system, so these scores are a design mechanism and not independent evidence that the "
                "output resembles the writer.")
    Stamp(fig,f"synthesise + recognise | {fn}, {SHAPE_SELECTED.name}")
    Save(fig,f"T51_candidate_selection_{writer}",f"best J={cands[best]['J']:.3f}")
    return [{k:v for k,v in c.items() if k!="img"} for c in cands]


if __name__=="__main__":
    print("T20-T51  synthesis visual tests")
    res = {}
    res["T48"] = T48("owen")
    res["T50"] = T50(["owen","dylan","yeukita"])
    res["T20"] = T20(["owen","dylan","yeukita","150"])
    res["T51"] = T51("owen","owen")
    res["T51_dylan"] = T51("dylan","dylan")
    (DATA/"synth_results.json").write_text(json.dumps(res,indent=1,default=float),
                                           encoding="utf-8")
    print("done")
