#u23539195 Dylan Braune.  visual tests for Section 3.7, integration, plus the
#software block diagram Section 3.2 still asks for.
#
#  python vt_system.py
from vt_common import *
import numpy as np
import json,tempfile
import segment as SEG
import recognise as REC
import synthesise as SY
import reproduce as RP


def Profile(writer):
    p = PROFILES/f"{writer}_curated.json"
    if not p.exists():
        p = PROFILES/f"{writer}.json"
    return json.loads(p.read_text(encoding="utf-8")),p.name


#======================================================================
#T59  THE PAGE VOTE
#per line, how much ink it carries and how it voted, against the pooled result
#======================================================================
def T59(writer="Owen",page=1,expect="owen"):
    #the page's stored full-resolution crops, which is the set the Design
    #draft's own two-crop vote trace is computed on
    folder = SEGCROPS/writer.capitalize()/f"{writer.capitalize()}{page:02d}"
    crops = sorted(folder.glob("*.png"))
    namer = REC.AuthorClassifierCNNNumpy(SHAPE_SELECTED)
    names = [k for k,_ in sorted(namer.authorMapping.items(),key=lambda kv:kv[1])]
    idx = namer.authorMapping[expect]

    rows = []
    for i,cp in enumerate(crops):
        crop = Gray(cp)
        _p,probs = REC.ClassifyImage(crop,namer)
        w = float((np.asarray(crop)<128).sum())
        rows.append(dict(i=i,w=w,probs=np.asarray(probs,dtype=float)))
    if not rows:
        print("  T59: no crops");return None

    W = np.array([r["w"] for r in rows])
    P = np.stack([r["probs"] for r in rows])
    pooled = (W[:,None]*P).sum(0)/max(W.sum(),1e-9)
    winner = int(pooled.argmax())
    agree = int((P.argmax(1)==idx).sum())

    fig = plt.figure(figsize=(7.4,4.8))
    gs = fig.add_gridspec(2,2,height_ratios=[1.5,1],width_ratios=[1.35,1],
                          hspace=0.45,wspace=0.3)

    axA = fig.add_subplot(gs[0,0])
    axA.imshow(P.T,cmap=CMAP,aspect="auto",interpolation="nearest",vmin=0,vmax=1)
    axA.set_yticks(range(len(names)));axA.set_yticklabels(names,fontsize=6.5)
    axA.set_xlabel("retained line crop",fontsize=7.5)
    axA.grid(False)
    axA.set_title(f"Per-line writer probabilities, {len(rows)} crops",fontsize=8.5)
    axA.plot(np.arange(len(rows)),np.full(len(rows),idx),ls="none",marker="_",
             ms=6,color=INK,mew=1.2)

    axB = fig.add_subplot(gs[0,1])
    order = np.argsort(-pooled)[:6]
    axB.barh(np.arange(len(order)),pooled[order],0.6,
             color=[DARK if names[o]==expect else SILVER for o in order])
    axB.set_yticks(np.arange(len(order)))
    axB.set_yticklabels([names[o] for o in order],fontsize=7)
    axB.invert_yaxis()
    axB.set_xlabel("ink-weighted pooled probability",fontsize=7.5)
    axB.set_xlim(0,1.05)
    for i,o in enumerate(order):
        axB.text(pooled[o]+0.02,i,f"{pooled[o]:.4f}",va="center",fontsize=6.6)
    axB.set_title(f"Pooled: {names[winner]}",fontsize=8.5)

    axC = fig.add_subplot(gs[1,:])
    xs = np.arange(len(rows))
    axC.bar(xs,W,0.6,color=SILVER,label="ink weight (pixels below grey 128)")
    axC.set_ylabel("ink weight",fontsize=7.5)
    axC.set_xlabel("retained line crop",fontsize=7.5)
    ax2 = axC.twinx()
    ax2.plot(xs,P[:,idx],color=DARK,lw=1.1,marker="o",ms=2.8,
             label=f"probability of {expect}")
    ax2.set_ylim(0,1.05);ax2.set_ylabel(f"p({expect})",fontsize=7.5)
    ax2.grid(False)
    h1,l1 = axC.get_legend_handles_labels();h2,l2 = ax2.get_legend_handles_labels()
    axC.legend(h1+h2,l1+l2,fontsize=6.8,ncol=2,loc="upper center",
               bbox_to_anchor=(0.5,1.26))

    Caption(fig,f"T59  Combining line evidence into one page decision. Each retained crop votes, weighted by how much ink it "
                f"carries, so a sparse line cannot outweigh a full one; {agree} of {len(rows)} individual crops agreed with the "
                "pooled result. Weighting by ink is also why crop rejection has to happen first: a dark artefact would otherwise "
                "carry a large weight. Illustration of Equation 3.30. The pooled probability and the agreement count are reported "
                "separately, and neither establishes accuracy without a reference.")
    Stamp(fig,f"recognise.ClassifyImage on stored crops | {writer}{page:02d}, {SHAPE_SELECTED.name}")
    Save(fig,"T59_page_vote",f"{names[winner]} {pooled[winner]:.4f}, {agree}/{len(rows)}")
    return dict(page=f"{writer}{page:02d}",crops=len(rows),pooled=names[winner],
                pooledP=float(pooled[winner]),linesAgreeing=agree)


#======================================================================
#T60  THE ROUND TRIP: TRAJECTORY, EMITTED FILE, PHYSICAL PAGE
#where a conversion error would show, and where an execution error would
#======================================================================
def T60(writers,text="Writing with style."):
    fig = plt.figure(figsize=(7.4,0.95*len(writers)+1.8))
    gs = fig.add_gridspec(len(writers),3,hspace=0.55,wspace=0.12,
                          width_ratios=[1.25,1.25,1.0])
    rows = []
    tmp = Path(tempfile.mkdtemp(prefix="vt_gcode_"))
    for r,(w,photo) in enumerate(writers):
        prof,fn = Profile(w)
        traj = SY.SynthesizeText(text,prof,mmPerXh=4.0,seed=2026)
        cfg = RP.GantryConfig()
        machine,scale = RP.ToMachine(traj,cfg,fitToBounds=True,optimizeOrder=True)
        path = tmp/f"{w}.gcode"
        RP.WriteGcode(traj,cfg,str(path),title=f"{w} visual test")
        parsed = RP.ParseGcode(str(path))
        back = parsed[0] if isinstance(parsed,tuple) else parsed

        ax = fig.add_subplot(gs[r,0])
        for s in machine:
            a = np.asarray(s,dtype=float)
            ax.plot(a[:,0],a[:,1],color=DARK,lw=0.8)
        ax.set_aspect("equal");ax.grid(False);ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(SILVER);sp.set_linewidth(0.5)
        ax.set_ylabel(w.capitalize(),fontsize=8,rotation=0,ha="right",
                      va="center",labelpad=10)
        if r==0: ax.set_title("(a) selected trajectory in machine mm",fontsize=8.3)

        ax = fig.add_subplot(gs[r,1])
        nV = 0
        for s in back:
            a = np.asarray(s,dtype=float)
            if len(a)>1:
                ax.plot(a[:,0],a[:,1],color=MID,lw=0.8)
            nV+=len(a)
        ax.set_aspect("equal");ax.grid(False);ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(SILVER);sp.set_linewidth(0.5)
        if r==0: ax.set_title("(b) re-parsed from the emitted G-code",fontsize=8.3)

        ax = fig.add_subplot(gs[r,2])
        if photo and photo.exists():
            #the pages were photographed on their side; stand them up, trim to
            #the writing, and take a band of the same proportion as the panels
            #beside it so the three are directly comparable
            a = np.rot90(Gray(photo,1400),k=1)
            #the dark page border would dominate a plain ink bbox, so the
            #edges are excluded and the band starts at the first row that
            #carries a real amount of writing
            H_,W_ = a.shape
            core = a[int(0.03*H_):int(0.97*H_),int(0.06*W_):int(0.94*W_)]
            rows_ = (core<120).sum(1)
            hit = np.nonzero(rows_>0.004*core.shape[1])[0]
            y0_ = int(0.03*H_)+(int(hit[0]) if len(hit) else 0)
            x0_,x1_ = int(0.04*W_),int(0.96*W_)
            band = max(10,int((x1_-x0_)/7.5))
            ShowGray(ax,a[y0_:y0_+band,x0_:x1_],None)
        else:
            ax.axis("off")
        if r==0: ax.set_title("(c) the first lines of a page this robot drew",fontsize=8.3)

        mv = sum(len(s) for s in machine)
        rows.append(dict(writer=w,strokesOut=len(machine),verticesOut=mv,
                         strokesBack=len(back),verticesBack=nV,
                         file=path.name,bytes=path.stat().st_size))

    Caption(fig,"T60  The two comparisons that localise a fault. Re-rendering the emitted motion file and comparing it with the "
                "selected trajectory checks the conversion, and would reveal an inverted axis, a scale change or a missing stroke "
                "before anything is drawn. Comparing the physical page with that render checks execution instead. Panel (c) is a "
                "photograph of real robot output and is a different sentence from (a) and (b); neither comparison measures "
                "positional accuracy, which is qualified in Results.")
    Stamp(fig,"synthesise + reproduce.WriteGcode / ParseGcode; photographs from ReportResults")
    Save(fig,"T60_round_trip")
    return rows


if __name__=="__main__":
    print("T15-T61  integration visual tests")
    res = {}
    from vt_blockdiagram import T15
    T15()
    from vt_jobstates import T61
    T61()
    res["T59"] = T59("Owen",1,"owen")
    pics = [("owen",REPRO/"AuthorO.jpg"),("dylan",REPRO/"AuthorD.jpg"),
            ("yeukita",REPRO/"AuthorY.jpg")]
    res["T60"] = T60(pics)
    (DATA/"system_results.json").write_text(json.dumps(res,indent=1,default=float),
                                            encoding="utf-8")
    print("done")
