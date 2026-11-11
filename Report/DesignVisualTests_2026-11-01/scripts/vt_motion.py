#u23539195 Dylan Braune.  visual tests for Section 3.3.2 (the drive) and
#Section 3.6.7 (machine mapping and G-code).
#
#nothing here touches the gantry.  the pulse figures evaluate the implemented
#stepping arithmetic on real generated moves; the G-code figures read files
#that were generated offline.
#
#  python vt_motion.py
from vt_common import *
import numpy as np
import json
import synthesise as SY
import reproduce as RP


def Profile(writer):
    p = PROFILES/f"{writer}_curated.json"
    if not p.exists():
        p = PROFILES/f"{writer}.json"
    return json.loads(p.read_text(encoding="utf-8")),p.name


#---Pretty: a profile name as it should be printed in a figure---
def Pretty(name):
    s = str(name)
    return Author(s)


#---Travel: pen-down and pen-up distance of a list of machine strokes---
def Travel(strokes):
    import math
    down = sum(sum(math.dist(a,b) for a,b in zip(s,s[1:]))
               for s in strokes if len(s)>1)
    up = sum(math.dist(a[-1],b[0]) for a,b in zip(strokes,strokes[1:]))
    return float(down),float(up)


#---Bounds: the extent of a list of machine strokes---
def Bounds(strokes):
    pts = [p for s in strokes for p in s]
    xs = [p[0] for p in pts];ys = [p[1] for p in pts]
    return min(xs),min(ys),max(xs),max(ys)


def Job(writer,text,mmPerXh=4.0,seed=2026):
    """one real trajectory and its machine placement, as the converter makes it"""
    prof,fn = Profile(writer)
    traj = SY.SynthesizeText(text,prof,mmPerXh=mmPerXh,seed=seed)
    cfg = RP.GantryConfig()
    machine,scale = RP.ToMachine(traj,cfg,fitToBounds=True,optimizeOrder=True)
    return prof,fn,traj,cfg,machine


#======================================================================
#T52  THE BRESENHAM STAIRCASE ON A REAL MOVE
#the integer accumulator, shown against the straight line it approximates
#======================================================================
def T52(cfg,machine):
    kx = cfg.fullStepsPerRev*cfg.microstepping/cfg.mmPerRevX
    ky = cfg.fullStepsPerRev*cfg.microstepping/cfg.mmPerRevY

    #---Walk: the implemented loop, major axis first, accumulator at NM//2
    def Walk(dx,dy):
        NM,Nm = max(abs(dx),abs(dy)),min(abs(dx),abs(dy))
        majorIsX = abs(dx)>=abs(dy)
        sx,sy = (1 if dx>0 else -1),(1 if dy>0 else -1)
        e = NM//2
        cx = cy = 0
        path = [(0,0)]
        for _ in range(NM):
            if majorIsX: cx+=sx
            else: cy+=sy
            e-=Nm
            if e<0:
                if majorIsX: cy+=sy
                else: cx+=sx
                e+=NM
            path.append((cx,cy))
        return np.array(path,float),NM,Nm,majorIsX

    #---Dev: how far the staircase departs from the straight line
    def Dev(path,dx,dy):
        t = np.linspace(0,1,len(path))
        ideal = np.stack([t*dx,t*dy],1)
        return np.abs(path-ideal).max(1)

    #the segment of the real job where the staircase departs furthest from
    #the straight line it approximates.  an exactly diagonal move is the one
    #case the interpolation gets right for free, so picking the most
    #diagonal move would show the mechanism at its least interesting
    best,bestDev = None,-1.0
    for s in machine:
        a_ = np.asarray(s,dtype=float)
        for p_,q_ in zip(a_,a_[1:]):
            dxi = int(round(q_[0]*kx))-int(round(p_[0]*kx))
            dyi = int(round(q_[1]*ky))-int(round(p_[1]*ky))
            if abs(dxi)<2 or abs(dyi)<2:continue
            if max(abs(dxi),abs(dyi))>50:continue
            pth,_NM,_Nm,_mx = Walk(dxi,dyi)
            d_ = float(Dev(pth,dxi,dyi).max())
            if d_>bestDev:
                best,bestDev = (p_,q_,dxi,dyi),d_
    p,q,dx,dy = best
    path,NM,Nm,majorIsX = Walk(dx,dy)
    dev = Dev(path,dx,dy)

    #the staircase is a tall narrow move, so it is given the narrow panel and
    #the deviation trace the wide one, instead of both sharing half the width
    fig,(axA,axB) = plt.subplots(1,2,figsize=(WIDTH,4.05),
                                 gridspec_kw=dict(width_ratios=[1,1.55],
                                                  wspace=0.32,left=0.105,
                                                  right=0.985,top=0.855,
                                                  bottom=0.235))

    axA.step(path[:,0],path[:,1],where="post",color=DARK,lw=1.5,
             label="commanded microsteps")
    axA.plot(path[:,0],path[:,1],ls="none",marker="o",ms=3.4,color=DARK)
    axA.plot([0,dx],[0,dy],color=SILVER,lw=1.6,ls="--",label="ideal straight line")
    axA.set_aspect("equal")
    axA.set_xlabel("x microsteps",fontsize=FS(8.2))
    axA.set_ylabel("y microsteps",fontsize=FS(8.2))
    axA.set_title(f"A real move: ({dx:+d}, {dy:+d}) microsteps\n"
                  f"major axis {'x' if majorIsX else 'y'}, "
                  f"accumulator starts at {NM//2}",fontsize=FS(8.5))
    #the legend sits under the panel: inside, it covered the staircase the
    #panel exists to show, and above it ran into the title
    axA.legend(fontsize=FS(8),loc="upper center",bbox_to_anchor=(0.5,-0.165),
               frameon=False,handlelength=1.8,labelspacing=0.3,
               borderaxespad=0.0,ncol=1)

    #deviation from the ideal line, in microsteps and in millimetres
    axB.plot(np.arange(len(dev)),dev,color=DARK,lw=1.5,marker="o",ms=3.0)
    axB.axhline(1.0,color=SILVER,lw=1.2,ls="--")
    axB.set_ylim(-0.04,1.22)
    axB.text(0.02,1.025,"one microstep: the bound never exceeded",
             fontsize=FS(8),ha="left",va="bottom",color=INK,
             transform=axB.get_yaxis_transform())
    axB.set_xlabel("iteration of the stepping loop",fontsize=FS(8.2))
    axB.set_ylabel("deviation from the ideal line\n(microsteps)",fontsize=FS(8.2))
    axB.set_title(f"Worst deviation {dev.max():.2f} microsteps\n"
                  f"= {dev.max()/kx*1000:.1f} $\\mu$m at "
                  f"{kx:.0f} microsteps/mm",fontsize=FS(8.5))

    Caption(fig,"T52  Coordinated two-axis motion on a segment taken from a real generated job. The major axis is pulsed every "
                "iteration and an integer accumulator inserts the minor-axis pulses along the way, so the commanded endpoint is "
                "reached exactly and the error never exceeds one microstep. The quantisation is far finer than the positional "
                "requirement, but it is command granularity: belt compliance, frame geometry and pen deflection all sit between "
                "this and the mark on paper.")
    Stamp(fig,"reproduce.BresenhamMove arithmetic, real generated job")
    Save(fig,"T52_bresenham_staircase",
         f"({dx},{dy}), worst {dev.max():.2f} microsteps")
    return dict(dx=int(dx),dy=int(dy),worstDeviationMicrosteps=float(dev.max()),
                microstepsPerMm=float(kx))


#======================================================================
#T53  THE DIRECTIONAL SPEED PENALTY
#predicted from the implementation alone, no measurement needed
#======================================================================
def T53(cfg,machine,feed=900.0):
    kx = cfg.fullStepsPerRev*cfg.microstepping/cfg.mmPerRevX
    v = feed/60.0

    #the analytic curve from Equation 3.26: an iteration that issues two
    #pulses sleeps twice, so the penalty follows the axis ratio
    ang = np.linspace(0,90,361)
    NM = np.maximum(np.abs(np.cos(np.radians(ang))),np.abs(np.sin(np.radians(ang))))
    Nm = np.minimum(np.abs(np.cos(np.radians(ang))),np.abs(np.sin(np.radians(ang))))
    ratio = (NM+Nm)/NM

    #and the same quantity evaluated on every segment of a real job
    angs,rats = [],[]
    for s in machine:
        a = np.asarray(s,dtype=float)
        for p,q in zip(a,a[1:]):
            dx = int(round(q[0]*kx))-int(round(p[0]*kx))
            dy = int(round(q[1]*kx))-int(round(p[1]*kx))
            NMi,Nmi = max(abs(dx),abs(dy)),min(abs(dx),abs(dy))
            if NMi==0:continue
            d = float(np.hypot(q[0]-p[0],q[1]-p[1]))
            tau = max(d/(v*NMi),2e-6)
            tdelay = (NMi+Nmi)*tau
            geom = d/v
            angs.append(np.degrees(np.arctan2(abs(dy),abs(dx))))
            rats.append(tdelay/max(geom,1e-12))

    fig,(axA,axB) = plt.subplots(1,2,figsize=(7.2,3.1),
                                 gridspec_kw=dict(width_ratios=[1,1],wspace=0.3))
    axA.plot(ang,ratio,color=DARK,lw=1.4,label="predicted by Equation 3.26")
    axA.plot(angs,rats,ls="none",marker="o",ms=2.0,mfc="none",mec=SILVER,
             mew=0.6,label=f"{len(angs)} real segments of one job")
    axA.axhline(1.0,color=INK,lw=0.8,ls=":")
    axA.axhline(2.0,color=INK,lw=0.8,ls=":")
    axA.set_xlim(0,90);axA.set_ylim(0.9,2.15)
    axA.set_xticks([0,15,30,45,60,75,90])
    axA.set_xlabel("move direction from the x axis (degrees)")
    axA.set_ylabel("delay / geometric feed budget")
    axA.set_title("A $45^\\circ$ move sleeps about twice as long\n"
                  "as an axis-parallel move of the same length",fontsize=FS(8.5))
    axA.legend(fontsize=FS(6.8),loc="lower right")

    #what that costs over the whole job
    axB.hist(rats,bins=28,color=PALE,edgecolor=DARK,linewidth=0.6)
    axB.axvline(float(np.mean(rats)),color=INK,lw=1.2,ls="--")
    axB.text(float(np.mean(rats)),axB.get_ylim()[1]*0.95,
             f"  mean {np.mean(rats):.3f}",fontsize=FS(7),va="top",color=INK)
    axB.set_xlabel("delay / geometric budget")
    axB.set_ylabel("segments")
    axB.set_title("Distribution over one real job",fontsize=FS(8.5))

    Caption(fig,"T53  A directional speed variation that follows from the implementation rather than from the mechanism. The "
                "stepping loop sleeps inside each axis-pulse call, so an iteration that issues both a major and a minor pulse "
                "takes twice as long as one that issues a single pulse. The requested feed is therefore not a measured trajectory "
                "speed, and a shared pulse scheduler would remove the dependence. The points are real segments of a generated "
                "job; the curve is the predicted ratio.")
    Stamp(fig,"reproduce._StepAxis timing model, real generated job")
    Save(fig,"T53_directional_pulse_penalty",f"mean {np.mean(rats):.3f}")
    return dict(segments=len(rats),meanRatio=float(np.mean(rats)),
                maxRatio=float(np.max(rats)))


#======================================================================
#T54  STROKE-ORDER OPTIMISATION
#a measured design win: how much pen-up travel the reordering saves
#======================================================================
def T54(writers,text="Writing with style, one line at a time."):
    fig = plt.figure(figsize=(7.2,1.5*len(writers)+2.2))
    gs = fig.add_gridspec(len(writers)+1,2,
                          height_ratios=[1]*len(writers)+[1.5],
                          hspace=0.5,wspace=0.1)
    rows = []
    for r,w in enumerate(writers):
        prof,fn = Profile(w)
        traj = SY.SynthesizeText(text,prof,mmPerXh=4.0,seed=5)
        cfg = RP.GantryConfig()
        raw,_ = RP.ToMachine(traj,cfg,fitToBounds=True,optimizeOrder=False)
        opt,_ = RP.ToMachine(traj,cfg,fitToBounds=True,optimizeOrder=True)
        for c,(m,lbl) in enumerate([(raw,"as synthesised"),(opt,"reordered")]):
            d,u = Travel(m)
            ax = fig.add_subplot(gs[r,c])
            for s in m:
                a = np.asarray(s,dtype=float)
                ax.plot(a[:,0],a[:,1],color=SILVER,lw=0.7)
            for a_,b_ in zip(m,m[1:]):
                ax.plot([a_[-1][0],b_[0][0]],[a_[-1][1],b_[0][1]],
                        color=DARK,lw=0.5,alpha=0.85)
            ax.set_aspect("equal");ax.grid(False)
            ax.set_xticks([]);ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(SILVER);s.set_linewidth(0.5)
            ax.set_title(f"{Author(w)}, {lbl}:  pen-up travel {u:.1f} mm",
                         fontsize=FS(8))
            if c==0: rawU = u
            else: rows.append((w,rawU,u,d))

    ax = fig.add_subplot(gs[len(writers),:])
    xs = np.arange(len(rows))
    ax.bar(xs-0.2,[r[1] for r in rows],0.38,color=SILVER,label="as synthesised")
    ax.bar(xs+0.2,[r[2] for r in rows],0.38,color=DARK,label="after reordering")
    ax.set_xticks(xs)
    ax.set_xticklabels([Author(r[0]) for r in rows],fontsize=FS(7.5))
    ax.set_ylabel("pen-up travel (mm)",fontsize=FS(7.5))
    for i,r in enumerate(rows):
        ax.text(i+0.2,r[2],f"-{100*(1-r[2]/max(r[1],1e-9)):.0f}%",
                ha="center",va="bottom",fontsize=FS(7),color=INK)
    ax.legend(fontsize=FS(6.8),ncol=2,loc="upper center",bbox_to_anchor=(0.5,1.22))

    Caption(fig,"T54  Ordering the strokes before writing the motion file. Silver is ink on the page and blue is pen-up travel "
                "between strokes. Strokes are sorted into text rows first and ordered within each row, so shortening the path "
                "cannot scramble reading order. The saving is non-writing time only: the ink laid down is identical, which is "
                "what makes this a free improvement rather than a trade-off.")
    Stamp(fig,"reproduce.ToMachine / OptimizeStrokeOrder")
    Save(fig,"T54_stroke_order")
    return [dict(writer=r[0],penUpBeforeMm=round(r[1],2),
                 penUpAfterMm=round(r[2],2),drawMm=round(r[3],2)) for r in rows]


#======================================================================
#T56  WHAT THE NOMINAL TIME BUDGET IS MADE OF
#the dimensional calculation, decomposed
#======================================================================
def T56(writers,text="Writing with style."):
    rows = []
    for w in writers:
        prof,fn = Profile(w)
        traj = SY.SynthesizeText(text,prof,mmPerXh=4.0,seed=2026)
        cfg = RP.GantryConfig()
        m,_ = RP.ToMachine(traj,cfg,fitToBounds=True,optimizeOrder=True)
        Dd,Du = Travel(m)
        Fd = float(cfg.drawFeedMmMin)
        Fu = float(cfg.travelFeedMmMin)
        up = float(cfg.penUpDwellMs)/1000.0
        dn = float(cfg.penDownDwellMs)/1000.0
        Np = 2*len(m)
        tDraw = 60*Dd/Fd
        tTrav = 60*Du/Fu
        tDwell = Np*(up+dn)/2
        rows.append((w,tDraw,tTrav,tDwell,Np,Dd,Du))

    fig,(axA,axB) = plt.subplots(1,2,figsize=(WIDTH,3.30),
                                 gridspec_kw=dict(width_ratios=[1.35,1],
                                                  wspace=0.26,left=0.098,
                                                  right=0.985,top=0.845,
                                                  bottom=0.165))
    xs = np.arange(len(rows))
    d = np.array([r[1] for r in rows]);t = np.array([r[2] for r in rows])
    w_ = np.array([r[3] for r in rows])
    names = [Pretty(r[0]) for r in rows]
    axA.bar(xs,d,0.60,color=DARK,label="drawing travel")
    axA.bar(xs,t,0.60,bottom=d,color=SEQ[3],label="pen-up travel")
    axA.bar(xs,w_,0.60,bottom=d+t,color=SILVER,label="pen dwell",
            edgecolor=INK,linewidth=0.4)
    axA.set_xticks(xs)
    axA.set_xticklabels(names,fontsize=FS(8))
    axA.set_ylabel("nominal job time (s)",fontsize=FS(8.2))
    axA.set_ylim(0,max(d+t+w_)*1.22)
    #the dwell seconds are the point of the figure, so they are printed
    #inside the band rather than left to be read off the axis
    for i,r in enumerate(rows):
        tot = r[1]+r[2]+r[3]
        axA.text(i,tot+0.45,f"{tot:.1f} s",ha="center",fontsize=FS(8),color=INK)
        axA.text(i,d[i]+t[i]+w_[i]/2,f"{w_[i]:.1f}",ha="center",va="center",
                 fontsize=FS(7.6),color=INK)
    axA.legend(fontsize=FS(8),ncol=3,loc="upper center",
               bbox_to_anchor=(0.5,1.175),frameon=False,handlelength=1.5,
               columnspacing=1.2,handletextpad=0.5)
    axA.set_title("Where a job's nominal time goes",fontsize=FS(8.5),pad=22)
    axA.tick_params(labelsize=FS(8))

    frac = 100*w_/(d+t+w_)
    axB.barh(xs,frac,0.60,color=DARK)
    axB.set_yticks(xs)
    axB.set_yticklabels(names,fontsize=FS(8))
    axB.invert_yaxis()
    axB.set_xlabel("pen dwell as a share of\nthe nominal budget (%)",
                   fontsize=FS(8.2))
    axB.set_xlim(0,100)
    axB.tick_params(labelsize=FS(8))
    for i,f in enumerate(frac):
        axB.text(f+2.0,i,f"{f:.1f}%",va="center",fontsize=FS(8),color=INK)
    axB.set_title("Dwell dominates every job",fontsize=FS(8.5),pad=22)

    Caption(fig,"T56  The converter's dimensional time budget decomposed for one sentence from four stored profiles: distance "
                "divided by feed for the drawing and travel moves, plus the configured dwell at every pen transition. The dwell "
                "term is the largest single component, so a timing argument that counted only geometric travel would omit most of "
                "the budget. This is the converter's estimate, not a measured completion time. Illustration of Equation 3.29.")
    Stamp(fig,"reproduce.ToMachine, GantryConfig feeds and dwells")
    Save(fig,"T56_time_budget")
    return [dict(writer=r[0],drawS=round(r[1],3),travelS=round(r[2],3),
                 dwellS=round(r[3],3),transitions=r[4],
                 drawMm=round(r[5],3),travelMm=round(r[6],3)) for r in rows]


#======================================================================
#T57  PLACEMENT INTO THE CALIBRATED WORKSPACE
#margins, bounds and the fit, on a real job
#======================================================================
def T57(writer="owen",text="Writing with style, one line at a time, on a ruled page."):
    prof,fn,traj,cfg,m = Job(writer,text)
    x0,y0,x1,y1 = Bounds(m)

    fig,ax = plt.subplots(figsize=(4.6,5.4))
    #the calibrated travel
    ax.add_patch(plt.Rectangle((cfg.boundsMinXmm,cfg.boundsMinYmm),
                               cfg.boundsMaxXmm-cfg.boundsMinXmm,
                               cfg.boundsMaxYmm-cfg.boundsMinYmm,
                               fill=False,ec=INK,lw=1.2))
    #the usable area inside the margins
    ax.add_patch(plt.Rectangle((cfg.originXmm+cfg.pageMarginXmm,
                                cfg.boundsMinYmm+cfg.pageMarginYmm),
                               cfg.boundsMaxXmm-cfg.originXmm-2*cfg.pageMarginXmm,
                               cfg.originYmm-cfg.pageMarginYmm-cfg.boundsMinYmm,
                               fill=False,ec=SILVER,lw=1.0,ls="--"))
    for s in m:
        a = np.asarray(s,dtype=float)
        ax.plot(a[:,0],a[:,1],color=DARK,lw=0.7)
    ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec=MID,lw=0.9,
                               ls=":"))
    ax.set_aspect("equal")
    ax.set_xlabel("machine x (mm)");ax.set_ylabel("machine y (mm)")
    ax.set_title(f"{Author(writer)}: a real job placed in the calibrated "
                 f"workspace\ntravel {cfg.boundsMaxXmm:.1f} x "
                 f"{cfg.boundsMaxYmm:.1f} mm, margins "
                 f"{cfg.pageMarginXmm:.0f}/{cfg.pageMarginYmm:.0f} mm",
                 fontsize=FS(8.5))
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([],[],color=INK,lw=1.2,label="calibrated travel"),
        Line2D([],[],color=SILVER,lw=1.0,ls="--",label="usable area inside margins"),
        Line2D([],[],color=MID,lw=0.9,ls=":",label="trajectory bounds"),
        Line2D([],[],color=DARK,lw=0.8,label="pen-down paths")],
        fontsize=FS(6.6),loc="lower center",bbox_to_anchor=(0.5,-0.22),ncol=2)

    Caption(fig,"T57  Placing a trajectory into machine coordinates. The margin is subtracted at both the left and the right, so "
                "a long line cannot run to the edge of the travel and meet the run-time clamp. Both the stored trajectory and the "
                "machine frame use y increasing upwards, so later lines are placed lower on the page without another image flip. "
                "Clamping remains a last boundary protection: a clamped point distorts a letter and is not successful layout. "
                "Illustration of Equation 3.28.")
    Stamp(fig,f"reproduce.ToMachine / GantryConfig | {fn}")
    Save(fig,"T57_machine_placement")
    return dict(boundsMm=[round(v,3) for v in (x0,y0,x1,y1)],
                travelX=cfg.boundsMaxXmm,travelY=cfg.boundsMaxYmm,
                marginX=cfg.pageMarginXmm,marginY=cfg.pageMarginYmm)


#======================================================================
#T55  DECIMATION AND THE NEAR-STROKE JOIN
#both save time and both can change the ink.  shown at several settings
#======================================================================
def T55(writer="owen",text="Writing with style."):
    prof,fn = Profile(writer)
    traj = SY.SynthesizeText(text,prof,mmPerXh=4.0,seed=2026)
    cfg = RP.GantryConfig()
    base,_ = RP.ToMachine(traj,cfg,fitToBounds=True,optimizeOrder=True)

    mins = [0.0,0.2,0.6,1.5,3.0]
    rows = []
    fig = plt.figure(figsize=(7.2,0.95*len(mins)+2.1))
    gs = fig.add_gridspec(len(mins)+1,1,height_ratios=[1]*len(mins)+[1.5],
                          hspace=0.55)
    for i,mm in enumerate(mins):
        strokes = [RP._Decimate(s,mm) if mm>0 else s for s in base]
        nV = sum(len(s) for s in strokes)
        ax = fig.add_subplot(gs[i])
        for s in strokes:
            a = np.asarray(s,dtype=float)
            if len(a)>1:
                ax.plot(a[:,0],a[:,1],color=DARK if mm<=0.6 else SILVER,lw=0.9)
                ax.plot(a[:,0],a[:,1],ls="none",marker="o",ms=1.1,color=INK,
                        alpha=0.5)
        ax.set_aspect("equal");ax.grid(False)
        ax.set_xticks([]);ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(SILVER);s.set_linewidth(0.5)
        ax.set_title(f"minimum segment {mm:.1f} mm:  {nV} vertices"
                     + ("   (the implemented default join distance is 0.6 mm)"
                        if mm==0.6 else ""),fontsize=FS(8.3))
        rows.append((mm,nV))

    ax = fig.add_subplot(gs[len(mins)])
    xs = np.arange(len(rows))
    ax.bar(xs,[r[1] for r in rows],0.55,
           color=[DARK if r[0]<=0.6 else SILVER for r in rows])
    ax.set_xticks(xs);ax.set_xticklabels([f"{r[0]:.1f} mm" for r in rows],
                                         fontsize=FS(7.5))
    ax.set_ylabel("vertices emitted",fontsize=FS(7.5))
    for i,r in enumerate(rows):
        ax.text(i,r[1],f"{r[1]}",ha="center",va="bottom",fontsize=FS(7),color=INK)
    ax.set_title("Each vertex is one G1 command and one pulse-loop entry",
                 fontsize=FS(8.5))

    Caption(fig,"T55  Path decimation. Removing short segments cuts the number of emitted commands, and because each vertex is a "
                "separate entry to the stepping loop the saving is real. It is bounded by geometry: beyond about a millimetre the "
                "curves visibly straighten, and a join distance set too generously closes gaps the writer left open. Dark blue "
                "marks the settings at or below the implemented default.")
    Stamp(fig,"reproduce._Decimate")
    Save(fig,"T55_decimation_ladder")
    return [dict(minSegMm=r[0],vertices=r[1]) for r in rows]


if __name__=="__main__":
    print("T52-T57  motion and machine-mapping visual tests")
    res = {}
    prof,fn,traj,cfg,machine = Job("owen","Writing with style.")
    res["T52"] = T52(cfg,machine)
    res["T53"] = T53(cfg,machine)
    res["T54"] = T54(["owen","dylan","yeukita"])
    res["T55"] = T55("owen")
    #the four named personal profiles, so the figure reads beside the rest
    #of the set; the earlier run used the numbered IAM profile 150 here
    res["T56"] = T56(["owen","dylan","yeukita","abhinav"])
    res["T57"] = T57("owen")
    (DATA/"motion_results.json").write_text(json.dumps(res,indent=1,default=float),
                                            encoding="utf-8")
    print("done")
