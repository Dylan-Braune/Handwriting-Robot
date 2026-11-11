#u23539195 Dylan Braune.  visual tests for Section 3.4, page conditioning
#and line segmentation.  every panel is produced by calling segment.py on a
#real photographed page; none of this measures a specification.
#
#  python -I scripts/vt_seg.py
from vt_common import *
import numpy as np
import segment as SEG


#======================================================================
#STAGES: run the conditioning chain on one page, keeping every intermediate
#this repeats the body of SEG.SegmentLines rather than calling it, because
#the figure needs the arrays it discards.  the order and the arguments are
#the same, so what is drawn is what the segmenter actually computes
#======================================================================
def Stages(pagePath,maxSide=None):
    page = Gray(pagePath,maxSide)
    full = np.asarray(page,dtype=np.uint8)
    half0 = SEG.Halve(full)

    #--- the angle search, scored exactly as Deskew scores it
    mask0,ratio0 = SEG.InkMask(half0.astype(np.float64))
    q = SEG.Halve(mask0.astype(np.uint8)*255)>127
    angles = np.arange(-4,4.01,0.5)
    scores = np.array([SEG.SkewScore(q,a) for a in angles])
    best = float(angles[int(scores.argmax())])

    fullA = SEG.Rotate(full,best,fill=255)
    halfA = SEG.Rotate(half0,best,fill=255)
    g = halfA.astype(np.float64)

    #--- the ratio image and its threshold, the two halves of Equation 3.1
    r = max(8,min(40,round(g.shape[0]/41)))
    ratio = g/np.maximum(SEG.BoxMean(g,r),1.0)
    thr = SEG.Otsu(ratio)
    maskRaw,_ = SEG.InkMask(g)

    #--- cleaning
    maskPrint,paper,P0,paperRemoved,removed = SEG.RemovePrintedStructure(maskRaw.copy(),g)
    maskClean,lab,nc,area,nArt = SEG.DropArtifactPieces(maskPrint.copy(),ratio,P0,*maskPrint.shape)

    #--- lines
    H,W = maskClean.shape
    P,centres = SEG.LineCentres(maskClean,H)
    cuts,cutsArr,L = SEG.SeamCuts(maskClean,centres,P,H,W)

    #--- ownership, both the first guess and the corrected result
    bandOwn,ownBeforeEF,region,areaC,ys,xs = SEG._BandOwnership(
        maskClean,lab,nc,cuts,cutsArr,L,P,H,W,centres)
    bandOnly = bandOwn.copy()
    own = SEG.AssignOwnership(maskClean,lab,nc,cuts,cutsArr,L,centres,P,H,W)

    #--- the row profile LinePitch works from, smoothed the same way
    prof = maskClean[:,int(0.05*W):int(0.97*W)].sum(1).astype(float)
    k = np.exp(-0.5*(np.arange(-12,13)/4.0)**2)
    sm = np.convolve(prof,k/k.sum(),"same")

    return dict(full=full,fullA=fullA,halfA=halfA,g=g,ratio=ratio,thr=thr,
                half0=half0,ratio0=ratio0,
                angles=angles,scores=scores,best=best,
                maskRaw=maskRaw,maskPrint=maskPrint,maskClean=maskClean,
                paper=paper,removed=removed,nArt=nArt,
                P=P,centres=centres,cuts=cuts,cutsArr=cutsArr,L=L,
                bandOnly=bandOnly,own=own,prof=prof,sm=sm,
                lab=lab,nc=nc,r=r)


#======================================================================
#T21  OTSU'S CRITERION ON THE RATIO HISTOGRAM
#the written trace gives bin 171 and a between-class variance of 564.98.
#this plots the whole curve so the reader can see the maximum is isolated
#======================================================================
def T21(S,tag="Owen01"):
    #the pre-deskew reduced page, which is the image the Design draft's own
    #worked trace is computed on.  using the post-deskew one would quote a
    #different threshold beside the same equation
    ratio = S["ratio0"]
    hi = 1.2
    hist = np.histogram(np.clip(ratio,0,hi),bins=256,range=(0,hi))[0].astype(float)
    p = hist/max(hist.sum(),1.0)
    w = np.cumsum(p)
    m = np.cumsum(p*np.arange(256))
    var = (m[-1]*w-m)**2/np.maximum(w*(1-w),1e-12)
    t = int(np.argmax(var))

    #the two class masses and means at the selected bin, as in the report
    w0 = float(w[t]); w1 = 1.0-w0
    mu0 = float(m[t]/max(w[t],1e-12))
    mu1 = float((m[-1]-m[t])/max(1-w[t],1e-12))
    bcv = w0*w1*(mu0-mu1)**2

    fig = plt.figure(figsize=(7.2,4.6))
    gs = fig.add_gridspec(3,2,height_ratios=[1,1,1],width_ratios=[1.5,1],
                          hspace=0.55,wspace=0.2)

    ax = fig.add_subplot(gs[0,0])
    ax.fill_between(np.arange(256),hist,color=PALE,lw=0)
    ax.plot(np.arange(256),hist,color=DARK,lw=0.8)
    ax.axvline(t,color=INK,lw=1.0,ls="--")
    ax.annotate(f"bin {t},  $t^*$={(t+0.5)*hi/256:.6f}",(t,hist.max()*0.55),
                xytext=(t-150,hist.max()*0.72),fontsize=FS(7.5),color=INK,
                arrowprops=dict(arrowstyle="->",color=INK,lw=0.7))
    ax.set_title("Ratio histogram, 256 bins over 0 to 1.2",fontsize=FS(8.5))
    ax.set_xlabel("bin index");ax.set_ylabel("pixels")
    ax.set_xlim(0,255)

    ax = fig.add_subplot(gs[1:,0])
    ax.plot(np.arange(256),var,color=DARK,lw=1.1)
    ax.axvline(t,color=INK,lw=1.0,ls="--")
    ax.plot([t],[var[t]],"o",ms=4,color=INK,zorder=5)
    ax.set_title(r"Between-class variance $\omega_0\omega_1(\mu_0-\mu_1)^2$",fontsize=FS(8.5))
    ax.set_xlabel("candidate threshold bin");ax.set_ylabel("variance")
    ax.set_xlim(0,255)
    ax.text(0.98,0.06,
            f"$\\omega_0$={w0:.5f}  $\\omega_1$={w1:.5f}\n"
            f"$\\mu_0$={mu0:.2f}  $\\mu_1$={mu1:.2f}\n"
            f"maximum {bcv:.2f}",
            transform=ax.transAxes,ha="right",va="bottom",fontsize=FS(7),
            bbox=dict(fc=PAPER,ec=SILVER,lw=0.5,pad=2.5))

    #a band of real writing at the selected threshold and either side of it,
    #zoomed far enough that individual strokes are legible
    Hh,Ww = ratio.shape
    band = (slice(int(0.30*Hh),int(0.40*Hh)),slice(int(0.08*Ww),int(0.72*Ww)))
    for cell,(bin_,lbl) in zip([gs[0,1],gs[1,1],gs[2,1]],
                               [(max(0,t-40),"40 bins lower: strokes lost"),
                                (t,"selected $t^*$"),
                                (min(255,t+40),"40 bins higher: rules and shadow admitted")]):
        axn = fig.add_subplot(cell)
        ShowMask(axn,(ratio<(bin_+0.5)*hi/256)[band],f"{lbl}  (bin {bin_})")

    Caption(fig,"T21  Otsu's criterion evaluated on the local-brightness ratio image of one real page, with the same band of "
                "writing thresholded at the selected bin and forty bins either side. The maximum is isolated, and the two "
                "detuned panels show what is lost and what is admitted. Illustration of Equations 3.1 and 3.3.")
    Stamp(fig,f"segment.Otsu / segment.InkMask | {tag}")
    Save(fig,"T21_otsu_criterion",f"bin {t}, bcv {bcv:.2f}")
    return dict(bin=t,threshold=(t+0.5)*hi/256,w0=w0,w1=w1,mu0=mu0,mu1=mu1,bcv=bcv)


#======================================================================
#T22  DESKEW OBJECTIVE OVER THE WHOLE SEARCH
#the report quotes three of the seventeen angles.  this shows all of them,
#which is what says whether the maximum is a peak or a plateau
#======================================================================
def T22(pages):
    fig = plt.figure(figsize=(7.2,4.4))
    gs = fig.add_gridspec(2,3,height_ratios=[1.15,1],hspace=0.45,wspace=0.22)

    ax = fig.add_subplot(gs[0,:])
    for i,(name,S) in enumerate(pages):
        s = S["scores"]/S["scores"].max()
        ax.plot(S["angles"],s,marker="o",ms=3.2,lw=1.1,
                color=CAT[i%len(CAT)],label=f"{name}  ({S['best']:+.1f}$^\\circ$)")
        ax.plot([S["best"]],[1.0],marker="o",ms=6,mfc="none",
                mec=CAT[i%len(CAT)],mew=1.2)
    ax.set_xlabel("trial angle (degrees)")
    ax.set_ylabel("squared row projection\n(normalised to each page's maximum)")
    #\Sigma, not \sum: matplotlib takes a large operator from a Computer
    #Modern extension face, and that would be a second font in the set
    ax.set_title(r"Deskew objective $\Sigma_y\, p_\theta(y)^2$ over the implemented "
                 r"$-4^\circ$ to $+4^\circ$ search at $0.5^\circ$ spacing",fontsize=FS(8.5))
    ax.legend(fontsize=FS(7),ncol=3,loc="lower center")
    ax.set_xlim(-4.3,4.3)

    #the first page at its chosen angle, and deliberately wrong
    name,S = pages[0]
    H = S["full"].shape[0]
    band = slice(int(0.18*H),int(0.34*H))
    for j,(a,lbl) in enumerate([(S["best"],f"selected {S['best']:+.1f}$^\\circ$"),
                                (S["best"]-3.0,f"{S['best']-3.0:+.1f}$^\\circ$"),
                                (S["best"]+3.0,f"{S['best']+3.0:+.1f}$^\\circ$")]):
        axn = fig.add_subplot(gs[1,j])
        rot = SEG.Rotate(S["full"],a,fill=255)
        ShowGray(axn,rot[band],f"{name}, {lbl}")

    Caption(fig,"T22  The complete angle search on five real pages, each normalised to its own maximum, with the "
                "selected angle ringed. The lower panels show one page at its selected angle and three degrees either side. "
                "A sharp peak means the objective is decisive on that page; a flat one is the sparse-page limitation stated in 3.4.6.")
    Stamp(fig,"segment.SkewScore / segment.Deskew")
    Save(fig,"T22_deskew_sweep")


#======================================================================
#T23  THE QUARTER-TURN DECISION AND ITS MARGIN
#one page is left on the wrong axis in the draft.  this shows why: the two
#scores are nearly tied, and the implementation refuses to turn on a tie
#======================================================================
def T23(items):
    names = [n for n,_,_ in items]
    s0 = np.array([a for _,a,_ in items])
    s90 = np.array([b for _,_,b in items])
    ratio = s90/np.maximum(s0,1e-9)

    fig,(axA,axB) = plt.subplots(1,2,figsize=(7.2,2.9),
                                 gridspec_kw=dict(width_ratios=[1.25,1],wspace=0.3))
    x = np.arange(len(names))
    axA.bar(x-0.2,s0,0.4,color=DARK,label="as photographed")
    axA.bar(x+0.2,s90,0.4,color=SILVER,label=r"rotated $90^\circ$")
    axA.set_xticks(x);axA.set_xticklabels(names,fontsize=FS(7),rotation=20,ha="right")
    axA.set_ylabel("nearly empty row fraction")
    axA.set_title("Line-axis evidence",fontsize=FS(8.5))
    axA.legend(fontsize=FS(7))

    axB.barh(x,ratio,0.5,color=[DARK if r>SEG.AXIS_TURN_MARGIN else SILVER for r in ratio])
    axB.axvline(SEG.AXIS_TURN_MARGIN,color=INK,lw=1.1,ls="--")
    axB.text(SEG.AXIS_TURN_MARGIN,len(names)-0.4,f"  margin {SEG.AXIS_TURN_MARGIN}",
             fontsize=FS(7),color=INK,va="top")
    axB.set_yticks(x);axB.set_yticklabels(names,fontsize=FS(7))
    axB.set_xlabel(r"$s_{90}/s_0$")
    axB.set_title("Turn accepted only above the margin",fontsize=FS(8.5))
    for i,r in enumerate(ratio):
        axB.text(r+0.02,i,f"{r:.3f}",fontsize=FS(6.8),va="center",color=INK)

    Caption(fig,"T23  The quarter-turn decision on real pages. Dark blue clears the 1.35 acceptance margin and would be "
                "turned; silver does not and is left as photographed. Leaving a page alone is the safe default, so a near tie "
                "resolves to no turn, which is the failure retained as capture C in 3.4.6.")
    Stamp(fig,"segment.LineAxisScore / segment.AXIS_TURN_MARGIN")
    Save(fig,"T23_quarter_turn_margin")


#======================================================================
#T24  ROW PROFILE, PITCH AND LINE CENTRES
#the three rules the draft states in prose, drawn on one real profile
#======================================================================
def T24(S,tag="Owen01"):
    sm,prof = S["sm"],S["prof"]
    H = len(sm)
    centres,P = S["centres"],S["P"]
    peaks = [i for i in range(1,H-1) if sm[i]>=sm[i-1] and sm[i]>sm[i+1] and sm[i]>0.06*sm.max()]

    rejected = [p for p in peaks if p not in centres]

    fig = plt.figure(figsize=(7.4,5.2))
    gs = fig.add_gridspec(1,3,width_ratios=[0.85,1.5,1.0],wspace=0.1)

    axP = fig.add_subplot(gs[0,0])
    ShowMask(axP,S["maskClean"],"Cleaned ink mask")
    for c in centres:
        axP.axhline(c,color=MID,lw=0.6)
    for c in rejected:
        axP.axhline(c,color=SILVER,lw=0.6,ls=":")

    axQ = fig.add_subplot(gs[0,1],sharey=axP)
    axQ.plot(prof,np.arange(H),color=SILVER,lw=0.5)
    axQ.plot(sm,np.arange(H),color=DARK,lw=0.9)
    axQ.axvline(0.06*sm.max(),color=INK,lw=0.8,ls=":")
    if rejected:
        axQ.plot(sm[rejected],rejected,"o",ms=3.4,mfc="none",mec=SILVER,mew=1.0)
    axQ.plot(sm[centres],centres,"o",ms=4.2,color=DARK,mec=INK,mew=0.4)
    axQ.invert_yaxis()
    axQ.set_xlabel("ink pixels per row")
    axQ.set_ylabel("row (half-resolution)")
    axQ.set_title(f"{len(peaks)} peaks above the floor,\n"
                  f"{len(centres)} accepted as line centres",fontsize=FS(8.5))
    axQ.grid(True,axis="y",alpha=0.25)

    from matplotlib.lines import Line2D
    axQ.legend(handles=[
        Line2D([],[],color=SILVER,lw=0.9,label="row ink count"),
        Line2D([],[],color=DARK,lw=1.2,label="smoothed profile"),
        Line2D([],[],color=INK,lw=0.9,ls=":",label="6% peak floor"),
        Line2D([],[],ls="none",marker="o",ms=4.2,color=DARK,mec=INK,label="accepted centre"),
        Line2D([],[],ls="none",marker="o",ms=4.2,mfc="none",mec=SILVER,label="peak rejected by $0.6P$"),
    ],fontsize=FS(6.5),loc="upper center",bbox_to_anchor=(0.5,-0.085),ncol=2)

    axR = fig.add_subplot(gs[0,2],sharey=axP)
    gaps = np.diff(centres)
    axR.plot(gaps,centres[:-1],"o-",color=DARK,ms=3.2,lw=0.8)
    axR.axvline(P,color=INK,lw=1.1,ls="--")
    axR.axvline(0.6*P,color=MID,lw=1.0,ls=":")
    axR.set_xlabel("rows between centres")
    axR.set_title(f"median pitch $P$={P}\n$0.6P$={0.6*P:.0f} minimum separation",fontsize=FS(8.5))
    axR.set_yticklabels([])
    axR.set_xlim(0,max(float(gaps.max())*1.12 if len(gaps) else P*1.5,P*1.3))
    axR.legend(handles=[
        Line2D([],[],color=DARK,lw=1.0,marker="o",ms=3.2,label="gap to next centre"),
        Line2D([],[],color=INK,lw=1.1,ls="--",label="median pitch $P$"),
        Line2D([],[],color=MID,lw=1.0,ls=":",label="$0.6P$ rule"),
    ],fontsize=FS(6.5),loc="upper center",bbox_to_anchor=(0.5,-0.085))

    Caption(fig,"T24  Why segmentation is driven by periodicity rather than by connected components. Every peak above the six "
                "per cent floor is a candidate; centres are then taken in order of strength subject to the 0.6P separation, and "
                "the median gap between the survivors is the pitch in which every later threshold is expressed. Gaps near twice "
                "the pitch are blank lines between paragraphs, which the median absorbs.")
    Stamp(fig,f"segment.LinePitch / segment.LineCentres | {tag}")
    Save(fig,"T24_profile_pitch_centres",f"P={P}, {len(centres)} centres")


#======================================================================
#T25  THE SEAM COST FIELD AND THE PATH THROUGH IT
#the central design argument of 3.4.3 and the one thing the section never
#shows.  the straight cut is drawn on the same gap for contrast
#======================================================================
#"only" cuts the figure down to the first selected gap for the Design
#section, where the point is the mechanism and one gap makes it; the full
#two-gap figure goes to the appendix.  both are drawn by this one function
#so the short version can never drift from the long one
def T25(S,tag="Owen01",only=False,stem="T25_seam_cost_field"):
    mask,centres,P = S["maskClean"],S["centres"],S["P"]
    H,W = mask.shape
    tailR = max(2,int(0.07*P))
    mdil = SEG.AnyWithinRadius(SEG.AnyWithinRadius(mask,tailR,0),tailR,1)
    mClosed = ~SEG.AnyWithinRadius(SEG.AnyWithinRadius(~mdil,tailR,0),tailR,1)
    m = mClosed.astype(np.float64)
    cols = np.arange(W)

    def Gap(c0,c1):
        yy = np.arange(c0,c1+1)[:,None]
        mid,halfGap = c0+0.5*(c1-c0),max(1.0,(c1-c0)/2.0)
        cost = 4.0*m[c0:c1+1]+0.012*((yy-mid)/halfGap)**2
        seam = c0+SEG.LeastCostSeam(cost,step=7)
        straight = int(round(mid))
        return cost,seam,straight,int(mask[np.clip(seam,0,H-1),cols].sum()),                int(mask[straight,cols].sum())

    pairs = list(zip(centres,centres[1:]))

    #the two gaps worth showing, chosen by stated rules rather than by eye:
    #the one where the seam avoids the most ink, and the one where it departs
    #furthest from a straight cut
    scored = []
    for c0,c1 in pairs:
        cost,seam,straight,kS,kR = Gap(c0,c1)
        scored.append((kR-kS,int(np.abs(seam-straight).max()),(c0,c1)))
    #a gap where the seam saves ink but runs almost straight shows nothing,
    #so the first panel is the gap that scores on both counts at once
    byInk = max(scored,key=lambda t:(t[0]*max(t[1],1),t[0]))
    byDev = max([x for x in scored if x[2]!=byInk[2]] or scored,
                key=lambda t:(t[1],t[0]))

    rows = [("(a)","the gap where the seam both avoids the most ink and bends furthest",byInk[2],True),
            ("(c)","the gap that departs furthest from a straight cut",byDev[2],False)]
    if only:
        rows = rows[:1]

    #a line gap is about ten times wider than it is tall, so drawn across one
    #column it is a few millimetres high.  each gap is therefore stacked as
    #two half-width bands, which doubles the scale of the ink, and the cost
    #field underneath keeps the whole width so the seam is seen end to end
    NB = 2
    panels = []
    for tagL,title,(c0,c1),annotate in rows:
        cost,seam,straight,kSeam,kStr = Gap(c0,c1)
        pad = int(0.50*P)
        top,bot = max(0,c0-pad),min(H,c1+pad)
        xmax = int(np.abs(seam-straight).argmax())
        for bi,(x0,x1) in enumerate(Bands(W,NB)):
            panels.append(("ink",tagL,title,c0,c1,kSeam,kStr,top,bot,x0,x1,
                           seam,straight,annotate and x0<=xmax<x1))
        panels.append(("cost",tagL,title,c0,c1,kSeam,kStr,top,bot,0,W,
                       seam,straight,False))

    #every panel is drawn at its true aspect, so the row heights come from
    #the data rather than from a guess
    LEFTF,RIGHTF = 0.115,0.985
    axW = WIDTH*(RIGHTF-LEFTF)
    hr = []
    for kind,_,_,c0,c1,_,_,top,bot,x0,x1,_,_,_ in panels:
        h = (bot-top) if kind=="ink" else (c1-c0)
        hr.append(max(0.42,axW*h/float(x1-x0)))

    #hr is already in inches, so the page is sized to hold the panels at
    #their true scale plus a fixed allowance for each two-line title, the
    #legend strip and the axis label; hspace is then derived, not guessed
    #the top strip holds the figure legend AND the two-line title of the
    #first panel, which is three lines of type at the sizes used here
    GAP,TOPSTRIP,BOTSTRIP = 0.62,0.95,0.46
    nGap = len(hr)-1
    figH = sum(hr)+nGap*GAP+TOPSTRIP+BOTSTRIP
    fig = plt.figure(figsize=(WIDTH,figH))
    gs = fig.add_gridspec(len(panels),1,height_ratios=hr,
                          hspace=GAP/(sum(hr)/len(hr)),
                          left=LEFTF,right=RIGHTF,
                          top=1.0-TOPSTRIP/figH,bottom=BOTSTRIP/figH)

    out = {}
    for pi,(kind,tagL,title,c0,c1,kSeam,kStr,top,bot,x0,x1,
            seam,straight,annotate) in enumerate(panels):
        ax = fig.add_subplot(gs[pi])
        if kind=="ink":
            ShowMask(ax,mask[top:bot,x0:x1],None)
            ax.plot(cols[x0:x1]-x0,seam[x0:x1]-top,color=DARK,lw=1.4)
            ax.axhline(straight-top,color=SILVER,lw=1.4,ls="--")
            ax.set_xlim(0,x1-x0);ax.set_ylim(bot-top,0)
            if x0==0:
                ax.set_title(f"{tagL}  {title}" "\n"
                             f"rows {c0}–{c1}: the seam crosses {kSeam} "
                             f"ink px, a straight cut crosses {kStr}",
                             fontsize=FS(8.0),pad=4,linespacing=1.35)
            ax.set_ylabel(f"columns\n{x0}–{x1}",fontsize=FS(7.2),rotation=0,
                          ha="right",va="center",labelpad=6)
            #nothing is drawn or written over the writing: the picture
            #carries the seam and the straight cut and nothing else, and
            #the clearance it wins is stated in the caption
            if pi==0:
                from matplotlib.lines import Line2D
                fig.legend(handles=[
                    Line2D([],[],color=DARK,lw=1.6,label="minimum-cost seam"),
                    Line2D([],[],color=SILVER,lw=1.6,ls="--",
                           label="straight cut at the midpoint")],
                    fontsize=FS(8),loc="upper center",
                    bbox_to_anchor=(0.55,0.999),ncol=2,frameon=False)
        else:
            cost,_,_,_,_ = Gap(c0,c1)
            im = ax.imshow(cost,cmap=CMAP,aspect="auto",interpolation="nearest",
                           extent=[0,W,c1,c0])
            ax.plot(cols,seam,color=INK,lw=1.2)
            ax.set_ylabel("page\nrow",fontsize=FS(7.2),rotation=0,ha="right",
                          va="center",labelpad=6)
            ax.grid(False)
            ax.tick_params(labelsize=FS(7.4))
            lab = "(b)" if tagL=="(a)" else "(d)"
            #the cost is stated once, under (b); (d) is the same field
            if lab=="(b)":
                ax.set_title(lab+"  the local cost in that gap, pale is cheap\n"
                             #\tilde, not \widetilde: the wide accent is an
                             #extensible glyph from a Computer Modern face,
                             #and it would be the one piece of this figure
                             #set that is not Times.  the symbol means the
                             #same closed mask as Equation 3.4
                             r"$C(y,x)=4\tilde{M}+0.012\,((y-y_m)/h_g)^2$",
                             fontsize=FSM(),pad=4,linespacing=1.4)
            else:
                ax.set_title(lab+"  the same cost field for the second gap",
                             fontsize=FS(8.3),pad=4)
            if pi==len(panels)-1:
                ax.set_xlabel("column of the half-resolution working image",
                              fontsize=FS(7.8),labelpad=2)
            else:
                #only the last cost panel is given a column scale; a second
                #copy of it only pushed the next panel's title into the ink
                ax.tick_params(axis="x",labelbottom=False,length=0)
            cb = fig.colorbar(im,ax=ax,pad=0.008,fraction=0.055,aspect=7)
            cb.ax.tick_params(labelsize=FS(7))
            cb.outline.set_linewidth(0.5)
            out[f"gap_{c0}_{c1}"] = dict(inkSeam=kSeam,inkStraight=kStr,
                                         maxDeviationPx=int(np.abs(seam-straight).max()))

    if only:
        Caption(fig,"The boundary between two text lines, on the gap of this page where the seam avoids the most ink. The line "
                    "is stacked in two half-width bands so the writing can be seen; (b) is the local cost field for the same gap, "
                    "over the full width. Ink is penalised four times more heavily than distance from the midline, so the "
                    "boundary bends around a descender rather than cutting it. Mechanism illustration of Equations 3.4 and 3.5. "
                    "The second selected gap and the complete figure are in the appendix.")
    else:
        Caption(fig,"T25  The boundary between two text lines, on the two gaps of this page selected by stated rules: (a,b) the gap "
                    "where the seam avoids the most ink, and (c,d) the gap where it departs furthest from a straight cut. Ink is "
                    "penalised four times more heavily than distance from the midline, so the boundary bends around a descender "
                    "rather than cutting it. Mechanism illustration of Equations 3.4 and 3.5.")
    Stamp(fig,f"segment.SeamCuts / segment.LeastCostSeam | {tag}")
    Save(fig,stem,str(out))
    return out


#======================================================================
#T26  SEAM WEIGHT SENSITIVITY
#the midpoint term is weak on purpose.  removing it moves the cut by an
#order of magnitude, which is what justifies keeping it at 0.012
#======================================================================
def T26(S,tag="Owen01"):
    mask,centres,P = S["maskClean"],S["centres"],S["P"]
    H,W = mask.shape
    tailR = max(2,int(0.07*P))
    mdil = SEG.AnyWithinRadius(SEG.AnyWithinRadius(mask,tailR,0),tailR,1)
    mClosed = ~SEG.AnyWithinRadius(SEG.AnyWithinRadius(~mdil,tailR,0),tailR,1)
    m = mClosed.astype(np.float64)

    weights = [0.0,0.003,0.012,0.05,0.2,1.0]
    gaps = list(zip(centres,centres[1:]))[:6]
    offs,inks = [],[]
    for wgt in weights:
        o,k = [],[]
        for c0,c1 in gaps:
            yy = np.arange(c0,c1+1)[:,None]
            mid,halfGap = c0+0.5*(c1-c0),max(1.0,(c1-c0)/2.0)
            cost = 4.0*m[c0:c1+1]+wgt*((yy-mid)/halfGap)**2
            seam = c0+SEG.LeastCostSeam(cost,step=7)
            o.append(float(np.abs(seam-mid).mean()))
            k.append(int(mask[np.clip(seam,0,H-1),np.arange(W)].sum()))
        offs.append(float(np.mean(o)));inks.append(float(np.mean(k)))

    fig,(axA,axB) = plt.subplots(1,2,figsize=(7.2,2.8),
                                 gridspec_kw=dict(wspace=0.32))
    sel = weights.index(0.012)
    cols = [DARK if i==sel else SILVER for i in range(len(weights))]
    axA.bar(range(len(weights)),offs,0.62,color=cols)
    axA.set_xticks(range(len(weights)))
    axA.set_xticklabels([f"{w:g}" for w in weights],fontsize=FS(7))
    axA.set_xlabel("midpoint weight");axA.set_ylabel("mean |offset| from midline (px)")
    axA.set_title("Where the seam sits",fontsize=FS(8.5))
    for i,v in enumerate(offs):
        axA.text(i,v,f"{v:.2f}",ha="center",va="bottom",fontsize=FS(6.6))

    axB.bar(range(len(weights)),inks,0.62,color=cols)
    axB.set_xticks(range(len(weights)))
    axB.set_xticklabels([f"{w:g}" for w in weights],fontsize=FS(7))
    axB.set_xlabel("midpoint weight");axB.set_ylabel("mean raw ink pixels crossed")
    axB.set_title("What it costs in severed ink",fontsize=FS(8.5))
    for i,v in enumerate(inks):
        axB.text(i,v,f"{v:.0f}",ha="center",va="bottom",fontsize=FS(6.6))

    Caption(fig,f"T26  The midpoint attraction swept over {len(gaps)} real gaps on one page; the implemented 0.012 is dark blue. "
                "Removing it lets the seam wander far from the midline on low-ink alternatives, while raising it forces the cut "
                "straight through ink. The term is deliberately weak, not absent.")
    Stamp(fig,f"segment.SeamCuts, midpoint weight swept | {tag}")
    Save(fig,"T26_seam_weight_sensitivity")
    return dict(weights=weights,offsets=offs,inkCrossed=inks)


#======================================================================
#T27  THE OWNERSHIP CORRECTIONS, BEFORE AND AFTER
#the band guess against the corrected map, on the places they disagree
#======================================================================
def T27(S,tag="Owen01"):
    band,own,mask = S["bandOnly"],S["own"],S["maskClean"]
    P = S["P"]
    H,W = mask.shape
    diff = (band!=own)&(mask)
    ys,xs = np.nonzero(diff)
    if len(ys)==0:
        print("  T27 skipped: the corrections changed nothing on this page")
        return None

    #cluster the disagreements into neighbourhoods and show the biggest few
    order = np.argsort(ys)
    ys,xs = ys[order],xs[order]
    groups,cur = [],[0]
    for i in range(1,len(ys)):
        if abs(int(ys[i])-int(ys[cur[-1]]))<=P and abs(int(xs[i])-int(xs[cur[-1]]))<=2*P:
            cur.append(i)
        else:
            groups.append(cur);cur=[i]
    groups.append(cur)

    #the biggest cluster is not the clearest one: a large diffuse patch
    #changes a few pixels in many places and reads as no change at all.
    #each cluster is scored by how much of the ink in its own neighbourhood
    #moved, so the cases shown are whole pieces that changed hands
    scoredG = []
    for gp in groups:
        gy,gx = ys[gp],xs[gp]
        wy0,wy1 = max(0,int(gy.min())-2),min(H,int(gy.max())+3)
        wx0,wx1 = max(0,int(gx.min())-2),min(W,int(gx.max())+3)
        localInk = max(1,int(mask[wy0:wy1,wx0:wx1].sum()))
        whole = len(gp)/float(localInk)
        scoredG.append((len(gp)*whole,len(gp),gp))
    scoredG.sort(key=lambda t:(-t[0],-t[1]))
    groups = [t[2] for t in scoredG[:3]]

    #each case is cropped to a square-ish window so the three sit side by
    #side at the same scale and fill the column width
    n = len(groups)
    pad = int(1.1*P)
    wins = []
    for gp in groups:
        gy,gx = ys[gp],xs[gp]
        y0,y1 = max(0,gy.min()-pad),min(H,gy.max()+pad)
        x0,x1 = max(0,gx.min()-pad),min(W,gx.max()+pad)
        wins.append([y0,y1,x0,x1])

    #the three windows are grown to one common shape so the panels are the
    #same size and the same scale, and the titles sit on one line
    ar = max((y1-y0)/float(x1-x0) for y0,y1,x0,x1 in wins)
    for k,(y0,y1,x0,x1) in enumerate(wins):
        need = int(round((y1-y0)/ar))
        if need>x1-x0:
            grow = need-(x1-x0)
            x0 = max(0,x0-grow//2);x1 = min(W,x0+need);x0 = max(0,x1-need)
        need = int(round(ar*(x1-x0)))
        if need>y1-y0:
            grow = need-(y1-y0)
            y0 = max(0,y0-grow//2);y1 = min(H,y0+need);y0 = max(0,y1-need)
        wins[k] = [y0,y1,x0,x1]
    panelW = (WIDTH-1.05)/n
    panelH = max(panelW*(y1-y0)/float(x1-x0) for y0,y1,x0,x1 in wins)
    fig,axes = plt.subplots(2,n,figsize=(WIDTH,2*panelH+1.05),squeeze=False,
                            gridspec_kw=dict(hspace=0.10,wspace=0.06,
                                             left=0.135,right=0.995,
                                             top=0.885,bottom=0.015))
    for j,gp in enumerate(groups):
        gy,gx = ys[gp],xs[gp]
        y0,y1,x0,x1 = wins[j]
        #the box around what actually changed, drawn identically in both rows
        by0,by1 = int(gy.min())-y0,int(gy.max())-y0
        bx0,bx1 = int(gx.min())-x0,int(gx.max())-x0
        fromL = int(np.bincount(band[gy,gx]+1).argmax())-1
        toL = int(np.bincount(own[gy,gx]+1).argmax())-1
        for row,(omap,lbl) in enumerate([(band,"seam-band guess"),(own,"after correction")]):
            ax = axes[row][j]
            view = omap[y0:y1,x0:x1]
            ink = mask[y0:y1,x0:x1]
            #one blue per line index, keyed to the index itself so the same
            #line keeps the same shade in both rows
            img = np.full(view.shape+(3,),1.0)
            for li in np.unique(view[ink]):
                img[ink&(view==li)] = np.array(P_hex(SEQ[3+(int(li)%3)]))/255.0
            #ink with no owner reads as silver, not as a line
            img[ink&(view<0)] = np.array(P_hex(SILVER))/255.0
            ax.imshow(img,interpolation="nearest",aspect="equal")
            ax.add_patch(plt.Rectangle((bx0-3,by0-3),bx1-bx0+6,by1-by0+6,
                                       fill=False,ec=INK,lw=1.1,ls="--"))
            ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
            for sp in ax.spines.values():
                sp.set_color(SILVER);sp.set_linewidth(0.6)
            if row==0:
                ax.set_title(f"case {j+1}:  {len(gp)} px,  line {fromL}"
                             f" → {toL}",fontsize=FS(8.2),pad=4)
            if j==0:
                ax.set_ylabel(lbl,fontsize=FS(8.2))

    fig.suptitle("Connected-component ownership correcting the seam-band assignment",
                 fontsize=FS(9),y=0.985)
    Caption(fig,"T27  The three largest places on one real page where the connected-component rules overrule the seam band. Each "
                "shade of blue is one line index and the dashed box marks the ink that moved, so the transfer is visible as a change "
                "of shade inside the box between the two rows. These are the detached-dot, enclosed-ring and residual-fragment rules "
                "of Algorithm 3.2 acting on real ink; without them an ascender or a descender is read as belonging to its neighbour.")
    Stamp(fig,f"segment._BandOwnership vs segment.AssignOwnership | {tag}")
    Save(fig,"T27_ownership_corrections",f"{len(groups)} cases, {int(diff.sum())} px moved")
    return dict(pixelsMoved=int(diff.sum()),cases=len(groups))


def P_hex(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2],16) for i in (0,2,4))


#======================================================================
#T29  THE FIXED CANVAS FIT
#what the 64 by 640 canvas costs a long line, which is the limitation
#3.4.5 states and never shows
#======================================================================
def T29(crops,tag=""):
    fig,axes = plt.subplots(len(crops),2,figsize=(7.2,1.25*len(crops)+0.6),
                            squeeze=False,
                            gridspec_kw=dict(width_ratios=[1.6,1],wspace=0.1,hspace=0.55))
    rows = []
    for i,(name,arr) in enumerate(crops):
        h,w = arr.shape
        a = min(64/h,640/w)
        fit = SEG.FitFixedCanvas(arr,64,640)
        ShowGray(axes[i][0],arr,None)
        axes[i][0].set_ylabel(name,fontsize=FS(7.5),rotation=0,ha="right",va="center",labelpad=14)
        axes[i][0].set_title(f"source {h}x{w} px",fontsize=FS(7.5),pad=2)
        ShowGray(axes[i][1],fit,f"canvas 64x640, scale a={a:.4f}  ->  {int(round(a*h))}x{min(640,int(round(a*w)))}",)
        rows.append((name,h,w,a,int(round(a*h))))
    fig.suptitle("Aspect-preserving fit onto the network canvas",fontsize=FS(9),y=0.995)
    Caption(fig,"T29  The shared scale a=min(64/h, 640/w) applied to real line crops of very different proportions. A long line "
                "is limited by width, so it occupies only part of the 64 available rows and loses vertical detail before the "
                "network sees it. Illustration of Equation 3.6 and the limitation stated with it.")
    Stamp(fig,"segment.FitFixedCanvas")
    Save(fig,"T29_fixed_canvas_fit")
    return rows


if __name__=="__main__":
    import json
    print("T21-T29  segmentation visual tests")
    res = {}

    #five real photographed pages, one per personal writer.  the geometry
    #figures run at the page's own resolution, which is what the segmenter
    #sees in service; T21 alone runs on the 1600-reduced page because that is
    #the image the Design draft's worked Otsu trace is computed on
    want = [("Owen",1),("Dylan",1),("Yeukita",1),("Abhinav",1),("Thiya",1)]
    pages = []
    for w,i in want:
        img,lab = PersonalPage(w,i)
        if not img.exists():
            print(f"  missing {img}");continue
        print(f"  conditioning {img.name} ...")
        pages.append((f"{w}{i:02d}",Stages(img)))

    imgO,_ = PersonalPage("Owen",1)
    print("  conditioning Owen01.jpg at the draft's 1600 reduction ...")
    res["T21"] = T21(Stages(imgO,PAGE_MAXSIDE),"Owen01, reduced to 1600 px")

    T22(pages)

    axis = []
    for name,St in pages:
        a = St["full"]
        axis.append((name,SEG.LineAxisScore(a),SEG.LineAxisScore(np.rot90(a))))
    T23(axis)
    res["T23"] = [dict(page=n,s0=a,s90=b,ratio=b/max(a,1e-9)) for n,a,b in axis]

    #each remaining figure runs on the page that exercises its rule hardest,
    #chosen by a stated score rather than by eye
    def Rejected(St):
        sm2 = St["sm"];H2 = len(sm2)
        pk = [i for i in range(1,H2-1) if sm2[i]>=sm2[i-1] and sm2[i]>sm2[i+1]
              and sm2[i]>0.06*sm2.max()]
        return len(pk)-len(St["centres"])
    pick = max(pages,key=lambda nv:Rejected(nv[1]))
    print(f"  T24 on {pick[0]}: {Rejected(pick[1])} peaks rejected by the 0.6P rule")
    T24(pick[1],pick[0])

    def SeamGain(St):
        mask,centres,P = St["maskClean"],St["centres"],St["P"]
        H,W = mask.shape
        tailR = max(2,int(0.07*P))
        md = SEG.AnyWithinRadius(SEG.AnyWithinRadius(mask,tailR,0),tailR,1)
        mc = ~SEG.AnyWithinRadius(SEG.AnyWithinRadius(~md,tailR,0),tailR,1)
        m = mc.astype(np.float64);cols=np.arange(W);best=0
        for c0,c1 in zip(centres,centres[1:]):
            yy=np.arange(c0,c1+1)[:,None]
            mid,hg=c0+0.5*(c1-c0),max(1.0,(c1-c0)/2.0)
            seam=c0+SEG.LeastCostSeam(4.0*m[c0:c1+1]+0.012*((yy-mid)/hg)**2,step=7)
            best=max(best,int(mask[int(round(mid)),cols].sum())
                          -int(mask[np.clip(seam,0,H-1),cols].sum()))
        return best
    pickS = max(pages,key=lambda nv:SeamGain(nv[1]))
    print(f"  T25/T26 on {pickS[0]}: best seam gain {SeamGain(pickS[1])} ink px")
    res["T25"] = T25(pickS[1],pickS[0])
    #the Design section carries the first selected gap only; the appendix
    #carries both.  both come from the same call path
    T25(pickS[1],pickS[0],only=True,stem="D25_seam_design")
    res["T26"] = T26(pickS[1],pickS[0])

    pickO = max(pages,key=lambda nv:int(((nv[1]["bandOnly"]!=nv[1]["own"])
                                         &nv[1]["maskClean"]).sum()))
    moved = int(((pickO[1]["bandOnly"]!=pickO[1]["own"])&pickO[1]["maskClean"]).sum())
    print(f"  T27 on {pickO[0]}: {moved} px reassigned by the ownership rules")
    res["T27"] = T27(pickO[1],pickO[0])

    #real crops of contrasting proportion for the canvas figure
    crops = []
    for nm in ["owen_1.png","dylan_1.png","yeukita_1.png","abhinav_1.png"]:
        q = CROPS/nm
        if q.exists():
            crops.append((Author(nm.split("_")[0]),Gray(q)))
    if crops:
        res["T29"] = T29(crops)

    (DATA/"seg_results.json").write_text(json.dumps(res,indent=1,default=float),
                                         encoding="utf-8")
    print("done")
