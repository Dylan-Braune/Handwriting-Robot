#u23539195 Dylan Braune.  visual tests for Section 3.6, style extraction.
#
#these run the real extractor on real handwriting with its known transcript,
#and read the final stored profiles.  no accuracy is claimed.
#
#  python vt_style.py
from vt_common import *
import numpy as np
import json
import segment as SEG
import recognise as REC
import style as ST


def Profile(writer):
    p = PROFILES/f"{writer}_curated.json"
    if not p.exists():
        p = PROFILES/f"{writer}.json"
    return json.loads(p.read_text(encoding="utf-8")),p.name


#---DrawGlyph: one stored glyph's strokes, in its normalised frame---
def DrawGlyph(ax,strokes,color=DARK,lw=1.2,alpha=1.0,zorder=2):
    for s in strokes:
        a = np.asarray(s,dtype=float)
        if len(a)>1:
            ax.plot(a[:,0],a[:,1],color=color,lw=lw,alpha=alpha,
                    solid_capstyle="round",zorder=zorder)


def GlyphFrame(ax,title=None):
    ax.set_aspect("equal");ax.grid(False)
    ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(SILVER);s.set_linewidth(0.5)
    if title:
        ax.set_title(title,fontsize=FS(7.5),pad=2)


#======================================================================
#T41  SKELETON, TRACED PATHS, SIMPLIFIED PATHS
#where a pen path comes from, and where the pen ORDER is invented
#======================================================================
def T41(gray,tag="",window=None):
    ink = SEG.BinarizeLine(gray)
    skel = SEG.Skeletonize(ink)
    polys = ST.TracePolylines(skel)
    simp = [ST.SimplifyPolyline(p,eps=0.6) for p in polys]

    #the blank rows a crop carries are dropped first, and the window is cut
    #to an aspect that leaves the panel a centimetre high on the page, so
    #the thinning and the path breaks can actually be seen
    y0,y1 = InkRows(gray,pad=0.10)
    ASPECT = 5.2
    if window is None:
        left,right = Window(gray[y0:y1],aspect=ASPECT)
    else:
        left,right = window

    nV = sum(len(p) for p in polys)
    nS = sum(len(p) for p in simp)

    panelH = (WIDTH-0.08)/ASPECT
    fig,axes = plt.subplots(4,1,figsize=(WIDTH,4*panelH+1.35),
                            gridspec_kw=dict(hspace=0.32,left=0.008,
                                             right=0.992,top=0.935,
                                             bottom=0.012))
    ShowGray(axes[0],gray[y0:y1,left:right],"(a)  grey crop",aspect="auto")
    ShowMask(axes[1],ink[y0:y1,left:right],"(b)  binarised ink, an area",
             aspect="auto")
    ShowMask(axes[2],skel[y0:y1,left:right],
             "(c)  Zhang–Suen centreline, one pixel wide",aspect="auto")

    ax = axes[3]
    ax.imshow(np.full_like(gray[y0:y1,left:right],255),cmap="gray",vmin=0,
              vmax=255,aspect="auto",interpolation="nearest")
    #each traced path in its own shade, so a pen lift is visible as a break
    for i,p in enumerate(simp):
        a = np.asarray(p,dtype=float)
        sel = (a[:,0]>=left)&(a[:,0]<right)
        if sel.sum()<2:continue
        ax.plot(a[sel,0]-left,a[sel,1]-y0,lw=1.5,
                color=[DARK,MID,SEQ[3],INK][i%4],solid_capstyle="round")
        ax.plot(a[sel,0][0]-left,a[sel,1][0]-y0,"o",ms=3.4,color=INK,zorder=4)
    ax.set_xlim(0,right-left);ax.set_ylim(y1-y0,0)
    ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
    for s in ax.spines.values():
        s.set_color(SILVER);s.set_linewidth(0.6)
    ax.set_title(f"(d)  {len(polys)} traced paths over the whole line, "
                 f"simplified from {nV} to {nS} vertices at "
                 "$\\epsilon$=0.6 px; a dot marks each path start",
                 fontsize=FS(8.3),pad=3)

    Caption(fig,"T41  How an area of ink becomes a pen path. Thinning gives a one-pixel centreline, the centreline is traced as a "
                "graph split at junctions with short spurs pruned and near-collinear ends rejoined, and the result is simplified "
                "within a bounded deviation. The shading separates the traced paths: where one letter is drawn in two shades the "
                "tracer has chosen a pen lift the writer may never have made, because a static image cannot record stroke order.")
    Stamp(fig,f"segment.Skeletonize / style.TracePolylines / SimplifyPolyline | {tag}")
    Save(fig,"T41_skeleton_trace_simplify",f"{len(polys)} paths, {nV}->{nS} vertices")
    return dict(paths=len(polys),verticesTraced=int(nV),verticesSimplified=int(nS))


#======================================================================
#T42  THE CUT-COST REFINEMENT
#the recogniser gives an approximate boundary; this is what moves it
#======================================================================
def T42(gray,text,model,tag=""):
    rawW = gray.shape[1]
    t = REC.TensorFromResized(REC.ResizeLineImageFixed(gray))
    lp = model.forward(t)[:,0,:]
    res = REC.CtcForcedAlign(lp,text)
    align,conf = res
    T = lp.shape[0]

    ink = SEG.BinarizeLine(gray)
    band = SEG.CoreBand(ink)
    top,base = band
    xh = float(base-top)

    #the implemented column cost, recomputed with the same constants
    colProf = ink.sum(axis=0).astype(np.float64)
    k = max(3,int(round(xh*0.12)))|1
    colSm = np.convolve(colProf,np.ones(k)/k,mode="same")
    rowsIdx = np.arange(ink.shape[0])[:,None]
    hiY = np.where(ink,rowsIdx,ink.shape[0]).min(axis=0).astype(np.float64)
    topAbove = np.clip((base-hiY)/max(1.0,xh),0.0,3.0)
    ref = float(np.percentile(colSm[colSm>0],60)) if (colSm>0).any() else 1.0
    thin = colSm/max(1e-6,ref)
    cutCost = thin+1.15*topAbove
    cutCost[colProf==0] = 0.0

    glyphs,stats = ST.ExtractLineGlyphs(gray,text,model)
    cuts = stats["cuts"]

    #initial boundaries straight from the alignment, before snapping
    init = []
    for ch,sp in align:
        if sp is None:continue
        init.append(0.5*(REC.FrameXToPixel(sp[0]/T*REC.INPUT_WIDTH,rawW,gray.shape[0])
                        +REC.FrameXToPixel(sp[1]/T*REC.INPUT_WIDTH,rawW,gray.shape[0])))
    mids = [0.5*(init[i]+init[i+1]) for i in range(len(init)-1)]
    final = [c[2] for c in cuts][1:]

    #the window with the largest refinement, chosen by a stated rule
    n = min(len(mids),len(final))
    if n:
        j = int(np.argmax([abs(final[i]-mids[i]) for i in range(n)]))
        centre = final[j]
    else:
        j,centre = 0,rawW/2
    half = max(int(6*xh),120)
    x0,x1 = int(max(0,centre-half)),int(min(rawW,centre+half))

    fig = plt.figure(figsize=(7.2,4.3))
    gs = fig.add_gridspec(3,1,height_ratios=[1.1,1.1,1.0],hspace=0.4)

    axA = fig.add_subplot(gs[0])
    ShowGray(axA,gray,None)
    for m in mids: axA.axvline(m,color=SILVER,lw=0.6)
    for f in final: axA.axvline(f,color=DARK,lw=0.6)
    axA.axvspan(x0,x1,color=MID,alpha=0.15,lw=0)
    axA.set_title("(a)  whole line: alignment midpoints in silver, refined cuts in blue",
                  fontsize=FS(8.3))

    axB = fig.add_subplot(gs[1])
    ShowGray(axB,gray[:,x0:x1],None)
    for m in mids:
        if x0<=m<x1: axB.axvline(m-x0,color=SILVER,lw=1.3,ls="--")
    for f in final:
        if x0<=f<x1: axB.axvline(f-x0,color=DARK,lw=1.3)
    axB.axhline(base,color=INK,lw=0.6,ls=":")
    axB.set_title(f"(b)  the largest refinement on this line: "
                  f"{abs(final[j]-mids[j]):.0f} px",fontsize=FS(8.3))

    axC = fig.add_subplot(gs[2])
    xx = np.arange(x0,x1)
    axC.fill_between(xx,thin[x0:x1],color=PALE,lw=0,label="ink term  $\\bar p_x/q_{0.60}$")
    axC.plot(xx,1.15*topAbove[x0:x1],color=SILVER,lw=1.0,
             label="height term  $1.15\\,(b-y_{top})/h_x$")
    axC.plot(xx,cutCost[x0:x1],color=DARK,lw=1.3,label="total cut cost $C_x$")
    for m in mids:
        if x0<=m<x1: axC.axvline(m,color=SILVER,lw=1.0,ls="--")
    for f in final:
        if x0<=f<x1: axC.axvline(f,color=DARK,lw=1.0)
    axC.set_xlim(x0,x1)
    axC.set_xlabel("column in the source crop",fontsize=FS(7.5))
    axC.set_ylabel("cost",fontsize=FS(7.5))
    axC.legend(fontsize=FS(6.4),ncol=3,loc="upper center",bbox_to_anchor=(0.5,1.30))

    Caption(fig,"T42  Refining a character boundary. The recogniser locates a character to a frame span, whose midpoint is only "
                "approximately where the letters separate. The cost prefers a column that is thin and low, which in cursive is "
                "the ligature crossing, and a genuinely blank column is given zero cost outright. One shared increasing boundary "
                "array is enforced so two neighbouring cuts cannot land on the same gap and leave an empty cell.")
    Stamp(fig,f"style.ExtractLineGlyphs cut cost | {tag}")
    Save(fig,"T42_cut_cost_refinement",f"max shift {abs(final[j]-mids[j]):.0f}px")
    return dict(alignConfidence=float(conf),maxShiftPx=float(abs(final[j]-mids[j])),
                characters=len(final)+1)


#======================================================================
#T43  THE STORED GLYPH COORDINATE FRAME
#image pixels on the left, the normalised frame that is stored on the right
#======================================================================
def T43(gray,text,model,want="V",tag=""):
    ink = SEG.BinarizeLine(gray)
    band = SEG.CoreBand(ink)
    top,base = band
    xh = float(base-top)
    slant = SEG.EstimateSlantDeg(ink)
    glyphs,stats = ST.ExtractLineGlyphs(gray,text,model)
    cuts = stats["cuts"]

    pick = None
    for (i,ch,left,right) in cuts:
        if ch==want and glyphs[i]:
            pick = (i,ch,left,right);break
    if pick is None:
        for (i,ch,left,right) in cuts:
            if glyphs[i] and ch.isalpha():
                pick = (i,ch,left,right);break
    i,ch,left,right = pick
    g = glyphs[i]

    fig = plt.figure(figsize=(WIDTH,3.35))
    gs = fig.add_gridspec(1,3,width_ratios=[1.35,0.95,1.35],wspace=0.20,
                          left=0.012,right=0.995,top=0.845,bottom=0.025)

    axA = fig.add_subplot(gs[0,0])
    pad = int(0.6*xh)
    a,b = int(max(0,left-pad)),int(min(gray.shape[1],right+pad))
    #the cell is cropped to the band the letter and its descender occupy,
    #instead of carrying the whole crop height as blank paper
    r0 = int(max(0,top-1.30*xh));r1 = int(min(gray.shape[0],base+0.45*xh))
    ShowGray(axA,gray[r0:r1,a:b],None)
    axA.axvline(left-a,color=DARK,lw=1.4)
    axA.axvline(right-a,color=DARK,lw=1.4)
    axA.axhline(base-r0,color=INK,lw=1.1,ls="--")
    axA.axhline(top-r0,color=SILVER,lw=1.1,ls="--")
    axA.text(0.99,1.0-(base-r0)/float(r1-r0),"baseline",transform=axA.transAxes,
             ha="right",va="bottom",fontsize=FS(7.4),color=INK,
             bbox=dict(fc=PAPER,ec="none",pad=0.8))
    axA.text(0.99,1.0-(top-r0)/float(r1-r0),"x-height",transform=axA.transAxes,
             ha="right",va="bottom",fontsize=FS(7.4),color="#5A6570",
             bbox=dict(fc=PAPER,ec="none",pad=0.8))
    axA.set_title(f"(a)  the cell for '{ch}' in image pixels\n"
                  f"baseline $b$={base}, $h_x$={xh:.0f} px, "
                  f"slant {slant:+.2f}$^\\circ$",fontsize=FSM())

    axB = fig.add_subplot(gs[0,1])
    DrawGlyph(axB,g["strokes"])
    axB.axhline(0,color=INK,lw=0.8,ls="--")
    axB.axhline(1,color=SILVER,lw=0.8,ls="--")
    axB.axvline(0,color=SILVER,lw=0.8,ls=":")
    GlyphFrame(axB,"(b)  stored, de-slanted\n$v$ in x-heights, baseline at 0")

    axC = fig.add_subplot(gs[0,2])
    axC.axis("off")
    axC.text(0.0,1.0,
        "$v=(b-y)/h_x$\n"
        "$u_0=(x-x_L)/h_x-v\\tan\\phi$\n"
        "$u=u_0-\\delta$\n\n"
        f"$x_L$ = {left:.1f} px\n"
        f"$b$ = {base} px,  $h_x$ = {xh:.0f} px\n"
        f"$\\tan\\phi$ = {np.tan(np.radians(slant)):+.7f}\n\n"
        f"stored width = {g['width']:.3f} $h_x$\n"
        f"advance = {g['advance']:.3f} $h_x$\n"
        f"lead $\\delta$ = {g['lead']:.3f} $h_x$\n"
        f"top = {g['top']:.3f},  bottom = {g['bot']:.3f}\n"
        f"entry = ({g['entryX']:.3f}, {g['entryY']:.3f})\n"
        f"exit  = ({g['exitX']:.3f}, {g['exitY']:.3f})\n"
        f"joined left/right = {g['connL']}/{g['connR']}",
        transform=axC.transAxes,va="top",ha="left",fontsize=FSM(),
        linespacing=1.55)

    Caption(fig,"T43  The transform that makes a glyph reusable. Image rows increase downwards and a letter's size depends on the "
                "photograph; the stored frame measures height upwards from the baseline in x-heights and removes the writer's "
                "slant, which is reapplied at placement. The lead is stored separately from the path, so the writer's own gap "
                "between the cell edge and the first ink survives instead of every letter being butted to its cell. "
                "Illustration of Equation 3.16.")
    Stamp(fig,f"style.ExtractLineGlyphs | {tag}")
    Save(fig,"T43_glyph_coordinates",f"'{ch}'")
    return dict(char=ch,xLeft=float(left),baseline=int(base),xHeight=float(xh),
                slantDeg=float(slant),stored={k:g[k] for k in
                ("width","advance","lead","top","bot","connL","connR")})


#======================================================================
#T45  THE VARIANT POOL, THE MEDOID AND THE PROTOTYPE
#how a library keeps handwriting variation without keeping bad cuts
#======================================================================
def T45(writers,char="a"):
    #the four column headings are stated once at the top instead of being
    #repeated on every writer, which frees the height for larger glyphs
    nR = len(writers)
    fig = plt.figure(figsize=(WIDTH,1.72*nR+0.78))
    gs = fig.add_gridspec(nR,4,width_ratios=[1.45,1,1,1.35],
                          hspace=0.16,wspace=0.26,
                          left=0.085,right=0.985,top=0.885,bottom=0.075)
    HEADS = (f"every stored '{char}' variant,\nthe medoid in blue",
             "the stored medoid\n(a real example)",
             "the denoised prototype\n(a summary)",
             "within-writer consensus:\ntotal $L_1$ distance")
    out = {}
    for r,w in enumerate(writers):
        prof,fn = Profile(w)
        pool = prof["glyphs"].get(char,[])
        if not pool:
            continue
        medoid = (prof.get("idealMedoid") or {}).get(char)
        proto = (prof.get("idealGlyphs") or {}).get(char)

        #the pairwise L1 descriptor rule the draft describes, recomputed on
        #the final pool so the selection is shown rather than asserted
        descs = []
        keep = []
        for g in pool:
            d = ST._ShapeGrid(g)
            if d is None:continue
            d = np.asarray(d,dtype=float)
            descs.append(d/max(np.linalg.norm(d),1e-9));keep.append(g)
        pool = keep
        if not pool:continue
        D = np.array([[np.abs(a-b).sum() for b in descs] for a in descs])
        tot = D.sum(1)
        pickIdx = int(np.argmin(tot))

        ax = fig.add_subplot(gs[r,0])
        for g in pool:
            DrawGlyph(ax,g["strokes"],color=SILVER,lw=1.0,alpha=0.85,zorder=1)
        DrawGlyph(ax,pool[pickIdx]["strokes"],color=DARK,lw=1.9,zorder=3)
        ax.axhline(0,color=INK,lw=0.6,ls="--",zorder=0)
        GlyphFrame(ax,HEADS[0] if r==0 else None)
        ax.set_ylabel(w.capitalize(),fontsize=FS(8.5),rotation=0,ha="right",
                      va="center",labelpad=8)

        ax = fig.add_subplot(gs[r,1])
        if medoid: DrawGlyph(ax,medoid["strokes"] if isinstance(medoid,dict) else medoid,lw=1.6)
        GlyphFrame(ax,HEADS[1] if r==0 else None)

        ax = fig.add_subplot(gs[r,2])
        if proto: DrawGlyph(ax,proto["strokes"] if isinstance(proto,dict) else proto,lw=1.6)
        GlyphFrame(ax,HEADS[2] if r==0 else None)

        ax = fig.add_subplot(gs[r,3])
        order = np.argsort(tot)
        ax.barh(np.arange(len(tot)),tot[order],0.72,
                color=[DARK if order[i]==pickIdx else SILVER
                       for i in range(len(tot))])
        ax.set_yticks(np.arange(len(tot)))
        ax.set_yticklabels([f"{order[i]}" for i in range(len(tot))],
                           fontsize=FS(7.2))
        ax.invert_yaxis()
        ax.tick_params(axis="y",length=2,pad=1.5,labelsize=FS(7.2))
        ax.tick_params(axis="x",labelsize=FS(8))
        ax.set_ylabel("variant",fontsize=FS(7.6),labelpad=1)
        if r==nR-1:
            ax.set_xlabel("total $L_1$ distance",fontsize=FSM(),labelpad=1)
        if r==0:
            ax.set_title(HEADS[3],fontsize=FSM(),pad=4)
        for s in ax.spines.values():
            s.set_color(SILVER)

        out[w] = dict(file=fn,variants=len(pool),medoidIndex=pickIdx,
                      totals=[round(float(t),4) for t in tot],
                      priorD=[g.get("priorD") for g in pool])

    Caption(fig,"T45  Keeping variation without keeping extraction errors. Every stored instance of one letter is overlaid in "
                "silver; the variant with the smallest total descriptor distance to the others is the medoid, an actual example "
                "the writer produced. The prototype beside it is a robust summary of the compatible variants and is a different "
                "object: one selects, the other combines. Shown for three writers, since the spread of a pool is a property of "
                "the hand and of how cleanly it segmented.")
    Stamp(fig,"style._ShapeGrid, final stored profiles")
    Save(fig,"T45_variant_pool_medoid")
    return out


#======================================================================
#T46  THE 12 BY 12 OCCUPANCY DESCRIPTOR
#what the prior and the consensus rule actually compare
#======================================================================
def T46(writers,char="a"):
    fig,axes = plt.subplots(2,len(writers),figsize=(1.5*len(writers)+0.8,3.4),
                            squeeze=False,
                            gridspec_kw=dict(hspace=0.3,wspace=0.22))
    for c,w in enumerate(writers):
        prof,_ = Profile(w)
        pool = prof["glyphs"].get(char,[])
        if not pool:continue
        g = pool[0]
        ax = axes[0][c]
        DrawGlyph(ax,g["strokes"])
        GlyphFrame(ax,f"{w.capitalize()}, '{char}'")
        v = ST._ShapeGrid(g)
        if v is None:continue
        n = int(round(len(v)**0.5))
        grid = np.asarray(v).reshape(n,n)
        ax = axes[1][c]
        #origin lower so the grid reads the same way up as the glyph above it
        ax.imshow(grid,cmap=CMAP,interpolation="nearest",origin="lower")
        ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
        ax.set_title(f"{n} x {n} occupancy, unit length",fontsize=FS(7))
    Caption(fig,"T46  The descriptor the shape prior and the consensus filter operate on. A glyph's strokes are reduced to a "
                "smoothed, normalised occupancy grid, so two forms are compared by where the ink falls rather than by how many "
                "points a path happens to contain. The same letter from different writers gives visibly different grids, which "
                "is what makes the comparison informative within a writer and tolerant across writers.")
    Stamp(fig,"style._ShapeGrid")
    Save(fig,"T46_shape_descriptor")


#======================================================================
#T47  THE SLANT SEARCH
#the same shape of evidence as the deskew sweep, one level down
#======================================================================
def T47(samples):
    fig,axes = plt.subplots(2,1,figsize=(7.2,4.2),
                            gridspec_kw=dict(height_ratios=[1,1.1],hspace=0.42))
    axT,axS = axes
    rows = []
    for i,(name,gray) in enumerate(samples):
        ink = SEG.BinarizeLine(gray)
        best = SEG.EstimateSlantDeg(ink)
        angles = np.arange(-30,30.1,1.0)
        H = ink.shape[0]
        score = []
        for a in angles:
            sh = np.tan(np.radians(a))
            yy,xx = np.nonzero(ink)
            k = np.rint(xx+sh*(H-1-yy)).astype(np.int64)
            k -= k.min()
            prof = np.bincount(k).astype(float)
            score.append(float((prof**2).sum()))
        score = np.array(score)/max(score)
        axS.plot(angles,score,lw=1.1,color=CAT[i%len(CAT)],
                 label=f"{name}  ({best:+.2f}$^\\circ$)")
        axS.plot([best],[np.interp(best,angles,score)],"o",ms=5,mfc="none",
                 mec=CAT[i%len(CAT)],mew=1.2)
        rows.append((name,float(best)))

    #the first writer's line, upright and sheared back
    name,gray = samples[0]
    ink = SEG.BinarizeLine(gray)
    best = SEG.EstimateSlantDeg(ink)
    xs = np.nonzero(ink.any(0))[0]
    left = int(xs.min());right = min(gray.shape[1],left+int(gray.shape[0]*4.5))
    import train as TR
    de = TR.MyAffine(gray,angleDeg=0.0,shearDeg=-best)
    axT.imshow(np.concatenate([gray[:,left:right],
                               np.full((gray.shape[0],12),255,np.uint8),
                               np.asarray(de,np.uint8)[:,left:right]],axis=1),
               cmap="gray",vmin=0,vmax=255,aspect="auto",interpolation="nearest")
    axT.set_xticks([]);axT.set_yticks([]);axT.grid(False)
    for s in axT.spines.values():
        s.set_color(SILVER);s.set_linewidth(0.6)
    axT.set_title(f"{name} as written (left) and sheared by the measured "
                  f"{-best:+.2f}$^\\circ$ (right)",fontsize=FS(8.5))

    axS.set_xlabel("shear angle (degrees)")
    axS.set_ylabel("squared column concentration\n(normalised per writer)")
    axS.set_title("Slant measured by shearing the ink until its columns are "
                  "most concentrated",fontsize=FS(8.5))
    axS.legend(fontsize=FS(6.8),ncol=4,loc="lower center")

    Caption(fig,"T47  Slant estimated over a whole line rather than from one letter. The ink is sheared through candidate angles "
                "and the angle giving the most concentrated column profile is stored. Different writers put the peak in clearly "
                "different places, which is what makes slant a usable style quantity; a flat curve would mean the measure could "
                "not separate them. It remains an estimate affected by letter content, not an observation of the hand.")
    Stamp(fig,"segment.EstimateSlantDeg")
    Save(fig,"T47_slant_search")
    return rows


#======================================================================
#T49  CONNECTEDNESS, COUNTED ON A REAL LINE
#the definition made visible: which boundaries a skeleton actually crosses
#======================================================================
def T49(gray,text,model,tag=""):
    ink = SEG.BinarizeLine(gray)
    glyphs,stats = ST.ExtractLineGlyphs(gray,text,model)
    cuts = stats["cuts"]
    xh = stats["xh"]

    adj,cross = [],[]
    for a,b in zip(cuts,cuts[1:]):
        ia,cha,la,ra = a
        ib,chb,lb,rb = b
        if cha==" " or chb==" ":continue
        if glyphs[ia] is None or glyphs[ib] is None:continue
        adj.append((ia,ib,ra))
        if glyphs[ia].get("connR") or glyphs[ib].get("connL"):
            cross.append(ra)
    c = len(cross)/max(1,len(adj))

    #the whole line across one column would be four millimetres high, so it
    #is stacked in two bands and the boundaries are drawn on both
    NB = 2
    y0,y1 = InkRows(gray,pad=0.12)
    bandH = (WIDTH-0.72)/NB*(y1-y0)/float(gray.shape[1])
    zoomH = 1.15
    fig = plt.figure(figsize=(WIDTH,NB*bandH+zoomH+1.05))
    gs = fig.add_gridspec(NB+1,1,height_ratios=[bandH]*NB+[zoomH],hspace=0.42,
                          left=0.105,right=0.995,top=0.925,bottom=0.025)
    for bi,(xa,xb) in enumerate(Bands(gray.shape[1],NB)):
        axb = fig.add_subplot(gs[bi])
        ShowGray(axb,gray[y0:y1,xa:xb],None,aspect="auto")
        for _,_,x in adj:
            if xa<=x<xb:
                axb.axvline(x-xa,color=SILVER,lw=1.0)
        for x in cross:
            if xa<=x<xb:
                axb.axvline(x-xa,color=DARK,lw=1.8)
        axb.set_xlim(0,xb-xa)
        axb.set_ylabel(f"columns\n{xa}–{xb}",fontsize=FS(7.4),rotation=0,
                       ha="right",va="center",labelpad=6)
        if bi==0:
            axb.set_title(f"(a)  {len(adj)} eligible adjacent non-space "
                          f"boundaries in silver, {len(cross)} crossed by a "
                          "traced centreline in blue",fontsize=FS(8.3),pad=4)

    #a zoom on a crossed boundary beside an uncrossed one, if both exist
    ax = fig.add_subplot(gs[NB])
    axes = [None,ax]
    if cross and adj:
        xc = cross[len(cross)//2]
        unc = [x for _,_,x in adj if x not in cross]
        xu = min(unc,key=lambda v:abs(v-xc)) if unc else xc
        w = int(2.2*xh)
        def Win(x):
            a0 = int(max(0,x-w));a1 = int(min(gray.shape[1],x+w))
            return gray[y0:y1,a0:a1],x-a0
        g1,p1 = Win(xc);g2,p2 = Win(xu)
        pad = np.full((y1-y0,30),255,np.uint8)
        ax.imshow(np.concatenate([g1,pad,g2],axis=1),cmap="gray",vmin=0,vmax=255,
                  aspect="auto",interpolation="nearest")
        ax.axvline(p1,color=DARK,lw=2.0)
        ax.axvline(g1.shape[1]+30+p2,color=SILVER,lw=2.0)
        ax.set_title("(b)  a crossed boundary (blue, left) beside an uncrossed "
                     "one (silver, right), at the same scale",
                     fontsize=FS(8.3),pad=4)
    ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
    for s in ax.spines.values():
        s.set_color(SILVER);s.set_linewidth(0.6)

    Caption(fig,f"T49  Connectedness as the implementation defines it: the fraction of eligible adjacent non-space glyph "
                f"boundaries that a traced centreline actually crosses, here {len(cross)}/{len(adj)} = {c:.4f} on one real line. "
                "It is measured on the writer's own ink, not by counting pen lifts in a generated file, which would instead "
                "measure the extractor's and the synthesiser's decisions. Illustration of Equation 3.18.")
    Stamp(fig,f"style.ExtractLineGlyphs | {tag}")
    Save(fig,"T49_connectedness",f"{len(cross)}/{len(adj)} = {c:.4f}")
    return dict(adjacent=len(adj),crossed=len(cross),connectedness=float(c))


#======================================================================
#T19  THE GLYPH QUALITY GATES
#which cuts the library keeps, which it rejects, and on what grounds
#======================================================================
def T19(writer="owen",chars=("a","e","o","t","n","r")):
    prof,fn = Profile(writer)
    rows = []
    for ch in chars:
        pool = prof["glyphs"].get(ch,[])
        if pool:
            rows.append((ch,pool))
    if not rows:
        return None

    nR = len(rows)
    maxN = max(len(p) for _,p in rows)
    cols = min(maxN,10)+1
    fig,axes = plt.subplots(nR,cols,figsize=(0.72*cols+1.0,0.95*nR+1.0),
                            squeeze=False,
                            gridspec_kw=dict(hspace=0.35,wspace=0.12))
    wideSet = set("mw")
    summary = {}
    for r,(ch,pool) in enumerate(rows):
        widths = np.array([g["width"] for g in pool])
        w40 = float(np.percentile(widths,40))
        for c in range(cols):
            ax = axes[r][c]
            GlyphFrame(ax,None)
            if c==0:
                ax.axis("off")
                ax.text(0.5,0.5,f"'{ch}'\n{len(pool)} kept",ha="center",va="center",
                        fontsize=FS(8),transform=ax.transAxes)
                continue
            k = c-1
            if k>=len(pool):
                ax.axis("off");continue
            g = pool[k]
            lim = 2.9 if ch in wideSet else 2.1
            ok = (g["width"]<=lim) and ST._ClassOk(ch,g["top"],g["bot"])
            DrawGlyph(ax,g["strokes"],color=DARK if ok else SILVER,
                      lw=1.3 if ok else 1.0)
            ax.axhline(0,color=SILVER,lw=0.5,ls="--",zorder=0)
            pd = g.get("priorD")
            ax.set_title(f"w {g['width']:.2f}" + (f"\nd {pd:.3f}" if pd is not None else ""),
                         fontsize=FS(5.6),pad=1)
        summary[ch] = dict(n=len(pool),width40=round(w40,3),
                           widthMax=round(float(widths.max()),3))

    fig.suptitle(f"Stored glyph pools for {writer.capitalize()}, with each variant's "
                 "width in x-heights and prior distance",fontsize=FS(9),y=1.0)
    Caption(fig,"T19  The variants that survived the library gates, with the two quantities the gates act on printed on each. "
                "Width is capped at 2.1 x-heights for ordinary characters and 2.9 for wide ones, and the height class is checked "
                "against what the letter should reach, so an 'l' that never rises above the band is rejected as a mis-cut rather "
                "than stored as a narrow letter. These are the pools after filtering: the gates only ever reject, never invent.")
    Stamp(fig,f"style gates, {fn}")
    Save(fig,"T19_glyph_quality_gates")
    return summary


if __name__=="__main__":
    print("T19-T49  style extraction visual tests")
    res = {}
    model = REC.PaperCRNNNumpy(READER_NPZ)

    gray = Gray(LineCrop("Owen",1,0))
    img,labp = PersonalPage("Owen",1)
    truth = PageLines(labp)[0]

    res["T41"] = T41(gray,"Owen01 line 00")
    res["T42"] = T42(gray,truth,model,"Owen01 line 00")
    res["T43"] = T43(gray,truth,model,"V","Owen01 line 00")

    res["T45"] = T45(["owen","dylan","yeukita"],"a")
    T46(["owen","dylan","yeukita","150"],"a")

    samples = [(w,Gray(LineCrop(w,1,0))) for w in
               ["Owen","Yeukita","Dylan","Abhinav"] if LineCrop(w,1,0).exists()]
    res["T47"] = T47(samples)

    res["T49"] = T49(gray,truth,model,"Owen01 line 00")
    res["T19"] = T19("owen")

    (DATA/"style_results.json").write_text(json.dumps(res,indent=1,default=float),
                                           encoding="utf-8")
    print("done")
