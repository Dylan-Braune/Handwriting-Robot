#u23539195 Dylan Braune.  visual tests for Section 3.5, the network.
#
#every panel runs the deployed reader checkpoint forward in NumPy on a real
#line crop through recognise.py.  nothing is trained and no accuracy is
#claimed: these show what the implemented layers compute.
#
#  python vt_net.py
from vt_common import *
import numpy as np
import segment as SEG
import recognise as REC


#---Crop: one real line crop, as the network receives it---
def Crop(writer="Owen",page=1,index=0):
    return Gray(LineCrop(writer,page,index))


#---Canvas: the 64 by 640 fitted crop and the network input tensor---
def Canvas(gray):
    fit = REC.ResizeLineImageFixed(gray)
    return fit,REC.TensorFromResized(fit)


#======================================================================
#T30  ONE CONVOLUTION, WRITTEN OUT
#the Design draft quotes sum WX, the bias and the resulting z for a named
#channel, row and column.  this shows the nine numbers that produced them,
#then follows the same activation through normalisation and ReLU
#======================================================================
def T30(model,fit,x,row=34,col=462,ch=0):
    sd = model.sd
    W = sd["stage1.0.weight"]           #(out,in,3,3)
    b = sd["stage1.0.bias"]
    #the padded input patch the implementation multiplies
    pad = np.pad(x[0,0],1,mode="constant")
    patch = pad[row:row+3,col:col+3]
    kern = W[ch,0]
    prod = kern*patch
    z = float(prod.sum()+b[ch])

    #and the batch-normalisation that follows it, from the stored statistics
    g = sd["stage1.1.weight"][ch];be = sd["stage1.1.bias"][ch]
    mu = sd["stage1.1.running_mean"][ch];var = sd["stage1.1.running_var"][ch]
    y = float(g*(z-mu)/np.sqrt(var+1e-5)+be)
    relu = max(0.0,y)

    fig = plt.figure(figsize=(7.2,4.3))
    gs = fig.add_gridspec(2,4,height_ratios=[1.15,1],width_ratios=[1.5,1,1,1],
                          hspace=0.45,wspace=0.3)

    #where on the line the sampled activation sits
    axL = fig.add_subplot(gs[0,:])
    ShowGray(axL,fit,None)
    axL.plot([col],[row],"o",ms=7,mfc="none",mec=INK,mew=1.3)
    axL.annotate(f"row {row}, column {col}",(col,row),
                 xytext=(col-150,row-22),fontsize=FS(7.5),color=INK,
                 arrowprops=dict(arrowstyle="->",color=INK,lw=0.8))
    axL.set_title("The 64 x 640 network input, with the sampled position marked",
                  fontsize=FS(8.5))

    def Grid(ax,M,title,fmt="{:+.3f}",cmap=None):
        ax.imshow(M,cmap=cmap or CMAP,interpolation="nearest",
                  vmin=-np.abs(M).max(),vmax=np.abs(M).max())
        for i in range(3):
            for j in range(3):
                v = M[i,j]
                t = (v+np.abs(M).max())/(2*np.abs(M).max()+1e-12)
                ax.text(j,i,fmt.format(v),ha="center",va="center",fontsize=FS(6.6),
                        color=PAPER if t>0.68 else INK)
        ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
        ax.set_title(title,fontsize=FS(8))

    Grid(fig.add_subplot(gs[1,0]),patch,"input patch $X$  (ink positive)")
    Grid(fig.add_subplot(gs[1,1]),kern,f"learned kernel $W$  (channel {ch})")
    Grid(fig.add_subplot(gs[1,2]),prod,"elementwise $W\\odot X$")

    axT = fig.add_subplot(gs[1,3])
    axT.axis("off")
    axT.text(0.0,1.0,
        "$\\sum W\\odot X$ = "+f"{prod.sum():+.6f}\n"
        f"bias $b$ = {b[ch]:+.6f}\n"
        "$z$ = "+f"{z:+.6f}\n\n"
        "batch norm, stored statistics\n"
        f"  $\\mu$={mu:.7f}  $\\sigma^2$={var:.8f}\n"
        f"  $\\gamma$={g:.7f}  $\\beta$={be:.7f}\n"
        "$y$ = "+f"{y:+.5f}\n\n"
        "ReLU$(y)$ = "+f"{relu:.5f}",
        transform=axT.transAxes,va="top",ha="left",fontsize=FS(7),
        linespacing=1.5,family="serif")

    Caption(fig,"T30  One convolution evaluated at a single position of a real line crop, with the nine input values, the nine "
                "learned weights and their products shown. The same activation is then carried through batch normalisation using "
                "the checkpoint's stored statistics and through the ReLU that follows. Illustration of Equations 3.7 and 3.8.")
    Stamp(fig,f"recognise.Conv2d / recognise.BatchNorm2d | {READER_NPZ.name}")
    Save(fig,"T30_convolution_arithmetic",f"z={z:+.6f}, y={y:+.5f}")
    return dict(sumWX=float(prod.sum()),bias=float(b[ch]),z=z,
                mu=float(mu),var=float(var),gamma=float(g),beta=float(be),
                y=y,relu=relu,row=row,col=col,channel=ch)


#======================================================================
#T31  THE LEARNED FIRST-LAYER KERNELS
#evidence that the checkpoint carries structure rather than noise
#======================================================================
def T31(model):
    W = model.sd["stage1.0.weight"][:,0]      #(32,3,3)
    n = W.shape[0]
    cols = 8;rows = int(np.ceil(n/cols))
    fig,axes = plt.subplots(rows,cols,figsize=(7.2,0.95*rows+0.75),
                            gridspec_kw=dict(hspace=0.28,wspace=0.12))
    for k in range(rows*cols):
        ax = axes.flat[k]
        ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
        if k>=n:
            ax.axis("off");continue
        M = W[k]
        ax.imshow(M,cmap=CMAP,interpolation="nearest",
                  vmin=-np.abs(M).max(),vmax=np.abs(M).max())
        ax.set_title(f"{k}",fontsize=FS(6),pad=1.5)
        for s in ax.spines.values():
            s.set_color(SILVER);s.set_linewidth(0.5)
    fig.suptitle(f"All {n} learned 3 x 3 kernels of the first convolution, "
                 "each scaled to its own range",fontsize=FS(9),y=1.0)
    Caption(fig,"T31  The first convolutional layer of the deployed reader checkpoint. Each tile is one learned kernel, scaled "
                "independently so its structure is visible. Oriented light-dark pairs are stroke and edge detectors; a layer that "
                "had not trained would show unstructured noise in every tile.")
    Stamp(fig,f"stage1.0.weight | {READER_NPZ.name}")
    Save(fig,"T31_learned_kernels",f"{n} kernels")
    return int(n)


#======================================================================
#T32 and T34  THE FEATURE STACK AND THE HEIGHT COLLAPSE
#the shapes in the backbone table, shown as the arrays they describe
#======================================================================
def T32(model,fit,x):
    sd = model.sd
    a1 = REC.ConvBlockForward(x,sd,"stage1",2)
    p1 = REC.MaxPool2d(a1,2,2)
    a2 = REC.ConvBlockForward(p1,sd,"stage2",4)
    p2 = REC.MaxPool2d(a2,2,2)
    a3 = REC.ConvBlockForward(p2,sd,"stage3",6)
    h = REC.AdaptiveMaxPoolHeight(a3)

    stages = [("input\n$1\\times64\\times640$",x[0,0],None),
              ("stage 1 out\n$32\\times64\\times640$",None,a1),
              ("after pool\n$32\\times32\\times320$",None,p1),
              ("stage 2 out\n$64\\times32\\times320$",None,a2),
              ("after pool\n$64\\times16\\times160$",None,p2),
              ("stage 3 out\n$128\\times16\\times160$",None,a3)]

    #the input is one channel, so it takes the full width and is drawn large
    #enough to read; every later stage shows three channels side by side
    rowH = 0.70
    fig = plt.figure(figsize=(WIDTH,6*rowH+1.45+0.30))
    gs = fig.add_gridspec(8,3,hspace=0.30,wspace=0.045,
                          height_ratios=[1,1,1,1,1,1,0.18,2.05],
                          left=0.175,right=0.995,top=0.935,bottom=0.075)

    for r,(title,raw,act) in enumerate(stages):
        #three channels per stage, the ones carrying the most energy, so the
        #panel is not a tour of dead filters
        if raw is not None:
            picks = [raw]
        else:
            e = act[0].reshape(act.shape[1],-1).std(1)
            picks = [act[0,int(i)] for i in np.argsort(-e)[:3]]
        spans = [gs[r,:]] if raw is not None else [gs[r,c] for c in range(3)]
        for c,cell in enumerate(spans):
            ax = fig.add_subplot(cell)
            M = picks[c]
            ax.imshow(M,cmap="gray_r" if raw is not None else CMAP,
                      aspect="auto",interpolation="nearest")
            ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
            for s in ax.spines.values():
                s.set_color(SILVER);s.set_linewidth(0.5)
            if c==0:
                ax.set_ylabel(title,fontsize=FS(8),rotation=0,ha="right",
                              va="center",labelpad=8)

    #the height collapse, which is the architectural decision
    axH = fig.add_subplot(gs[7,:])
    axH.imshow(h[0,:,0,:],cmap=CMAP,aspect="auto",interpolation="nearest")
    axH.set_yticks([0,64,127]);axH.tick_params(labelsize=FS(8))
    axH.set_xlabel("time step (160 frames)",fontsize=FS(8.2))
    axH.set_ylabel("channel\n(128)",fontsize=FS(8),rotation=0,ha="right",
                   va="center",labelpad=8)
    axH.grid(False)
    axH.set_title("maximum over the remaining height,  "
                  "$128\\times16\\times160 \\rightarrow 128\\times1\\times160$:"
                  "\n"
                  "height is discarded and width becomes the CTC time axis",
                  fontsize=FS(8.3),pad=5,linespacing=1.35)

    fig.suptitle("The convolutional stack on one real line, three strongest "
                 "channels per stage",fontsize=FS(9),y=0.995)
    Caption(fig,"T32 and T34  Feature maps at their true resolution as the crop passes through the implemented backbone. Spatial "
                "detail collapses while the receptive field grows, and the final operation discards height entirely. That last step "
                "is what makes the sequence model possible: after it, each of the 160 columns is one time step with 128 features. "
                "Illustration of the shapes in the backbone table.")
    Stamp(fig,f"recognise.ConvBlockForward / AdaptiveMaxPoolHeight | {READER_NPZ.name}")
    Save(fig,"T32_feature_stack")
    return dict(shapes={k:list(v.shape) for k,v in
                        [("stage1",a1),("pool1",p1),("stage2",a2),
                         ("pool2",p2),("stage3",a3),("heightPooled",h)]})


#======================================================================
#T33  MAX POOLING, NUMERICALLY AND ON A REAL MAP
#======================================================================
def T33(model,x):
    sd = model.sd
    a1 = REC.ConvBlockForward(x,sd,"stage1",2)
    p1 = REC.MaxPool2d(a1,2,2)
    e = a1[0].reshape(a1.shape[1],-1).std(1)
    ch = int(np.argmax(e))

    #a 4 by 4 window of that same real feature map, and its 2 by 2 result
    sub = a1[0,ch,30:34,300:304]
    out = np.array([[sub[0:2,0:2].max(),sub[0:2,2:4].max()],
                    [sub[2:4,0:2].max(),sub[2:4,2:4].max()]])

    fig = plt.figure(figsize=(7.2,3.1))
    gs = fig.add_gridspec(1,4,width_ratios=[1,0.72,1.6,1.6],wspace=0.3)

    def Num(ax,M,title):
        ax.imshow(M,cmap=CMAP,interpolation="nearest",
                  vmin=M.min(),vmax=max(M.max(),1e-9))
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=FS(6.6),
                        color=PAPER if M[i,j]>0.62*max(M.max(),1e-9) else INK)
        ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
        ax.set_title(title,fontsize=FS(8))
        #the 2 by 2 window boundaries, so the grouping is unambiguous
        if M.shape[0]==4:
            for t in (1.5,):
                ax.axhline(t,color=INK,lw=1.0);ax.axvline(t,color=INK,lw=1.0)

    Num(fig.add_subplot(gs[0,0]),sub,f"a real $4\\times4$ window\n(channel {ch}, rows 30-33)")
    Num(fig.add_subplot(gs[0,1]),out,"after $2\\times2$ max\nstride 2")

    axA = fig.add_subplot(gs[0,2])
    axA.imshow(a1[0,ch],cmap=CMAP,aspect="auto",interpolation="nearest")
    axA.set_xticks([]);axA.set_yticks([]);axA.grid(False)
    axA.set_title(f"channel {ch} before:  {a1.shape[2]} x {a1.shape[3]}",fontsize=FS(8))
    axB = fig.add_subplot(gs[0,3])
    axB.imshow(p1[0,ch],cmap=CMAP,aspect="auto",interpolation="nearest")
    axB.set_xticks([]);axB.set_yticks([]);axB.grid(False)
    axB.set_title(f"and after:  {p1.shape[2]} x {p1.shape[3]}",fontsize=FS(8))

    Caption(fig,"T33  Max pooling taken from a real feature map rather than from an invented example. Each 2 x 2 window keeps its "
                "largest response, so height and width halve while the channel count is unchanged. The strongest response survives "
                "a small shift of the stroke that produced it, which is why the map on the right is a coarser but not a weaker "
                "description of the same line.")
    Stamp(fig,f"recognise.MaxPool2d | {READER_NPZ.name}")
    Save(fig,"T33_max_pooling")


#======================================================================
#T35  WHICH PIXELS EACH TIME STEP SEES
#the 160 frames mapped back onto the crop through the fitted scale
#======================================================================
def T35(gray,model):
    fit,x = Canvas(gray)
    lp = model.forward(x)[:,0,:]
    T = lp.shape[0]
    h,w = gray.shape
    edges = [REC.FrameXToPixel(640.0*t/T,w,h) for t in range(T+1)]

    fig,(axA,axB) = plt.subplots(2,1,figsize=(7.2,2.9),
                                 gridspec_kw=dict(height_ratios=[1,1],hspace=0.42))
    ShowGray(axA,gray,None)
    for t in range(0,T+1,10):
        axA.axvline(edges[t],color=MID,lw=0.5)
    axA.set_title(f"Source crop {h} x {w} px, every tenth frame boundary drawn",
                  fontsize=FS(8.5))

    ShowGray(axB,fit,None)
    step = 640.0/T
    for t in range(0,T+1,10):
        axB.axvline(t*step,color=MID,lw=0.5)
    axB.set_title(f"The same boundaries on the 64 x 640 canvas: "
                  f"{T} frames of {step:.0f} canvas columns each",fontsize=FS(8.5))

    Caption(fig,"T35  The correspondence between an output time step and a position in the original crop. A frame edge at t of T "
                "maps to canvas column 640t/T and back through the fitted scale to a source pixel. This is what lets an alignment "
                "produced in frame coordinates be turned into a cut in the image, and it is the step that connects the recogniser "
                "to glyph extraction.")
    Stamp(fig,"recognise.FrameXToPixel")
    Save(fig,"T35_frame_to_pixel",f"{T} frames over {w} px")
    return dict(frames=int(T),sourceWidth=int(w),canvasStep=float(step))


#======================================================================
#T36  THE CTC LATTICE, THE FORCED PATH AND THE GREEDY PATH
#the figure the section most needs: blanks, collapsing, alignment against
#decoding, and the derivative all in one place
#======================================================================
def T36(gray,text,model,tag=""):
    fit,x = Canvas(gray)
    lp = model.forward(x)[:,0,:]            #(T,classes) log probs
    T,K = lp.shape
    pred = REC.DecodeCtc(lp[:,None,:])[0]
    al = REC.CtcForcedAlign(lp,text)
    spans = al[0] if al else []
    conf = al[1] if al else float("nan")

    probs = np.exp(lp)
    greedy = lp.argmax(1)

    #only the rows the transcript uses, so the lattice is readable
    used = [0]+[REC.CHAR_TO_IDX[c] for c in text if c in REC.CHAR_TO_IDX]
    seen,rows = set(),[]
    for u in used:
        if u not in seen:
            seen.add(u);rows.append(u)
    lab = ["blank"]+[REC.IDX_TO_CHAR[r] for r in rows[1:]]

    #the lattice needs one readable row per class, so its height follows the
    #number of classes rather than a fixed ratio, and the input line is
    #stacked in two bands so the writing can be read against the alignment
    NB = 2
    bandH = (WIDTH-0.72)/NB*fit.shape[0]/fit.shape[1]
    latH = max(2.6,0.135*len(rows))
    probH = 0.95
    gapH = 0.34                               #room for the decoding statement
    fig = plt.figure(figsize=(WIDTH,NB*bandH+latH+gapH+probH+1.30))
    gs = fig.add_gridspec(NB+3,1,hspace=0.40,
                          height_ratios=[bandH]*NB+[latH,gapH,probH],
                          left=0.115,right=0.995,top=0.955,bottom=0.062)

    for bi,(x0,x1) in enumerate(Bands(fit.shape[1],NB)):
        axI = fig.add_subplot(gs[bi])
        ShowGray(axI,fit[:,x0:x1],None,aspect="auto")
        if bi==0:
            axI.set_title("input line on the network canvas, known transcript:"
                          "\n" f"“{text}”",fontsize=FS(8.3),pad=4,
                          linespacing=1.35)
        axI.set_ylabel(f"canvas\ncols {x0}–{x1}",fontsize=FS(7.4),
                       rotation=0,ha="right",va="center",labelpad=6)

    axL = fig.add_subplot(gs[NB])
    M = probs[:,rows].T
    axL.imshow(M,cmap=CMAP,aspect="auto",interpolation="nearest",
               vmin=0,vmax=1,extent=[0,T,len(rows)-0.5,-0.5])
    axL.set_yticks(range(len(rows)));axL.set_yticklabels(lab,fontsize=FS(7.2))
    axL.grid(False)
    axL.tick_params(axis="y",length=2,pad=1.5)
    axL.tick_params(axis="x",length=0)
    axL.set_ylabel("output class",fontsize=FS(8.2),labelpad=2)
    axL.set_xticklabels([])

    #the greedy best-path decision at every frame, as open silver markers
    rowOf = {r:i for i,r in enumerate(rows)}
    gy = [rowOf.get(int(g),np.nan) for g in greedy]
    axL.plot(np.arange(T)+0.5,gy,ls="none",marker="s",ms=2.6,mfc="none",
             mec=SILVER,mew=0.8)

    #the forced-alignment path: one bar per character over its frame span
    for ch,sp in spans:
        if sp is None:continue
        r = rowOf.get(REC.CHAR_TO_IDX[ch])
        if r is None:continue
        axL.plot([sp[0],sp[1]],[r,r],color=INK,lw=2.2,solid_capstyle="butt")

    from matplotlib.lines import Line2D
    axL.legend(handles=[
        Line2D([],[],color=INK,lw=2.2,label="forced alignment to the known text"),
        Line2D([],[],ls="none",marker="s",ms=3.4,mfc="none",mec=SILVER,
               label="greedy best path per frame")],
        fontsize=FS(6.6),loc="lower center",bbox_to_anchor=(0.5,1.01),ncol=2,frameon=False)

    ok = ("matches the transcript exactly" if pred==text
          else "differs from the transcript by %d edit(s)"%REC.Levenshtein(pred,text))
    axL.text(0.5,-0.022,f"best-path decoding collapses to “{pred}”\n"
             f"— {ok}",transform=axL.transAxes,ha="center",va="top",
             fontsize=FS(7.8),color=INK,linespacing=1.4)

    axB = fig.add_subplot(gs[NB+2],sharex=axL)
    axB.fill_between(np.arange(T),probs[:,0],color=SILVER,lw=0)
    axB.plot(np.arange(T),1.0-probs[:,0],color=DARK,lw=1.0)
    axB.set_ylim(0,1.02);axB.set_xlim(0,T)
    axB.set_xlabel("time step (frame)",fontsize=FS(8.2))
    axB.set_ylabel("probability",fontsize=FS(8.2))
    axB.legend(handles=[
        Line2D([],[],color=SILVER,lw=5,label="blank"),
        Line2D([],[],color=DARK,lw=1.2,label="any character")],
        fontsize=FS(8),ncol=2,loc="lower left",bbox_to_anchor=(0.0,1.015),
        frameon=False,handlelength=1.6,columnspacing=1.4)
    axB.text(1.0,1.03,f"blank carries {probs[:,0].mean()*100:.0f}% of the "
             "probability mass on average",transform=axB.transAxes,
             ha="right",va="bottom",fontsize=FS(8.2),color=INK)
    axB.tick_params(labelbottom=True,labelsize=FS(8))

    Caption(fig,"T36  The reader's complete output for one real line. Rows are the classes the transcript uses, and the black "
                "segments are the frames the forced alignment assigns to each character. Decoding and alignment search the same "
                "lattice differently: decoding takes the best class per frame and collapses, while alignment maximises over only "
                "the paths that spell the known text. The blank class holds most of the probability mass between characters, which "
                "is how the collapse rule distinguishes a repeated letter from a held one.")
    Stamp(fig,f"recognise.PaperCRNNNumpy / CtcForcedAlign | {tag}")
    Save(fig,"T36_ctc_lattice",f"T={T}, conf={conf:.4f}")
    return dict(frames=int(T),decoded=pred,truth=text,alignConfidence=float(conf),
                blankMass=float(probs[:,0].mean()))


#======================================================================
#T37  PER-FRAME POSTERIORS AS SPIKES
#the same output read as a time series, which is easier than the lattice
#======================================================================
def T37(gray,text,model,tag=""):
    fit,x = Canvas(gray)
    lp = model.forward(x)[:,0,:]
    probs = np.exp(lp)
    T = probs.shape[0]
    best = probs[:,1:].argmax(1)+1
    peak = probs[np.arange(T),best]

    fig,(axA,axB) = plt.subplots(2,1,figsize=(7.4,3.0),
                                 gridspec_kw=dict(height_ratios=[1,1.25],hspace=0.33))
    ShowGray(axA,fit,None)
    axA.set_title(f"“{text}”",fontsize=FS(8.3))

    axB.vlines(np.arange(T),0,peak,color=DARK,lw=1.1)
    axB.plot(np.arange(T),probs[:,0],color=SILVER,lw=0.9,label="blank")
    for t in range(T):
        if peak[t]>0.45:
            axB.text(t,peak[t]+0.03,REC.IDX_TO_CHAR.get(int(best[t]),""),
                     ha="center",fontsize=FS(5.6),color=INK)
    axB.set_xlim(0,T);axB.set_ylim(0,1.18)
    axB.set_xlabel("time step (frame)",fontsize=FS(7.5))
    axB.set_ylabel("probability",fontsize=FS(7.5))
    axB.set_title("strongest non-blank class per frame, labelled above 0.45",
                  fontsize=FS(8.3))
    axB.legend(fontsize=FS(6.5),loc="upper right")

    Caption(fig,"T37  The same output read along time. Character probability appears as isolated spikes separated by blank, which "
                "is the output shape CTC training produces: the network commits to a character at one or two frames and emits blank "
                "elsewhere rather than spreading a letter across its full width.")
    Stamp(fig,f"recognise.PaperCRNNNumpy | {tag}")
    Save(fig,"T37_frame_posteriors")


#======================================================================
#T40  THE STROKE-WIDTH SWEEP, SHOWN
#the table reports four accuracies; this shows what they chose between
#======================================================================
def T40(samples):
    widths = [0.06,0.09,0.11,0.17]
    acc = {0.06:98.06,0.09:97.42,0.11:98.06,0.17:99.35}
    ep = {0.06:38,0.09:43,0.11:39,0.17:43}

    #a line crop is mostly blank paper above and below the writing, and at
    #five panels across a column that leaves the ink a millimetre high.  each
    #crop is therefore trimmed to the rows that carry ink and cut to a short
    #window, so the same letters are drawn several times larger
    nR = len(samples)
    panelW = (WIDTH-0.80)/5
    ASPECT = 2.0
    rowH = panelW/ASPECT
    fig = plt.figure(figsize=(WIDTH,nR*rowH+2.25))
    gs = fig.add_gridspec(nR+1,5,height_ratios=[rowH]*nR+[1.55],
                          width_ratios=[1,1,1,1,1],hspace=0.34,wspace=0.055,
                          left=0.125,right=0.995,top=0.955,bottom=0.085)

    kept = {}
    for r,(name,gray) in enumerate(samples):
        ink = SEG.BinarizeLine(gray)
        band = SEG.CoreBand(ink)
        xh = float(band[1]-band[0]) if band else max(8.0,gray.shape[0]/3.0)
        y0,y1 = InkRows(gray,pad=0.08)
        left,right = Window(gray[y0:y1],aspect=ASPECT)
        ax = fig.add_subplot(gs[r,0])
        ShowGray(ax,gray[y0:y1,left:right],"original ink" if r==0 else None,
                 aspect="auto")
        ax.set_ylabel(name,fontsize=FS(8),rotation=0,ha="right",va="center",
                      labelpad=6)
        for c,rho in enumerate(widths):
            w = int(np.clip(round(rho*xh),2,9))
            k = 2*(w//2)+1
            kept[(name,rho)] = k
            norm = REC.StrokeNormalize(gray,penWidthXh=rho)
            src = norm if norm is not None else gray
            ax = fig.add_subplot(gs[r,c+1])
            ShowGray(ax,src[y0:y1,left:right],
                     f"$\\rho$={rho:.2f},  $k$={k}" if r==0 else f"$k$={k}",
                     aspect="auto")

    axA = fig.add_subplot(gs[nR,:])
    xs = np.arange(len(widths))
    bars = axA.bar(xs,[acc[w] for w in widths],0.5,
                   color=[DARK if w==0.17 else SILVER for w in widths])
    axA.set_xticks(xs)
    axA.set_xticklabels([f"$\\rho$={w:.2f}\nbest epoch {ep[w]}" for w in widths],
                        fontsize=FS(7))
    axA.set_ylim(96.5,100)
    axA.set_ylabel("held-out writer accuracy (%)",fontsize=FS(7.5))
    for i,w in enumerate(widths):
        axA.text(i,acc[w]+0.07,f"{acc[w]:.2f}%",ha="center",fontsize=FS(7),color=INK)
    axA.set_title("Supplied eleven-writer validation sweep; the selected setting is dark blue",
                  fontsize=FS(8.5))

    Caption(fig,"T40  What the stroke-width sweep was choosing between. Each column redraws the same real crops at one width "
                "fraction, and the dilation kernel k printed on every panel is the quantity that actually changes: nearby fractions "
                "can give the same odd kernel and therefore the same image. Accuracies are the supplied sweep, reproduced here for "
                "comparison, not measured in this figure.")
    Stamp(fig,"recognise.StrokeNormalize; accuracies from the supplied sweep")
    Save(fig,"T40_stroke_width_sweep")
    return {f"{n}@{r}":k for (n,r),k in kept.items()}


if __name__=="__main__":
    import json
    print("T30-T40  network visual tests")
    res = {}
    print(f"  loading {READER_NPZ.name} ...")
    model = REC.PaperCRNNNumpy(READER_NPZ)

    #Owen01 line 00 is the crop the Design draft's worked traces are computed
    #on, so the arithmetic in T30 and T36 matches the numbers in the text
    gray = Crop("Owen",1,0)
    fit,x = Canvas(gray)

    res["T30"] = T30(model,fit,x)
    res["T31"] = T31(model)
    res["T32"] = T32(model,fit,x)
    T33(model,x)
    res["T35"] = T35(gray,model)

    img,labp = PersonalPage("Owen",1)
    lines = PageLines(labp)
    truth = lines[0] if lines else REC.ReadText(gray,model)
    res["T36"] = T36(gray,truth,model,"Owen01 line 00")
    T37(gray,truth,model,"Owen01 line 00")

    #the stroke-width sweep shown on four different writers, since a width
    #that suits one hand need not suit another
    samples = []
    for w in ["Owen","Yeukita","Dylan","Abhinav"]:
        q = LineCrop(w,1,0)
        if q.exists():
            samples.append((w,Gray(q)))
    res["T40"] = T40(samples)

    (DATA/"net_results.json").write_text(json.dumps(res,indent=1,default=float),
                                         encoding="utf-8")
    print("done")
