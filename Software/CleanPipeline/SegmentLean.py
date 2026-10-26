#u23539195 Dylan Braune Clean Line Segmentation code, with image conditioning
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

import MyImage

InputHeight,InputWidth = 64,640   #exact input size the classifiers were trained on


#resizes/pads one line crop to the fixed classifier input size, never stretched unevenly
def ToClassifierFormat(img,height = InputHeight,width = InputWidth,pad_value=255):
    gray = np.asarray(img.convert("L") if hasattr(img,'convert') else img,
                      dtype=np.uint8)
    return Image.fromarray(MyImage.FitFixedCanvas(gray,height,width,
                                                     pad_value))


#mean over a (2r+1)x(2r+1) window using an integral image, fast box blur
def BoxMean(a,r):
    p = np.pad(a,r,mode="edge").astype(np.float64)
    ii = np.pad(p.cumsum(0).cumsum(1),((1,0),(1,0)))   #integral image
    k = 2*r+1
    return (ii[k:,k:]-ii[:-k,k:]-ii[k:,:-k]+ii[:-k,:-k])/(k*k)


#true where any pixel within r along `axis` is set, used for dilation
def AnyWithinRadius(m,r,axis):
    a = np.moveaxis(m.astype(np.int32),axis,0)
    n = a.shape[0]
    c = np.cumsum(np.pad(a,[(r+1,r)]+[(0,0)]*(a.ndim-1)),0)
    return np.moveaxis((c[2*r+1:2*r+1+n]-c[:n])>0,0,axis)


#marks pixels lying on a run of >= n consecutive set pixels along `axis`.
#two stacked prefix sums: the first finds where a full-length run STARTS,
#the second turns that into "a run covers this pixel", which is what the
#rule-line removal needs (every pixel of an underline, not just its first)
def LongRunMask(m,n,axis):
    a = np.moveaxis(m.astype(np.int32),axis,0)
    L = a.shape[0]
    if L<n:
        return np.zeros(m.shape,bool)
    c = np.cumsum(np.pad(a,((1,0),(0,0))),0)
    hit = ((c[n:]-c[:-n])==n).astype(np.int32)   #run starts at i
    ch = np.cumsum(np.pad(hit,((1,0),(0,0))),0)
    j = np.arange(L)
    lo,hi = np.maximum(0,j-n+1),np.minimum(j,L-n)   #a run covering j starts in [lo,hi]
    cov = ((ch[hi+1]-ch[lo])>0)&(hi>=lo)[:, None]
    return np.moveaxis(cov,0,axis)


#otsu's method: the threshold that maximises between class variance
def OtsuThreshold(v):
    h,_ = np.histogram(np.clip(v,0,1.2),bins=256,range=(0,1.2))
    p = h/h.sum()
    w,m = np.cumsum(p),np.cumsum(p*np.arange(256))
    s = (m[-1]*w-m)**2/np.maximum(w*(1-w),1e-12)   #between class variance per bin
    return (int(np.argmax(s))+0.5)*1.2/256


#ink mask: pixel is darker than its own locally blurred background
def InkMask(gray):
    r = max(8,min(40,round(gray.shape[0]/41)))   #background blur radius scales with image size
    ratio = gray/np.maximum(BoxMean(gray,r),1.0)
    mask = ratio<OtsuThreshold(ratio)
    solid = BoxMean(mask.astype(np.float64),4)>0.999   #9x9 solid ink: binder interior, shadow
    return mask&~(BoxMean(solid.astype(np.float64),3)>0),ratio


#line spacing P: the gap between the row profile's own peaks, or a fallback
def LinePitch(mask):
    #a direct peak gap tracks irregular spacing correctly, checked against 16 real pages.
    #autocorrelation only kicks in when peaks can't be resolved at all, since it can lock
    #onto a multiple of the true spacing on a page with uneven paragraph gaps. one line with
    #nothing to compare falls back to the page's own letter size (close to P/2.2 in practice)
    W = mask.shape[1]
    prof = mask[:,int(0.05*W):int(0.97*W)].sum(1).astype(float)
    k = np.exp(-0.5*(np.arange(-12,13)/4.0)**2)   #gaussian smoothing kernel
    sm = np.convolve(prof,k/k.sum(),"same")

    peaks = [i for i in range(1,len(sm)-1) if sm[i]>=sm[i-1] and sm[i]>sm[i+1] and sm[i]>0.06*sm.max()]
    if len(peaks)>=2:
        return int(np.median(np.diff(sorted(peaks)))),sm   #direct gap: works for 2+ lines, irregular spacing too

    a = sm-sm.mean()
    ac = np.correlate(a,a,"full")[len(a)-1:]   #autocorrelation fallback
    seg = ac[15:min(200,len(ac)//3)]
    idx = next((i for i in range(1,len(seg)-1)
                if seg[i]>=0.7*seg.max() and seg[i]>=seg[i-1] and seg[i]>=seg[i+1]), None)
    if idx is not None:
        return 15+idx,sm

    lab,nc = LabelConnectedComponents(mask)
    if nc>0:
        ys,xs = np.nonzero(mask)
        pl = lab[ys,xs]
        top,bot = np.full(nc+1,mask.shape[0]),np.zeros(nc+1,int)
        np.minimum.at(top,pl,ys); np.maximum.at(bot,pl,ys)
        area = np.bincount(pl,minlength=nc+1)
        hts = bot-top+1
        sel = (area>=8)&(hts>=3)&(hts<=mask.shape[0])
        if sel[1:].any():
            return max(15,int(round(2.2*np.median(hts[1:][sel[1:]])))),sm   #one line: go by letter size
    return 48,sm   #no usable ink: an arbitrary but harmless default


#left to right minimum cost path (row index per column), dynamic programming
def LeastCostSeam(cost,step=3):
    h,W = cost.shape
    acc = cost.copy()
    back = np.zeros((h,W),np.int8)
    idx = np.arange(h)
    shifts = range(-step,step+1)
    for x in range(1,W):
        p = acc[:,x-1]
        cand = np.full((len(shifts),h),np.inf)
        for i,d in enumerate(shifts):
            if d<0:
                cand[i,-d:] = p[:d]+0.02*abs(d)   #small cost for moving off straight ahead
            elif d>0:
                cand[i,:-d] = p[d:]+0.02*d
            else:
                cand[i] = p
        j = cand.argmin(0)
        acc[:,x]+=cand[j,idx]
        back[:,x] = j-step
    y = np.empty(W,int)
    y[-1] = acc[:,-1].argmin()
    for x in range(W-1,0,-1):
        y[x-1] = y[x]+back[y[x],x]   #backtrack the cheapest path
    return y


#8 connected component labelling via row runs and union find
def LabelConnectedComponents(mask):
    H,W = mask.shape
    lab = np.zeros((H,W),np.int32)
    parent = [0]

    #union find root lookup with path compression
    def find(a):
        while parent[a]!=a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    prev = []
    for y in range(H):
        d = np.diff(np.r_[0,mask[y].view(np.int8),0])
        cur,k = [],0
        for a,b in zip(np.nonzero(d==1)[0].tolist(),np.nonzero(d==-1)[0].tolist()):
            while k<len(prev) and prev[k][1]<a:
                k+=1
            l,kk = 0,k
            while kk<len(prev) and prev[kk][0]<=b:
                r = find(prev[kk][2])
                if l==0:
                    l = r
                elif r!=l:
                    parent[max(r,l)] = min(r,l)   #merge touching runs from the row above
                    l = min(r,l)
                kk+=1
            if l==0:
                parent.append(len(parent))
                l = len(parent)-1
            cur.append((a,b,l))
            lab[y,a:b] = l
        prev = cur
    roots = np.array([find(i) for i in range(len(parent))])
    _,comp = np.unique(roots,return_inverse=True)
    return comp[lab].astype(np.int32),int(comp.max())


#largest bright area of a photo after the handwriting is blurred away
def PaperRegionMask(g,P):
    H,W = g.shape
    s = 4
    c = BoxMean(g[:H//s*s,:W//s*s].reshape(H//s,s,W//s,s).mean((1,3)),max(1,int(0.15*P/s)))
    dark = c<0.45*np.median(c)
    if dark.mean()<0.002:
        return np.ones((H,W),bool)   #no dark surround: nothing to cut away
    lab,n = LabelConnectedComponents(~dark)
    big = np.kron(lab==np.bincount(lab.ravel())[1:].argmax()+1,np.ones((s,s),bool))
    full = np.ones((H,W),bool)
    full[:big.shape[0],:big.shape[1]] = big
    r = int(0.4*P)
    return ~AnyWithinRadius(AnyWithinRadius(~full,r,0),r,1)   #shrink a little from the paper edge


#how level the text lines are at a trial angle
#
#J(angle) = sum over rows of (ink in that row)^2
#
#the total amount of ink does not change when the page turns, so a plain sum
#is the same at every angle and tells you nothing.  the sum of SQUARES is
#biggest when the same ink is packed into the fewest rows, which is exactly
#when the lines are level
#
#the page is never actually rotated to work this out.  rotating a point by a
#about the centre puts it in row
#   k = (y-cy)*cos(a) - (x-cx)*sin(a) + cy
#so the ink coordinates are mapped straight into row numbers and counted
#with bincount.  that is one pass over the ink pixels per angle instead of a
#whole image rotation, measured 31x faster, and it avoids resampling the
#mask 17 times which would blur it a little more each time
def SkewScore(mask,angleDeg):
    ys,xs = np.nonzero(mask)
    if len(ys)==0:
        return 0.0
    h = mask.shape[0]
    cy,cx = (h-1)/2.0,(mask.shape[1]-1)/2.0
    th = np.radians(float(angleDeg))
    k = np.rint((ys-cy)*np.cos(th)-(xs-cx)*np.sin(th)+cy).astype(np.int64)
    #shift so the lowest row index is 0 and let the profile be as long as it
    #needs to be.  do NOT clip to the page height: clipping piles all the
    #ink that falls past an edge into one row, and because the score squares
    #each row that fake spike scores higher than any real line.  it made the
    #search pick an angle 4 degrees out on a tilted page
    k -= k.min()
    prof = np.bincount(k).astype(np.float64)
    return float((prof**2).sum())


#full page to line crops pipeline, the main entry point for this file
def SegmentLines(path):
    t0 = time.time()
    full = Image.open(path).convert("L")
    #2:1 box average, my own code not a library.  box not point sampling
    #so a thin pale stroke lands in some output pixel instead of vanishing
    halfA = MyImage.Halve(np.asarray(full,dtype=np.uint8))

    #1 deskew: rotate to whichever angle gives the sharpest row profile
    mask,_ = InkMask(halfA.astype(np.float64))
    #score the skew on a quarter size mask, 16x fewer pixels per angle
    q = MyImage.Halve(mask.astype(np.uint8)*255)>127
    best = max(np.arange(-4,4.01,0.5),key=lambda a: SkewScore(q,a))
    #fill 255 because whatever rotates in from outside the photo is paper,
    #fill 0 above because there it is a mask and outside must add no ink
    fullA = MyImage.Rotate(np.asarray(full,dtype=np.uint8),best,fill=255)
    halfA = MyImage.Rotate(halfA,best,fill=255)
    g = halfA.astype(np.float64)
    mask,ratio = InkMask(g)
    H,W = mask.shape

    #2 long thin straight strokes are not handwriting: underlines, rules, binder edge
    P,_ = LinePitch(mask)
    paper = PaperRegionMask(g,P)
    paperRemoved = int((mask&~paper).sum())
    mask&=paper
    flat = AnyWithinRadius(LongRunMask(AnyWithinRadius(mask,1,0),int(1.25*P),1),1,0)&mask   #tolerant of a slight tilt
    upright = AnyWithinRadius(LongRunMask(AnyWithinRadius(mask,3,1),int(1.4*P),0),3,1)&mask   #wobbly vertical page edge too
    removed = int((flat|upright).sum())
    mask = mask&~flat&~upright

    #2b page boundary artifacts: edge slivers, corner shadows, fold creases
    rc = max(1,int(0.1*P))
    dil = AnyWithinRadius(AnyWithinRadius(mask,rc,0),rc,1)
    closed = ~AnyWithinRadius(AnyWithinRadius(~dil,rc,0),rc,1)   #tiny pen lift gaps joined
    lab,nc = LabelConnectedComponents(closed)
    ys,xs = np.nonzero(mask)
    pl = lab[ys,xs]
    area = np.bincount(pl,minlength=nc+1).astype(float)
    top,bot = np.full(nc+1,H),np.zeros(nc+1,int)
    xmin,xmax = np.full(nc+1,W),np.zeros(nc+1,int)
    np.minimum.at(top,pl,ys)
    np.maximum.at(bot,pl,ys)
    np.minimum.at(xmin,pl,xs)
    np.maximum.at(xmax,pl,xs)
    tall = (bot-top+1)>2.2*P
    thin = (xmax-xmin+1)<=0.5*P
    edgeTol = max(3,int(0.5*P))   #this close to the paper boundary counts as page, not text
    touchL,touchR = xmin<=edgeTol,xmax>=W-1-edgeTol
    touchT,touchB = top<=edgeTol,bot>=H-1-edgeTol
    thinV = (bot-top+1)<=0.5*P
    letters = (area>=0.065*P*P)&~tall
    xL = np.percentile(xmin[letters],5)
    vbar = ((xmax-xmin+1)<=0.12*P)&((bot-top+1)>=1.2*P)   #thin and taller than any letter
    otsuT = OtsuThreshold(ratio)
    meanRatio = np.bincount(pl,weights=ratio[ys,xs],minlength=nc+1)/np.maximum(area,1)
    faint = meanRatio>0.85*otsuT   #barely past the ink threshold, real pen strokes are much darker
    faintVbar = faint&((xmax-xmin+1)<=0.2*P)&((bot-top+1)>=0.6*P)   #a printed margin rule or crease

    density = area/np.maximum((bot-top+1)*(xmax-xmin+1),1)
    edgeHeight = (bot-top+1)<=1.2*P
    edgeShadow = (touchT|touchB)&edgeHeight&(density<0.15)&faint&((xmax-xmin+1)>P)   #a crease or fold, not text

    drop = (touchL|(touchR&thin)|((touchT|touchB)&thinV&thin)
            |(tall&thin)|vbar|faintVbar|edgeShadow|(xmax<xL-1.2*P))
    drop[0] = False
    nArtifacts = int(drop.sum())
    mask&=~drop[lab]

    #3 line spacing and centres, recomputed on the cleaned up mask
    P,sm = LinePitch(mask)
    peaks = [i for i in range(1,H-1) if sm[i]>=sm[i-1] and sm[i]>sm[i+1] and sm[i]>0.06*sm.max()]
    centres = []
    for c in sorted(peaks,key=lambda i:-sm[i]):
        if all(abs(c-o)>=0.6*P for o in centres):
            centres.append(c)
    centres.sort()

    #4 rough boundaries: least ink seam between each pair of neighbouring centres
    cuts = [np.full(W,max(0,centres[0]-int(0.75*P)))]
    tailR = max(2,int(0.07*P))
    #closing (dilate then erode), not plain dilation, fills the thin gap a seam could thread
    #through beside a letter stem while leaving the stroke's outer shape close to real, so the
    #cut traces the letter instead of a bloated blob around it
    mdil = AnyWithinRadius(AnyWithinRadius(mask,tailR,0),tailR,1)
    mClosed = ~AnyWithinRadius(AnyWithinRadius(~mdil,tailR,0),tailR,1)
    m = mClosed.astype(np.float64)

    for c0,c1 in zip(centres,centres[1:]):
        yy = np.arange(c0,c1+1)[:, None]
        mid,half_w = c0+0.5*(c1-c0),max(1.0,(c1-c0)/2.0)   #a stray in the gap is usually the line above's tail
        #bigger per column step lets the cut react fast enough for a tall narrow tail, a light
        #pull back to the midline keeps it tracing real ink instead of a generic v shape
        cuts.append(c0+LeastCostSeam(4.0*m[c0:c1+1]+0.012*((yy-mid)/half_w)**2,step=7))

    cuts.append(np.full(W,min(H-1,centres[-1]+int(0.75*P))))
    L = len(cuts)-1

    cutsArr = np.array(cuts)

    #5 pieces owned by the line holding most of their middle band
    ys,xs = np.nonzero(mask)
    pl = lab[ys,xs]
    Y = np.arange(H)[:, None]
    region = sum((Y>=cuts[n][None,:]).astype(np.int16) for n in range(1,L)) if L>1 else np.zeros((H,W),np.int16)
    rp = region[ys,xs]
    ok = (ys>=cutsArr[0][xs])&(ys<cutsArr[-1][xs])
    area = np.bincount(pl,minlength=nc+1).astype(float)

    A = np.bincount(pl,minlength=nc+1).astype(float)
    cx = np.bincount(pl,xs,nc+1)/np.maximum(A,1)
    cy = np.bincount(pl,ys,nc+1)/np.maximum(A,1)
    cnt = np.bincount(pl[ok]*L+rp[ok],minlength=(nc+1)*L).reshape(nc+1,L)
    substantial = cnt>=0.15*np.maximum(cnt.sum(1,keepdims=True),1)   #a real share of this piece's ink
    regionIdx = np.where(substantial,np.arange(L)[None,:],L)
    #when a piece's ink genuinely spans two lines, reserve it for the topmost line it
    #touches, since that's the crop that goes wrong if it loses the piece
    major = np.where(substantial.any(1),regionIdx.min(1),cnt.argmax(1))

    minor = (cnt>0)&(cnt<0.12*P*P)&(cnt<0.35*cnt.sum(1,keepdims=True))&(np.abs(np.arange(L)[None,:]-major[:, None])==1)
    key = pl[ok]*L+rp[ok]
    xlo,xhi = np.full((nc+1)*L,W),np.zeros((nc+1)*L,int)
    np.minimum.at(xlo,key,xs[ok])
    np.maximum.at(xhi,key,xs[ok])
    minor&=((xhi-xlo+1).reshape(nc+1,L)<=0.55*P)   #a cut off tail is narrow, a word wide part is its own line
    big = (cnt.sum(1)>=0.05*P*P)
    tiny = np.zeros(nc+1,bool)
    for l in np.nonzero((cnt.sum(1)>0)&(cnt.sum(1)<0.06*P*P))[0]:   #small stray piece outside its line's band
        n = major[l]
        near = big&(major==n)&(np.abs(cx-cx[l])<=6*P)
        tiny[l] = True
        speck = cnt[l].sum()<0.01*P*P
        if not near.any() or (speck and not (near&(np.abs(cx-cx[l])<=3*P)).any()):
            major[l] = -1   #nowhere near its line's writing: drop
            continue
        yc = np.median(cy[near&(np.abs(cx-cx[l])<=1.5*P)]) if (near&(np.abs(cx-cx[l])<=1.5*P)).any() else np.median(cy[near])
        if cy[l]-yc>0.5*P and n+1<L:
            major[l] = -1 if speck else n+1   #a speck out of band drops, a real mark moves
        elif yc-cy[l]>0.7*P and n>0:
            major[l] = -1 if speck else n-1
    for l in np.nonzero((cnt.sum(1)>=0.06*P*P)&(cnt.sum(1)<0.15*P*P)&(major>=0))[0]:   #small piece to nearest line centre
        pick,bd = major[l],np.inf
        for n in (major[l]-1,major[l],major[l]+1):
            rows = big&(major==n)&(np.abs(cx-cx[l])<=1.5*P)
            if 0<=n<L and rows.any() and abs(np.median(cy[rows])-cy[l])<bd:
                pick,bd = n,abs(np.median(cy[rows])-cy[l])
        major[l],tiny[l] = pick, True
    top2,bot2 = np.full(nc+1,H),np.zeros(nc+1,int)   #a dot goes with the letter body directly below it
    x0b,x1b = np.full(nc+1,W),np.zeros(nc+1,int)
    np.minimum.at(top2,pl,ys)
    np.maximum.at(bot2,pl,ys)
    np.minimum.at(x0b,pl,xs)
    np.maximum.at(x1b,pl,xs)
    hh,ww = bot2-top2+1,x1b-x0b+1
    dot = (A>0)&(A<0.1*P*P)&(np.maximum(hh,ww)<=0.45*P)&(np.maximum(hh,ww)<=2.2*np.maximum(np.minimum(hh,ww),1))
    stem = (A>=0.03*P*P)&~dot
    for l in np.nonzero(dot&(major>=0))[0]:
        xa,xb = max(0,x0b[l]-2),x1b[l]+3
        for gap in range(1,int(0.4*P)+1):
            if bot2[l]+gap>=H:
                break
            row = lab[bot2[l]+gap,xa:xb]
            hit = [v for v in np.unique(row[row>0]) if v!=l and stem[v] and major[v]>=0]
            if hit:
                n2 = major[hit[0]]   #an i dot floats 0.25-0.7 spacings above its line's centre
                rows = big&(major==n2)&(np.abs(cx-cx[l])<=1.5*P)
                if rows.any() and 0.25*P<=np.median(cy[rows])-cy[l]<=0.7*P:
                    major[l],tiny[l] = n2, True
                break
    own = np.full((H,W),-1,np.int16)
    own[ys[ok],xs[ok]] = np.where(tiny[pl]|minor[pl,rp],major[pl],rp)[ok]   #a small cut off part rejoins its main part

    own_before_EF = own.copy()
    #a dot joined to the line above by gap closing (an i dot touching a descender) goes with the stem below it
    raw,nr = LabelConnectedComponents(mask)
    pr = raw[ys,xs]
    Ar = np.bincount(pr,minlength=nr+1).astype(float)
    rt,rb = np.full(nr+1,H),np.zeros(nr+1,int)
    rl_,rr = np.full(nr+1,W),np.zeros(nr+1,int)
    np.minimum.at(rt,pr,ys); np.maximum.at(rb,pr,ys); np.minimum.at(rl_,pr,xs); np.maximum.at(rr,pr,xs)
    h0,w0 = rb-rt+1,rr-rl_+1
    dotr = (Ar>=0.015*P*P)&(Ar<0.1*P*P)&(np.maximum(h0,w0)<=0.5*P)&(np.maximum(h0,w0)<=2.2*np.maximum(np.minimum(h0,w0),1))
    stemr = (Ar>=0.03*P*P)&~dotr
    for d in np.nonzero(dotr)[0]:
        for gap in range(1,int(0.4*P)+1):
            if rb[d]+gap>=H:
                break
            row = raw[rb[d]+gap,max(0,rl_[d]-2):rr[d]+3]
            hit = [v for v in np.unique(row[row>0]) if v!=d and stemr[v]]
            if hit:
                n2 = region[rb[d]+gap,max(0,rl_[d]-2)+np.nonzero(row==hit[0])[0][0]]   #line of the stem where it meets the dot
                dv = pr==d
                ov = own[ys[dv],xs[dv]]
                if (ov>=0).any():
                    n1 = np.bincount(ov[ov>=0],minlength=L).argmax()
                    closedPiece = lab[ys[dv][0],xs[dv][0]]   #a genuine i dot stays its own small piece even after closing
                    wholeWord = area[closedPiece]>=0.2*P*P
                    if n2==n1+1 and not wholeWord and (rt[d]+rb[d])/2>=centres[n1]+0.45*P:
                        keep = ov==n1   #a letter that broke apart in the raw mask, not a floating dot
                        own[ys[dv][keep],xs[dv][keep]] = n2
                break

    #a ring shaped blob at the bottom of a stroke (an i dot touching the descender above) goes with the stem below it
    colbot = np.full((nr+1,W),-1,np.int32)
    np.maximum.at(colbot,(pr,xs),ys)
    wr = int(0.35*P)
    for c in range(1,nr+1):
        cols = np.nonzero(colbot[c]>=0)[0]
        if len(cols)==0 or Ar[c]<0.015*P*P:
            continue
        order = cols[np.argsort(-colbot[c][cols])]
        taken = np.zeros(W,bool)
        for xb in order[:60]:
            if taken[xb]:
                continue
            taken[max(0,xb-wr):xb+wr+1] = True
            yb = colbot[c][xb]
            xa,xe = max(0,xb-wr),min(W,xb+wr+1)
            widest,rt = 0, None
            for r in range(yb,max(0,yb-int(0.5*P)),-1):   #walk up while the blob stays wide, stem is where it narrows
                cc = np.nonzero(raw[r,xa:xe]==c)[0]
                wd = (cc.max()-cc.min()+1) if len(cc) else 0
                widest = max(widest,wd)
                if wd<0.45*widest:
                    rt = r
                    break
            else:
                rt = max(0,yb-int(0.5*P))
            h = yb-rt
            if not (0.2*P<=widest<=0.45*P and 0.6<=h/max(widest,1)<=1.6) or yb-rt>=0.5*P:
                continue
            blob = (raw[rt+1:yb+1,xa:xe]==c)
            bc = np.nonzero(blob.any(0))[0]
            xl,xr = xa+bc.min(),xa+bc.max()
            if LabelConnectedComponents(~np.pad(blob[:,bc.min():bc.max()+1],1))[1]<2:   #a dot ring encloses a hole, a hook does not
                continue
            n = region[yb,xb]
            for gap in range(1,int(0.4*P)+1):
                if yb+gap>=H:
                    break
                row = raw[yb+gap,max(0,xl-2):xr+3]
                hit = np.nonzero((row>0)&(row!=c))[0]
                if len(hit):
                    ty,tx = yb+gap,max(0,xl-2)+hit[0]
                    n2 = region[ty,tx]
                    wholeWord = area[lab[yb,xb]]>=0.2*P*P   #a real ring letter like "o" in a word, not a floating dot
                    if n2==n+1 and not wholeWord and ty>=centres[n2]-0.45*P:
                        sub = own[rt+1:yb+1,xl:xr+1]
                        sub[blob[:,xl-xa:xr-xa+1]&(sub==n)] = n2
                    break

    #a fragment with no ink in its own line's x height band is not that line's letter: it goes
    #back to the line whose ink it touches, or to a neighbour only if it sits in that line's band
    own0,moves = own.copy(),[]
    locked = own!=own_before_EF   #pixels already placed by the two steps above
    for n in range(L):
        comp,nb = LabelConnectedComponents((own0==n)&mask)
        if nb==0:
            continue
        yy,xx = np.nonzero(comp)
        pc = comp[yy,xx]
        inband = np.bincount(pc,weights=(np.abs(yy-centres[n])<=0.28*P).astype(float),minlength=nb+1)
        size = np.bincount(pc,minlength=nb+1)
        for c in np.nonzero((inband==0)&(size>=0.006*P*P))[0]:
            if c==0:
                continue
            fy,fx = yy[pc==c],xx[pc==c]
            if locked[fy,fx].any() and len(fy)<0.04*P*P and max(np.ptp(fy),np.ptp(fx))+1<=0.4*P:
                continue   #a dot placed above, not a tail loop
            ya,yb,xa,xb = max(0,fy.min()-3),min(H,fy.max()+4),max(0,fx.min()-3),min(W,fx.max()+4)
            frag = np.zeros((yb-ya,xb-xa),bool)
            frag[fy-ya,fx-xa] = True
            near = AnyWithinRadius(AnyWithinRadius(frag,2,0),2,1)&mask[ya:yb,xa:xb]&~frag
            ov = own0[ya:yb,xa:xb][near]
            cnt_ = np.bincount(ov[(ov>=0)&(ov!=n)],minlength=L)
            cy_ = fy.mean()
            if cnt_.max()>=3:
                m = int(cnt_.argmax())
                if abs(m-n)==1 and (m>n)==(cy_>centres[n]):
                    moves.append((fy,fx,m))
            else:
                best_m = min((m for m in (n-1,n+1) if 0<=m<L),key=lambda m: abs(cy_-centres[m]),default=None)
                if best_m is not None and abs(cy_-centres[best_m])<=0.2*P:   #sits in the other line's x height band
                    moves.append((fy,fx,best_m))
    for fy,fx,m in moves:
        own[fy,fx] = m

    #6 crops from the full resolution, lighting normalised page
    r = max(8,min(40,round(H/41)))   #same scale aware radius as InkMask
    #the background was measured at half size, stretch it back up so the
    #full resolution page can be divided by it
    bg_full = MyImage.Resize(BoxMean(g,r).astype(np.float32),fullA.shape[1],fullA.shape[0],'linear')
    norm = np.clip(fullA.astype(np.float32)/np.maximum(bg_full,1.0),0,1)*255
    f = fullA.shape[1]/W
    perLine = np.bincount(own[own>=0],minlength=L)
    minInk = max(60,0.12*np.median(perLine[perLine>0]))
    lines = []
    for n in range(L):
        ink = own==n
        yy,xx = np.nonzero(ink)
        if len(yy)<minInk or yy.max()-yy.min()<max(6,0.2*P):   #a short lowercase word is still a real line
            continue
        rowInk = np.bincount(yy,minlength=H)>0
        gapAt = np.flatnonzero(~rowInk[yy.min():yy.max()+1])+yy.min()
        if len(gapAt):   #keep only the tallest contiguous run of real ink, drop stray marks far away
            runs,s = [],yy.min()
            for g in gapAt:
                if g>s:
                    runs.append((s,g-1))
                s = g+1
            runs.append((s,yy.max()))
            runs = [(a,b) for a,b in runs if b>=a]
            mainRun = max(runs,key=lambda ab: int(ink[ab[0]:ab[1]+1].sum()))
            if mainRun[1]-mainRun[0]+1<yy.max()-yy.min()+1-0.8*P:   #only trim a real paragraph sized gap
                keep = (yy>=mainRun[0])&(yy<=mainRun[1])
                ink = np.zeros_like(ink)
                ink[yy[keep],xx[keep]] = True
                yy,xx = yy[keep],xx[keep]
        pad = int(0.15*P)
        y0,y1 = max(0,yy.min()-pad),min(H,yy.max()+1+pad)
        x0,x1 = max(0,xx.min()-pad),min(W,xx.max()+1+pad)
        Y0,Y1,X0,X1 = int(y0*f),int(y1*f),int(x0*f),int(x1*f)
        inkC = ink[y0:y1,x0:x1]
        #blow the half size mask up to the crop size.  bilinear on purpose,
        #the soft edge it gives is what makes the stroke edge look natural
        alpha = MyImage.Resize((inkC*255).astype(np.uint8),X1-X0,Y1-Y0,'linear').astype(np.float32)/255.0
        haloR = max(1,int(0.20*P))   #near this line's own ink only, bridges a letter the threshold missed
        halo = AnyWithinRadius(AnyWithinRadius(inkC,haloR,0),haloR,1)
        haloAlpha = MyImage.Resize((halo*255).astype(np.uint8),X1-X0,Y1-Y0,'linear').astype(np.float32)/255.0
        region = norm[Y0:Y1,X0:X1]
        localR = max(3,min(20,round(region.shape[0]/8)))
        localRatio = region/np.maximum(BoxMean(region,localR),1.0)   #this line's own contrast, not the whole page's
        localMask = localRatio<OtsuThreshold(localRatio)
        ownC = own[y0:y1,x0:x1]
        notOther = ~((ownC>=0)&(ownC!=n))   #never recover a pixel the page already gave to a neighbour
        #nearest here, not bilinear: this is a yes/no ownership flag and a
        #blurred edge would half give a pixel to two lines at once
        notOtherFull = MyImage.Resize((notOther*255).astype(np.uint8),X1-X0,Y1-Y0,'nearest')>127
        alpha = np.maximum(alpha,localMask*haloAlpha*notOtherFull)
        inkPix = region[alpha>0.5]
        black = np.percentile(inkPix,40) if len(inkPix) else 0.0   #typical ink darkness, not just the darkest pixel
        white = np.percentile(region,75)

        linear = np.clip((region-black)*(255.0/max(white-black,1.0)),0,255)   #contrast stretch
        darkness = np.clip((255.0-linear)/255.0,0,1)**0.55   #gamma<1 pulls faint strokes much darker
        enhanced = 255.0*(1.0-darkness)
        crop = (alpha*enhanced+(1-alpha)*255).astype(np.uint8)
        margin = int(pad*f)   #same left gap every crop starts with
        firstCol = int(np.argmax((alpha>0.1).any(axis=0))) if (alpha>0.1).any() else margin
        shift = firstCol-margin
        if shift>0:
            crop = np.pad(crop[:,shift:],((0,0),(0,shift)),constant_values=255)
        img = Image.fromarray(crop)
        lines.append(dict(order=len(lines),image=img,clf=ToClassifierFormat(img),bbox=(X0,Y0,X1,Y1)))

    #debug overlay: red tint outside the paper region, alternating cut colours
    ov = Image.fromarray(np.clip(ratio*255,0,255).astype(np.uint8)).convert("RGB")
    shade = np.asarray(ov).astype(np.int16)
    shade[~paper] = (shade[~paper]*0.5+np.array([120,0,0])).astype(np.int16)
    ov = Image.fromarray(np.clip(shade,0,255).astype(np.uint8))
    d = ImageDraw.Draw(ov)
    for n,cu in enumerate(cuts):
        d.line(list(zip(range(W),cu.tolist())),fill=(255,0,0) if n%2 else (0,140,255),width=2)
    return lines,ov,dict(angle=float(best),pitch=int(P),paperCoverage=round(float(paper.mean()),3),inkOutsidePaper=paperRemoved,removedStraightPixels=removed,artifactPieces=nArtifacts,
                           nLines=len(lines),seconds=round(time.time()-t0,1))


if __name__=="__main__":
    src,out = Path(sys.argv[1]),Path(sys.argv[2])
    out.mkdir(parents=True,exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    lines,ov,info = SegmentLines(src)
    (out/"classifier").mkdir(exist_ok=True)
    for ln in lines:
        ln["image"].save(out/f"{ln['order']:02d}.png")   #native resolution crop
        ln["clf"].save(out/"classifier"/f"{ln['order']:02d}.png")   #640x64 classifier input
    ov.save(out/"_cuts_overlay.jpg",quality=88)
    print(info)
