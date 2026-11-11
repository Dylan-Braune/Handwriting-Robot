#u23539195 D.R. Braune.  the image toolkit, and page to line segmentation
#one photographed page in, deskewed line crops out
#arrays in, arrays out, nothing here opens a file
import numpy as np

InputHeight,InputWidth = 64,640   #the crop size the classifiers take

#----------------------------------------------------------------------
#Integral: running total of the image, cell (y,x) holds everything above
#and left of it, so any rectangle sum is four lookups
#----------------------------------------------------------------------
def Integral(img):
    ii = np.zeros((img.shape[0]+1,img.shape[1]+1),np.float64)
    np.cumsum(np.cumsum(img,axis=0),axis=1,out=ii[1:,1:])
    return ii

#---BoxSum: the window sum around every pixel at once, from that running total---
def BoxSum(img,ry,rx):
    h,w = img.shape
    ii = Integral(img)
    y,x = np.arange(h),np.arange(w)
    y1,y2 = np.clip(y-ry,0,h)[:,None],np.clip(y+ry+1,0,h)[:,None]
    x1,x2 = np.clip(x-rx,0,w)[None,:],np.clip(x+rx+1,0,w)[None,:]
    return ii[y2,x2]-ii[y1,x2]-ii[y2,x1]+ii[y1,x1]   #BR - TR - BL + TL

#---BoxCount: how many pixels each window covered, which is smaller at the edges---
def BoxCount(ry,rx,h,w):
    y,x = np.arange(h),np.arange(w)
    cy = (np.clip(y+ry+1,0,h)-np.clip(y-ry,0,h))[:,None]
    cx = (np.clip(x+rx+1,0,w)-np.clip(x-rx,0,w))[None,:]
    return cy*cx

#---BoxMean: the mean value in a window around every pixel, used as a local background---
def BoxMean(a,r):
    p = np.pad(a,r,mode="edge").astype(np.float64)   #edge repeated, so every window is full
    ii = Integral(p)
    k = 2*r+1
    return (ii[k:,k:]-ii[:-k,k:]-ii[k:,:-k]+ii[:-k,:-k])/(k*k)

#---Dilate: grows every shape outwards, a pixel turns on if any set pixel is in range---
def Dilate(mask,ky,kx):
    return BoxSum(mask.astype(np.float64),ky//2,kx//2)>0

#---Erode: shrinks every shape inwards, a pixel stays on only if the window is full---
def Erode(mask,ky,kx):
    ry,rx = ky//2,kx//2
    return BoxSum(mask.astype(np.float64),ry,rx)>=BoxCount(ry,rx,*mask.shape)-0.5

#---Close: dilate then erode by the same radius, bridging gaps without fattening---
def Close(mask,ky,kx):
    return Erode(Dilate(mask,ky,kx),ky,kx)

#---AnyWithinRadius: dilates along one axis only, true where anything is set within r places---
def AnyWithinRadius(m,r,axis):
    a = np.moveaxis(m.astype(np.int32),axis,0)
    n = a.shape[0]
    c = np.cumsum(np.pad(a,[(r+1,r)]+[(0,0)]*(a.ndim-1)),0)
    return np.moveaxis((c[2*r+1:2*r+1+n]-c[:n])>0,0,axis)

#---LongRunMask: keeps only ink lying on an unbroken run of at least n pixels along one axis---
def LongRunMask(m,n,axis):
    a = np.moveaxis(m.astype(np.int32),axis,0)
    L = a.shape[0]
    if L<n:
        return np.zeros(m.shape,bool)
    c = np.cumsum(np.pad(a,((1,0),(0,0))),0)
    hit = ((c[n:]-c[:-n])==n).astype(np.int32)   #a run of n starts at i when c[i+n]-c[i]==n
    ch = np.cumsum(np.pad(hit,((1,0),(0,0))),0)   #a second sum turns starts into coverage
    j = np.arange(L)
    lo,hi = np.maximum(0,j-n+1),np.minimum(j,L-n)
    return np.moveaxis(((ch[hi+1]-ch[lo])>0)&(hi>=lo)[:,None],0,axis)

#---RowRunLength: for every pixel, the length of the unbroken run of ink through it along its row---
def RowRunLength(mask):
    h,w = mask.shape
    a = mask.astype(np.int32)
    starts = np.zeros_like(a)
    starts[:,0] = a[:,0]
    starts[:,1:] = (a[:,1:]==1)&(a[:,:-1]==0)   #a run starts where the row steps 0 to 1
    runId = np.cumsum(starts.reshape(-1)).reshape(h,w)
    runId[a==0] = 0
    if runId.max()==0:
        return np.zeros_like(a)
    counts = np.bincount(runId.reshape(-1))
    counts[0] = 0
    return counts[runId]

#---Otsu: the threshold maximising between-class variance over the histogram---
def Otsu(v,hi=None):
    a = np.asarray(v)
    isInt = np.issubdtype(a.dtype,np.integer)
    if isInt:
        hist = np.bincount(a.ravel(),minlength=256).astype(np.float64)
    else:
        hi = 1.2 if hi is None else float(hi)
        hist = np.histogram(np.clip(a,0,hi),bins=256,range=(0,hi))[0].astype(np.float64)
    p = hist/max(hist.sum(),1.0)
    w = np.cumsum(p)   #mass at or below each bin
    m = np.cumsum(p*np.arange(256))
    var = (m[-1]*w-m)**2/np.maximum(w*(1-w),1e-12)
    t = int(np.argmax(var))
    return t if isInt else (t+0.5)*hi/256   #bin index for ints, bin centre for floats

#----------------------------------------------------------------------
#BinarizeLine: a grey line crop to a clean ink mask, threshold, close pinholes, drop
#specks, and remove long thin runs, which are printed rules
#----------------------------------------------------------------------
def BinarizeLine(gray,stripRules=True,closeGaps=True):
    ink = np.asarray(gray)<Otsu(np.asarray(gray,dtype=np.uint8))
    h = ink.shape[0]
    if stripRules and ink.any():
        hRun = RowRunLength(ink)
        vRun = RowRunLength(ink.T).T
        thin = vRun<=max(3,int(0.12*h))
        rule = (hRun>=max(24,int(1.1*h)))&thin   #long and thin
        if rule.any():
            ink = ink&~(rule&thin)
    if closeGaps and ink.any():
        k = int(np.clip(round(0.045*h),2,4))
        ink = Close(ink,k,k)
    labels,n = LabelComponents(ink)
    if n:
        areas = np.bincount(labels.ravel())
        keep = areas>=6
        keep[0] = False
        ink = keep[labels]
    return ink

#---CoreBand: the band of rows holding most of the ink, which is the x-height band---
def CoreBand(ink):
    prof = ink.sum(axis=1).astype(np.float64)
    if prof.max()<=0:
        return None
    rows = np.nonzero(prof>=0.45*prof.max())[0]
    top,base = int(rows[0]),int(rows[-1])
    return (top,base) if base-top>=3 else None

#----------------------------------------------------------------------
#LabelComponents: numbers every eight-connected blob of ink, row by row, merging the runs
#that touch the row above through a union-find
#----------------------------------------------------------------------
def LabelComponents(mask):
    H,W = mask.shape
    lab = np.zeros((H,W),np.int32)
    parent = [0]

    def Find(a):
        while parent[a]!=a:
            parent[a] = parent[parent[a]]   #flatten the chain while walking it
            a = parent[a]
        return a

    prev = []
    for y in range(H):
        d = np.diff(np.r_[0,mask[y].view(np.int8),0])   #runs start at 0 to 1, end at 1 to 0
        cur,k = [],0
        for a,b in zip(np.nonzero(d==1)[0].tolist(),np.nonzero(d==-1)[0].tolist()):
            while k<len(prev) and prev[k][1]<a:
                k+=1
            l,kk = 0,k
            while kk<len(prev) and prev[kk][0]<=b:   #<=b counts a diagonal touch
                r = Find(prev[kk][2])
                if l==0:
                    l = r
                elif r!=l:
                    parent[max(r,l)] = min(r,l)
                    l = min(r,l)
                kk+=1
            if l==0:
                parent.append(len(parent))
                l = len(parent)-1
            cur.append((a,b,l))
            lab[y,a:b] = l
        prev = cur
    roots = np.array([Find(i) for i in range(len(parent))])
    _,comp = np.unique(roots,return_inverse=True)   #renumber 1..n after the merges
    return comp[lab].astype(np.int32),int(comp.max())

#---PieceBounds: the bounding box and pixel count of every numbered piece, in one pass---
def PieceBounds(labelAt,ys,xs,n,H,W):
    top,bot = np.full(n+1,H),np.zeros(n+1,int)
    left,right = np.full(n+1,W),np.zeros(n+1,int)
    np.minimum.at(top,labelAt,ys)
    np.maximum.at(bot,labelAt,ys)
    np.minimum.at(left,labelAt,xs)
    np.maximum.at(right,labelAt,xs)
    area = np.bincount(labelAt,minlength=n+1).astype(float)
    return top,bot,left,right,area

#----------------------------------------------------------------------
#Skeletonize: zhang-suen thinning, repeatedly deletes edge pixels not needed to
#keep the shape connected, leaving a one-pixel-wide centreline
#----------------------------------------------------------------------
def Skeletonize(mask):
    img = mask.astype(np.uint8).copy()

    def Ring(P):
        p = np.pad(P,1)   #the eight neighbours of every pixel, clockwise from north
        return (p[:-2,1:-1],p[:-2,2:],p[1:-1,2:],p[2:,2:],
                p[2:,1:-1],p[2:,:-2],p[1:-1,:-2],p[:-2,:-2])

    changed = True
    while changed:
        changed = False
        for step in (0,1):   #two sub-passes, alternating corners
            P2,P3,P4,P5,P6,P7,P8,P9 = Ring(img)
            r = [P2,P3,P4,P5,P6,P7,P8,P9,P2]
            B = sum(r[:8])   #how many neighbours are set
            A = sum(((r[k]==0)&(r[k+1]==1)).astype(np.uint8) for k in range(8))   #0 to 1 steps round the ring
            if step==0:
                c1,c2 = (P2*P4*P6)==0,(P4*P6*P8)==0
            else:
                c1,c2 = (P2*P4*P8)==0,(P2*P6*P8)==0
            go = (img==1)&(B>=2)&(B<=6)&(A==1)&c1&c2   #a==1 means deleting it cannot split the stroke
            if go.any():
                img[go] = 0
                changed = True
    return img.astype(bool)

#----------------------------------------------------------------------
#_Weights: the resampling weights for one axis, the filter widens when shrinking so
#every input pixel still contributes
#----------------------------------------------------------------------
def _Weights(inSize,outSize,kernel):
    scale = inSize/float(outSize)
    filterScale = max(1.0,scale)
    support = (1.0 if kernel=='linear' else 0.5)*filterScale
    centres = (np.arange(outSize)+0.5)*scale   #output centre, in input coordinates
    half = int(np.ceil(support))+1
    cols = np.floor(centres)[:,None]+np.arange(-half,half+1)[None,:]
    t = (cols+0.5-centres[:,None])/filterScale   #distance from the window centre
    if kernel=='linear':
        w = np.where(np.abs(t)<1.0,1.0-np.abs(t),0.0)   #triangle
    else:
        w = np.where(np.abs(t)<=0.5,1.0,0.0)   #box
    cc = np.clip(cols,0,inSize-1).astype(np.int64)   #fold a window running off the image back on
    out = np.zeros((outSize,inSize),np.float64)
    np.add.at(out,(np.repeat(np.arange(outSize),cols.shape[1]),cc.ravel()),w.ravel())
    tot = out.sum(axis=1,keepdims=True)
    return out/np.where(tot>0,tot,1.0)   #each row sums to one

#---Resize: scales an image by applying those weights along one axis then the other---
def Resize(arr,newW,newH,kernel='linear'):
    a = np.asarray(arr)
    newW,newH = max(1,int(newW)),max(1,int(newH))
    h,w = a.shape[0],a.shape[1]
    if h==newH and w==newW:
        return a.copy()
    if kernel=='nearest':   #for label maps, where an averaged value exists nowhere
        sy = np.clip(((np.arange(newH)+0.5)*(h/float(newH))).astype(np.int64),0,h-1)
        sx = np.clip(((np.arange(newW)+0.5)*(w/float(newW))).astype(np.int64),0,w-1)
        return a[sy[:,None],sx[None,:]]
    f = a.astype(np.float64)
    if newH!=h:
        f = np.tensordot(_Weights(h,newH,kernel),f,axes=([1],[0]))
    if newW!=w:
        f = np.tensordot(f,_Weights(w,newW,kernel).T,axes=([1],[0]))
        if f.ndim==3:
            f = np.moveaxis(f,-1,1)   #tensordot left width last, put it back
    if np.issubdtype(a.dtype,np.integer):
        i = np.iinfo(a.dtype)
        return np.clip(np.rint(f),i.min,i.max).astype(a.dtype)
    return f.astype(a.dtype)

#---Halve: exact two-to-one box average---
def Halve(a):
    a = np.asarray(a)
    return Resize(a,a.shape[1]//2,a.shape[0]//2,'box')

#----------------------------------------------------------------------
#FitFixedCanvas: puts a crop on the fixed classifier canvas, scaled by one factor for both
#axes and padded with paper
#----------------------------------------------------------------------
def FitFixedCanvas(gray,height,width,padValue=255):
    a = np.asarray(gray)
    h,w = a.shape[0],a.shape[1]
    canvas = np.full((height,width),padValue,np.uint8)
    if w<=0 or h<=0:
        return canvas
    s = min(height/float(h),width/float(w))
    newW,newH = max(1,int(round(w*s))),max(1,int(round(h*s)))
    small = Resize(a.astype(np.uint8),newW,newH,'linear')
    y0 = max(0,(height-newH)//2)   #vertically centred, left flush
    canvas[y0:y0+newH,:newW] = small[:height-y0,:width]
    return canvas

#---_Sample: reads the image at fractional positions, mixing the four neighbours---
def _Sample(a,sx,sy,fill,edge='area'):
    h,w = a.shape[0],a.shape[1]
    x0,y0 = np.floor(sx).astype(np.int64),np.floor(sy).astype(np.int64)
    fx,fy = sx-x0,sy-y0
    if edge=='area':
        inside = (sx>-0.5)&(sx<w-0.5)&(sy>-0.5)&(sy<h-0.5)   #pixel i covers i-0.5 to i+0.5
    else:
        inside = (sx>=0)&(sx<=w-1)&(sy>=0)&(sy<=h-1)
    xi,yi = np.clip(x0,0,w-1),np.clip(y0,0,h-1)
    xj,yj = np.clip(x0+1,0,w-1),np.clip(y0+1,0,h-1)
    top = a[yi,xi]*(1-fx)+a[yi,xj]*fx
    bot = a[yj,xi]*(1-fx)+a[yj,xj]*fx
    return np.where(inside,top*(1-fy)+bot*fy,float(fill))

#---_Like: rounds a float result back to the input dtype---
def _Like(out,like):
    if np.issubdtype(np.asarray(like).dtype,np.integer):
        return np.clip(np.rint(out),0,255).astype(np.uint8)
    return out

#---Rotate: turns the image about its centre, reading backwards from each output pixel---
def Rotate(gray,deg,fill=255):
    a = np.asarray(gray).astype(np.float64)
    h,w = a.shape[0],a.shape[1]
    th = np.radians(float(deg))
    c,s = np.cos(th),np.sin(th)
    cy,cx = (h-1)/2.0,(w-1)/2.0
    yy,xx = np.meshgrid(np.arange(h,dtype=np.float64),
                        np.arange(w,dtype=np.float64),indexing='ij')
    dx,dy = xx-cx,yy-cy
    #rows run downward, so this is the transpose of the textbook matrix
    return _Like(_Sample(a,c*dx-s*dy+cx,s*dx+c*dy+cy,fill),gray)

#---Warp: affine with six coefficients or perspective with eight, same backward read---
def Warp(gray,coeffs,fill=255):
    a = np.asarray(gray).astype(np.float64)
    h,w = a.shape[0],a.shape[1]
    yy,xx = np.meshgrid(np.arange(h,dtype=np.float64),
                        np.arange(w,dtype=np.float64),indexing='ij')
    X,Y = xx+0.5,yy+0.5   #the coefficients act on the pixel centre
    k = [float(v) for v in coeffs]
    if len(k)==6:
        sx,sy = k[0]*X+k[1]*Y+k[2]-0.5,k[3]*X+k[4]*Y+k[5]-0.5
    elif len(k)==8:
        den = k[6]*X+k[7]*Y+1.0
        den = np.where(np.abs(den)<1e-12,1e-12,den)
        sx,sy = (k[0]*X+k[1]*Y+k[2])/den-0.5,(k[3]*X+k[4]*Y+k[5])/den-0.5
    else:
        raise ValueError("Warp wants 6 coefficients for affine or 8 for perspective")
    return _Like(_Sample(a,sx,sy,fill,edge='centre'),gray)

#---Blur: gaussian blur, run as a one-dimensional kernel down then across---
def Blur(gray,sigma,mode='edge'):
    a = np.asarray(gray).astype(np.float64)
    s = float(sigma)
    if s<=0.0:
        return np.asarray(gray).copy()
    r = max(1,int(np.ceil(3.0*s)))   #cut off at three sigma
    t = np.arange(-r,r+1,dtype=np.float64)
    k = np.exp(-(t*t)/(2.0*s*s))
    k /= k.sum()
    pk = dict(edge='edge',constant='constant')[mode]
    pad = np.pad(a,((r,r),(r,r)),mode=pk)
    tmp = np.zeros_like(a)
    for i,wgt in enumerate(k):
        tmp += wgt*pad[i:i+a.shape[0],r:r+a.shape[1]]
    pad2 = np.pad(tmp,((0,0),(r,r)),mode=pk)
    out = np.zeros_like(a)
    for i,wgt in enumerate(k):
        out += wgt*pad2[:,i:i+a.shape[1]]
    return _Like(out,gray)

#----------------------------------------------------------------------
#Elastic: bends strokes locally by reading the image at positions moved by two
#smooth random fields
#----------------------------------------------------------------------
def Elastic(gray,sigma,alpha,rng):
    a = np.asarray(gray).astype(np.float64)
    h,w = a.shape
    dx = Blur(rng.uniform(-1,1,(h,w)),sigma,mode='constant')*alpha
    dy = Blur(rng.uniform(-1,1,(h,w)),sigma,mode='constant')*alpha
    yy,xx = np.meshgrid(np.arange(h,dtype=np.float64),
                        np.arange(w,dtype=np.float64),indexing='ij')
    return _Like(_Sample(a,xx+dx,yy+dy,255,edge='centre'),gray)

#----------------------------------------------------------------------
#DrawPolyline: draws a polyline with a round pen, filling where the distance to the
#nearest point on a segment is within half the pen width
#----------------------------------------------------------------------
def DrawPolyline(img,pts,value,width):
    if len(pts)==0:
        return img
    h,w = img.shape[0],img.shape[1]
    r = max(0.5,float(width)/2.0)
    val = np.uint8(np.clip(round(float(value)),0,255))
    segs = [(pts[i],pts[i+1]) for i in range(len(pts)-1)] or [(pts[0],pts[0])]
    for (ax,ay),(bx,by) in segs:
        x0,x1 = max(0,int(min(ax,bx)-r)-1),min(w,int(max(ax,bx)+r)+2)
        y0,y1 = max(0,int(min(ay,by)-r)-1),min(h,int(max(ay,by)+r)+2)
        if x1<=x0 or y1<=y0:
            continue
        yy,xx = np.meshgrid(np.arange(y0,y1,dtype=np.float64),
                            np.arange(x0,x1,dtype=np.float64),indexing='ij')
        vx,vy = bx-ax,by-ay
        L2 = vx*vx+vy*vy
        if L2<=1e-12:
            d = np.hypot(xx-ax,yy-ay)
        else:
            t = np.clip(((xx-ax)*vx+(yy-ay)*vy)/L2,0.0,1.0)   #how far along the segment
            d = np.hypot(xx-(ax+t*vx),yy-(ay+t*vy))
        img[y0:y1,x0:x1] = np.where(d<=r,val,img[y0:y1,x0:x1])
    return img

#----------------------------------------------------------------------
#EstimateSlantDeg: shears the ink by each trial angle and keeps the angle whose column
#profile is sharpest
#----------------------------------------------------------------------
def EstimateSlantDeg(ink):
    ys,xs = np.nonzero(ink)
    if len(xs)<50:
        return 0.0
    yc = ys.mean()
    best,bestScore = 0.0,-1.0
    for deg in np.arange(-45,45.5,1.5):
        xsh = (xs+(ys-yc)*-np.tan(np.radians(deg))).astype(np.int64)
        prof = np.bincount(xsh-xsh.min()).astype(np.float64)
        score = float((prof**2).sum())
        if score>bestScore:
            bestScore,best = score,float(deg)
    return -best   #positive is a forward lean, so deslanting is x -= tan(slant)*y

#---PolylineLength: total distance walked along a list of points---
def PolylineLength(p):
    a = np.asarray(p,dtype=np.float64)
    if len(a)<2:
        return 0.0
    return float(np.hypot(*np.diff(a,axis=0).T).sum())

#---InkMask: which pixels are ink, by comparing each to its own blurred background---
def InkMask(gray):
    r = max(8,min(40,round(gray.shape[0]/41)))   #background radius scales with the image
    ratio = gray/np.maximum(BoxMean(gray,r),1.0)
    mask = ratio<Otsu(ratio)
    solid = BoxMean(mask.astype(np.float64),4)>0.999   #a solid 9x9 block is not writing
    return mask&~(BoxMean(solid.astype(np.float64),3)>0),ratio

#----------------------------------------------------------------------
#PaperRegionMask: the sheet inside the photograph, the biggest bright blob, found on a
#block-averaged copy and blown back up
#----------------------------------------------------------------------
def PaperRegionMask(g,P):
    H,W = g.shape
    s = 4
    c = BoxMean(g[:H//s*s,:W//s*s].reshape(H//s,s,W//s,s).mean((1,3)),max(1,int(0.15*P/s)))
    dark = c<0.45*np.median(c)
    if dark.mean()<0.002:
        return np.ones((H,W),bool)
    lab,n = LabelComponents(~dark)
    big = np.kron(lab==np.bincount(lab.ravel())[1:].argmax()+1,np.ones((s,s),bool))
    full = np.ones((H,W),bool)
    full[:big.shape[0],:big.shape[1]] = big
    r = int(0.4*P)
    return ~AnyWithinRadius(AnyWithinRadius(~full,r,0),r,1)   #shrink in from the paper edge

#the smallest gap between profile peaks that counts as line spacing, in
#letter heights
PEAK_MIN_PITCH_XH = 1.5

#the lowest pitch that may be reported, in half-resolution pixels
MIN_LINE_PITCH = 15

#how shallow the dip between two peaks must be before they merge into one
#line, as a fraction of the smaller peak, 1.0 disables the merge
LINE_MERGE_VALLEY = 1.0

#---_LetterHeight: the median height of the ink blobs, ignoring specks and page-tall pieces---
def _LetterHeight(mask):
    lab,nc = LabelComponents(mask)
    if nc<=0:
        return None
    ys,xs = np.nonzero(mask)
    pl = lab[ys,xs]
    top,bot = np.full(nc+1,mask.shape[0]),np.zeros(nc+1,int)
    np.minimum.at(top,pl,ys)
    np.maximum.at(bot,pl,ys)
    area = np.bincount(pl,minlength=nc+1)
    hts = bot-top+1
    sel = (area>=8)&(hts>=3)&(hts<=mask.shape[0])
    if not sel[1:].any():
        return None
    return float(np.median(hts[1:][sel[1:]]))

#----------------------------------------------------------------------
#LinePitch: the spacing between lines, the median gap between peaks of the smoothed
#ink-per-row profile, with autocorrelation and letter height as fallbacks
#----------------------------------------------------------------------
def LinePitch(mask):
    W = mask.shape[1]
    prof = mask[:,int(0.05*W):int(0.97*W)].sum(1).astype(float)
    k = np.exp(-0.5*(np.arange(-12,13)/4.0)**2)
    sm = np.convolve(prof,k/k.sum(),"same")

    h = _LetterHeight(mask)
    floor = (PEAK_MIN_PITCH_XH*h) if h else 0.0   #a gap below this is within one line

    peaks = [i for i in range(1,len(sm)-1) if sm[i]>=sm[i-1] and sm[i]>sm[i+1] and sm[i]>0.06*sm.max()]
    if len(peaks)>=2:
        p = int(np.median(np.diff(sorted(peaks))))
        if p>=floor:
            return p,sm

    a = sm-sm.mean()
    ac = np.correlate(a,a,"full")[len(a)-1:]
    seg = ac[15:min(200,len(ac)//3)]
    idx = next((i for i in range(1,len(seg)-1)
                if seg[i]>=0.7*seg.max() and seg[i]>=seg[i-1] and seg[i]>=seg[i+1]), None)
    if idx is not None and 15+idx>=floor:
        return 15+idx,sm

    if h:
        return max(MIN_LINE_PITCH,int(round(2.2*h))),sm
    return 48,sm

#----------------------------------------------------------------------
#LeastCostSeam: the cheapest path left to right across a cost field, keeping the best
#running cost of reaching each row and backtracking at the end
#----------------------------------------------------------------------
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
                cand[i,-d:] = p[:d]+0.02*abs(d)   #a small cost for moving off straight ahead
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
        y[x-1] = y[x]+back[y[x],x]
    return y

#----------------------------------------------------------------------
#SkewScore: the sum of the squared ink per row at a trial angle, which peaks when the
#page is level
#----------------------------------------------------------------------
def SkewScore(mask,angleDeg):
    ys,xs = np.nonzero(mask)
    if len(ys)==0:
        return 0.0
    h = mask.shape[0]
    cy,cx = (h-1)/2.0,(mask.shape[1]-1)/2.0
    th = np.radians(float(angleDeg))
    k = np.rint((ys-cy)*np.cos(th)-(xs-cx)*np.sin(th)+cy).astype(np.int64)
    k -= k.min()   #shifted to zero, not clipped: clipping piles off-page ink into one row
    prof = np.bincount(k).astype(np.float64)
    return float((prof**2).sum())

#----------------------------------------------------------------------
#Deskew: scores a range of angles on a quarter-size mask, rotates to the best, and
#returns the full and half-resolution pages with the ink mask and ratio
#----------------------------------------------------------------------
def Deskew(page):
    full = np.asarray(page,dtype=np.uint8)
    if full.ndim==3:
        full = np.rint(full[...,:3].mean(2)).astype(np.uint8)
    halfA = Halve(full)

    mask,_ = InkMask(halfA.astype(np.float64))
    q = Halve(mask.astype(np.uint8)*255)>127
    best = max(np.arange(-4,4.01,0.5),key=lambda a: SkewScore(q,a))
    #fill 255: whatever rotates in from outside the photograph is paper
    fullA = Rotate(full,best,fill=255)
    halfA = Rotate(halfA,best,fill=255)
    g = halfA.astype(np.float64)
    mask,ratio = InkMask(g)
    return fullA,halfA,mask,ratio,best

#---RemovePrintedStructure: masks off everything outside the sheet, then deletes long straight runs---
def RemovePrintedStructure(mask,g):
    P,_ = LinePitch(mask)
    paper = PaperRegionMask(g,P)
    paperRemoved = int((mask&~paper).sum())
    mask&=paper
    flat = AnyWithinRadius(LongRunMask(AnyWithinRadius(mask,1,0),int(1.25*P),1),1,0)&mask
    upright = AnyWithinRadius(LongRunMask(AnyWithinRadius(mask,3,1),int(1.4*P),0),3,1)&mask
    removed = int((flat|upright).sum())
    mask = mask&~flat&~upright
    return mask,paper,P,paperRemoved,removed

#----------------------------------------------------------------------
#DropArtifactPieces: drops whole pieces that are not writing, edge slivers, pieces far too big
#or too thin, faint vertical bars and crease shadows
#----------------------------------------------------------------------
def DropArtifactPieces(mask,ratio,P,H,W):
    rc = max(1,int(0.1*P))
    dil = AnyWithinRadius(AnyWithinRadius(mask,rc,0),rc,1)
    closed = ~AnyWithinRadius(AnyWithinRadius(~dil,rc,0),rc,1)
    lab,nc = LabelComponents(closed)
    ys,xs = np.nonzero(mask)
    pl = lab[ys,xs]
    area = np.bincount(pl,minlength=nc+1).astype(float)
    top,bot,xmin,xmax,_ = PieceBounds(pl,ys,xs,nc,H,W)
    tall = (bot-top+1)>2.2*P
    thin = (xmax-xmin+1)<=0.5*P
    edgeTol = max(3,int(0.5*P))   #this close to the paper boundary counts as page
    touchL,touchR = xmin<=edgeTol,xmax>=W-1-edgeTol
    touchT,touchB = top<=edgeTol,bot>=H-1-edgeTol
    thinV = (bot-top+1)<=0.5*P
    letters = (area>=0.065*P*P)&~tall
    #where the text block starts on the left, judged from the letters alone
    xL = np.percentile(xmin[letters],5) if letters.any() else -np.inf
    vbar = ((xmax-xmin+1)<=0.12*P)&((bot-top+1)>=1.2*P)
    otsuT = Otsu(ratio)
    #bincount with weights sums the ratio per piece, giving its mean darkness
    meanRatio = np.bincount(pl,weights=ratio[ys,xs],minlength=nc+1)/np.maximum(area,1)
    faint = meanRatio>0.85*otsuT
    faintVbar = faint&((xmax-xmin+1)<=0.2*P)&((bot-top+1)>=0.6*P)

    density = area/np.maximum((bot-top+1)*(xmax-xmin+1),1)
    edgeHeight = (bot-top+1)<=1.2*P
    edgeShadow = (touchT|touchB)&edgeHeight&(density<0.15)&faint&((xmax-xmin+1)>P)

    drop = (touchL|(touchR&thin)|((touchT|touchB)&thinV&thin)
            |(tall&thin)|vbar|faintVbar|edgeShadow|(xmax<xL-1.2*P))
    drop[0] = False
    nArtifacts = int(drop.sum())
    mask&=~drop[lab]
    return mask,lab,nc,area,nArtifacts

#----------------------------------------------------------------------
#LineCentres: the row each line sits on, profile peaks kept at least 0.6 pitches apart,
#then optionally merged where the dip between two of them is shallow
#----------------------------------------------------------------------
def LineCentres(mask,H,mergeValley=None):
    P,sm = LinePitch(mask)
    peaks = [i for i in range(1,H-1) if sm[i]>=sm[i-1] and sm[i]>sm[i+1] and sm[i]>0.06*sm.max()]
    centres = []
    for c in sorted(peaks,key=lambda i:-sm[i]):
        if all(abs(c-o)>=0.6*P for o in centres):
            centres.append(c)
    centres.sort()

    thr = LINE_MERGE_VALLEY if mergeValley is None else float(mergeValley)
    while len(centres)>1 and thr<1.0:
        ratios = [(float(sm[a:b+1].min())/max(float(min(sm[a],sm[b])),1e-9),i)
                  for i,(a,b) in enumerate(zip(centres,centres[1:]))]
        worst,i = max(ratios)
        if worst<=thr:
            break
        centres.pop(i+1 if sm[centres[i]]>=sm[centres[i+1]] else i)

    return P,centres

#----------------------------------------------------------------------
#SeamCuts: a least-ink seam between each pair of line centres, so the cut goes around
#descenders instead of through them
#----------------------------------------------------------------------
def SeamCuts(mask,centres,P,H,W):
    cuts = [np.full(W,max(0,centres[0]-int(0.75*P)))]
    tailR = max(2,int(0.07*P))
    #closing, not plain dilation: it fills the thin gap a seam could thread
    #through beside a stem while leaving the stroke's outer shape near real
    mdil = AnyWithinRadius(AnyWithinRadius(mask,tailR,0),tailR,1)
    mClosed = ~AnyWithinRadius(AnyWithinRadius(~mdil,tailR,0),tailR,1)
    m = mClosed.astype(np.float64)

    for c0,c1 in zip(centres,centres[1:]):
        yy = np.arange(c0,c1+1)[:, None]
        mid,halfGap = c0+0.5*(c1-c0),max(1.0,(c1-c0)/2.0)
        #the quadratic term pulls the seam gently back towards the midline
        cuts.append(c0+LeastCostSeam(4.0*m[c0:c1+1]+0.012*((yy-mid)/halfGap)**2,step=7))

    cuts.append(np.full(W,min(H-1,centres[-1]+int(0.75*P))))
    L = len(cuts)-1
    cutsArr = np.array(cuts)
    return cuts,cutsArr,L

#----------------------------------------------------------------------
#_BandOwnership: the first guess at which line owns each piece of ink, the band holding the
#most of it, with small pieces, cut-off tails and dots pulled to their line
#----------------------------------------------------------------------
def _BandOwnership(mask,lab,nc,cuts,cutsArr,L,P,H,W,centres):
    ys,xs = np.nonzero(mask)
    pl = lab[ys,xs]
    Y = np.arange(H)[:, None]
    #band index = how many seam boundaries sit above this pixel
    region = sum((Y>=cuts[n][None,:]).astype(np.int16) for n in range(1,L)) if L>1 else np.zeros((H,W),np.int16)
    rp = region[ys,xs]
    ok = (ys>=cutsArr[0][xs])&(ys<cutsArr[-1][xs])
    area = np.bincount(pl,minlength=nc+1).astype(float)

    A = np.bincount(pl,minlength=nc+1).astype(float)
    cx = np.bincount(pl,xs,nc+1)/np.maximum(A,1)
    cy = np.bincount(pl,ys,nc+1)/np.maximum(A,1)
    #(piece,band) packed into one integer, so one bincount gives the table
    cnt = np.bincount(pl[ok]*L+rp[ok],minlength=(nc+1)*L).reshape(nc+1,L)
    substantial = cnt>=0.15*np.maximum(cnt.sum(1,keepdims=True),1)
    regionIdx = np.where(substantial,np.arange(L)[None,:],L)
    #a piece spanning two lines is reserved for the topmost it touches
    major = np.where(substantial.any(1),regionIdx.min(1),cnt.argmax(1))

    minor = (cnt>0)&(cnt<0.12*P*P)&(cnt<0.35*cnt.sum(1,keepdims=True))&(np.abs(np.arange(L)[None,:]-major[:, None])==1)
    key = pl[ok]*L+rp[ok]
    xlo,xhi = np.full((nc+1)*L,W),np.zeros((nc+1)*L,int)
    np.minimum.at(xlo,key,xs[ok])
    np.maximum.at(xhi,key,xs[ok])
    minor&=((xhi-xlo+1).reshape(nc+1,L)<=0.55*P)   #a cut-off tail is narrow
    big = (cnt.sum(1)>=0.05*P*P)
    tiny = np.zeros(nc+1,bool)
    for l in np.nonzero((cnt.sum(1)>0)&(cnt.sum(1)<0.06*P*P))[0]:
        n = major[l]
        near = big&(major==n)&(np.abs(cx-cx[l])<=6*P)
        tiny[l] = True
        speck = cnt[l].sum()<0.01*P*P
        if not near.any() or (speck and not (near&(np.abs(cx-cx[l])<=3*P)).any()):
            major[l] = -1   #nowhere near its line's writing
            continue
        yc = np.median(cy[near&(np.abs(cx-cx[l])<=1.5*P)]) if (near&(np.abs(cx-cx[l])<=1.5*P)).any() else np.median(cy[near])
        if cy[l]-yc>0.5*P and n+1<L:
            major[l] = -1 if speck else n+1
        elif yc-cy[l]>0.7*P and n>0:
            major[l] = -1 if speck else n-1
    for l in np.nonzero((cnt.sum(1)>=0.06*P*P)&(cnt.sum(1)<0.15*P*P)&(major>=0))[0]:
        pick,bd = major[l],np.inf
        for n in (major[l]-1,major[l],major[l]+1):
            rows = big&(major==n)&(np.abs(cx-cx[l])<=1.5*P)
            if 0<=n<L and rows.any() and abs(np.median(cy[rows])-cy[l])<bd:
                pick,bd = n,abs(np.median(cy[rows])-cy[l])
        major[l],tiny[l] = pick, True
    #a dot goes with the letter body directly below it
    top2,bot2,x0b,x1b,_ = PieceBounds(pl,ys,xs,nc,H,W)
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
                n2 = major[hit[0]]
                rows = big&(major==n2)&(np.abs(cx-cx[l])<=1.5*P)
                if rows.any() and 0.25*P<=np.median(cy[rows])-cy[l]<=0.7*P:
                    major[l],tiny[l] = n2, True
                break
    own = np.full((H,W),-1,np.int16)
    own[ys[ok],xs[ok]] = np.where(tiny[pl]|minor[pl,rp],major[pl],rp)[ok]
    ownBeforeEF = own.copy()
    return own,ownBeforeEF,region,area,ys,xs

#----------------------------------------------------------------------
#_DotsJoinedAbove: returns a small high-sitting piece, fused to a descender above it by the
#gap closing, to the line its own stem belongs to
#----------------------------------------------------------------------
def _DotsJoinedAbove(own,mask,lab,area,region,centres,L,P,H,W,ys,xs):
    raw,nr = LabelComponents(mask)
    pr = raw[ys,xs]
    Ar = np.bincount(pr,minlength=nr+1).astype(float)
    rt,rb,rowLeft,rr,_ = PieceBounds(pr,ys,xs,nr,H,W)
    h0,w0 = rb-rt+1,rr-rowLeft+1
    dotr = (Ar>=0.015*P*P)&(Ar<0.1*P*P)&(np.maximum(h0,w0)<=0.5*P)&(np.maximum(h0,w0)<=2.2*np.maximum(np.minimum(h0,w0),1))
    stemr = (Ar>=0.03*P*P)&~dotr
    for d in np.nonzero(dotr)[0]:
        for gap in range(1,int(0.4*P)+1):
            if rb[d]+gap>=H:
                break
            row = raw[rb[d]+gap,max(0,rowLeft[d]-2):rr[d]+3]
            hit = [v for v in np.unique(row[row>0]) if v!=d and stemr[v]]
            if hit:
                n2 = region[rb[d]+gap,max(0,rowLeft[d]-2)+np.nonzero(row==hit[0])[0][0]]
                dv = pr==d
                ov = own[ys[dv],xs[dv]]
                if (ov>=0).any():
                    n1 = np.bincount(ov[ov>=0],minlength=L).argmax()
                    closedPiece = lab[ys[dv][0],xs[dv][0]]
                    wholeWord = area[closedPiece]>=0.2*P*P   #a word, not a floating dot
                    if n2==n1+1 and not wholeWord and (rt[d]+rb[d])/2>=centres[n1]+0.45*P:
                        keep = ov==n1
                        own[ys[dv][keep],xs[dv][keep]] = n2
                break
    return raw,nr,pr,Ar

#---_RingsToStem: gives a closed ring sitting on a stem to the line that stem belongs to---
def _RingsToStem(own,raw,nr,pr,Ar,lab,area,region,centres,P,H,W,ys,xs):
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
            for r in range(yb,max(0,yb-int(0.5*P)),-1):   #walk up while the blob stays wide
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
            if LabelComponents(~np.pad(blob[:,bc.min():bc.max()+1],1))[1]<2:   #a ring encloses a hole
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
                    wholeWord = area[lab[yb,xb]]>=0.2*P*P
                    if n2==n+1 and not wholeWord and ty>=centres[n2]-0.45*P:
                        sub = own[rt+1:yb+1,xl:xr+1]
                        sub[blob[:,xl-xa:xr-xa+1]&(sub==n)] = n2
                    break

#----------------------------------------------------------------------
#_FragmentsToNeighbour: rehomes a piece with no ink in its own x-height band to the neighbouring
#line it reaches into
#----------------------------------------------------------------------
def _FragmentsToNeighbour(own,ownBeforeEF,mask,centres,L,P,H,W):
    own0,moves = own.copy(),[]
    locked = own!=ownBeforeEF   #pixels already placed by the two steps above
    for n in range(L):
        comp,nb = LabelComponents((own0==n)&mask)
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
            cnt = np.bincount(ov[(ov>=0)&(ov!=n)],minlength=L)
            cyPiece = fy.mean()
            if cnt.max()>=3:
                m = int(cnt.argmax())
                if abs(m-n)==1 and (m>n)==(cyPiece>centres[n]):
                    moves.append((fy,fx,m))
            else:
                bestBand = min((m for m in (n-1,n+1) if 0<=m<L),key=lambda m: abs(cyPiece-centres[m]),default=None)
                if bestBand is not None and abs(cyPiece-centres[bestBand])<=0.2*P:
                    moves.append((fy,fx,bestBand))
    for fy,fx,m in moves:
        own[fy,fx] = m

#---AssignOwnership: the band guess followed by the three corrections, in order---
def AssignOwnership(mask,lab,nc,cuts,cutsArr,L,centres,P,H,W):
    own,ownBeforeEF,region,area,ys,xs = _BandOwnership(
        mask,lab,nc,cuts,cutsArr,L,P,H,W,centres)
    raw,nr,pr,Ar = _DotsJoinedAbove(
        own,mask,lab,area,region,centres,L,P,H,W,ys,xs)
    _RingsToStem(own,raw,nr,pr,Ar,lab,area,region,centres,P,H,W,ys,xs)
    _FragmentsToNeighbour(own,ownBeforeEF,mask,centres,L,P,H,W)
    return own

#----------------------------------------------------------------------
#_BuildLineCrop: one band in, one finished crop out, trimmed to its ink, padded, contrast
#stretched, and None when too little is left to be a line of writing
#----------------------------------------------------------------------
def _BuildLineCrop(n,own,norm,P,f,H,W,minInk):
    ink = own==n
    yy,xx = np.nonzero(ink)
    if len(yy)<minInk or yy.max()-yy.min()<max(6,0.2*P):
        return None
    rowInk = np.bincount(yy,minlength=H)>0
    gapAt = np.flatnonzero(~rowInk[yy.min():yy.max()+1])+yy.min()
    if len(gapAt):   #keep the tallest contiguous run of ink, dropping distant marks
        runs,s = [],yy.min()
        for g in gapAt:
            if g>s:
                runs.append((s,g-1))
            s = g+1
        runs.append((s,yy.max()))
        runs = [(a,b) for a,b in runs if b>=a]
        mainRun = max(runs,key=lambda ab: int(ink[ab[0]:ab[1]+1].sum()))
        if mainRun[1]-mainRun[0]+1<yy.max()-yy.min()+1-0.8*P:
            keep = (yy>=mainRun[0])&(yy<=mainRun[1])
            ink = np.zeros_like(ink)
            ink[yy[keep],xx[keep]] = True
            yy,xx = yy[keep],xx[keep]
    pad = int(0.15*P)
    y0,y1 = max(0,yy.min()-pad),min(H,yy.max()+1+pad)
    x0,x1 = max(0,xx.min()-pad),min(W,xx.max()+1+pad)
    Y0,Y1,X0,X1 = int(y0*f),int(y1*f),int(x0*f),int(x1*f)
    inkC = ink[y0:y1,x0:x1]
    #bilinear: the soft edge it gives is what makes the stroke edge natural
    alpha = Resize((inkC*255).astype(np.uint8),X1-X0,Y1-Y0,'linear').astype(np.float32)/255.0
    haloR = max(1,int(0.20*P))
    halo = AnyWithinRadius(AnyWithinRadius(inkC,haloR,0),haloR,1)
    haloAlpha = Resize((halo*255).astype(np.uint8),X1-X0,Y1-Y0,'linear').astype(np.float32)/255.0
    region = norm[Y0:Y1,X0:X1]
    localR = max(3,min(20,round(region.shape[0]/8)))
    localRatio = region/np.maximum(BoxMean(region,localR),1.0)   #this line's own contrast
    localMask = localRatio<Otsu(localRatio)
    ownC = own[y0:y1,x0:x1]
    notOther = ~((ownC>=0)&(ownC!=n))   #never recover a pixel given to a neighbour
    #nearest, not bilinear: this is a yes or no ownership flag
    notOtherFull = Resize((notOther*255).astype(np.uint8),X1-X0,Y1-Y0,'nearest')>127
    alpha = np.maximum(alpha,localMask*haloAlpha*notOtherFull)
    inkPix = region[alpha>0.5]
    black = np.percentile(inkPix,40) if len(inkPix) else 0.0   #typical ink darkness
    white = np.percentile(region,75)

    linear = np.clip((region-black)*(255.0/max(white-black,1.0)),0,255)
    darkness = np.clip((255.0-linear)/255.0,0,1)**0.55   #gamma below one darkens faint strokes
    enhanced = 255.0*(1.0-darkness)
    crop = (alpha*enhanced+(1-alpha)*255).astype(np.uint8)
    margin = int(pad*f)   #the same left gap on every crop
    firstCol = int(np.argmax((alpha>0.1).any(axis=0))) if (alpha>0.1).any() else margin
    shift = firstCol-margin
    if shift>0:
        crop = np.pad(crop[:,shift:],((0,0),(0,shift)),constant_values=255)
    return dict(image=crop,
                clf=FitFixedCanvas(crop,InputHeight,InputWidth),
                bbox=(X0,Y0,X1,Y1))

#---_SheetCrop: crops the dark surround away, leaving the biggest bright blob---
def _SheetCrop(g):
    bright = g>0.60*np.percentile(g,95)
    lab,n = LabelComponents(bright)
    if n==0:
        return g
    i = np.bincount(lab.ravel())[1:].argmax()+1
    yy,xx = np.nonzero(lab==i)
    return g[yy.min():yy.max()+1,xx.min():xx.max()+1]

#----------------------------------------------------------------------
#LineAxisScore: the fraction of rows that are nearly empty, which is high only on the
#axis the lines of writing run along
#----------------------------------------------------------------------
def LineAxisScore(page):
    g = np.asarray(page)[::4,::4].astype(np.float64)
    g = _SheetCrop(g)
    if g.size<100 or min(g.shape)<8:
        return 0.0
    mask,_ratio = InkMask(g)
    rows = mask.sum(axis=1).astype(np.float64)
    m = rows.mean()
    return float((rows<0.35*m).mean()) if m>0 else 0.0

#how much better the sideways score must be before the page is turned
AXIS_TURN_MARGIN = 1.35

#----------------------------------------------------------------------
#UprightPage: stands a photographed page on the correct axis, which end is up is
#settled elsewhere, by reading the lines both ways
#----------------------------------------------------------------------
def UprightPage(page):
    a = np.asarray(page)
    s0 = LineAxisScore(a)
    s90 = LineAxisScore(np.rot90(a))
    if s90>AXIS_TURN_MARGIN*max(s0,1e-9):
        return np.ascontiguousarray(np.rot90(a)),90,s0,s90
    return a,0,s0,s90

#----------------------------------------------------------------------
#DropSolidCrops: drops a crop that is a solid block of dark rather than writing, it must
#be over the absolute ink limit and over a multiple of the page median
#----------------------------------------------------------------------
def DropSolidCrops(lines,maxAbsPct=15.0,maxMedianRatio=3.0):
    if not lines:
        return lines
    ink = [float((np.asarray(l['image'])<128).mean()*100.0) for l in lines]
    med = float(np.median(ink))
    kept = []
    for l,v in zip(lines,ink):
        l['inkPct'] = v
        if med>0 and v>maxAbsPct and v>maxMedianRatio*med:
            continue
        l['order'] = len(kept)
        kept.append(l)
    return kept if kept else lines

#----------------------------------------------------------------------
#SegmentLines: the whole job, deskew, remove the printing, drop artefacts, find the
#centres, cut the seams, assign ownership and build the crops
#----------------------------------------------------------------------
def SegmentLines(page,mergeValley=None):
    fullA,halfA,mask,ratio,best = Deskew(page)
    g = halfA.astype(np.float64)
    H,W = mask.shape
    mask,paper,P,paperRemoved,removed = RemovePrintedStructure(mask,g)
    mask,lab,nc,area,nArtifacts = DropArtifactPieces(mask,ratio,P,H,W)
    P,centres = LineCentres(mask,H,mergeValley)
    cuts,cutsArr,L = SeamCuts(mask,centres,P,H,W)
    own = AssignOwnership(mask,lab,nc,cuts,cutsArr,L,centres,P,H,W)

    #the crops come off the full-resolution page, though every decision
    #above was taken at half size
    r = max(8,min(40,round(H/41)))
    bgFull = Resize(BoxMean(g,r).astype(np.float32),fullA.shape[1],fullA.shape[0],'linear')
    norm = np.clip(fullA.astype(np.float32)/np.maximum(bgFull,1.0),0,1)*255
    f = fullA.shape[1]/W   #half-size coordinate to full size
    perLine = np.bincount(own[own>=0],minlength=L)
    minInk = max(60,0.12*np.median(perLine[perLine>0]))
    lines = []
    for n in range(L):
        crop = _BuildLineCrop(n,own,norm,P,f,H,W,minInk)
        if crop is None:
            continue
        crop['order'] = len(lines)
        lines.append(crop)
    nBuilt = len(lines)
    lines = DropSolidCrops(lines)
    nDropped = nBuilt-len(lines)

    return lines,dict(cuts=cuts,ratio=ratio,paper=paper),dict(
        angle=float(best),pitch=int(P),
        paperCoverage=round(float(paper.mean()),3),
        inkOutsidePaper=paperRemoved,removedStraightPixels=removed,
        artifactPieces=nArtifacts,nLines=len(lines),dropped=nDropped)
