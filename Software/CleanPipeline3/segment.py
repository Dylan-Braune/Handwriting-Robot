#u23539195 Dylan Braune.  the image toolkit, and page to line segmentation.
#one photographed page in, clean deskewed line crops out.  FU 2.1, 2.2, R2
#
#the first half of this file is the image toolkit the whole pipeline shares:
#window sums, morphology, thresholding, components, thinning, resampling and
#drawing.  the second half is the segmenter that uses it.  other files import
#what they need from here, which is why there is no separate imaging module
#
#arrays in, arrays out.  no file opening and no image library anywhere
import numpy as np

InputHeight,InputWidth = 64,640   #exact input size the classifiers were trained on


#======================================================================
#TOOLKIT 1: adding up pixels inside a rectangle
#everything below that needs a local average or a local count is built on
#these, because they cost the same no matter how big the window is
#======================================================================
#----------------------------------------------------------------------
#Integral: a running total of the image, for fast window sums
#cell (y,x) holds the sum of everything above and left of it.  once you
#have that, the sum inside ANY rectangle is four lookups: bottom right
#minus the two edges plus the corner that got taken off twice
#that is why a big blur here costs the same as a small one
#----------------------------------------------------------------------
def Integral(img):
    ii = np.zeros((img.shape[0]+1,img.shape[1]+1),np.float64)   #one spare row and col of zeros
    np.cumsum(np.cumsum(img,axis=0),axis=1,out=ii[1:,1:])   #ii[y,x] = sum of all above and left
    return ii

#----------------------------------------------------------------------
#BoxSum: total ink in a window around every pixel at once
#slides a (2r+1) square over the whole image using the running total,
#so every pixel gets its own window sum in one pass
#----------------------------------------------------------------------
def BoxSum(img,ry,rx):
    h,w = img.shape
    ii = Integral(img)
    y,x = np.arange(h),np.arange(w)
    y1,y2 = np.clip(y-ry,0,h)[:,None],np.clip(y+ry+1,0,h)[:,None]   #window clipped at the edge
    x1,x2 = np.clip(x-rx,0,w)[None,:],np.clip(x+rx+1,0,w)[None,:]
    return ii[y2,x2]-ii[y1,x2]-ii[y2,x1]+ii[y1,x1]   #sum = BR - TR - BL + TL

#----------------------------------------------------------------------
#BoxCount: how many pixels that window actually covered
#at the edges the window hangs off the image, so the count is smaller
#than the full square.  dividing by this instead of the square gives an
#honest average at the borders
#----------------------------------------------------------------------
def BoxCount(ry,rx,h,w):
    y,x = np.arange(h),np.arange(w)
    cy = (np.clip(y+ry+1,0,h)-np.clip(y-ry,0,h))[:,None]
    cx = (np.clip(x+rx+1,0,w)-np.clip(x-rx,0,w))[None,:]
    return cy*cx

#----------------------------------------------------------------------
#BoxMean: the average value in a window around every pixel
# used as a local background level,
#so a page that is brighter on one side still reads correctly
#----------------------------------------------------------------------
def BoxMean(a,r):
    p = np.pad(a,r,mode="edge").astype(np.float64)   #edge repeated, not clipped, so every window has (2r+1)^2
    ii = Integral(p)
    k = 2*r+1
    return (ii[k:,k:]-ii[:-k,k:]-ii[k:,:-k]+ii[:-k,:-k])/(k*k)   #same 4 lookups, divided by the count



#======================================================================
#TOOLKIT 2: growing and shrinking shapes
#all three are window sums with a test on the result, no loops over pixels
#======================================================================
#----------------------------------------------------------------------
#Dilate: grow every shape outwards by r pixels
#a pixel turns on if ANY set pixel is within the window, which is the
#same as asking whether the window sum is above zero
#----------------------------------------------------------------------
def Dilate(mask,ky,kx):
    return BoxSum(mask.astype(np.float64),ky//2,kx//2)>0   #sum>0 means at least one was set

#----------------------------------------------------------------------
#Erode: shrink every shape inwards by r pixels
#a pixel stays on only if EVERY pixel in the window is set, so the sum
#has to equal the covered count.  the opposite of dilate
#----------------------------------------------------------------------
def Erode(mask,ky,kx):
    ry,rx = ky//2,kx//2
    return BoxSum(mask.astype(np.float64),ry,rx)>=BoxCount(ry,rx,*mask.shape)-0.5   #sum == count

#----------------------------------------------------------------------
#Close: bridge small gaps without fattening the shape
#dilate then erode by the same radius.  the dilate joins anything closer
#than the window, the erode gives back the thickness it added, so a
#broken stroke heals but a thin stroke stays thin
#----------------------------------------------------------------------
def Close(mask,ky,kx):
    return Erode(Dilate(mask,ky,kx),ky,kx)   #outside shape stays put, inside gaps fill

#---AnyWithinRadius: dilate along one axis only, true where anything is set within r places---
def AnyWithinRadius(m,r,axis):
    a = np.moveaxis(m.astype(np.int32),axis,0)   #work on axis 0 then move back
    n = a.shape[0]
    c = np.cumsum(np.pad(a,[(r+1,r)]+[(0,0)]*(a.ndim-1)),0)
    return np.moveaxis((c[2*r+1:2*r+1+n]-c[:n])>0,0,axis)   #count over the span grew at all



#======================================================================
#TOOLKIT 3: telling a printed rule from handwriting
#a printed line runs straight and unbroken for far longer than a pen stroke
#======================================================================
#----------------------------------------------------------------------
#LongRunMask: find ink that runs straight for a long way
#measures how long the unbroken run through each pixel is, along rows
#or columns, and keeps only the runs longer than a limit.  that picks
#out ruled lines and the binder edge while leaving letters alone
#----------------------------------------------------------------------
def LongRunMask(m,n,axis):
    a = np.moveaxis(m.astype(np.int32),axis,0)
    L = a.shape[0]
    if L<n:
        return np.zeros(m.shape,bool)
    c = np.cumsum(np.pad(a,((1,0),(0,0))),0)
    hit = ((c[n:]-c[:-n])==n).astype(np.int32)   #a run of n starts at i when c[i+n]-c[i]==n
    ch = np.cumsum(np.pad(hit,((1,0),(0,0))),0)   #second sum turns starts into "a run covers me"
    j = np.arange(L)
    lo,hi = np.maximum(0,j-n+1),np.minimum(j,L-n)   #a covering run must start in [lo,hi]
    return np.moveaxis(((ch[hi+1]-ch[lo])>0)&(hi>=lo)[:,None],0,axis)

#---RowRunLength: for every pixel, how long the unbroken run of ink through it is along its row---
def RowRunLength(mask):
    h,w = mask.shape
    a = mask.astype(np.int32)
    starts = np.zeros_like(a)
    starts[:,0] = a[:,0]
    starts[:,1:] = (a[:,1:]==1)&(a[:,:-1]==0)   #a run starts where the row steps 0->1
    runId = np.cumsum(starts.reshape(-1)).reshape(h,w)   #number every run
    runId[a==0] = 0
    if runId.max()==0:
        return np.zeros_like(a)
    counts = np.bincount(runId.reshape(-1))   #how long each numbered run is
    counts[0] = 0
    return counts[runId]   #hand each pixel its own run's length



#======================================================================
#TOOLKIT 4: deciding what is ink and what is paper
#one threshold for a whole crop, then a tidy up pass
#======================================================================
#----------------------------------------------------------------------
#Otsu: pick the brightness that best splits ink from paper
#tries every possible cut of the histogram and keeps the one where the
#two sides are furthest apart.  no fixed number, so it copes with a dark
#photo and a bright one the same way
#----------------------------------------------------------------------
def Otsu(v,hi=None):
    a = np.asarray(v)
    isInt = np.issubdtype(a.dtype,np.integer)
    if isInt:
        hist = np.bincount(a.ravel(),minlength=256).astype(np.float64)
    else:
        hi = 1.2 if hi is None else float(hi)   #ratio image, nothing above 1.2 carries ink
        hist = np.histogram(np.clip(a,0,hi),bins=256,range=(0,hi))[0].astype(np.float64)   #float image, bin it to 256
    p = hist/max(hist.sum(),1.0)
    w = np.cumsum(p)   #fraction of pixels at or below each bin
    m = np.cumsum(p*np.arange(256))   #their mean contribution
    var = (m[-1]*w-m)**2/np.maximum(w*(1-w),1e-12)   #guard the empty halves where wB*wF is 0
    t = int(np.argmax(var))
    return t if isInt else (t+0.5)*hi/256   #bin index for ints, bin centre for floats

#----------------------------------------------------------------------
#BinarizeLine: a grey line crop in, a clean ink mask out
#thresholds with Otsu, closes pinholes, drops specks too small to be
#a pen mark, and throws away long straight runs that are printed rules
#rather than writing
#----------------------------------------------------------------------
def BinarizeLine(gray,stripRules=True,closeGaps=True):
    ink = np.asarray(gray)<Otsu(np.asarray(gray,dtype=np.uint8))
    h = ink.shape[0]
    if stripRules and ink.any():
        hRun = RowRunLength(ink)   #run length across
        vRun = RowRunLength(ink.T).T   #and down, same function transposed
        thin = vRun<=max(3,int(0.12*h))
        rule = (hRun>=max(24,int(1.1*h)))&thin   #long AND thin is a rule, not a letter
        if rule.any():
            ink = ink&~(rule&thin)   #second thin test keeps a real stroke crossing the rule
    if closeGaps and ink.any():
        k = int(np.clip(round(0.045*h),2,4))
        ink = Close(ink,k,k)   #a fast hand breaks thin strokes, this rejoins a hairline gap
    labels,n = LabelComponents(ink)
    if n:
        areas = np.bincount(labels.ravel())
        keep = areas>=6   #under 6 pixels is noise, not ink
        keep[0] = False
        ink = keep[labels]
    return ink

#----------------------------------------------------------------------
#CoreBand: find the band holding the lower case letter bodies
#counts ink per row and keeps the rows holding most of it, which is the
#x height band between the baseline and the top of an 'a'.  gives the
#scale everything else is measured in
#----------------------------------------------------------------------
def CoreBand(ink):
    prof = ink.sum(axis=1).astype(np.float64)
    if prof.max()<=0:
        return None
    rows = np.nonzero(prof>=0.45*prof.max())[0]
    top,base = int(rows[0]),int(rows[-1])
    return (top,base) if base-top>=3 else None   #a band under 3 rows is not a band



#======================================================================
#TOOLKIT 5: finding separate pieces of ink
#a piece is one blob of touching pixels, like a letter or a dot
#======================================================================
#----------------------------------------------------------------------
#LabelComponents: number every separate blob of ink
#works row by row.  each row of ink becomes runs, runs touching the
#row above join the same group, and a small union find keeps track of
#which group numbers turned out to be the same piece
#----------------------------------------------------------------------
def LabelComponents(mask):
    H,W = mask.shape
    lab = np.zeros((H,W),np.int32)
    parent = [0]   #each label points at a lower one, following the chain gives the group

    def find(a):
        while parent[a]!=a:
            parent[a] = parent[parent[a]]   #flatten the chain as we walk it
            a = parent[a]
        return a

    prev = []
    for y in range(H):
        d = np.diff(np.r_[0,mask[y].view(np.int8),0])   #runs start at 0->1 and end at 1->0
        cur,k = [],0
        for a,b in zip(np.nonzero(d==1)[0].tolist(),np.nonzero(d==-1)[0].tolist()):
            while k<len(prev) and prev[k][1]<a:   #skip runs above that ended before this one starts
                k+=1
            l,kk = 0,k
            while kk<len(prev) and prev[kk][0]<=b:   #<=b so a diagonal touch counts, 8 connectivity
                r = find(prev[kk][2])
                if l==0:
                    l = r
                elif r!=l:
                    parent[max(r,l)] = min(r,l)   #two groups above join through this run
                    l = min(r,l)
                kk+=1
            if l==0:   #touches nothing above, so start a new label
                parent.append(len(parent))
                l = len(parent)-1
            cur.append((a,b,l))
            lab[y,a:b] = l
        prev = cur
    roots = np.array([find(i) for i in range(len(parent))])
    _,comp = np.unique(roots,return_inverse=True)   #labels were handed out before the merges, renumber 1..n
    return comp[lab].astype(np.int32),int(comp.max())

#----------------------------------------------------------------------
#PieceBounds: the box and pixel count of every numbered piece
#one pass with minimum and maximum per label, so you get where each
#piece sits and how big it is without looping over pieces
#----------------------------------------------------------------------
def PieceBounds(labelAt,ys,xs,n,H,W):
    top,bot = np.full(n+1,H),np.zeros(n+1,int)
    left,right = np.full(n+1,W),np.zeros(n+1,int)
    np.minimum.at(top,labelAt,ys)
    np.maximum.at(bot,labelAt,ys)
    np.minimum.at(left,labelAt,xs)
    np.maximum.at(right,labelAt,xs)
    area = np.bincount(labelAt,minlength=n+1).astype(float)
    return top,bot,left,right,area



#======================================================================
#TOOLKIT 6: thinning a stroke to its centre line
#needed by the glyph extractor, which follows the centre of a pen stroke
#======================================================================
#----------------------------------------------------------------------
#Skeletonize: shave a stroke down to a one pixel wide line
#Zhang-Suen thinning: repeatedly delete edge pixels that are not needed
#to keep the shape connected, until nothing more can go.  a thick pen
#stroke becomes the path the pen travelled
#----------------------------------------------------------------------
def Skeletonize(mask):
    img = mask.astype(np.uint8).copy()

    def ring(P):
        p = np.pad(P,1)   #the 8 neighbours of every pixel at once, clockwise from north
        return (p[:-2,1:-1],p[:-2,2:],p[1:-1,2:],p[2:,2:],
                p[2:,1:-1],p[2:,:-2],p[1:-1,:-2],p[:-2,:-2])

    changed = True
    while changed:
        changed = False
        for step in (0,1):   #two sub passes, alternating corners so the line does not drift
            P2,P3,P4,P5,P6,P7,P8,P9 = ring(img)
            r = [P2,P3,P4,P5,P6,P7,P8,P9,P2]
            B = sum(r[:8])   #how many neighbours are set
            A = sum(((r[k]==0)&(r[k+1]==1)).astype(np.uint8) for k in range(8))   #0->1 steps round the ring
            if step==0:
                c1,c2 = (P2*P4*P6)==0,(P4*P6*P8)==0
            else:
                c1,c2 = (P2*P4*P8)==0,(P2*P6*P8)==0
            go = (img==1)&(B>=2)&(B<=6)&(A==1)&c1&c2   #A==1 means deleting it cannot split the stroke
            if go.any():
                img[go] = 0
                changed = True
    return img.astype(bool)



#======================================================================
#TOOLKIT 7: resizing
#one resampler, used for the page reduction and the classifier input
#======================================================================
#----------------------------------------------------------------------
#_Weights: which input pixels feed each output pixel, and how much
#builds the weights once per axis.  when shrinking, the filter widens so
#every input pixel still contributes, which is what stops a reduced page
#from dropping thin strokes
#----------------------------------------------------------------------
def _Weights(inSize,outSize,kernel):
    scale = inSize/float(outSize)
    filterScale = max(1.0,scale)   #only widen when shrinking, scale>1 is a shrink
    support = (1.0 if kernel=='linear' else 0.5)*filterScale
    centres = (np.arange(outSize)+0.5)*scale   #centre of output pixel i, in input coordinates
    half = int(np.ceil(support))+1
    #broadcast a column of window centres against a row of offsets, giving
    #one row of source indices per output pixel in a single array
    cols = np.floor(centres)[:,None]+np.arange(-half,half+1)[None,:]
    t = (cols+0.5-centres[:,None])/filterScale   #distance from the window centre, in filter widths
    if kernel=='linear':
        w = np.where(np.abs(t)<1.0,1.0-np.abs(t),0.0)   #triangle, w = 1-|t|
    else:
        w = np.where(np.abs(t)<=0.5,1.0,0.0)   #box, flat inside then nothing
    cc = np.clip(cols,0,inSize-1).astype(np.int64)   #fold a window running off the image back on
    out = np.zeros((outSize,inSize),np.float64)
    np.add.at(out,(np.repeat(np.arange(outSize),cols.shape[1]),cc.ravel()),w.ravel())   #add.at: folded weights must sum, not overwrite
    tot = out.sum(axis=1,keepdims=True)
    return out/np.where(tot>0,tot,1.0)   #each row sums to 1, so no dark border on a padded crop

#----------------------------------------------------------------------
#Resize: scale an image with those weights
#applies the weights along one axis then the other, which gives the same
#answer as doing both at once but far fewer multiplications
#----------------------------------------------------------------------
def Resize(arr,newW,newH,kernel='linear'):
    a = np.asarray(arr)
    newW,newH = max(1,int(newW)),max(1,int(newH))
    h,w = a.shape[0],a.shape[1]
    if h==newH and w==newW:
        return a.copy()
    if kernel=='nearest':   #for LABELS not brightness: averaging line 3 and 4 gives 3.5, which exists nowhere
        sy = np.clip(((np.arange(newH)+0.5)*(h/float(newH))).astype(np.int64),0,h-1)
        sx = np.clip(((np.arange(newW)+0.5)*(w/float(newW))).astype(np.int64),0,w-1)
        return a[sy[:,None],sx[None,:]]
    f = a.astype(np.float64)
    if newH!=h:
        f = np.tensordot(_Weights(h,newH,kernel),f,axes=([1],[0]))   #rows pass
    if newW!=w:
        f = np.tensordot(f,_Weights(w,newW,kernel).T,axes=([1],[0]))   #then columns
        if f.ndim==3:
            f = np.moveaxis(f,-1,1)   #tensordot left width last on a 3-d array, put it back
    if np.issubdtype(a.dtype,np.integer):
        i = np.iinfo(a.dtype)
        return np.clip(np.rint(f),i.min,i.max).astype(a.dtype)
    return f.astype(a.dtype)

#---Halve: exact 2 to 1 average, the quick page reduction---
def Halve(a):
    a = np.asarray(a)
    return Resize(a,a.shape[1]//2,a.shape[0]//2,'box')

#----------------------------------------------------------------------
#FitFixedCanvas: put a crop on the fixed classifier canvas
#the networks were trained on one exact size, so a crop is scaled by
#ONE factor for both axes, never stretched, and the leftover space is
#filled with paper.  stretching would change the letter shapes
#----------------------------------------------------------------------
def FitFixedCanvas(gray,height,width,padValue=255):
    a = np.asarray(gray)
    h,w = a.shape[0],a.shape[1]
    canvas = np.full((height,width),padValue,np.uint8)
    if w<=0 or h<=0:
        return canvas
    s = min(height/float(h),width/float(w))   #whichever axis runs out of room first
    newW,newH = max(1,int(round(w*s))),max(1,int(round(h*s)))
    small = Resize(a.astype(np.uint8),newW,newH,'linear')
    y0 = max(0,(height-newH)//2)   #vertically centred, left flush
    canvas[y0:y0+newH,:newW] = small[:height-y0,:width]   #slices absorb a 1 pixel rounding overshoot
    return canvas



#======================================================================
#TOOLKIT 8: turning, bending and drawing
#all of these read the source at fractional positions, hence the sampler
#======================================================================
#----------------------------------------------------------------------
#_Sample: read the image at a position between pixels
#mixes the four neighbours by how close each one is, so a rotated or
#bent image comes out smooth instead of jagged
#----------------------------------------------------------------------
def _Sample(a,sx,sy,fill,edge='area'):
    h,w = a.shape[0],a.shape[1]
    x0,y0 = np.floor(sx).astype(np.int64),np.floor(sy).astype(np.int64)   #the pixel up and left of the sample point
    fx,fy = sx-x0,sy-y0   #fractional part, how far between the two pixels
    if edge=='area':
        inside = (sx>-0.5)&(sx<w-0.5)&(sy>-0.5)&(sy<h-0.5)   #pixel i covers i-0.5..i+0.5, so the image spans -0.5..w-0.5
    else:
        inside = (sx>=0)&(sx<=w-1)&(sy>=0)&(sy<=h-1)   #strict: anything off the pixel centres is fill
    xi,yi = np.clip(x0,0,w-1),np.clip(y0,0,h-1)
    xj,yj = np.clip(x0+1,0,w-1),np.clip(y0+1,0,h-1)   #at an edge both clamp to the same pixel
    top = a[yi,xi]*(1-fx)+a[yi,xj]*fx
    bot = a[yj,xi]*(1-fx)+a[yj,xj]*fx
    return np.where(inside,top*(1-fy)+bot*fy,float(fill))

#---_Like: round a float result back to the type that came in---
def _Like(out,like):
    if np.issubdtype(np.asarray(like).dtype,np.integer):
        return np.clip(np.rint(out),0,255).astype(np.uint8)
    return out

#----------------------------------------------------------------------
#Rotate: turn the image about its centre, same canvas size
#works backwards: for each OUTPUT pixel it asks where that came from in
#the input, which guarantees no gaps.  anything off the edge is paper
#----------------------------------------------------------------------
def Rotate(gray,deg,fill=255):
    a = np.asarray(gray).astype(np.float64)
    h,w = a.shape[0],a.shape[1]
    th = np.radians(float(deg))
    c,s = np.cos(th),np.sin(th)
    cy,cx = (h-1)/2.0,(w-1)/2.0
    yy,xx = np.meshgrid(np.arange(h,dtype=np.float64),
                        np.arange(w,dtype=np.float64),indexing='ij')
    dx,dy = xx-cx,yy-cy
    #sx = cos*dx-sin*dy+cx, sy = sin*dx+cos*dy+cy.  rows run DOWNWARD so this
    #is the transpose of the textbook matrix, which makes deg mean the same
    #thing here as in the skew search
    return _Like(_Sample(a,c*dx-s*dy+cx,s*dx+c*dy+cy,fill),gray)

#----------------------------------------------------------------------
#Warp: affine or perspective, whatever the six coefficients say
#same backwards idea as Rotate, but the source position is a general
#mapping, so one function covers shear, scale and keystone together
#----------------------------------------------------------------------
def Warp(gray,coeffs,fill=255):
    a = np.asarray(gray).astype(np.float64)
    h,w = a.shape[0],a.shape[1]
    yy,xx = np.meshgrid(np.arange(h,dtype=np.float64),
                        np.arange(w,dtype=np.float64),indexing='ij')
    X,Y = xx+0.5,yy+0.5   #coeffs act on the pixel CENTRE, at i+0.5
    k = [float(v) for v in coeffs]
    if len(k)==6:
        sx,sy = k[0]*X+k[1]*Y+k[2]-0.5,k[3]*X+k[4]*Y+k[5]-0.5   #0.5 off again to get an index
    elif len(k)==8:
        den = k[6]*X+k[7]*Y+1.0
        den = np.where(np.abs(den)<1e-12,1e-12,den)   #zero denominator is a point at infinity, nudge it
        sx,sy = (k[0]*X+k[1]*Y+k[2])/den-0.5,(k[3]*X+k[4]*Y+k[5])/den-0.5
    else:
        raise ValueError("Warp wants 6 coefficients for affine or 8 for perspective")
    return _Like(_Sample(a,sx,sy,fill,edge='centre'),gray)

#----------------------------------------------------------------------
#Blur: soften the image with a gaussian
#runs a 1-d gaussian down the rows then across the columns, which is the
#same result as a 2-d one and much cheaper
#----------------------------------------------------------------------
def Blur(gray,sigma,mode='edge'):
    a = np.asarray(gray).astype(np.float64)
    s = float(sigma)
    if s<=0.0:
        return np.asarray(gray).copy()
    r = max(1,int(np.ceil(3.0*s)))   #cut off at 3 sigma, past that the weight is under 0.3 percent
    t = np.arange(-r,r+1,dtype=np.float64)
    k = np.exp(-(t*t)/(2.0*s*s))
    k /= k.sum()
    pk = dict(edge='edge',constant='constant')[mode]   #edge for a photo, constant for a zero centred field
    pad = np.pad(a,((r,r),(r,r)),mode=pk)
    tmp = np.zeros_like(a)
    for i,wgt in enumerate(k):
        tmp += wgt*pad[i:i+a.shape[0],r:r+a.shape[1]]   #rows pass, sum of shifted copies
    pad2 = np.pad(tmp,((0,0),(r,r)),mode=pk)
    out = np.zeros_like(a)
    for i,wgt in enumerate(k):
        out += wgt*pad2[:,i:i+a.shape[1]]   #columns pass, same again
    return _Like(out,gray)

#----------------------------------------------------------------------
#Elastic: bend strokes locally, for training augmentation
#two smooth random fields say how far each pixel moves in x and in y,
#then the image is read at the moved positions.  makes handwriting that
#is plausibly the same hand but not the same sample
#----------------------------------------------------------------------
def Elastic(gray,sigma,alpha,rng):
    a = np.asarray(gray).astype(np.float64)
    h,w = a.shape
    dx = Blur(rng.uniform(-1,1,(h,w)),sigma,mode='constant')*alpha   #blurred so neighbours move together
    dy = Blur(rng.uniform(-1,1,(h,w)),sigma,mode='constant')*alpha
    yy,xx = np.meshgrid(np.arange(h,dtype=np.float64),
                        np.arange(w,dtype=np.float64),indexing='ij')
    return _Like(_Sample(a,xx+dx,yy+dy,255,edge='centre'),gray)   #255 outside, the page is white there

#----------------------------------------------------------------------
#DrawPolyline: draw a line with a round pen
#for each pixel near the line it measures the distance to the nearest
#point ON the segment, and fills if that is within half a pen width.
#round ends and round corners come out of this for free
#----------------------------------------------------------------------
def DrawPolyline(img,pts,value,width):
    if len(pts)==0:
        return img
    h,w = img.shape[0],img.shape[1]
    r = max(0.5,float(width)/2.0)
    val = np.uint8(np.clip(round(float(value)),0,255))
    segs = [(pts[i],pts[i+1]) for i in range(len(pts)-1)] or [(pts[0],pts[0])]
    for (ax,ay),(bx,by) in segs:
        x0,x1 = max(0,int(min(ax,bx)-r)-1),min(w,int(max(ax,bx)+r)+2)   #only the box this segment can reach
        y0,y1 = max(0,int(min(ay,by)-r)-1),min(h,int(max(ay,by)+r)+2)
        if x1<=x0 or y1<=y0:
            continue
        yy,xx = np.meshgrid(np.arange(y0,y1,dtype=np.float64),
                            np.arange(x0,x1,dtype=np.float64),indexing='ij')
        vx,vy = bx-ax,by-ay
        L2 = vx*vx+vy*vy
        if L2<=1e-12:
            d = np.hypot(xx-ax,yy-ay)   #zero length segment, just distance to the point
        else:
            t = np.clip(((xx-ax)*vx+(yy-ay)*vy)/L2,0.0,1.0)   #how far along the segment, 0..1
            d = np.hypot(xx-(ax+t*vx),yy-(ay+t*vy))
        img[y0:y1,x0:x1] = np.where(d<=r,val,img[y0:y1,x0:x1])
    return img



#======================================================================
#TOOLKIT 9: measuring a piece of writing
#small measurements the rest of the pipeline asks for
#======================================================================
#----------------------------------------------------------------------
#EstimateSlantDeg: how far the writing leans
#shears the ink by each trial angle and keeps the angle where the
#column profile is sharpest, because upright strokes stack into tall
#narrow columns while leaning ones smear across many
#----------------------------------------------------------------------
def EstimateSlantDeg(ink):
    ys,xs = np.nonzero(ink)
    if len(xs)<50:
        return 0.0   #too little ink to measure, call it upright
    yc = ys.mean()
    best,bestScore = 0.0,-1.0
    for deg in np.arange(-45,45.5,1.5):
        xsh = (xs+(ys-yc)*-np.tan(np.radians(deg))).astype(np.int64)
        prof = np.bincount(xsh-xsh.min()).astype(np.float64)
        score = float((prof**2).sum())
        if score>bestScore:
            bestScore,best = score,float(deg)
    return -best   #positive means a normal forward lean, so deslanting is x -= tan(slant)*y

#---PolylineLength: total distance walked along a list of points---
def PolylineLength(p):
    a = np.asarray(p,dtype=np.float64)
    if len(a)<2:
        return 0.0
    return float(np.hypot(*np.diff(a,axis=0).T).sum())



#======================================================================
#THE SEGMENTER: page conditioning
#from here down is the segmenter proper, which uses the toolkit above
#======================================================================
#----------------------------------------------------------------------
#InkMask: which pixels are ink, allowing for uneven lighting
#compares each pixel to its OWN blurred background rather than to one
#number for the page, so a shadow across the paper does not become ink
#----------------------------------------------------------------------
def InkMask(gray):
    r = max(8,min(40,round(gray.shape[0]/41)))   #background blur radius scales with image size
    ratio = gray/np.maximum(BoxMean(gray,r),1.0)
    mask = ratio<Otsu(ratio)
    solid = BoxMean(mask.astype(np.float64),4)>0.999   #9x9 solid ink: binder interior, shadow
    return mask&~(BoxMean(solid.astype(np.float64),3)>0),ratio

#----------------------------------------------------------------------
#PaperRegionMask: find the sheet inside the photograph
#the biggest bright blob is the paper.  found on a shrunk copy for
#speed, then blown back up to full size
#----------------------------------------------------------------------
def PaperRegionMask(g,P):
    H,W = g.shape
    s = 4
    c = BoxMean(g[:H//s*s,:W//s*s].reshape(H//s,s,W//s,s).mean((1,3)),max(1,int(0.15*P/s)))
    dark = c<0.45*np.median(c)
    if dark.mean()<0.002:
        return np.ones((H,W),bool)   #no dark surround: nothing to cut away
    lab,n = LabelComponents(~dark)
    #biggest label is the sheet.  kron with a block of ones blows the
    #block averaged mask back up to the working image size
    big = np.kron(lab==np.bincount(lab.ravel())[1:].argmax()+1,np.ones((s,s),bool))
    full = np.ones((H,W),bool)
    full[:big.shape[0],:big.shape[1]] = big
    r = int(0.4*P)
    return ~AnyWithinRadius(AnyWithinRadius(~full,r,0),r,1)   #shrink a little from the paper edge

#---PEAK_MIN_PITCH_XH: a peak gap smaller than this many letter heights is---
#not the space between two lines, it is two bands of the same line.  real
#line spacing runs about 2.2 letter heights, so 1.5 leaves room for a
#tightly written page while still rejecting a within-line gap
PEAK_MIN_PITCH_XH = 1.5

#---MIN_LINE_PITCH: never report a pitch below this, in half-res pixels---
MIN_LINE_PITCH = 15

#---TRUST_ALL_PEAK_GAPS: True restores the behaviour before the fix of---
#2026-10-28, where every peak in the row profile counted as a line
TRUST_ALL_PEAK_GAPS = False


#----------------------------------------------------------------------
#_LetterHeight: how tall this page's letters are, in half-res pixels
#the median height of the ink blobs, ignoring specks and anything as tall
#as the page.  a letter is the one thing on a page whose size does not
#depend on knowing the line spacing first, so it is what the pitch
#estimate is sanity checked against
#----------------------------------------------------------------------
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
#LinePitch: the spacing between lines of writing
#looks at the gaps between peaks in the ink per row profile and takes
#the typical one.  everything later is measured in this pitch
#----------------------------------------------------------------------
def LinePitch(mask):
    W = mask.shape[1]   #peak gaps first, they track irregular paragraph spacing
    prof = mask[:,int(0.05*W):int(0.97*W)].sum(1).astype(float)
    k = np.exp(-0.5*(np.arange(-12,13)/4.0)**2)   #gaussian smoothing kernel
    sm = np.convolve(prof,k/k.sum(),"same")

    #how tall this page's letters are, worked out before the peaks are read
    #rather than only as a last resort: it is what says whether a gap between
    #two peaks is big enough to be the space between two lines
    h = _LetterHeight(mask)
    floor = (PEAK_MIN_PITCH_XH*h) if h else 0.0

    peaks = [i for i in range(1,len(sm)-1) if sm[i]>=sm[i-1] and sm[i]>sm[i+1] and sm[i]>0.06*sm.max()]
    if len(peaks)>=2:
        p = int(np.median(np.diff(sorted(peaks))))
        #a page holding ONE line still raises three peaks -- the ascenders,
        #the x-height band and the descenders -- and the gaps between those
        #are letter sized, not line sized.  taking them for line spacing put
        #the pitch at 15 where it should have been 51, and every length
        #downstream is measured in it, so the crop lost its ascenders and
        #descenders entirely
        if TRUST_ALL_PEAK_GAPS or p>=floor:
            return p,sm   #direct gap: works for 2+ lines, irregular spacing too

    a = sm-sm.mean()
    ac = np.correlate(a,a,"full")[len(a)-1:]   #autocorrelation fallback
    seg = ac[15:min(200,len(ac)//3)]
    idx = next((i for i in range(1,len(seg)-1)
                if seg[i]>=0.7*seg.max() and seg[i]>=seg[i-1] and seg[i]>=seg[i+1]), None)
    if idx is not None and (TRUST_ALL_PEAK_GAPS or 15+idx>=floor):
        return 15+idx,sm

    if h:
        return max(MIN_LINE_PITCH,int(round(2.2*h))),sm   #one line: go by letter size
    return 48,sm   #no usable ink: an arbitrary but harmless default

#----------------------------------------------------------------------
#LeastCostSeam: the cheapest path left to right across the page
#walks column by column, and for each row keeps the best running cost
#of reaching it, so the final path is the one crossing the least ink.
#used to cut between two lines without slicing through a letter
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

#----------------------------------------------------------------------
#SkewScore: how level the lines look at a trial angle
#adds up the square of the ink in each row.  when the page is level the
#ink piles into a few rows and the total is large, when it is tilted the
#same ink is spread over many rows and the total drops
#----------------------------------------------------------------------
def SkewScore(mask,angleDeg):
    ys,xs = np.nonzero(mask)
    if len(ys)==0:
        return 0.0
    h = mask.shape[0]
    cy,cx = (h-1)/2.0,(mask.shape[1]-1)/2.0
    th = np.radians(float(angleDeg))
    k = np.rint((ys-cy)*np.cos(th)-(xs-cx)*np.sin(th)+cy).astype(np.int64)   #which row each ink pixel lands in at this angle
    k -= k.min()   #shift to 0, do NOT clip to the page height: clipping piles
                   #off-page ink into one row and the squaring makes that fake
                   #spike score higher than any real line
    prof = np.bincount(k).astype(np.float64)
    return float((prof**2).sum())

#----------------------------------------------------------------------
#Deskew: straighten the page
#scores a range of angles with SkewScore, takes the best, and rotates.
#the search bins the ink by coordinate instead of rotating the image once
#per angle, which is why it is quick
#----------------------------------------------------------------------
def Deskew(page):


    #page is a 2-d grey uint8 array.  loading the photo is the caller's job,
    #see testing/ for the helper that does it
    full = np.asarray(page,dtype=np.uint8)
    if full.ndim==3:
        #average the colour channels, the whole block works on grey only
        full = np.rint(full[...,:3].mean(2)).astype(np.uint8)
    #2:1 box average.  box not point sampling, so a thin pale stroke lands
    #in some output pixel instead of vanishing between samples
    halfA = Halve(full)

    #1 deskew: rotate to whichever angle gives the sharpest row profile
    mask,_ = InkMask(halfA.astype(np.float64))
    #score the skew on a quarter size mask, 16x fewer pixels per angle
    q = Halve(mask.astype(np.uint8)*255)>127
    best = max(np.arange(-4,4.01,0.5),key=lambda a: SkewScore(q,a))
    #fill 255 because whatever rotates in from outside the photo is paper,
    #fill 0 above because there it is a mask and outside must add no ink
    fullA = Rotate(full,best,fill=255)
    halfA = Rotate(halfA,best,fill=255)
    g = halfA.astype(np.float64)
    mask,ratio = InkMask(g)
    H,W = mask.shape



    return fullA,halfA,mask,ratio,best

#----------------------------------------------------------------------
#RemovePrintedStructure: rub out the ruled lines
#masks off everything outside the sheet, then deletes long straight
#runs, which are printed rules and the binder edge, not writing
#----------------------------------------------------------------------
def RemovePrintedStructure(mask,g):


    P,_ = LinePitch(mask)
    paper = PaperRegionMask(g,P)
    paperRemoved = int((mask&~paper).sum())
    mask&=paper
    flat = AnyWithinRadius(LongRunMask(AnyWithinRadius(mask,1,0),int(1.25*P),1),1,0)&mask   #tolerant of a slight tilt
    upright = AnyWithinRadius(LongRunMask(AnyWithinRadius(mask,3,1),int(1.4*P),0),3,1)&mask   #wobbly vertical page edge too
    removed = int((flat|upright).sum())
    mask = mask&~flat&~upright



    return mask,paper,P,paperRemoved,removed

#----------------------------------------------------------------------
#DropArtifactPieces: throw away blobs that are not writing
#judges each piece as a whole: edge slivers, things far too big or
#too thin, and leftovers of the ruling.  whole pieces, so a letter is
#never half removed
#----------------------------------------------------------------------
def DropArtifactPieces(mask,ratio,P,H,W):


    rc = max(1,int(0.1*P))
    dil = AnyWithinRadius(AnyWithinRadius(mask,rc,0),rc,1)
    closed = ~AnyWithinRadius(AnyWithinRadius(~dil,rc,0),rc,1)   #tiny pen lift gaps joined
    lab,nc = LabelComponents(closed)
    ys,xs = np.nonzero(mask)
    pl = lab[ys,xs]
    area = np.bincount(pl,minlength=nc+1).astype(float)
    top,bot,xmin,xmax,_ = PieceBounds(pl,ys,xs,nc,H,W)
    tall = (bot-top+1)>2.2*P
    thin = (xmax-xmin+1)<=0.5*P
    edgeTol = max(3,int(0.5*P))   #this close to the paper boundary counts as page, not text
    touchL,touchR = xmin<=edgeTol,xmax>=W-1-edgeTol
    touchT,touchB = top<=edgeTol,bot>=H-1-edgeTol
    thinV = (bot-top+1)<=0.5*P
    letters = (area>=0.065*P*P)&~tall
    #where the text block starts on the left, judged from the letters alone.
    #with nothing that looks like a letter there is no margin to estimate, and
    #asking numpy for a percentile of an empty selection raises IndexError --
    #which reached the web server as a 500 instead of "no handwriting found".
    #-inf simply stops the left-of-margin test below from dropping anything
    xL = np.percentile(xmin[letters],5) if letters.any() else -np.inf
    vbar = ((xmax-xmin+1)<=0.12*P)&((bot-top+1)>=1.2*P)   #thin and taller than any letter
    otsuT = Otsu(ratio)
    #bincount with weights sums the ratio per piece, so dividing by the
    #pixel count gives each piece's mean darkness in one pass
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



    return mask,lab,nc,area,nArtifacts



#======================================================================
#THE SEGMENTER: finding and cutting the lines
#where each line sits, where to cut, and who owns each piece of ink
#======================================================================
#----------------------------------------------------------------------
#LineCentres: the row each line of writing sits on
#peaks of the ink per row profile, forced to stay at least a bit over
#half a pitch apart so one line cannot be counted twice
#----------------------------------------------------------------------
def LineCentres(mask,H):


    P,sm = LinePitch(mask)
    peaks = [i for i in range(1,H-1) if sm[i]>=sm[i-1] and sm[i]>sm[i+1] and sm[i]>0.06*sm.max()]
    centres = []
    for c in sorted(peaks,key=lambda i:-sm[i]):
        if all(abs(c-o)>=0.6*P for o in centres):
            centres.append(c)
    centres.sort()



    return P,centres

#----------------------------------------------------------------------
#SeamCuts: the cut between each pair of lines
#a least ink seam between neighbouring centres.  a straight horizontal cut
#would slice through descenders, a seam goes around them
#----------------------------------------------------------------------
def SeamCuts(mask,centres,P,H,W):


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
        mid,halfGap = c0+0.5*(c1-c0),max(1.0,(c1-c0)/2.0)   #a stray in the gap is usually the line above's tail
        #bigger per column step lets the cut react fast enough for a tall narrow tail, a light
        #pull back to the midline keeps it tracing real ink instead of a generic v shape
        cuts.append(c0+LeastCostSeam(4.0*m[c0:c1+1]+0.012*((yy-mid)/halfGap)**2,step=7))

    cuts.append(np.full(W,min(H-1,centres[-1]+int(0.75*P))))
    L = len(cuts)-1

    cutsArr = np.array(cuts)



    return cuts,cutsArr,L

#----------------------------------------------------------------------
#_BandOwnership: first guess at which line owns each piece
#a piece goes to the line holding most of it, so a tall letter is not
#split in half by the seam above or below it
#----------------------------------------------------------------------
def _BandOwnership(mask,lab,nc,cuts,cutsArr,L,P,H,W,centres):


    ys,xs = np.nonzero(mask)
    pl = lab[ys,xs]
    Y = np.arange(H)[:, None]
    #band index = how many seam boundaries sit above this pixel, so just
    #count the ones it is below.  no loop over pixels needed
    region = sum((Y>=cuts[n][None,:]).astype(np.int16) for n in range(1,L)) if L>1 else np.zeros((H,W),np.int16)
    rp = region[ys,xs]
    ok = (ys>=cutsArr[0][xs])&(ys<cutsArr[-1][xs])
    area = np.bincount(pl,minlength=nc+1).astype(float)

    A = np.bincount(pl,minlength=nc+1).astype(float)
    cx = np.bincount(pl,xs,nc+1)/np.maximum(A,1)
    cy = np.bincount(pl,ys,nc+1)/np.maximum(A,1)
    #pack (piece,band) into one integer piece*L+band so a single bincount
    #gives the whole piece by band table, then unpack it with reshape
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
    #a dot goes with the letter body directly below it, so every piece needs its box
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
                n2 = major[hit[0]]   #an i dot floats 0.25-0.7 spacings above its line's centre
                rows = big&(major==n2)&(np.abs(cx-cx[l])<=1.5*P)
                if rows.any() and 0.25*P<=np.median(cy[rows])-cy[l]<=0.7*P:
                    major[l],tiny[l] = n2, True
                break
    own = np.full((H,W),-1,np.int16)
    own[ys[ok],xs[ok]] = np.where(tiny[pl]|minor[pl,rp],major[pl],rp)[ok]   #a small cut off part rejoins its main part

    ownBeforeEF = own.copy()


    return own,ownBeforeEF,region,area,ys,xs

#----------------------------------------------------------------------
#_DotsJoinedAbove: fix an i dot welded to the line above
#closing small gaps can fuse a dot to a descender from the line above.
#a small piece sitting high goes back to the line it belongs to
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
                n2 = region[rb[d]+gap,max(0,rowLeft[d]-2)+np.nonzero(row==hit[0])[0][0]]   #line of the stem where it meets the dot
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



    return raw,nr,pr,Ar

#----------------------------------------------------------------------
#_RingsToStem: fix a loop welded to the stem below it
#same welding problem, the other way round: a ring sitting on a stem is
#given to the line the stem belongs to
#----------------------------------------------------------------------
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
            if LabelComponents(~np.pad(blob[:,bc.min():bc.max()+1],1))[1]<2:   #a dot ring encloses a hole, a hook does not
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

#----------------------------------------------------------------------
#_FragmentsToNeighbour: rehome a piece with no body ink
#a piece with nothing in its own x height band is not part of that
#line at all, so it goes to the neighbour it does reach into
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
                if bestBand is not None and abs(cyPiece-centres[bestBand])<=0.2*P:   #sits in the other line's x height band
                    moves.append((fy,fx,bestBand))
    for fy,fx,m in moves:
        own[fy,fx] = m

#----------------------------------------------------------------------
#AssignOwnership: the band guess plus the four corrections
#runs the default, then each fix in turn.  kept separate so a rule can
#be read, tested or turned off on its own
#----------------------------------------------------------------------
def AssignOwnership(mask,lab,nc,cuts,cutsArr,L,centres,P,H,W):
    own,ownBeforeEF,region,area,ys,xs = _BandOwnership(
        mask,lab,nc,cuts,cutsArr,L,P,H,W,centres)
    raw,nr,pr,Ar = _DotsJoinedAbove(
        own,mask,lab,area,region,centres,L,P,H,W,ys,xs)
    _RingsToStem(own,raw,nr,pr,Ar,lab,area,region,centres,P,H,W,ys,xs)
    _FragmentsToNeighbour(own,ownBeforeEF,mask,centres,L,P,H,W)
    return own

#----------------------------------------------------------------------
#_BuildLineCrop: one band in, one finished crop out
#takes the pieces this line owns, trims to the ink, pads a little and
#returns None if what is left is too small to be a line of writing
#----------------------------------------------------------------------
def _BuildLineCrop(n,own,norm,P,f,H,W,minInk):


    ink = own==n
    yy,xx = np.nonzero(ink)
    if len(yy)<minInk or yy.max()-yy.min()<max(6,0.2*P):   #a short lowercase word is still a real line
        return None
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
    alpha = Resize((inkC*255).astype(np.uint8),X1-X0,Y1-Y0,'linear').astype(np.float32)/255.0
    haloR = max(1,int(0.20*P))   #near this line's own ink only, bridges a letter the threshold missed
    halo = AnyWithinRadius(AnyWithinRadius(inkC,haloR,0),haloR,1)
    haloAlpha = Resize((halo*255).astype(np.uint8),X1-X0,Y1-Y0,'linear').astype(np.float32)/255.0
    region = norm[Y0:Y1,X0:X1]
    localR = max(3,min(20,round(region.shape[0]/8)))
    localRatio = region/np.maximum(BoxMean(region,localR),1.0)   #this line's own contrast, not the whole page's
    localMask = localRatio<Otsu(localRatio)
    ownC = own[y0:y1,x0:x1]
    notOther = ~((ownC>=0)&(ownC!=n))   #never recover a pixel the page already gave to a neighbour
    #nearest here, not bilinear: this is a yes/no ownership flag and a
    #blurred edge would half give a pixel to two lines at once
    notOtherFull = Resize((notOther*255).astype(np.uint8),X1-X0,Y1-Y0,'nearest')>127
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
    return dict(image=crop,
                clf=FitFixedCanvas(crop,InputHeight,InputWidth),
                bbox=(X0,Y0,X1,Y1))

#----------------------------------------------------------------------
#SegmentLines: the whole job, one call
#deskew, remove the printing, drop artefacts, find the centres, cut the
#seams, assign ownership, build the crops.  each stage is its own
#function above so the sequence here reads as the method
#----------------------------------------------------------------------
def SegmentLines(page,timer=None):
    #timer is optional and changes nothing about the result.  when the web
    #server passes one in, the three phases below are recorded separately so
    #the page can report conditioning apart from segmentation, which is what
    #the timing requirements are written against
    import time as _t
    marks = []
    def Mark(name):
        marks.append((name,_t.perf_counter()))
    Mark("start")

    fullA,halfA,mask,ratio,best = Deskew(page)
    g = halfA.astype(np.float64)
    H,W = mask.shape
    mask,paper,P,paperRemoved,removed = RemovePrintedStructure(mask,g)
    mask,lab,nc,area,nArtifacts = DropArtifactPieces(mask,ratio,P,H,W)
    Mark("conditioning")                 #flatten, deskew, drop the printing and the specks
    P,centres = LineCentres(mask,H)
    cuts,cutsArr,L = SeamCuts(mask,centres,P,H,W)
    own = AssignOwnership(mask,lab,nc,cuts,cutsArr,L,centres,P,H,W)
    Mark("segmenting")                   #where the lines are and which ink belongs to which

    #crops come off the FULL resolution page, so the recogniser gets every
    #pixel the camera caught though every decision above was at half size
    r = max(8,min(40,round(H/41)))   #same scale aware radius as InkMask
    bgFull = Resize(BoxMean(g,r).astype(np.float32),fullA.shape[1],fullA.shape[0],'linear')
    norm = np.clip(fullA.astype(np.float32)/np.maximum(bgFull,1.0),0,1)*255
    f = fullA.shape[1]/W   #half size coordinate to full size
    perLine = np.bincount(own[own>=0],minlength=L)
    minInk = max(60,0.12*np.median(perLine[perLine>0]))
    lines = []
    for n in range(L):
        crop = _BuildLineCrop(n,own,norm,P,f,H,W,minInk)
        if crop is None:
            continue
        crop['order'] = len(lines)
        lines.append(crop)
    Mark("cropping")                     #cut each line out of the full resolution page

    if timer is not None:
        for i in range(1,len(marks)):
            timer.Add(marks[i][0],marks[i][1]-marks[i-1][1])

    #cuts and the ratio image go back raw so the web code can draw an overlay
    return lines,dict(cuts=cuts,ratio=ratio,paper=paper),dict(
        angle=float(best),pitch=int(P),
        paperCoverage=round(float(paper.mean()),3),
        inkOutsidePaper=paperRemoved,removedStraightPixels=removed,
        artifactPieces=nArtifacts,nLines=len(lines))
