#u23539195 Dylan Braune.  my own image geometry code, written for this
#project.  this is NOT a library, every function here is hand written
#
#why this file exists
#reading and writing png files is file i/o and stays with pil, but every
#operation that CHANGES pixel values is part of the design.  resizing a crop,
#rotating a page level, warping a training image, blurring a stroke and
#drawing the synthesised ink are all things the networks are then judged on,
#so they are written here instead of called from a library
#
#why not just take the nearest source pixel when resizing
#line crops are nearly always SHRUNK on the way to the 64x640 classifier
#input, a crop is about 130 px tall.  point sampling a shrink skips whole
#rows, so a one pixel pen stroke that falls between two sample positions
#vanishes and thin strokes break up.  the fix is standard, the filter must
#get WIDER as the image gets smaller so every input pixel lands in some
#output window.  that is what filterScale does below, it turns the filter
#into an area average when shrinking and plain linear interpolation when
#growing
import numpy as np


#--- shared bilinear sampler ---------------------------------------------
#takes a float source position per output pixel and reads the source with
#bilinear interpolation.  used by Rotate and Warp so the sampling is written
#once and both cannot drift apart
#
#bilinear means blend the 4 pixels around the source point
#  top = p00*(1-fx) + p01*fx
#  bot = p10*(1-fx) + p11*fx
#  out = top*(1-fy) + bot*fy
#where fx fy are the fractional parts of the source position
#edge tells it what counts as "still on the image"
#  'area'   pixel i covers i-0.5 to i+0.5 so the image spans -0.5 to w-0.5
#           and a point at x = -0.3 still reads the edge pixel.  used by
#           Rotate, where the whole page is valid content and a half pixel
#           strip of fill along the top would be wrong
#  'centre' the point must land between the first and last pixel CENTRE.
#           used by Warp, where a shear pushes part of the canvas off the
#           image and that part must come back as fill, not as a smeared
#           copy of the last row
def _SampleBilinear(a,sx,sy,fill,edge='area'):
    h,w = a.shape[0],a.shape[1]
    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    fx,fy = sx-x0,sy-y0
    if edge=='area':
        inside = (sx>-0.5)&(sx<w-0.5)&(sy>-0.5)&(sy<h-0.5)
    else:
        inside = (sx>=0)&(sx<=w-1)&(sy>=0)&(sy<=h-1)
    #clamp the pixel and its neighbour to the last real index.  at the edge
    #the two then point at the same pixel so the blend just returns it
    xi = np.clip(x0,0,w-1)
    yi = np.clip(y0,0,h-1)
    xi1 = np.clip(x0+1,0,w-1)
    yi1 = np.clip(y0+1,0,h-1)
    p00,p01 = a[yi,xi],a[yi,xi1]
    p10,p11 = a[yi1,xi],a[yi1,xi1]
    top = p00*(1-fx)+p01*fx
    bot = p10*(1-fx)+p11*fx
    return np.where(inside,top*(1-fy)+bot*fy,float(fill))


#rounds a float result back to the dtype the caller gave us
def _AsInput(out,like):
    if np.issubdtype(np.asarray(like).dtype,np.integer):
        return np.clip(np.rint(out),0,255).astype(np.uint8)
    return out


#--- separable resampling -------------------------------------------------
#one output pixel is a weighted sum of a run of input pixels.  both axes use
#the same 1-d construction so the weights are built once per axis as an
#(outSize,inSize) matrix and applied with two matrix multiplies

#triangle or linear kernel, full weight at the centre falling to zero one
#filter width away.  w(t) = 1-|t| for |t|<1 else 0
def _Triangle(t):
    a = np.abs(t)
    return np.where(a<1.0,1.0-a,0.0)


#box kernel, equal weight across the window and zero outside.  cheaper and
#sharper than the triangle when halving by an exact integer factor
def _Box(t):
    return np.where(np.abs(t)<=0.5,1.0,0.0)


_KERNELS = {'linear': (_Triangle,1.0), 'box': (_Box,0.5)}


#weights mapping inSize input samples to outSize output samples.
#centre of output pixel i sits at (i+0.5)*scale in input coordinates, and
#the window round it is `support` wide.  support is widened by the reduction
#factor so a shrink averages instead of skipping
def _ResampleWeights(inSize,outSize,kernel):
    fn,support = _KERNELS[kernel]
    scale = inSize/float(outSize)
    #only widen when shrinking, scale>1 means shrinking
    filterScale = max(1.0,scale)
    support = support*filterScale
    centres = (np.arange(outSize)+0.5)*scale
    #widest window any output pixel needs, so all the weights fit one array
    half = int(np.ceil(support))+1
    offs = np.arange(-half,half+1)
    cols = np.floor(centres)[:,None]+offs[None,:]
    #distance from the window centre measured in filter widths
    t = (cols+0.5-centres[:,None])/filterScale
    w = fn(t)
    #clamp to the edge instead of letting the window run off the image.  an
    #edge pixel's weight is folded onto the nearest real column which keeps
    #each row summing to 1 and stops a dark border appearing on a padded crop
    colsClamped = np.clip(cols,0,inSize-1).astype(np.int64)
    out = np.zeros((outSize,inSize),np.float64)
    #add.at because several offsets can clamp onto the SAME column at an
    #edge and their weights must add up, a plain assignment would keep only
    #the last one and darken the border
    np.add.at(out,(np.repeat(np.arange(outSize),cols.shape[1]),
                   colsClamped.ravel()),w.ravel())
    tot = out.sum(axis=1,keepdims=True)
    return out/np.where(tot>0,tot,1.0)


#resample a 2-d (H,W) or 3-d (H,W,C) array to (newH,newW).
#kernel 'linear' for general scaling, 'box' for an exact integer shrink
def Resize(arr,newW,newH,kernel='linear'):
    a = np.asarray(arr)
    newW,newH = max(1,int(newW)),max(1,int(newH))
    h,w = a.shape[0],a.shape[1]
    if h==newH and w==newW:
        return a.copy()
    #nearest just picks the closest source pixel, no blending.  needed where
    #the values are LABELS and not brightness: the ownership map holds a
    #line index per pixel, and averaging index 3 and index 4 would give 3.5
    #which is a line that does not exist
    if kernel=='nearest':
        srcY = np.clip(((np.arange(newH)+0.5)*(h/float(newH))).astype(np.int64),0,h-1)
        srcX = np.clip(((np.arange(newW)+0.5)*(w/float(newW))).astype(np.int64),0,w-1)
        return a[srcY[:,None],srcX[None,:]]
    f = a.astype(np.float64)
    #rows first then columns, the two passes are independent so the order
    #does not matter to the result
    if newH!=h:
        f = np.tensordot(_ResampleWeights(h,newH,kernel),f,axes=([1],[0]))
    if newW!=w:
        f = np.tensordot(f,_ResampleWeights(w,newW,kernel).T,
                         axes=([1],[0]) if f.ndim==2 else ([1],[0]))
        if f.ndim==3:
            #tensordot leaves the width axis at the end for a 3-d array so
            #put it back in the middle, shape goes (H,C,W) -> (H,W,C)
            f = np.moveaxis(f,-1,1)
    if np.issubdtype(a.dtype,np.integer):
        info = np.iinfo(a.dtype)
        return np.clip(np.rint(f),info.min,info.max).astype(a.dtype)
    return f.astype(a.dtype)


#halve both axes by exact box averaging, which is what the page reduction
#wants.  kept separate from Resize only so the call site reads clearly
def Halve(arr):
    a = np.asarray(arr)
    return Resize(a,a.shape[1]//2,a.shape[0]//2,'box')


#--- fixed canvas fit -----------------------------------------------------
#scales a line crop by ONE factor on both axes, never squashed unevenly
#because that distorts stroke proportions, then lays it on a padded canvas
#left aligned and vertically centred.  this is the exact geometry every
#classifier in this project was trained with so it must not change
def FitFixedCanvas(gray,height,width,padValue=255):
    a = np.asarray(gray)
    h,w = a.shape[0],a.shape[1]
    canvas = np.full((height,width),padValue,np.uint8)
    if w<=0 or h<=0:
        return canvas
    #one scale, whichever axis runs out of room first
    s = min(height/float(h),width/float(w))
    newW,newH = max(1,int(round(w*s))),max(1,int(round(h*s)))
    small = Resize(a.astype(np.uint8),newW,newH,'linear')
    y0 = max(0,(height-newH)//2)
    #the slices guard against a rounding overshoot of one pixel
    canvas[y0:y0+newH,:newW] = small[:height-y0,:width]
    return canvas


#--- rotation -------------------------------------------------------------
#rotates about the image centre by `deg`, keeps the same canvas size and
#fills whatever rotates in from outside with `fill`
#
#inverse mapping, for every OUTPUT pixel work out where it came FROM in the
#input.  doing it forwards would leave holes because two output pixels can
#claim the same source pixel and others none
def Rotate(gray,deg,fill=255):
    a = np.asarray(gray).astype(np.float64)
    h,w = a.shape[0],a.shape[1]
    th = np.radians(float(deg))
    c,s = np.cos(th),np.sin(th)
    cy,cx = (h-1)/2.0,(w-1)/2.0
    yy,xx = np.meshgrid(np.arange(h,dtype=np.float64),
                        np.arange(w,dtype=np.float64),indexing='ij')
    dx,dy = xx-cx,yy-cy
    #image rows run DOWNWARD so a counter clockwise turn on screen is a
    #clockwise one in (row,column) terms.  that is why the signs below are
    #the transpose of the textbook rotation matrix, and it makes `deg` mean
    #the same thing here as in the deskew search
    #  sx = cos*dx - sin*dy + cx
    #  sy = sin*dx + cos*dy + cy
    sx = c*dx-s*dy+cx
    sy = s*dx+c*dy+cy
    return _AsInput(_SampleBilinear(a,sx,sy,fill),gray)


#--- affine and perspective warp -----------------------------------------
#one function for both, the only difference is how many coefficients.
#coefficients map an OUTPUT pixel back to its SOURCE, same convention the
#augmentation code already solves for
#
#6 coefficients, affine
#  sx = a*x + b*y + c
#  sy = d*x + e*y + f
#8 coefficients, perspective, divide by a third row that varies with x,y
#  den = g*x + h*y + 1
#  sx  = (a*x + b*y + c)/den
#  sy  = (d*x + e*y + f)/den
def Warp(gray,coeffs,fill=255):
    a = np.asarray(gray).astype(np.float64)
    h,w = a.shape[0],a.shape[1]
    yy,xx = np.meshgrid(np.arange(h,dtype=np.float64),
                        np.arange(w,dtype=np.float64),indexing='ij')
    #the coefficients act on the CENTRE of the output pixel, not its index.
    #pixel i covers i-0.5 to i+0.5 so its centre is at i+0.5, and the
    #answer comes back as a centre too and has 0.5 taken off again to turn
    #it into an index.  leaving these two halves out shifts the whole result
    #by half a pixel, which is small but shows up as a blur on thin strokes
    X,Y = xx+0.5,yy+0.5
    k = [float(v) for v in coeffs]
    if len(k)==6:
        sx = k[0]*X+k[1]*Y+k[2]-0.5
        sy = k[3]*X+k[4]*Y+k[5]-0.5
    elif len(k)==8:
        den = k[6]*X+k[7]*Y+1.0
        #a denominator at zero is a point sent to infinity.  push it to a
        #tiny number so the divide is safe, the pixel then lands far off the
        #image and gets the fill value anyway
        den = np.where(np.abs(den)<1e-12,1e-12,den)
        sx = (k[0]*X+k[1]*Y+k[2])/den-0.5
        sy = (k[3]*X+k[4]*Y+k[5])/den-0.5
    else:
        raise ValueError("Warp needs 6 coefficients for affine or 8 for perspective")
    return _AsInput(_SampleBilinear(a,sx,sy,fill),gray)


#--- gaussian blur --------------------------------------------------------
#separable gaussian, blur down the rows then across the columns.  a 2-d
#gaussian is the product of two 1-d ones so two 1-d passes give the same
#answer as one 2-d kernel for far less work, (2r+1)+(2r+1) multiplies per
#pixel instead of (2r+1)*(2r+1)
#
#  k[t] = exp(-t*t / (2*sigma*sigma))   then divided by its own sum
#
#cut off at 3 sigma because the weight past that is under 0.3 percent
def Blur(gray,sigma):
    a = np.asarray(gray).astype(np.float64)
    s = float(sigma)
    if s<=0.0:
        return np.asarray(gray).copy()
    r = max(1,int(np.ceil(3.0*s)))
    t = np.arange(-r,r+1,dtype=np.float64)
    k = np.exp(-(t*t)/(2.0*s*s))
    k /= k.sum()
    #edge replication, repeat the border pixel outwards.  zero padding would
    #pull a dark edge into a white page
    pad = np.pad(a,((r,r),(r,r)),mode='edge')
    #rows pass, sum of 2r+1 shifted copies each times its weight
    tmp = np.zeros_like(a)
    for i,wgt in enumerate(k):
        tmp += wgt*pad[i:i+a.shape[0],r:r+a.shape[1]]
    #columns pass, same again on the result of the rows pass
    tmp2 = np.pad(tmp,((0,0),(r,r)),mode='edge')
    out = np.zeros_like(a)
    for i,wgt in enumerate(k):
        out += wgt*tmp2[:,i:i+a.shape[1]]
    return _AsInput(out,gray)


#--- drawing the synthesised ink -----------------------------------------
#draws one stroke as a thick polyline straight into a uint8 array.
#
#a thick segment is every pixel whose distance to the segment is at most
#half the pen width, which is a rectangle with a half circle on each end.
#doing it by distance rather than by filling a rotated rectangle means the
#round pen ends and the round joins between segments come out for free, no
#separate end caps and no mitre to get wrong
#
#distance from a point P to the segment A..B
#  t = clamp( dot(P-A, B-A) / |B-A|^2 , 0, 1)    how far along, 0..1
#  C = A + t*(B-A)                              nearest point on segment
#  d = |P - C|
def DrawPolyline(img,pts,value,width):
    if len(pts)==0:
        return img
    h,w = img.shape[0],img.shape[1]
    r = max(0.5,float(width)/2.0)
    val = np.uint8(np.clip(round(float(value)),0,255))
    #a single point is still a dot of ink, treat it as a zero length segment
    segs = [(pts[i],pts[i+1]) for i in range(len(pts)-1)] or [(pts[0],pts[0])]
    for (ax,ay),(bx,by) in segs:
        #only look at the box the segment can possibly touch, +1 for rounding
        x0 = int(np.floor(min(ax,bx)-r))-1
        x1 = int(np.ceil(max(ax,bx)+r))+1
        y0 = int(np.floor(min(ay,by)-r))-1
        y1 = int(np.ceil(max(ay,by)+r))+1
        x0,y0 = max(0,x0),max(0,y0)
        x1,y1 = min(w,x1),min(h,y1)
        if x1<=x0 or y1<=y0:
            continue
        yy,xx = np.meshgrid(np.arange(y0,y1,dtype=np.float64),
                            np.arange(x0,x1,dtype=np.float64),indexing='ij')
        vx,vy = bx-ax,by-ay
        L2 = vx*vx+vy*vy
        if L2<=1e-12:
            #zero length, distance is just to the single point
            d = np.hypot(xx-ax,yy-ay)
        else:
            t = ((xx-ax)*vx+(yy-ay)*vy)/L2
            t = np.clip(t,0.0,1.0)
            d = np.hypot(xx-(ax+t*vx),yy-(ay+t*vy))
        img[y0:y1,x0:x1] = np.where(d<=r,val,img[y0:y1,x0:x1])
    return img
