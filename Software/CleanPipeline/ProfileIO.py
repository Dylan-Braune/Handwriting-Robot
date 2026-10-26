#the handful of things the live server needs from the style profile pipeline:
#loading a saved profile, forced aligning a recognizer's output against known
#text, and the binarize/skeletonize/band primitives used for stroke normalisation.
#kept separate from BuildStyleProfile.py's ~1000 lines of offline glyph library
#building code, which is only ever run by hand, never touched by a live request
from pathlib import Path

import numpy as np
from PIL import Image

from Charset import CHAR_TO_IDX

#this file lives in Software/CleanPipeline, a sibling of Software/CNN where
#the actual style profiles still live -- data is not duplicated
CNN_DATA_DIR = Path(__file__).resolve().parent.parent/"CNN"
PROFILE_DIR = CNN_DATA_DIR/"NOGIT"/"StyleProfiles10"
PEN_WIDTH_XH = 0.14   #constant pen width used to normalise a stroke, in x-heights


#summed area table (integral image) with a zero row/col on top/left
def BoxIntegral(img):
    ii = np.zeros((img.shape[0]+1,img.shape[1]+1),np.float64)
    np.cumsum(np.cumsum(img,axis=0),axis=1,out=ii[1:,1:])
    return ii


#sum over a (2ry+1)x(2rx+1) window centred per pixel, edge clamped
def BoxSum(img,ry,rx):
    h,w = img.shape
    ii = BoxIntegral(img)
    y = np.arange(h)
    x = np.arange(w)
    y1 = np.clip(y-ry,0,h)[:, None]
    y2 = np.clip(y+ry+1,0,h)[:, None]
    x1 = np.clip(x-rx,0,w)[None,:]
    x2 = np.clip(x+rx+1,0,w)[None,:]
    return ii[y2,x2]-ii[y1,x2]-ii[y2,x1]+ii[y1,x1]


#pixel count of the clamped window at each position, for the erode threshold
def BoxCount(ry,rx,h,w):
    y = np.arange(h)
    x = np.arange(w)
    cy = (np.clip(y+ry+1,0,h)-np.clip(y-ry,0,h))[:, None]
    cx = (np.clip(x+rx+1,0,w)-np.clip(x-rx,0,w))[None,:]
    return cy*cx


#otsu's method: the threshold that maximises between class variance
def OtsuThresholdValue(gray):
    hist = np.bincount(gray.ravel(),minlength=256).astype(np.float64)
    total = hist.sum()
    sumAll = np.dot(np.arange(256),hist)
    sumB = wB = 0.0
    maxVar,thresh = 0.0,0
    for t in range(256):
        wB+=hist[t]
        if wB==0:
            continue
        wF = total-wB
        if wF==0:
            break
        sumB+=t*hist[t]
        mB = sumB/wB
        mF = (sumAll-sumB)/wF
        var = wB*wF*(mB-mF)**2   #between class variance at this split
        if var>maxVar:
            maxVar,thresh = var,t
    return thresh


#binary dilation via box sum, kernel ky x kx (odd sizes, centre anchored)
def Dilate(mask,ky,kx):
    ry,rx = ky//2,kx//2
    s = BoxSum(mask.astype(np.float64),ry,rx)
    return s>0.5


#binary erosion via box sum: every pixel in the window must be set
def Erode(mask,ky,kx):
    ry,rx = ky//2,kx//2
    s = BoxSum(mask.astype(np.float64),ry,rx)
    cnt = BoxCount(ry,rx,*mask.shape)
    return s>=cnt-0.5


#morphological closing: dilate then erode, joins hairline gaps
def Close(mask,ky,kx):
    return Erode(Dilate(mask,ky,kx),ky,kx)


#union find with path compression, used by the run based component labeller
class RunUnionFind:
    __slots__ = ('parent',)

    def __init__(self,n):
        self.parent = list(range(n))

    def find(self,a):
        p = self.parent
        while p[a]!=a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self,a,b):
        ra,rb = self.find(a),self.find(b)
        if ra!=rb:
            self.parent[ra] = rb


#per row (start,end) column index pairs of ink runs
def RowRuns(mask):
    h,w = mask.shape
    padded = np.zeros((h,w+2),bool)
    padded[:,1:-1] = mask
    d = np.diff(padded.astype(np.int8),axis=1)
    runs = []
    for r in range(h):
        starts = np.nonzero(d[r]==1)[0]
        ends = np.nonzero(d[r]==-1)[0]
        runs.append((starts,ends))
    return runs


#8 connected component labelling via row runs and union find, returns (labels,count)
def LabelComponents(mask,connectivity=8):
    h,w = mask.shape
    rowRuns = RowRuns(mask)

    runIdStart = []
    total = 0
    for r in range(h):
        runIdStart.append(total)
        total+=len(rowRuns[r][0])
    if total==0:
        return np.zeros((h,w),np.int32),0

    uf = RunUnionFind(total)
    pad = 1 if connectivity==8 else 0
    for r in range(1,h):
        s1,e1 = rowRuns[r-1]
        s2,e2 = rowRuns[r]
        if len(s1)==0 or len(s2)==0:
            continue
        i = j = 0
        while i<len(s1) and j<len(s2):   #runs [s1[i],e1[i]) and [s2[j],e2[j]) touch?
            if s1[i]<e2[j]+pad and s2[j]<e1[i]+pad:
                uf.union(runIdStart[r-1]+i,runIdStart[r]+j)
            if e1[i]<e2[j]:
                i+=1
            else:
                j+=1

    rootLabel = {}
    runLabel = np.zeros(total,np.int32)
    nextLabel = 1
    for k in range(total):
        root = uf.find(k)
        lab = rootLabel.get(root)
        if lab is None:
            lab = nextLabel
            rootLabel[root] = lab
            nextLabel+=1
        runLabel[k] = lab

    labels = np.zeros((h,w),np.int32)
    for r in range(h):
        starts,ends = rowRuns[r]
        base = runIdStart[r]
        for i in range(len(starts)):
            labels[r,starts[i]:ends[i]] = runLabel[base+i]
    return labels,nextLabel-1


#ctc forced alignment: viterbi over the blank interleaved label sequence,
#returns per character (t0,t1) timestep spans, or none if it cannot align
def CtcForcedAlign(logProbs,text):
    labels = [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]
    if not labels:
        return None
    ext = [0]
    for l in labels:
        ext+=[l,0]
    S,T = len(ext),logProbs.shape[0]
    if T<len(labels):
        return None
    NEG = -1e30
    dp = np.full((T,S),NEG)
    bp = np.zeros((T,S),np.int32)
    dp[0,0] = logProbs[0,ext[0]]
    if S>1:
        dp[0,1] = logProbs[0,ext[1]]
    for t in range(1,T):
        emit = logProbs[t,ext]
        stay = dp[t-1]
        prev1 = np.concatenate(([NEG],dp[t-1,:-1]))
        prev2 = np.concatenate(([NEG,NEG],dp[t-1,:-2]))
        for s in range(S):   #skip only onto a non blank that differs two back, standard ctc topology
            if s>=2 and (ext[s]==0 or ext[s]==ext[s-2]):
                prev2[s] = NEG
        cand = np.stack([stay,prev1,prev2])
        best = np.argmax(cand,axis=0)
        dp[t] = cand[best,np.arange(S)]+emit
        bp[t] = best
    endS = S-1 if S==1 else (S-1 if dp[T-1,S-1]>=dp[T-1,S-2] else S-2)
    path = np.zeros(T,np.int32)
    s = endS
    for t in range(T-1,-1,-1):
        path[t] = s
        s-=bp[t,s]
    spans = []
    for ci in range(len(labels)):
        sIdx = 1+2*ci
        ts = np.nonzero(path==sIdx)[0]
        if len(ts)==0:
            spans.append(None)
        else:
            spans.append((int(ts[0]),int(ts[-1])+1))
    kept = [c for c in text if c in CHAR_TO_IDX]
    conf = float(np.mean([logProbs[t,ext[path[t]]] for t in range(T)]))   #mean per frame log prob along the path
    return list(zip(kept,spans)),conf


#zhang suen thinning, vectorised, reduces a stroke to a 1 pixel wide centreline
def Skeletonize(mask):
    img = mask.astype(np.uint8).copy()

    #8 neighbour ring around every pixel, padded
    def neighbours(P):
        p = np.pad(P,1)
        return (p[:-2,1:-1],p[:-2,2:],p[1:-1,2:],p[2:,2:],
                p[2:,1:-1],p[2:,:-2],p[1:-1,:-2],p[:-2,:-2])

    changed = True
    while changed:
        changed = False
        for step in (0,1):   #two sub iterations per pass, standard zhang suen
            P2,P3,P4,P5,P6,P7,P8,P9 = neighbours(img)
            ring = [P2,P3,P4,P5,P6,P7,P8,P9,P2]
            B = sum(ring[:8])   #number of set neighbours
            A = sum(((ring[k]==0)&(ring[k+1]==1)).astype(np.uint8)
                    for k in range(8))   #number of 0 to 1 transitions around the ring
            if step==0:
                c1 = (P2*P4*P6)==0
                c2 = (P4*P6*P8)==0
            else:
                c1 = (P2*P4*P8)==0
                c2 = (P2*P6*P8)==0
            cond = (img==1)&(B>=2)&(B<=6)&(A==1)&c1&c2
            if cond.any():
                img[cond] = 0
                changed = True
    return img.astype(bool)


#horizontal run length at every pixel
def HorizontalRunLength(mask):
    h,w = mask.shape
    a = mask.astype(np.int32)
    starts = np.zeros_like(a)
    starts[:,0] = a[:,0]
    starts[:,1:] = (a[:,1:]==1)&(a[:,:-1]==0)
    runId = np.cumsum(starts.reshape(-1)).reshape(h,w)
    runId[a==0] = 0
    if runId.max()==0:
        return np.zeros_like(a)
    counts = np.bincount(runId.reshape(-1))
    counts[0] = 0
    return counts[runId]


#binarises a line and removes underlines/ruled lines: a stroke running far
#longer than a letter while staying pen width thin is a rule, not handwriting.
#leaving them in glues every letter together and drags the slant estimate flat
def BinarizeLine(gray,stripRules=True,closeGaps=True):
    thr = OtsuThresholdValue(gray)
    ink = gray<thr
    if stripRules and ink.any():
        h = ink.shape[0]
        hRun = HorizontalRunLength(ink)
        vRun = HorizontalRunLength(ink.T).T
        rule = (hRun>=max(24,int(1.1*h)))&(vRun<=max(3,int(0.12*h)))
        if rule.any():
            ink = ink&~(rule&(vRun<=max(3,int(0.12*h))))   #keep real strokes crossing the rule
    if closeGaps and ink.any():
        #a light fast hand breaks thin strokes into pieces when scanned, a small
        #closing rejoins a hairline break without merging neighbouring letters
        h = ink.shape[0]
        k = int(np.clip(round(0.045*h),2,4))
        ink = Close(ink,k,k)
    labels,n = LabelComponents(ink,connectivity=8)
    if n:
        areas = np.bincount(labels.ravel())
        keep = areas>=6
        keep[0] = False
        ink = keep[labels]
    return ink


#baseline/topline of the x height band, rows with >=45% of peak ink density
def CoreBand(ink):
    prof = ink.sum(axis=1).astype(np.float64)
    if prof.max()<=0:
        return None
    rows = np.nonzero(prof>=0.45*prof.max())[0]
    top,base = int(rows[0]),int(rows[-1])
    if base-top<3:
        return None
    return top,base


#real ink -> centreline -> re-inked at one constant width -> inverted. removes the
#author's own pen/pressure while keeping every geometric property of the writing,
#the same transform the gantry performs when it redraws a hand with its own pen
def StrokeNormalize(gray,penWidthXh=PEN_WIDTH_XH):
    ink = BinarizeLine(gray)
    if not ink.any():
        return None
    band = CoreBand(ink)
    xh = float(band[1]-band[0]) if band else 0.0
    if xh<4:
        xh = max(8.0,ink.shape[0]/3.0)
    w = int(np.clip(round(penWidthXh*xh),2,9))
    sk = Skeletonize(ink)
    if not sk.any():
        return None
    k = 2*(w//2)+1
    out = Dilate(sk,k,k)
    return Image.fromarray(((~out)*255).astype(np.uint8))


#shear search: the slant is whichever shear makes the column projection sharpest.
#positive = normal forward lean, so deslanting is x -= tan(slant)*y, y up from baseline
def EstimateSlantDeg(ink):
    ys,xs = np.nonzero(ink)
    if len(xs)<50:
        return 0.0
    yc = ys.mean()
    best,bestScore = 0.0,-1.0
    for deg in np.arange(-45,45.5,1.5):
        sh = np.tan(np.radians(deg))
        xsh = (xs+(ys-yc)*-sh).astype(np.int64)
        prof = np.bincount(xsh-xsh.min())
        score = float((prof.astype(np.float64)**2).sum())
        if score>bestScore:
            bestScore,best = score,float(deg)
    return -best
