#u23539195 Dylan Braune.  building a writer's style profile.  FU 5, R5
#
#the job: take pages somebody wrote, cut out every letter, throw away the bad
#cuts, and store what is left as that writer's handwriting, measured in
#x-heights so it can be redrawn at any size
#
#the order below is the order the work happens in:
#  trace     ink to centre lines
#  cut       centre lines to one glyph per character, using the reader's
#            own idea of where each character sits
#  filter    five gates, cheapest first, drop what is not a letter
#  measure   slant, x-height, spacing, how joined up the hand is
import json
import random

import numpy as np

import paths
from recognise import (CtcForcedAlign,CHARSET,CHAR_TO_IDX,ReadText,
                       ResizeLineImageFixed,TensorFromResized,
                       FrameXToPixel,INPUT_WIDTH,INPUT_HEIGHT)
from segment import (BinarizeLine,CoreBand,EstimateSlantDeg,LabelComponents,PolylineLength,Skeletonize)


#======================================================================
#WHERE THE DATA IS, AND THE LIMITS THE FILTERS USE
#every number a filter compares against is here, named, so it can be tuned
#======================================================================
#---DATA_DIR: the data folder beside the code---
DATA_DIR = paths.Data.parents[1]/"Data"/"Datasets"/"IAMpages10"

#---CACHE_DIR: cached extractions, so a rebuild need not redo the cutting---
CACHE_DIR = paths.Cache/"line_cache_authors10"

#---NOGIT_DIR: the big files that stay off git---
NOGIT_DIR = paths.Cache

#---TEXT_WEIGHTS: the reader used to place the character boundaries---
TEXT_WEIGHTS = paths.Weights/"paper_cnn_bilstm_ctc_best.npz"

#---PERSONAL_WEIGHTS: the reader tuned on the photographed writers---
PERSONAL_WEIGHTS = paths.Weights/"paper_cnn_bilstm_ctc_personal_best.npz"

#---RAW_DIR: where the uncut extraction cache per writer lives---
RAW_DIR = paths.Cache/"GlyphCache10"

#---MAX_VARIANTS_PER_CHAR: how many shapes of one letter are kept---
MAX_VARIANTS_PER_CHAR = 12

#---ALIGN_CONF_PCT: only trust boundaries from lines the reader read confidently---
ALIGN_CONF_PCT = 25

#---FRAGMENT_PEN_FRAC: a letter needs this much of the usual pen path, else it is a fragment---
FRAGMENT_PEN_FRAC = 0.5

#---MIN_PEN_ABS: and never less than this, in x-heights---
MIN_PEN_ABS = 0.30

#---MIN_LONGEST_STROKE: its longest single stroke must be at least this---
MIN_LONGEST_STROKE = 0.55

#---DOT_MIN_AREA: smaller than this and it is dust, not an i dot---
DOT_MIN_AREA = 0.004

#---MAX_GLYPH_W: wider than this means the cut swallowed a neighbour---
MAX_GLYPH_W = 2.1

#---MAX_GLYPH_W_WIDE: the same limit for letters that really are wide---
MAX_GLYPH_W_WIDE = 2.9

#---_WIDE_CHARS: which letters get that wider allowance---
_WIDE_CHARS = set('mwMW')

#---_OFFS: the eight neighbours of a pixel, for following a centre line---
_OFFS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]



#======================================================================
#STEP 1: INK TO CENTRE LINES
#the thinned ink is a picture, these turn it into paths the pen followed
#======================================================================
#----------------------------------------------------------------------
#TracePolylines: follow the thinned ink into lists of points
#walks from each loose end along connected pixels, so a stroke becomes
#an ordered path instead of a set of pixels.  then tidies: whiskers off,
#and paths that meet at a junction and carry on straight are rejoined
#----------------------------------------------------------------------
def TracePolylines(skel):
    pts = set((int(y),int(x)) for y,x in zip(*np.nonzero(skel)))
    if not pts:
        return []

    def nbrs(p):
        y,x = p
        out = []
        for dy,dx in _OFFS:
            q = (y+dy,x+dx)
            if q not in pts:
                continue
            #a diagonal step is redundant when an orthogonal 2-step path exists
            if dy and dx and ((y+dy,x) in pts or (y,x+dx) in pts):
                continue
            out.append(q)
        return out

    deg = {p: len(nbrs(p)) for p in pts}
    usedEdges = set()
    polylines = []

    def walk(start,first):
        chain = [start,first]
        usedEdges.add((start,first))
        usedEdges.add((first,start))
        cur,prev = first,start
        while deg[cur]==2:
            nxt = [n for n in nbrs(cur) if n!=prev]
            if not nxt:
                break
            n = nxt[0]
            if (cur,n) in usedEdges:
                break
            usedEdges.add((cur,n))
            usedEdges.add((n,cur))
            chain.append(n)
            prev,cur = cur,n
        return chain

    seeds = sorted([p for p in pts if deg[p]!=2]) or sorted(pts)[:1]
    for p in seeds:
        for n in nbrs(p):
            if (p,n) not in usedEdges:
                polylines.append(walk(p,n))
    #leftover pure loops
    covered = {q for ch in polylines for q in ch}
    for p in sorted(pts-covered):
        if deg.get(p,0)==2:
            ns = nbrs(p)
            if ns and (p,ns[0]) not in usedEdges:
                polylines.append(walk(p,ns[0]))

    chains = [[(float(x),float(y)) for (y,x) in ch] for ch in polylines
              if len(ch)>=2]
    chains = _PruneSpurs(chains)
    return _JoinAtJunctions(chains)

#---_Dir: which way a path is heading at one of its ends---
def _Dir(pts,fromEnd,n=4):
    p = np.asarray(pts,np.float64)
    if fromEnd:
        a,b = p[max(0,len(p)-1-n)],p[-1]
    else:
        a,b = p[min(len(p)-1,n)],p[0]
    v = b-a
    L = np.hypot(*v)
    return v/L if L>1e-6 else np.array([1.0,0.0])

#----------------------------------------------------------------------
#_PruneSpurs: cut off the little whiskers thinning leaves
#a thick junction thins into a short stub going nowhere.  anything
#shorter than a minimum and attached at one end only is removed
#----------------------------------------------------------------------
def _PruneSpurs(chains,minLen=2.5):
    if len(chains)<=1:
        return chains
    ends = {}
    for c in chains:
        for e in (tuple(c[0]),tuple(c[-1])):
            ends[e] = ends.get(e,0)+1
    out = []
    for c in chains:
        L = PolylineLength(c)
        dangling = ends[tuple(c[0])]==1 or ends[tuple(c[-1])]==1
        if L<minLen and dangling and len(chains)>1:
            continue
        out.append(c)
    return out or chains

#----------------------------------------------------------------------
#_JoinAtJunctions: rejoin paths that are really one stroke
#two paths meeting at a junction and continuing in roughly the same
#direction were one pen stroke before thinning split them
#----------------------------------------------------------------------
def _JoinAtJunctions(chains,maxAngleDeg=75.0):
    chains = [list(c) for c in chains]
    changed = True
    while changed and len(chains)>1:
        changed = False
        best = None
        for i in range(len(chains)):
            for j in range(len(chains)):
                if i==j:
                    continue
                for iEnd in (0,1):
                    for jEnd in (0,1):
                        pi = chains[i][-1] if iEnd else chains[i][0]
                        pj = chains[j][-1] if jEnd else chains[j][0]
                        if abs(pi[0]-pj[0])>1.5 or abs(pi[1]-pj[1])>1.5:
                            continue
                        di = _Dir(chains[i],iEnd)
                        dj = _Dir(chains[j],jEnd)
                        #continuing straight through means di ~ -dj
                        cosang = float(-np.dot(di,dj))
                        ang = np.degrees(np.arccos(np.clip(cosang,-1,1)))   #clip first, rounding can push a unit dot past 1
                        if ang<maxAngleDeg and (best is None or ang<best[0]):
                            best = (ang,i,j,iEnd,jEnd)
        if best is not None:
            _,i,j,iEnd,jEnd = best
            a = chains[i] if iEnd else chains[i][::-1]
            b = chains[j] if not jEnd else chains[j][::-1]
            merged = a+b[1:] if a[-1]==b[0] else a+b
            for k in sorted((i,j),reverse=True):
                del chains[k]
            chains.append(merged)
            changed = True
    return chains

#----------------------------------------------------------------------
#SimplifyPolyline: drop points that add nothing
#keeps a point only if removing it would move the path by more than a
#tolerance, so a straight run of pixels becomes two points
#----------------------------------------------------------------------
def SimplifyPolyline(poly,eps=0.6):
    if len(poly)<3:
        return poly
    p = np.asarray(poly,np.float64)
    fromStart = np.hypot(*(p-p[0]).T)
    if fromStart[-1]<=max(1.5,0.25*fromStart.max()):
        far = int(np.argmax(fromStart))
        if 0<far<len(poly)-1:
            return SimplifyPolyline(poly[:far+1],eps)[:-1]+ \
                   SimplifyPolyline(poly[far:],eps)
    a,b = p[0],p[-1]
    ab = b-a
    L = np.hypot(*ab)+1e-9
    rel = p-a
    d = np.abs(ab[0]*rel[:,1]-ab[1]*rel[:,0])/L
    i = int(np.argmax(d))
    if d[i]>eps:
        return SimplifyPolyline(poly[:i+1],eps)[:-1]+ \
               SimplifyPolyline(poly[i:],eps)
    return [poly[0],poly[-1]]



#======================================================================
#STEP 2: CUTTING ONE LINE INTO LETTERS
#the reader says WHERE each character is, the cut happens between them
#======================================================================
#----------------------------------------------------------------------
#_TrimLigatureTails: drop the joining stroke of the next letter
#in joined writing the cut leaves a tail belonging to the neighbour.
#a low, outward running start or end is that tail, so it is trimmed,
#and where the pen entered and left is remembered instead
#----------------------------------------------------------------------
def _TrimLigatureTails(strokes,connL=True,connR=True,lowY=0.42,minKeep=0.18,
                       climbY=1.6,climbSlope=2.2):
    if not strokes:
        return strokes,(0.0,0.0),(0.0,0.0)
    allPts = [p for s in strokes for p in s]
    entry,exitPt = strokes[0][0],strokes[-1][-1]
    xs = [p[0] for p in allPts]
    span = max(xs)-min(xs)
    if span<=1e-6:
        return strokes,entry,exitPt

    #how many leading points form a low(ish), outward-running carrier
    def runLen(s,connected,maxHeight,maxSlope):
        if not connected:
            return 0
        i = 0
        while i+1<len(s):
            dx = s[i+1][0]-s[i][0]
            dy = s[i+1][1]-s[i][1]
            if s[i][1]>=maxHeight or dx<0 or abs(dy)>maxSlope*max(dx,1e-6):
                break
            i+=1
        return i

    out = []
    for k,s in enumerate(strokes):
        t = list(s)
        if k==0:
            i = runLen(t,connL,lowY,0.7) or runLen(t,connL,climbY,climbSlope)
            if i and (t[-1][0]-t[i][0])>minKeep*span:
                t = t[i:]
        if k==len(strokes)-1 and len(t)>=2:
            #the exit connector leaves to the right, so mirror x to reuse the same test
            rev = [(-x,y) for (x,y) in t[::-1]]
            r = runLen(rev,connR,lowY,0.7) or runLen(rev,connR,climbY,climbSlope)
            if r and (t[len(t)-1-r][0]-t[0][0])>minKeep*span:
                t = t[:len(t)-r]
        if len(t)>=2:
            out.append(t)
    if not out:
        return strokes,entry,exitPt
    return out,out[0][0],out[-1][-1]

#----------------------------------------------------------------------
#ExtractLineGlyphs: one line and its text in, one glyph per letter out
#asks the reader for each character's column span, snaps the boundary to
#the cheapest column nearby (thin ink, low down, which is where an honest
#cut is), thins the whole line once, splits the centre lines at those
#boundaries, trims the joins, and stores each letter upright in x-heights
#----------------------------------------------------------------------
def ExtractLineGlyphs(gray,text,model):
    rawW = gray.shape[1]
    #read the line, then ask the recogniser WHERE it saw each character
    t = TensorFromResized(ResizeLineImageFixed(gray))
    lp = model.forward(t)[:,0,:]
    res = CtcForcedAlign(lp,text)
    if res is None:
        return None, None
    align,conf = res
    T = lp.shape[0]

    ink = BinarizeLine(gray)
    band = CoreBand(ink)
    if band is None:
        return None, None
    top,base = band
    xh = float(base-top)
    slant = EstimateSlantDeg(ink)
    shear = np.tan(np.radians(slant))

    #cut cost per column: thin (one stroke wide) and low (near baseline) --
    #honest boundaries in cursive are the ligature crossings
    colProf = ink.sum(axis=0).astype(np.float64)
    k = max(3,int(round(xh*0.12)))|1
    colSm = np.convolve(colProf,np.ones(k)/k,mode='same')
    rowsIdx = np.arange(ink.shape[0])[:, None]
    hiY = np.where(ink,rowsIdx,ink.shape[0]).min(axis=0).astype(np.float64)
    #height of the highest ink in each column, up from baseline
    topAbove = np.clip((base-hiY)/max(1.0,xh),0.0,3.0)
    thin = colSm/max(1e-6,float(np.percentile(colSm[colSm>0],60))
                       if (colSm>0).any() else 1.0)

    cutCost = thin+1.15*topAbove
    cutCost[colProf==0] = 0.0   #a true gap is always a fine cut
    snapR = max(3,int(round(xh*0.5)))

    #cheapest column by the thin/low cost near x -- a true gap scores 0 so it wins
    def _Snap(x,floor=None):
        a = int(np.clip(round(x)-snapR,0,rawW-1))
        b = int(np.clip(round(x)+snapR+1,1,rawW))
        if b-a<2:
            return float(x)
        win = cutCost[a:b].copy()
        #mild preference for staying near the recognizer's own boundary
        off = np.abs(np.arange(a,b)-x)/max(1.0,snapR)
        return float(a+int(np.argmin(win+0.6*off)))

    #frame index -> canvas x -> original pixel x, via FrameXToPixel's real inverse
    origH = gray.shape[0]
    charXs = []
    for ch,sp in align:
        if sp is None:
            charXs.append((ch, None, None))
            continue
        x0 = FrameXToPixel(sp[0]/T*INPUT_WIDTH,rawW,origH)
        x1 = FrameXToPixel(sp[1]/T*INPUT_WIDTH,rawW,origH)
        charXs.append((ch,x0,x1))
    valid = [(i,c) for i,c in enumerate(charXs) if c[1] is not None]
    centres = [0.5*(c[1]+c[2]) for _,c in valid]

    #one shared, strictly increasing boundary array -- independent snapping
    #could leave two neighbouring boundaries on the same column (an empty cell)
    nV = len(valid)
    minCell = max(1.0,0.08*xh)
    bnd = [0.0]*(nV+1)
    bnd[0] = max(0.0,centres[0]-(0.5*(centres[1]-centres[0])
                                    if nV>1 else xh))
    bnd[nV] = min(float(rawW),centres[-1]+(0.5*(centres[-1]-centres[-2])
                                              if nV>1 else xh))
    for k2 in range(1,nV):
        s = _Snap(0.5*(centres[k2-1]+centres[k2]),floor=bnd[k2-1])
        lo = bnd[k2-1]+minCell
        #leave room for the remaining cells, but never go backwards
        hi = max(lo,bnd[nV]-(nV-k2)*minCell)
        bnd[k2] = min(max(s,lo),hi)
    cuts = []
    for k2 in range(nV):
        i,(ch,x0,x1) = valid[k2]
        cuts.append((i,ch,bnd[k2],bnd[k2+1],x0,x1))

    glyphs = [None]*len(charXs)
    inkArea = float(ink.sum())

    #skeletonize the whole line once, then cut the traced centerlines at the
    #boundaries -- thinning each column slice separately bakes a spurious spine along the cut
    skelAll = Skeletonize(ink)
    polysAll = TracePolylines(skelAll)
    skelLenTotal = sum(
        float(np.hypot(*(np.diff(np.asarray(p),axis=0).T)).sum())
        for p in polysAll) or 1.0

    bounds = [(int(round(c[2])),int(round(c[3]))) for c in cuts]

    #component map, kept only to find the dots the skeleton tracer drops
    compLab,_nComp = LabelComponents(ink)

    segsFor = {i: [] for i in range(len(charXs))}
    crossings = set()
    tracedComps = set()
    for p in polysAll:
        cur,curT = [], None
        for pt in p:
            px,py = int(round(pt[0])),int(round(pt[1]))
            if 0<=py<compLab.shape[0] and 0<=px<compLab.shape[1]:
                tracedComps.add(int(compLab[py,px]))
            t = None
            for k2,(L,R) in enumerate(bounds):
                if L<=pt[0]<R:
                    t = cuts[k2][0]
                    break
            if t!=curT:
                if curT is not None and len(cur)>=2:
                    segsFor[curT].append(cur)
                if curT is not None and t is not None:
                    crossings.add((min(curT,t),max(curT,t)))
                cur = [pt] if t is not None else []
                curT = t
            elif t is not None:
                cur.append(pt)
        if curT is not None and len(cur)>=2:
            segsFor[curT].append(cur)

    #a dot thins to a single skeleton pixel (not a polyline) -- recover any
    #component the tracer produced nothing for as a short dash across its own width
    minDotPx = max(2,int(round(DOT_MIN_AREA*xh*xh)))
    for cid in np.unique(compLab):
        if cid==0 or int(cid) in tracedComps:
            continue
        ysC,xsC = np.nonzero(compLab==cid)
        if ysC.size<minDotPx:
            continue
        xc = float(xsC.mean())
        owner = None
        for k2,(L,R) in enumerate(bounds):
            if L<=xc<R:
                owner = cuts[k2][0]
                break
        if owner is None:
            continue
        yc = float(ysC.mean())
        x0c,x1c = float(xsC.min()),float(xsC.max())
        if x1c-x0c<1.0:
            x0c,x1c = x0c-0.5,x1c+0.5
        segsFor[owner].append([(x0c,yc),(x1c,yc)])

    for (i,ch,left,right,ax0,ax1) in cuts:
        if ch==' ':
            glyphs[i] = dict(char=' ',width=(right-left)/xh)
            continue
        polys = [s for s in segsFor[i] if len(s)>=2]
        if not polys:
            glyphs[i] = None
            continue
        #a centerline crossing a boundary means the pen never lifted there --
        #the honest definition of "connected" for measuring cursiveness
        connL = any(i in pr for pr in crossings if i-1 in pr)
        connR = any(i in pr for pr in crossings if i+1 in pr)

        #normalize strokes: origin at the left cut, baseline, y up, /xh, de-slanted
        norm = []
        for p in polys:
            p = SimplifyPolyline(p,eps=0.6)
            q = []
            for (x,y) in p:
                yn = (base-y)/xh
                xn = (x-left)/xh
                xn = xn-shear*yn
                q.append((round(xn,3),round(yn,3)))
            norm.append(q)
        #sort strokes left-to-right by start, each oriented left-to-right
        norm = [(s if s[0][0]<=s[-1][0] else s[::-1]) for s in norm]
        norm.sort(key=lambda s: min(pt[0] for pt in s))
        #trim the ligature connector on each confirmed-joined side, remember
        #where the pen entered/left so synthesis draws its own connector
        norm,entry,exitPt = _TrimLigatureTails(norm,connL=connL,connR=connR)
        if not norm:
            glyphs[i] = None
            continue
        #re-align in the RE-SLANTED frame the glyph will be drawn in
        xr = [px+shear*py for s in norm for (px,py) in s]
        x0r,x1r = min(xr),max(xr)
        norm = [[(round(px-x0r,3),py) for (px,py) in s] for s in norm]
        ysAll = [pt[1] for s in norm for pt in s]
        glyphs[i] = dict(
            char=ch,strokes=norm,
            width=round(x1r-x0r,3),
            advance=round((right-left)/xh,3),
            #gap from the cell's left edge to where the ink starts -- keeps
            #the author's own letter gaps instead of butting glyphs together
            lead=round(float(np.clip(x0r,0.0,1.5)),3),
            top=round(max(ysAll),3),bot=round(min(ysAll),3),
            entryX=round(entry[0]-x0r,3),entryY=round(entry[1],3),
            exitX=round(exitPt[0]-x0r,3),exitY=round(exitPt[1],3),
            connL=connL,connR=connR)

    stats = dict(slant=slant,xh=xh,alignConf=conf,
                 strokeW=inkArea/max(1.0,skelLenTotal),
                 #the territory each character was actually cut from, kept for debugging
                 cuts=[(int(i),ch,float(left),float(right))
                       for (i,ch,left,right,_ax0,_ax1) in cuts])
    return glyphs,stats



#======================================================================
#STEP 3: IS THIS SHAPE ACTUALLY A LETTER
#five gates, cheapest first.  a gate only ever rejects, it never invents
#======================================================================
#---_ASCEND: letters that should reach above the x-height band---
_ASCEND = set('bdfhklt')|set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

#---_DESCEND: letters that should drop below the baseline---
_DESCEND = set('gjpqy')

#---_XONLY: letters that should stay inside the band---
_XONLY = set('acemnorsuvwxz')

#----------------------------------------------------------------------
#_ClassOk: does the ink reach where this letter should reach
#an 'l' that never goes above the x-height band was mis-cut, so the
#expected height class is a cheap and reliable reject
#----------------------------------------------------------------------
def _ClassOk(ch,top,bot):
    if ch in _XONLY:
        return top<=1.42 and bot>=-0.42
    if ch in _ASCEND:
        return top>=1.25
    if ch in _DESCEND:
        return bot<=-0.20
    return True

#---_PenLength: total pen path of a glyph, a fragment has too little---
def _PenLength(g):
    tot = 0.0
    for s in g['strokes']:
        p = np.asarray(s,np.float64)
        if len(p)>=2:
            tot+=PolylineLength(p)
    return tot

#---_LongestStroke: length of its single longest stroke---
def _LongestStroke(g):
    best = 0.0
    for s in g['strokes']:
        p = np.asarray(s,np.float64)
        if len(p)>=2:
            best = max(best,PolylineLength(p))
    return best

#----------------------------------------------------------------------
#_ShapeGrid: describe a shape as a small grid of numbers
#lays the glyph on a 12 by 12 grid and counts how much ink falls in each
#cell, then blurs it a little so two writings of the same letter do not
#have to line up exactly.  comparing letters is then comparing grids
#----------------------------------------------------------------------
def _ShapeGrid(g,n=12):
    xs,ys = [],[]
    for st in g['strokes']:
        for (x0,y0),(x1,y1) in zip(st,st[1:]):
            d = max(abs(x1-x0),abs(y1-y0))
            k = max(2,int(d*24))
            for t in np.linspace(0.0,1.0,k):
                xs.append(x0+t*(x1-x0))
                ys.append(y0+t*(y1-y0))
        if len(st)==1:
            xs.append(st[0][0])
            ys.append(st[0][1])
    if not xs:
        return None
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    w = max(1e-3,xs.max()-xs.min())
    h = max(1e-3,ys.max()-ys.min())
    gx = np.clip(((xs-xs.min())/w*(n-1)).astype(int),0,n-1)
    gy = np.clip(((ys-ys.min())/h*(n-1)).astype(int),0,n-1)
    grid = np.zeros((n,n),np.float32)
    np.add.at(grid,(gy,gx),1.0)   #scatter add, several stroke points can land in one cell
    grid = np.sqrt(grid)   #damp long dwells
    #3x3 blur: tolerate small shifts between two writings of one letter
    pad = np.pad(grid,1)
    grid = sum(pad[a:a+n,b:b+n]*(0.25 if (a==1 and b==1) else 0.09375)
               for a in range(3) for b in range(3))
    v = grid.ravel()
    nrm = float(np.linalg.norm(v))
    return v/nrm if nrm>1e-9 else None

#---_LETTER_PRIOR: cached reference shape per letter---
_LETTER_PRIOR = {}

#----------------------------------------------------------------------
#BuildLetterPrior: what each letter looks like across all writers
#pools every writer's variants of a letter and takes the middle value in
#each grid cell.  used ONLY to reject a writer's own bad cuts, never to
#put somebody else's letter into their profile
#----------------------------------------------------------------------
def BuildLetterPrior(libsByAuthor,minSamples=12):
    byChar = {}
    for lib in libsByAuthor.values():
        for ch,vs in lib.items():
            byChar.setdefault(ch,[]).extend(vs)
    prior = {}
    for ch,vs in byChar.items():
        grids = [g for g in (_ShapeGrid(v) for v in vs) if g is not None]
        if len(grids)<minSamples:
            continue
        G = np.vstack(grids)
        proto = np.median(G,axis=0)   #robust to fragments in the pool
        nrm = float(np.linalg.norm(proto))
        if nrm>1e-9:
            prior[ch] = proto/nrm
    return prior

#---_PriorScore: how far one glyph sits from that reference shape---
def _PriorScore(g,ch,prior):
    proto = prior.get(ch)
    grid = _ShapeGrid(g)
    if proto is None or grid is None:
        return 0.0
    return float(1.0-np.dot(grid,proto))

#---PriorFilter: drop the variants least like the letter is supposed to look---
def PriorFilter(variants,ch,prior,keepFrac=0.7,minKeep=3):
    if not prior or ch not in prior or len(variants)<=minKeep:
        return variants
    scored = sorted(variants,key=lambda g: _PriorScore(g,ch,prior))
    k = max(minKeep,int(round(keepFrac*len(scored))))
    return scored[:k]

#----------------------------------------------------------------------
#_ConsensusFilter: drop variants that disagree with the rest
#finds the most typical variant this writer produced and removes the
#ones furthest from it, which catches glyphs holding part of a
#neighbouring letter
#----------------------------------------------------------------------
def _ConsensusFilter(variants,keepFrac=0.75):
    if len(variants)<=3:
        return variants
    grids = [_ShapeGrid(g) for g in variants]
    ok = [(g,d) for g,d in zip(variants,grids) if d is not None]
    if len(ok)<=3:
        return [g for g,_ in ok] or variants
    G = np.stack([d for _,d in ok])
    D = np.abs(G[:, None,:]-G[None,:,:]).sum(-1)
    medoid = int(np.argmin(D.sum(1)))
    dist = D[medoid]
    order = np.argsort(dist)
    n = max(3,int(round(len(ok)*keepFrac)))
    return [ok[i][0] for i in order[:n]]



#======================================================================
#STEP 4: TIDYING THE SHAPES THAT SURVIVED
#smoothing, and working out one clean letterform to fall back on
#======================================================================
#---_ResampleN: respace a path to n evenly spread points---
def _ResampleN(poly,n):
    p = np.asarray(poly,np.float64)
    if len(p)<2:
        return np.zeros((n,2))
    seg = np.hypot(*np.diff(p,axis=0).T)
    cum = np.concatenate([[0.0],np.cumsum(seg)])
    if cum[-1]<1e-9:
        return np.repeat(p[:1],n,axis=0)
    t = np.linspace(0.0,cum[-1],n)
    return np.stack([np.interp(t,cum,p[:,0]),np.interp(t,cum,p[:,1])],1)

#---_AlignStroke: line two strokes up so they can be averaged---
def _AlignStroke(s,N):
    s = np.asarray(s,float)
    if len(s)<2:
        return None
    if float(np.hypot(*(s[0]-s[-1])))<0.30 and len(s)>=4:
        i0 = int(np.argmin(s[:,0]))
        s = np.concatenate([s[i0:],s[:i0+1]])
    elif s[0,0]>s[-1,0]:
        s = s[::-1]
    return _ResampleN(s,N)

#----------------------------------------------------------------------
#_DenoisedPrototype: average several writings into one clean shape
#respaces each variant to the same number of points, lines them up and
#takes the middle value at each point.  the wobble cancels out and the
#shape the writer meant is what is left
#----------------------------------------------------------------------
def _DenoisedPrototype(variants,minN=3,N=44):
    byNc = {}
    for g in variants:
        if all(len(s)>=2 for s in g['strokes']):
            byNc.setdefault(len(g['strokes']),[]).append(g)
    if not byNc:
        return None
    nc = max(byNc,key=lambda k: len(byNc[k]))
    group = byNc[nc]
    if len(group)<minN or nc>5:
        return None
    stacks = [[] for _ in range(nc)]
    for g in group:
        ss = sorted(g['strokes'],key=lambda s: min(p[0] for p in s))
        ok = True
        for i,s in enumerate(ss):
            a = _AlignStroke(s,N)
            if a is None:
                ok = False
                break
            stacks[i].append(a)
        if not ok:
            for st in stacks:
                if st:
                    st.pop()
    proto = []
    for st in stacks:
        if len(st)<minN:
            return None
        A = np.stack(st)
        #two rounds of shift-to-mean-then-remean, sharpens a cursive hand's cut-position noise
        med = np.median(A,axis=0)
        for _ in range(2):
            shifted = []
            for c in A:
                best,bestd = c,1e9
                for dx in np.linspace(-0.22,0.22,9):
                    cc = c.copy()
                    cc[:,0]+=dx
                    d = float(np.hypot(*(cc-med).T).mean())
                    if d<bestd:
                        bestd,best = d,cc
                shifted.append(best)
            A = np.stack(shifted)
            med = np.median(A,axis=0)
        dev = np.sqrt(((A-med)**2).sum(-1)).mean(1)
        A = A[dev<=np.percentile(dev,80)]
        proto.append(np.median(A,axis=0))
    x0 = min(float(s[:,0].min()) for s in proto)
    return [[[round(float(x-x0),3),round(float(y),3)] for x,y in s]
            for s in proto]

#---SmoothStroke: take the pixel staircase out of a path---
def SmoothStroke(pts,step=0.055,passes=2):
    p = np.asarray(pts,np.float64)
    if len(p)<3:
        return [[round(float(x),3),round(float(y),3)] for x,y in p]
    seg = np.hypot(*np.diff(p,axis=0).T)
    cum = np.concatenate([[0.0],np.cumsum(seg)])
    total = float(cum[-1])
    if total<1e-6:
        return [[round(float(x),3),round(float(y),3)] for x,y in p]
    n = max(3,int(round(total/step))+1)
    t = np.linspace(0.0,total,n)
    q = np.stack([np.interp(t,cum,p[:,0]),np.interp(t,cum,p[:,1])],1)
    for _ in range(passes):
        inner = 0.25*q[:-2]+0.5*q[1:-1]+0.25*q[2:]
        q = np.vstack([q[:1],inner,q[-1:]])
    return [[round(float(x),3),round(float(y),3)] for x,y in q]



#======================================================================
#STEP 5: MEASURING THE HAND AS A WHOLE
#the numbers synthesis needs to lay letters out like this writer does
#======================================================================
#----------------------------------------------------------------------
#RenderRefStats: how this writer's real line looks to the networks
#ink density, aspect, paper and ink levels, word gaps from real gaps,
#slant and reach.  synthesis has to reproduce these, not just shapes
#----------------------------------------------------------------------
def RenderRefStats(gray):
    a = np.asarray(ResizeLineImageFixed(gray),dtype=np.float32)
    #paper isn't pure white and ink isn't pure black on a real scan --
    #rendering at those extremes is a domain shift the model reacts to
    bright = a[a>=200]
    dark = a[a<128]
    out = dict(inkFrac=float((a<128).mean()),
               aspect=float(a.shape[1])/max(1.0,a.shape[0]),
               slantMeas=np.nan,ascMeas=np.nan,descMeas=np.nan,
               paperLevel=float(np.median(bright)) if bright.size else 255.0,
               inkLevel=float(np.median(dark)) if dark.size else 40.0,
               midFrac=float(((a>=128)&(a<200)).mean()),
               wordGap=np.nan,heightXh=np.nan)
    #word spacing from real ink gaps, not the recognizer's cramped space territory
    ink = BinarizeLine(gray)
    band = CoreBand(ink)
    if band is not None and ink.any():
        top,base = band
        xh = float(base-top)
        if xh>=4:
            col = ink.any(axis=0)
            gaps,run = [],0
            for v in col:
                if v:
                    if run:
                        gaps.append(run)
                    run = 0
                else:
                    run+=1
            wide = [g/xh for g in gaps if g/xh>0.5]
            if wide:
                out['wordGap'] = float(np.median(wide))
            ys = np.nonzero(ink.any(axis=1))[0]
            out['heightXh'] = float(ys.max()-ys.min()+1)/xh
            #measured the same way as on the synthesized render, so they're directly comparable
            out['slantMeas'] = float(EstimateSlantDeg(ink))
            out['ascMeas'] = float(base-ys.min())/xh
            out['descMeas'] = float(base-ys.max())/xh
    return out

#---_MedianWordGap: the typical gap this writer leaves between words---
def _MedianWordGap(refs,spaceAdvs):
    vals = [r['wordGap'] for r in (refs or [])
            if r.get('wordGap')==r.get('wordGap')]   #drop nan
    if vals:
        return float(np.median(vals))
    return float(np.median(spaceAdvs)) if spaceAdvs else 1.2



#======================================================================
#PUTTING IT TOGETHER
#the two entry points: extract a writer's pages, then build their profile
#======================================================================
#----------------------------------------------------------------------
#ExtractAuthorRaw: run the cutting over every line of a writer
#returns the uncut, unfiltered result per line, which is what gets
#cached, so changing a filter does not mean redoing the slow part
#----------------------------------------------------------------------
def ExtractAuthorRaw(authorId,lineItems,model):
    parsed = []
    for gray,text in lineItems:
        glyphs,stats = ExtractLineGlyphs(gray,text,model)
        if glyphs is not None:
            stats = dict(stats)
            stats['ref'] = RenderRefStats(gray)
            parsed.append((glyphs,stats))
    return parsed

#----------------------------------------------------------------------
#BuildAuthorProfile: raw glyphs in, the stored profile out
#runs the five gates in order, keeps the most typical variants per
#letter, works out one clean fallback letterform, and measures the
#hand: slant, x-height, reach, word gap and how joined up it is
#----------------------------------------------------------------------
def BuildAuthorProfile(authorId,parsed,refs=None,prior=None,
                       verbose=True):
    lib = {}
    slants,xhs,strokeWs = [],[],[]
    ascRatios,descRatios = [],[]
    advances = {}
    spaceAdvs = []
    connPairs,adjPairs = 0,0

    if not parsed:
        return None
    confs = [st['alignConf'] for _,st in parsed]
    #only trust char boundaries from lines the recognizer actually read
    confCut = (float(np.percentile(confs,ALIGN_CONF_PCT))
               if len(confs)>=8 else -1e9)

    for glyphs,stats in parsed:
        if stats['alignConf']<confCut:
            continue
        slants.append(stats['slant'])
        xhs.append(stats['xh'])
        strokeWs.append(stats['strokeW']/max(1.0,stats['xh']))
        for k,g in enumerate(glyphs):
            if g is None:
                continue
            ch = g['char']
            if ch==' ':
                spaceAdvs.append(g['width'])
                continue
            if ch.islower():
                if g['top']>1.35:
                    ascRatios.append(g['top'])
                if g['bot']<-0.25:
                    descRatios.append(g['bot'])
            advances.setdefault(ch,[]).append(g['advance'])
            lib.setdefault(ch,[]).append(g)
            #connectedness between adjacent in-word letters
            if k+1<len(glyphs) and glyphs[k+1] is not None and \
                    glyphs[k+1].get('char',' ')!=' ' and 'connR' in g:
                adjPairs+=1
                if g['connR']:
                    connPairs+=1

    if not xhs:
        return None

    #prune each char's variants, keep the most typical MAX_VARIANTS
    lib2 = {}
    for ch,insts in lib.items():
        wMax = MAX_GLYPH_W_WIDE if ch in _WIDE_CHARS else MAX_GLYPH_W
        #this character's own pen path norm, so a one-stroke 'i' and a looping 'o' each get their own floor
        penFloor = max(MIN_PEN_ABS,FRAGMENT_PEN_FRAC*
                       float(np.median([_PenLength(g) for g in insts])))
        good = [g for g in insts
                if 0.05<g['width']<wMax and (g['top']-g['bot'])>0.25
                and len(g['strokes'])<=6
                and _ClassOk(ch,g['top'],g['bot'])
                #ink much wider than the recognizer's cell means the cut swallowed a neighbour
                and g['lead']+g['width']<=1.45*g['advance']+0.25
                #a letter needs a letter's worth of pen path
                and _PenLength(g)>=penFloor
                and _LongestStroke(g)>=MIN_LONGEST_STROKE]
        if not good:
            continue
        #anchor on a low percentile so doubled letters don't inflate the median
        wRef = float(np.percentile([g['width'] for g in good],40))
        hMed = float(np.median([g['top']-g['bot'] for g in good]))
        tight = [g for g in good
                 if 0.6*wRef<=g['width']<=1.5*wRef
                 and 0.55*hMed<=(g['top']-g['bot'])<=1.7*hMed]
        good = tight if len(tight)>=3 else good
        #reject malformed variants against the cross-author prior, then internal consensus
        good = PriorFilter(good,ch,prior)
        for g in good:
            #how far this sits from the cross-author idea of the letter --
            #synthesis uses it to decide whether the author's own shape is trustworthy
            g['priorD'] = round(_PriorScore(g,ch,prior),4)
        good = _ConsensusFilter(good)
        good.sort(key=lambda g: abs(g['width']-wRef)/max(0.2,wRef)+
                  abs((g['top']-g['bot'])-hMed)/max(0.2,hMed))
        good = good[:MAX_VARIANTS_PER_CHAR]
        #de-staircase the kept variants too -- at a low anchor level these ARE the output
        for g in good:
            g['strokes'] = [SmoothStroke(s) for s in g['strokes']]
        lib2[ch] = good

    #idealMedoid: the single real variant most typical of the author's own set
    medoid = {}
    for ch,vs in lib2.items():
        grids = [(_ShapeGrid(g),g) for g in vs]
        grids = [(v,g) for v,g in grids if v is not None]
        if len(grids)<2:
            continue
        G = np.stack([v for v,_ in grids])
        D = np.abs(G[:, None,:]-G[None,:,:]).sum(-1)
        med = grids[int(np.argmin(D.sum(1)))][1]
        medoid[ch] = dict(
            strokes=[SmoothStroke(s) for s in med['strokes']],
            priorD=round(float(med.get('priorD',0.5)),4),
            advance=round(float(med.get('advance',0.9)),3),
            lead=round(float(med.get('lead',0.0)),3),
            width=round(float(med.get('width',0.9)),3),
            top=round(float(med.get('top',1.0)),3),
            bot=round(float(med.get('bot',0.0)),3),
            entryY=round(float(med.get('entryY',0.3)),3),
            exitY=round(float(med.get('exitY',0.3)),3),
            nPool=len(vs))

    #idealGlyphs: the denoised per-stroke median prototype, prepended as an
    #extra variant when it measurably beats the raw variants on priorD
    ideal = {}
    for ch,vs in lib2.items():
        proto = _DenoisedPrototype(vs)
        if proto is None:
            continue
        pd = _PriorScore({'strokes': proto},ch,prior)
        rawPd = float(np.median([g.get('priorD',0.5) for g in vs]))
        pts = [p for s in proto for p in s]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ideal[ch] = dict(
            strokes=[SmoothStroke(s) for s in proto],priorD=round(pd,4),
            advance=round(float(np.median([g['advance'] for g in vs])),3),
            lead=round(float(np.median([g.get('lead',0.0) for g in vs])),3),
            width=round(max(xs)-min(xs),3),
            top=round(max(ys),3),bot=round(min(ys),3),
            entryY=round(float(np.median([g.get('entryY',0.3) for g in vs])),3),
            exitY=round(float(np.median([g.get('exitY',0.3) for g in vs])),3),
            nPool=len(vs))
        if pd<rawPd-0.02 and pd<0.32:
            g0 = dict(vs[0])
            g0.update(strokes=proto,priorD=round(pd,4),denoised=True,
                      width=round(max(xs)-min(xs),3),
                      top=round(max(ys),3),bot=round(min(ys),3))
            lib2[ch] = [g0]+vs

    refs = refs or [st['ref'] for _,st in parsed if 'ref' in st]
    prof = dict(
        authorId=authorId,
        #the author's own ink density / line proportions, for synthesis's stroke-weight calibration
        inkFracRef=round(float(np.median([r['inkFrac'] for r in refs])),5)
        if refs else None,
        aspectRef=round(float(np.median([r['aspect'] for r in refs])),2)
        if refs else None,
        paperLevel=round(float(np.median([r['paperLevel'] for r in refs])),1)
        if refs else 255.0,
        inkLevel=round(float(np.median([r['inkLevel'] for r in refs])),1)
        if refs else 40.0,
        midFracRef=round(float(np.median([r['midFrac'] for r in refs])),4)
        if refs else 0.0,
        slantMeasRef=round(float(np.nanmedian([r['slantMeas'] for r in refs])),3)
        if refs else None,
        ascMeasRef=round(float(np.nanmedian([r['ascMeas'] for r in refs])),3)
        if refs else None,
        descMeasRef=round(float(np.nanmedian([r['descMeas'] for r in refs])),3)
        if refs else None,

        slantDeg=round(float(np.median(slants)),2),
        xHeightPx=round(float(np.median(xhs)),1),
        strokeWidthXh=round(float(np.median(strokeWs)),3),
        ascender=round(float(np.median(ascRatios)) if ascRatios else 1.7,2),
        descender=round(float(np.median(descRatios)) if descRatios else -0.6,2),
        wordSpaceXh=round(_MedianWordGap(refs,spaceAdvs),2),
        connectedness=round(connPairs/max(1,adjPairs),3),
        letterAdvance={ch: round(float(np.median(v)),3)
                       for ch,v in advances.items()},
        nLines=len(xhs),
        glyphs=lib2,
        idealGlyphs=ideal,
        idealMedoid=medoid,
    )
    if verbose:
        nVar = sum(len(v) for v in lib2.values())
        print(f"  {authorId}: {len(xhs)} lines, {len(lib2)} chars, "
              f"{nVar} variants | slant {prof['slantDeg']:+.1f} deg, "
              f"conn {prof['connectedness']:.2f}, "
              f"asc {prof['ascender']:.2f}, desc {prof['descender']:.2f}")
    return prof
