#u23539195 D.R. Braune.  building a writer's style profile
#the ink is traced to centrelines, cut into one glyph per character at the
#reader's own character boundaries, the bad cuts are dropped, and the hand
#the hand itself is described too, every shape in x-heights so it redraws
#at any size
import json
from pathlib import Path

import numpy as np
from PIL import Image

from recognise import (CharAcc,CtcForcedAlign,ReadText,ResizeLineImageFixed,
                       TensorFromResized,FrameXToPixel,INPUT_WIDTH)
from segment import (BinarizeLine,CoreBand,EstimateSlantDeg,LabelComponents,
                     PolylineLength,SegmentLines,Skeletonize)

#how many shapes of one letter are kept
MAX_VARIANTS_PER_CHAR = 12   #up to twelve shapes of each letter are stored

#boundaries are trusted only from lines above this reading-confidence percentile
ALIGN_CONF_PCT = 25   #the weakest quarter of lines by reading confidence are left out

#a letter needs this fraction of the usual pen path, else it is a fragment
FRAGMENT_PEN_FRAC = 0.5    # half the usual pen path for that letter

#and never less than this, in x-heights
MIN_PEN_ABS = 0.30

#its longest single stroke must be at least this
MIN_LONGEST_STROKE = 0.55

#smaller than this area, in squared x-heights, is dust rather than an i dot
DOT_MIN_AREA = 0.004   #four thousandths of a squared x-height

#wider than this means the cut swallowed a neighbour
MAX_GLYPH_W = 2.1   # ordinary letters may run to 2 x-heights wide
MAX_GLYPH_W_WIDE = 2.9   #and m and w to nearly 3
_WIDE_CHARS = set('mwMW')

#the only letters carrying a crossbar over their neighbour
_BAR_CHARS = set('tfTF')

#a crossbar's width range, in x-heights
BAR_MIN_W = 0.25
BAR_MAX_W = 2.0

#how much wider than tall a stroke must be to be a bar
BAR_FLATNESS = 2.5   # a bar runs at least two and a half times wider than tall

#and how far above the baseline it must sit, in x-heights
BAR_MIN_HEIGHT = 0.6   #and sits six tenths of an x-height up

#the eight neighbours of a pixel, for following a centreline
_OFFS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]   #the eight neighbours clockwise from the top left

#----------------------------------------------------------------------
#TracePolylines: follows the thinned ink from each loose end along connected pixels,
#stopping at every junction, then prunes stubs and rejoins the chains that
#leave a junction in roughly the same direction
#----------------------------------------------------------------------
def TracePolylines(skel):
    pts = set((int(y),int(x)) for y,x in zip(*np.nonzero(skel)))    #the skeleton as a set of coordinates for fast lookups
    if not pts:
        return []

    def Nbrs(p):
        y,x = p
        out = []
        for dy,dx in _OFFS:
            q = (y+dy,x+dx)
            if q not in pts:
                continue
            #a diagonal step is redundant when an orthogonal two-step path exists
            if dy and dx and ((y+dy,x) in pts or (y,x+dx) in pts):   #a diagonal is dropped when an L shaped route exists
                continue
            out.append(q)
        return out

    deg = {p: len(Nbrs(p)) for p in pts}    #degree 1 is a loose end 2 is mid stroke and 3 or more a junction
    usedEdges = set()
    polylines = []

    def Walk(start,first):
        chain = [start,first]
        usedEdges.add((start,first))
        usedEdges.add((first,start))
        cur,prev = first,start
        while deg[cur]==2:   # the walk runs on while each pixel has one way forward
            nxt = [n for n in Nbrs(cur) if n!=prev]
            if not nxt:
                break
            n = nxt[0]
            if (cur,n) in usedEdges:   #every step is taken once
                break
            usedEdges.add((cur,n))
            usedEdges.add((n,cur))
            chain.append(n)
            prev,cur = cur,n
        return chain

    seeds = sorted([p for p in pts if deg[p]!=2]) or sorted(pts)[:1]   #walks start at loose ends and junctions
    for p in seeds:
        for n in Nbrs(p):
            if (p,n) not in usedEdges:
                polylines.append(Walk(p,n))
    #whatever closed loops are left over
    covered = {q for ch in polylines for q in ch}    #a closed loop has no end to start from
    for p in sorted(pts-covered):
        if deg.get(p,0)==2:
            ns = Nbrs(p)
            if ns and (p,ns[0]) not in usedEdges:
                polylines.append(Walk(p,ns[0]))

    chains = [[(float(x),float(y)) for (y,x) in ch] for ch in polylines   #rows and columns swapped round into x and y
              if len(ch)>=2]
    chains = _PruneSpurs(chains)
    return _JoinAtJunctions(chains)

#---_Dir: the direction a path is heading at one of its ends---
def _Dir(pts,fromEnd,n=4):
    p = np.asarray(pts,np.float64)
    if fromEnd:
        a,b = p[max(0,len(p)-1-n)],p[-1]   #the last four points give the heading at the end
    else:
        a,b = p[min(len(p)-1,n)],p[0]
    v = b-a
    L = np.hypot(*v)   #divided by its length to make a unit direction
    return v/L if L>1e-6 else np.array([1.0,0.0])

#---_PruneSpurs: removes the short stubs thinning leaves at a thick junction---
def _PruneSpurs(chains,minLen=2.5):
    if len(chains)<=1:
        return chains
    ends = {}
    for c in chains:
        for e in (tuple(c[0]),tuple(c[-1])):
            ends[e] = ends.get(e,0)+1   #how many chains share each endpoint
    out = []
    for c in chains:
        L = PolylineLength(c)
        dangling = ends[tuple(c[0])]==1 or ends[tuple(c[-1])]==1  #an end touching nothing else is free
        if L<minLen and dangling and len(chains)>1:   #short chains with a free end are thinning whiskers
            continue
        out.append(c)
    return out or chains

#----------------------------------------------------------------------
#_JoinAtJunctions: merges two paths that meet at a junction and continue in roughly the same
#direction, which means they were one pen stroke before thinning split them
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
                        if abs(pi[0]-pj[0])>1.5 or abs(pi[1]-pj[1])>1.5:    #the two ends have to sit within 1.5 pixels
                            continue
                        di = _Dir(chains[i],iEnd)
                        dj = _Dir(chains[j],jEnd)
                        cosang = float(-np.dot(di,dj))   #straight through means di is -dj
                        ang = np.degrees(np.arccos(np.clip(cosang,-1,1)))   #that cosine as an angle away from straight
                        if ang<maxAngleDeg and (best is None or ang<best[0]):   #the straightest pair under 75 degrees goes first
                            best = (ang,i,j,iEnd,jEnd)
        if best is not None:
            _,i,j,iEnd,jEnd = best
            a = chains[i] if iEnd else chains[i][::-1]   #both chains turned so they run end to start
            b = chains[j] if not jEnd else chains[j][::-1]
            merged = a+b[1:] if a[-1]==b[0] else a+b
            for k in sorted((i,j),reverse=True):
                del chains[k]
            chains.append(merged)
            changed = True
    return chains

#---SimplifyPolyline: keeps a point only where removing it would move the path by more than eps---
def SimplifyPolyline(poly,eps=0.6):
    if len(poly)<3:
        return poly
    p = np.asarray(poly,np.float64)
    fromStart = np.hypot(*(p-p[0]).T)    #distance from the first point to every other
    if fromStart[-1]<=max(1.5,0.25*fromStart.max()):  #a path ending back near its start is a loop
        far = int(np.argmax(fromStart))  #splitting it at the furthest point gives two open halves
        if 0<far<len(poly)-1:
            return SimplifyPolyline(poly[:far+1],eps)[:-1]+ \
                   SimplifyPolyline(poly[far:],eps)
    a,b = p[0],p[-1]
    ab = b-a
    L = np.hypot(*ab)+1e-9   #length of the line joining the two ends
    rel = p-a
    d = np.abs(ab[0]*rel[:,1]-ab[1]*rel[:,0])/L    #the cross product over that length is the perpendicular distance
    i = int(np.argmax(d))
    if d[i]>eps:  #a point further than eps is kept and both halves recurse
        return SimplifyPolyline(poly[:i+1],eps)[:-1]+ \
               SimplifyPolyline(poly[i:],eps)
    return [poly[0],poly[-1]]   # otherwise the straight line stands in for the whole path

#----------------------------------------------------------------------
#_TrimLigatureTails: trims a low, outward-running carrier stroke off each joined side of a
#glyph and reports where the pen entered and left
#----------------------------------------------------------------------
def _TrimLigatureTails(strokes,connL=True,connR=True,lowY=0.42,minKeep=0.18,
                       climbY=1.6,climbSlope=2.2):
    if not strokes:
        return strokes,(0.0,0.0),(0.0,0.0)
    allPts = [p for s in strokes for p in s]
    entry,exitPt = strokes[0][0],strokes[-1][-1]
    xs = [p[0] for p in allPts]
    span = max(xs)-min(xs)    #the width of the whole glyph
    if span<=1e-6:
        return strokes,entry,exitPt

    #how many leading points form a low, outward-running carrier
    def RunLen(s,connected,maxHeight,maxSlope):
        if not connected:
            return 0
        i = 0
        while i+1<len(s):
            dx = s[i+1][0]-s[i][0]   #step along x
            dy = s[i+1][1]-s[i][1]   # and step up or down
            if s[i][1]>=maxHeight or dx<0 or abs(dy)>maxSlope*max(dx,1e-6):  #a carrier stays low runs rightwards and keeps its slope down
                break
            i+=1
        return i

    out = []
    for k,s in enumerate(strokes):
        t = list(s)
        if k==0:
            i = RunLen(t,connL,lowY,0.7) or RunLen(t,connL,climbY,climbSlope)  #the low test first then the looser climbing one
            if i and (t[-1][0]-t[i][0])>minKeep*span:   #the trim leaves at least 18 percent of the width behind
                t = t[i:]
        if k==len(strokes)-1 and len(t)>=2:
            #the exit connector leaves rightwards, so x is mirrored to reuse the same check
            rev = [(-x,y) for (x,y) in t[::-1]]  #x mirrored so the exit tail reads like an entry tail
            r = RunLen(rev,connR,lowY,0.7) or RunLen(rev,connR,climbY,climbSlope)
            if r and (t[len(t)-1-r][0]-t[0][0])>minKeep*span:
                t = t[:len(t)-r]
        if len(t)>=2:
            out.append(t)
    if not out:
        return strokes,entry,exitPt
    return out,out[0][0],out[-1][-1]   #the trimmed strokes with the new entry and exit points

#----------------------------------------------------------------------
#ExtractLineGlyphs: one line and its text in, one glyph per character out, the reader gives
#each character's column span, the boundary snaps to the cheapest nearby
#column, the whole line is thinned once, and the centrelines are cut at
#those boundaries and stored upright in x-heights
#----------------------------------------------------------------------
def ExtractLineGlyphs(gray,text,model):
    rawW = gray.shape[1]   # width of the crop in its own pixels
    t = TensorFromResized(ResizeLineImageFixed(gray))
    lp = model.Forward(t)[:,0,:]   #log probabilities for 160 columns across 75 classes
    res = CtcForcedAlign(lp,text)   #the known text drives a viterbi path through those scores
    if res is None:
        return None, None
    align,conf = res
    T = lp.shape[0]

    ink = BinarizeLine(gray)
    band = CoreBand(ink)
    if band is None:
        return None, None
    top,base = band
    xh = float(base-top)   #baseline less band top is the x-height every measurement uses
    slant = EstimateSlantDeg(ink)    #the shear that packs the ink into the fewest columns
    shear = np.tan(np.radians(slant))   # that angle as a shear factor

    #the cut cost per column: thin ink, low down
    colProf = ink.sum(axis=0).astype(np.float64)    #ink per column
    k = max(3,int(round(xh*0.12)))|1   # an odd smoothing window about an eighth of an x-height
    colSm = np.convolve(colProf,np.ones(k)/k,mode='same')   #the column profile smoothed by that window
    rowsIdx = np.arange(ink.shape[0])[:, None]
    hiY = np.where(ink,rowsIdx,ink.shape[0]).min(axis=0).astype(np.float64)   #the highest ink row in each column
    topAbove = np.clip((base-hiY)/max(1.0,xh),0.0,3.0)   #highest ink per column, above the baseline
    thin = colSm/max(1e-6,float(np.percentile(colSm[colSm>0],60))  #column ink against a typical column at the 60th percentile
                       if (colSm>0).any() else 1.0)

    cutCost = thin+1.15*topAbove   # thin low columns cost little and bodies and ascenders cost more
    cutCost[colProf==0] = 0.0   #a true gap is always a free cut
    snapR = max(3,int(round(xh*0.5)))    #a boundary may move half an x-height either way

    #the cheapest column within snapR of x, with a mild pull towards x itself
    def _Snap(x,floor=None):
        a = int(np.clip(round(x)-snapR,0,rawW-1))
        b = int(np.clip(round(x)+snapR+1,1,rawW))
        if b-a<2:
            return float(x)
        win = cutCost[a:b].copy()   #the costs inside that window
        off = np.abs(np.arange(a,b)-x)/max(1.0,snapR)  #distance from the starting guess as a fraction of the radius
        return float(a+int(np.argmin(win+0.6*off)))   # the cheapest column with a mild pull back towards the guess

    #frame index to canvas x to original pixel x
    origH = gray.shape[0]
    charXs = []
    for ch,sp in align:
        if sp is None:
            charXs.append((ch, None, None))
            continue
        x0 = FrameXToPixel(sp[0]/T*INPUT_WIDTH,rawW,origH)   #frame index to canvas column to crop pixel
        x1 = FrameXToPixel(sp[1]/T*INPUT_WIDTH,rawW,origH)
        charXs.append((ch,x0,x1))
    valid = [(i,c) for i,c in enumerate(charXs) if c[1] is not None]
    centres = [0.5*(c[1]+c[2]) for _,c in valid]   #the middle of each character span

    #one shared, strictly increasing boundary array, so two neighbouring
    #boundaries cannot land on the same column and leave an empty cell
    nV = len(valid)
    minCell = max(1.0,0.08*xh)   #no cell narrower than 8 percent of an x-height
    bnd = [0.0]*(nV+1)
    bnd[0] = max(0.0,centres[0]-(0.5*(centres[1]-centres[0])    # the left edge sits half a character spacing before the first centre
                                    if nV>1 else xh))
    bnd[nV] = min(float(rawW),centres[-1]+(0.5*(centres[-1]-centres[-2])   #and the right edge the same distance after the last
                                              if nV>1 else xh))
    for k2 in range(1,nV):
        s = _Snap(0.5*(centres[k2-1]+centres[k2]),floor=bnd[k2-1])   #each inner boundary starts midway between two centres
        lo = bnd[k2-1]+minCell   #kept at least one cell width past the boundary before it
        hi = max(lo,bnd[nV]-(nV-k2)*minCell)   #leave room for the remaining cells
        bnd[k2] = min(max(s,lo),hi)   # the snapped position held inside those two limits
    cuts = []
    for k2 in range(nV):
        i,(ch,x0,x1) = valid[k2]
        cuts.append((i,ch,bnd[k2],bnd[k2+1],x0,x1))   # index character left cut right cut and the reader span

    glyphs = [None]*len(charXs)
    inkArea = float(ink.sum())  #total ink pixels in the line

    #the whole line is thinned once, then the traced centrelines are cut at
    #the boundaries: thinning each column slice separately bakes a spurious
    #spine along the cut
    skelAll = Skeletonize(ink)   # the whole line thinned to one pixel wide
    polysAll = TracePolylines(skelAll)   #those centrelines walked into polylines
    skelLenTotal = sum(  #total centreline length across the line
        float(np.hypot(*(np.diff(np.asarray(p),axis=0).T)).sum())
        for p in polysAll) or 1.0

    bounds = [(int(round(c[2])),int(round(c[3]))) for c in cuts]   #the cut positions rounded to whole columns

    #the component map, kept to find the dots the tracer drops
    compLab,_nComp = LabelComponents(ink)   #every separate blob of ink numbered

    segsFor = {i: [] for i in range(len(charXs))}
    crossings = set()
    tracedComps = set()

    #which cut a column belongs to
    def _CellAt(x):
        for k2,(L,R) in enumerate(bounds):
            if L<=x<R:
                return k2
        return None

    #notes the components a path covers, so the dot recovery below does not
    #mistake a traced stroke for an untraced blob
    def _MarkComps(p):
        for pt in p:
            px,py = int(round(pt[0])),int(round(pt[1]))
            if 0<=py<compLab.shape[0] and 0<=px<compLab.shape[1]:
                tracedComps.add(int(compLab[py,px]))

    #the t or f whose crossbar this path is: short, much wider than tall,
    #well above the baseline, and starting in or next to a bar letter's cell
    def _BarOwner(p):
        xs = [q[0] for q in p]
        ys = [q[1] for q in p]
        dx,dy = max(xs)-min(xs), max(ys)-min(ys)   #width and height of this path
        if dx<BAR_MIN_W*xh or dx>BAR_MAX_W*xh:  # a bar runs between a quarter and two x-heights wide
            return None
        if dx<BAR_FLATNESS*max(dy,1e-6):   #and at least two and a half times wider than tall
            return None
        if float(np.median([(base-y)/max(1.0,xh) for y in ys]))<BAR_MIN_HEIGHT:    # its median height above the baseline passes 0.6 x-heights
            return None
        k2 = _CellAt(min(xs))
        if k2 is None:
            return None
        if cuts[k2][1] in _BAR_CHARS:
            return k2 if _CellAt(max(xs))!=k2 else None   #a bar starting in a t cell and reaching past it
        #a bar lying wholly inside the following letter's cell still belongs
        #to the t before it
        prev = k2-1
        if prev>=0 and cuts[prev][1] in _BAR_CHARS:
            return prev
        return None

    for p in polysAll:
        owner = _BarOwner(p)
        if owner is not None:
            #the whole bar goes to the letter it belongs to, and no crossing
            #is recorded: a bar over the next letter is not a pen join
            _MarkComps(p)
            segsFor[cuts[owner][0]].append(list(p))   # the whole bar goes to the letter it belongs to
            continue
        cur,curT = [], None
        for pt in p:
            px,py = int(round(pt[0])),int(round(pt[1]))
            if 0<=py<compLab.shape[0] and 0<=px<compLab.shape[1]:
                tracedComps.add(int(compLab[py,px]))
            t = None
            for k2,(L,R) in enumerate(bounds):
                if L<=pt[0]<R:   #which cell this point of the path falls in
                    t = cuts[k2][0]
                    break
            if t!=curT:
                if curT is not None and len(cur)>=2:
                    segsFor[curT].append(cur)   #the piece inside the previous cell is closed off
                if curT is not None and t is not None:
                    crossings.add((min(curT,t),max(curT,t)))   # a path crossing a boundary records those two letters as joined
                cur = [pt] if t is not None else []
                curT = t
            elif t is not None:
                cur.append(pt)
        if curT is not None and len(cur)>=2:
            segsFor[curT].append(cur)

    #a dot thins to a single pixel rather than a polyline, so any component
    #the tracer produced nothing for is recovered as a short dash
    minDotPx = max(2,int(round(DOT_MIN_AREA*xh*xh)))   #the smallest blob still counted as a dot
    for cid in np.unique(compLab):
        if cid==0 or int(cid) in tracedComps:  # blobs the tracer already covered are left alone
            continue
        ysC,xsC = np.nonzero(compLab==cid)
        if ysC.size<minDotPx:
            continue
        xc = float(xsC.mean())   #the blob centre decides which cell takes it
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
        segsFor[owner].append([(x0c,yc),(x1c,yc)])    #the dot goes in as a short horizontal dash

    for (i,ch,left,right,ax0,ax1) in cuts:
        if ch==' ':
            glyphs[i] = dict(char=' ',width=(right-left)/xh)  #a space carries only its width
            continue
        polys = [s for s in segsFor[i] if len(s)>=2]
        if not polys:
            glyphs[i] = None
            continue
        #a centreline crossing a boundary means the pen never lifted there
        connL = any(i in pr for pr in crossings if i-1 in pr)   # joined on the left when a centreline crossed that boundary
        connR = any(i in pr for pr in crossings if i+1 in pr)

        #origin at the left cut and the baseline, y up, divided by x-height,
        #and de-slanted
        norm = []
        for p in polys:
            p = SimplifyPolyline(p,eps=0.6)   #points moving the path under 0.6 pixels are dropped
            q = []
            for (x,y) in p:
                yn = (base-y)/xh  #height above the baseline in x-heights with y now pointing up
                xn = (x-left)/xh   #distance from the left cut in x-heights
                xn = xn-shear*yn    # the slant taken out so the letter is stored upright
                q.append((round(xn,3),round(yn,3)))
            norm.append(q)
        #each stroke oriented left to right, then sorted by its start
        norm = [(s if s[0][0]<=s[-1][0] else s[::-1]) for s in norm]   # every stroke turned to run left to right
        norm.sort(key=lambda s: min(pt[0] for pt in s))   #and ordered by where it starts
        norm,entry,exitPt = _TrimLigatureTails(norm,connL=connL,connR=connR)
        if not norm:
            glyphs[i] = None
            continue
        #re-aligned in the re-slanted frame the glyph will be drawn in
        xr = [px+shear*py for s in norm for (px,py) in s]   #the slant put back to find the ink edges as drawn
        x0r,x1r = min(xr),max(xr)
        norm = [[(round(px-x0r,3),py) for (px,py) in s] for s in norm]   #shifted so the ink starts at zero
        ysAll = [pt[1] for s in norm for pt in s]
        glyphs[i] = dict(
            char=ch,strokes=norm,
            width=round(x1r-x0r,3),    #ink width in x-heights
            advance=round((right-left)/xh,3),   #how far the pen moves on to the next letter
            #the gap from the cell's left edge to where the ink starts
            lead=round(float(np.clip(x0r,0.0,1.5)),3),
            top=round(max(ysAll),3),bot=round(min(ysAll),3),   #the highest and lowest points reached
            entryX=round(entry[0]-x0r,3),entryY=round(entry[1],3),   #where the pen comes in
            exitX=round(exitPt[0]-x0r,3),exitY=round(exitPt[1],3),   #and where it leaves
            connL=connL,connR=connR)

    stats = dict(slant=slant,xh=xh,alignConf=conf,
                 strokeW=inkArea/max(1.0,skelLenTotal),  #ink area over centreline length is the pen width
                 cuts=[(int(i),ch,float(left),float(right))
                       for (i,ch,left,right,_ax0,_ax1) in cuts])
    return glyphs,stats

#letters that should reach above the x-height band
_ASCEND = set('bdfhklt')|set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

#letters that should drop below the baseline
_DESCEND = set('gjpqy')

#letters that should stay inside the band
_XONLY = set('acemnorsuvwxz')

#---_ClassOk: whether the ink reaches where this letter's height class says it should---
def _ClassOk(ch,top,bot):
    if ch in _XONLY:
        return top<=1.42 and bot>=-0.42   # an x only letter stays inside the band
    if ch in _ASCEND:
        return top>=1.25   #an ascender reaches above it
    if ch in _DESCEND:
        return bot<=-0.20   # and a descender drops below the baseline
    return True

#---_PenLength: the total pen path of a glyph---
def _PenLength(g):
    tot = 0.0
    for s in g['strokes']:
        p = np.asarray(s,np.float64)
        if len(p)>=2:
            tot+=PolylineLength(p)  #every stroke length added up
    return tot

#---_LongestStroke: the length of its single longest stroke---
def _LongestStroke(g):
    best = 0.0
    for s in g['strokes']:
        p = np.asarray(s,np.float64)
        if len(p)>=2:
            best = max(best,PolylineLength(p))
    return best

#----------------------------------------------------------------------
#_ShapeGrid: describes a shape as a normalised 12 by 12 grid of ink counts, blurred a
#little so two writings of one letter need not line up exactly
#----------------------------------------------------------------------
def _ShapeGrid(g,n=12):
    xs,ys = [],[]
    for st in g['strokes']:
        for (x0,y0),(x1,y1) in zip(st,st[1:]):
            d = max(abs(x1-x0),abs(y1-y0))
            k = max(2,int(d*24))   #each segment sampled densely enough to count properly
            for t in np.linspace(0.0,1.0,k):    #points walked evenly along the segment
                xs.append(x0+t*(x1-x0))
                ys.append(y0+t*(y1-y0))
        if len(st)==1:
            xs.append(st[0][0])
            ys.append(st[0][1])
    if not xs:
        return None
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    w = max(1e-3,xs.max()-xs.min())    #the grid stretches to the glyph own bounding box
    h = max(1e-3,ys.max()-ys.min())
    gx = np.clip(((xs-xs.min())/w*(n-1)).astype(int),0,n-1)    #each point placed into one of twelve columns
    gy = np.clip(((ys-ys.min())/h*(n-1)).astype(int),0,n-1)
    grid = np.zeros((n,n),np.float32)
    np.add.at(grid,(gy,gx),1.0)   #several stroke points can land in one cell
    grid = np.sqrt(grid)   #damp long dwells
    pad = np.pad(grid,1)   #3x3 blur
    grid = sum(pad[a:a+n,b:b+n]*(0.25 if (a==1 and b==1) else 0.09375)   #centre weight a quarter and the eight neighbours the rest
               for a in range(3) for b in range(3))
    v = grid.ravel()
    nrm = float(np.linalg.norm(v))   # scaled to unit length so a dot product is a cosine
    return v/nrm if nrm>1e-9 else None

#----------------------------------------------------------------------
#BuildLetterPrior: pools every writer's variants of a letter and takes the median grid cell,
#used only to reject a writer's own bad cuts
#----------------------------------------------------------------------
def BuildLetterPrior(storeByAuthor,minSamples=12):
    byChar = {}
    for store in storeByAuthor.values():
        for ch,vs in store.items():
            byChar.setdefault(ch,[]).extend(vs)
    prior = {}
    for ch,vs in byChar.items():
        grids = [g for g in (_ShapeGrid(v) for v in vs) if g is not None]
        if len(grids)<minSamples:   #a reference needs at least twelve samples of the letter
            continue
        G = np.vstack(grids)
        proto = np.median(G,axis=0)  #the median cell value across every writer
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
    return float(1.0-np.dot(grid,proto))   #one less the cosine gives the distance from that reference

#---PriorFilter: drops the variants least like the letter is supposed to look---
def PriorFilter(variants,ch,prior,keepFrac=0.7,minKeep=3):
    if not prior or ch not in prior or len(variants)<=minKeep:
        return variants
    scored = sorted(variants,key=lambda g: _PriorScore(g,ch,prior))
    k = max(minKeep,int(round(keepFrac*len(scored))))   #the closest 70 percent are kept and at least three
    return scored[:k]

#---_ConsensusFilter: drops the variants furthest from this writer's own most typical one---
def _ConsensusFilter(variants,keepFrac=0.75):
    if len(variants)<=3:
        return variants
    grids = [_ShapeGrid(g) for g in variants]
    ok = [(g,d) for g,d in zip(variants,grids) if d is not None]
    if len(ok)<=3:
        return [g for g,_ in ok] or variants
    G = np.stack([d for _,d in ok])
    D = np.abs(G[:, None,:]-G[None,:,:]).sum(-1)   #every pairwise distance between the shape grids
    medoid = int(np.argmin(D.sum(1)))   #the variant with the smallest total distance to the rest
    dist = D[medoid]
    order = np.argsort(dist)
    n = max(3,int(round(len(ok)*keepFrac)))   #the nearest 75 percent to it are kept
    return [ok[i][0] for i in order[:n]]

#---_ResampleN: respaces a path to n evenly spread points---
def _ResampleN(poly,n):
    p = np.asarray(poly,np.float64)
    if len(p)<2:
        return np.zeros((n,2))
    seg = np.hypot(*np.diff(p,axis=0).T)   #the step length between consecutive points
    cum = np.concatenate([[0.0],np.cumsum(seg)])   #running distance along the path
    if cum[-1]<1e-9:
        return np.repeat(p[:1],n,axis=0)
    t = np.linspace(0.0,cum[-1],n)   #n positions evenly spread by distance
    return np.stack([np.interp(t,cum,p[:,0]),np.interp(t,cum,p[:,1])],1)  #x and y read off at those positions

#---_AlignStroke: orients one stroke consistently and respaces it, so two can be averaged---
def _AlignStroke(s,N):
    s = np.asarray(s,float)
    if len(s)<2:
        return None
    if float(np.hypot(*(s[0]-s[-1])))<0.30 and len(s)>=4:   #a closed loop is rotated to start at its leftmost point
        i0 = int(np.argmin(s[:,0]))
        s = np.concatenate([s[i0:],s[:i0+1]])
    elif s[0,0]>s[-1,0]:  # an open stroke is turned to run left to right
        s = s[::-1]
    return _ResampleN(s,N)

#----------------------------------------------------------------------
#_DenoisedPrototype: averages several writings of a letter into one clean shape by taking the
#median point of the aligned, respaced variants
#----------------------------------------------------------------------
def _DenoisedPrototype(variants,minN=3,N=44):
    byNc = {}
    for g in variants:
        if all(len(s)>=2 for s in g['strokes']):
            byNc.setdefault(len(g['strokes']),[]).append(g)
    if not byNc:
        return None
    nc = max(byNc,key=lambda k: len(byNc[k]))   #the stroke count most of the variants share
    group = byNc[nc]
    if len(group)<minN or nc>5:   #three variants at least and five strokes at most
        return None
    stacks = [[] for _ in range(nc)]
    for g in group:
        ss = sorted(g['strokes'],key=lambda s: min(p[0] for p in s))   #strokes lined up by where each starts
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
        #two rounds of shift to the median, then re-median
        med = np.median(A,axis=0)   #the point by point median of the stack
        for _ in range(2):
            shifted = []
            for c in A:
                best,bestd = c,1e9
                for dx in np.linspace(-0.22,0.22,9):   # each variant slid sideways to sit best against it
                    cc = c.copy()
                    cc[:,0]+=dx
                    d = float(np.hypot(*(cc-med).T).mean())   #mean distance at that shift
                    if d<bestd:
                        bestd,best = d,cc
                shifted.append(best)
            A = np.stack(shifted)
            med = np.median(A,axis=0)
        dev = np.sqrt(((A-med)**2).sum(-1)).mean(1)   #how far each variant sits from the median overall
        A = A[dev<=np.percentile(dev,80)]   #the furthest fifth is dropped
        proto.append(np.median(A,axis=0))  #the median of what is left is the prototype stroke
    x0 = min(float(s[:,0].min()) for s in proto)   # shifted so the prototype starts at zero
    return [[[round(float(x-x0),3),round(float(y),3)] for x,y in s]
            for s in proto]

#----------------------------------------------------------------------
#SmoothStroke: respaces a path evenly and runs a three-point average over it, taking the
#pixel staircase out
#----------------------------------------------------------------------
def SmoothStroke(pts,step=0.055,passes=2):
    p = np.asarray(pts,np.float64)
    if len(p)<3:
        return [[round(float(x),3),round(float(y),3)] for x,y in p]
    seg = np.hypot(*np.diff(p,axis=0).T)
    cum = np.concatenate([[0.0],np.cumsum(seg)])
    total = float(cum[-1])
    if total<1e-6:
        return [[round(float(x),3),round(float(y),3)] for x,y in p]
    n = max(3,int(round(total/step))+1)  #a point every 0.055 of an x-height
    t = np.linspace(0.0,total,n)
    q = np.stack([np.interp(t,cum,p[:,0]),np.interp(t,cum,p[:,1])],1)
    for _ in range(passes):
        inner = 0.25*q[:-2]+0.5*q[1:-1]+0.25*q[2:]   #a quarter half quarter average over each triple
        q = np.vstack([q[:1],inner,q[-1:]])   #the two end points stay where they are
    return [[round(float(x),3),round(float(y),3)] for x,y in q]

#----------------------------------------------------------------------
#RenderRefStats: measures one real line as the networks see it, ink density, aspect, paper
#and ink levels, word gap, slant and reach
#----------------------------------------------------------------------
def RenderRefStats(gray):
    a = np.asarray(ResizeLineImageFixed(gray),dtype=np.float32)   #measured on the canvas the networks see
    bright = a[a>=200]
    dark = a[a<128]
    out = dict(inkFrac=float((a<128).mean()),   #share of the canvas under 128 grey
               aspect=float(a.shape[1])/max(1.0,a.shape[0]),
               slantMeas=np.nan,ascMeas=np.nan,descMeas=np.nan,
               paperLevel=float(np.median(bright)) if bright.size else 255.0,   # median of the bright pixels stands for paper
               inkLevel=float(np.median(dark)) if dark.size else 40.0,   #and the median of the dark ones for ink
               midFrac=float(((a>=128)&(a<200)).mean()),
               wordGap=np.nan,heightXh=np.nan)
    #the word gap comes from real ink gaps, not the reader's space territory
    ink = BinarizeLine(gray)
    band = CoreBand(ink)
    if band is not None and ink.any():
        top,base = band
        xh = float(base-top)
        if xh>=4:
            col = ink.any(axis=0)   #which columns hold any ink at all
            gaps,run = [],0
            for v in col:
                if v:
                    if run:
                        gaps.append(run)
                    run = 0
                else:
                    run+=1
            wide = [g/xh for g in gaps if g/xh>0.5]    # gaps over half an x-height count as word gaps
            if wide:
                out['wordGap'] = float(np.median(wide))
            ys = np.nonzero(ink.any(axis=1))[0]
            out['heightXh'] = float(ys.max()-ys.min()+1)/xh   # the full ink height in x-heights
            #taken the same way as on a synthesised render, so the two compare
            out['slantMeas'] = float(EstimateSlantDeg(ink))
            out['ascMeas'] = float(base-ys.min())/xh  #how far the ink reaches above the baseline
            out['descMeas'] = float(base-ys.max())/xh   # and how far below it
    return out

#---_MedianWordGap: the typical gap this writer leaves between words---
def _MedianWordGap(refs,spaceAdvs):
    vals = [r['wordGap'] for r in (refs or [])
            if r.get('wordGap')==r.get('wordGap')]   #drop nan
    if vals:
        return float(np.median(vals))
    return float(np.median(spaceAdvs)) if spaceAdvs else 1.2    #measured gaps first then space widths then 1.2 x-heights

#----------------------------------------------------------------------
#ExtractAuthorRaw: runs the cutting over every line of a writer, returning the uncut,
#unfiltered result per line
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
#BuildAuthorProfile: raw glyphs in, the stored profile out, runs the filters in order, keeps
#the most typical variants per letter, derives one clean fallback
#letterform, and measures slant, x-height, reach, word gap and joinedness
#----------------------------------------------------------------------
def BuildAuthorProfile(authorId,parsed,refs=None,prior=None,
                       verbose=True):
    store = {}
    slants,xhs,strokeWs = [],[],[]
    ascRatios,descRatios = [],[]
    advances = {}
    spaceAdvs = []
    connPairs,adjPairs = 0,0

    if not parsed:
        return None
    confs = [st['alignConf'] for _,st in parsed]
    confCut = (float(np.percentile(confs,ALIGN_CONF_PCT))   #the 25th percentile of this writer own alignment confidence
               if len(confs)>=8 else -1e9)

    for glyphs,stats in parsed:
        if stats['alignConf']<confCut:   #lines below that cut are set aside
            continue
        slants.append(stats['slant'])
        xhs.append(stats['xh'])
        strokeWs.append(stats['strokeW']/max(1.0,stats['xh']))   #pen width stored as a fraction of the x-height
        for k,g in enumerate(glyphs):
            if g is None:
                continue
            ch = g['char']
            if ch==' ':
                spaceAdvs.append(g['width'])
                continue
            if ch.islower():
                if g['top']>1.35:   #a lower case letter reaching this high is an ascender
                    ascRatios.append(g['top'])
                if g['bot']<-0.25:   #and one dropping this low is a descender
                    descRatios.append(g['bot'])
            advances.setdefault(ch,[]).append(g['advance'])
            store.setdefault(ch,[]).append(g)
            #joinedness, counted over adjacent in-word letters
            if k+1<len(glyphs) and glyphs[k+1] is not None and \
                    glyphs[k+1].get('char',' ')!=' ' and 'connR' in g:   #two letters side by side inside a word
                adjPairs+=1
                if g['connR']:
                    connPairs+=1  #and a pen join counted between them

    if not xhs:
        return None

    storeKept = {}
    for ch,insts in store.items():
        wMax = MAX_GLYPH_W_WIDE if ch in _WIDE_CHARS else MAX_GLYPH_W   #m and w get the wider limit
        #this character's own pen-path norm, so an i and an o each get a floor
        penFloor = max(MIN_PEN_ABS,FRAGMENT_PEN_FRAC*   #half the median pen path of this very letter
                       float(np.median([_PenLength(g) for g in insts])))
        good = [g for g in insts
                if 0.05<g['width']<wMax and (g['top']-g['bot'])>0.25    #sensible width and height
                and len(g['strokes'])<=6   #six strokes at most
                and _ClassOk(ch,g['top'],g['bot'])   # reaching where the letter height class says
                #ink much wider than its cell means the cut swallowed a neighbour
                and g['lead']+g['width']<=1.45*g['advance']+0.25   #ink staying inside its own cell
                and _PenLength(g)>=penFloor  #enough pen path to be a whole letter
                and _LongestStroke(g)>=MIN_LONGEST_STROKE]  # and one stroke of real length
        if not good:
            continue
        #a low percentile, so doubled letters do not inflate the reference width
        wRef = float(np.percentile([g['width'] for g in good],40))   # the 40th percentile width is the reference
        hMed = float(np.median([g['top']-g['bot'] for g in good]))
        tight = [g for g in good   #variants near that width and the median height
                 if 0.6*wRef<=g['width']<=1.5*wRef
                 and 0.55*hMed<=(g['top']-g['bot'])<=1.7*hMed]
        good = tight if len(tight)>=3 else good
        good = PriorFilter(good,ch,prior)   #the cross writer shape check
        for g in good:
            g['priorD'] = round(_PriorScore(g,ch,prior),4)   # each variant keeps its own distance from the reference
        good = _ConsensusFilter(good)   #then the within writer consistency check
        good.sort(key=lambda g: abs(g['width']-wRef)/max(0.2,wRef)+   #ranked by how close they sit to the typical size
                  abs((g['top']-g['bot'])-hMed)/max(0.2,hMed))
        good = good[:MAX_VARIANTS_PER_CHAR]   # the best twelve are stored
        for g in good:
            g['strokes'] = [SmoothStroke(s) for s in g['strokes']]    #the pixel staircase smoothed out of each stroke
        storeKept[ch] = good

    #the single real variant most typical of this writer's own set
    medoid = {}
    for ch,vs in storeKept.items():
        grids = [(_ShapeGrid(g),g) for g in vs]
        grids = [(v,g) for v,g in grids if v is not None]
        if len(grids)<2:
            continue
        G = np.stack([v for v,_ in grids])
        D = np.abs(G[:, None,:]-G[None,:,:]).sum(-1)  #all pairwise distances inside this letter pool
        med = grids[int(np.argmin(D.sum(1)))][1]    #the most typical real variant is the medoid
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

    #the denoised per-stroke median prototype, prepended as an extra
    #variant when it scores better than the raw ones
    ideal = {}
    for ch,vs in storeKept.items():
        proto = _DenoisedPrototype(vs)
        if proto is None:
            continue
        pd = _PriorScore({'strokes': proto},ch,prior)   #how letter like the prototype turned out
        rawPd = float(np.median([g.get('priorD',0.5) for g in vs]))   #against the median of the raw variants
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
        if pd<rawPd-0.02 and pd<0.32:   # a clearly better prototype goes to the front of the pool
            g0 = dict(vs[0])
            g0.update(strokes=proto,priorD=round(pd,4),denoised=True,
                      width=round(max(xs)-min(xs),3),
                      top=round(max(ys),3),bot=round(min(ys),3))
            storeKept[ch] = [g0]+vs

    refs = refs or [st['ref'] for _,st in parsed if 'ref' in st]
    prof = dict(
        authorId=authorId,
        inkFracRef=round(float(np.median([r['inkFrac'] for r in refs])),5)   #median appearance across the real lines
        if refs else None,
        aspectRef=round(float(np.median([r['aspect'] for r in refs])),2)
        if refs else None,
        paperLevel=round(float(np.median([r['paperLevel'] for r in refs])),1)
        if refs else 255.0,
        inkLevel=round(float(np.median([r['inkLevel'] for r in refs])),1)
        if refs else 40.0,
        midFracRef=round(float(np.median([r['midFrac'] for r in refs])),4)
        if refs else 0.0,
        slantMeasRef=round(float(np.nanmedian([r['slantMeas'] for r in refs])),3)   # nanmedian so lines with no band drop out
        if refs else None,
        ascMeasRef=round(float(np.nanmedian([r['ascMeas'] for r in refs])),3)
        if refs else None,
        descMeasRef=round(float(np.nanmedian([r['descMeas'] for r in refs])),3)
        if refs else None,

        slantDeg=round(float(np.median(slants)),2),   #the writer slant in degrees
        xHeightPx=round(float(np.median(xhs)),1),   # and the x-height in source pixels
        strokeWidthXh=round(float(np.median(strokeWs)),3),
        ascender=round(float(np.median(ascRatios)) if ascRatios else 1.7,2),   #how high ascenders reach with 1.7 as the fallback
        descender=round(float(np.median(descRatios)) if descRatios else -0.6,2),   #and how low descenders drop
        wordSpaceXh=round(_MedianWordGap(refs,spaceAdvs),2),
        connectedness=round(connPairs/max(1,adjPairs),3),   #joined pairs over adjacent pairs
        letterAdvance={ch: round(float(np.median(v)),3)   #the median advance of each character
                       for ch,v in advances.items()},
        nLines=len(xhs),
        glyphs=storeKept,
        idealGlyphs=ideal,
        idealMedoid=medoid,
    )
    if verbose:
        nVar = sum(len(v) for v in storeKept.values())
        print(f"  {authorId}: {len(xhs)} lines, {len(storeKept)} chars, "
              f"{nVar} variants | slant {prof['slantDeg']:+.1f} deg, "
              f"conn {prof['connectedness']:.2f}, "
              f"asc {prof['ascender']:.2f}, desc {prof['descender']:.2f}")
    return prof


#----------------------------------------------------------------------
#PairCropsToTranscript: which written line each transcript line is
#
#the obvious pairing is crop i with transcript line i, which is what the
#offline build used.  it is wrong on a real photograph, and wrong SILENTLY,
#which is worse.  measured on one page of six: the segmenter returned 14
#crops for 9 transcript lines, and the third crop was a one-character
#fragment.  from that crop on, every pairing was off by one, so the letter
#cutter was told that "that God will not love you any less" was the picture
#of "C.S. Lewis. It is, of course, quite true".  the cuts it then makes are
#garbage, and nothing downstream can tell: a glyph library does not know
#what it was supposed to look like.
#
#so the crops are READ, once each, and matched against the transcript with
#a monotonic alignment: writing runs down a page in order, so the pairing
#must be increasing in both, but either side may skip -- a junk crop pairs
#with nothing, and a line the segmenter missed leaves its transcript line
#unmatched.  a pair below the floor is not made at all, because a wrong
#pairing is more damaging than a missing one.
#
#the reader is poor on some of these lines, which is the point: it does not
#have to read a line correctly to show that this crop is that line rather
#than the one below it.
#
#the floor is 0.15 because it was measured rather than picked.  swept over
#seven pages from three writers at 0.00, 0.15, 0.25, 0.35 and 0.45:
#
#  Dylan01 24/24  Dylan02 25/25  Thiya01 16/16  Thiya02 11/11
#      all four unchanged at every floor, mean pair accuracy 97 to 99 per
#      cent -- on a page the reader can read, the alignment is exact and
#      the floor does nothing at all
#  Aston01  9/9 at 0.25 falling to 7/9 at 0.35
#  Aston05 17/17 at 0.00, 16 at 0.15, 15 at 0.25, 11 at 0.35, 8 at 0.45
#
#and the pairs a 0.35 floor removed were CORRECT ones: crop 1 of Aston05
#read "Shn C. Wiltweck" against "John C. MAXWELL" and scored 0.33.  that is
#the reader being poor on an unfamiliar hand, not the pairing being wrong,
#and throwing the line away costs a line of letters for nothing.  0.15
#keeps those and still drops a pair carrying no reading evidence at all.
#
#it is worth a third more letterforms.  the same six pages, built three
#ways:
#
#  crop i with transcript line i      35 characters, 112 variants, 43 lines
#  aligned, floor 0.35                40 characters, 138 variants, 32 lines
#  aligned, floor 0.15                43 characters, 145 variants, 44 lines
#
#the middle row is the one to read: it got MORE letterforms out of ELEVEN
#FEWER lines than the naive pairing did, which is what a correct alignment
#should do -- the lines the naive pairing added were mislabelled, and the
#extraction filters were throwing most of their cuts away anyway
#----------------------------------------------------------------------
PAIR_FLOOR = 0.15   #a pairing needs at least 15 percent reading agreement


def PairCropsToTranscript(crops, truth, reader):
    read = [ReadText(Image.fromarray(c).convert("L"), reader) for c in crops]   #every crop read once by the reader
    n, m = len(crops), len(truth)
    S = [[CharAcc(read[i], truth[j]) for j in range(m)] for i in range(n)]   # character accuracy of every crop against every transcript line
    #dp[i][j]: the best total over the first i crops and first j transcript
    #lines.  three moves: drop this crop, leave this transcript line
    #unmatched, or pair the two
    dp = [[0.0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, m+1):
            best = max(dp[i-1][j], dp[i][j-1])   #dropping the crop or leaving the transcript line unmatched
            s = S[i-1][j-1]
            if s >= PAIR_FLOOR and dp[i-1][j-1]+s > best:   #pairing the two when it scores better and clears the floor
                best = dp[i-1][j-1]+s
            dp[i][j] = best
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        s = S[i-1][j-1]
        if s >= PAIR_FLOOR and abs(dp[i][j]-(dp[i-1][j-1]+s)) < 1e-12:   # this cell came from a pair so the pair is recorded
            pairs.append((i-1, j-1, s))
            i, j = i-1, j-1
        elif dp[i][j] == dp[i-1][j]:   #or from dropping the crop
            i -= 1
        else:
            j -= 1   #otherwise from leaving the transcript line unmatched
    pairs.reverse()   #backtracking builds them last first
    return pairs, read


def BuildProfileFromPages(name, pages, outPath, reader,
                          priorLibs=None, inNamer=False):
    """pages is a list of (grayArray, [transcript lines]).  Writes a profile.

    reader is the text model the pairing and the letter cutting both use.
    priorLibs is {author: profile} for the cross-writer letter prior -- the
    stored profiles as they come off disk, whole, so a writer's own bad cuts
    can be rejected.  the glyph dict is taken out of each here, so handing
    over glyph dicts directly yields an EMPTY prior and no filtering.  inNamer is reported, not used: a new writer can be written
    but not judged until the namer is retrained.
    """
    items = []
    perPage = []
    for i, (gray, truth) in enumerate(pages):
        lines, _dbg, _info = SegmentLines(gray)  #the page cut into line crops
        crops = [np.asarray(ln["image"]) for ln in lines]
        truth = [t for t in truth if t.strip() and t.strip() != "MESS"]   #blank lines and ones marked MESS are left out
        pairs, _read = PairCropsToTranscript(crops, truth, reader)
        mean = (sum(s for _c, _t, s in pairs)/len(pairs)) if pairs else 0.0   #mean reading agreement across the pairs of this page
        perPage.append({"page": i+1, "lines_found": len(crops),
                        "transcript_lines": len(truth), "paired": len(pairs),
                        "mean_pair_accuracy_pct": round(100*mean, 1)})
        print("[add] page %d: %d crops, %d transcript lines, %d paired "
              "(mean %.1f%%)"
              % (i+1, len(crops), len(truth), len(pairs), 100*mean), flush=True)
        if len(pairs) < len(truth):
            print("[add]   %d transcript line(s) matched no crop on this page"
                  % (len(truth)-len(pairs)), flush=True)
        for ci, ti, _s in pairs:
            items.append((crops[ci], truth[ti]))
    if not items:
        raise ValueError("no written line could be matched to a transcript line. "
                         "Check that each label file holds the lines of ITS page, "
                         "one per written line and in order, and that the "
                         "photograph is of the whole sheet")

    print("[add] cutting letters out of %d lines" % len(items), flush=True)
    parsed = ExtractAuthorRaw(name, items, reader)   # every paired line cut into glyphs
    if not parsed:
        raise ValueError("no line survived extraction; the transcripts may not "
                         "match the pages")
    print("[add] %d/%d lines usable" % (len(parsed), len(items)), flush=True)

    #the pooled letter prior: the stored libraries, plus this writer's raw cuts
    rawLibs = {}
    for a, prof in (priorLibs or {}).items():
        rawLibs[a] = {ch: vs for ch, vs in (prof.get("glyphs") or {}).items()}   #the stored writers supply their kept variants
    mine = {}
    for glyphs, _st in parsed:
        for g in glyphs:
            if g and g.get("char", " ") != " " and "strokes" in g:
                mine.setdefault(g["char"], []).append(g)
    rawLibs[name] = mine   #and this writer supplies raw uncut ones
    prior = BuildLetterPrior(rawLibs)   # the pooled median shape of each letter
    print("[add] letter prior over %d characters from %d writers"
          % (len(prior), len(rawLibs)), flush=True)

    refs = [st["ref"] for _g, st in parsed if "ref" in st]
    prof = BuildAuthorProfile(name, parsed, refs=refs, prior=prior)   #filtering measuring and packing into the profile
    if prof is None:
        raise ValueError("not enough usable lines to measure a style from")
    Path(outPath).write_text(json.dumps(prof), encoding="utf-8")   # the whole profile saved as one json file
    nVar = sum(len(v) for v in (prof.get("glyphs") or {}).values())
    print("[add] saved %s: %d characters, %d variants, %d lines"
          % (Path(outPath).name, len(prof.get("glyphs") or {}), nVar,
             prof.get("nLines")), flush=True)
    return {"author": name, "file": Path(outPath).name,
            "characters": len(prof.get("glyphs") or {}),
            "variants": nVar, "lines_used": prof.get("nLines"),
            "slant_deg": prof.get("slantDeg"),
            "connectedness": prof.get("connectedness"),
            "x_height_px": prof.get("xHeightPx"),
            "in_namer_classes": bool(inNamer),
            "pages": perPage}
