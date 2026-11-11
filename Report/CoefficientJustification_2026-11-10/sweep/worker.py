#u23539195 coefficient justification: runs ONE pipeline stage inside ONE patched copy
#of the submitted code and writes the arrays and numbers a sweep panel needs
#it is always started with the patched copy's folder as the working directory, so
#"import segment" is the patched segment, never the submitted one
#   python worker.py <task> <paramsJson> <outPrefix>
import json
import sys
import time
from pathlib import Path

import numpy as np

import paths
import segment as SEG
import recognise as REC
import train as TR


#---Pack: a list of polylines as one point array plus the length of each---
def Pack(polys):
    if not polys:
        return np.zeros((0,2),np.float32),np.zeros(0,np.int32)
    pts = np.concatenate([np.asarray(p,np.float32).reshape(-1,2) for p in polys],0)
    lens = np.asarray([len(p) for p in polys],np.int32)
    return pts,lens


#---LabelLines: a page's typed transcript, one written line per line---
def LabelLines(pagePath):
    lp = Path(str(pagePath).replace("Pages","Labels").replace(".jpg","_labels.txt"))
    if not lp.exists():
        return []
    return [l.strip() for l in lp.read_text(encoding="utf-8",errors="replace").splitlines()
            if l.strip()]


#---PageUpright: reads a page off disk and stands it on the correct axis---
def PageUpright(pagePath):
    page = TR.ReadImageFile(str(pagePath))
    up,rot,s0,s90 = SEG.UprightPage(page)
    return up,dict(rotation=rot,axisScore0=round(float(s0),4),
                   axisScore90=round(float(s90),4))


#---Reader: the text reader, from whichever checkpoint the caller names---
def Reader(name=None):
    return REC.PaperCRNNNumpy(paths.Weights/name if name else None)


#---Namer: the writer classifier, from whichever checkpoint the caller names---
def Namer(name=None):
    return REC.AuthorClassifierCNNNumpy(paths.Weights/name if name else None)


#======================================================================
#the tasks.  each returns (metrics dict, arrays dict)
#======================================================================

#---TaskPageSegment: a whole page to line crops, with the seams and ownership kept---
def TaskPageSegment(p):
    up,info = PageUpright(paths.Data/p["page"])
    t0 = time.time()
    lines,diag,stats = SEG.SegmentLines(up,mergeValley=p.get("mergeValley"))
    secs = time.time()-t0
    truth = LabelLines(paths.Data/p["page"])
    arrays = {}
    cuts = np.asarray(diag["cuts"],np.int32)
    arrays["cuts"] = cuts
    arrays["paper"] = diag["paper"].astype(np.uint8)
    half = SEG.Halve(up if up.ndim==2 else up[...,:3].mean(2).astype(np.uint8))
    arrays["half"] = np.asarray(half,np.uint8)
    for i in p.get("lines",[0]):
        if i < len(lines):
            arrays["crop%d" % i] = lines[i]["image"]
    #how well the crops read, which is what a line-finding coefficient costs or buys
    acc = None
    if p.get("score"):
        rd = Reader(p.get("textWeights"))
        n = min(len(lines),len(truth),int(p.get("scoreLines",8)))
        accs = [REC.CharAcc(REC.ReadText(lines[i]["image"],rd),truth[i]) for i in range(n)]
        acc = float(np.mean(accs)) if accs else None
    m = dict(stats)
    m.update(info)
    m.update(nLinesTruth=len(truth),seconds=round(secs,2),
             charAcc=(round(acc,4) if acc is not None else None),
             inkPct=[round(float(l.get("inkPct",float("nan"))),3) for l in lines][:30])
    return m,arrays


#---TaskLineInk: one line crop to the binary ink mask BinarizeLine produces---
def TaskLineInk(p):
    up,_ = PageUpright(paths.Data/p["page"])
    lines,_,_ = SEG.SegmentLines(up)
    crop = lines[p.get("lineIdx",0)]["image"]
    ink = SEG.BinarizeLine(crop,stripRules=p.get("stripRules",True),
                           closeGaps=p.get("closeGaps",True))
    band = SEG.CoreBand(ink)
    lab,n = SEG.LabelComponents(ink)
    m = dict(inkPixels=int(ink.sum()),components=int(n),
             coreBand=(list(map(int,band)) if band else None),
             xh=(float(band[1]-band[0]) if band else None),
             slantDeg=round(float(SEG.EstimateSlantDeg(ink)),3))
    return m,dict(crop=crop,ink=ink.astype(np.uint8))


#---TaskStrokeNorm: one line redrawn at a fixed pen width, and who the namer says wrote it---
def TaskStrokeNorm(p):
    up,_ = PageUpright(paths.Data/p["page"])
    lines,_,_ = SEG.SegmentLines(up)
    idx = p.get("lineIdx",0)
    crop = lines[idx]["image"]
    norm = REC.StrokeNormalize(np.asarray(crop,np.uint8))
    nm = Namer(p.get("authorWeights"))
    names = sorted(nm.authorMapping,key=lambda k: nm.authorMapping[k])
    #scored on the normalised crop, which is what ClassifyImage feeds the namer
    predIdx,probs = REC.ClassifyImage(crop,nm)
    want = p.get("author")
    m = dict(pred=names[predIdx],predProb=round(float(probs[predIdx]),4),
             authors=names,
             trueProb=(round(float(probs[nm.authorMapping[want]]),4)
                       if want in nm.authorMapping else None),
             correct=(names[predIdx]==want if want else None),
             inkFrac=(round(float((norm<128).mean()),5) if norm is not None else None))
    arrays = dict(crop=crop,probs=probs.astype(np.float32))
    if norm is not None:
        arrays["norm"] = norm
    return m,arrays


#---TaskTrace: the thinned centrelines of one line, as polylines---
def TaskTrace(p):
    import style as STY
    up,_ = PageUpright(paths.Data/p["page"])
    lines,_,_ = SEG.SegmentLines(up)
    crop = lines[p.get("lineIdx",0)]["image"]
    ink = SEG.BinarizeLine(crop)
    sk = SEG.Skeletonize(ink)
    polys = STY.TracePolylines(sk)
    pts,lens = Pack(polys)
    tot = float(sum(SEG.PolylineLength(q) for q in polys))
    m = dict(nPolylines=len(polys),totalLength=round(tot,1),
             skeletonPixels=int(sk.sum()),inkPixels=int(ink.sum()),
             meanPoints=round(float(np.mean(lens)) if len(lens) else 0.0,1))
    return m,dict(crop=crop,ink=ink.astype(np.uint8),sk=sk.astype(np.uint8),
                  pts=pts,lens=lens)


#---TaskGlyphCut: one line cut into one glyph per character at the reader's boundaries---
def TaskGlyphCut(p):
    import style as STY
    up,_ = PageUpright(paths.Data/p["page"])
    lines,_,_ = SEG.SegmentLines(up)
    truth = LabelLines(paths.Data/p["page"])
    idx = p.get("lineIdx",0)
    crop = lines[idx]["image"]
    text = truth[idx] if idx < len(truth) else ""
    rd = Reader(p.get("textWeights"))
    glyphs,stats = STY.ExtractLineGlyphs(crop,text,rd)
    if glyphs is None:
        return dict(failed=True,text=text),dict(crop=crop)
    kept = [g for g in glyphs if g is not None and "strokes" in g]
    polys,chars,widths = [],[],[]
    for g in kept:
        for s in g["strokes"]:
            polys.append(s)
        chars.append(g["char"])
        widths.append(g["width"])
    pts,lens = Pack(polys)
    perGlyph = np.asarray([len(g["strokes"]) for g in kept],np.int32)
    m = dict(text=text,nChars=len(text),nGlyphs=len(kept),
             nNone=int(sum(1 for g in glyphs if g is None)),
             alignConf=round(float(stats["alignConf"]),3),
             xh=round(float(stats["xh"]),2),
             slant=round(float(stats["slant"]),2),
             strokeW=round(float(stats["strokeW"]),3),
             chars="".join(chars),
             meanWidth=round(float(np.mean(widths)) if widths else 0.0,3),
             cuts=[[int(c[0]),c[1],round(float(c[2]),1),round(float(c[3]),1)]
                   for c in stats["cuts"]])
    return m,dict(crop=crop,pts=pts,lens=lens,perGlyph=perGlyph)


#---TaskProfile: a whole writer's profile built from their pages, then described---
def TaskProfile(p):
    import style as STY
    rd = Reader(p.get("textWeights"))
    items = []
    maxLines = int(p.get("maxLines",40))
    for img,lab in paths.PersonalWriterPages(p["author"]):
        if not lab.exists():
            continue
        up,_ = PageUpright(img)
        lines,_,_ = SEG.SegmentLines(up)
        truth = [l.strip() for l in lab.read_text(encoding="utf-8",errors="replace").splitlines()
                 if l.strip()]
        for crop,text in zip([l["image"] for l in lines],truth):
            items.append((crop,text))
        if len(items) >= maxLines:
            break
    items = items[:maxLines]
    parsed = STY.ExtractAuthorRaw(p["author"],items,rd)
    #BuildAuthorProfile takes prior=None by default, but _PriorScore calls
    #prior.get(ch) unconditionally, so the default path raises.  the deployed
    #pipeline always passes a prior pooled over every writer; this study has one
    #writer in hand, so the prior is pooled over that writer's own raw glyphs and
    #the difference is recorded in the document
    rawStore = {}
    for glyphs,_st in parsed:
        for g in glyphs:
            if g and "strokes" in g:
                rawStore.setdefault(g["char"],[]).append(g)
    prior = STY.BuildLetterPrior({p["author"]: rawStore},
                                 minSamples=int(p.get("priorMinSamples",4)))
    prof = STY.BuildAuthorProfile(p["author"],parsed,prior=prior,verbose=False)
    if prof is None:
        return dict(failed=True,nItems=len(items)),{}
    nVar = sum(len(v) for v in prof["glyphs"].values())
    m = dict(nLinesUsed=len(items),nLines=prof["nLines"],
             nChars=len(prof["glyphs"]),nVariants=nVar,
             connectedness=prof["connectedness"],slantDeg=prof["slantDeg"],
             strokeWidthXh=prof["strokeWidthXh"],ascender=prof["ascender"],
             descender=prof["descender"],wordSpaceXh=prof["wordSpaceXh"],
             xHeightPx=prof["xHeightPx"],
             medianPriorD=round(float(np.median(
                 [g.get("priorD",0.5) for v in prof["glyphs"].values() for g in v])),4)
             if nVar else None)
    outProf = p.get("saveProfile")
    if outProf:
        Path(outProf).write_text(json.dumps(prof),encoding="utf-8")
    #the writer's own stored letterforms, so a profile coefficient can be SEEN:
    #the most typical variant of each sample character, laid out side by side
    sample = p.get("sampleChars","aeghnorst")
    polys,per,shown = [],[],[]
    for ch in sample:
        vs = prof["glyphs"].get(ch)
        if not vs:
            continue
        g = min(vs,key=lambda q: q.get("priorD",0.5))
        polys += [s for s in g["strokes"] if len(s) >= 2]
        per.append(len([s for s in g["strokes"] if len(s) >= 2]))
        shown.append(ch)
    pts,lens = Pack(polys)
    m["shown"] = "".join(shown)
    return m,dict(pts=pts,lens=lens,perGlyph=np.asarray(per,np.int32))


#---TaskSynth: a profile and a sentence to a rendered line, scored by both judges---
def TaskSynth(p):
    import synthesise as SY
    prof = json.loads(Path(p["profile"]).read_text(encoding="utf-8"))
    text = p["text"]
    doScore = p.get("score",True)
    rd = Reader(p.get("textWeights")) if doScore else None
    nm = Namer(p.get("authorWeights")) if doScore else None
    t0 = time.time()
    if p.get("bestOf"):
        traj = SY.SynthesizeJointBestOf(
            p.get("author") or prof.get("authorId"),text,prof,
            nTries=int(p.get("nTries",6)),mmPerXh=float(p.get("mmPerXh",4.0)),
            lineWidthMm=float(p.get("lineWidthMm",180.0)),
            jitter=float(p.get("jitter",0.5)),seed=int(p.get("seed",0)),
            reader=rd,authorModel=nm,
            legibility=p.get("legibility"))
    else:
        traj = SY.SynthesizeText(text,prof,mmPerXh=float(p.get("mmPerXh",4.0)),
                                 seed=int(p.get("seed",0)),
                                 lineWidthMm=float(p.get("lineWidthMm",180.0)),
                                 jitter=float(p.get("jitter",0.15)),
                                 legibility=p.get("legibility"))
    secs = time.time()-t0
    img = SY.RenderTrajectory(traj,pxPerMm=float(p.get("pxPerMm",18.0)))
    down,up_ = traj.PenTravelMm()
    #one seed is one sample of a random process, and the differences between
    #synthesis settings are a character or two, so the sweep is repeated over
    #several seeds and the mean and spread reported.  the panel shows the first
    #seed, so the picture and the headline number come from the same place
    extraSeeds = [int(s) for s in (p.get("seeds") or [])]
    seedStats = None
    if extraSeeds and doScore:
        visible = text.replace("\n"," ")
        wantedAuthor = p.get("author") or prof.get("authorId")
        accs,wids,joints = [],[],[]
        for sd in extraSeeds:
            tj = SY.SynthesizeText(text,prof,mmPerXh=float(p.get("mmPerXh",4.0)),
                                   seed=sd,
                                   lineWidthMm=float(p.get("lineWidthMm",180.0)),
                                   jitter=float(p.get("jitter",0.15)),
                                   legibility=p.get("legibility"))
            im = SY.RenderTrajectory(tj,pxPerMm=float(p.get("pxPerMm",18.0)))
            a = REC.CharAcc(REC.ReadText(im,rd),visible)
            _pi,pr = REC.ClassifyImage(im,nm)
            w = (float(pr[nm.authorMapping[wantedAuthor]])
                 if wantedAuthor in nm.authorMapping else float("nan"))
            accs.append(a)
            wids.append(w)
            joints.append(2*a*w/(a+w) if (a+w) > 1e-9 else 0.0)
        seedStats = dict(
            nSeeds=len(extraSeeds),
            #the per-seed values are kept so two settings can be compared on the
            #same seeds rather than only by their means
            seedList=list(extraSeeds),
            charAccAll=[round(float(a),4) for a in accs],
            harmonicAll=[round(float(j),4) for j in joints],
            charAccMean=round(float(np.mean(accs)),4),
            charAccSd=round(float(np.std(accs,ddof=1)) if len(accs) > 1 else 0.0,4),
            widConfMean=round(float(np.mean(wids)),4),
            widConfSd=round(float(np.std(wids,ddof=1)) if len(wids) > 1 else 0.0,4),
            harmonicMean=round(float(np.mean(joints)),4),
            harmonicSd=round(float(np.std(joints,ddof=1)) if len(joints) > 1 else 0.0,4))
    m = dict(nStrokes=len(traj.strokes),seconds=round(secs,2),
             penDownMm=round(down,2),penUpMm=round(up_,2),
             lines=traj.meta.get("lines"),
             glyphSources=traj.meta.get("glyphSources"),
             lineStepMm=round(float(traj.meta.get("lineStepMm",0.0)),3),
             jointScore=traj.meta.get("jointScore"))
    if doScore:
        visible = text.replace("\n"," ")
        pred = REC.ReadText(img,rd)
        acc = REC.CharAcc(pred,visible)
        wantedAuthor = p.get("author") or prof.get("authorId")
        _pi,probs = REC.ClassifyImage(img,nm)
        wid = (float(probs[nm.authorMapping[wantedAuthor]])
               if wantedAuthor in nm.authorMapping else float("nan"))
        m.update(read=pred,charAcc=round(acc,4),widConf=round(wid,4),
                 harmonic=round(2*acc*wid/(acc+wid),4) if (acc+wid) > 1e-9 else 0.0)
    if seedStats:
        m.update(seedStats)
    pts,lens = Pack(traj.strokes)
    return m,dict(img=img,pts=pts,lens=lens)


#---TaskGcode: a trajectory to a G-code file, with the machine numbers it reports---
def TaskGcode(p):
    import synthesise as SY
    import reproduce as RP
    prof = json.loads(Path(p["profile"]).read_text(encoding="utf-8"))
    cfg = RP.GantryConfig()
    #useXhForRuling takes the deployed path, where the x-height is derived from
    #the ruling and the writer's own ratio rather than being passed in; that is
    #the only way the ruling coefficients are reachable
    if p.get("useXhForRuling"):
        mmPerXh = RP.XhForRuling(prof,cfg,text=p["text"],probe=True)
    else:
        mmPerXh = float(p.get("mmPerXh",4.0))
    traj = SY.SynthesizeText(p["text"],prof,mmPerXh=mmPerXh,
                             seed=int(p.get("seed",0)),
                             lineWidthMm=float(p.get("lineWidthMm",
                                                     cfg.usableWidthMm)),
                             jitter=float(p.get("jitter",0.15)))
    out = Path(p["gcodeOut"])
    stats = RP.WriteGcode(traj,cfg,out,title=p.get("title","sweep"))
    machine,scale = RP.ToMachine(traj,cfg)
    pts,lens = Pack(machine)
    m = {k:v for k,v in stats.items() if k != "placement"}
    m["placement"] = stats["placement"]
    m["mmPerXh"] = round(float(mmPerXh),4)
    m["cfg"] = dict(penUpDwellMs=cfg.penUpDwellMs,penDownDwellMs=cfg.penDownDwellMs,
                    joinStrokesMm=cfg.joinStrokesMm,closeRingMm=cfg.closeRingMm,
                    minSegmentMm=cfg.minSegmentMm,drawFeed=cfg.drawFeedMmMin,
                    travelFeed=cfg.travelFeedMmMin,
                    stepsPerMmX=round(cfg.stepsPerMmX,4),
                    stepsPerMmY=round(cfg.stepsPerMmY,4),
                    rulingXhFrac=cfg.rulingXhFrac,
                    rulingMaxExtent=cfg.rulingMaxExtent)
    return m,dict(pts=pts,lens=lens)


#---TaskBresenham: the exact step pattern BresenhamMove would issue, with no hardware---
def TaskBresenham(p):
    #the stepping arithmetic is lifted from BresenhamMove, which cannot run off
    #the board because it drives gpio lines, so the same error term is walked here
    dx,dy = int(p["dx"]),int(p["dy"])
    ax,ay = abs(dx),abs(dy)
    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1
    x = y = 0
    path = [(0,0)]
    if ax >= ay:
        err = ax//2
        for _ in range(ax):
            x += sx
            err -= ay
            if err < 0:
                y += sy
                err += ax
            path.append((x,y))
    else:
        err = ay//2
        for _ in range(ay):
            y += sy
            err -= ax
            if err < 0:
                x += sx
                err += ay
            path.append((x,y))
    a = np.asarray(path,np.float64)
    #perpendicular distance from the ideal straight line, in whole steps
    if ax or ay:
        L = np.hypot(dx,dy)
        dev = np.abs(dx*a[:,1]-dy*a[:,0])/max(L,1e-9)
    else:
        dev = np.zeros(1)
    m = dict(steps=len(path)-1,maxDeviationSteps=round(float(dev.max()),4),
             rmsDeviationSteps=round(float(np.sqrt((dev**2).mean())),4))
    return m,dict(path=a.astype(np.float32),dev=dev.astype(np.float32))


TASKS = {
    "page_segment": TaskPageSegment,
    "line_ink": TaskLineInk,
    "stroke_norm": TaskStrokeNorm,
    "trace": TaskTrace,
    "glyph_cut": TaskGlyphCut,
    "profile": TaskProfile,
    "synth": TaskSynth,
    "gcode": TaskGcode,
    "bresenham": TaskBresenham,
}


def Main():
    task = sys.argv[1]
    params = json.loads(sys.argv[2])
    outPrefix = Path(sys.argv[3])
    outPrefix.parent.mkdir(parents=True,exist_ok=True)
    t0 = time.time()
    metrics,arrays = TASKS[task](params)
    metrics["_task"] = task
    metrics["_wallSeconds"] = round(time.time()-t0,2)
    if arrays:
        np.savez_compressed(str(outPrefix)+".npz",**arrays)
    (Path(str(outPrefix)+".json")).write_text(
        json.dumps(metrics,indent=1,default=str),encoding="utf-8")
    print(json.dumps({"ok":True,"task":task,
                      "seconds":metrics["_wallSeconds"]}))


if __name__ == "__main__":
    Main()
