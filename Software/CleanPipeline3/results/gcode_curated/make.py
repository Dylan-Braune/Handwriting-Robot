"""The same A4 page from each writer's original profile and from his curated one.

The variant filters score how typical a shape is, which cannot tell a badly cut
letter from an unusual one, so the library holds some glyphs that are simply
wrong.  Those were removed by hand in the Curate page.  Removing them cuts the
pool the synthesiser draws from, which could help (fewer wrong letterforms) or
hurt (less of the writer's real variety, and a smaller pool for the best-of-N
search to pick from), so it has to be measured rather than assumed.

Method is gcode_a4_pages/make.py unchanged, run twice per writer: synthesise
best-of-N, write G-code, parse it back with the separate reader, draw it on an
A4 canvas at the photographed pages' scale, then segment, read and name it.
Same sentence, same seed and same nTries both times, so the only difference
between the two rows is which glyphs were available.
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import numpy as np
from PIL import Image, ImageDraw
import paths, recognise as REC, reproduce as REP, segment as SEG, synthesise as SYN

TEXT   = "remove the inconsistencies on the belt pulley"
NTRIES = 14
SEED   = 23
A4W_MM, A4H_MM = 210.0, 297.0
PX_PER_MM = 2448.0/A4W_MM          #the same scale as the photographed pages

reader = REC.PaperCRNNNumpy(checkpointPath=REC.DEFAULT_TEXT_WEIGHTS)
inkM   = REC.AuthorClassifierCNNNumpy(checkpointPath=REC.DEFAULT_AUTHOR_WEIGHTS)
shapeM = REC.AuthorClassifierCNNNumpy(checkpointPath=REC.SHAPE_AUTHOR_WEIGHTS)
inkBy  = {v:k for k,v in inkM.authorMapping.items()}
shBy   = {v:k for k,v in shapeM.authorMapping.items()}
cfg    = REP.GantryConfig()


def Sheet(strokes):
    """machine millimetres onto an A4 canvas, sheet top at the work-area origin"""
    W,H = int(A4W_MM*PX_PER_MM), int(A4H_MM*PX_PER_MM)
    im = Image.new("L",(W,H),255)
    d = ImageDraw.Draw(im)
    for s in strokes:
        pts = [(x*PX_PER_MM, (cfg.originYmm-y)*PX_PER_MM) for (x,y) in s]
        if len(pts) > 1:
            d.line(pts, fill=0, width=max(2,int(0.35*PX_PER_MM)))
    return np.asarray(im, dtype=np.uint8)


def PenUpTravelMm(strokes):
    """how far the pen travels lifted: the cost of the stroke ordering"""
    tot, at = 0.0, None
    for s in strokes:
        if not s:
            continue
        if at is not None:
            tot += float(np.hypot(s[0][0]-at[0], s[0][1]-at[1]))
        at = s[-1]
    return tot


#----------------------------------------------------------------------
#OnePage: one writer, one profile, start to finish
#returns what the page measured, so the two profiles can be put side by
#side.  nothing here is specific to curation: it is the ordinary
#reproduction path, which is the point
#----------------------------------------------------------------------
def OnePage(writer, profilePath, tag, idx):
    prof = json.load(open(profilePath, encoding="utf-8"))
    nVariants = sum(len(v) for v in prof["glyphs"].values())
    traj = SYN.SynthesizeJointBestOf(writer, TEXT, prof, nTries=NTRIES, seed=SEED,
                                     reader=reader, authorModel=shapeM,
                                     authorMapping=shapeM.authorMapping)
    gpath = HERE/("%d_%s_%s.gcode"%(idx,writer,tag))
    written = REP.WriteGcode(traj, cfg, str(gpath),
                             title="A4 page, %s, %s profile"%(writer,tag))
    back = REP.ParseGcode(str(gpath))        #the separate reader, not the writer
    page = Sheet(back)
    Image.fromarray(page).save(HERE/("%d_%s_%s_page.png"%(idx,writer,tag)))

    lines,_dbg,_info = SEG.SegmentLines(page)
    texts, inkTot, shTot = [], None, None
    for ln in lines:
        crop = Image.fromarray(np.asarray(ln["image"])).convert("L")
        texts.append(REC.ReadText(crop, reader))
        _k,pr = REC.ClassifyImage(crop, inkM, strokeNormalise=False)
        inkTot = np.asarray(pr,float) if inkTot is None else inkTot+pr
        _k,pr = REC.ClassifyImage(crop, shapeM)
        shTot = np.asarray(pr,float) if shTot is None else shTot+pr
    got = " ".join(texts)
    inkAvg = inkTot/max(1,len(lines))
    shAvg  = shTot/max(1,len(lines))

    ov = Image.fromarray(page).convert("RGB"); dd = ImageDraw.Draw(ov)
    for ln in lines:
        b = ln["bbox"]; dd.rectangle([b[0],b[1],b[2],b[3]], outline=(230,30,30), width=5)
    ov.save(HERE/("%d_%s_%s_page_segmented.png"%(idx,writer,tag)))

    return dict(writer=writer, tag=tag, variants=nVariants, lines=len(lines),
                acc=REC.CharAcc(got, TEXT), read=got,
                inkName=inkBy[int(np.argmax(inkAvg))], inkConf=float(inkAvg.max()),
                shapeName=shBy[int(np.argmax(shAvg))],
                shapeSelf=float(shAvg[shapeM.authorMapping[writer]]),
                pulses=written["penPulses"], strokes=written["strokes"],
                travel=PenUpTravelMm(back), gcode=gpath.name)


rows = []
for i,w in enumerate(paths.PersonalWriters,1):
    orig = paths.Profiles/("%s.json"%w)
    cur  = paths.Profiles/("%s_curated.json"%w)
    if not cur.exists():
        print("%-9s skipped: no curated profile on disk" % w, flush=True)
        continue
    for tag,p in (("original",orig),("curated",cur)):
        r = OnePage(w, p, tag, i)
        rows.append(r)
        print("%-9s %-8s %4d variants | %d line(s) | char %5.1f%% | ink %-8s %3.0f%% | "
              "shape %-8s self %3.0f%% | %d strokes, %d pulses, %.0f mm pen-up"
              % (w, tag, r["variants"], r["lines"], 100*r["acc"], r["inkName"],
                 100*r["inkConf"], r["shapeName"], 100*r["shapeSelf"],
                 r["strokes"], r["pulses"], r["travel"]), flush=True)

#----------------------------------------------------------------------
#the write-up.  the per-writer deltas matter more than the mean: a pool
#cut by a third is a different experiment from one cut by a tenth
#----------------------------------------------------------------------
with open(HERE/"RESULTS.txt","w",encoding="utf-8") as f:
    f.write('sentence: "%s"\n' % TEXT)
    f.write("nTries %d, seed %d, margins %.0f mm left / %.0f mm top, A4 at %.2f px/mm\n"
            % (NTRIES,SEED,cfg.pageMarginXmm,cfg.pageMarginYmm,PX_PER_MM))
    f.write("identical settings for both profiles: the only difference is which "
            "glyph variants were available\n\n")
    f.write("%-9s %-9s %9s %6s %8s  %-14s %-14s\n"
            % ("writer","profile","variants","lines","char %","ink namer","shape namer"))
    for r in rows:
        f.write("%-9s %-9s %9d %6d %7.1f  %-14s %-14s\n"
                % (r["writer"],r["tag"],r["variants"],r["lines"],100*r["acc"],
                   "%s %.0f%%"%(r["inkName"],100*r["inkConf"]),
                   "%s %.0f%%"%(r["shapeName"],100*r["shapeSelf"])))
    f.write("\nchange, curated minus original\n")
    f.write("%-9s %10s %10s %10s %10s %12s\n"
            % ("writer","variants","char %","shape self","strokes","pen-up mm"))
    byW = {}
    for r in rows:
        byW.setdefault(r["writer"],{})[r["tag"]] = r
    for w,d in byW.items():
        if "original" not in d or "curated" not in d:
            continue
        o,c = d["original"],d["curated"]
        f.write("%-9s %10d %10.1f %10.1f %10d %12.0f\n"
                % (w, c["variants"]-o["variants"], 100*(c["acc"]-o["acc"]),
                   100*(c["shapeSelf"]-o["shapeSelf"]), c["strokes"]-o["strokes"],
                   c["travel"]-o["travel"]))
    f.write("\nread back\n")
    for r in rows:
        f.write("%-9s %-9s %s\n" % (r["writer"],r["tag"],r["read"]))
    missing = [w for w in paths.PersonalWriters
               if not (paths.Profiles/("%s_curated.json"%w)).exists()]
    if missing:
        f.write("\nnot tested, no curated profile: %s\n" % ", ".join(missing))
print("wrote", HERE/"RESULTS.txt")
