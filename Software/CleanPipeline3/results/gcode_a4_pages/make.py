"""G-code for each writer, laid out on an A4 sheet, then read back as if photographed.

The point is to check that what the gantry will draw can be segmented and
recognised.  The G-code is written, parsed back with the separate reader, drawn
onto an A4-sized canvas at the sheet's real scale, and then put through the
ordinary classify path: segment, read, name.  Any loss here is the price of
going through G-code, before the machine has introduced any of its own.
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

rows = []
for i,w in enumerate(paths.PersonalWriters,1):
    prof = json.load(open(paths.Profiles/("%s.json"%w), encoding="utf-8"))
    traj = SYN.SynthesizeJointBestOf(w, TEXT, prof, nTries=NTRIES, seed=SEED,
                                     reader=reader, authorModel=shapeM,
                                     authorMapping=shapeM.authorMapping)
    gpath = HERE/("%d_%s.gcode"%(i,w))
    written = REP.WriteGcode(traj, cfg, str(gpath), title="A4 page, %s"%w)
    back = REP.ParseGcode(str(gpath))        #the separate reader, not the writer
    page = Sheet(back)
    Image.fromarray(page).save(HERE/("%d_%s_page.png"%(i,w)))

    lines,_dbg,info = SEG.SegmentLines(page)
    texts, inkTot, shTot = [], None, None
    for ln in lines:
        crop = Image.fromarray(np.asarray(ln["image"])).convert("L")
        texts.append(REC.ReadText(crop, reader))
        _k,pr = REC.ClassifyImage(crop, inkM, strokeNormalise=False)
        inkTot = np.asarray(pr,float) if inkTot is None else inkTot+pr
        _k,pr = REC.ClassifyImage(crop, shapeM)
        shTot = np.asarray(pr,float) if shTot is None else shTot+pr
    got = " ".join(texts)
    acc = REC.CharAcc(got, TEXT)
    inkAvg, shAvg = inkTot/max(1,len(lines)), shTot/max(1,len(lines))
    inkName, shName = inkBy[int(np.argmax(inkAvg))], shBy[int(np.argmax(shAvg))]

    ov = Image.fromarray(page).convert("RGB"); dd = ImageDraw.Draw(ov)
    for k,ln in enumerate(lines):
        b = ln["bbox"]; dd.rectangle([b[0],b[1],b[2],b[3]], outline=(230,30,30), width=5)
    ov.save(HERE/("%d_%s_page_segmented.png"%(i,w)))

    rows.append((w,len(lines),acc,inkName,float(inkAvg.max()),shName,
                 float(shAvg[shapeM.authorMapping[w]]),got,written["penPulses"]))
    print("%-9s %d line(s) | char %5.1f%% | ink namer %-8s %3.0f%% | shape namer %-8s (self %3.0f%%) | %d pulses"
          % (w,len(lines),100*acc,inkName,100*float(inkAvg.max()),shName,
             100*float(shAvg[shapeM.authorMapping[w]]),written["penPulses"]), flush=True)

with open(HERE/"RESULTS.txt","w",encoding="utf-8") as f:
    f.write('sentence: "%s"\n' % TEXT)
    f.write("nTries %d, seed %d, margins %.0f mm left / %.0f mm top, A4 at %.2f px/mm\n\n"
            % (NTRIES,SEED,cfg.pageMarginXmm,cfg.pageMarginYmm,PX_PER_MM))
    f.write("%-10s %5s %8s  %-10s %-10s %s\n"
            % ("writer","lines","char %","ink namer","shape namer","read back"))
    for w,n,acc,ia,ic,sa,sc,got,pp in rows:
        f.write("%-10s %5d %7.1f  %-10s %-10s %s\n"
                % (w,n,100*acc,"%s %.0f%%"%(ia,100*ic),"%s %.0f%%"%(sa,100*sc),got))
print("wrote", HERE/"RESULTS.txt")
