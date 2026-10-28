"""All thirteen writers, a long sentence and a short one, through the whole path.

Every published accuracy figure for this system was measured on one sentence
length, and sentence length moves the numbers by more than any change to the
synthesiser does: the same writers scored about forty points apart on a 21
character sentence and a 44 character one.  A reader that drops a word loses a
larger share of a short line, while a long line gives the segmenter more
chances to cut badly, so the two lengths do not merely scale.

Method is gcode_a4_pages/make.py: synthesise best-of-N, write the G-code, parse
it back with the separate reader, draw it on an A4 canvas at the photographed
pages' scale, then segment, read and name it.  Nothing here touches the gantry,
so what it measures is the loss from going through G-code, before the machine
adds any of its own.

Both namers are reported, but only the stroke-normalised one means anything on
this output: the ink namer reads real photographs, and a drawn page has one pen
width everywhere, so it carries no ink evidence at all.
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import numpy as np
from PIL import Image, ImageDraw
import paths, recognise as REC, reproduce as REP, segment as SEG, synthesise as SYN

SENTENCES = [("long",  "remove the inconsistencies on the belt pulley"),
             ("short", "this gantry writes today")]
NTRIES = 14
SEED   = 23
A4W_MM, A4H_MM = 210.0, 297.0
PX_PER_MM = 2448.0/A4W_MM          #the same scale as the photographed pages

reader = REC.PaperCRNNNumpy(checkpointPath=REC.DEFAULT_TEXT_WEIGHTS)
inkM   = REC.AuthorClassifierCNNNumpy(checkpointPath=REC.DEFAULT_AUTHOR_WEIGHTS)
shapeM = REC.AuthorClassifierCNNNumpy(checkpointPath=REC.SHAPE_AUTHOR_WEIGHTS)
inkBy  = {v: k for k, v in inkM.authorMapping.items()}
shBy   = {v: k for k, v in shapeM.authorMapping.items()}
cfg    = REP.GantryConfig()

WRITERS = sorted(p.stem for p in paths.Profiles.glob("*.json")
                 if "_curated" not in p.name)


def Sheet(strokes):
    """machine millimetres onto an A4 canvas, sheet top at the work-area origin"""
    W, H = int(A4W_MM*PX_PER_MM), int(A4H_MM*PX_PER_MM)
    im = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(im)
    for s in strokes:
        pts = [(x*PX_PER_MM, (cfg.originYmm-y)*PX_PER_MM) for (x, y) in s]
        if len(pts) > 1:
            d.line(pts, fill=0, width=max(2, int(0.35*PX_PER_MM)))
    return np.asarray(im, dtype=np.uint8)


def PenUpTravelMm(strokes):
    tot, at = 0.0, None
    for s in strokes:
        if not s:
            continue
        if at is not None:
            tot += float(np.hypot(s[0][0]-at[0], s[0][1]-at[1]))
        at = s[-1]
    return tot


#---ProfileFor: the writer's curated library if they have one---
#these were curated by hand against the ORIGINAL cut, so they are not
#affected by the crossbar fix in style.py: that changes what a rebuild
#would produce, and nothing here rebuilds
def ProfileFor(writer):
    cur = paths.Profiles/("%s_curated.json" % writer)
    return (cur, "curated") if cur.exists() else (paths.Profiles/("%s.json" % writer),
                                                  "original")


def OnePage(writer, text, tag, idx):
    profPath, profTag = ProfileFor(writer)
    prof = json.load(open(profPath, encoding="utf-8"))
    traj = SYN.SynthesizeJointBestOf(writer, text, prof, nTries=NTRIES, seed=SEED,
                                     reader=reader, authorModel=shapeM,
                                     authorMapping=shapeM.authorMapping)
    stem = "%02d_%s_%s" % (idx, writer, tag)
    gpath = HERE/(stem+".gcode")
    written = REP.WriteGcode(traj, cfg, str(gpath),
                             title="A4 page, %s, %s sentence" % (writer, tag))
    back = REP.ParseGcode(str(gpath))
    page = Sheet(back)
    Image.fromarray(page).save(HERE/(stem+"_page.png"))

    lines, _dbg, _info = SEG.SegmentLines(page)
    texts, inkTot, shTot = [], None, None
    for ln in lines:
        crop = Image.fromarray(np.asarray(ln["image"])).convert("L")
        texts.append(REC.ReadText(crop, reader))
        _k, pr = REC.ClassifyImage(crop, inkM, strokeNormalise=False)
        inkTot = np.asarray(pr, float) if inkTot is None else inkTot+pr
        _k, pr = REC.ClassifyImage(crop, shapeM)
        shTot = np.asarray(pr, float) if shTot is None else shTot+pr
    got = " ".join(texts)
    inkAvg = inkTot/max(1, len(lines))
    shAvg = shTot/max(1, len(lines))

    ov = Image.fromarray(page).convert("RGB")
    dd = ImageDraw.Draw(ov)
    for ln in lines:
        b = ln["bbox"]
        dd.rectangle([b[0], b[1], b[2], b[3]], outline=(230, 30, 30), width=5)
    ov.save(HERE/(stem+"_page_segmented.png"))

    return dict(writer=writer, tag=tag, profile=profTag, chars=len(text), lines=len(lines),
                acc=REC.CharAcc(got, text), read=got,
                inkName=inkBy[int(np.argmax(inkAvg))], inkConf=float(inkAvg.max()),
                shapeName=shBy[int(np.argmax(shAvg))],
                shapeTop=float(shAvg.max()),
                shapeSelf=float(shAvg[shapeM.authorMapping[writer]]),
                strokes=written["strokes"], pulses=written["penPulses"],
                travel=PenUpTravelMm(back), gcode=gpath.name)


rows = []
idx = 0
for tag, text in SENTENCES:
    for w in WRITERS:
        idx += 1
        r = OnePage(w, text, tag, idx)
        rows.append(r)
        print("%-9s %-8s %-6s %2d chars | %d line(s) | char %5.1f%% | shape %-8s top %3.0f%% "
              "self %3.0f%% %s | ink %-8s %3.0f%% | %d strokes %.0f mm pen-up"
              % (r["writer"], r["profile"], r["tag"], r["chars"], r["lines"], 100*r["acc"],
                 r["shapeName"], 100*r["shapeTop"], 100*r["shapeSelf"],
                 "OK " if r["shapeName"] == r["writer"] else "WRONG",
                 r["inkName"], 100*r["inkConf"], r["strokes"], r["travel"]),
              flush=True)


#----------------------------------------------------------------------
#the write-up.  the per-length summary is the point: one number per
#sentence hides the thing the experiment was run to see
#----------------------------------------------------------------------
def Summary(v):
    acc = np.array([r["acc"] for r in v])
    self_ = np.array([r["shapeSelf"] for r in v])
    right = sum(1 for r in v if r["shapeName"] == r["writer"])
    return (100*acc.mean(), 100*np.median(acc), 100*acc.min(), 100*acc.max(),
            100*self_.mean(), right, len(v), 100.0*right/len(v))


with open(HERE/"RESULTS.txt", "w", encoding="utf-8") as f:
    f.write("all %d writers with a style profile, two sentence lengths\n" % len(WRITERS))
    for tag, text in SENTENCES:
        f.write('  %-6s %2d chars: "%s"\n' % (tag, len(text), text))
    f.write("nTries %d, seed %d, margins %.0f mm left / %.0f mm top, A4 at %.2f px/mm\n\n"
            % (NTRIES, SEED, cfg.pageMarginXmm, cfg.pageMarginYmm, PX_PER_MM))

    f.write("%-9s %-6s %6s %6s %8s  %-16s %-14s\n"
            % ("writer", "length", "chars", "lines", "char %", "shape namer", "ink namer"))
    for r in rows:
        f.write("%-9s %-6s %6d %6d %7.1f  %-16s %-14s\n"
                % (r["writer"], r["tag"], r["chars"], r["lines"], 100*r["acc"],
                   "%s %.0f%% %s" % (r["shapeName"], 100*r["shapeTop"],
                                     "ok" if r["shapeName"] == r["writer"] else "WRONG"),
                   "%s %.0f%%" % (r["inkName"], 100*r["inkConf"])))

    f.write("\nby sentence length\n")
    f.write("%-8s %7s %7s %7s %7s %10s %14s\n"
            % ("length", "mean", "median", "worst", "best", "self conf", "named right"))
    for tag, _t in SENTENCES:
        v = [r for r in rows if r["tag"] == tag]
        m, md, lo, hi, sc, right, n, pct = Summary(v)
        f.write("%-8s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %9.1f%% %7d/%-2d %4.0f%%\n"
                % (tag, m, md, lo, hi, sc, right, n, pct))
    v = rows
    m, md, lo, hi, sc, right, n, pct = Summary(v)
    f.write("%-8s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %9.1f%% %7d/%-2d %4.0f%%\n"
            % ("both", m, md, lo, hi, sc, right, n, pct))

    f.write("\npersonal writers only (the five photographed hands)\n")
    for tag, _t in SENTENCES:
        v = [r for r in rows if r["tag"] == tag and r["writer"] in paths.PersonalWriters]
        if v:
            m, md, lo, hi, sc, right, n, pct = Summary(v)
            f.write("%-8s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %9.1f%% %7d/%-2d %4.0f%%\n"
                    % (tag, m, md, lo, hi, sc, right, n, pct))

    f.write("\nread back\n")
    for r in rows:
        f.write("%-9s %-6s %s\n" % (r["writer"], r["tag"], r["read"]))
print("wrote", HERE/"RESULTS.txt")
