"""One try, best letterform only: the library's ceiling with the search switched off.

Every other run here draws N candidate renderings and keeps the one the reader
and the namer like best, so the number it reports mixes two things -- how good
the library is, and how lucky the search got.  This removes the second.  N is 1,
and every character is drawn with the writer's single best letterform rather
than sampled from their variants, so the result is deterministic: the same page
every time, for the same library.

That makes it the right measurement for judging a library (and for judging
curation), and the wrong one for judging the system: a real page varies its
letterforms, and drawing the same shape for every 'e' loses the variety the
writer classifier keys on.  Expect text accuracy to hold up and writer
identification to suffer.

Two senses of "best" exist in a profile and both are tried:
    medoid      the most typical variant the writer actually wrote
    prototype   an average of several of their variants, denoised

A character whose best letterform is still too far from typical
(priorD above IDEAL_ANCHOR_MAX_PRIORD) falls back to a generic alphabet that
is nobody's handwriting.  How often that happens is counted and reported,
because it would otherwise silently explain a bad writer-identification score.
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
MODES = [("medoid", "medoid"), ("prototype", "prototype")]
SEED  = 23
A4W_MM, A4H_MM = 210.0, 297.0
PX_PER_MM = 2448.0/A4W_MM

reader = REC.PaperCRNNNumpy(checkpointPath=REC.DEFAULT_TEXT_WEIGHTS)
shapeM = REC.AuthorClassifierCNNNumpy(checkpointPath=REC.SHAPE_AUTHOR_WEIGHTS)
shBy   = {v: k for k, v in shapeM.authorMapping.items()}
cfg    = REP.GantryConfig()

WRITERS = sorted(p.stem for p in paths.Profiles.glob("*.json")
                 if "_curated" not in p.name)


#----------------------------------------------------------------------
#AllIdeal: perCharLam for every character, whatever its position
#SynthesizeText reads this with .get(pos,0.0) and treats anything at or
#above 0.999 as "use the best letterform".  a plain dict would need every
#visible position enumerated correctly, and a defaultdict does not help
#because .get never triggers __missing__
#----------------------------------------------------------------------
class AllIdeal:
    def __bool__(self):
        return True

    def get(self, _k, _d=0.0):
        return 1.0


#----------------------------------------------------------------------
#GenericFraction: how much of this sentence is not the writer's hand
#_GlyphSource reports 'core' whether _CoreGlyph returned the writer's own
#best letterform or gave up and substituted the generic alphabet, so the
#usage counts cannot tell them apart.  this asks the profile the same
#question _CoreGlyph asks: is there a best letterform for this character,
#and is it close enough to typical to be used
#----------------------------------------------------------------------
def GenericFraction(prof, text, mode):
    src = (prof.get("idealMedoid") if mode == "medoid"
           else prof.get("idealGlyphs")) or {}
    chars = [c for c in text if c != " "]
    gen = 0
    for c in chars:
        own = src.get(c)
        if not own or float(own.get("priorD", 1.0)) > SYN.IDEAL_ANCHOR_MAX_PRIORD:
            gen += 1
    return 100.0*gen/max(1, len(chars))


def ProfileFor(writer):
    cur = paths.Profiles/("%s_curated.json" % writer)
    return (cur, "curated") if cur.exists() else (paths.Profiles/("%s.json" % writer),
                                                  "original")


def Sheet(strokes):
    W, H = int(A4W_MM*PX_PER_MM), int(A4H_MM*PX_PER_MM)
    im = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(im)
    for s in strokes:
        pts = [(x*PX_PER_MM, (cfg.originYmm-y)*PX_PER_MM) for (x, y) in s]
        if len(pts) > 1:
            d.line(pts, fill=0, width=max(2, int(0.35*PX_PER_MM)))
    return np.asarray(im, dtype=np.uint8)


def OnePage(writer, text, tag, mode, idx):
    profPath, profTag = ProfileFor(writer)
    prof = json.load(open(profPath, encoding="utf-8"))
    SYN.IDEAL_SOURCE = mode
    #one try, no search, every character on its best letterform
    traj = SYN.SynthesizeText(text, prof, seed=SEED, perCharLam=AllIdeal())
    src = dict(traj.meta.get("glyphSources") or {})
    stem = "%02d_%s_%s_%s" % (idx, writer, tag, mode)
    gpath = HERE/(stem+".gcode")
    written = REP.WriteGcode(traj, cfg, str(gpath),
                             title="%s, %s, best %s letterform" % (writer, tag, mode))
    back = REP.ParseGcode(str(gpath))
    page = Sheet(back)
    Image.fromarray(page).save(HERE/(stem+"_page.png"))

    lines, _dbg, _info = SEG.SegmentLines(page)
    texts, shTot = [], None
    for ln in lines:
        crop = Image.fromarray(np.asarray(ln["image"])).convert("L")
        texts.append(REC.ReadText(crop, reader))
        _k, pr = REC.ClassifyImage(crop, shapeM)
        shTot = np.asarray(pr, float) if shTot is None else shTot+pr
    got = " ".join(texts)
    shAvg = shTot/max(1, len(lines))

    ov = Image.fromarray(page).convert("RGB")
    dd = ImageDraw.Draw(ov)
    for ln in lines:
        b = ln["bbox"]
        dd.rectangle([b[0], b[1], b[2], b[3]], outline=(230, 30, 30), width=5)
    ov.save(HERE/(stem+"_page_segmented.png"))

    nAll = max(1, sum(src.values()))
    generic = GenericFraction(prof, text, mode)
    return dict(writer=writer, tag=tag, mode=mode, profile=profTag,
                chars=len(text), lines=len(lines),
                acc=REC.CharAcc(got, text), read=got,
                shapeName=shBy[int(np.argmax(shAvg))],
                shapeTop=float(shAvg.max()),
                shapeSelf=float(shAvg[shapeM.authorMapping[writer]]),
                fromIdeal=100.0*src.get("core", 0)/nAll,
                fromGeneric=generic,
                sources=src, strokes=written["strokes"])


rows = []
idx = 0
for mode, _m in MODES:
    for tag, text in SENTENCES:
        for w in WRITERS:
            idx += 1
            r = OnePage(w, text, tag, mode, idx)
            rows.append(r)
            print("%-9s %-9s %-6s | char %5.1f%% | shape %-8s top %3.0f%% self %3.0f%% %s "
                  "| ideal %3.0f%% generic %3.0f%% | %d strokes"
                  % (r["writer"], r["mode"], r["tag"], 100*r["acc"], r["shapeName"],
                     100*r["shapeTop"], 100*r["shapeSelf"],
                     "OK " if r["shapeName"] == r["writer"] else "WRONG",
                     r["fromIdeal"], r["fromGeneric"], r["strokes"]), flush=True)


def Summary(v):
    acc = np.array([r["acc"] for r in v])
    right = sum(1 for r in v if r["shapeName"] == r["writer"])
    return (100*acc.mean(), 100*np.median(acc), 100*acc.min(), 100*acc.max(),
            100*np.mean([r["shapeSelf"] for r in v]), right, len(v),
            100.0*right/len(v), np.mean([r["fromGeneric"] for r in v]))


with open(HERE/"RESULTS.txt", "w", encoding="utf-8") as f:
    f.write("one try, no best-of-N search, every character on its best letterform\n")
    for tag, text in SENTENCES:
        f.write('  %-6s %2d chars: "%s"\n' % (tag, len(text), text))
    f.write("seed %d, deterministic.  curated profile where one exists\n\n" % SEED)

    f.write("%-9s %-10s %-6s %8s  %-16s %9s %9s\n"
            % ("writer", "best", "length", "char %", "shape namer", "ideal %", "generic %"))
    for r in rows:
        f.write("%-9s %-10s %-6s %7.1f  %-16s %8.0f%% %8.0f%%\n"
                % (r["writer"], r["mode"], r["tag"], 100*r["acc"],
                   "%s %.0f%% %s" % (r["shapeName"], 100*r["shapeTop"],
                                     "ok" if r["shapeName"] == r["writer"] else "WRONG"),
                   r["fromIdeal"], r["fromGeneric"]))

    f.write("\n%-20s %7s %7s %7s %7s %10s %13s %9s\n"
            % ("group", "mean", "median", "worst", "best", "self conf",
               "named right", "generic"))
    for mode, _m in MODES:
        for tag, _t in SENTENCES:
            v = [r for r in rows if r["mode"] == mode and r["tag"] == tag]
            m, md, lo, hi, sc, right, n, pct, gen = Summary(v)
            f.write("%-20s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %9.1f%% %6d/%-2d %4.0f%% %8.0f%%\n"
                    % ("%s, %s" % (mode, tag), m, md, lo, hi, sc, right, n, pct, gen))
    for mode, _m in MODES:
        v = [r for r in rows if r["mode"] == mode]
        m, md, lo, hi, sc, right, n, pct, gen = Summary(v)
        f.write("%-20s %6.1f%% %6.1f%% %6.1f%% %6.1f%% %9.1f%% %6d/%-2d %4.0f%% %8.0f%%\n"
                % ("%s, both" % mode, m, md, lo, hi, sc, right, n, pct, gen))

    f.write("\nread back\n")
    for r in rows:
        f.write("%-9s %-10s %-6s %s\n" % (r["writer"], r["mode"], r["tag"], r["read"]))
print("wrote", HERE/"RESULTS.txt")
