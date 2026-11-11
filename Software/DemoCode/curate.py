"""Pick which of a writer's letterforms to keep, and rebuild the profile from those.

    python curate.py sheet  dylan                 draw every variant, numbered
    python curate.py list   dylan                 count them per character
    python curate.py drop   dylan --drop "a:2,5 e:0 o:1"
    python curate.py drop   dylan --keep "a:0,1 e:2"      the other way round

A profile holds several variants of each letter, chosen by filters that score
how typical a shape is. Those filters cannot tell a badly cut letter from an
unusual one, so a glyph that is simply wrong can survive. This lets a person
look at them and say which to keep.

`sheet` writes one numbered image per character, so the variants can be seen
before anything is removed. The numbers under each are what `drop` and `keep`
take. The Curate page in the web app does the same job by clicking: it draws
every variant from `Variants` below and sends the marked ones back to `Apply`,
which is the same code this file's `drop` ends up in.

Removing variants changes more than the library: the prototype letterform, the
most typical real variant, and the letter advances are all derived from the set
and are recomputed here the same way BuildAuthorProfile computes them. Nothing
is written over: the result is saved as `<writer>_curated.json`, and synthesis
can be pointed at it to compare against the original.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paths
import style as ST

OUT = HERE/"results"/"glyph_curation"


#---ParseSpec: "a:2,5 e:0" into {'a': {2,5}, 'e': {0}}---
def ParseSpec(spec):
    out = {}
    for part in (spec or "").split():
        if ":" not in part:
            raise SystemExit("expected char:indices, got %r" % part)
        ch, idx = part.split(":", 1)
        if len(ch) != 1:
            raise SystemExit("one character per entry, got %r" % ch)
        out.setdefault(ch, set()).update(int(i) for i in idx.split(",") if i != "")
    return out


#---LoadProfile: a writer's profile, curated version if one exists and is asked for---
def LoadProfile(writer, curated=False):
    name = "%s_curated.json" % writer if curated else "%s.json" % writer
    path = paths.Profiles/name
    if not path.exists():
        raise SystemExit("no profile at %s" % path)
    return json.load(open(path, encoding="utf-8")), path


#======================================================================
#SEEING THEM
#======================================================================
#---DrawGlyph: one variant on its own small canvas---
def DrawGlyph(g, cell=150, pad=14):
    from PIL import Image, ImageDraw
    im = Image.new("L", (cell, cell), 255)
    d = ImageDraw.Draw(im)
    strokes = g.get("strokes") or []
    pts = [p for s in strokes for p in s]
    if not pts:
        return im
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w = max(1e-6, max(xs)-min(xs))
    h = max(1e-6, max(ys)-min(ys))
    k = (cell-2*pad)/max(w, h)
    ox = (cell-w*k)/2.0
    oy = (cell-h*k)/2.0
    for s in strokes:
        q = [(ox+(p[0]-min(xs))*k, cell-(oy+(p[1]-min(ys))*k)) for p in s]
        if len(q) > 1:
            d.line(q, fill=0, width=2)
    return im


#----------------------------------------------------------------------
#Sheet: one image per character, every variant numbered
#the number under a variant is its index, which is what drop and keep
#take.  a variant marked proto is the denoised prototype the builder
#prepended rather than one the writer actually wrote
#----------------------------------------------------------------------
def Sheet(writer, curated=False, cell=150):
    from PIL import Image, ImageDraw
    prof, path = LoadProfile(writer, curated)
    glyphs = prof.get("glyphs") or {}
    outDir = OUT/writer
    outDir.mkdir(parents=True, exist_ok=True)
    made = []
    for ch in sorted(glyphs):
        vs = glyphs[ch]
        cols = min(10, max(1, len(vs)))
        rows = (len(vs)+cols-1)//cols
        im = Image.new("L", (cols*cell, rows*(cell+22)+26), 255)
        d = ImageDraw.Draw(im)
        d.text((6, 6), "%s  %r  %d variants" % (writer, ch, len(vs)), fill=0)
        for i, g in enumerate(vs):
            cx, cy = (i % cols)*cell, 26+(i//cols)*(cell+22)
            im.paste(DrawGlyph(g, cell), (cx, cy))
            tag = "%d%s" % (i, "  proto" if g.get("denoised") else "")
            d.text((cx+4, cy+cell+4), tag, fill=0)
            d.text((cx+4, cy+cell+12), "pd %.2f" % float(g.get("priorD", 0.5)), fill=0)
        safe = "".join(c if c.isalnum() else "_%02x" % ord(c) for c in ch)
        f = outDir/("%s.png" % safe)
        im.save(f)
        made.append((ch, len(vs), f.name))
    index = outDir/"INDEX.txt"
    with open(index, "w", encoding="utf-8") as fh:
        fh.write("%s, from %s\n\n" % (writer, path.name))
        fh.write("the number under a variant is its index, for --drop and --keep\n")
        fh.write("pd is the prior distance: lower means closer to the usual shape\n\n")
        for ch, n, name in made:
            fh.write("%-4r %3d variants  %s\n" % (ch, n, name))
    print("wrote %d character sheets to %s" % (len(made), outDir))
    print("  index: %s" % index)
    return made


#---Listing: how many variants each character has---
def Listing(writer, curated=False):
    prof, path = LoadProfile(writer, curated)
    glyphs = prof.get("glyphs") or {}
    print("%s, from %s: %d characters, %d variants in total"
          % (writer, path.name, len(glyphs), sum(len(v) for v in glyphs.values())))
    for ch in sorted(glyphs):
        vs = glyphs[ch]
        pd = [float(g.get("priorD", 0.5)) for g in vs]
        proto = sum(1 for g in vs if g.get("denoised"))
        print("  %-4r %3d variants   prior distance min %.2f median %.2f max %.2f%s"
              % (ch, len(vs), min(pd), float(np.median(pd)), max(pd),
                 "   (%d prototype)" % proto if proto else ""))


#======================================================================
#REBUILDING WHAT THE SET DECIDES
#the library is not the only thing that comes from the variants: the
#prototype letterform, the most typical real variant and the letter
#advances are all derived from them, so all three are recomputed rather
#than left describing glyphs that are no longer there
#======================================================================
def Medoid(ch, vs):
    grids = [(ST._ShapeGrid(g), g) for g in vs]
    grids = [(v, g) for v, g in grids if v is not None]
    if len(grids) < 2:
        return None
    G = np.stack([v for v, _ in grids])
    D = np.abs(G[:, None, :]-G[None, :, :]).sum(-1)
    med = grids[int(np.argmin(D.sum(1)))][1]
    return dict(strokes=[ST.SmoothStroke(s) for s in med["strokes"]],
                priorD=round(float(med.get("priorD", 0.5)), 4),
                advance=round(float(med.get("advance", 0.9)), 3),
                lead=round(float(med.get("lead", 0.0)), 3),
                width=round(float(med.get("width", 0.9)), 3),
                top=round(float(med.get("top", 1.0)), 3),
                bot=round(float(med.get("bot", 0.0)), 3),
                entryY=round(float(med.get("entryY", 0.3)), 3),
                exitY=round(float(med.get("exitY", 0.3)), 3),
                nPool=len(vs))


def Prototype(ch, vs):
    proto = ST._DenoisedPrototype(vs)
    if proto is None:
        return None
    pts = [p for s in proto for p in s]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return dict(strokes=[ST.SmoothStroke(s) for s in proto],
                priorD=round(float(np.median([g.get("priorD", 0.5) for g in vs])), 4),
                advance=round(float(np.median([g["advance"] for g in vs])), 3),
                lead=round(float(np.median([g.get("lead", 0.0) for g in vs])), 3),
                width=round(max(xs)-min(xs), 3),
                top=round(max(ys), 3), bot=round(min(ys), 3),
                entryY=round(float(np.median([g.get("entryY", 0.3) for g in vs])), 3),
                exitY=round(float(np.median([g.get("exitY", 0.3) for g in vs])), 3),
                nPool=len(vs))


#----------------------------------------------------------------------
#Variants: the whole set in a form something else can draw
#the command line draws its own sheets, but the curation page in the web
#app needs the strokes themselves so it can draw them and be clicked on.
#the indices here are the ones Apply takes, so what is seen and what is
#removed cannot drift apart
#----------------------------------------------------------------------
def Variants(writer, curated=False):
    prof, path = LoadProfile(writer, curated)
    glyphs = prof.get("glyphs") or {}
    chars = []
    for ch in sorted(glyphs):
        vs = []
        for i, g in enumerate(glyphs[ch]):
            vs.append(dict(index=i,
                           priorD=round(float(g.get("priorD", 0.5)), 4),
                           advance=round(float(g.get("advance", 0.9)), 3),
                           denoised=bool(g.get("denoised")),
                           strokes=[[[round(float(p[0]), 4), round(float(p[1]), 4)]
                                     for p in s] for s in (g.get("strokes") or [])]))
        chars.append(dict(ch=ch, variants=vs))
    return dict(writer=writer, source=path.name, characters=chars,
                nChars=len(chars), nVariants=sum(len(c["variants"]) for c in chars),
                curatedFrom=prof.get("curatedFrom"),
                hasCurated=(paths.Profiles/("%s_curated.json" % writer)).exists())


#---Drop: the command line's way in, char:index strings---
def Drop(writer, dropSpec=None, keepSpec=None, curated=False, outName=None):
    return Apply(writer, ParseSpec(dropSpec), ParseSpec(keepSpec), curated, outName)


#---Apply: the real work, on {char: {indices}} as the page sends them---
def Apply(writer, drop=None, keep=None, curated=False, outName=None):
    prof, path = LoadProfile(writer, curated)
    glyphs = dict(prof.get("glyphs") or {})
    drop = {k: set(v) for k, v in (drop or {}).items() if v}
    keep = {k: set(v) for k, v in (keep or {}).items()}
    for ch in list(drop)+list(keep):
        if ch not in glyphs:
            raise SystemExit("%r has no variants in %s" % (ch, path.name))

    removed, emptied = 0, []
    for ch, vs in list(glyphs.items()):
        if ch in keep:
            chosen = [g for i, g in enumerate(vs) if i in keep[ch]]
        elif ch in drop:
            chosen = [g for i, g in enumerate(vs) if i not in drop[ch]]
        else:
            continue
        removed += len(vs)-len(chosen)
        if not chosen:
            emptied.append(ch)
            del glyphs[ch]
        else:
            glyphs[ch] = chosen

    prof["glyphs"] = glyphs
    #the three things the set decides, recomputed from what is left
    ideal = dict(prof.get("idealGlyphs") or {})
    medoid = dict(prof.get("idealMedoid") or {})
    adv = dict(prof.get("letterAdvance") or {})
    touched = set(drop) | set(keep)
    for ch in touched:
        if ch not in glyphs:
            ideal.pop(ch, None)
            medoid.pop(ch, None)
            adv.pop(ch, None)
            continue
        vs = glyphs[ch]
        p = Prototype(ch, vs)
        m = Medoid(ch, vs)
        if p is None:
            ideal.pop(ch, None)
        else:
            ideal[ch] = p
        if m is None:
            medoid.pop(ch, None)
        else:
            medoid[ch] = m
        adv[ch] = round(float(np.median([g["advance"] for g in vs])), 3)
    prof["idealGlyphs"] = ideal
    prof["idealMedoid"] = medoid
    prof["letterAdvance"] = adv
    #connectedness is a ratio over adjacent pairs seen while reading the pages,
    #and which letters sat next to which is not kept, so it cannot be recomputed
    #faithfully here.  it is left alone, and that is recorded in the file
    prof["curatedFrom"] = path.name
    prof["curatedRemoved"] = removed
    prof["curatedNote"] = ("glyphs, idealGlyphs, idealMedoid and letterAdvance "
                           "were rebuilt from the kept variants; connectedness "
                           "is carried over unchanged because the adjacency it "
                           "was counted from is not stored")

    out = paths.Profiles/(outName or ("%s_curated.json" % writer))
    out.write_text(json.dumps(prof), encoding="utf-8")
    summary = dict(writer=writer, source=path.name, out=out.name, path=str(out),
                   removed=removed, emptied=emptied,
                   rebuilt=sorted(c for c in touched if c in glyphs),
                   nChars=len(glyphs),
                   nVariants=sum(len(v) for v in glyphs.values()),
                   note=prof["curatedNote"])
    print("removed %d variants" % removed)
    if emptied:
        print("  characters left with none, and dropped: %s" % ", ".join(repr(c) for c in emptied))
    print("  recomputed prototype, medoid and advance for: %s"
          % ", ".join(repr(c) for c in summary["rebuilt"]))
    print("  %d characters and %d variants remain" % (summary["nChars"], summary["nVariants"]))
    print("  wrote %s" % out)
    return summary


def Main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=["sheet", "list", "drop"])
    ap.add_argument("writer")
    ap.add_argument("--drop", default=None, help='e.g. "a:2,5 e:0"')
    ap.add_argument("--keep", default=None, help='e.g. "a:0,1" keeps only those')
    ap.add_argument("--curated", action="store_true",
                    help="work from <writer>_curated.json instead of the original")
    ap.add_argument("--out", default=None, help="name for the written profile")
    a = ap.parse_args()
    if a.action == "sheet":
        Sheet(a.writer, a.curated)
    elif a.action == "list":
        Listing(a.writer, a.curated)
    else:
        if not (a.drop or a.keep):
            raise SystemExit("give --drop or --keep")
        Drop(a.writer, a.drop, a.keep, a.curated, a.out)


if __name__ == "__main__":
    Main()
