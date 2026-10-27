"""Trace a photographed page's own ink into G-code, so the gantry redraws it.

    python retrace.py <image> [-o out.gcode] [--preview out.png]

The reproduction path normally goes photograph -> read -> style -> synthesise
-> G-code, and any error in the drawn result could have come from any of those
stages.  This takes the shortest possible route instead: the same conditioning
and line separation, then the ink itself thinned to its centre line and
followed, with no recognition, no glyph library and no synthesis.

What comes out is the writer's actual handwriting as a pen path, so drawing it
and comparing against the original page measures the gantry alone: its
resolution, its backlash, its pen control.  Whatever differs is mechanical,
because nothing in between made a decision.

It is a measuring instrument, not part of the reproduction pipeline.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import paths
import reproduce as REP
import segment as SEG
import style as ST
import synthesise as SYN
import timing as TIMING
import train as TR


#---MIN_STROKE_PTS: paths shorter than this are speckle, not writing---
MIN_STROKE_PTS = 4

#---MIN_STROKE_XH: and a stroke shorter than this fraction of an x-height---
MIN_STROKE_XH = 0.12


#----------------------------------------------------------------------
#PageStrokes: a photographed page as pen paths in page coordinates
#each line is conditioned and cut out by the ordinary segmenter, then its
#ink is thinned and followed.  the crop's own offset puts every path back
#where it belongs on the page, so the result is one coordinate system
#----------------------------------------------------------------------
def PageStrokes(gray, timer=None):
    lines, _dbg, info = SEG.SegmentLines(gray, timer=timer)
    strokes = []
    perLine = []
    for ln in lines:
        crop = np.asarray(ln["image"], dtype=np.uint8)
        ink = SEG.BinarizeLine(crop)
        if not ink.any():
            perLine.append(0)
            continue
        polys = ST.TracePolylines(SEG.Skeletonize(ink))
        x0, y0 = (ln["bbox"][0], ln["bbox"][1]) if "bbox" in ln else (0, 0)
        kept = 0
        for p in polys:
            if len(p) < MIN_STROKE_PTS:
                continue
            #y is flipped here: the page counts rows downwards, a trajectory
            #counts millimetres upwards, and the writer flips nothing
            strokes.append([(float(x)+x0, -(float(y)+y0)) for (x, y) in p])
            kept += 1
        perLine.append(kept)
    return strokes, dict(lines=len(lines), strokesPerLine=perLine, **info)


#---PruneSpecks: drop paths too short to be writing---
def PruneSpecks(strokes, xhPx):
    limit = MIN_STROKE_XH*max(1.0, xhPx)
    out = []
    for s in strokes:
        span = max(max(p[0] for p in s)-min(p[0] for p in s),
                   max(p[1] for p in s)-min(p[1] for p in s))
        if len(s) >= MIN_STROKE_PTS and span >= limit:
            out.append(s)
    return out


#----------------------------------------------------------------------
#RetracePage: the whole job, photograph in and a g-code file out
#the trajectory is handed to the ordinary writer, so the page is placed,
#scaled to the work area and ordered by exactly the same code a
#synthesised line goes through.  only the source of the strokes differs
#----------------------------------------------------------------------
def RetracePage(imagePath, outPath, cfg=None, previewPath=None):
    cfg = cfg or REP.GantryConfig()
    run = TIMING.Run("retrace")

    with run.Step("loading"):
        gray = TR.ReadImageFile(imagePath)

    strokes, info = PageStrokes(gray, timer=run)
    with run.Step("tracing"):
        strokes = PruneSpecks(strokes, info.get("pitch", 40)*0.4)
    if not strokes:
        raise SystemExit("no ink found to trace in %s" % imagePath)

    traj = SYN.Trajectory(strokes, dict(author="retrace", source=str(imagePath),
                                        lines=info["lines"]))
    with run.Step("gcode"):
        written = REP.WriteGcode(traj, cfg, str(outPath),
                                 title="retrace of %s" % Path(imagePath).name)

    preview = None
    if previewPath:
        with run.Step("preview"):
            preview = SYN.RenderTrajectory(traj, pxPerMm=6.0)
            from PIL import Image
            Image.fromarray(np.asarray(preview, dtype=np.uint8)).save(previewPath)

    run.chars = 0
    run.lines = info["lines"]
    #WriteGcode reports a stroke count of its own, after ordering and clamping
    out = dict(written)
    out.update(gcode=str(outPath), tracedStrokes=len(strokes), timing=run.AsDict(),
               preview=str(previewPath) if previewPath else None)
    return out


def Main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--preview", default=None)
    args = ap.parse_args()
    out = args.out or str(Path(args.image).with_suffix(".gcode").name)
    r = RetracePage(args.image, out, previewPath=args.preview)
    print("traced %d lines into %d strokes" % (r["timing"]["lines"], r["tracedStrokes"]))
    print("  %s: %d g-code lines, %d pen pulses, scale %.3f"
          % (r["gcode"], r["lines"], r["penPulses"], r["scale"]))
    if r["clipped"]:
        print("  WARNING: %d point(s) clamped to the work area" % r["clipped"])
    if r["preview"]:
        print("  preview: %s" % r["preview"])


if __name__ == "__main__":
    Main()
