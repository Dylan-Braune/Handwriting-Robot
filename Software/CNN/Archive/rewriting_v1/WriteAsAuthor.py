"""
WriteAsAuthor.py -- END-TO-END entry point for the writing side.

    text string + author (1-10)
        -> style profile (built once by BuildStyleProfile.py, reused from disk)
        -> pen trajectory in that author's handwriting
        -> preview image + plotter preview
        -> G-code file + step/direction schedule
        -> simulation check (does the emitted G-code redraw the trajectory?)

Usage (interactive -- just run it):
    python WriteAsAuthor.py

Or scripted:
    python WriteAsAuthor.py "Hello world" 3
    python WriteAsAuthor.py "Hello world" 151        (author id also works)

All machine constants (steps/mm, bed size, feeds, pen pulse timing) live in
WriteGCode.GantryConfig -- see the header of that file.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import WriteGCode as GW
import SynthesizeHandwriting as SY

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "NOGIT" / "WriteJobs"


def ResolveAuthor(profiles, token):
    """Accepts a 1-10 index or a raw author folder id."""
    ids = sorted(profiles)
    token = str(token).strip()
    if token in profiles:
        return token
    if token.isdigit():
        n = int(token)
        if 1 <= n <= len(ids):
            return ids[n - 1]
    raise SystemExit(f"Unknown author {token!r}. Choose 1-{len(ids)} or one "
                     f"of: {', '.join(ids)}")


def Run(text, authorToken, mmPerXh=4.0, seed=None, jobName=None,
        cfg=None, outDir=None, legibilityTries=4):
    profiles = SY.LoadAllProfiles()
    if not profiles:
        raise SystemExit("No style profiles found -- run BuildStyleProfile.py first.")
    author = ResolveAuthor(profiles, authorToken)
    prof = profiles[author]
    cfg = cfg or GW.GantryConfig()
    outDir = Path(outDir or (OUT_DIR / (jobName or f"{author}")))
    outDir.mkdir(parents=True, exist_ok=True)

    # Best-of-N: draw the line a few times and keep the clearest reading.
    # Every candidate is the same author's library and the same measured
    # parameters -- only the per-letter variant choice and the jitter
    # differ -- so this buys legibility without spending style. Measured
    # over the novel-sentence set it improves BOTH: text 70.0% -> 74.5%
    # char, and author matching 81.7% -> 85.0%.
    traj = SY.SynthesizeLegible(
        text, prof, nTries=legibilityTries, mmPerXh=mmPerXh, seed=seed,
        lineWidthMm=cfg.boundsMaxXmm - cfg.originXmm - 5)
    down, up = traj.PenTravelMm()

    previewPath = outDir / "synth_preview.png"
    SY.RenderTrajectory(traj, pxPerMm=9.0, profile=prof).save(previewPath)

    gcodePath = outDir / "job.gcode"
    gRes = GW.WriteGcode(traj, cfg, gcodePath, title=f"author {author}")
    stepPath = outDir / "steps.csv"
    sRes = GW.WriteStepSchedule(traj, cfg, stepPath)
    plotPath = outDir / "plotter_preview.png"
    GW.RenderGcodePreview(gcodePath, cfg, path=plotPath)
    sim = GW.SimulateAndCompare(traj, cfg, gcodePath)

    print(f"\n=== Wrote '{text[:60]}{'...' if len(text) > 60 else ''}' "
          f"as author {author} ===")
    print(f"  glyph sources     : {traj.meta['glyphSources']}")
    print(f"  strokes           : {len(traj.strokes)} "
          f"(pen-down {down:.0f} mm, pen-up travel {up:.0f} mm)")
    print(f"  style             : slant {prof['slantDeg']:+.1f} deg, "
          f"connectedness {prof['connectedness']:.2f}, "
          f"x-height {mmPerXh:.1f} mm")
    print(f"  G-code            : {gcodePath}  "
          f"({gRes['lines']} lines, {gRes['penPulses']} pen pulses, "
          f"scale {gRes['scale']:.2f})")
    print(f"  step schedule     : {stepPath}  "
          f"({sRes['rows']} rows, {sRes['steps']} microsteps, "
          f"{sRes['seconds']:.1f} s estimated)")
    print(f"  previews          : {previewPath}\n"
          f"                      {plotPath}")
    print(f"  simulation check  : IoU {sim['iou']:.3f}, mean deviation "
          f"{sim['meanDevMm']:.3f} mm, max {sim['maxDevMm']:.3f} mm")
    return dict(author=author, traj=traj, gcode=gRes, steps=sRes, sim=sim,
                outDir=outDir)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        Run(sys.argv[1], sys.argv[2],
            mmPerXh=float(sys.argv[3]) if len(sys.argv) > 3 else 4.0)
    else:
        profiles = SY.LoadAllProfiles()
        ids = sorted(profiles)
        print("Authors:")
        for i, a in enumerate(ids, 1):
            p = profiles[a]
            print(f"  {i:2d}) {a}  slant {p['slantDeg']:+5.1f} deg, "
                  f"connectedness {p['connectedness']:.2f}, "
                  f"{len(p['glyphs'])} glyph classes")
        text = input("\nText to write: ").strip() or "Hello world"
        who = input(f"Author [1-{len(ids)} or id]: ").strip() or "1"
        xh = input("x-height in mm (blank = 4.0): ").strip()
        Run(text, who, mmPerXh=float(xh) if xh else 4.0)
