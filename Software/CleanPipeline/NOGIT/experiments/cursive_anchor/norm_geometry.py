"""Is StrokeNormalize actually normalising? Measure, don't eyeball.

For yeukita's real held-out lines and her synthesised lines of the same
text, report the geometry the judge ends up looking at: image height,
measured x-height, the pen width StrokeNormalize derives from it, and the
mean ink thickness after normalisation (ink pixels / skeleton pixels).

    cd Software/CleanPipeline
    python NOGIT/experiments/cursive_anchor/norm_geometry.py [author]
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline")
import numpy as np

import SynthesizeHandwriting as SY
import ProfileIO as PIO
import BuildStyleProfile as SP
from TrainText import IAMLineDatasetRaw, DecodePng
from TrainAuthor import AddPersonalSamples

AUTHOR = sys.argv[1] if len(sys.argv) > 1 else "yeukita"
N = 6


def geom(pil, tag):
    gray = np.array(pil.convert("L"))
    ink = PIO.BinarizeLine(gray)
    band = PIO.CoreBand(ink)
    xh = float(band[1] - band[0]) if band else float("nan")
    w = int(np.clip(round(PIO.PEN_WIDTH_XH * (xh if xh == xh and xh >= 4 else
                                              max(8.0, ink.shape[0] / 3.0))), 2, 9))
    sk = PIO.Skeletonize(ink)
    thickRaw = ink.sum() / max(1, sk.sum())
    norm = PIO.StrokeNormalize(gray)
    if norm is None:
        return None
    nink = PIO.BinarizeLine(np.array(norm))
    nsk = PIO.Skeletonize(nink)
    thickNorm = nink.sum() / max(1, nsk.sum())
    return dict(tag=tag, h=gray.shape[0], w_img=gray.shape[1], xh=xh, penW=w,
                thickRaw=thickRaw, thickNorm=thickNorm,
                inkFrac=ink.mean(), inkFracNorm=nink.mean())


base = IAMLineDatasetRaw(rootDir=str(SP.DATA_DIR), cacheDir=str(SP.CACHE_DIR))
AddPersonalSamples(base)
held = [s for s in base.samples
        if s["page_key"].split("/")[0] == AUTHOR and s["is_holdout"]]
held = sorted(held, key=lambda s: -len(s["text"]))[:N]
prof = SY.LoadAllProfiles()[AUTHOR]

hdr = (f"{'line':26s} {'src':6s} {'imgH':>5s} {'imgW':>6s} {'x-height':>9s} "
       f"{'penW':>5s} {'thick raw':>10s} {'thick norm':>11s} {'inkFrac':>8s}")
print(hdr)
print("-" * len(hdr))
acc = {"real": [], "synth": []}
for s in held:
    text = s["text"]
    real = DecodePng(s["image_png"]).convert("L")
    traj = SY.SynthesizeText(text, prof, seed=1, lineWidthMm=10_000.0)
    synth = SY.RenderTrajectory(traj, profile=prof)
    for tag, pil in (("real", real), ("synth", synth)):
        g = geom(pil, tag)
        if g is None:
            continue
        acc[tag].append(g)
        print(f"{s['page_key'][:26]:26s} {tag:6s} {g['h']:5d} {g['w_img']:6d} "
              f"{g['xh']:9.1f} {g['penW']:5d} {g['thickRaw']:10.2f} "
              f"{g['thickNorm']:11.2f} {g['inkFrac']:8.3f}")

print()
for tag in ("real", "synth"):
    v = acc[tag]
    if not v:
        continue
    print(f"{tag:6s} mean: x-height={np.mean([g['xh'] for g in v]):6.1f}  "
          f"penW={np.mean([g['penW'] for g in v]):4.1f}  "
          f"thick raw={np.mean([g['thickRaw'] for g in v]):5.2f}  "
          f"thick normalised={np.mean([g['thickNorm'] for g in v]):5.2f}  "
          f"inkFrac={np.mean([g['inkFrac'] for g in v]):.3f} -> "
          f"{np.mean([g['inkFracNorm'] for g in v]):.3f}")

print(f"\nprofile says: strokeWidthXh={prof.get('strokeWidthXh')}, "
      f"xHeightPx={prof.get('xHeightPx')}, inkFracRef={prof.get('inkFracRef')}")
print(f"PEN_WIDTH_XH={PIO.PEN_WIDTH_XH}, and StrokeNormalize clamps the "
      f"dilation width to [2,9] px")
