"""What the writer-ID judge actually SEES for yeukita: real vs synthesised.

The judge for a reproduced author is the stroke-normalised shape classifier
(the robot writes with one constant pen, so ink density is not a style
channel it can reproduce). This script therefore shows, side by side, the
StrokeNormalize() output for

  * her real held-out lines, and
  * her synthesised lines of the same text,

each with the shape classifier's verdict on the NORMALISED image, so a
style loss can be separated from a normalisation artefact.

    cd Software/CleanPipeline
    python NOGIT/experiments/cursive_anchor/yeukita_norm_check.py [author]
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline")
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import SynthesizeHandwriting as SY
import Evaluate as EV
import ProfileIO as PIO
import BuildStyleProfile as SP
from Classify import PaperCRNNNumpy, ClassifyImage, ReadText, CharAcc
from TrainText import IAMLineDatasetRaw, DecodePng
from TrainAuthor import AddPersonalSamples

OUT_DIR = Path(__file__).resolve().parent / "new_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
AUTHOR = sys.argv[1] if len(sys.argv) > 1 else "yeukita"
N_LINES = 6

print("loading models...")
textModel = PaperCRNNNumpy(
    checkpoint_path=EV.CNN_DATA_DIR / "weights" / "paper_cnn_bilstm_ctc_joint_best.npz")
shapeModel, shapeMap, shapeI2A = EV.LoadShapeModel()
print("  shape judge:", shapeModel.checkpoint_path.name)
prof = SY.LoadAllProfiles()[AUTHOR]

base = IAMLineDatasetRaw(rootDir=str(SP.DATA_DIR), cacheDir=str(SP.CACHE_DIR))
AddPersonalSamples(base)
held = [s for s in base.samples
        if s["page_key"].split("/")[0] == AUTHOR and s["is_holdout"]]
held.sort(key=lambda s: s["page_key"])
#longest lines first: a 2-word line is too little ink for a style verdict
held = sorted(held, key=lambda s: -len(s["text"]))[:N_LINES]


def judge(pil):
    """classify BOTH ways, and also hand back what the judge sees"""
    norm = PIO.StrokeNormalize(np.array(pil.convert("L")))
    rawIdx, rawP = ClassifyImage(pil, shapeModel, apply_stroke_normalize=False)
    if norm is None:
        return None, (shapeI2A[rawIdx], float(rawP[shapeMap[AUTHOR]])), None
    nIdx, nP = ClassifyImage(norm, shapeModel, apply_stroke_normalize=False)
    return norm, (shapeI2A[rawIdx], float(rawP[shapeMap[AUTHOR]])), \
        (shapeI2A[nIdx], float(nP[shapeMap[AUTHOR]]))


rows = []
for s in held:
    text = s["text"]
    real = DecodePng(s["image_png"]).convert("L")
    rNorm, rRaw, rN = judge(real)

    traj = SY.SynthesizeText(text, prof, seed=1, lineWidthMm=10_000.0)
    synth = SY.RenderTrajectory(traj, profile=prof)
    sNorm, sRaw, sN = judge(synth)

    pred = ReadText(synth, textModel)
    rows.append(dict(text=text, real=real, rNorm=rNorm, rRaw=rRaw, rN=rN,
                     synth=synth, sNorm=sNorm, sRaw=sRaw, sN=sN,
                     charAcc=CharAcc(pred.lower(), text.lower()), pred=pred,
                     lam=traj.meta.get("legibilityLambda")))
    print(f"\n{s['page_key']}  {text!r}")
    print(f"  REAL   raw: {rRaw[0]:9s} p={rRaw[1]*100:5.1f}%   "
          f"normalised: {rN[0]:9s} p={rN[1]*100:5.1f}%")
    print(f"  SYNTH  raw: {sRaw[0]:9s} p={sRaw[1]*100:5.1f}%   "
          f"normalised: {sN[0]:9s} p={sN[1]*100:5.1f}%   char={rows[-1]['charAcc']*100:.1f}%")

print()
for tag, k in (("REAL  raw       ", "rRaw"), ("REAL  normalised", "rN"),
               ("SYNTH raw       ", "sRaw"), ("SYNTH normalised", "sN")):
    sub = [r[k] for r in rows if r[k]]
    print(f"{tag}: p({AUTHOR})={np.mean([p for _, p in sub])*100:5.1f}%   "
          f"ID correct={np.mean([a == AUTHOR for a, _ in sub])*100:5.1f}%")


def font(size, bold=False):
    try:
        return ImageFont.truetype(
            "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf", size)
    except Exception:
        return ImageFont.load_default()


#sheet: for each line, 4 strips -- real, real normalised, synth, synth normalised
PANELS = [("real", "real", "rRaw"), ("real, stroke-normalised", "rNorm", "rN"),
          ("synthesised", "synth", "sRaw"), ("synthesised, stroke-normalised", "sNorm", "sN")]
MAXW = 1500


def fit(im):
    if im.width <= MAXW:
        return im.convert("RGB")
    h = max(1, int(im.height * MAXW / im.width))
    return im.convert("RGB").resize((MAXW, h))


blocks = []
for r in rows:
    strips = []
    for label, imk, sk in PANELS:
        im = r[imk]
        if im is None:
            continue
        strips.append((label, fit(im), r[sk]))
    blocks.append((r, strips))

W = MAXW + 40
H = 70 + sum(40 + sum(im.height + 26 for _, im, _ in st) for _, st in blocks)
sheet = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(sheet)
d.text((10, 8), f"{AUTHOR}: what the stroke-normalised shape judge sees "
                f"({shapeModel.checkpoint_path.name})", fill=(0, 0, 0), font=font(15, True))
d.text((10, 30), f"synthesis at the new default lam={rows[0]['lam']}; "
                 f"p = classifier confidence that this line is {AUTHOR}",
       fill=(90, 90, 90), font=font(11))
y = 52
for r, strips in blocks:
    d.text((10, y), f"truth: {r['text']!r}", fill=(0, 0, 0), font=font(12, True))
    y += 20
    for label, im, (who, p) in strips:
        ok = who == AUTHOR
        d.text((14, y), f"{label}   ->  {who}  p={p*100:.1f}%",
               fill=(0, 120, 0) if ok else (180, 0, 0), font=font(11, True))
        sheet.paste(im, (20, y + 14))
        y += im.height + 26
    y += 14
    d.line([(0, y - 8), (W, y - 8)], fill=(225, 225, 225))
out = OUT_DIR / f"norm_check_{AUTHOR}.png"
sheet.save(out)
print("\nwrote", out)
