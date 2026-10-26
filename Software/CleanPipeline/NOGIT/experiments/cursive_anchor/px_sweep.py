"""Does the render resolution explain the writer-ID loss after normalisation?

Her real lines come out of the scanner at x-height ~42 px; synthesis at the
default pxPerMm=18 renders them at ~31 px. StrokeNormalize dilates by a
whole number of pixels, so the judge sees a coarser stroke for synthesis
than for real ink of the same nominal width. This sweeps the render
resolution only -- no code change, no synthesis change -- and reports what
the stroke-normalised shape judge makes of each.

    cd Software/CleanPipeline
    python NOGIT/experiments/cursive_anchor/px_sweep.py [author]
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
PX = [18.0, 22.0, 25.0, 30.0, 36.0]
N = 6

print("loading models...")
textModel = PaperCRNNNumpy(
    checkpoint_path=EV.CNN_DATA_DIR / "weights" / "paper_cnn_bilstm_ctc_joint_best.npz")
shapeModel, shapeMap, shapeI2A = EV.LoadShapeModel()
prof = SY.LoadAllProfiles()[AUTHOR]

base = IAMLineDatasetRaw(rootDir=str(SP.DATA_DIR), cacheDir=str(SP.CACHE_DIR))
AddPersonalSamples(base)
held = [s for s in base.samples
        if s["page_key"].split("/")[0] == AUTHOR and s["is_holdout"]]
held = sorted(held, key=lambda s: -len(s["text"]))[:N]


def xheight(pil):
    ink = PIO.BinarizeLine(np.array(pil.convert("L")))
    band = PIO.CoreBand(ink)
    return float(band[1] - band[0]) if band else float("nan")


def verdict(pil):
    """stroke-normalise exactly as the judge does, then classify"""
    norm = PIO.StrokeNormalize(np.array(pil.convert("L")))
    if norm is None:
        return None, None, None
    i, p = ClassifyImage(norm, shapeModel, apply_stroke_normalize=False)
    return shapeI2A[i], float(p[shapeMap[AUTHOR]]), norm


#reference: what her real ink scores
realP, realX = [], []
for s in held:
    real = DecodePng(s["image_png"]).convert("L")
    who, p, _ = verdict(real)
    realP.append((who == AUTHOR, p))
    realX.append(xheight(real))
print(f"\nREAL  x-height={np.mean(realX):.1f}px  "
      f"p({AUTHOR})={np.mean([p for _, p in realP])*100:.1f}%  "
      f"ID correct={np.mean([ok for ok, _ in realP])*100:.0f}%")

rows = []
for px in PX:
    ps, oks, xs, chars = [], [], [], []
    for s in held:
        text = s["text"]
        traj = SY.SynthesizeText(text, prof, seed=1, lineWidthMm=10_000.0)
        img = SY.RenderTrajectory(traj, pxPerMm=px, profile=prof)
        who, p, _ = verdict(img)
        if p is None:
            continue
        ps.append(p)
        oks.append(who == AUTHOR)
        xs.append(xheight(img))
        pred = ReadText(img, textModel)
        chars.append(CharAcc(pred.lower(), text.lower()))
    rows.append(dict(px=px, xh=np.mean(xs), p=np.mean(ps), ok=np.mean(oks),
                     char=np.mean(chars)))
    print(f"SYNTH pxPerMm={px:5.1f}  x-height={np.mean(xs):5.1f}px  "
          f"p({AUTHOR})={np.mean(ps)*100:5.1f}%  "
          f"ID correct={np.mean(oks)*100:5.0f}%  char={np.mean(chars)*100:5.1f}%")


def font(size, bold=False):
    try:
        return ImageFont.truetype(
            "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf", size)
    except Exception:
        return ImageFont.load_default()


#visual: the same line normalised at each resolution, against her real ink
text = held[0]["text"]
panels = [("REAL ink", DecodePng(held[0]["image_png"]).convert("L"), None)]
for px in PX:
    traj = SY.SynthesizeText(text, prof, seed=1, lineWidthMm=10_000.0)
    panels.append((f"pxPerMm={px:.0f}",
                   SY.RenderTrajectory(traj, pxPerMm=px, profile=prof), px))

MAXW = 1500
shots = []
for label, pil, px in panels:
    who, p, norm = verdict(pil)
    im = norm if norm is not None else pil
    im = im.convert("RGB")
    if im.width > MAXW:
        im = im.resize((MAXW, max(1, int(im.height * MAXW / im.width))))
    shots.append((label, im, who, p, xheight(pil)))

H = 56 + sum(im.height + 24 for _, im, _, _, _ in shots)
sheet = Image.new("RGB", (MAXW + 40, H), "white")
d = ImageDraw.Draw(sheet)
d.text((10, 8), f"{AUTHOR}: stroke-normalised view at each render resolution",
       fill=(0, 0, 0), font=font(15, True))
d.text((10, 30), f"truth: {text!r}", fill=(90, 90, 90), font=font(11))
y = 50
for label, im, who, p, xh in shots:
    ok = who == AUTHOR
    d.text((14, y), f"{label}   x-height={xh:.0f}px   ->  {who}  p={p*100:.1f}%",
           fill=(0, 120, 0) if ok else (180, 0, 0), font=font(11, True))
    sheet.paste(im, (20, y + 14))
    y += im.height + 24
out = OUT_DIR / f"px_sweep_{AUTHOR}.png"
sheet.save(out)
print("\nwrote", out)
