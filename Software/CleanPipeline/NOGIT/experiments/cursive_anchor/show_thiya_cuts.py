import sys
sys.path.insert(0, r"C:\Users\braun\OneDrive\Documents\GitHub\Handwriting-Robot\Software\CleanPipeline")
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import BuildStyleProfile as SP
from Classify import PaperCRNNNumpy
from ProfileIO import CtcForcedAlign, BinarizeLine, CoreBand, EstimateSlantDeg
from TrainText import IAMLineDatasetRaw, DecodePng, FrameXToPixel, INPUT_WIDTH
from TrainAuthor import AddPersonalSamples
from BuildStyleProfile import TensorFromResized, ResizeLineImageFixed


def font(size, bold=False):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf", size)
    except Exception:
        return ImageFont.load_default()


def compute_cuts(gray, text, model):
    rawW = gray.shape[1]
    pil = Image.fromarray(gray)
    t = TensorFromResized(ResizeLineImageFixed(pil))
    lp = model.forward(t)[:, 0, :]
    res = CtcForcedAlign(lp, text)
    if res is None:
        return None
    align, conf = res
    T = lp.shape[0]

    ink = BinarizeLine(gray)
    band = CoreBand(ink)
    if band is None:
        return None
    top, base_y = band
    xh = float(base_y - top)

    colProf = ink.sum(axis=0).astype(np.float64)
    k = max(3, int(round(xh * 0.12))) | 1
    colSm = np.convolve(colProf, np.ones(k) / k, mode='same')
    rowsIdx = np.arange(ink.shape[0])[:, None]
    hiY = np.where(ink, rowsIdx, ink.shape[0]).min(axis=0).astype(np.float64)
    topAbove = np.clip((base_y - hiY) / max(1.0, xh), 0.0, 3.0)
    thin = colSm / max(1e-6, float(np.percentile(colSm[colSm > 0], 60)) if (colSm > 0).any() else 1.0)
    cutCost = thin + 1.15 * topAbove
    cutCost[colProf == 0] = 0.0
    snapR = max(3, int(round(xh * 0.5)))

    def _Snap(x):
        a = int(np.clip(round(x) - snapR, 0, rawW - 1))
        b = int(np.clip(round(x) + snapR + 1, 1, rawW))
        if b - a < 2:
            return float(x)
        win = cutCost[a:b].copy()
        off = np.abs(np.arange(a, b) - x) / max(1.0, snapR)
        return float(a + int(np.argmin(win + 0.6 * off)))

    origH = gray.shape[0]
    charXs = []
    for ch, sp in align:
        if sp is None:
            charXs.append((ch, None, None))
            continue
        x0 = FrameXToPixel(sp[0] / T * INPUT_WIDTH, rawW, origH)
        x1 = FrameXToPixel(sp[1] / T * INPUT_WIDTH, rawW, origH)
        charXs.append((ch, x0, x1))
    valid = [(i, c) for i, c in enumerate(charXs) if c[1] is not None]
    if not valid:
        return None
    centres = [0.5 * (c[1] + c[2]) for _, c in valid]

    nV = len(valid)
    minCell = max(1.0, 0.08 * xh)
    bnd = [0.0] * (nV + 1)
    bnd[0] = max(0.0, centres[0] - (0.5 * (centres[1] - centres[0]) if nV > 1 else xh))
    bnd[nV] = min(float(rawW), centres[-1] + (0.5 * (centres[-1] - centres[-2]) if nV > 1 else xh))
    for k2 in range(1, nV):
        s = _Snap(0.5 * (centres[k2 - 1] + centres[k2]))
        lo = bnd[k2 - 1] + minCell
        hi = max(lo, bnd[nV] - (nV - k2) * minCell)
        bnd[k2] = min(max(s, lo), hi)

    cuts = []
    for k2 in range(nV):
        i, (ch, x0, x1) = valid[k2]
        cuts.append((ch, bnd[k2], bnd[k2 + 1], x0, x1))
    return cuts, nV, len(text)


def render_line(gray, text, cuts_info, scale=3):
    cuts, nV, nText = cuts_info
    origH, rawW = gray.shape
    img = Image.fromarray(gray).convert("RGB").resize((rawW * scale, origH * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    f = font(13, bold=True)
    for ch, b0, b1, x0, x1 in cuts:
        draw.line([(b0 * scale, 0), (b0 * scale, origH * scale)], fill=(220, 0, 0), width=2)
        for yy in range(0, origH * scale, 8):
            draw.line([(x0 * scale, yy), (x0 * scale, yy + 4)], fill=(0, 80, 220), width=1)
            draw.line([(x1 * scale, yy), (x1 * scale, yy + 4)], fill=(0, 160, 220), width=1)
        cx = 0.5 * (b0 + b1) * scale
        draw.text((cx - 4, origH * scale - 14), ch, fill=(0, 120, 0), font=f)
    draw.line([(cuts[-1][2] * scale, 0), (cuts[-1][2] * scale, origH * scale)], fill=(220, 0, 0), width=2)
    return img


# --- get thiya's first N non-holdout lines, in page order ---
N = 10
base = IAMLineDatasetRaw(rootDir=str(SP.DATA_DIR), cacheDir=str(SP.CACHE_DIR))
AddPersonalSamples(base)
base.samples = [s for s in base.samples if s["page_key"].split("/")[0] == "thiya" and not s["is_holdout"]]
base.samples.sort(key=lambda s: s["page_key"])
picked = base.samples[:N]

model = PaperCRNNNumpy(checkpoint_path=SP.PERSONAL_WEIGHTS)

rows = []
label_f = font(11)
for s in picked:
    gray = np.array(DecodePng(s["image_png"]).convert("L"))
    text = s["text"]
    info = compute_cuts(gray, text, model)
    if info is None:
        print("skip (no alignment):", repr(text))
        continue
    cuts, nV, nText = info
    img = render_line(gray, text, info, scale=3)
    rows.append((img, text, nV, nText))
    print(f"{s['page_key']}: {nV}/{nText} aligned  text={text!r}")

maxW = max(im.width for im, *_ in rows)
row_h = max(im.height for im, *_ in rows) + 30
H = row_h * len(rows)
sheet = Image.new("RGB", (maxW, H), "white")
draw = ImageDraw.Draw(sheet)
y = 0
for im, text, nV, nText in rows:
    sheet.paste(im, (0, y))
    draw.text((4, y + im.height + 2), f"{text!r}  ({nV}/{nText} aligned)", fill=(0, 0, 0), font=label_f)
    y += row_h
    draw.line([(0, y - 2), (maxW, y - 2)], fill=(200, 200, 200))

legend_h = 40
canvas = Image.new("RGB", (sheet.width, sheet.height + legend_h), "white")
canvas.paste(sheet, (0, 0))
ld = ImageDraw.Draw(canvas)
ld.text((10, sheet.height + 4), "red solid = final snapped glyph-cell boundary (what BuildStyleProfile cuts on)", fill=(220, 0, 0), font=label_f)
ld.text((10, sheet.height + 20), "blue dashes = raw CTC-aligned span before snapping (x0 dark / x1 light)", fill=(0, 80, 220), font=label_f)

out = str(Path(__file__).resolve().parent / "thiya_cuts_10lines.png")
canvas.save(out)
print("wrote", out, canvas.size)
