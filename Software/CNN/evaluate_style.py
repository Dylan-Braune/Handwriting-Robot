"""
evaluate_style.py -- does the synthesized handwriting actually look like the
author it claims to be?

Three independent measurements, all on HELD-OUT text (the profiles are
fitted only on non-holdout pages, and the evaluation text comes from each
author's holdout page, so the synthesizer is never scored on text it was
fitted to):

  1. WRITER-ID ACCURACY -- render each synthesized line the same way the
     training crops look (black ink, white paper, 64x640 aspect-preserved)
     and ask the 10-author classifier who wrote it. Target: >= 85% of
     author X's synthesized lines classified as X.
  2. FEATURE AGREEMENT -- slant, x-height-normalized stroke width, letter
     pitch and word spacing measured on the synthesized renders vs on the
     author's real lines, reported as a per-feature relative error.
  3. SIDE-BY-SIDE SHEETS -- real line above, synthesized same-text line
     below, per author, for visual judgement.

Run:
    python evaluate_style.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import style_profile as SP
import synthesize_handwriting as SY
from train_author_classifier import AuthorClassifierCNN
from train_paper_cnn_bilstm_ctc import (
    IAMLineDatasetRaw, _decode_png, resize_line_image_fixed,
    tensor_from_resized,
)

SCRIPT_DIR = Path(__file__).resolve().parent
# prefer the fully fine-tuned model when it exists, otherwise the fast
# frozen-backbone one -- both are saved in the same format
_W1 = SCRIPT_DIR / "NOGIT" / "weights" / "author_classifier_10_weights.pt"
_W2 = SCRIPT_DIR / "NOGIT" / "weights" / "author_fast_10_weights.pt"
AUTHOR_WEIGHTS = _W1 if _W1.exists() else _W2
OUT_DIR = SCRIPT_DIR / "NOGIT" / "StyleEval"


def LoadAuthorModel(device):
    ck = torch.load(AUTHOR_WEIGHTS, map_location=device, weights_only=False)
    mapping = ck["author_mapping"]
    model = AuthorClassifierCNN(num_authors=len(mapping)).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    idxToAuthor = {v: k for k, v in mapping.items()}
    return model, mapping, idxToAuthor


def ClassifyImage(model, pilImg, device):
    t = tensor_from_resized(resize_line_image_fixed(pilImg)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(t)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return int(np.argmax(probs)), probs


# ---------------------------------------------------------------------------
# Feature measurement on a rendered/real line image
# ---------------------------------------------------------------------------
def MeasureLine(grayArr):
    ink = SP.BinarizeLine(grayArr)
    if ink.sum() < 40:
        return None
    band = SP.CoreBand(ink)
    if band is None:
        return None
    top, base = band
    xh = float(base - top)
    if xh < 4:
        return None
    slant = SP.EstimateSlantDeg(ink)
    # word spacing: gaps in the column profile wider than 0.5 xh
    col = ink.any(axis=0)
    gaps, run = [], 0
    for v in col:
        if v:
            if run:
                gaps.append(run)
            run = 0
        else:
            run += 1
    wordGaps = [g / xh for g in gaps if g / xh > 0.5]
    inkFrac = float(ink.sum()) / max(1.0, xh * ink.shape[1])
    return dict(slant=slant, xh=xh,
                wordGap=float(np.median(wordGaps)) if wordGaps else np.nan,
                inkPerXh=inkFrac)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def Evaluate(mmPerXh=4.0, pxPerMm=18.0, nSeeds=3, jitter=1.0, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mapping, idxToAuthor = LoadAuthorModel(device)
    profiles = SY.LoadAllProfiles()

    base = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    holdout, trainLines = {}, {}
    for s in base.samples:
        a = s["page_key"].split("/")[0]
        (holdout if s["is_holdout"] else trainLines).setdefault(a, []).append(s)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    perAuthor, allRows = {}, []
    featRows = []

    for a in sorted(profiles):
        prof = profiles[a]
        lines = holdout.get(a) or trainLines.get(a, [])
        texts = [s["text"] for s in lines if len(s["text"]) >= 12][:12]
        if not texts:
            continue
        correct = tot = 0
        for ti, text in enumerate(texts):
            for seed in range(nSeeds):
                traj = SY.SynthesizeText(text, prof, mmPerXh=mmPerXh,
                                         seed=1000 * ti + seed, jitter=jitter,
                                         lineWidthMm=10_000.0)
                img = SY.RenderTrajectory(traj, pxPerMm=pxPerMm, profile=prof)
                pred, probs = ClassifyImage(model, img, device)
                ok = idxToAuthor[pred] == a
                correct += ok
                tot += 1
                allRows.append((a, idxToAuthor[pred], float(probs[mapping[a]])))
                if seed == 0:
                    m = MeasureLine(np.array(img))
                    if m:
                        featRows.append((a, "synth", m))
        perAuthor[a] = correct / max(1, tot)
        # real-line features for the same author
        for s in lines[:12]:
            g = np.array(_decode_png(s["image_png"]).convert("L"))
            m = MeasureLine(g)
            if m:
                featRows.append((a, "real", m))
        if verbose:
            print(f"  {a}: writer-ID {perAuthor[a] * 100:5.1f}%  ({correct}/{tot})")

    overall = float(np.mean(list(perAuthor.values()))) if perAuthor else 0.0
    return dict(perAuthor=perAuthor, overall=overall, rows=allRows,
                features=featRows)


def FeatureReport(featRows):
    byA = {}
    for a, kind, m in featRows:
        byA.setdefault(a, {"real": [], "synth": []})[kind].append(m)
    out = {}
    for a, d in sorted(byA.items()):
        if not d["real"] or not d["synth"]:
            continue
        row = {}
        for key in ("slant", "wordGap", "inkPerXh"):
            r = np.nanmedian([m[key] for m in d["real"]])
            s = np.nanmedian([m[key] for m in d["synth"]])
            row[key] = (float(r), float(s))
        out[a] = row
    return out


def BuildComparisonSheet(profiles, holdoutTexts, path, mmPerXh=4.0,
                         pxPerMm=18.0, maxW=1500):
    """Real line (top) vs synthesized same text (bottom), per author."""
    rows = []
    for a in sorted(profiles):
        item = holdoutTexts.get(a)
        if item is None:
            continue
        realImg, text = item
        traj = SY.SynthesizeText(text, profiles[a], mmPerXh=mmPerXh, seed=7,
                                 lineWidthMm=10_000.0)
        synImg = SY.RenderTrajectory(traj, pxPerMm=pxPerMm, profile=profiles[a])
        h = 62
        def fit(im):
            s = h / im.height
            w = max(1, int(im.width * s))
            im = im.resize((w, h), Image.Resampling.LANCZOS)
            if im.width > maxW:
                im = im.crop((0, 0, maxW, h))
            return im
        rows.append((a, text, fit(realImg.convert("L")), fit(synImg)))

    if not rows:
        return None
    lab = 108
    W = lab + min(maxW, max(max(r[2].width, r[3].width) for r in rows)) + 16
    H = sum(2 * 62 + 34 for _ in rows) + 16
    sheet = Image.new("L", (W, H), 245)
    d = ImageDraw.Draw(sheet)
    y = 8
    for a, text, realIm, synIm in rows:
        d.text((6, y + 22), f"{a}", fill=0)
        d.text((6, y + 40), "real", fill=90)
        d.text((6, y + 100), "synth", fill=90)
        sheet.paste(realIm, (lab, y + 16))
        sheet.paste(synIm, (lab, y + 16 + 62))
        d.text((lab, y + 2), text[:78], fill=110)
        d.line([(0, y + 140), (W, y + 140)], fill=200)
        y += 2 * 62 + 34
    sheet.save(path)
    return path


def MachineDistort(traj, cfg, rng, backlashMm=0.15, jitterMm=0.05):
    """Apply the errors a real gantry adds on top of a perfect trajectory:
      * microstep quantization (X/Y land on 1/stepsPerMm grid),
      * belt backlash -- a constant lost-motion offset applied whenever the
        direction of travel on an axis reverses,
      * small random positioning noise per move.
    Returns a NEW trajectory; re-running writer-ID on it measures the
    digital -> physical style loss directly."""
    import synthesize_handwriting as _SY
    qx, qy = 1.0 / cfg.stepsPerMmX, 1.0 / cfg.stepsPerMmY
    out = []
    for s in traj.strokes:
        q = []
        lastDx = lastDy = 0.0
        offx = offy = 0.0
        prev = None
        for (x, y) in s:
            if prev is not None:
                dx, dy = x - prev[0], y - prev[1]
                if dx * lastDx < 0:
                    offx += backlashMm * (1 if dx > 0 else -1)
                if dy * lastDy < 0:
                    offy += backlashMm * (1 if dy > 0 else -1)
                if abs(dx) > 1e-9:
                    lastDx = dx
                if abs(dy) > 1e-9:
                    lastDy = dy
            X = round((x + offx + rng.gauss(0, jitterMm)) / qx) * qx
            Y = round((y + offy + rng.gauss(0, jitterMm)) / qy) * qy
            q.append((X, Y))
            prev = (x, y)
        out.append(q)
    return _SY.Trajectory(out, dict(traj.meta))


def EvaluatePhysical(mmPerXh=4.0, pxPerMm=18.0, nSeeds=2,
                     backlashMm=0.15, jitterMm=0.05):
    """Writer-ID accuracy on machine-distorted trajectories -- the estimate
    of what survives once the gantry actually draws it."""
    import random
    import gcode_writer as GW
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mapping, idxToAuthor = LoadAuthorModel(device)
    profiles = SY.LoadAllProfiles()
    base = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR),
                             cache_dir=str(SP.CACHE_DIR))
    holdout = {}
    for s in base.samples:
        if s["is_holdout"]:
            holdout.setdefault(s["page_key"].split("/")[0], []).append(s)
    cfg = GW.GantryConfig()
    rng = random.Random(12345)
    per = {}
    for a in sorted(profiles):
        texts = [s["text"] for s in holdout.get(a, []) if len(s["text"]) >= 12][:8]
        if not texts:
            continue
        ok = tot = 0
        for ti, text in enumerate(texts):
            for seed in range(nSeeds):
                traj = SY.SynthesizeText(text, profiles[a], mmPerXh=mmPerXh,
                                         seed=1000 * ti + seed,
                                         lineWidthMm=10_000.0)
                dist = MachineDistort(traj, cfg, rng, backlashMm, jitterMm)
                img = SY.RenderTrajectory(dist, pxPerMm=pxPerMm,
                                          profile=profiles[a])
                pred, _ = ClassifyImage(model, img, device)
                ok += idxToAuthor[pred] == a
                tot += 1
        per[a] = ok / max(1, tot)
    overall = float(np.mean(list(per.values()))) if per else 0.0
    return dict(perAuthor=per, overall=overall,
                backlashMm=backlashMm, jitterMm=jitterMm)


def CollectHoldoutSamples(maxPerAuthor=1):
    """One real held-out line image + its text per author, for the
    side-by-side sheet (same text, real vs synthesized)."""
    base = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR),
                             cache_dir=str(SP.CACHE_DIR))
    out, seen = {}, {}
    for s in base.samples:
        a = s["page_key"].split("/")[0]
        if not s["is_holdout"] or seen.get(a, 0) >= maxPerAuthor:
            continue
        if not (18 <= len(s["text"]) <= 62):
            continue
        out[a] = (_decode_png(s["image_png"]), s["text"])
        seen[a] = seen.get(a, 0) + 1
    return out


def RunFullEvaluation(mmPerXh=4.0, pxPerMm=18.0, nSeeds=3, jitter=1.0):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Style evaluation (synthesized vs real, held-out text) ===")
    res = Evaluate(mmPerXh=mmPerXh, pxPerMm=pxPerMm, nSeeds=nSeeds,
                   jitter=jitter)
    print(f"\nOverall writer-ID accuracy on synthesized held-out text: "
          f"{res['overall'] * 100:.1f}%   (target 85%)")
    nPass = sum(1 for v in res["perAuthor"].values() if v >= 0.85)
    print(f"Authors at or above 85%: {nPass}/{len(res['perAuthor'])}")

    fr = FeatureReport(res["features"])
    print("\nFeature agreement (real -> synth):")
    for a, row in fr.items():
        print(f"  {a}: " + "  ".join(
            f"{k} {v[0]:.2f}->{v[1]:.2f}" for k, v in row.items()))

    profiles = SY.LoadAllProfiles()
    samples = CollectHoldoutSamples()
    sheet = BuildComparisonSheet(profiles, samples,
                                 OUT_DIR / "real_vs_synth.png",
                                 mmPerXh=mmPerXh, pxPerMm=pxPerMm)
    print(f"\nSide-by-side sheet: {sheet}")
    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(dict(perAuthor=res["perAuthor"], overall=res["overall"],
                       features={a: {k: list(v) for k, v in row.items()}
                                 for a, row in fr.items()}), f, indent=2)
    return res, fr


if __name__ == "__main__":
    RunFullEvaluation()
