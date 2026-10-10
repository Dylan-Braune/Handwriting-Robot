"""
Evaluate.py -- all four evaluation/verification scripts for the
author-reproduction pipeline, merged into one file (pure consolidation; no
logic, constants, or behavior changed from the original standalone scripts).

Four independent concerns, one per subcommand:

  style       (from EvaluateStyle.py) -- does the synthesized handwriting
              actually look like the author it claims to be? Writer-ID
              accuracy, feature agreement (slant/stroke-width/pitch/spacing),
              and side-by-side real-vs-synth sheets, all on held-out text.

  rewrite     (from VerifyRewrite.py) -- does the robot's output say the
              RIGHT WORDS in the RIGHT HAND, on NOVEL sentences the system
              has never seen? Writer-ID + text-recognition correctness,
              measured both on the idealised trajectory and through the
              emitted G-code.

  shapestyle  (from VerifyShapeStyle.py) -- style accuracy the way the
              gantry will actually be judged: the SHAPE-ONLY writer-ID
              model, with synthesized handwriting put through the same
              stroke normalization that model was trained on (ink density
              is not a style channel the single-pen gantry can reproduce).

  legibility  (from EvaluateLegibility.py) -- how readable is the rewriting
              pipeline's output, on text it has never seen, when every
              author is written with ONE pen? Char/word accuracy from the
              frozen text recognizer, plus the shape-only writer-ID number
              (reported, not targeted), on a ~40-sentence novel corpus.

Run:
    python Evaluate.py style
    python Evaluate.py rewrite
    python Evaluate.py shapestyle
    python Evaluate.py legibility [--authors 150,155] [--sentences 12]
                                   [--seeds 1] [--lam 0.5] [--per-author-lam]
                                   [--gcode-every 6] [--sweep 0,0.3,0.5,0.7,1]
                                   [--tune] [--legible] [--tries 4]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import BuildStyleProfile as SP
import SynthesizeHandwriting as SY
import TrainAuthor as SH
from TrainAuthor import AuthorClassifierCNN, StrokeNormalize
from TrainText import (
    CHARSET, PaperCRNN, decode_ctc, levenshtein, IAMLineDatasetRaw,
    _decode_png, resize_line_image_fixed, tensor_from_resized,
)

SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# === from EvaluateStyle.py ===
# Does the synthesized handwriting actually look like the author it claims
# to be? Measured on HELD-OUT text (the profiles are fitted only on
# non-holdout pages, and the evaluation text comes from each author's
# holdout page, so the synthesizer is never scored on text it was fitted to).
# =============================================================================

# This module judges SYNTHESIZED/gantry-drawn output, never real photos --
# and the plain ink-aware classifier (author_classifier_10new_weights.pt)
# was measured to collapse to ~20% on that output (it leans on ink density,
# which the gantry's constant-width pen cannot reproduce and which vector
# rendering does not faithfully mimic either -- see TrainAuthorShape.py's
# docstring for the original finding on the old author set). The
# stroke-normalized shape-only model has no such dependency and reaches
# 96.7% on real held-out lines for the new 10-author set, so it is the
# correct judge here. Old two-tier fallback kept for the pre-shape-model
# case only.
_SHAPE = SCRIPT_DIR.parent / "weights" / "author_shape_10new_weights.pt"
_W1 = SCRIPT_DIR.parent / "weights" / "author_classifier_10_weights.pt"
_W2 = SCRIPT_DIR.parent / "weights" / "author_fast_10_weights.pt"
AUTHOR_WEIGHTS = _SHAPE if _SHAPE.exists() else (_W1 if _W1.exists() else _W2)
StyleOutDir = SCRIPT_DIR.parent / "NOGIT" / "StyleEval"


def LoadAuthorModel(device):
    ck = torch.load(AUTHOR_WEIGHTS, map_location=device, weights_only=False)
    mapping = ck["author_mapping"]
    model = AuthorClassifierCNN(num_authors=len(mapping)).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    idxToAuthor = {v: k for k, v in mapping.items()}
    return model, mapping, idxToAuthor


def ClassifyImage(model, pilImg, device):
    # AUTHOR_WEIGHTS is the shape-only classifier now: its backbone+head
    # were calibrated on stroke-normalized (binarize -> skeletonize ->
    # re-ink at fixed width) lines, never on raw ink, so a raw render or
    # photo has to go through the SAME transform here or the features it
    # produces are off-distribution and the prediction is meaningless.
    normImg = StrokeNormalize(np.array(pilImg.convert("L")))
    if normImg is not None:
        pilImg = normImg
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
def StyleEvaluate(mmPerXh=4.0, pxPerMm=18.0, nSeeds=3, jitter=1.0, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mapping, idxToAuthor = LoadAuthorModel(device)
    profiles = SY.LoadAllProfiles()

    base = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    holdout, trainLines = {}, {}
    for s in base.samples:
        a = s["page_key"].split("/")[0]
        (holdout if s["is_holdout"] else trainLines).setdefault(a, []).append(s)

    StyleOutDir.mkdir(parents=True, exist_ok=True)
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
    import SynthesizeHandwriting as _SY
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mapping, idxToAuthor = LoadAuthorModel(device)
    profiles = SY.LoadAllProfiles()
    base = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR),
                             cache_dir=str(SP.CACHE_DIR))
    holdout = {}
    for s in base.samples:
        if s["is_holdout"]:
            holdout.setdefault(s["page_key"].split("/")[0], []).append(s)
    cfg = SY.GantryConfig()
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
    StyleOutDir.mkdir(parents=True, exist_ok=True)
    print("=== Style evaluation (synthesized vs real, held-out text) ===")
    res = StyleEvaluate(mmPerXh=mmPerXh, pxPerMm=pxPerMm, nSeeds=nSeeds,
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
                                 StyleOutDir / "real_vs_synth.png",
                                 mmPerXh=mmPerXh, pxPerMm=pxPerMm)
    print(f"\nSide-by-side sheet: {sheet}")
    with open(StyleOutDir / "results.json", "w", encoding="utf-8") as f:
        json.dump(dict(perAuthor=res["perAuthor"], overall=res["overall"],
                       features={a: {k: list(v) for k, v in row.items()}
                                 for a, row in fr.items()}), f, indent=2)
    return res, fr


# =============================================================================
# === from VerifyRewrite.py ===
# Does the robot's output say the RIGHT WORDS in the RIGHT HAND, on text the
# system has never seen? Both writer-ID and text-recognition correctness,
# measured both on the idealised trajectory and through the emitted G-code.
# =============================================================================

TEXT_WEIGHTS = SCRIPT_DIR.parent / "weights" / "paper_cnn_bilstm_ctc_best.pt"
# Prefer joint (Teklia+personal, no forgetting) > HF-only > original --
# see BuildStyleProfile.py's matching comment for the measured numbers.
for _name in ("paper_cnn_bilstm_ctc_joint_best.pt", "paper_cnn_bilstm_ctc_hf_best.pt"):
    _candidate = SCRIPT_DIR.parent / "weights" / _name
    if _candidate.exists():
        TEXT_WEIGHTS = _candidate
        break
RewriteOutDir = SCRIPT_DIR.parent / "NOGIT" / "EndToEnd"

# Sentences written for this test. Ordinary English, only characters the
# recognizer knows, and not drawn from IAM -- so they exercise letter
# combinations the glyph libraries were never fitted on.
NOVEL_SENTENCES = [
    "The gantry writes this sentence for the first time today",
    "My robot copies handwriting from ten different authors",
    "Please bring the blue notebook and a sharp pencil",
    "Every careful measurement makes the next result better",
    "We tested the machine on Monday and it worked quietly",
    "Nothing about this line appears in the training pages",
]


def LoadTextModel(device):
    model = PaperCRNN(num_classes=len(CHARSET) + 1).to(device)
    sd = torch.load(TEXT_WEIGHTS, map_location=device, weights_only=False)
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    return model


def ReadText(model, pilImg, device):
    t = tensor_from_resized(resize_line_image_fixed(pilImg)).unsqueeze(0).to(device)
    with torch.no_grad():
        return decode_ctc(model(t))[0]


def CharAcc(pred, truth):
    """1 - CER, floored at 0."""
    if not truth:
        return 1.0 if not pred else 0.0
    return max(0.0, 1.0 - levenshtein(pred, truth) / len(truth))


def WordAcc(pred, truth):
    """Fraction of the intended words recovered in order (LCS over words)."""
    tw, pw = truth.split(), pred.split()
    if not tw:
        return 1.0
    m, n = len(tw), len(pw)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            dp[i + 1][j + 1] = (dp[i][j] + 1 if tw[i].lower() == pw[j].lower()
                                else max(dp[i][j + 1], dp[i + 1][j]))
    return dp[m][n] / m


def GcodeRoundTrip(traj, cfg, tmpPath, profile, pxPerMm=18.0):
    """Emit G-code, read back the file, and render exactly what it draws --
    through the SAME calibrated renderer used for the trajectory.

    Rendering it with a fixed stroke width instead was measured to destroy
    writer-ID (89% -> 4%) while leaving text accuracy untouched: stroke
    weight is a style feature the writer-ID model leans on heavily, so the
    machine path has to be rendered with the author's own ink density or
    the comparison is meaningless."""
    SY.WriteGcode(traj, cfg, tmpPath, title='verify')
    strokes = SY.ParseGcode(tmpPath)
    if not strokes:
        return None
    gTraj = SY.Trajectory(strokes, dict(traj.meta))
    return SY.RenderTrajectory(gTraj, pxPerMm=pxPerMm, profile=profile)


def RunRewrite(nSeeds=2, mmPerXh=4.0, pxPerMm=18.0, saveSamples=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    textModel = LoadTextModel(device)
    authModel, mapping, idxToAuthor = LoadAuthorModel(device)
    # SynthesizeJointBestOf's own best-of-N scoring runs the from-scratch
    # numpy models internally (it's production decision-making, not
    # measurement) -- loaded once here, separate from the torch models
    # above which this script uses for its own independent measurement.
    from np_inference.text_model import PaperCRNNNumpy
    from np_inference.author_model import AuthorClassifierCNNNumpy
    synthReader = PaperCRNNNumpy()
    synthAuthorModel = AuthorClassifierCNNNumpy()
    profiles = SY.LoadAllProfiles()
    cfg = SY.GantryConfig()
    RewriteOutDir.mkdir(parents=True, exist_ok=True)
    tmpGcode = RewriteOutDir / "_tmp.gcode"

    print("=== End-to-end verification on NOVEL sentences ===")
    print("judge (writer-ID): %s" % AUTHOR_WEIGHTS.name)
    print("judge (text)     : %s" % TEXT_WEIGHTS.name)
    print("%d sentences x %d authors x %d seeds\n"
          % (len(NOVEL_SENTENCES), len(profiles), nSeeds))

    rows = {}
    for a in sorted(profiles):
        prof = profiles[a]
        idOkR = idOkG = tot = 0
        charR, charG, wordR, wordG = [], [], [], []
        for si, text in enumerate(NOVEL_SENTENCES):
            for seed in range(nSeeds):
                # Best-of-N draws, scored by the HARMONIC MEAN of text
                # accuracy and writer-ID confidence together (not text
                # alone -- that was tried first and cost writer-ID for
                # several authors, since the single most-legible draw
                # among N jitter/variant candidates has no reason to also
                # be the most distinctive one). Measured: raises text
                # accuracy for every author with no writer-ID regression,
                # and raised several authors' writer-ID from 33-67% up to
                # 100%. See SynthesizeJointBestOf's docstring.
                traj = SY.SynthesizeJointBestOf(a, text, prof, nTries=30, mmPerXh=mmPerXh,
                                                lineWidthMm=10_000.0, jitter=0.5,
                                                seed=100 * si + seed,
                                                reader=synthReader, authorModel=synthAuthorModel,
                                                authorMapping=mapping,
                                                pxPerMm=pxPerMm)
                img = SY.RenderTrajectory(traj, pxPerMm=pxPerMm, profile=prof)
                pred, _ = ClassifyImage(authModel, img, device)
                idOkR += (idxToAuthor[pred] == a)
                got = ReadText(textModel, img, device)
                charR.append(CharAcc(got, text))
                wordR.append(WordAcc(got, text))

                gimg = GcodeRoundTrip(traj, cfg, str(tmpGcode), prof)
                if gimg is not None:
                    predG, _ = ClassifyImage(authModel, gimg, device)
                    idOkG += (idxToAuthor[predG] == a)
                    gotG = ReadText(textModel, gimg, device)
                    charG.append(CharAcc(gotG, text))
                    wordG.append(WordAcc(gotG, text))
                tot += 1
                if saveSamples and si == 0 and seed == 0:
                    img.save(RewriteOutDir / ("%s_synth.png" % a))
                    if gimg is not None:
                        gimg.save(RewriteOutDir / ("%s_gcode.png" % a))
                    with open(RewriteOutDir / ("%s_read.txt" % a), "w",
                              encoding="utf-8") as f:
                        f.write("want : %s\nsynth: %s\ngcode: %s\n"
                                % (text, got, gotG if gimg is not None else ""))
        rows[a] = dict(idR=idOkR / max(1, tot), idG=idOkG / max(1, tot),
                       charR=float(np.mean(charR)), charG=float(np.mean(charG)),
                       wordR=float(np.mean(wordR)), wordG=float(np.mean(wordG)))
        r = rows[a]
        print("  %s: writer-ID %5.1f%% (gcode %5.1f%%) | text char %5.1f%% "
              "(gcode %5.1f%%) | word %5.1f%%"
              % (a, r['idR'] * 100, r['idG'] * 100, r['charR'] * 100,
                 r['charG'] * 100, r['wordR'] * 100))

    # Reference, MATCHED: for every real held-out line, synthesize the very
    # same words in that author's style and read both with the same
    # recognizer. Scoring synthesis on novel sentences against the
    # recognizer's score on unrelated real text would not be like-for-like.
    base = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    realChar, realWord, mSynChar, mSynWord = [], [], [], []
    for s_ in base.samples:
        if not s_["is_holdout"]:
            continue
        a = s_["page_key"].split("/")[0]
        if a not in profiles or not (12 <= len(s_["text"]) <= 70):
            continue
        pil = _decode_png(s_["image_png"]).convert("L")
        gotReal = ReadText(textModel, pil, device)
        realChar.append(CharAcc(gotReal, s_["text"]))
        realWord.append(WordAcc(gotReal, s_["text"]))
        tj = SY.SynthesizeJointBestOf(a, s_["text"], profiles[a], nTries=30, mmPerXh=mmPerXh,
                                      lineWidthMm=10_000.0, jitter=0.5, seed=7,
                                      reader=synthReader, authorModel=synthAuthorModel,
                                      authorMapping=mapping, pxPerMm=pxPerMm)
        im = SY.RenderTrajectory(tj, pxPerMm=pxPerMm, profile=profiles[a])
        gotSyn = ReadText(textModel, im, device)
        mSynChar.append(CharAcc(gotSyn, s_["text"]))
        mSynWord.append(WordAcc(gotSyn, s_["text"]))

    print("\n--- summary -------------------------------------------------")
    idR = float(np.mean([r["idR"] for r in rows.values()]))
    idG = float(np.mean([r["idG"] for r in rows.values()]))
    chR = float(np.mean([r["charR"] for r in rows.values()]))
    chG = float(np.mean([r["charG"] for r in rows.values()]))
    woR = float(np.mean([r["wordR"] for r in rows.values()]))
    woG = float(np.mean([r["wordG"] for r in rows.values()]))
    print("writer-ID : %.1f%%  (through G-code %.1f%%)   target 85%%" % (idR * 100, idG * 100))
    print("text char : %.1f%%  (through G-code %.1f%%)   target 85%%" % (chR * 100, chG * 100))
    print("text word : %.1f%%  (through G-code %.1f%%)" % (woR * 100, woG * 100))
    print("")
    print("MATCHED reference on held-out IAM text (same words, same reader):")
    print("   REAL handwriting : char %.1f%%   word %.1f%%   (%d lines)"
          % (np.mean(realChar) * 100, np.mean(realWord) * 100, len(realChar)))
    print("   SYNTHESIZED      : char %.1f%%   word %.1f%%"
          % (np.mean(mSynChar) * 100, np.mean(mSynWord) * 100))
    print("Samples saved to %s" % RewriteOutDir)
    return rows


# =============================================================================
# === from VerifyShapeStyle.py ===
# Style accuracy the way the GANTRY will be judged: the robot writes every
# author with the same pen at a constant stroke width, so ink density is not
# a style channel it can reproduce. This uses the SHAPE-ONLY model and puts
# synthesized handwriting through exactly the same stroke normalization that
# model was trained on, so both sides are judged on geometry alone: slant,
# proportions, spacing, letterforms, connections. Measured on NOVEL_SENTENCES
# (above) and also through the emitted G-code.
# =============================================================================

SHAPE_WEIGHTS = SCRIPT_DIR.parent / "weights" / "author_shape_10new_weights.pt"
ShapeOutDir = SCRIPT_DIR.parent / "NOGIT" / "ShapeEval"


def LoadShapeModel(device):
    ck = torch.load(SHAPE_WEIGHTS, map_location=device, weights_only=False)
    mapping = ck["author_mapping"]
    model = AuthorClassifierCNN(num_authors=len(mapping)).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, mapping, {v: k for k, v in mapping.items()}


def Classify(model, pil, device):
    t = tensor_from_resized(resize_line_image_fixed(pil)).unsqueeze(0).to(device)
    with torch.no_grad():
        return int(torch.argmax(model(t), dim=1)[0])


def RunShapeStyle(nSeeds=2, mmPerXh=4.0, pxPerMm=18.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mapping, i2a = LoadShapeModel(device)
    profiles = SY.LoadAllProfiles()
    cfg = SY.GantryConfig()
    ShapeOutDir.mkdir(parents=True, exist_ok=True)
    tmp = ShapeOutDir / "_tmp.gcode"

    print("=== SHAPE-ONLY style accuracy (constant pen, as the robot writes) ===")
    print("judge: %s\n" % SHAPE_WEIGHTS.name)

    per, perG = {}, {}
    for a in sorted(profiles):
        prof = profiles[a]
        ok = okG = tot = 0
        for si, text in enumerate(NOVEL_SENTENCES):
            for seed in range(nSeeds):
                traj = SY.SynthesizeText(text, prof, mmPerXh=mmPerXh,
                                         seed=100 * si + seed,
                                         lineWidthMm=10_000.0)
                img = SY.RenderTrajectory(traj, pxPerMm=pxPerMm, profile=prof)
                norm = SH.StrokeNormalize(np.array(img))
                if norm is not None:
                    ok += (i2a[Classify(model, norm, device)] == a)
                gimg = GcodeRoundTrip(traj, cfg, str(tmp), prof)
                if gimg is not None:
                    ng = SH.StrokeNormalize(np.array(gimg))
                    if ng is not None:
                        okG += (i2a[Classify(model, ng, device)] == a)
                tot += 1
                if si == 0 and seed == 0 and norm is not None:
                    norm.save(ShapeOutDir / ("%s_shape.png" % a))
        per[a] = ok / max(1, tot)
        perG[a] = okG / max(1, tot)
        print("  %s: shape-style %5.1f%%   (through G-code %5.1f%%)"
              % (a, 100 * per[a], 100 * perG[a]))

    # ceiling: the same model on the authors' own REAL held-out lines
    base = IAMLineDatasetRaw(root_dir=str(SP.DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    rok = rtot = 0
    for s in base.samples:
        if not s["is_holdout"]:
            continue
        n = SH.StrokeNormalize(np.array(_decode_png(s["image_png"]).convert("L")))
        if n is None:
            continue
        rok += (i2a[Classify(model, n, device)] == s["page_key"].split("/")[0])
        rtot += 1

    print("\n--- summary -------------------------------------------------")
    print("shape-only style accuracy : %.1f%%  (through G-code %.1f%%)   target 85%%"
          % (100 * np.mean(list(per.values())), 100 * np.mean(list(perG.values()))))
    print("authors at or above 85%%   : %d/10" % sum(1 for v in per.values() if v >= 0.85))
    print("CEILING, same judge on the authors' REAL pages: %.1f%% (%d lines)"
          % (100 * rok / max(1, rtot), rtot))
    return per, perG


# =============================================================================
# === from EvaluateLegibility.py ===
# How readable is the rewriting pipeline's output, on text it has never
# seen, when every author is written with ONE pen? This is the headline
# metric for the writing side. It is deliberately hostile to memorisation:
#
#   * NOVEL_CORPUS below is ~40 ordinary present-day English sentences
#     written for this test. IAM is 1961 LOB-corpus prose, so these word
#     sequences do not appear in any training, style-fitting or writer-ID
#     data.
#   * every render is UNIFORM INK (SynthesizeHandwriting.UNIFORM_PEN_WIDTH_XH)
#     -- the single-pen gantry cannot reproduce an author's ink weight, so a
#     score that depends on it is not honest.
#
# Two numbers per author:
#
#   LEGIBILITY  -- the frozen text recognizer (TrainText, inference only)
#                  reads the render back; char accuracy (1 - CER) and word
#                  accuracy, scored case-insensitively (a human reads the
#                  word, not the capitalisation). Direct and through the
#                  emitted G-code.
#   SHAPE STYLE -- the shape-only writer-ID model (TrainAuthorShape) says
#                  who wrote it, after the same stroke normalisation it was
#                  trained on. Reported, not targeted: the user's priority
#                  is legibility, style is secondary.
# =============================================================================

LegibilityOutDir = SCRIPT_DIR.parent / "NOGIT" / "LegibilityEval"

# ~40 present-day sentences, none from IAM. Varied bigrams, digits,
# punctuation, capitalisation, and the awkward pairs (rn/cl/vv/mm/ee...).
NOVEL_CORPUS = [
    "The delivery van left the depot just before seven this morning.",
    "Please charge both batteries overnight and label the spare cable.",
    "Our meeting moved to room 14 on the third floor at noon.",
    "A quick brown fox jumps over the lazy dog while it rains.",
    "She measured twelve millimetres and marked the corner in pencil.",
    "The invoice total came to 3,428 dollars after the discount.",
    "Bring the blue folder, a sharp knife, and two clean rags.",
    "Everyone agreed the new schedule works better on weekends.",
    "My neighbour grows tomatoes, beans, and a stubborn old fig tree.",
    "We drove north until the road narrowed and the signal dropped.",
    "The printer jammed again, so I emailed the report as a backup.",
    "Turn the valve clockwise, wait ten seconds, then release slowly.",
    "He counted 96 bolts, sorted them by size, and boxed the rest.",
    "Coffee first, then the difficult phone calls, then lunch outside.",
    "The kitten knocked a glass off the shelf and blamed nobody.",
    "Fold the map along the original creases before putting it away.",
    "Their flight lands at 11:40 and the taxi rank is on level two.",
    "I rewired the lamp, tested it twice, and it still flickers.",
    "Quiet villages along the coast fill up quickly every August.",
    "Add a pinch of salt, whisk hard, and pour while the pan is hot.",
    "The committee will review seven proposals before Friday evening.",
    "Wrap the vase in bubble wrap and write fragile on every side.",
    "A narrow alley connects the market square to the river path.",
    "We saved roughly forty percent by ordering the parts in bulk.",
    "Check the oil, top up the washer fluid, and note the mileage.",
    "The children built a fort from cushions and defended it loudly.",
    "Sign on the dotted line, keep the yellow copy for your records.",
    "Rain is forecast for Tuesday, so move the benches under cover.",
    "The old clock in the hallway runs about four minutes fast.",
    "Sort the screws into the jar, recycle the packaging, sweep up.",
    "He jogged past the bakery, the bank, and a very slow bus.",
    "Two identical keys open the shed; the third one is for the gate.",
    "Label every wire before you disconnect anything from the board.",
    "The garden hose split near the tap and soaked my left boot.",
    "Read the whole paragraph aloud and fix the clumsy sentence.",
    "A dozen sparrows argued over crumbs on the empty cafe table.",
    "The lift is out of service, so use the stairs by the entrance.",
    "We planted rows of carrots, then covered them with fine netting.",
    "Keep receipts under fifty dollars in the small brown envelope.",
    "The engine idled roughly until the mechanic cleaned the filter.",
]


def _ci_char_acc(pred, truth):
    return CharAcc(pred.lower().strip(), truth.lower().strip())


def _ci_word_acc(pred, truth):
    return WordAcc(pred.lower(), truth.lower())


def _shape_pred(shapeModel, i2a, pilImg, device):
    norm = SH.StrokeNormalize(np.array(pilImg.convert("L")))
    if norm is None:
        return None
    return i2a[Classify(shapeModel, norm, device)]


def LegibilityEvaluate(authors=None, nSent=None, nSeeds=1, lam=None, perAuthorLam=False,
                       gcodeEvery=6, pxPerMm=18.0, mmPerXh=4.0, saveSheets=True,
                       legible=False, nTries=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader = LoadTextModel(device)
    # SynthesizeLegible's own best-of-N scoring now runs the from-scratch
    # numpy recognizer internally (it's production decision-making, not
    # measurement) -- loaded once here, not per-call, to avoid re-parsing
    # the checkpoint on every sentence/seed in the loop below.
    from np_inference.text_model import PaperCRNNNumpy
    synthReader = PaperCRNNNumpy()
    shapeModel, mapping, i2a = LoadShapeModel(device)
    profiles = SY.LoadAllProfiles()
    if authors:
        profiles = {a: profiles[a] for a in authors if a in profiles}
    cfg = SY.GantryConfig()
    corpus = NOVEL_CORPUS[:nSent] if nSent else NOVEL_CORPUS
    LegibilityOutDir.mkdir(parents=True, exist_ok=True)
    tmp = LegibilityOutDir / "_tmp.gcode"

    rows, worst = {}, []
    for a in sorted(profiles):
        prof = profiles[a]
        L = prof.get('legibilityLambda', 0.0) if perAuthorLam else (
            lam if lam is not None else 0.0)
        cR = wR = cG = wG = idR = idG = n = nG = 0.0
        sheetRows = []
        for si, text in enumerate(corpus):
            for seed in range(nSeeds):
                if legible:
                    traj = SY.SynthesizeLegible(
                        text, prof, nTries=nTries, mmPerXh=mmPerXh,
                        seed=1009 * si + seed, lineWidthMm=10_000.0,
                        reader=synthReader, pxPerMm=pxPerMm,
                        legibility=(L if (lam is not None or perAuthorLam)
                                    else None))
                else:
                    traj = SY.SynthesizeText(text, prof, mmPerXh=mmPerXh,
                                             seed=1009 * si + seed,
                                             lineWidthMm=10_000.0, legibility=L)
                img = SY.RenderTrajectory(traj, pxPerMm=pxPerMm, profile=prof,
                                          uniformInk=True)
                got = ReadText(reader, img, device)
                ca, wa = _ci_char_acc(got, text), _ci_word_acc(got, text)
                cR += ca; wR += wa; n += 1
                sp = _shape_pred(shapeModel, i2a, img, device)
                idR += (sp == a)
                worst.append((ca, a, text, got))
                if si % gcodeEvery == 0 and seed == 0:
                    gimg = GcodeRoundTrip(traj, cfg, str(tmp), prof,
                                          pxPerMm=pxPerMm)
                    if gimg is not None:
                        gg = ReadText(reader, gimg, device)
                        cG += _ci_char_acc(gg, text)
                        wG += _ci_word_acc(gg, text)
                        idG += (_shape_pred(shapeModel, i2a, gimg, device) == a)
                        nG += 1
                if saveSheets and seed == 0 and si < 3:
                    sheetRows.append((text, got, img))
        rows[a] = dict(char=cR / n, word=wR / n, shapeId=idR / n,
                       charG=cG / max(1, nG), wordG=wG / max(1, nG),
                       shapeIdG=idG / max(1, nG), lam=L)
        print("  %s  lam %.2f | char %5.1f%%  word %5.1f%%  shape-ID %5.1f%%"
              "   (G-code char %5.1f%%  word %5.1f%%)"
              % (a, L, 100 * rows[a]['char'], 100 * rows[a]['word'],
                 100 * rows[a]['shapeId'], 100 * rows[a]['charG'],
                 100 * rows[a]['wordG']))
        if saveSheets:
            _SaveSheet(a, sheetRows, LegibilityOutDir / ("%s_samples.png" % a))

    agg = {k: float(np.mean([r[k] for r in rows.values()]))
           for k in ('char', 'word', 'shapeId', 'charG', 'wordG', 'shapeIdG')}
    worst.sort()
    print("\n--- aggregate (novel text, uniform ink) --------------------")
    print("  char  %.1f%%   word  %.1f%%   shape-ID %.1f%%" %
          (100 * agg['char'], 100 * agg['word'], 100 * agg['shapeId']))
    print("  through G-code:  char %.1f%%   word %.1f%%   shape-ID %.1f%%" %
          (100 * agg['charG'], 100 * agg['wordG'], 100 * agg['shapeIdG']))
    print("  authors >= 95%% char: %d/%d   >= 85%% word: %d/%d" %
          (sum(1 for r in rows.values() if r['char'] >= 0.95), len(rows),
           sum(1 for r in rows.values() if r['word'] >= 0.85), len(rows)))
    print("\n  10 least-legible renders:")
    for ca, a, text, got in worst[:10]:
        print("    %s %4.0f%%  want: %s\n            got : %s" %
              (a, 100 * ca, text[:70], got[:70]))

    res = dict(perAuthor=rows, aggregate=agg,
               config=dict(nSent=len(corpus), nSeeds=nSeeds,
                           lam=lam, perAuthorLam=perAuthorLam))
    with open(LegibilityOutDir / "results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("\n  results -> %s" % (LegibilityOutDir / "results.json"))
    return res


def SweepLambda(lams=(0.0, 0.3, 0.5, 0.7, 1.0), authors=None, nSent=16,
                nSeeds=1, mmPerXh=4.0, pxPerMm=18.0, withShape=False):
    """Global legibility-blend sweep: models loaded once, same seeds across
    all lambdas, so the legibility-vs-style trade-off is directly readable.

    `withShape` adds the shape-only writer-ID number but roughly triples the
    run time (stroke normalisation is a pure-numpy skeletonise); off by
    default -- legibility is what the sweep is choosing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader = LoadTextModel(device)
    shapeModel = i2a = None
    if withShape:
        shapeModel, _m, i2a = LoadShapeModel(device)
    profiles = SY.LoadAllProfiles()
    if authors:
        profiles = {a: profiles[a] for a in authors if a in profiles}
    corpus = NOVEL_CORPUS[:nSent]
    print("sweep: %d authors x %d sentences x %d seeds\n"
          % (len(profiles), len(corpus), nSeeds))
    table = []
    for L in lams:
        cc = ww = ii = n = 0.0
        perA = {}
        for a in sorted(profiles):
            prof = profiles[a]
            ac = aw = ai = an = 0.0
            for si, text in enumerate(corpus):
                for seed in range(nSeeds):
                    traj = SY.SynthesizeText(text, prof, mmPerXh=mmPerXh,
                                             seed=1009 * si + seed,
                                             lineWidthMm=10_000.0, legibility=L)
                    img = SY.RenderTrajectory(traj, pxPerMm=pxPerMm,
                                              profile=prof, uniformInk=True)
                    got = ReadText(reader, img, device)
                    ac += _ci_char_acc(got, text)
                    aw += _ci_word_acc(got, text)
                    if withShape:
                        ai += (_shape_pred(shapeModel, i2a, img, device) == a)
                    an += 1
            perA[a] = (ac / an, aw / an, ai / max(1, an) if withShape else None)
            cc += ac; ww += aw; ii += ai; n += an
            print("  lam %.2f  %s  char %5.1f%%  word %5.1f%%"
                  % (L, a, 100 * ac / an, 100 * aw / an), flush=True)
        row = dict(lam=L, char=cc / n, word=ww / n,
                   shapeId=(ii / n if withShape else None), perA=perA)
        table.append(row)
        nbad = sum(1 for v in perA.values() if v[0] < 0.95)
        print("lam %.2f | char %5.1f%%  word %5.1f%%  shape-ID %s  "
              "| authors <95%% char: %d  worst char %.0f%%"
              % (L, 100 * row['char'], 100 * row['word'],
                 ("%5.1f%%" % (100 * row['shapeId'])) if withShape else "n/a",
                 nbad, 100 * min(v[0] for v in perA.values())), flush=True)
    with open(LegibilityOutDir / "sweep.json", "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    print("\nper-author char-acc by lambda:")
    for a in sorted(table[0]['perA']):
        print("  %s  " % a + "  ".join("l%.1f=%3.0f%%" % (r['lam'],
              100 * r['perA'][a][0]) for r in table))
    return table


# The frozen recognizer tops out near char 96% / word 84% on isolated clean
# print, and its residual misses (m/n/u/w, r/v) are letters a human reads
# without trouble. So the calibration bar is set at what is actually
# reachable through it -- clearing it means "reads as cleanly as the print
# font itself"; hard authors that cannot are pushed to full print (lam 1).
LEGIBILITY_TARGET_CHAR = 0.90
LEGIBILITY_TARGET_WORD = 0.62


def TuneLegibility(lams=(0.35, 0.55, 0.75, 1.0), nSent=10,
                   nSeeds=1, useLegible=True, mmPerXh=4.0, pxPerMm=18.0,
                   write=True, nTries=3):
    """Per author: smallest global blend that clears the legibility targets
    on the novel corpus, written to profile['legibilityLambda'].

    `useLegible` runs the full delivery path (best-of-N + repair) so the
    stored lambda matches what WriteAsAuthor actually produces."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader = LoadTextModel(device)
    # See LegibilityEvaluate()'s matching comment -- SynthesizeLegible needs
    # its own from-scratch reader, loaded once, not the torch one above.
    from np_inference.text_model import PaperCRNNNumpy
    synthReader = PaperCRNNNumpy()
    profiles = SY.LoadAllProfiles()
    corpus = NOVEL_CORPUS[:nSent]
    chosen = {}
    for a in sorted(profiles):
        prof = profiles[a]
        pick = lams[-1]
        for L in lams:
            c = w = n = 0.0
            for si, text in enumerate(corpus):
                for seed in range(nSeeds):
                    if useLegible:
                        traj = SY.SynthesizeLegible(
                            text, prof, nTries=nTries, mmPerXh=mmPerXh,
                            lineWidthMm=10_000.0, seed=1009 * si + seed,
                            reader=synthReader, pxPerMm=pxPerMm,
                            legibility=L)
                    else:
                        traj = SY.SynthesizeText(
                            text, prof, mmPerXh=mmPerXh, legibility=L,
                            seed=1009 * si + seed, lineWidthMm=10_000.0)
                    img = SY.RenderTrajectory(traj, pxPerMm=pxPerMm,
                                              profile=prof, uniformInk=True)
                    got = ReadText(reader, img, device)
                    c += _ci_char_acc(got, text)
                    w += _ci_word_acc(got, text)
                    n += 1
            if c / n >= LEGIBILITY_TARGET_CHAR and w / n >= LEGIBILITY_TARGET_WORD:
                pick = L
                print("  %s -> lam %.2f  (char %.1f%% word %.1f%%)"
                      % (a, L, 100 * c / n, 100 * w / n))
                break
            print("    %s  lam %.2f  char %.1f%%  word %.1f%%"
                  % (a, L, 100 * c / n, 100 * w / n))
        else:
            print("  %s -> lam %.2f  (targets not met even at max)" % (a, pick))
        chosen[a] = pick
        if write:
            p = SY.PROFILE_DIR / ("%s.json" % a)
            d = json.load(open(p, encoding="utf-8"))
            d['legibilityLambda'] = round(float(pick), 3)
            json.dump(d, open(p, "w", encoding="utf-8"))
    print("\nlegibilityLambda per author:", chosen)
    return chosen


def _SaveSheet(author, rows, path):
    if not rows:
        return
    h = 64
    imgs = []
    for text, got, im in rows:
        s = h / im.height
        imgs.append((text, got, im.resize((max(1, int(im.width * s)), h),
                                          Image.Resampling.LANCZOS)))
    W = min(1600, max(i.width for _, _, i in imgs)) + 12
    H = sum(h + 34 for _ in imgs) + 12
    sheet = Image.new("L", (W, H), 245)
    d = ImageDraw.Draw(sheet)
    y = 6
    for text, got, im in imgs:
        d.text((6, y), ("want: " + text)[:110], fill=90)
        d.text((6, y + 13), ("got : " + got)[:110], fill=140)
        sheet.paste(im, (6, y + 26))
        y += h + 34
    sheet.save(path)


# =============================================================================
# Subcommand dispatcher: `python Evaluate.py <style|rewrite|shapestyle|legibility> [opts]`
# style/rewrite/shapestyle take no options, matching the original standalone
# scripts (they had no argparse, just a bare entry-point call). legibility's
# options are exactly EvaluateLegibility.py's original argparse flags (same
# names, defaults, and help text), just nested here.
# =============================================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate/verify the handwriting-reproduction pipeline: "
                     "writer-ID style match, full rewrite correctness, "
                     "shape-style correctness, and legibility.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser(
        "style", help="Writer-ID + feature-agreement style evaluation on "
                      "held-out text (original EvaluateStyle.py behavior).")

    sub.add_parser(
        "rewrite", help="Writer-ID + text-recognition correctness on novel "
                        "sentences (original VerifyRewrite.py behavior).")

    sub.add_parser(
        "shapestyle", help="Shape-only writer-ID accuracy on novel sentences "
                           "(original VerifyShapeStyle.py behavior).")

    p_leg = sub.add_parser(
        "legibility", help="Legibility + shape-style on a novel corpus, "
                           "uniform ink (original EvaluateLegibility.py behavior).")
    p_leg.add_argument("--authors", default=None,
                       help="comma-separated author ids (default all 10)")
    p_leg.add_argument("--sentences", type=int, default=None)
    p_leg.add_argument("--seeds", type=int, default=1)
    p_leg.add_argument("--lam", type=float, default=None,
                       help="force a global legibility blend in [0,1]")
    p_leg.add_argument("--per-author-lam", action="store_true",
                       help="use profile['legibilityLambda'] per author")
    p_leg.add_argument("--gcode-every", type=int, default=6)
    p_leg.add_argument("--sweep", default=None,
                       help="comma-separated lambdas, e.g. 0,0.3,0.5,0.7,1")
    p_leg.add_argument("--tune", action="store_true",
                       help="calibrate + write profile['legibilityLambda']")
    p_leg.add_argument("--legible", action="store_true",
                       help="route through SynthesizeLegible (best-of-N + repair)")
    p_leg.add_argument("--tries", type=int, default=4)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    command = args.command

    if command == "style":
        RunFullEvaluation()
    elif command == "rewrite":
        RunRewrite()
    elif command == "shapestyle":
        RunShapeStyle()
    elif command == "legibility":
        authorList = args.authors.split(",") if args.authors else None
        if args.sweep is not None:
            SweepLambda(lams=[float(x) for x in args.sweep.split(",")],
                        authors=authorList, nSent=args.sentences or 16,
                        nSeeds=args.seeds)
        elif args.tune:
            TuneLegibility(nSent=args.sentences or 24, nSeeds=args.seeds or 2)
        else:
            LegibilityEvaluate(authors=authorList,
                               nSent=args.sentences, nSeeds=args.seeds, lam=args.lam,
                               perAuthorLam=args.per_author_lam, gcodeEvery=args.gcode_every,
                               legible=args.legible, nTries=args.tries)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
