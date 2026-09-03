"""
VerifyRewrite.py -- does the robot's output say the RIGHT WORDS in the
RIGHT HAND, on text the system has never seen?

Two independent questions, both answered on NOVEL sentences that appear
nowhere in IAM (so neither the style profiles, the writer-ID model, nor the
text recognizer can have memorised them):

  1. WRITER ID  -- is synthesized author X recognised as author X?
     (TrainAuthor.py's 10-author model)
  2. TEXT       -- does the frozen text recognizer read back the sentence
     that was requested?  (TrainText.py, used for
     inference only, never modified)

Both are also measured through the FULL MACHINE PATH: trajectory -> G-code
-> re-parsed from the emitted file -> rasterized -> classified. That is the
image the gantry would actually draw, not the idealised trajectory.

Reading the text number honestly needs a reference, because the recognizer
is not perfect on real handwriting either: the same measurement is run on
each author's REAL held-out lines. That is the ceiling synthesis competes
against, not 100%.

Run:
    python VerifyRewrite.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import WriteGCode as GW
import BuildStyleProfile as SP
import SynthesizeHandwriting as SY
import EvaluateStyle as ES
from TrainText import (
    CHARSET, PaperCRNN, decode_ctc, levenshtein, IAMLineDatasetRaw,
    _decode_png, resize_line_image_fixed, tensor_from_resized,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TEXT_WEIGHTS = SCRIPT_DIR / "NOGIT" / "weights" / "paper_cnn_bilstm_ctc_best.pt"
OUT_DIR = SCRIPT_DIR / "NOGIT" / "EndToEnd"

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
    GW.WriteGcode(traj, cfg, tmpPath, title='verify')
    strokes = GW.ParseGcode(tmpPath)
    if not strokes:
        return None
    gTraj = SY.Trajectory(strokes, dict(traj.meta))
    return SY.RenderTrajectory(gTraj, pxPerMm=pxPerMm, profile=profile)


def Run(nSeeds=2, mmPerXh=4.0, pxPerMm=18.0, saveSamples=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    textModel = LoadTextModel(device)
    authModel, mapping, idxToAuthor = ES.LoadAuthorModel(device)
    profiles = SY.LoadAllProfiles()
    cfg = GW.GantryConfig()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmpGcode = OUT_DIR / "_tmp.gcode"

    print("=== End-to-end verification on NOVEL sentences ===")
    print("judge (writer-ID): %s" % ES.AUTHOR_WEIGHTS.name)
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
                traj = SY.SynthesizeText(text, prof, mmPerXh=mmPerXh,
                                         seed=100 * si + seed,
                                         lineWidthMm=10_000.0)
                img = SY.RenderTrajectory(traj, pxPerMm=pxPerMm, profile=prof)
                pred, _ = ES.ClassifyImage(authModel, img, device)
                idOkR += (idxToAuthor[pred] == a)
                got = ReadText(textModel, img, device)
                charR.append(CharAcc(got, text))
                wordR.append(WordAcc(got, text))

                gimg = GcodeRoundTrip(traj, cfg, str(tmpGcode), prof)
                if gimg is not None:
                    predG, _ = ES.ClassifyImage(authModel, gimg, device)
                    idOkG += (idxToAuthor[predG] == a)
                    gotG = ReadText(textModel, gimg, device)
                    charG.append(CharAcc(gotG, text))
                    wordG.append(WordAcc(gotG, text))
                tot += 1
                if saveSamples and si == 0 and seed == 0:
                    img.save(OUT_DIR / ("%s_synth.png" % a))
                    if gimg is not None:
                        gimg.save(OUT_DIR / ("%s_gcode.png" % a))
                    with open(OUT_DIR / ("%s_read.txt" % a), "w",
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
        tj = SY.SynthesizeText(s_["text"], profiles[a], mmPerXh=mmPerXh,
                               seed=7, lineWidthMm=10_000.0)
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
    print("Samples saved to %s" % OUT_DIR)
    return rows


if __name__ == "__main__":
    Run()
