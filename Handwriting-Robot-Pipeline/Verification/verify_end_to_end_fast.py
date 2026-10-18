"""Stage 5 (full-coverage path): end-to-end writer-ID + text accuracy for ALL
13 authors, sized to actually finish on a CPU-only machine.

WHY THIS EXISTS ALONGSIDE Evaluate.py rewrite
---------------------------------------------
Evaluate.py rewrite is the official measurement, and it is running unmodified
in this session. Its cost, though, is enormous on CPU: 13 authors x 6 novel
sentences x 2 seeds = 156 lines, each synthesized as a best-of-30 search, PLUS
an 87-line matched reference where every line is again a best-of-30 search.
That is 780 + 87 x 30 synthesis attempts, and on this machine it runs at
roughly 80 minutes per author -- on the order of 16 hours in total.

This script measures the SAME things, on the SAME code path
(SynthesizeJointBestOf, the production method; RenderTrajectory; the same
G-code round trip), for ALL 13 authors, but with a scope a CPU can finish:
2 novel sentences x 1 seed x best-of-8 per author = 26 scored lines.

The tradeoff is stated plainly rather than hidden: fewer samples per author
means each per-author percentage moves in 50-point steps and is correspondingly
noisier. Treat this as the complete-coverage picture and Evaluate.py rewrite as
the high-precision one.

Run from Software/CNN:
    python NOGIT/full_pipeline_run/verify_end_to_end_fast.py
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# path setup for the clean package layout
import sys as _sys
from pathlib import Path as _p
_root = _p(__file__).resolve().parent.parent
_cnn = _root / 'Software' / 'CNN'
_ar = _cnn / 'AuthorReproductionStuff'
_gantry = _root / 'Software' / 'GantryControl'
for _q in (_cnn, _ar, _ar / 'np_inference', _gantry):
    if str(_q) not in _sys.path:
        _sys.path.insert(0, str(_q))


THIS_DIR = Path(__file__).resolve().parent
CNN_DIR = _cnn

import SynthesizeHandwriting as SY  # noqa: E402
from Evaluate import (  # noqa: E402
    AUTHOR_WEIGHTS, TEXT_WEIGHTS, ClassifyImage, GcodeRoundTrip, LoadAuthorModel,
    LoadTextModel, ReadText, CharAcc, WordAcc,
)
from np_inference.text_model import PaperCRNNNumpy  # noqa: E402
from np_inference.author_model import AuthorClassifierCNNNumpy  # noqa: E402


SENTENCES = [
    "The gantry writes this sentence for the first time today",
    "My robot copies handwriting from ten different authors",
    "Please bring the blue notebook and a sharp pencil",
]
N_TRIES = 8
MM_PER_XH = 4.0
PX_PER_MM = 18.0
OUT_IMG = CNN_DIR / "NOGIT" / "full_pipeline_run" / "e2e_fast"
T0 = time.time()


def say(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def main():
    device = torch.device("cpu")
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    say(f"judge (writer-ID): {AUTHOR_WEIGHTS.name}")
    say(f"judge (text)     : {TEXT_WEIGHTS.name}")

    textModel = LoadTextModel(device)
    authModel, mapping, idxToAuthor = LoadAuthorModel(device)
    say("torch judges loaded")
    synthReader = PaperCRNNNumpy()
    synthAuthor = AuthorClassifierCNNNumpy()
    say("numpy synthesis scorers loaded")

    profiles = SY.LoadAllProfiles()
    cfg = SY.GantryConfig()
    say(f"profiles: {len(profiles)} -> {sorted(profiles)}")
    tmpGcode = OUT_IMG / "_tmp_fast.gcode"

    rows = []
    per_author = {}
    for ai, a in enumerate(sorted(profiles), 1):
        prof = profiles[a]
        idR = idG = tot = 0
        charR, charG, wordR, wordG = [], [], [], []
        for si, text in enumerate(SENTENCES):
            t0 = time.time()
            traj = SY.SynthesizeJointBestOf(
                a, text, prof, nTries=N_TRIES, mmPerXh=MM_PER_XH,
                lineWidthMm=10_000.0, jitter=0.5, seed=si,
                reader=synthReader, authorModel=synthAuthor,
                authorMapping=mapping, pxPerMm=PX_PER_MM)
            img = SY.RenderTrajectory(traj, pxPerMm=PX_PER_MM, profile=prof)
            pred, probs = ClassifyImage(authModel, img, device)
            okR = int(idxToAuthor[pred] == a)
            got = ReadText(textModel, img, device)
            caR, waR = CharAcc(got, text), WordAcc(got, text)
            idR += okR
            charR.append(caR)
            wordR.append(waR)

            gimg = GcodeRoundTrip(traj, cfg, str(tmpGcode), prof)
            okG, gotG, caG, waG = None, None, None, None
            if gimg is not None:
                predG, _ = ClassifyImage(authModel, gimg, device)
                okG = int(idxToAuthor[predG] == a)
                gotG = ReadText(textModel, gimg, device)
                caG, waG = CharAcc(gotG, text), WordAcc(gotG, text)
                idG += okG
                charG.append(caG)
                wordG.append(waG)
            tot += 1
            if si == 0:
                img.save(OUT_IMG / f"{a}_synth.png")
                if gimg is not None:
                    gimg.save(OUT_IMG / f"{a}_gcode.png")
                with open(OUT_IMG / f"{a}_read.txt", "w", encoding="utf-8") as f:
                    f.write(f"want : {text}\nsynth: {got}\ngcode: {gotG}\n")
            rows.append(dict(author=a, sentence=text, writer_id_render=okR,
                             writer_id_gcode=okG, char_render=caR, char_gcode=caG,
                             word_render=waR, word_gcode=waG,
                             read_render=got, read_gcode=gotG,
                             confidence=float(probs.max())))
            say(f"  {a} s{si+1}/{len(SENTENCES)}: ID {'OK' if okR else 'MISS'} "
                f"char {caR*100:.0f}% | gcode ID {'OK' if okG else 'MISS'} "
                f"char {'' if caG is None else f'{caG*100:.0f}%'} | {time.time()-t0:.0f}s")
        per_author[a] = dict(
            idR=idR / max(1, tot), idG=idG / max(1, tot), n=tot,
            charR=float(np.mean(charR)), charG=float(np.mean(charG)) if charG else None,
            wordR=float(np.mean(wordR)), wordG=float(np.mean(wordG)) if wordG else None)
        r = per_author[a]
        say(f"== {a} ({ai}/{len(profiles)}): writer-ID {r['idR']*100:.0f}% "
            f"(gcode {r['idG']*100:.0f}%) | char {r['charR']*100:.0f}% | word {r['wordR']*100:.0f}%")

    print()
    print("=" * 92, flush=True)
    print("FULL-COVERAGE END-TO-END (13 authors, 2 novel sentences each, best-of-8)", flush=True)
    print("=" * 92, flush=True)
    print(f"{'author':<10} {'writerID':>9} {'wID gcode':>10} {'char':>8} {'char gcode':>11} "
          f"{'word':>8}", flush=True)
    for a in sorted(per_author):
        r = per_author[a]
        print(f"{a:<10} {r['idR']*100:8.0f}% {r['idG']*100:9.0f}% {r['charR']*100:7.0f}% "
              f"{(r['charG'] or 0)*100:10.0f}% {r['wordR']*100:7.0f}%", flush=True)
    idR = float(np.mean([r["idR"] for r in per_author.values()]))
    idG = float(np.mean([r["idG"] for r in per_author.values()]))
    chR = float(np.mean([r["charR"] for r in per_author.values()]))
    chG = float(np.mean([r["charG"] for r in per_author.values() if r["charG"] is not None]))
    woR = float(np.mean([r["wordR"] for r in per_author.values()]))
    print("-" * 92, flush=True)
    print(f"{'MEAN':<10} {idR*100:8.0f}% {idG*100:9.0f}% {chR*100:7.0f}% {chG*100:10.0f}% "
          f"{woR*100:7.0f}%", flush=True)
    print(f"\ntargets: writer-ID 85%, char 85%", flush=True)

    with open(THIS_DIR / "stage5_fast_summary.json", "w", encoding="utf-8") as f:
        json.dump(dict(scope=dict(sentences=len(SENTENCES), tries=N_TRIES,
                                  authors=len(per_author)),
                       per_author=per_author,
                       mean=dict(idR=idR, idG=idG, charR=chR, charG=chG, wordR=woR),
                       sentences=SENTENCES,
                       judge_id=AUTHOR_WEIGHTS.name, judge_text=TEXT_WEIGHTS.name,
                       elapsed_seconds=round(time.time() - T0, 1)), f, indent=2)
    with open(THIS_DIR / "stage5_fast_per_line.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    say("wrote stage5_fast_summary.json + stage5_fast_per_line.csv")
    say("DONE")


if __name__ == "__main__":
    main()
