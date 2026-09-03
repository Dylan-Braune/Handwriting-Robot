"""
verify_shape_style.py -- style accuracy the way the GANTRY will be judged.

The robot writes every author with the same pen at a constant stroke width,
so ink density is not a style channel it can reproduce. Measuring against a
writer-ID model that leans on ink weight therefore flatters the result: the
model in train_author_classifier.py scores 100% on real pages but only
11-16% on those same pages once every author is re-inked at one width
(chance is 10%) -- it reads the pen, not the hand.

This script uses the SHAPE-ONLY model (train_author_shape.py), and puts
synthesized handwriting through exactly the same stroke normalization that
model was trained on, so both sides are judged on geometry alone: slant,
proportions, spacing, letterforms, connections.

Everything is measured on NOVEL sentences (see verify_end_to_end.py) that
appear nowhere in IAM, and also through the emitted G-code.

Run:
    python verify_shape_style.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gcode_writer as GW
import style_profile as SP
import synthesize_handwriting as SY
import train_author_shape as SH
import verify_end_to_end as V
from train_author_classifier import AuthorClassifierCNN
from train_paper_cnn_bilstm_ctc import (
    IAMLineDatasetRaw, _decode_png, resize_line_image_fixed,
    tensor_from_resized,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SHAPE_WEIGHTS = SCRIPT_DIR / "NOGIT" / "weights" / "author_shape_10_weights.pt"
OUT_DIR = SCRIPT_DIR / "NOGIT" / "ShapeEval"


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


def Run(nSeeds=2, mmPerXh=4.0, pxPerMm=18.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mapping, i2a = LoadShapeModel(device)
    profiles = SY.LoadAllProfiles()
    cfg = GW.GantryConfig()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / "_tmp.gcode"

    print("=== SHAPE-ONLY style accuracy (constant pen, as the robot writes) ===")
    print("judge: %s\n" % SHAPE_WEIGHTS.name)

    per, perG = {}, {}
    for a in sorted(profiles):
        prof = profiles[a]
        ok = okG = tot = 0
        for si, text in enumerate(V.NOVEL_SENTENCES):
            for seed in range(nSeeds):
                traj = SY.SynthesizeText(text, prof, mmPerXh=mmPerXh,
                                         seed=100 * si + seed,
                                         lineWidthMm=10_000.0)
                img = SY.RenderTrajectory(traj, pxPerMm=pxPerMm, profile=prof)
                norm = SH.StrokeNormalize(np.array(img))
                if norm is not None:
                    ok += (i2a[Classify(model, norm, device)] == a)
                gimg = V.GcodeRoundTrip(traj, cfg, str(tmp), prof)
                if gimg is not None:
                    ng = SH.StrokeNormalize(np.array(gimg))
                    if ng is not None:
                        okG += (i2a[Classify(model, ng, device)] == a)
                tot += 1
                if si == 0 and seed == 0 and norm is not None:
                    norm.save(OUT_DIR / ("%s_shape.png" % a))
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


if __name__ == "__main__":
    Run()
