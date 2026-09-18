"""
build_pw_comparison.py -- one sheet per author: anchor vs style-transfer at
pos_weight=1.0 vs pos_weight=0.5, all on the SAME held-out sentence for that
author, using the plain thin-line renderer (clearest for eyeballing, not the
classifier-matched renderer EvaluateStyleTransferRNN.py uses for scoring).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import SynthesizeHandwriting as SY
from StyleTransferRNN import (
    PAIR_CACHE_DIR, WEIGHTS_DIR, N_POINTS, MM_PER_XH,
    StyleTransferRNN, trajectory_to_xy, _resample_arclength, render_xy,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "NOGIT" / "StyleTransferEval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PW1 = WEIGHTS_DIR / "styletransfer_best_pw1.0.pt"
CKPT_PW05 = WEIGHTS_DIR / "styletransfer_best_pw0.5.pt"


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    writers = ck["writers"]
    model = StyleTransferRNN(len(writers), hidden=ck["hidden"], layers=ck["layers"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, writers


@torch.no_grad()
def render_for(model, writers, prof, writer_id, text, device):
    traj = SY.SynthesizeText(text, prof, mmPerXh=MM_PER_XH, seed=0,
                             lineWidthMm=100_000.0, legibility=1.0)
    anchor_xy, anchor_eos = trajectory_to_xy(traj, MM_PER_XH)
    anchor_xy = anchor_xy - anchor_xy[0]
    a_rs, a_eos_rs = _resample_arclength(anchor_xy, N_POINTS, eos=anchor_eos)
    anchor_d = np.diff(a_rs, axis=0, prepend=a_rs[:1]).astype(np.float32)
    w_idx = torch.tensor([writers.index(writer_id)], device=device)
    pred_d = model(torch.from_numpy(anchor_d).unsqueeze(0).to(device), w_idx)
    pred_xy = np.cumsum(pred_d[0].cpu().numpy(), axis=0)
    return a_rs, a_eos_rs, pred_xy


def main():
    device = torch.device("cpu")
    model_pw1, writers1 = load_model(CKPT_PW1, device)
    model_pw05, writers05 = load_model(CKPT_PW05, device)
    profiles = SY.LoadAllProfiles()

    writers = sorted(p.stem for p in PAIR_CACHE_DIR.glob("*.pkl"))
    label_w = 90
    rows = []
    for w in writers:
        pairs = pickle.load(open(PAIR_CACHE_DIR / f"{w}.pkl", "rb"))
        holdout = [p for p in pairs if p["is_holdout"]]
        if not holdout:
            continue
        text = holdout[0]["text"]
        prof = profiles[w]

        anchor_xy, anchor_eos, pred_pw1 = render_for(model_pw1, writers1, prof, w, text, device)
        anchor_im = render_xy(anchor_xy, eos=anchor_eos)
        synth_pw1_im = render_xy(pred_pw1, eos=anchor_eos)

        _, _, pred_pw05 = render_for(model_pw05, writers05, prof, w, text, device)
        synth_pw05_im = render_xy(pred_pw05, eos=anchor_eos)

        rows.append((w, text, anchor_im, synth_pw1_im, synth_pw05_im))

    # No cropping -- a fixed crop width was hiding most of several
    # sentences (some render past 2000px at this scale) and made them
    # look empty/sparse when they weren't. Size the sheet to the widest
    # actual render instead, and paste every image in full.
    max_w = max(im.width for r in rows for im in r[2:5])
    row_h = max(im.height for r in rows for im in r[2:5]) + 25
    W = label_w + max_w
    H = row_h * len(rows) * 3 + 40
    sheet = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(sheet)
    y = 0
    for w, text, anchor_im, pw1_im, pw05_im in rows:
        d.text((5, y + 5), f"author {w}", fill=0)
        d.text((5, y + row_h - 15), '"' + text[:40] + '"', fill=0)
        y += 5
        for label, im in [("anchor", anchor_im), ("pw=1.0", pw1_im), ("pw=0.5", pw05_im)]:
            d.text((5, y + 20), label, fill=0)
            sheet.paste(im, (label_w, y))
            y += row_h
        y += 10
    out_path = OUT_DIR / "pos_weight_comparison_all_authors_full.png"
    sheet.save(out_path)
    print(f"-> {out_path}  {sheet.size}")


if __name__ == "__main__":
    main()
