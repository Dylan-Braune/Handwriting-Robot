"""
EvaluateStyleTransferRNN.py -- does StyleTransferRNN's output actually look
like the right author AND say the right words, on HELD-OUT lines?

Same two questions as EvaluateStyle.py / VerifyRewrite.py, applied to the
anchor+residual style-transfer model instead of the plain trajectory
library, so the two approaches can be compared on the same footing:

  1. WRITER ID   -- the frozen 10-author classifier (TrainAuthor.py) reads
                     the rendered style-transferred line back. Compared
                     against the SAME classifier's accuracy on the plain
                     anchor (legibility=1.0, no style at all) -- the
                     interesting number is whether style transfer helps
                     writer-ID over the anchor baseline, not just whether
                     it beats chance.
  2. TEXT         -- the frozen text recognizer (TrainText.py) reads the
                     rendered line back; char accuracy against the known
                     ground-truth text (these are the SAME held-out lines
                     StyleTransferRNN's own validation split uses, i.e.
                     never trained on).

Also dumps a side-by-side sheet per author (real / anchor / style-transfer)
for eyeballing.

Run:
    python EvaluateStyleTransferRNN.py
    python EvaluateStyleTransferRNN.py --max-per-author 5
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import SynthesizeHandwriting as SY
from SynthesizeHandwriting import Trajectory
from EvaluateStyle import LoadAuthorModel, ClassifyImage
from VerifyRewrite import LoadTextModel, ReadText, CharAcc
from StyleTransferRNN import (
    PAIR_CACHE_DIR, BEST_PATH, N_POINTS, MM_PER_XH,
    StyleTransferRNN, trajectory_to_xy, _resample_arclength, render_xy,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parent / "NOGIT" / "StyleTransferEval"


def xy_eos_to_trajectory(xy, eos, mm_per_xh=MM_PER_XH):
    """(N,2) xy in x-height units + (N,) pen-lift flags -> a Trajectory in
    mm, using the SAME real rendering pipeline (RenderTrajectory: proper
    pen width, soft edges, ink-density calibration) the classifier was
    actually trained on -- render_xy's plain 2px polyline is a much
    cruder stand-in that may not match the classifier's input
    distribution at all, so writer-ID scored on it isn't a fair test."""
    pts_mm = xy * mm_per_xh
    strokes, chain = [], [pts_mm[0].tolist()]
    for i in range(1, len(pts_mm)):
        if eos[i - 1] > 0.5:
            if len(chain) >= 2:
                strokes.append(chain)
            chain = [pts_mm[i].tolist()]
        else:
            chain.append(pts_mm[i].tolist())
    if len(chain) >= 2:
        strokes.append(chain)
    return Trajectory(strokes, dict(mmPerXh=mm_per_xh))


@torch.no_grad()
def sample_for_text(model, writers, prof, writer_id, text, device):
    """Like StyleTransferRNN.sample(), but takes an already-loaded model +
    profile map so we don't reload the checkpoint per line."""
    traj = SY.SynthesizeText(text, prof, mmPerXh=MM_PER_XH, seed=0,
                             lineWidthMm=100_000.0, legibility=1.0)
    anchor_xy, anchor_eos = trajectory_to_xy(traj, MM_PER_XH)
    if anchor_xy is None:
        return None
    anchor_xy = anchor_xy - anchor_xy[0]
    a_rs, a_eos_rs = _resample_arclength(anchor_xy, N_POINTS, eos=anchor_eos)
    anchor_d = np.diff(a_rs, axis=0, prepend=a_rs[:1]).astype(np.float32)
    w_idx = torch.tensor([writers.index(writer_id)], device=device)
    pred_d = model(torch.from_numpy(anchor_d).unsqueeze(0).to(device), w_idx)
    pred_xy = np.cumsum(pred_d[0].cpu().numpy(), axis=0)
    return pred_xy, a_rs, a_eos_rs


def main(max_per_author=8):
    device = torch.device("cpu")
    if not BEST_PATH.exists():
        print(f"No trained model at {BEST_PATH} -- run training first.")
        return
    ck = torch.load(BEST_PATH, map_location=device, weights_only=False)
    writers = ck["writers"]
    model = StyleTransferRNN(len(writers), hidden=ck["hidden"], layers=ck["layers"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    profiles = SY.LoadAllProfiles()
    author_model, author_map, idx_to_author = LoadAuthorModel(device)
    text_model = LoadTextModel(device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for w in writers:
        pairs = pickle.load(open(PAIR_CACHE_DIR / f"{w}.pkl", "rb"))
        holdout = [p for p in pairs if p["is_holdout"]][:max_per_author]
        if not holdout:
            continue
        prof = profiles[w]
        sheet_ims = []
        for r in holdout:
            text = r["text"]
            out = sample_for_text(model, writers, prof, w, text, device)
            if out is None:
                continue
            pred_xy, anchor_xy, anchor_eos = out
            # for the classifier/text-recognizer scoring, render through the
            # SAME pipeline (RenderTrajectory: real pen width, soft edges,
            # ink-density match) those models were actually trained on
            synth_traj = xy_eos_to_trajectory(pred_xy, anchor_eos)
            anchor_traj = xy_eos_to_trajectory(anchor_xy, anchor_eos)
            synth_render = SY.RenderTrajectory(synth_traj, pxPerMm=18.0, profile=prof)
            anchor_render = SY.RenderTrajectory(anchor_traj, pxPerMm=18.0, profile=prof)
            # plain thin-line renders too, just for the eyeball comparison sheet
            synth_im = render_xy(pred_xy, eos=anchor_eos)
            anchor_im = render_xy(anchor_xy, eos=anchor_eos)
            real_im = render_xy(r["target"])

            # writer-ID + text on the style-transferred render
            wi_pred, _ = ClassifyImage(author_model, synth_render, device)
            wi_correct = idx_to_author[wi_pred] == w
            text_pred = ReadText(text_model, synth_render, device)
            char_acc = CharAcc(text_pred, text)

            # same two measurements on the raw anchor, as a baseline
            wi_pred_a, _ = ClassifyImage(author_model, anchor_render, device)
            wi_correct_a = idx_to_author[wi_pred_a] == w
            text_pred_a = ReadText(text_model, anchor_render, device)
            char_acc_a = CharAcc(text_pred_a, text)

            rows.append(dict(author=w, text=text,
                             wi_correct=wi_correct, char_acc=char_acc,
                             wi_correct_anchor=wi_correct_a, char_acc_anchor=char_acc_a))
            if len(sheet_ims) < 4:
                sheet_ims.append((real_im, anchor_im, synth_im, text))

        if sheet_ims:
            pad, rowH = 10, 90
            W = max(im.width for triple in sheet_ims for im in triple[:3]) + 220
            sheet = Image.new("L", (W, rowH * len(sheet_ims) + pad), 255)
            d = ImageDraw.Draw(sheet)
            for i, (real_im, anchor_im, synth_im, text) in enumerate(sheet_ims):
                y = i * rowH
                d.text((5, y), f"real:", fill=0)
                sheet.paste(real_im.crop((0, 0, min(real_im.width, 300), real_im.height)), (60, y))
                d.text((5, y + 30), f"anchor:", fill=0)
                sheet.paste(anchor_im.crop((0, 0, min(anchor_im.width, 300), anchor_im.height)), (60, y + 30))
                d.text((5, y + 60), f"synth:", fill=0)
                sheet.paste(synth_im.crop((0, 0, min(synth_im.width, 300), synth_im.height)), (60, y + 60))
            sheet.save(OUT_DIR / f"{w}_compare.png")

    if not rows:
        print("No holdout rows evaluated.")
        return

    print(f"\n{'author':>8} {'n':>4} {'writerID synth':>15} {'writerID anchor':>16} "
          f"{'charAcc synth':>14} {'charAcc anchor':>15}")
    all_wi, all_wi_a, all_ca, all_ca_a = [], [], [], []
    for w in writers:
        wrows = [r for r in rows if r["author"] == w]
        if not wrows:
            continue
        wi = np.mean([r["wi_correct"] for r in wrows])
        wi_a = np.mean([r["wi_correct_anchor"] for r in wrows])
        ca = np.mean([r["char_acc"] for r in wrows])
        ca_a = np.mean([r["char_acc_anchor"] for r in wrows])
        all_wi.append(wi); all_wi_a.append(wi_a); all_ca.append(ca); all_ca_a.append(ca_a)
        print(f"{w:>8} {len(wrows):>4} {wi:>14.1%} {wi_a:>16.1%} {ca:>14.1%} {ca_a:>15.1%}")
    print(f"{'MEAN':>8} {len(rows):>4} {np.mean(all_wi):>14.1%} {np.mean(all_wi_a):>16.1%} "
          f"{np.mean(all_ca):>14.1%} {np.mean(all_ca_a):>15.1%}")
    print(f"\nComparison sheets -> {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-author", type=int, default=8)
    args = ap.parse_args()
    main(max_per_author=args.max_per_author)
