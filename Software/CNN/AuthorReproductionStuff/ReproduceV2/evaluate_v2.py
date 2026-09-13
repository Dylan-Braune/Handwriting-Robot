"""
evaluate_v2.py -- backend-agnostic version of EvaluateStyle.RunFullEvaluation
for the revised pipeline (../REVISED_METHOD.md stage H).

For every author: take a real HELD-OUT line, reproduce its text with the
chosen backend, and report
  - line CER      (PaperCRNN, legibility)
  - shape-ID      (author_shape_10_weights.pt -- NOT the ink model)
  - style prob    (same judge, continuous)
plus a real-vs-synth comparison sheet (real line on top, synth same text
below), identical format to EvaluateStyle's.

Usage:
    python evaluate_v2.py                       # legacy_synth backend
    python evaluate_v2.py --backend one_dm      # once its checkpoint is set up
    python evaluate_v2.py --n 3                 # 3 lines/author instead of 1
"""

import argparse
import json

import numpy as np
import torch
from PIL import Image, ImageDraw

import _env
from _env import AUTHORS, OUT_DIR, DATA_DIR
from TrainText import IAMLineDatasetRaw, _decode_png
import BuildStyleProfile as SP

from htg_backend import BACKENDS
from reproduce_v2 import (
    load_text_model, load_shape_model, reproduce, shape_prob, ctc_read,
    levenshtein,
)


def holdout_lines(n_per_author):
    base = IAMLineDatasetRaw(root_dir=str(DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    out = {}
    for s in base.samples:
        a = s["page_key"].split("/")[0]
        if not s.get("is_holdout") or a not in AUTHORS:
            continue
        if not (18 <= len(s["text"]) <= 44):
            continue
        out.setdefault(a, [])
        if len(out[a]) < n_per_author:
            out[a].append((_decode_png(s["image_png"]).convert("L"), s["text"]))
    return out


def sheet(rows, path, h=62, maxw=1500):
    def fit(im):
        s = h / im.height
        im = im.resize((max(1, int(im.width * s)), h), Image.Resampling.LANCZOS)
        return im.crop((0, 0, min(maxw, im.width), h))
    lab = 120
    W = lab + min(maxw, max(max(fit(r[2]).width, fit(r[3]).width) for r in rows)) + 16
    H = sum(2 * h + 34 for _ in rows) + 16
    sh = Image.new("L", (W, H), 245)
    d = ImageDraw.Draw(sh)
    y = 8
    for a, text, real, syn in rows:
        d.text((6, y + 20), str(a), fill=0)
        d.text((6, y + 40), "real", fill=90)
        d.text((6, y + 40 + h), "v2", fill=90)
        d.text((lab, y), text[:80], fill=110)
        sh.paste(fit(real), (lab, y + 14))
        sh.paste(fit(syn), (lab, y + 14 + h))
        d.line([(0, y + 2 * h + 20), (W, y + 2 * h + 20)], fill=200)
        y += 2 * h + 34
    sh.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="legacy_synth", choices=list(BACKENDS))
    ap.add_argument("--n", type=int, default=1, help="lines per author")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}   [backend] {args.backend}")
    text_model = load_text_model(device)
    shape_model, mapping = load_shape_model(device)
    models = (text_model, shape_model, mapping)
    backend = BACKENDS[args.backend]()

    samples = holdout_lines(args.n)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, per = [], {}
    real_ceiling_hits = real_ceiling_tot = 0
    for a in AUTHORS:
        if a not in samples:
            continue
        cers, ids, stys = [], [], []
        first_pair = None
        for real_img, text in samples[a]:
            r = reproduce(text, a, backend=backend, k=args.k, device=device,
                          _models=models, verbose=False)
            cers.append(r["cer"])
            stys.append(r["style"])
            # top-1 shape id of the synth line
            probs = _shape_probs(shape_model, mapping, r["image"], device)
            ids.append(int(np.argmax(probs)) == mapping[a])
            # real-line ceiling for the same judge
            rp = _shape_probs(shape_model, mapping, real_img, device)
            real_ceiling_hits += int(np.argmax(rp)) == mapping[a]
            real_ceiling_tot += 1
            if first_pair is None:
                first_pair = (real_img, r["image"], text)
        per[a] = dict(cer=float(np.mean(cers)), style=float(np.mean(stys)),
                      shape_id=float(np.mean(ids)))
        rows.append((a, first_pair[2], first_pair[0], first_pair[1]))
        print(f"  {a}: CER {per[a]['cer']:.2%}  shape-ID {per[a]['shape_id']:.0%}  "
              f"style {per[a]['style']:.2f}")

    overall = dict(
        cer=float(np.mean([v["cer"] for v in per.values()])),
        shape_id=float(np.mean([v["shape_id"] for v in per.values()])),
        style=float(np.mean([v["style"] for v in per.values()])),
        n_at_85=sum(v["shape_id"] >= 0.85 for v in per.values()),
        real_ceiling=real_ceiling_hits / max(1, real_ceiling_tot),
    )
    print("\n--- summary ------------------------------------------------")
    print(f"  mean line CER      : {overall['cer']:.2%}")
    print(f"  mean shape-ID      : {overall['shape_id']:.1%}   "
          f"({overall['n_at_85']}/{len(per)} authors >= 85%)")
    print(f"  mean style prob    : {overall['style']:.2f}")
    print(f"  real-line ceiling  : {overall['real_ceiling']:.1%} "
          f"(same judge on the authors' own held-out lines)")

    out_sheet = OUT_DIR / f"real_vs_v2_{args.backend}.png"
    sheet(rows, out_sheet)
    with open(OUT_DIR / f"results_{args.backend}.json", "w", encoding="utf-8") as f:
        json.dump(dict(perAuthor=per, overall=overall, backend=args.backend), f,
                  indent=2)
    print(f"\n  sheet : {out_sheet}")


def _shape_probs(model, mapping, pil, device):
    import TrainAuthorShape as SH
    from TrainText import resize_line_image_fixed, tensor_from_resized
    norm = SH.StrokeNormalize(np.array(pil.convert("L")))
    if norm is None:
        return np.zeros(len(mapping))
    t = tensor_from_resized(resize_line_image_fixed(norm)).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.softmax(model(t), dim=1)[0].cpu().numpy()


if __name__ == "__main__":
    main()
