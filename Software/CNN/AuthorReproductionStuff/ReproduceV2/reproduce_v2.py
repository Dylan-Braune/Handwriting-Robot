"""
reproduce_v2.py -- revised author-reproduction pipeline, IMAGE OUTPUT ONLY.

See ../REVISED_METHOD.md.

  line-level backend (diffbrush, legacy_synth): ask for K whole-line
    candidates + 1 archetype floor; keep the most CTC-legible backend line.
  word-level backend (one_dm): K candidates per word, drop blown-out
    renders, keep the most legible, compose into a line.

The frozen judges are PaperCRNN (CER / legibility) and the shape-only
writer-ID model (style). The style judge can't score diffusion texture, so
for the neural backends selection is legibility-first among candidates that
already share one consistent hand by construction.

Usage:
    python reproduce_v2.py "Only Mr. Lucas's actions, therefore, arose" 153 --backend diffbrush
    python reproduce_v2.py "some text" 151 --backend legacy_synth --k 6
"""

import argparse
import random

import numpy as np
import torch
from PIL import Image

import _env
from _env import PROFILE_DIR, TEXT_WEIGHTS, SHAPE_WEIGHTS, OUT_DIR, DATA_DIR

from TrainText import (
    CHARSET, PaperCRNN, decode_ctc, levenshtein,
    resize_line_image_fixed, tensor_from_resized,
    IAMLineDatasetRaw, _decode_png,
)
from TrainAuthor import AuthorClassifierCNN
import BuildStyleProfile as SP
import TrainAuthorShape as SH
import json

from htg_backend import BACKENDS, PrintArchetypeBackend


# ---------------------------------------------------------------------------
# frozen judges
# ---------------------------------------------------------------------------
def load_text_model(device):
    m = PaperCRNN(num_classes=len(CHARSET) + 1).to(device)
    m.load_state_dict(torch.load(TEXT_WEIGHTS, map_location=device, weights_only=False))
    m.eval()
    return m


def load_shape_model(device):
    ck = torch.load(SHAPE_WEIGHTS, map_location=device, weights_only=False)
    mapping = ck["author_mapping"]
    m = AuthorClassifierCNN(num_authors=len(mapping)).to(device)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, mapping


def _as_line(pil, aspect=9.0):
    """Pad a word/line crop with white to at least `aspect`:1 before feeding
    the recogniser -- PaperCRNN was trained on full lines and mis-reads a
    near-square word crop that gets stretched to 640x64."""
    pil = pil.convert("L")
    w, h = pil.size
    target_w = max(w, int(h * aspect))
    if target_w == w:
        return pil
    out = Image.new("L", (target_w, h), 255)
    out.paste(pil, ((target_w - w) // 2, 0))
    return out


def ctc_read(model, pil, device):
    t = tensor_from_resized(resize_line_image_fixed(_as_line(pil))).unsqueeze(0).to(device)
    with torch.no_grad():
        return decode_ctc(model(t))[0]


def shape_prob(model, mapping, author_id, pil, device):
    """P(this line was written by author_id) under the shape-only judge,
    after the same stroke normalisation the judge was trained on."""
    norm = SH.StrokeNormalize(np.array(pil.convert("L")))
    if norm is None:
        return 0.0
    t = tensor_from_resized(resize_line_image_fixed(norm)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0].cpu().numpy()
    return float(probs[mapping[author_id]]) if author_id in mapping else 0.0


# ---------------------------------------------------------------------------
# reference material
# ---------------------------------------------------------------------------
def load_profile(author_id):
    with open(PROFILE_DIR / f"{author_id}.json", encoding="utf-8") as f:
        return json.load(f)


def holdout_refs(author_id, n=5):
    base = IAMLineDatasetRaw(root_dir=str(DATA_DIR), cache_dir=str(SP.CACHE_DIR))
    out = []
    for s in base.samples:
        if s["page_key"].split("/")[0] != author_id:
            continue
        if not s.get("is_holdout"):
            continue
        out.append(_decode_png(s["image_png"]).convert("L"))
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# stage G: compose word images into a line
# ---------------------------------------------------------------------------
def _ink_crop(im, thresh=245):
    a = np.asarray(im.convert("L"))
    ys, xs = np.where(a < thresh)
    if len(xs) == 0:
        return im
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def compose_line(word_imgs, profile, seed=0, pad=16, target_ink_h=44):
    """Crop each word to its ink, normalise every word to a common ink
    height (independently generated word crops otherwise drift in size),
    then lay them on one baseline."""
    rng = random.Random(seed)
    crops = [_ink_crop(im) for im in word_imgs]
    med_h = float(np.median([c.height for c in crops])) or target_ink_h
    norm = []
    for c in crops:
        # scale toward the median word height, but clamp so one odd word
        # can't blow up or shrink away
        s = float(np.clip(med_h / max(1, c.height), 0.75, 1.7))
        norm.append(c.resize((max(1, round(c.width * s)),
                              max(1, round(c.height * s))),
                             Image.Resampling.LANCZOS))
    word_imgs = norm
    gap = int(np.clip(0.5 * med_h * profile.get("wordSpaceXh", 1.4) / 1.4, 14, 46))
    line_h = int(max(im.height for im in word_imgs) + 4 * pad)
    total_w = sum(im.width for im in word_imgs) + gap * (len(word_imgs) - 1) + 2 * pad
    canvas = Image.new("L", (total_w, line_h), 255)
    x = pad
    baseline = line_h // 2
    for im in word_imgs:
        dy = int(rng.uniform(-0.05, 0.05) * med_h)   # per-word baseline drift
        y = baseline - im.height // 2 + dy
        canvas.paste(im, (x, y))
        x += im.width + gap
    # tiny whole-line slope
    slope = rng.uniform(-2.0, 2.0)
    canvas = canvas.transform(
        canvas.size, Image.Transform.AFFINE,
        (1, 0, 0, slope / canvas.width, 1, 0),
        resample=Image.Resampling.BILINEAR, fillcolor=255)
    return canvas.crop(canvas.getbbox() or (0, 0, canvas.width, canvas.height))


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def score_img(img, target, text_model, shape_model, mapping, author_id,
              device, w_leg=0.45, w_sty=0.55):
    pred = ctc_read(text_model, img, device)
    cer = levenshtein(pred, target) / max(1, len(target))
    sty = shape_prob(shape_model, mapping, author_id, img, device)
    return dict(pred=pred, cer=cer, sty=sty,
                combined=w_leg * (1.0 - cer) + w_sty * sty)


# ---------------------------------------------------------------------------
# line-level backend (legacy synth): K whole-line candidates, pick best
# ---------------------------------------------------------------------------
def _reproduce_line_backend(backend, text, style, profile, k, seed, judges,
                            device, verbose):
    text_model, shape_model, mapping, author_id = judges
    cands = [("backend", im)
             for im in backend.generate_line(text, style, k=k, seed=seed)]
    cands.append(("archetype", compose_line(
        [PrintArchetypeBackend().generate(w, style)[0] for w in text.split()],
        profile, seed=seed)))
    scored = []
    for tag, img in cands:
        s = score_img(img, text, text_model, shape_model, mapping, author_id, device)
        s.update(tag=tag, img=img)
        scored.append(s)
        if verbose:
            print(f"  [{tag:9s}] CER {s['cer']:.2%}  style {s['sty']:.2f}  "
                  f"score {s['combined']:.3f}")
    # the neural backend keeps a consistent hand across the whole line, which
    # the style judge can't score on diffusion texture -- so among readable
    # backend candidates, take the most legible; only fall through to the
    # archetype line if every backend candidate is genuinely unreadable.
    gen = [s for s in scored if s["tag"] == "backend"]
    readable = [s for s in gen if s["cer"] <= 0.5]
    if readable:
        best = min(readable, key=lambda s: s["cer"])
    elif gen:
        best = min(gen, key=lambda s: s["cer"])
    else:
        best = max(scored, key=lambda s: s["combined"])
    return best, (1 if best["tag"] == "archetype" else 0)


# ---------------------------------------------------------------------------
# word-level backend (One-DM): best-of-K per word -> compose -> line repair
# ---------------------------------------------------------------------------
def _ink_fraction(im):
    a = np.asarray(im.convert("L"), np.float32) / 255.0
    return float((a < 0.6).mean())


def _reproduce_word_backend(backend, text, style, profile, k, seed, judges,
                            device, verbose):
    """One-DM makes each word in a consistent hand by construction (one
    anchor style crop). So candidate selection is: drop the rare blown-out
    / blocky-print failures (ink fraction out of band), then take the most
    CTC-legible of what's left. Archetype only if nothing is readable."""
    text_model, shape_model, mapping, author_id = judges
    words = text.split()
    chosen, n_arch = [], 0
    for wi, word in enumerate(words):
        cands = backend.generate(word, style, k=k, seed=seed + 1000 * wi)
        rows = []
        for img in cands:
            ink = _ink_fraction(img)
            pred = ctc_read(text_model, img, device)
            cer = levenshtein(pred, word) / max(1, len(word))
            rows.append(dict(img=img, ink=ink, cer=cer, pred=pred))
        ok = [r for r in rows if 0.02 <= r["ink"] <= 0.16]      # cursive band
        if ok:
            best = min(ok, key=lambda r: (r["cer"], abs(r["ink"] - 0.08)))
        else:                                                   # all blown out
            best = min(rows, key=lambda r: (r["ink"], r["cer"]))
        if best["cer"] > 0.6:                                   # unreadable
            best = dict(img=PrintArchetypeBackend().generate(word, style)[0],
                        ink=0, cer=best["cer"], pred="", is_arch=True)
            n_arch += 1
        chosen.append(best["img"])
        if verbose:
            print(f"  {word!r:16s} {'arch' if best.get('is_arch') else 'one_dm':6s}"
                  f" CER {best['cer']:.2f}  ink {best['ink']:.3f}")

    line = compose_line(chosen, profile, seed=seed)
    s = score_img(line, text, text_model, shape_model, mapping, author_id, device)
    s.update(tag="one_dm", img=line)
    return s, n_arch


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------
def reproduce(text, author_id, backend=None, k=6, seed=0, device=None,
              _models=None, verbose=True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model, shape_model, mapping = _models or (
        load_text_model(device), *load_shape_model(device))
    from htg_backend import LegacySynthBackend
    backend = backend or LegacySynthBackend()
    profile = load_profile(author_id)
    refs = holdout_refs(author_id)
    style = backend.describe_style(author_id, refs, profile)
    judges = (text_model, shape_model, mapping, author_id)

    fn = (_reproduce_line_backend if callable(getattr(backend, "generate_line", None))
          else _reproduce_word_backend)
    best, n_arch = fn(backend, text, style, profile, k, seed, judges, device, verbose)

    return dict(image=best["img"], pred=best["pred"], cer=best["cer"],
                style=best["sty"], source=best["tag"],
                n_words=len(text.split()), n_archetype=n_arch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("author")
    ap.add_argument("--backend", default="legacy_synth", choices=list(BACKENDS))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}   [backend] {args.backend}")
    backend = BACKENDS[args.backend]()

    r = reproduce(args.text, args.author, backend=backend, k=args.k, seed=args.seed,
                  device=device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.author}_{args.backend}.png"
    r["image"].save(out)
    print(f"\n  line CER {r['cer']:.2%} | style {r['style']:.2f} | "
          f"{r['n_archetype']}/{r['n_words']} words fell back to archetype")
    print(f"  reader saw: {r['pred']!r}")
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
