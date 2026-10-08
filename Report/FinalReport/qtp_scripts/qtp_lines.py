"""
qtp_lines.py -- the REAL handwriting lines used by QTP1 (and the recogniser ceiling).

IAM writers (150 151 152 153 384 551 552 588): the cached line images that the training
scripts use (Software/CNN/NOGIT/line_cache_authors10, built by LineImageCache) with their
transcriptions; `heldout` = the writer's held-out PAGE (last labelled page, the page the
style profile never saw; IAMLineDatasetRaw rule).
Personal writers (yeukita, dylan): ProcessPage TEXT crops of the personal photos paired
with *_labels.txt (see qtp0_segment_cache.py).  Three different 'held-out' flags exist:
  heldout_profile : lines NOT used to build the style profile (BuildStyleProfile split)
  heldout_text    : validation split of the text recogniser   (TrainText joint split)
  heldout_author  : validation split of the writer classifiers (TrainAuthor split)
For IAM writers all three coincide with the held-out page.
"""
import io
import pickle
import re

import numpy as np
from PIL import Image

import qtp_common as C
import qtp_splits as S


def _decode(png):
    return np.array(Image.open(io.BytesIO(png)).convert("L"))


def charset_ok(text):
    from np_inference.text_model import CHAR_TO_IDX
    return all(c in CHAR_TO_IDX for c in text)


def load_real_lines():
    """list of dict(writer, kind, page, line_idx, text, gray(uint8), heldout_profile,
    heldout_text, heldout_author)."""
    out = []
    with open(C.CACHE / "iam_lines.pkl", "rb") as f:
        iam = pickle.load(f)
    for r in iam:
        out.append(dict(writer=r["author"], kind="iam", page=r["page_key"].split("/")[1],
                        line_idx=r["line_idx"], text=r["text"], gray=_decode(r["png"]),
                        heldout_profile=bool(r["is_holdout"]), heldout_text=bool(r["is_holdout"]),
                        heldout_author=bool(r["is_holdout"])))
    d = S.load_seg_cache()
    for pr in d["personal"]:
        for i, (crop, text) in enumerate(zip(pr["crops"], pr["labels"])):
            g = np.asarray(crop)
            if g.ndim == 3:
                g = g[..., 0]
            out.append(dict(writer=pr["author"], kind="personal", page=pr["page"], line_idx=i,
                            text=text, gray=g.astype(np.uint8),
                            heldout_profile=i in pr["profile_val_idx"],
                            heldout_text=i in pr["text_val_idx"],
                            heldout_author=i in pr["author_val_idx"]))
    return out


def norm_text(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def pick_overlay_texts(lines, k=3, lo=22, hi=60):
    """K held-out (w.r.t. the STYLE PROFILE) real lines per writer, charset-clean, 22-60
    characters, spaced evenly through the writer's eligible held-out lines."""
    picks = []
    for w in C.ALL_AUTHORS:
        elig = [l for l in lines if l["writer"] == w and l["heldout_profile"]
                and lo <= len(l["text"].strip()) <= hi and charset_ok(l["text"].strip())]
        elig.sort(key=lambda l: (l["page"], l["line_idx"]))
        if not elig:
            continue
        idx = sorted(set(int(round(x)) for x in np.linspace(0, len(elig) - 1, min(k, len(elig)))))
        for j in idx:
            picks.append(elig[j])
    return picks
