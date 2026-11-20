"""
qtp_splits.py -- reproduce, line by line, which personal (yeukita/dylan) lines were
held out by each of the three modules that split them:

  text   : TrainText.collect_personal_lines_joint  (text recogniser, 'joint' checkpoint)
           rows FILTERED to charset/length first; ONE random.Random(SPLIT_SEED) shared across
           all pages, order yeukita pages then dylan pages; per-page shuffle; val=first
           max(1, round(0.15 n)).
  author : TrainAuthor.add_personal_samples        (writer classifiers)
           rows = all n paired lines (unfiltered); ONE shared Random(SPLIT_SEED); same
           per-page shuffle / n_val rule; empty-target lines skipped afterwards.
  profile: BuildStyleProfile.build_personal_line_items (style profiles used by synthesis)
           rows = all n paired lines; a FRESH random.Random(SPLIT_SEED) PER AUTHOR.

All three use VAL_FRACTION=0.15, SPLIT_SEED=0 (authors_config.py).  The split relies on
the number of TEXT crops/labels per page being the same as when the models were trained.
"""
import pickle
import random

import qtp_common as C


def compute_splits(page_rows, max_safe, val_fraction=0.15, seed=0):
    from np_inference.text_model import CHAR_TO_IDX
    # text
    rng = random.Random(seed)
    for pr in page_rows:
        kept = []
        for i, t in enumerate(pr["labels"]):
            k = "".join(c for c in t.strip() if c in CHAR_TO_IDX)
            if 0 < len(k) <= max_safe:
                kept.append(i)
        order = list(kept)
        rng.shuffle(order)
        nv = max(1, round(len(order) * val_fraction))
        pr["text_val_idx"] = sorted(order[:nv])
        pr["text_kept_idx"] = kept
    # author (shared rng, unfiltered)
    rng = random.Random(seed)
    for pr in page_rows:
        n = len(pr["labels"])
        order = list(range(n))
        rng.shuffle(order)
        nv = max(1, round(n * val_fraction))
        pr["author_val_idx"] = sorted(order[:nv])
    # profile (fresh rng per author)
    rngs = {}
    for pr in page_rows:
        rng = rngs.setdefault(pr["author"], random.Random(seed))
        n = len(pr["labels"])
        order = list(range(n))
        rng.shuffle(order)
        nv = max(1, round(n * val_fraction))
        pr["profile_val_idx"] = sorted(order[:nv])
    return page_rows


def load_seg_cache():
    with open(C.CACHE / "seg_cache.pkl", "rb") as f:
        d = pickle.load(f)
    compute_splits(d["personal"], d.get("max_safe") or 120, d["val_fraction"], d["split_seed"])  # MAX_SAFE_LABEL_CHARS = int((640//4)*0.75) = 120
    return d


def split_overlap_report(page_rows):
    """Per author: how many validation lines each split has and how they overlap."""
    out = {}
    for a in C.PERSONAL_AUTHORS:
        sets = {k: set() for k in ("text", "author", "profile")}
        total = 0
        for pi, pr in enumerate(page_rows):
            if pr["author"] != a:
                continue
            total += len(pr["labels"])
            for k, key in (("text", "text_val_idx"), ("author", "author_val_idx"),
                           ("profile", "profile_val_idx")):
                sets[k] |= {(pi, i) for i in pr[key]}
        out[a] = dict(total_lines=total,
                      n_text_val=len(sets["text"]), n_author_val=len(sets["author"]),
                      n_profile_val=len(sets["profile"]),
                      text_eq_author=sets["text"] == sets["author"],
                      author_eq_profile=sets["author"] == sets["profile"],
                      author_and_profile=len(sets["author"] & sets["profile"]),
                      text_and_profile=len(sets["text"] & sets["profile"]),
                      text_and_author=len(sets["text"] & sets["author"]))
    return out
