"""
BuildStyleProfile10Authors.py -- extends BuildStyleProfile.py's pipeline to
the NEW 10-author set: 8 kept dataset authors (150,151,152,153,384,551,552,
588 -- dropped 154/155 for being redundant with 150/151/152's style
cluster) + 2 personal authors (yeukita, dylan, from Software/CNN/NOGIT/
yeukita and dylan).

WHY A SEPARATE SCRIPT, NOT AN EDIT TO BuildStyleProfile.py
    BuildStyleProfile.py's BuildAll() is hardwired to IAMLineDatasetRaw
    (IAMpages10 folder layout). The 8 kept dataset authors already have
    fully-extracted, cached raw glyph libraries in NOGIT/GlyphCache10/ from
    earlier work -- loaded straight from cache here, NOT re-extracted, so
    this only pays the (expensive, forced-alignment) extraction cost for
    the 2 new personal authors. The two personal authors are segmented
    with SegmentPage.ProcessPage (the same personal-page pipeline
    ClassifyText.py uses for non-IAM pages) instead of IAMLineDatasetRaw,
    and forced-aligned with the PERSONAL fine-tuned recognizer (not the
    general hf model) -- per the measured finding that the personal
    fine-tune reads their handwriting far better (90%+ vs whatever the
    general model gets on it) even though it's worse on general text.

    Everything downstream (BuildLetterPrior pooling, BuildAuthorProfile,
    JSON output) is reused UNCHANGED from BuildStyleProfile.py, and all 10
    profiles are written to the SAME NOGIT/StyleProfiles10/ directory so
    SynthesizeHandwriting.py/WriteAsAuthor.py etc. don't need to change at
    all to pick up yeukita/dylan as author IDs.

Run:
    python BuildStyleProfile10Authors.py
"""
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import SegmentPage as PS
from ExtractIAMLines import ReadLabelLines
from TrainText import CHAR_TO_IDX, CHARSET, PaperCRNN
import BuildStyleProfile as BSP

SCRIPT_DIR = Path(__file__).resolve().parent
NOGIT_DIR = SCRIPT_DIR.parent / "NOGIT"
RAW_DIR = BSP.RAW_DIR              # NOGIT/GlyphCache10
PROFILE_DIR = BSP.PROFILE_DIR      # NOGIT/StyleProfiles10
PERSONAL_WEIGHTS = NOGIT_DIR / "weights" / "paper_cnn_bilstm_ctc_personal_best.pt"

DATASET_AUTHORS = ["150", "151", "152", "153", "384", "551", "552", "588"]
PERSONAL_AUTHORS = ["yeukita", "dylan", "robert", "owen", "abhinav"]
VAL_FRACTION = 0.15          # same holdout fraction as TrainTextPersonal.py
SPLIT_SEED = 0                # same seed too -- same lines held out from both


def build_personal_line_items(author, model, device):
    """Segments every photo for one personal author, splits its lines into
    train/holdout (per-page, same convention as TrainTextPersonal.py), and
    returns (train_items, holdout_texts) where train_items is a list of
    (gray_crop, text) ready for ExtractAuthorRaw."""
    rng = random.Random(SPLIT_SEED)
    train_items, holdout_texts = [], []
    for img_path in sorted((NOGIT_DIR / author).glob("*.jpg")):
        results, _preview, _meta = PS.ProcessPage(str(img_path))
        crops = [r["raw_crop"] for r in results if r["tag"] == "TEXT"]
        gt = ReadLabelLines(str(img_path))
        gt = [g for g in gt if g.strip() != "MESS"]
        n = min(len(crops), len(gt))
        rows = list(zip(crops[:n], gt[:n]))
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * VAL_FRACTION))
        for crop, text in rows[n_val:]:
            train_items.append((crop, text))
        for _crop, text in rows[:n_val]:
            holdout_texts.append(text)
    return train_items, holdout_texts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    allParsed = {}   # author -> (parsed, refs)

    # ---- 8 kept dataset authors: load straight from existing cache ----
    for a in DATASET_AUTHORS:
        rawPath = RAW_DIR / f"{a}.pkl"
        if not rawPath.exists():
            raise SystemExit(f"[Error] expected cached raw glyphs at {rawPath} -- "
                              f"run BuildStyleProfile.py's BuildAll() first if missing.")
        with open(rawPath, "rb") as f:
            parsed = pickle.load(f)
        # refs (RenderRefStats) aren't cached separately in the original
        # pipeline -- they're cheap (no model, no forced alignment) so just
        # recompute from the same cached parsed data's own stats, which
        # already embeds 'ref' per line (see ExtractAuthorRaw).
        refs = [stats["ref"] for _glyphs, stats in parsed if "ref" in stats]
        allParsed[a] = (parsed, refs)
        print(f"[Dataset] {a}: {len(parsed)} lines loaded from cache")

    # ---- 2 personal authors: fresh extraction with the personal model ----
    personal_model = PaperCRNN(num_classes=len(CHARSET) + 1).to(device)
    sd = torch.load(PERSONAL_WEIGHTS, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    personal_model.load_state_dict(sd)
    personal_model.eval()

    for a in PERSONAL_AUTHORS:
        rawPath = RAW_DIR / f"{a}.pkl"
        if rawPath.exists():
            with open(rawPath, "rb") as f:
                parsed = pickle.load(f)
            print(f"[Personal] {a}: {len(parsed)} lines loaded from cache")
        else:
            train_items, holdout_texts = build_personal_line_items(a, personal_model, device)
            print(f"[Personal] {a}: {len(train_items)} train lines, "
                  f"{len(holdout_texts)} held out")
            parsed = BSP.ExtractAuthorRaw(a, train_items, personal_model, device)
            with open(rawPath, "wb") as f:
                pickle.dump(parsed, f)
            print(f"[Personal] {a}: {len(parsed)}/{len(train_items)} lines usable after extraction")
        refs = [stats["ref"] for _glyphs, stats in parsed if "ref" in stats]
        allParsed[a] = (parsed, refs)

    # ---- shared cross-author letter prior (needs ALL 10 pooled) ----
    rawLibs = {}
    for a2, (parsed2, _r) in allParsed.items():
        lib = {}
        for glyphs, _st in parsed2:
            for g in glyphs:
                if g and g.get('char', ' ') != ' ' and 'strokes' in g:
                    lib.setdefault(g['char'], []).append(g)
        rawLibs[a2] = lib
    prior = BSP.BuildLetterPrior(rawLibs)
    print(f"\n[Prior] learned for {len(prior)} characters across {len(allParsed)} authors")

    for a2 in sorted(allParsed):
        parsed2, refs2 = allParsed[a2]
        prof = BSP.BuildAuthorProfile(a2, parsed2, refs=refs2, prior=prior)
        if prof is None:
            print(f"  {a2}: FAILED (no usable lines)")
            continue
        with open(PROFILE_DIR / f"{a2}.json", "w", encoding="utf-8") as f:
            json.dump(prof, f)
        print(f"  {a2}: profile saved")

    print(f"\nProfiles saved to {PROFILE_DIR}")
    print(f"Final 10-author set: {sorted(allParsed)}")


if __name__ == "__main__":
    main()
