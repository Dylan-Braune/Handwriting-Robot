"""Stage 8: build the visual examples the report embeds, one panel per author.

Produces, under Report/FullPipelineRun/images/:

  seg_<author>.jpg        real page + the segmentation's own cut overlay, and a
                          stacked strip of the actual line crops it produced
  classify_<author>.png   the author's real lines with the writer-ID verdict
                          underneath each one (predicted author + confidence)
  repro_<author>.png      what the pipeline DRAWS for that author: the rendered
                          synthesis, the same thing after the G-code round trip,
                          and for comparison a real line by that same author

All three read artifacts earlier stages already wrote; this script computes no
new accuracy numbers, it only renders what was measured.

Run from Software/CNN:
    python NOGIT/full_pipeline_run/make_visual_examples.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

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
REPO = _root

from TrainText import IAMLineDatasetRaw, _decode_png  # noqa: E402
from authors_config import DATASET_AUTHORS, PERSONAL_AUTHORS  # noqa: E402


CROPS = _cnn / "Crops"
END2END = CNN_DIR / "NOGIT" / "EndToEnd"
CACHE_DIR = CNN_DIR / "NOGIT" / "line_cache_authors10"
DATA_DIR = _root / "Data" / "Datasets" / "IAMpages10"
IMAGES = _root / "Verification" / "output" / "images"
GUTTER = 8


def rgb(im):
    return im.convert("RGB") if im.mode != "RGB" else im


def stack_vertical(imgs, width, gutter=GUTTER, bg=245):
    """Scale each image to `width` and stack with a light gutter."""
    scaled = []
    for im in imgs:
        im = rgb(im)
        h = max(1, round(im.height * (width / im.width)))
        scaled.append(im.resize((width, h), Image.Resampling.LANCZOS))
    H = sum(i.height for i in scaled) + gutter * (len(scaled) + 1)
    out = Image.new("RGB", (width + 2 * gutter, H), (bg, bg, bg))
    y = gutter
    for im in scaled:
        out.paste(im, (gutter, y))
        y += im.height + gutter
    return out


# -----------------------------------------------------------------------------
# 1. Segmentation panel
# -----------------------------------------------------------------------------
def segmentation_panel(author, page, crop_dir):
    ov = crop_dir / "_cuts_overlay.jpg"
    crops = sorted(crop_dir.glob("[0-9][0-9].png"))
    if not crops:
        return None
    parts = []
    if ov.exists():
        parts.append(Image.open(ov))
    crops_img = stack_vertical([Image.open(c) for c in crops], 1100)
    parts.append(crops_img)
    return stack_vertical(parts, 1100)


# -----------------------------------------------------------------------------
# 2. Classification panel -- real line + verdict
# -----------------------------------------------------------------------------
def classification_panel(rows_for_author, title):
    """rows_for_author: list of (path, truth, predicted, confidence, correct)."""
    if not rows_for_author:
        return None
    CELL_W = 1100
    LINE_H = 62
    LBL_H = 26
    n = len(rows_for_author)
    out = Image.new("RGB", (CELL_W, (LINE_H + LBL_H) * n + 40), "white")
    d = ImageDraw.Draw(out)
    d.text((6, 6), title, fill=(0, 0, 0))
    y = 32
    for path, truth, pred, conf, ok in rows_for_author:
        im = Image.open(path).convert("L")
        s = min(LINE_H / max(1, im.height), (CELL_W - 8) / max(1, im.width))
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))),
                       Image.Resampling.LANCZOS)
        out.paste(im.convert("RGB"), (6, y + (LINE_H - im.height) // 2))
        mark = "OK " if ok else "MISS"
        col = (0, 120, 0) if ok else (200, 0, 0)
        d.text((6, y + LINE_H + 4),
               f"{mark} true={truth}  pred={pred}  conf={conf*100:.1f}%", fill=col)
        y += LINE_H + LBL_H
    return out


# -----------------------------------------------------------------------------
# 3. Reproduction panel
# -----------------------------------------------------------------------------
def reproduction_panel(author, real_sample):
    parts = []
    for tag, name in (("synthesised (rendered)", f"{author}_synth.png"),
                      ("after G-code round-trip", f"{author}_gcode.png")):
        p = END2END / name
        if not p.exists():
            continue
        im = Image.open(p).convert("RGB")
        band = Image.new("RGB", (im.width, im.height + 24), "white")
        ImageDraw.Draw(band).text((6, 6), tag, fill=(0, 0, 160))
        band.paste(im, (0, 24))
        parts.append(band)
    if real_sample is not None:
        im = rgb(real_sample)
        band = Image.new("RGB", (im.width, im.height + 24), "white")
        ImageDraw.Draw(band).text((6, 6), "REAL line by this author (for comparison)",
                                  fill=(0, 0, 160))
        band.paste(im, (0, 24))
        parts.append(band)
    if not parts:
        return None
    return stack_vertical(parts, 1100)


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)

    # per-line predictions from Stage 4
    pred_csv = THIS_DIR / "stage4_per_line_predictions.csv"
    preds = defaultdict(list)
    if pred_csv.exists():
        with open(pred_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                preds[r["model"]].append(r)
        print(f"[Preds] loaded {len(preds)} model tables from {pred_csv.name}")
    else:
        print(f"[Warn] {pred_csv.name} missing -- classification panels skipped "
              f"(run verify_author_classification.py first)")

    # real IAM line per author, for the reproduction comparison
    base = IAMLineDatasetRaw(root_dir=str(DATA_DIR), cache_dir=str(CACHE_DIR))
    iam_real = {}
    for s in base.samples:
        a = s["page_key"].split("/")[0]
        if a in DATASET_AUTHORS and a not in iam_real and 20 <= len(s["text"]) <= 60:
            iam_real[a] = _decode_png(s["image_png"]).convert("L")

    authors = DATASET_AUTHORS + [a for a in PERSONAL_AUTHORS]
    done = []
    for a in authors:
        is_personal = a in PERSONAL_AUTHORS
        cap = a.capitalize() if is_personal else a

        # --- find this author's crop page dirs (personal only) ---------------
        page_dirs = []
        if is_personal:
            adir = CROPS / cap
            if adir.exists():
                page_dirs = sorted(d for d in adir.iterdir() if d.is_dir())

        # --- Stage 8.1 segmentation panel ------------------------------------
        if page_dirs:
            pd = page_dirs[0]
            p = segmentation_panel(a, pd.name, pd)
            if p:
                out = IMAGES / f"seg_{a}.jpg"
                p.save(out, quality=88)
                print(f"[seg] {a}: {out.name} ({p.width}x{p.height})")

        # --- Stage 8.2 classification panel ----------------------------------
        for model in ("shape", "ink"):
            rows = [r for r in preds.get(model, []) if r["true_author"] == a]
            if not rows:
                continue
            # prefer the personal-photo crops for personal authors
            sel = [r for r in rows if r["source"] == "personal-photo"] or rows
            sel = sel[:12]
            prepared = []
            for r in sel:
                gid = r["gid"]
                if r["source"] == "personal-photo":
                    auth, page, stem = gid.split("/")
                    path = CROPS / auth.capitalize() / page / "classifier" / f"{stem}.png"
                    if not path.exists():
                        path = CROPS / auth.capitalize() / page / f"{stem}.png"
                else:
                    path = None
                if path is None or not path.exists():
                    continue
                prepared.append((path, r["true_author"], r["predicted"],
                                 float(r["confidence"]), r["correct"] == "1"))
            if not prepared:
                continue
            ttl = (f"{a} -- {model} writer-ID on real lines "
                   f"({sum(1 for x in prepared if x[4])}/{len(prepared)} correct)")
            p = classification_panel(prepared, ttl)
            if p:
                out = IMAGES / f"classify_{model}_{a}.png"
                p.save(out)
                print(f"[classify] {a}/{model}: {out.name} ({p.width}x{p.height})")

        # --- Stage 8.3 reproduction panel ------------------------------------
        # Real comparison line: an actual IAM holdout line for dataset authors;
        # for the personal authors, their own real crop from Stage 1.
        real = iam_real.get(a)
        if real is None and page_dirs:
            if is_personal:
                adir = CROPS / cap
                if adir.exists():
                    page_dirs = sorted(d for d in adir.iterdir() if d.is_dir())
            if page_dirs:
                cands = sorted(page_dirs[0].glob("[0-9][0-9].png")) or \
                        sorted((page_dirs[0] / "classifier").glob("[0-9][0-9].png"))
                if cands:
                    real = Image.open(cands[len(cands) // 2]).convert("L")
        p = reproduction_panel(a, real)
        if p:
            out = IMAGES / f"repro_{a}.png"
            p.save(out)
            print(f"[repro] {a}: {out.name} ({p.width}x{p.height})")
        done.append(a)

    # contact sheets
    sheets = CNN_DIR / "SegmentationBest" / "GlyphSheets"
    for a in authors:
        src = sheets / f"{a}.png"
        if src.exists():
            Image.open(src).save(IMAGES / f"glyphs_{a}.png")
    print(f"\n[Glyphs] copied {len(authors)} glyph sheets")
    print(f"[Done] images in {IMAGES}")
    print(f"[Authors] {len(done)}: {', '.join(done)}")


if __name__ == "__main__":
    main()
