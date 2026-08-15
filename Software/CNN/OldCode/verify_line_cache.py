"""
Sanity-checks your line data two ways: (1) that each page's segmented line
count matches its *_labels.txt, and (2) that the cached (image, label) pairs
in line_cache_raw/ (built by train_paper_cnn_bilstm_ctc.py) actually look
like they go together.

CHECK 1 -- page-level line count (deterministic, no OCR, reliable):
For every page under your data dir, re-runs FullLineBoxMaker.ExtractLinePatches
and compares the number of detected lines against the number of lines in that
page's *_labels.txt. A mismatch here means the segmentation heuristic split
or merged lines differently than the label file expects for that page --
this is the same check verify_all_authors.py does.

CHECK 2 -- cached-sample spot check (OCR-assisted, best-effort):
For a sample of cached (image, label) pairs, this:
  a) Measures ink density (fraction of dark pixels). A crop that's almost
     entirely blank, or almost entirely dark, is a reliable, OCR-independent
     sign something went wrong (empty region grabbed, or a rule line /
     margin got included instead of handwriting).
  b) Runs Tesseract OCR on the crop and compares it to the stored label
     text with a normalized edit-distance similarity score.

     IMPORTANT: Tesseract is a printed-text OCR engine. It was not built for
     cursive handwriting, and will score genuinely CORRECT pairs poorly --
     don't read the average similarity as a data-quality percentage. It's
     useful as a coarse filter (e.g. catching an obvious off-by-one where
     the image and label are for entirely different words), not as proof
     of correctness on its own.

Either way, the script writes its most-suspicious findings (blank crops,
over-inked crops, lowest text-similarity crops) out to a small review
folder as actual PNG images next to their expected label text, so you can
open a handful and visually confirm the pairing yourself -- that visual
check is the real verification, everything else here is just triage to
help you find which handful to look at.

Usage:
    python verify_line_cache.py                       # full page check + OCR-sample 500 cached lines
    python verify_line_cache.py --ocr-samples 2000     # check more cached lines
    python verify_line_cache.py --ocr-samples 999999   # check every cached line (slow)
    python verify_line_cache.py --skip-ocr             # only the fast, reliable page-count check
    python verify_line_cache.py --skip-page-check       # only the cached-sample spot check
"""

import argparse
import io
import os
import random
from pathlib import Path

import numpy as np
import pytesseract
import torch
from PIL import Image

from FullLineBoxMaker import ExtractLinePatches, ReadLabelLines

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[1] / "Data" / "Datasets" / "IAMpages671"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "line_cache_raw"
DEFAULT_REVIEW_DIR = SCRIPT_DIR / "cache_verification_review"


def levenshtein(a, b):
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    previous = list(range(len(b) + 1))
    for ca in a:
        current = [previous[0] + 1]
        for j, cb in enumerate(b):
            current.append(min(current[j] + 1, previous[j + 1] + 1, previous[j] + (ca != cb)))
        previous = current
    return previous[-1]


def similarity(a, b):
    if not a and not b:
        return 1.0
    dist = levenshtein(a, b)
    return 1.0 - dist / max(1, max(len(a), len(b)))


def resolve_dataset_root(root_dir):
    data_dir = root_dir / "data"
    return data_dir if data_dir.exists() and any(p.is_dir() for p in data_dir.iterdir()) else root_dir


# -----------------------------------------------------------------------------
# Check 1: page-level line count.
# -----------------------------------------------------------------------------
def check_page_line_counts(data_dir):
    root_dir = resolve_dataset_root(Path(data_dir))
    author_folders = sorted(p.name for p in root_dir.iterdir() if p.is_dir())

    total, matched = 0, 0
    mismatches = []

    print(f"[Page check] Scanning {len(author_folders)} author folder(s) under {root_dir}...")
    for a_idx, author_id in enumerate(author_folders, 1):
        author_dir = root_dir / author_id
        for img_path in sorted(author_dir.glob("*.png")):
            label_lines = ReadLabelLines(str(img_path))
            if not label_lines:
                continue
            expected = len(label_lines)
            try:
                samples, _texts, _ = ExtractLinePatches(
                    str(img_path), expectedLineCount=expected, labelLines=label_lines
                )
            except Exception as e:
                mismatches.append((author_id, img_path.name, expected, f"ERROR: {e}"))
                total += 1
                continue

            detected = len(samples)
            total += 1
            if detected == expected:
                matched += 1
            else:
                mismatches.append((author_id, img_path.name, expected, detected))

        if a_idx % 100 == 0:
            print(f"  ... {a_idx}/{len(author_folders)} authors checked ({matched}/{total} pages matched so far)")

    print(f"[Page check] {matched}/{total} pages have exactly the expected number of lines "
          f"({100 * matched / max(1, total):.2f}%).")
    if mismatches:
        print(f"[Page check] {len(mismatches)} page(s) with a count mismatch:")
        for author_id, name, expected, detected in mismatches[:40]:
            print(f"    {author_id}/{name}: expected {expected}, got {detected}")
        if len(mismatches) > 40:
            print(f"    ... and {len(mismatches) - 40} more.")
    return total, matched, mismatches


# -----------------------------------------------------------------------------
# Check 2: OCR-assisted spot check on the cached (image, label) pairs.
# -----------------------------------------------------------------------------
def load_cached_samples(cache_dir):
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.pt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path} -- has line_cache_raw/ been built yet?")

    manifest = torch.load(manifest_path, weights_only=False)
    samples = []
    for shard_name in manifest.get("shard_files", []):
        shard_path = cache_dir / "shards" / shard_name
        if shard_path.exists():
            samples.extend(torch.load(shard_path, weights_only=False))
    return samples


def ink_ratio(pil_img, dark_threshold=200):
    arr = np.array(pil_img.convert("L"), dtype=np.uint8)
    return float(np.mean(arr < dark_threshold))


def check_cached_samples(cache_dir, n_samples, seed=7):
    samples = load_cached_samples(cache_dir)
    print(f"[Sample check] Loaded {len(samples)} cached samples from {cache_dir}.")

    rng = random.Random(seed)
    chosen = samples if n_samples >= len(samples) else rng.sample(samples, n_samples)
    print(f"[Sample check] OCR-checking {len(chosen)} of them (this is the slow part)...")

    results = []
    for i, sample in enumerate(chosen, 1):
        pil_img = Image.open(io.BytesIO(sample["image_png"])).convert("L")
        expected_text = sample["text"]

        density = ink_ratio(pil_img)
        try:
            ocr_text = pytesseract.image_to_string(pil_img, config="--psm 7").strip()
        except Exception as e:
            ocr_text = f"<OCR ERROR: {e}>"

        sim = similarity(ocr_text.lower(), expected_text.lower())
        results.append(
            {
                "expected": expected_text,
                "ocr": ocr_text,
                "similarity": sim,
                "ink_ratio": density,
                "image": pil_img,
            }
        )

        if i % 50 == 0 or i == len(chosen):
            print(f"\r[Sample check] {i}/{len(chosen)} checked", end="", flush=True)
    print()

    return results


def summarize_and_export(results, review_dir, blank_threshold=0.002, full_threshold=0.6, worst_n=40):
    sims = [r["similarity"] for r in results]
    avg_sim = sum(sims) / max(1, len(sims))
    print(f"\n[Sample check] Average OCR/label text similarity: {avg_sim:.2%} "
          f"-- expected to be LOW even for correct pairs (Tesseract can't really read cursive "
          f"handwriting). Don't treat this number as a data-quality score.")

    blank = [r for r in results if r["ink_ratio"] < blank_threshold]
    too_full = [r for r in results if r["ink_ratio"] > full_threshold]
    blank_ids = {id(r) for r in blank}
    full_ids = {id(r) for r in too_full}

    print(f"[Sample check] {len(blank)} sample(s) look suspiciously BLANK (ink ratio < {blank_threshold:.1%}) "
          f"-- likely an empty/mis-cropped region.")
    print(f"[Sample check] {len(too_full)} sample(s) look suspiciously OVER-INKED (ink ratio > {full_threshold:.0%}) "
          f"-- likely grabbed a rule line, margin, or the wrong region entirely.")

    review_dir = Path(review_dir)
    review_dir.mkdir(exist_ok=True)
    for old_file in review_dir.glob("*"):
        old_file.unlink()

    remaining = [r for r in results if id(r) not in blank_ids and id(r) not in full_ids]
    worst_text_matches = sorted(remaining, key=lambda r: r["similarity"])[:worst_n]

    to_export = blank + too_full + worst_text_matches
    print(f"\n[Sample check] Writing {len(to_export)} flagged sample(s) to {review_dir} "
          f"(image + a .txt with the expected label and Tesseract's guess).")
    for idx, r in enumerate(to_export):
        if id(r) in blank_ids:
            tag = "blank"
        elif id(r) in full_ids:
            tag = "overinked"
        else:
            tag = "lowsim"
        base = review_dir / f"{idx:03d}_{tag}"
        r["image"].save(base.with_suffix(".png"))
        with open(base.with_suffix(".txt"), "w", encoding="utf-8") as f:
            f.write(f"expected:   {r['expected']}\n")
            f.write(f"ocr guess:  {r['ocr']}\n")
            f.write(f"similarity: {r['similarity']:.2%}\n")
            f.write(f"ink_ratio:  {r['ink_ratio']:.4f}\n")

    print(f"[Sample check] Open a few files in {review_dir} and confirm the image actually shows "
          f"the expected handwritten text -- that visual check is the real verification here.")


def main():
    parser = argparse.ArgumentParser(description="Verify line_cache_raw/ against your page labels.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--ocr-samples", type=int, default=500,
                         help="How many cached lines to OCR-check. Use a small number for a quick look, "
                              "or a number >= your dataset size to check everything.")
    parser.add_argument("--skip-ocr", action="store_true", help="Only run the fast, deterministic page-count check.")
    parser.add_argument("--skip-page-check", action="store_true", help="Only run the OCR-assisted cache spot check.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    args = parser.parse_args()

    if not args.skip_page_check:
        check_page_line_counts(args.data_dir)
        print()

    if not args.skip_ocr:
        results = check_cached_samples(args.cache_dir, args.ocr_samples)
        summarize_and_export(results, args.review_dir)


if __name__ == "__main__":
    main()
