"""Stage 4: writer-ID (author classification) verification across all 13
authors, plus an independent numpy-vs-torch agreement check.

WHAT THIS MEASURES
------------------
Writer-ID accuracy is scored on REAL handwriting, not on synthesis:

  * personal authors (abhinav, dylan, owen, thiya, yeukita)
        -> the fresh crops Stage 1 wrote to
           SegmentationBest/Crops/<Author>/<Page>/classifier/
  * IAM dataset authors (150 151 152 153 384 551 552 588)
        -> the decoded line crops in NOGIT/line_cache_authors10, the exact
           images TrainAuthor.py trained on, read through IAMLineDatasetRaw so
           preprocessing is identical to training.

Both deployed heads are scored on the SAME lines:
  * ink    -- author_classifier_10new_weights.pt   (raw line)
  * shape  -- author_shape_10new_weights.pt        (stroke-normalised line;
              the model server.py / np_inference prefer by default)

THE AGREEMENT CHECK
-------------------
np_inference's premise is that a from-scratch numpy forward pass reproduces the
torch model exactly (the production path never imports torch). That is an
assumption worth testing rather than trusting, so this compares torch
AuthorClassifierCNN predictions against AuthorClassifierCNNNumpy predictions
on the same tensors and reports top-1 agreement plus max abs logit difference.
It deliberately imports torch, which np_inference does not.

MEMORY
------
An earlier version of this script held every PIL image for all ~1,500 lines at
once and the process died silently around 6 GB. This version keeps only a file
path or the compressed PNG bytes per line and materialises one image at a time,
reports progress every 100 lines, and flushes every print. Memory is bounded by
the model weights plus one line image.

Run from Software/CNN:
    python NOGIT/full_pipeline_run/verify_author_classification.py
"""
import csv
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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

from TrainAuthor import AuthorClassifierCNN, StrokeNormalize  # noqa: E402
from TrainText import (  # noqa: E402
    IAMLineDatasetRaw, _decode_png, resize_line_image_fixed, tensor_from_resized,
)
from authors_config import DATASET_AUTHORS, PERSONAL_AUTHORS  # noqa: E402
from np_inference.author_model import AuthorClassifierCNNNumpy  # noqa: E402


WEIGHTS_DIR = _cnn / "weights"
INK_WEIGHTS = WEIGHTS_DIR / "author_classifier_10new_weights.pt"
SHAPE_WEIGHTS = WEIGHTS_DIR / "author_shape_10new_weights.pt"
CROPS_ROOT = _cnn / "Crops"
CACHE_DIR = CNN_DIR / "NOGIT" / "line_cache_authors10"
DATA_DIR = _root / "Data" / "Datasets" / "IAMpages10"

T0 = time.time()


def say(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def collect_lines():
    """-> list of lightweight descriptors (no decoded images retained).

    Each descriptor is either {"path": <file>} or {"png": <bytes>} plus the
    author / source / page / gid labels. PNG bytes are the compressed form the
    cache already stores, so retention is cheap.
    """
    out = []
    for a in PERSONAL_AUTHORS:
        adir = CROPS_ROOT / a.capitalize()
        if not adir.exists():
            say(f"WARN no crops for {a}")
            continue
        for page_dir in sorted(d for d in adir.iterdir() if d.is_dir()):
            cdir = page_dir / "classifier"
            if not cdir.exists():
                cdir = page_dir
            for p in sorted(cdir.glob("[0-9][0-9].png")):
                out.append(dict(author=a, source="personal-photo",
                                gid=f"{a}/{page_dir.name}/{p.stem}",
                                page=page_dir.name, path=p))
    say(f"personal-photo lines: {sum(1 for d in out if d['source']=='personal-photo')}")

    base = IAMLineDatasetRaw(root_dir=str(DATA_DIR), cache_dir=str(CACHE_DIR))
    n_iam = 0
    for s in base.samples:
        a = s["page_key"].split("/")[0]
        if a not in DATASET_AUTHORS:
            continue
        out.append(dict(author=a, source="IAM-real-line",
                        gid=f"{a}/{s['page_key'].split('/')[1]}#{s['line_idx']}",
                        page=s["page_key"].split("/")[1], png=s["image_png"]))
        n_iam += 1
    say(f"IAM real-line lines: {n_iam}")
    del base
    gc.collect()
    say(f"total lines: {len(out)}")
    return out


def load_raw(d):
    if "path" in d:
        return Image.open(d["path"]).convert("L")
    return _decode_png(d["png"]).convert("L")


def main():
    device = torch.device("cpu")
    lines = collect_lines()

    say("loading torch models...")
    def load_torch(path):
        m = AuthorClassifierCNN(num_authors=13).to(device)
        sd = torch.load(path, map_location=device, weights_only=False)
        m.load_state_dict(sd["model_state_dict"])
        m.eval()
        return m, sd["author_mapping"]

    ink, mapping = load_torch(INK_WEIGHTS)
    shape, mapping_s = load_torch(SHAPE_WEIGHTS)
    assert mapping == mapping_s, "checkpoint class orders disagree"
    idx2author = {v: k for k, v in mapping.items()}
    authors_sorted = [idx2author[i] for i in range(len(idx2author))]
    say(f"class order: {authors_sorted}")

    say("loading numpy models...")
    ink_np = AuthorClassifierCNNNumpy(INK_WEIGHTS)
    shape_np = AuthorClassifierCNNNumpy(SHAPE_WEIGHTS)

    # -------------------------------------------------------------------------
    # Score both heads line by line
    # -------------------------------------------------------------------------
    # results[model][line_index] = (pred_idx, conf, probs_list)
    results = {"ink": {}, "shape": {}}
    agreement = {}
    for model_name, tmodel, nmodel, normalize in (
            ("ink", ink, ink_np, False),
            ("shape", shape, shape_np, True)):
        ckpt = THIS_DIR / f"stage4_predictions_{model_name}.json"
        ckpt_ag = THIS_DIR / f"stage4_agreement_{model_name}.json"
        if ckpt.exists() and ckpt_ag.exists():
            say(f"--- {model_name}: reusing {ckpt.name} "
                f"(delete it to force a re-score)")
            results[model_name] = {int(k): tuple(v) for k, v in
                                   json.loads(ckpt.read_text(encoding="utf-8")).items()}
            agreement[model_name] = json.loads(ckpt_ag.read_text(encoding="utf-8"))
            continue
        say(f"--- scoring {model_name} head ({'stroke-normalised' if normalize else 'raw'} input)")
        tl_all, nl_all, keep = [], [], []
        skipped = 0
        per = results[model_name]
        for i, d in enumerate(lines):
            pil = load_raw(d)
            if normalize:
                norm = StrokeNormalize(np.array(pil))
                if norm is None:
                    skipped += 1
                    if (i + 1) % 100 == 0:
                        say(f"  {model_name}: {i+1}/{len(lines)} lines "
                            f"(skipped {skipped}, kept {len(keep)})")
                    continue
                pil = norm
            resized = resize_line_image_fixed(pil)
            t = tensor_from_resized(resized)
            with torch.no_grad():
                tlog = tmodel(t.unsqueeze(0)).numpy()[0]
            nlog = nmodel.forward(t.numpy()[None, ...])[0]
            prob = torch.softmax(torch.tensor(nlog), dim=0).numpy()
            pidx = int(prob.argmax())
            per[i] = (pidx, float(prob[pidx]), prob.tolist())
            tl_all.append(tlog)
            nl_all.append(nlog)
            keep.append(i)
            if (i + 1) % 100 == 0:
                say(f"  {model_name}: {i+1}/{len(lines)} lines "
                    f"(skipped {skipped}, kept {len(keep)})")
            pil = None
        tl = np.array(tl_all)
        nl = np.array(nl_all)
        agree = float((tl.argmax(1) == nl.argmax(1)).mean()) if len(keep) else 0.0
        maxdiff = float(np.abs(tl - nl).max()) if len(keep) else 0.0
        agreement[model_name] = dict(n=len(keep), top1=agree, maxdiff=maxdiff, skipped=skipped)
        say(f"  {model_name}: kept {len(keep)}/{len(lines)} (skipped {skipped}) | "
            f"numpy-vs-torch top-1 {agree*100:.2f}% | max |logit diff| {maxdiff:.3e}")
        del tl_all, nl_all, tl, nl
        # Checkpoint the expensive per-line verdicts immediately. The scoring
        # pass costs ~45 minutes on this CPU; a crash in report aggregation
        # (which happened once) must not force a re-run to recover them.
        with open(THIS_DIR / f"stage4_predictions_{model_name}.json", "w",
                  encoding="utf-8") as f:
            json.dump({str(k): v for k, v in per.items()}, f)
        with open(THIS_DIR / f"stage4_agreement_{model_name}.json", "w",
                  encoding="utf-8") as f:
            json.dump(agreement[model_name], f, indent=2)
        say(f"  checkpointed {len(per)} verdicts -> stage4_predictions_{model_name}.json")
        gc.collect()

    # -------------------------------------------------------------------------
    # Per-author accuracy, confusion, confidence
    # -------------------------------------------------------------------------
    report = {}
    # defaultdict(int) was wrong here: confusion[m][a] must itself be a
    # defaultdict so that confusion[m][a][pred] += 1 can create a new key.
    confusion = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    csv_rows = []
    for model_name in ("ink", "shape"):
        per = results[model_name]
        by_author = defaultdict(lambda: [0, 0, []])
        for i, l in enumerate(lines):
            if i not in per:
                continue
            pidx, conf, _pv = per[i]
            pred = idx2author[pidx]
            ok = pred == l["author"]
            rec = by_author[l["author"]]
            rec[1] += 1
            rec[0] += int(ok)
            rec[2].append(conf)
            confusion[model_name][l["author"]][pred] += 1
            csv_rows.append((model_name, l["author"], l["source"], l["page"], l["gid"],
                             pred, f"{conf:.4f}", int(ok)))
        report[model_name] = {a: dict(correct=v[0], total=v[1], acc=v[0] / max(1, v[1]),
                                      mean_conf=float(np.mean(v[2])) if v[2] else 0.0)
                              for a, v in sorted(by_author.items())}

    print()
    print("=" * 92, flush=True)
    print("WRITER-ID ACCURACY ON REAL HANDWRITING (both deployed heads, same lines)", flush=True)
    print("=" * 92, flush=True)
    print(f"{'author':<10} {'source':<15}" +
          "".join(f"{n+' acc':>11} {n+' n':>7} {n+' conf':>10}" for n in ("ink", "shape")),
          flush=True)
    overall = {}
    for a in authors_sorted:
        src = "personal-photo" if a in PERSONAL_AUTHORS else "IAM-real-line"
        row = f"{a:<10} {src:<15}"
        for n in ("ink", "shape"):
            d = report[n].get(a)
            row += (f"{d['acc']*100:10.1f}% {d['total']:7d} {d['mean_conf']*100:9.1f}%"
                    if d else f"{'-':>11} {0:>7} {'-':>10}")
        print(row, flush=True)
    print("-" * 92, flush=True)
    for n in ("ink", "shape"):
        tot = sum(d["total"] for d in report[n].values())
        cor = sum(d["correct"] for d in report[n].values())
        overall[n] = dict(correct=cor, total=tot, acc=cor / max(1, tot))
        print(f"  {n:5s} overall: {cor}/{tot} = {cor/max(1,tot)*100:.1f}%", flush=True)

    for n in ("ink", "shape"):
        out = THIS_DIR / f"stage4_confusion_{n}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred"] + authors_sorted)
            for a in authors_sorted:
                w.writerow([a] + [confusion[n][a][b] for b in authors_sorted])
        say(f"confusion {n} -> {out.name}")

    out_csv = THIS_DIR / "stage4_per_line_predictions.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "true_author", "source", "page", "gid", "predicted",
                    "confidence", "correct"])
        w.writerows(csv_rows)
    say(f"per-line -> {out_csv.name}")

    summary = dict(
        n_lines=len(lines),
        agreement=[dict(model=k, n=v["n"], top1_agreement=v["top1"],
                        max_logit_diff=v["maxdiff"], skipped=v["skipped"])
                   for k, v in agreement.items()],
        per_author=report, overall=overall, index_order=authors_sorted,
        elapsed_seconds=round(time.time() - T0, 1),
    )
    with open(THIS_DIR / "stage4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    say("summary -> stage4_summary.json")
    say("DONE")


if __name__ == "__main__":
    main()
