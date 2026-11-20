"""
qtp0_segment_cache.py -- run SegmentPage.ProcessPage ONCE on every personal
photo (and on the held-out IAM pages) and cache the TEXT crops, so QTP1/QTP2
do not each pay ~45 s per page.  NOT a timing measurement (several pages are
processed in parallel); QTP4 re-measures ProcessPage timing alone, serially.

Also reproduces the train/validation split of the personal lines exactly as
TrainText.collect_personal_lines_joint (text recogniser) and
TrainAuthor.add_personal_samples (writer classifier) do it.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtp_common as C


def _seg(path):
    import SegmentPage as PS
    t = time.perf_counter()
    results, _prev, _meta = PS.ProcessPage(str(path))
    dt = time.perf_counter() - t
    keep = [dict(order=r["order"], tag=r["tag"], bbox=r["bbox"], raw_crop=r["raw_crop"])
            for r in results]
    return str(path), keep, dt


def iam_holdout_pages():
    """Held-out page per IAM author exactly as IAMLineDatasetRaw: the LAST
    labelled page (sorted by file name) when the author has >= 3 labelled pages."""
    out = {}
    for a in C.DATASET_AUTHORS:
        pages = sorted((C.IAM_DIR / a).glob("*.png"))
        labelled = [p for p in pages if C.read_labels(p)]
        if len(labelled) >= 3:
            out[a] = labelled[-1]
    return out


def main():
    pages = C.personal_pages()
    iam = iam_holdout_pages()
    jobs = [p for _, p in pages] + list(iam.values())
    print(f"segmenting {len(jobs)} pages ({len(pages)} personal + {len(iam)} IAM held-out) "
          f"with 4 worker processes", flush=True)
    t0 = time.time()
    seg = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for path, keep, dt in ex.map(_seg, jobs):
            seg[path] = keep
            print(f"  {os.path.basename(path)}: {sum(1 for r in keep if r['tag']=='TEXT')} TEXT / "
                  f"{len(keep)} regions ({dt:.0f} s, parallel)", flush=True)
    print("segmentation wall time %.0f s" % (time.time() - t0))

    # ---- reproduce the splits -------------------------------------------
    from np_inference.text_model import CHAR_TO_IDX
    MAX_SAFE = None
    try:
        import re
        src = open(C.CNN / "TrainText.py", encoding="utf-8").read()
        m = re.search(r"^MAX_SAFE_LABEL_CHARS\s*=\s*(\d+)", src, re.M)
        MAX_SAFE = int(m.group(1)) if m else None
    except Exception:
        pass
    from authors_config import VAL_FRACTION, SPLIT_SEED

    page_rows = []   # per page: dict(author,page,crops,labels,n_crops,n_labels)
    for author, p in pages:
        res = seg[str(p)]
        crops = [r["raw_crop"] for r in res if r["tag"] == "TEXT"]
        gt = [g for g in C.read_labels(p) if g.strip() != "MESS"]
        n = min(len(crops), len(gt))
        page_rows.append(dict(author=author, page=p.name, crops=crops[:n], labels=gt[:n],
                              n_crops=len(crops), n_labels=len(gt), n_label_lines_raw=len(C.read_labels(p)),
                              n_regions=len(res)))

    # text-recogniser split (TrainText.collect_personal_lines_joint): rows are first
    # FILTERED (chars in charset, 0 < len <= MAX_SAFE), then shuffled per page with ONE
    # shared Random(SPLIT_SEED); val = first max(1, round(0.15*n)).
    rng = random.Random(SPLIT_SEED)
    for pr in page_rows:
        kept_idx = []
        for i, t in enumerate(pr["labels"]):
            kept = "".join(c for c in t.strip() if c in CHAR_TO_IDX)
            if 0 < len(kept) <= (MAX_SAFE or 10**9):
                kept_idx.append(i)
        order = list(kept_idx)
        rng.shuffle(order)
        n_val = max(1, round(len(order) * VAL_FRACTION))
        pr["text_val_idx"] = sorted(order[:n_val])
        pr["text_kept_idx"] = kept_idx

    # writer-classifier split (TrainAuthor.add_personal_samples): rows = ALL n paired
    # lines (unfiltered), shuffled per page with one shared Random(SPLIT_SEED);
    # holdout = first max(1, round(0.15*n)); empty-target lines then skipped.
    rng = random.Random(SPLIT_SEED)
    for pr in page_rows:
        n = len(pr["labels"])
        order = list(range(n))
        rng.shuffle(order)
        n_val = max(1, round(n * VAL_FRACTION))
        pr["author_val_idx"] = sorted(order[:n_val])

    for pr in page_rows:
        same = pr["text_val_idx"] == pr["author_val_idx"]
        print(f"  {pr['author']}/{pr['page'][:40]:40s} crops={pr['n_crops']} labels={pr['n_labels']} "
              f"text_val={pr['text_val_idx']} author_val={pr['author_val_idx']} same={same}")

    iam_rows = {}
    for a, p in iam.items():
        res = seg[str(p)]
        crops = [r["raw_crop"] for r in res if r["tag"] == "TEXT"]
        labels = C.read_labels(p)
        iam_rows[a] = dict(page=p.name, crops=crops, labels=labels, n_regions=len(res))
        print(f"  IAM {a}/{p.name}: {len(crops)} TEXT crops, {len(labels)} label lines")

    with open(C.CACHE / "seg_cache.pkl", "wb") as f:
        pickle.dump(dict(personal=page_rows, iam=iam_rows, max_safe=MAX_SAFE,
                         val_fraction=VAL_FRACTION, split_seed=SPLIT_SEED), f)
    print("saved", C.CACHE / "seg_cache.pkl")


if __name__ == "__main__":
    main()
