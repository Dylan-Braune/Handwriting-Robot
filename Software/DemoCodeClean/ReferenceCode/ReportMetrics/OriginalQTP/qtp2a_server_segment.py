"""
qtp2a_server_segment.py -- segment every personal photo and every held-out IAM page the way
server.py /api/classify does it: open image -> convert('RGB') -> np.array(convert('L')) ->
save grey PNG -> SegmentPage.ProcessPage(png) -> keep tag == 'TEXT' crops (r['raw_crop']).
(The training scripts call ProcessPage on the colour .jpg directly; ProcessPage has a
colour-aware ink-recovery stage, so the two paths may differ -- QTP2 reports both.)
Not a timing measurement (runs next to other jobs).  Results cached for qtp2_classify.py.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import pickle
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtp_common as C
from qtp0_segment_cache import iam_holdout_pages


def server_segment(path):
    import SegmentPage as PS
    pil = Image.open(path).convert("RGB")
    gray = np.array(pil.convert("L"))
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "_classify_tmp.png")
        Image.fromarray(gray).save(tmp)
        t = time.perf_counter()
        results, _prev, _meta = PS.ProcessPage(tmp)
        dt = time.perf_counter() - t
    crops = [r["raw_crop"] for r in results if r.get("tag") == "TEXT"]
    return str(path), crops, dt, pil.size


def main():
    jobs = [p for _, p in C.personal_pages()] + list(iam_holdout_pages().values())
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"server-path segmentation of {len(jobs)} pages, {workers} worker(s)", flush=True)
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for path, crops, dt, size in ex.map(server_segment, jobs):
            out[path] = dict(crops=crops, seconds_contended=dt, size=size)
            print(f"  {os.path.basename(path)}: {len(crops)} TEXT crops ({dt:.0f} s, contended)", flush=True)
    with open(C.CACHE / "seg_cache_server.pkl", "wb") as f:
        pickle.dump(out, f)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
