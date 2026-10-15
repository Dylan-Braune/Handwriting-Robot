"""
qtp1a_realfeat.py -- hand-crafted features (qtp_features.line_features) of EVERY real line
(1043 lines: 689 IAM cached lines of the 8 IAM writers + 354 personal lines), cached to
qtp_data/qtp1a_real_features.json.  Each row records the writer, page, line index, text,
and the three held-out flags (see qtp_lines.py).
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtp_common as C
import qtp_features as QF
import qtp_lines as L

_LINES = None


def _init(lines):
    global _LINES
    _LINES = lines


def _one(i):
    l = _LINES[i]
    f = QF.line_features(l["gray"], l["text"].strip())
    return i, f


def main():
    lines = L.load_real_lines()
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    rows = [None] * len(lines)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(lines,)) as ex:
        for k, (i, f) in enumerate(ex.map(_one, range(len(lines)), chunksize=8)):
            rows[i] = f
            if k % 100 == 0:
                print(k, len(lines), f"{time.time()-t0:.0f}s", flush=True)
    out = []
    n_fail = 0
    for l, f in zip(lines, rows):
        if f is None:
            n_fail += 1
            continue
        out.append(dict(writer=l["writer"], kind=l["kind"], page=l["page"], line_idx=l["line_idx"],
                        text=l["text"], heldout_profile=l["heldout_profile"],
                        heldout_text=l["heldout_text"], heldout_author=l["heldout_author"], **f))
    print("lines measured", len(out), "failed", n_fail)
    C.save_json(dict(rows=out, n_failed=n_fail, features=QF.FEATURES, target_xh_px=QF.TARGET_XH),
                "qtp1a_real_features.json")


if __name__ == "__main__":
    main()
