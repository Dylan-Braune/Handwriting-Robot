"""
qtp1a_synthfeat.py -- hand-crafted features (qtp_features.line_features, the SAME extractor as
for the real lines) of every synthesised A-job line (10 writers x 6 novel sentences) ->
qtp_data/qtp1a_synth_features.json.  Also features of the B-job lines (matched-text synthesis).
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import io
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtp_common as C
import qtp_features as QF


def _one(path):
    with open(path, "rb") as f:
        r = pickle.load(f)
    g = np.array(Image.open(io.BytesIO(r["png"])).convert("L"))
    f = QF.line_features(g, r["text"].strip())
    return dict(name=r["name"], writer=r["writer"], text=r["text"], seed=r["seed"],
                seconds=r["seconds"], jointScore=r["meta"].get("jointScore"),
                jointTries=r["meta"].get("jointTries"),
                img_h=g.shape[0], img_w=g.shape[1], **(f or {}))


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    files = sorted((C.DATA_OUT / "synth").glob("*.pkl"))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_one, files, chunksize=4))
    A = [r for r in rows if r["name"].startswith("A_")]
    B = [r for r in rows if r["name"].startswith("B_")]
    print("A lines", len(A), "B lines", len(B))
    C.save_json(dict(A=A, B=B, features=QF.FEATURES), "qtp1a_synth_features.json")


if __name__ == "__main__":
    main()
