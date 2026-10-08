"""
qtp_common.py -- shared helpers for the qualification test (QTP) scripts.

Nothing in the existing pipeline is modified; this module only imports it.
All QTP scripts run with any working directory; sys.path is set up here the
same way Evaluate.py / server.py do it.
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CNN = REPO / "Software" / "CNN"
ARS = CNN / "AuthorReproductionStuff"
GANTRY_DIR = REPO / "Software" / "GantryControl"
NOGIT = CNN / "NOGIT"
WEIGHTS = CNN / "weights"
PROFILE_DIR = NOGIT / "StyleProfiles10"
IAM_DIR = REPO / "Data" / "Datasets" / "IAMpages10"
IAM_CACHE = NOGIT / "line_cache_authors10"
FINAL = REPO / "Report" / "FinalReport"
DATA_OUT = FINAL / "qtp_data"
CACHE = NOGIT / "qtp_cache"          # heavy intermediates (pickles of crops/PNGs), not part of the deliverable
CACHE.mkdir(parents=True, exist_ok=True)
FIG_OUT = FINAL / "Figures"
DATA_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

for p in (str(ARS), str(CNN), str(GANTRY_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

PERSONAL_AUTHORS = ["yeukita", "dylan"]          # same order as authors_config.py
DATASET_AUTHORS = ["150", "151", "152", "153", "384", "551", "552", "588"]
ALL_AUTHORS = sorted(DATASET_AUTHORS + PERSONAL_AUTHORS)

HF_CKPT = WEIGHTS / "paper_cnn_bilstm_ctc_hf_best.pt"
JOINT_CKPT = WEIGHTS / "paper_cnn_bilstm_ctc_joint_best.pt"


def personal_pages():
    """[(author, path)] in the exact order used by TrainText/TrainAuthor."""
    out = []
    for a in PERSONAL_AUTHORS:
        for p in sorted((NOGIT / a).glob("*.jpg")):
            out.append((a, p))
    return out


def read_labels(img_path):
    """Same as ExtractIAMLines.ReadLabelLines but without importing pytesseract."""
    import re
    lp = os.path.splitext(str(img_path))[0] + "_labels.txt"
    if not os.path.exists(lp):
        return []
    with open(lp, "r", encoding="utf-8") as f:
        raw = [l.strip() for l in f.readlines() if l.strip()]
    return [re.sub(r'^\d+\t', '', l) for l in raw]


def machine_info():
    import numpy as np
    info = dict(platform=platform.platform(), processor=platform.processor(),
                cpu_count=os.cpu_count(), python=sys.version.split()[0],
                numpy=np.__version__,
                label="dev laptop, NOT ODROID N2+ (timings are laptop timings)")
    try:
        import subprocess
        out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True,
                             text=True, timeout=15).stdout.split("\n")
        info["cpu_name"] = [l.strip() for l in out if l.strip() and l.strip() != "Name"][0]
    except Exception:
        pass
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        info[k] = os.environ.get(k)
    return info


def save_json(obj, name):
    p = DATA_OUT / name
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return p


def load_json(name):
    with open(DATA_OUT / name, encoding="utf-8") as f:
        return json.load(f)


def mean_min_max(xs):
    import numpy as np
    xs = list(xs)
    return dict(mean=float(np.mean(xs)), min=float(np.min(xs)), max=float(np.max(xs)),
                n=len(xs), values=[float(x) for x in xs])


# --------------------------------------------------------------------------
# matplotlib house style: white background, sans-serif, >= 9 pt, 300 dpi,
# width <= 16 cm
# --------------------------------------------------------------------------
CM = 1 / 2.54
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def mpl_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 300, "figure.dpi": 100,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    })
    return plt


def savefig(fig, name):
    p = FIG_OUT / name
    fig.savefig(p, dpi=300, bbox_inches="tight")
    return p
