"""Path + shared-model setup for the MyHandwriting methods."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CNN_DIR = HERE.parent                       # Software/CNN
REPO = CNN_DIR.parents[1]                    # repo root
for p in (str(CNN_DIR), str(CNN_DIR / "AuthorReproductionStuff")):
    if p not in sys.path:
        sys.path.insert(0, p)

PAGES_DIR = HERE / "pages"
OUT_DIR = HERE / "out"
LIB_DIR = HERE / "lib"
NOGIT_DIR = CNN_DIR / "NOGIT"            # also consumed by diffbrush_infer / onedm_infer
NOGIT_WEIGHTS = NOGIT_DIR / "weights"
TEXT_WEIGHTS = NOGIT_WEIGHTS / "paper_cnn_bilstm_ctc_best.pt"
# use the HF-trained recogniser if the student dropped it in
_hf = NOGIT_WEIGHTS / "paper_cnn_bilstm_ctc_hf_best.pt"
if _hf.exists():
    TEXT_WEIGHTS = _hf

for d in (OUT_DIR, LIB_DIR):
    d.mkdir(exist_ok=True)
