"""ReproductionImprovement -- path setup.

CONVENTION FOR THIS FOLDER
--------------------------
Only files this work actually CHANGES live here. Everything else is imported
straight out of the original folders, so there is exactly one copy of every
unmodified module and no drift between the two.

    changed, so copied here :  BuildStyleProfile.py, Evaluate.py
    pulled, never copied    :  SynthesizeHandwriting.py, ProfileIO.py,
                               np_inference/, authors_config.py,
                               SegmentLean.py, SegmentPage.py, ExtractIAMLines.py

Every entry point in this folder starts with `import paths` so the pull works.

Importing this module also gives you REPO, CNN and DATA_DIR so nothing has to
recompute them.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # Software/CNN/ReproductionImprovement
CNN = HERE.parent                               # Software/CNN
ARS = CNN / "AuthorReproductionStuff"           # the original pipeline folder
REPO = CNN.parents[1]                           # repo root
DATA_DIR = REPO / "Data" / "Datasets" / "IAMpages10"

# `HERE` first so our changed copies win; then the original folders so every
# unmodified module resolves without being duplicated.
for _p in (HERE, ARS, CNN):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)
