"""
Path setup for the ReproduceV2 package.

AuthorReproductionStuff/ was moved out of Software/CNN/ but its scripts still
resolve `NOGIT/` and imports relative to the old location. This module fixes
that for everything under ReproduceV2/ -- import it FIRST:

    from _env import CNN_DIR, NOGIT_DIR, WEIGHTS_DIR, PROFILE_DIR

It puts both Software/CNN/ and Software/CNN/AuthorReproductionStuff/ on
sys.path (so `import TrainText` and `import SynthesizeHandwriting` both work)
and points the NOGIT paths at Software/CNN/NOGIT/ regardless of where this
file is run from.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent                 # .../AuthorReproductionStuff/ReproduceV2
REPRO_DIR = _HERE
AUTHOR_STUFF_DIR = _HERE.parent                          # .../AuthorReproductionStuff
CNN_DIR = _HERE.parents[1]                               # .../Software/CNN

for p in (str(CNN_DIR), str(AUTHOR_STUFF_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

NOGIT_DIR = CNN_DIR / "NOGIT"
WEIGHTS_DIR = NOGIT_DIR / "weights"
PROFILE_DIR = NOGIT_DIR / "StyleProfiles10"
DATA_DIR = CNN_DIR.parents[1] / "Data" / "Datasets" / "IAMpages10"
OUT_DIR = NOGIT_DIR / "ReproduceV2"

TEXT_WEIGHTS = WEIGHTS_DIR / "paper_cnn_bilstm_ctc_best.pt"
SHAPE_WEIGHTS = WEIGHTS_DIR / "author_shape_10_weights.pt"

AUTHORS = ["150", "151", "152", "153", "154", "155", "384", "551", "552", "588"]


def _repin_legacy_paths():
    """BuildStyleProfile / TrainAuthorShape compute their NOGIT + Data paths
    relative to AuthorReproductionStuff/ (stale after the folder move), which
    makes them rebuild the 137 MB line cache in the wrong place. Force the
    canonical Software/CNN/NOGIT locations."""
    try:
        import BuildStyleProfile as _SP
        _SP.SCRIPT_DIR = CNN_DIR
        _SP.DATA_DIR = DATA_DIR
        _SP.CACHE_DIR = NOGIT_DIR / "line_cache_authors10"
        _SP.PROFILE_DIR = PROFILE_DIR
        _SP.TEXT_WEIGHTS = TEXT_WEIGHTS
    except Exception:
        pass


_repin_legacy_paths()
