"""Opt-in sanity check: monkey-patch SegmentPage.LabelComponents/BoxSum with
the C versions and confirm ProcessPage produces identical output on a real
photo. Run from repo root: python C_Acceleration/segmentation/pipeline_sanity.py"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Software" / "CNN"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import SegmentPage as SP
import accel_py as A

PHOTO = Path(__file__).resolve().parent.parent.parent / "Software" / "CNN" / "NOGIT" / "dylan" / \
    "preprocessing_page_localization_illumination_binarisation.jpg"


def signature(results):
    return tuple((r.get("tag"), tuple(round(v, 1) for v in r.get("bbox", ()))) for r in results)


def run():
    t0 = time.time()
    baseline, _, _ = SP.ProcessPage(str(PHOTO))
    t_py = time.time() - t0
    sig_py = signature(baseline)

    orig_label, orig_box = SP.LabelComponents, SP.BoxSum
    SP.LabelComponents = A.label_components
    SP.BoxSum = A.box_sum
    try:
        t0 = time.time()
        accelerated, _, _ = SP.ProcessPage(str(PHOTO))
        t_c = time.time() - t0
    finally:
        SP.LabelComponents, SP.BoxSum = orig_label, orig_box

    sig_c = signature(accelerated)
    assert sig_py == sig_c, "ProcessPage output changed with accel functions active"
    print(f"ProcessPage output identical: {len(baseline)} items, Python {t_py:.1f}s, "
          f"with accel {t_c:.1f}s ({(1 - t_c / t_py) * 100:.1f}% faster)")


if __name__ == "__main__":
    run()
