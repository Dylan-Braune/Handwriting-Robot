"""
qtp4b_c_accel.py -- OPTIONAL: re-measure the effect of the C-accelerated LabelComponents / BoxSum
(C_Acceleration/segmentation, README claims ProcessPage ~25-32 % faster) on the dev laptop.

The C_Acceleration folder itself is NOT touched: accel.c and accel_py.py are COPIED into
qtp_scripts/c_accel_build/ , compiled there with the MinGW gcc that build_windows.bat points to,
and ProcessPage is run with SegmentPage.LabelComponents / SegmentPage.BoxSum monkey-patched
in this process only (as C_Acceleration/segmentation/pipeline_sanity.py does).  Baseline and
accelerated runs alternate (3 each) on one photo; outputs are compared for identity.
Laptop numbers, NOT the ODROID.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtp_common as C
import SegmentPage as SP

SRC = C.REPO / "C_Acceleration" / "segmentation"
BUILD = Path = None
GCC = r"C:\Users\braun\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin\gcc.exe"
PHOTO = C.NOGIT / "dylan" / "preprocessing_page_localization_illumination_binarisation.jpg"
REPS = 3


def signature(results):
    return tuple((r.get("tag"), tuple(round(float(v), 1) for v in r.get("bbox", ()))) for r in results)


def main():
    build = C.FINAL / "qtp_scripts" / "c_accel_build"
    build.mkdir(exist_ok=True)
    for f in ("accel.c", "accel_py.py"):
        shutil.copy(SRC / f, build / f)
    out = dict(machine=C.machine_info(), photo=PHOTO.name, repeats=REPS)
    r = subprocess.run([GCC, "-O3", "-shared", "-o", str(build / "libaccel.dll"), str(build / "accel.c"), "-lm"],
                       capture_output=True, text=True)
    out["build"] = dict(returncode=r.returncode, stderr=r.stderr[-500:])
    if r.returncode != 0 or not (build / "libaccel.dll").exists():
        out["status"] = "BUILD FAILED"
        C.save_json(out, "qtp4b_c_accel.json")
        print("build failed", r.stderr)
        return
    sys.path.insert(0, str(build))
    import accel_py as A
    out["available"] = A.AVAILABLE
    pil = Image.open(PHOTO).convert("RGB")
    tmp = os.path.join(tempfile.mkdtemp(), "page.png")
    Image.fromarray(np.array(pil.convert("L"))).save(tmp)
    orig_label, orig_box = SP.LabelComponents, SP.BoxSum
    base_t, acc_t = [], []
    sig_b = sig_a = None
    for i in range(REPS):
        t = time.perf_counter(); res, _, _ = SP.ProcessPage(tmp); base_t.append(time.perf_counter() - t)
        sig_b = signature(res)
        SP.LabelComponents, SP.BoxSum = A.label_components, A.box_sum
        try:
            t = time.perf_counter(); res, _, _ = SP.ProcessPage(tmp); acc_t.append(time.perf_counter() - t)
            sig_a = signature(res)
        finally:
            SP.LabelComponents, SP.BoxSum = orig_label, orig_box
        print(f"rep{i}: baseline {base_t[-1]:.1f}s  accelerated {acc_t[-1]:.1f}s", flush=True)
    mb, ma = float(np.mean(base_t)), float(np.mean(acc_t))
    out.update(status="ok", baseline_s=dict(mean=mb, min=min(base_t), max=max(base_t), values=base_t),
               accelerated_s=dict(mean=ma, min=min(acc_t), max=max(acc_t), values=acc_t),
               improvement_percent_of_baseline=100 * (1 - ma / mb), identical_output=(sig_a == sig_b),
               n_items=len(sig_b))
    C.save_json(out, "qtp4b_c_accel.json")
    print(out["improvement_percent_of_baseline"], "% faster; identical output:", sig_a == sig_b)


if __name__ == "__main__":
    main()
