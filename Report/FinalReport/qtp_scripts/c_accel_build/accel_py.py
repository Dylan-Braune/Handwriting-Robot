"""
accel_py.py -- ctypes wrapper around the compiled accel.c acceleration
library (libaccel.so on Linux/ODROID, libaccel.dll on Windows).

This module is purely ADDITIVE: nothing in SegmentPage.py imports it, and
nothing here modifies any existing file. A caller can opt in explicitly,
e.g.:

    import accel.accel_py as accel
    if accel.AVAILABLE['label_components']:
        SegmentPage.LabelComponents = accel.label_components

Public API (mirrors the numeric contract of SegmentPage.py's pure-Python
primitives -- originally RawImageOps.py, since merged into SegmentPage.py):

    label_components(mask) -> (labels: np.ndarray[int32], count: int)
        mask: 2D array, bool or uint8, 0/1 (any truthy/falsy values are
              treated as 0/1). Returns a SAME-PARTITION labeling (labels
              1..count, 0 = background) -- not necessarily the same label
              NUMBERING as SegmentPage.LabelComponents, but the same
              grouping of pixels into components.

    box_sum(img, ry, rx) -> np.ndarray[float64]
        img: 2D float64 (or castable) array. Numerically matches
             SegmentPage.BoxSum(img, ry, rx) to float64 precision.

If the compiled library cannot be found/loaded, both functions raise
RuntimeError when called, and AVAILABLE reports False for everything, so
calling code can check before opting in.
"""

import ctypes
import os
import platform

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

AVAILABLE = {
    "label_components": False,
    "box_sum": False,
}

_lib = None
_load_error = None


def _candidate_lib_names():
    if os.name == "nt":
        return ["libaccel.dll", "accel.dll"]
    return ["libaccel.so", "accel.so"]


def _load_library():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return
    names = _candidate_lib_names()
    tried = []
    for name in names:
        path = os.path.join(_THIS_DIR, name)
        tried.append(path)
        if os.path.isfile(path):
            try:
                _lib = ctypes.CDLL(path)
                break
            except OSError as e:
                _load_error = f"found {path} but failed to load it: {e}"
                return
    if _lib is None:
        _load_error = (
            "could not find a compiled accel library. Tried: "
            + ", ".join(tried)
            + f" (platform={platform.system()}). Build it first -- see "
              "build_windows.bat / build_linux.sh in this directory."
        )
        return

    # box_sum(const double *img, int h, int w, int ry, int rx, double *out)
    try:
        _lib.box_sum.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        _lib.box_sum.restype = None
        AVAILABLE["box_sum"] = True
    except AttributeError:
        pass

    # label_components(const uint8_t *mask, int h, int w, int connectivity,
    #                   int32_t *out_labels) -> int32_t
    try:
        _lib.label_components.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
        ]
        _lib.label_components.restype = ctypes.c_int32
        AVAILABLE["label_components"] = True
    except AttributeError:
        pass


_load_library()


def _require(name):
    if not AVAILABLE.get(name, False):
        msg = f"accel.{name} is not available"
        if _load_error:
            msg += f": {_load_error}"
        raise RuntimeError(msg)


def box_sum(img, ry, rx):
    """C-accelerated equivalent of SegmentPage.BoxSum(img, ry, rx).

    img must be 2D and is cast to float64 (contiguous) before the call.
    Returns a float64 array of the same shape as img.
    """
    _require("box_sum")
    arr = np.ascontiguousarray(img, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("box_sum: img must be 2D")
    h, w = arr.shape
    out = np.empty((h, w), dtype=np.float64)
    _lib.box_sum(
        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(h),
        ctypes.c_int(w),
        ctypes.c_int(int(ry)),
        ctypes.c_int(int(rx)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return out


def label_components(mask, connectivity=8):
    """C-accelerated equivalent of SegmentPage.LabelComponents(mask,
    connectivity). mask must be 2D, any dtype (cast to uint8 0/1).

    Returns (labels: int32 ndarray same shape as mask, count: int).

    NOTE: label NUMBERING is not guaranteed to match the pure-Python
    version (it depends on scan order internally) -- only the PARTITION
    of foreground pixels into components and the total count match.
    """
    _require("label_components")
    arr = np.ascontiguousarray(mask, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("label_components: mask must be 2D")
    if connectivity not in (4, 8):
        raise ValueError("label_components: connectivity must be 4 or 8")
    h, w = arr.shape
    out = np.empty((h, w), dtype=np.int32)
    count = _lib.label_components(
        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int(h),
        ctypes.c_int(w),
        ctypes.c_int(connectivity),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    if count < 0:
        raise RuntimeError("label_components: C allocation failure")
    return out, int(count)
