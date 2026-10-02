"""ctypes wrapper around accel.c's compiled conv2d_forward / lstm_forward.

This module is purely additive: nothing in the existing codebase imports
it, and it does not modify np_inference/layers.py. It exposes drop-in
replacements with the SAME signature/return shape as the pure-numpy
functions in np_inference/layers.py, so a caller can opt in explicitly
(see README.md) without touching this directory's own files or the
production files.

Library selection: picks `libaccel.dll` on Windows and `libaccel.so` on
Linux (same accel.c source, built separately on each platform -- see
build_windows.bat / build_linux.sh). If the compiled library is missing
or fails to load, AVAILABLE["conv2d"] / AVAILABLE["lstm"] are False and
calling the wrapped functions raises RuntimeError rather than silently
falling back, so a caller always knows which path actually ran.
"""

import ctypes
import platform
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

AVAILABLE = {"conv2d": False, "lstm": False}

_lib = None
_load_error = None


def _candidate_lib_names():
    if platform.system() == "Windows":
        return ["libaccel.dll", "accel.dll"]
    return ["libaccel.so", "accel.so"]


def _try_load():
    global _lib, _load_error
    for name in _candidate_lib_names():
        path = _HERE / name
        if path.exists():
            try:
                _lib = ctypes.CDLL(str(path))
                return
            except OSError as exc:  # pragma: no cover - env dependent
                _load_error = exc
    if _lib is None and _load_error is None:
        _load_error = FileNotFoundError(
            f"No compiled accel library found in {_HERE} "
            f"(looked for {_candidate_lib_names()}). "
            "Build it first: see build_windows.bat / build_linux.sh."
        )


_try_load()

if _lib is not None:
    try:
        _lib.conv2d_forward.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
        ]
        _lib.conv2d_forward.restype = None
        AVAILABLE["conv2d"] = True
    except AttributeError as exc:  # pragma: no cover
        _load_error = exc

    try:
        _lib.lstm_forward.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ]
        _lib.lstm_forward.restype = None
        AVAILABLE["lstm"] = True
    except AttributeError as exc:  # pragma: no cover
        _load_error = exc


def _dptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def conv2d(x, weight, bias, stride=1, padding=1):
    """Drop-in replacement for np_inference.layers.conv2d.

    x: (B,C,H,W) ndarray, weight: (OutC,InC,kh,kw) ndarray,
    bias: (OutC,) ndarray or None. Returns (B,OutC,outH,outW) float64
    ndarray, matching the pure-numpy version's dtype behaviour (computed
    in float64 regardless of input dtype, same as numpy's own promotion
    when mixing the two here).
    """
    if not AVAILABLE["conv2d"]:
        raise RuntimeError(
            f"accel conv2d_forward not available ({_load_error!r}); "
            "build libaccel first (see build_windows.bat / build_linux.sh)."
        )

    x = np.ascontiguousarray(x, dtype=np.float64)
    weight = np.ascontiguousarray(weight, dtype=np.float64)
    B, C, H, W = x.shape
    OutC, InC, kh, kw = weight.shape

    if bias is not None:
        bias_arr = np.ascontiguousarray(bias, dtype=np.float64)
        bias_ptr = _dptr(bias_arr)
    else:
        bias_ptr = None

    outH = (H + 2 * padding - kh) // stride + 1
    outW = (W + 2 * padding - kw) // stride + 1
    out = np.empty((B, OutC, outH, outW), dtype=np.float64)

    _lib.conv2d_forward(
        _dptr(x), B, C, H, W,
        _dptr(weight), OutC, InC, kh, kw,
        bias_ptr,
        stride, padding,
        _dptr(out), outH, outW,
    )
    return out


def lstm_forward(x, weight_ih, weight_hh, bias_ih, bias_hh, hidden_size):
    """Drop-in replacement for np_inference.layers.lstm_forward.

    x: (T,B,input_size) ndarray. Returns (outputs, (h, c)) matching the
    pure-numpy version: outputs (T,B,H), h and c each (B,H), all float64.
    """
    if not AVAILABLE["lstm"]:
        raise RuntimeError(
            f"accel lstm_forward not available ({_load_error!r}); "
            "build libaccel first (see build_windows.bat / build_linux.sh)."
        )

    x = np.ascontiguousarray(x, dtype=np.float64)
    weight_ih = np.ascontiguousarray(weight_ih, dtype=np.float64)
    weight_hh = np.ascontiguousarray(weight_hh, dtype=np.float64)
    bias_ih = np.ascontiguousarray(bias_ih, dtype=np.float64)
    bias_hh = np.ascontiguousarray(bias_hh, dtype=np.float64)

    T, B, input_size = x.shape
    H = hidden_size

    out = np.empty((T, B, H), dtype=np.float64)
    h_out = np.empty((B, H), dtype=np.float64)
    c_out = np.empty((B, H), dtype=np.float64)

    _lib.lstm_forward(
        _dptr(x), T, B, input_size,
        _dptr(weight_ih), _dptr(weight_hh),
        _dptr(bias_ih), _dptr(bias_hh),
        H,
        _dptr(out), _dptr(h_out), _dptr(c_out),
    )
    return out, (h_out, c_out)
