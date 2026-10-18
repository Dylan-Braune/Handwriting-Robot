"""
camera_capture.py -- captures one still image from the camera attached to
whatever machine this runs on (Odroid in production, your laptop for
testing).

Your plugged-in camera enumerates as a standard USB Video Class (UVC)
webcam (Windows shows it as "UCB Camera", VID_0BDA -- Realtek, a generic
UVC chipset), NOT a Raspberry-Pi-style CSI ribbon camera. That means the
right tool is OpenCV's cv2.VideoCapture, which talks to any UVC device
through the OS's normal camera driver on BOTH Windows and Linux -- so the
exact same code path is what you're testing on your laptop right now and
what runs on the Odroid later, no platform-specific branching needed.

Usage:
    from camera_capture import capture_image, camera_available
    path = capture_image()   # returns a Path to the saved JPEG
"""
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = SCRIPT_DIR.parent / "NOGIT" / "CameraCaptures"
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

# Which /dev/videoN (Linux) or device index (Windows) to open. 0 is "first
# camera the OS finds" -- if your Odroid has more than one video device
# (e.g. a webcam AND an onboard HDMI capture chip), change this to the
# right index. Override via the CAMERA_INDEX env var without editing code.
import os
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))

# A UVC webcam's very first frame after opening is often dark/unfocused
# while its auto-exposure settles -- discard this many frames before
# keeping one, same fix every OpenCV capture tutorial recommends.
WARMUP_FRAMES = 5


def camera_available(index=None):
    """Cheap check: can we actually open and read one frame from the
    camera? Used by the server to report a clean 'no camera' error
    instead of a stack trace when testing off the Odroid."""
    import cv2
    idx = CAMERA_INDEX if index is None else index
    cap = cv2.VideoCapture(idx)
    try:
        if not cap.isOpened():
            return False
        ok, _frame = cap.read()
        return ok
    finally:
        cap.release()


_preview_cap = None
_preview_lock = None


def _get_lock():
    global _preview_lock
    if _preview_lock is None:
        import threading
        _preview_lock = threading.Lock()
    return _preview_lock


def preview_start(index=None):
    """Opens (and keeps open) the camera for live preview streaming. Safe
    to call repeatedly -- a no-op if already open. Kept separate from
    capture_image() so a short-lived capture doesn't leave the sensor
    powered on between uses: the whole point of a start/stop toggle is
    that the camera only runs while someone's actually looking at the
    live feed to frame the shot and let autofocus settle, not 24/7."""
    import cv2
    global _preview_cap
    with _get_lock():
        if _preview_cap is not None:
            return
        idx = CAMERA_INDEX if index is None else index
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open camera at index {idx} for preview.")
        _preview_cap = cap


def preview_stop():
    """Releases the preview camera handle. Always call this once you're
    done framing the shot -- see the module docstring's note on heat."""
    global _preview_cap
    with _get_lock():
        if _preview_cap is not None:
            _preview_cap.release()
            _preview_cap = None


def preview_active():
    return _preview_cap is not None


def preview_frame_jpeg():
    """Grabs one frame from the already-open preview camera and returns it
    JPEG-encoded bytes, for an MJPEG-style streaming endpoint. Raises
    RuntimeError if preview_start() hasn't been called."""
    import cv2
    with _get_lock():
        if _preview_cap is None:
            raise RuntimeError("Preview not started -- call preview_start() first.")
        ok, frame = _preview_cap.read()
    if not ok:
        raise RuntimeError("Failed to read a preview frame.")
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to JPEG-encode preview frame.")
    return buf.tobytes()


def capture_image(timeout=10, index=None):
    """Captures one still image, returns its Path. Raises RuntimeError if
    no camera could be opened/read -- callers (server.py) should catch
    this and report it as a clean error, not a crash.

    If a live preview is currently open (preview_start()), this grabs the
    frame from THAT handle instead of opening a second one -- most webcams
    only allow one process/handle to hold the device at a time, and this
    is also the intended flow: watch the live preview until autofocus
    settles on the page, then capture "the good image" from what's
    already showing, no re-opening/re-warming needed."""
    import cv2

    with _get_lock():
        if _preview_cap is not None:
            ok, frame = _preview_cap.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the open preview.")
            out_path = CAPTURE_DIR / f"capture_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(out_path), frame)
            return out_path

    idx = CAMERA_INDEX if index is None else index
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open camera at index {idx}. Is it plugged in, and "
            "not already in use by another program (close any other app "
            "using the webcam, e.g. a video call)?"
        )

    deadline = time.time() + timeout
    frame = None
    try:
        for _ in range(WARMUP_FRAMES):
            if time.time() > deadline:
                break
            ok, frame = cap.read()
            if not ok:
                frame = None
    finally:
        cap.release()

    if frame is None:
        raise RuntimeError(
            f"Camera at index {idx} opened but produced no readable frame "
            "within the timeout."
        )

    out_path = CAPTURE_DIR / f"capture_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(str(out_path), frame)
    if not out_path.exists():
        raise RuntimeError(f"cv2.imwrite reported no error but wrote no file: {out_path}")
    return out_path


if __name__ == "__main__":
    p = capture_image()
    print(f"Captured: {p}")
