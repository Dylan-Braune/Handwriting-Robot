#captures one still image from the camera attached to whatever machine
#this runs on (Odroid in production, your laptop for testing).
#
#your plugged-in camera enumerates as a standard USB Video Class (UVC)
#webcam (Windows shows it as "UCB Camera", VID_0BDA -- Realtek, a generic
#UVC chipset), NOT a Raspberry-Pi-style CSI ribbon camera. That means the
#right tool is OpenCV's cv2.VideoCapture, which talks to any UVC device
#through the OS's normal camera driver on BOTH Windows and Linux -- so the
#exact same code path is what you're testing on your laptop right now and
#what runs on the Odroid later, no platform-specific branching needed.
#
#usage:
#    from Camera import CaptureImage
#    path = CaptureImage()   # returns a Path to the saved JPEG
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
#this file lives in Software/CleanPipeline, a sibling of Software/CNN
#where the NOGIT captures folder still lives
CNN_DATA_DIR = Path(__file__).resolve().parent.parent/"CNN"
CAPTURE_DIR = CNN_DATA_DIR/"NOGIT"/"CameraCaptures"
CAPTURE_DIR.mkdir(parents=True,exist_ok=True)

import os
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX","0"))   #which /dev/videoN (Linux) or device index (Windows) to open

WARMUP_FRAMES = 5   #a UVC webcam's first frame after opening is often dark/unfocused


_previewCap = None
_previewLock = None


def _GetLock():
    global _previewLock
    if _previewLock is None:
        import threading
        _previewLock = threading.Lock()
    return _previewLock


#opens (and keeps open) the camera for live preview streaming. Safe to
#call repeatedly -- a no-op if already open. Kept separate from
#CaptureImage() so a short-lived capture doesn't leave the sensor powered
#on between uses -- the camera only runs while someone's actually
#watching the live feed to frame the shot and let autofocus settle.
def PreviewStart(index=None):
    import cv2
    global _previewCap
    with _GetLock():
        if _previewCap is not None:
            return
        idx = CAMERA_INDEX if index is None else index
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open camera at index {idx} for preview.")
        _previewCap = cap


#releases the preview camera handle. Always call this once you're done
#framing the shot -- see the module comment's note on heat.
def PreviewStop():
    global _previewCap
    with _GetLock():
        if _previewCap is not None:
            _previewCap.release()
            _previewCap = None


def PreviewActive():
    return _previewCap is not None


#grabs one frame from the already-open preview camera and returns
#JPEG-encoded bytes, for an MJPEG-style streaming endpoint. Raises
#RuntimeError if PreviewStart() hasn't been called.
def PreviewFrameJpeg():
    import cv2
    with _GetLock():
        if _previewCap is None:
            raise RuntimeError("Preview not started -- call preview_start() first.")
        ok,frame = _previewCap.read()
    if not ok:
        raise RuntimeError("Failed to read a preview frame.")
    ok,buf = cv2.imencode(".jpg",frame)
    if not ok:
        raise RuntimeError("Failed to JPEG-encode preview frame.")
    return buf.tobytes()


#captures one still image, returns its Path. Raises RuntimeError if no
#camera could be opened/read -- callers (Server.py) should catch this
#and report it as a clean error, not a crash.
#if a live preview is currently open (PreviewStart()), this grabs the
#frame from THAT handle instead of opening a second one -- most webcams
#only allow one process/handle to hold the device at a time, and this is
#also the intended flow: watch the live preview until autofocus settles
#on the page, then capture "the good image" from what's already showing.
def CaptureImage(timeout=10,index=None):
    import cv2

    with _GetLock():
        if _previewCap is not None:
            ok,frame = _previewCap.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the open preview.")
            outPath = CAPTURE_DIR/f"capture_{int(time.time()*1000)}.jpg"
            cv2.imwrite(str(outPath),frame)
            return outPath

    idx = CAMERA_INDEX if index is None else index
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open camera at index {idx}. Is it plugged in, and "
            "not already in use by another program (close any other app "
            "using the webcam, e.g. a video call)?"
        )

    deadline = time.time()+timeout
    frame = None
    try:
        for _ in range(WARMUP_FRAMES):
            if time.time()>deadline:
                break
            ok,frame = cap.read()
            if not ok:
                frame = None
    finally:
        cap.release()

    if frame is None:
        raise RuntimeError(
            f"Camera at index {idx} opened but produced no readable frame "
            "within the timeout."
        )

    outPath = CAPTURE_DIR/f"capture_{int(time.time()*1000)}.jpg"
    cv2.imwrite(str(outPath),frame)
    if not outPath.exists():
        raise RuntimeError(f"cv2.imwrite reported no error but wrote no file: {outPath}")
    return outPath

if __name__=="__main__":
    p = CaptureImage()
    print(f"Captured: {p}")
