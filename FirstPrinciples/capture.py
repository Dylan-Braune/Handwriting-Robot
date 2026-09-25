"""
capture.py -- gets a page image into the program (FU 1.1, 1.2).

The camera and its USB driver are OFF-THE-SHELF items (the proposal's
system requirements table lists "a camera, camera driver" as taken off
the shelf). OpenCV's VideoCapture is used ONLY to grab a raw frame and
to decode/encode image FILES -- never to process, filter or analyse an
image. All real image processing happens by hand in conditioning.py /
segmentation.py using numpy only.
"""
import cv2
import numpy as np


def _to_gray(bgr_frame):
    """Averages the 3 colour channels by hand -- NOT cv2.cvtColor,
    since colour conversion counts as image processing."""
    b = bgr_frame[:, :, 0].astype(np.float32)
    g = bgr_frame[:, :, 1].astype(np.float32)
    r = bgr_frame[:, :, 2].astype(np.float32)
    return ((b + g + r) / 3.0).astype(np.uint8)


def capture_page(camera_index=0):
    """Grabs one frame from the camera. Returns a grayscale numpy
    array (rows x cols), pixel values 0-255."""
    cam = cv2.VideoCapture(camera_index)
    ok, frame = cam.read()
    cam.release()
    if not ok:
        raise RuntimeError("Could not read a frame from the camera")
    return _to_gray(frame)


def load_page(path):
    """Reads an existing image file from disk. cv2.imread here is just
    a file DECODER (JPEG/PNG bytes -> raw pixel array), the same role
    as the camera driver above."""
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(path)
    return _to_gray(frame)


def save_image(path, array):
    """Writes a numpy array back out as an image file (file ENCODER)."""
    cv2.imwrite(path, array)
