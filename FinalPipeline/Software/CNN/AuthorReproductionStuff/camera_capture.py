"""
camera_capture.py -- captures one still image from whatever camera module
is attached to the Odroid.

I (the assistant) do not know the exact camera hardware/library on your
Odroid, so this defaults to shelling out to common Linux camera CLI
tools (tried in order below) rather than guessing a specific Python
camera SDK that might not be installed. ADJUST `CAPTURE_COMMANDS` to
match your actual setup -- e.g. if you're using `picamera2` in Python
directly, replace `capture_image()`'s body with that library's call
instead of a subprocess.

Usage:
    from camera_capture import capture_image
    path = capture_image()   # returns a Path to the saved JPEG
"""
import shutil
import subprocess
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = SCRIPT_DIR.parent / "NOGIT" / "CameraCaptures"
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

# Tried in order -- first one whose executable exists on PATH is used.
# {path} is substituted with the output file path.
CAPTURE_COMMANDS = [
    # Raspberry Pi / libcamera-based cameras (common on many SBCs, incl.
    # some Odroid camera modules that provide a libcamera-compatible driver)
    ["libcamera-still", "-n", "-o", "{path}", "--width", "1920", "--height", "1080"],
    # Older Raspberry Pi camera stack
    ["raspistill", "-n", "-o", "{path}"],
    # Generic USB webcam via fswebcam
    ["fswebcam", "-r", "1920x1080", "--no-banner", "{path}"],
]


def _find_command():
    for cmd in CAPTURE_COMMANDS:
        exe = cmd[0]
        if shutil.which(exe):
            return cmd
    return None


def capture_image(timeout=10):
    """Captures one still image, returns its Path. Raises RuntimeError if
    no known camera command is available -- in that case, either install
    one of the tools above, or replace this function's body with a direct
    call into your camera's own Python SDK (e.g. picamera2)."""
    cmd_template = _find_command()
    if cmd_template is None:
        raise RuntimeError(
            "No known camera capture command found on PATH (tried: "
            f"{[c[0] for c in CAPTURE_COMMANDS]}). Edit camera_capture.py "
            "to call your actual camera's capture method directly."
        )
    out_path = CAPTURE_DIR / f"capture_{int(time.time() * 1000)}.jpg"
    cmd = [part.format(path=str(out_path)) for part in cmd_template]
    subprocess.run(cmd, check=True, timeout=timeout, capture_output=True)
    if not out_path.exists():
        raise RuntimeError(f"Camera command ran but produced no file: {' '.join(cmd)}")
    return out_path


if __name__ == "__main__":
    p = capture_image()
    print(f"Captured: {p}")
