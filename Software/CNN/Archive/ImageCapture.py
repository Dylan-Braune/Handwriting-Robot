import cv2
import time

# Initialize camera (0 is usually default webcam)
cap = cv2.VideoCapture(1)

# Set resolution to full 8MP
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2488)

# Pause briefly for 1–2 seconds to let autofocus & auto-exposure settle
time.sleep(1.5)

# Capture single frame
ret, frame = cap.read()
if ret:
    cv2.imwrite("photo_8mp.jpg", frame)

# Release the camera handle immediately (shuts off power/stream)
cap.release()