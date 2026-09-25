# FirstPrinciples

A from-scratch reimplementation matching the actual scope in the project
proposal (`ProjectProposal2026DylanFinalRev0...pdf`), separate from the
main `Software/` codebase (which uses PyTorch/PIL/pytesseract and is far
more advanced than the proposal requires). Nothing in `Software/` has
been touched -- this is a parallel, simpler pipeline to test and
eventually replace it with, once you're happy it meets spec.

## Library rule followed here

Per the proposal (section 2/4): *library functions only for hardware
interfacing*. Image processing, character separation, the neural
network, and pen-trajectory mapping are all hand-written using plain
Python + numpy (numpy is a maths library, like a calculator -- it does
matrix multiplication, not image processing or learning).

`cv2` appears in exactly one file (`capture.py`) and only for the two
things the proposal explicitly allows off-the-shelf: the camera driver
(`VideoCapture`) and image file decode/encode (`imread`/`imwrite`,
which just turn a JPEG/PNG file into a raw pixel array or back -- the
same job a camera driver does). No `cv2` processing function
(`cvtColor`, `threshold`, `resize`, `findContours`, etc.) is used
anywhere. `gpiod` in `motor.py` is a hardware GPIO library, same
category as the camera driver.

Verification/testing scripts (accuracy checking, plotting, loading a
reference dataset) are NOT held to this rule -- only the actual
training/classification/reproduction methodology is.

## Files, mapped to the proposal's functional block diagram (section 3.2)

| File | Functional unit(s) |
|---|---|
| `capture.py` | FU 1.1, 1.2 -- camera driver, image transfer |
| `conditioning.py` | FU 2.1 -- image pre-conditioner (binarize, deskew) |
| `segmentation.py` | FU 2.2 -- character separation |
| `network.py` | FU 2.3, 2.4 -- classifier + backpropagation |
| `trajectory.py` | FU 2.5 -- character image to trajectory map |
| `storage.py` | FU 2.6, 2.7 -- permanent trajectory map storage/loading |
| `motor.py` | FU 2.8, 3.1, 3.2, 3.4 -- stepper pulse generation |
| `main.py` | ties all of the above into train/classify/write commands |

## Running it

```
python main.py
> train page1.jpg abcdef 0
> classify page2.jpg
> write hello 0
```

Character size, hidden-layer size, and the character set are all
constants at the top of `main.py` -- adjust them there.
