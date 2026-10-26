# Block D — Hardware control and the web service

| File | Lines | Role |
|---|---|---|
| `Gantry.py` | 825 | Odroid N2+ direct-GPIO gantry driver. Bit-bangs step/dir, interprets G-code, homes and calibrates, enforces end-stops. |
| `Server.py` | 590 | Flask JSON API: the whole pipeline behind HTTP. |
| `Camera.py` | 149 | USB webcam capture and live preview. |
| `WebRenderHelpers.py` | 142 | Glyph-grid and sample-crop images for the info page. |

---

## D.1 `Gantry.py`

### D.1.1 The hardware it drives

There is **no microcontroller**. The Odroid's own GPIO pins drive the A4988 step/dir inputs directly through `gpiod`, and end-stops are read in the same loop as every step pulse — which is what makes "stop immediately on any trigger" a real guarantee rather than best-effort.

| Signal | Pin (gpiochip0 offset) | Physical |
|---|---|---|
| X dir / step | 64 / 68 | 7, 11 |
| Y dir / step | 81 / 69 | 12, 13 |
| X min / max | 72 / 65 | 15, 16 |
| Y min / max | 66 / 67 | 18, 22 |
| Pen switch (input) | 70 | 33 |
| Pen motor (output) | 71 | 35 |

All five switches are active-high, triggered at 3.3 V, each with its own ~10 kΩ pull-down to ground fitted in hardware. The pen switch is **feedback only**: high means pen up, low means pen down. The pen motor output drives a cam that toggles pen height each time it runs — it is not a "move to position" actuator.

`MICROSTEPS = 16`, `STEPS_PER_REV = 200*16 = 3200`, and `STEPS_PER_MM = 80.0` is used **only before `Calibrate()` has run**; afterwards the measured values in `Calibration` are authoritative.

### D.1.2 Importability away from the hardware

```python
try:
    import gpiod
    from gpiod.line import Direction,Value
    _GPIOD_IMPORT_ERROR = None
except ImportError as _e:
    gpiod = None
    Direction = Value = None
    _GPIOD_IMPORT_ERROR = _e
```

`gpiod` only exists on the Odroid, and all hardware access happens inside `Connect()` and the functions that require it — never at import time. So `Server.py` can import this module on a laptop, and every route that touches the gantry returns a clean JSON error instead of a stack trace.

Two exception types carry that distinction outward: `GantryNotConnectedError` (no `gpiod`, or the chip could not be opened) and `NotCalibratedError` (`Calibrate()` has not succeeded this session). The second is deliberately **never persisted to disk**: a mechanical bump between sessions would silently invalidate a saved value, so calibration must be redone every restart.

### D.1.3 End-stop handling

**`TriggeredEndstop(ignore)`** returns the name of the first triggered switch not in `ignore`, debounced:

```python
if req.get_value(pin)!=Value.ACTIVE: continue
for _ in range(ENDSTOP_CONFIRM_SAMPLES-1):     # 3 samples total
    time.sleep(ENDSTOP_DEBOUNCE_S)             # 6 ms apart
    if req.get_value(pin)!=Value.ACTIVE: break
else:
    return name
```

A pin must read active on **three consecutive** samples 6 ms apart before it is trusted. A single re-check was not enough: the motors have no flyback diode and share a ground return, and that electrical noise can hold a switch input high for several milliseconds at a time — well past a one-shot debounce window. The sampling only begins once a pin is seen high at all, so the common "nothing triggered" case adds no delay.

`ignore` **skips** those pins during the scan rather than filtering the result afterwards. That matters: a switch legitimately still triggered elsewhere — the other axis resting at its own limit — could otherwise hide a genuinely different switch further down the scan.

`_AllTriggeredEndstops()` returns every triggered switch, since more than one can be genuinely active at once: sitting in a corner, or right after homing where both axes still rest against the switch they just found.

**`AxisDirValue(invert, positive)`** is the single source of truth for direction:

```python
forward = positive!=invert   # xor
return Value.ACTIVE if forward else Value.INACTIVE
```

Every direction decision in the file — drawing, homing, the RPM test — goes through it, so they cannot disagree about which pin value means which physical direction. `INVERT_X = INVERT_Y = True` for this machine's wiring.

**Emergency stop.** `_TriggerEmergencyStop(hit)` latches the state, drops both step pins, and then **retracts off every switch currently triggered**:

```python
allHits = _AllTriggeredEndstops() or [hit]
for h in allHits:
    _RetractOffSwitch(h, ignore=set(allHits)-{h})
```

`RETRACT_MM = 5.0` backs the tripped axis off its switch, with `_RETREAT_SIGN` giving the away direction per switch. Without this, clearing the e-stop with `r` would leave the gantry unable to move, because the switch reads triggered again the instant it is cleared. Retracting off *all* of them handles a corner hit, or a job aborting right after calibration with both axes resting on a switch. The retract stops early, without raising, if some *other* unexpected switch trips — that is a real mechanical problem rather than something to push through. The function is idempotent (`if estopped: return`), so it is safe to call from both the main loop and a G-code job, whichever notices first.

A tripped e-stop **never auto-clears**. Only an explicit `r` clears it, and that refuses while a switch is still physically triggered.

### D.1.4 `BresenhamMove` — coordinated two-axis motion

Bresenham (1965), which is exactly on-topic here: it is a plotter.

```python
if ax>=ay:                      # x is the major axis
    err = ax//2
    for _ in range(ax):
        if TriggeredEndstop() is not None: return
        pulse(STEP1_PIN); CurrentXSteps+=sx
        err-=ay
        if err<0:               # minor axis catches up by one step
            pulse(STEP2_PIN); CurrentYSteps+=sy
            err+=ax
```

The axis with the larger step count is the **major** axis and is stepped every iteration; the minor axis steps only when its accumulated error crosses zero. `err` tracks (minor progress) − (major progress), scaled by the major count, so the minor axis ends up stepped in proportion to `dx:dy` **without ever using floating point**. The result is that a diagonal move traces a straight line instead of one axis finishing early and the pen drawing an L.

`err` is initialised to `ax//2` rather than 0, which centres the rounding so the error is symmetric about the ideal line rather than biased to one side.

The end-stop is checked **before every pulse**, and the function simply returns when one trips, leaving the caller's loop to handle the e-stop state. Both step counters are updated as it goes, so position is always known.

### D.1.5 `SetPen` — closing the loop on a toggle actuator

```python
if PenIsUp()==targetUp: return True     # already there -- the toggle motor must not run
req.set_value(PEN_MOTOR_PIN,Value.ACTIVE)
while time.time()<deadline:
    if TriggeredEndstop() is not None: break
    if PenIsUp()==targetUp: reached = True; break
    time.sleep(0.002)
if reached and targetUp: time.sleep(PEN_UP_OVERRUN_S)
req.set_value(PEN_MOTOR_PIN,Value.INACTIVE)
```

The motor runs only until the **switch confirms** the target state, which is what makes an actuator with no position feedback usable. The early return when already in the target state is essential, because running the toggle motor again would invert the pen. `PEN_UP_OVERRUN_S = 0.04` keeps it running a little past the moment the switch first makes contact, so the mechanism finishes seating. A timeout is a real fault and is reported, not swallowed, and the function refuses to run at all while an end-stop is tripped.

### D.1.6 `Calibrate` — direction-agnostic homing

The elegant part of this file: calibration never needs to know in advance which direction leads to which switch.

`_StepAxis(stepPin, dirPin, dirValue, stopWhen, ignore)` steps one axis in one direction until `stopWhen(triggered_name)` is satisfied, returning the switch name and the step count. Any *other* triggered switch not in `ignore` means crossed wiring and raises immediately. `HOMING_MAX_STEPS = 200_000` without a trigger is also a fault, not a silent hang.

`_CalibrateAxis` then:

1. Drives one way until **either** of that axis's switches triggers, whichever comes first.
2. Infers that the other switch is the one in the opposite direction.
3. Reverses and counts the steps until it reaches it.

That count is the switch-to-switch travel in steps, and dividing by the measured physical distance gives the real steps per millimetre:

```python
Calibration["steps_per_mm_x"] = xTravel/X_SWITCH_TRAVEL_MM    # 204.0 mm
Calibration["steps_per_mm_y"] = yTravel/Y_SWITCH_TRAVEL_MM    # 262.0 mm
```

So the nominal 80 steps/mm from the pulley geometry is replaced by a measured value that absorbs belt stretch and pulley tolerance.

Two `ignore` sets make this robust. The reverse leg ignores `firstHit`, because it may still read triggered for the first few steps of the reverse move — release lag, not a fault. And the Y axis passes `OtherAxisNames={"X_MIN","X_MAX"}`, because X has just been calibrated and may legitimately be resting on its own switch.

Each axis is retracted `RETRACT_MM` off the switch it ends on **immediately after that axis finishes**, rather than at the end. Otherwise it is left resting switch-down, which would trip the main loop's e-stop the instant calibration mode ends — before the final park move even runs.

`usable_width/height = SWITCH_TRAVEL - 2*EDGE_TOLERANCE_MM`, with `EDGE_TOLERANCE_MM = 5.0` kept clear of both switches on both axes. Calibration finishes by parking at `(0,0)` in that frame, which is the safe start corner.

### D.1.7 G-code interpretation

`ParseGcodeLine` strips `;` comments and `(...)` comments, then `([A-Za-z])\s*(-?[0-9]*\.?[0-9]+)` pulls out letter/value words. The first `G` or `M` word becomes the command and every other letter a parameter.

**`_ClampToUsableArea`** pulls any target into the calibrated safe rectangle and prints when it does — a second line of defence on top of the physical switches, so a bad coordinate in an uploaded file is pulled back in bounds rather than ever reaching a switch.

**`_GcodeLinearMove`** converts absolute millimetres to a step delta:

```python
x0Mm = CurrentXSteps/spmmX - EDGE_TOLERANCE_MM
TargetXSteps = round((targetXMm+EDGE_TOLERANCE_MM)*spmmX)
dxSteps = TargetXSteps-CurrentXSteps
```

The delta is computed from the **current actual step count**, not from an accumulated float position, so repeated small moves cannot drift. The `EDGE_TOLERANCE_MM` offset encodes that `mm = 0` in the G-code frame is 5 mm in from the min switch, not the switch itself.

Feed rate becomes a per-step delay:

```python
feedMmS = max(feedMmMin/60.0, GCODE_MIN_FEED_MM_S)
stepDelayS = max((distMm/feedMmS)/majorSteps, CIRCLE_STEP_DELAY_S)
```

Total move time is distance over speed, divided evenly across the major axis's steps. `GCODE_MIN_FEED_MM_S = 0.1` is a floor so a stray `F0` cannot divide by zero, and `CIRCLE_STEP_DELAY_S = 2 µs` is the floor on pulse width.

**`_GcodeArcMove`** implements `G2`/`G3` by interpolation, since there is no native arc mode. Per the G-code specification the centre is `start + (I, J)`, so the radius is just `hypot(I, J)`. Start and end angles come from `atan2` of each endpoint relative to the centre, and then:

```python
if clockwise:
    while endAngle>=startAngle: endAngle -= 2*pi     # force a strictly decreasing sweep
else:
    while endAngle<=startAngle: endAngle += 2*pi     # strictly increasing
```

A clockwise arc must sweep with a decreasing angle and a counter-clockwise one with an increasing angle, so whichever came out going the wrong way has full turns added or subtracted until it is consistent — which is what makes a full circle, where start and end coincide, sweep 2π instead of 0. The arc is then chopped into `max(4, |span|*radius/0.4)` segments, about 0.4 mm per chord, and each waypoint goes through the same linear-move path as a regular `G1`.

**`RunGcodeFile`** refuses to run until `Calibrate()` has succeeded this session — checked here, so it applies equally to the CLI and the web server. Then per line it checks the end-stop *before* executing, and dispatches:

| Command | Behaviour |
|---|---|
| `G0`/`G1` | linear move; `F` persists until changed |
| `G2`/`G3` | arc, centre at `start + (I,J)` |
| `G4 P` | dwell, polling the end-stop every 5 ms so a dwell is still interruptible |
| `G90`/`G91` | absolute / relative |
| `G20` | inches — **values used as-is, not converted**, and it prints that |
| `G21` | millimetres, the native frame |
| `M3`/`M5` | pen down / up through `SetPen`; a non-confirming pen prints a warning and continues |
| `M2`/`M30` | program end |
| anything else | skipped with a message naming the line |

Both the absolute and relative branches track `curXMm`/`curYMm`, which is what makes `G91` work and lets an arc know its own start point.

`WriteGcodeFile(path)` is the server's entry point: connect if needed, run the file, leave the connection open, and return `{estopped, tripped_switch}` so the API can report a mid-run abort.

### D.1.8 The main loop and the CLI

The standalone program runs an `InputListener` thread for stdin commands while the main thread drives motion. The division of labour is deliberate:

```python
if CurrentMode=="CALIBRATING":
    time.sleep(0.01); continue
hit = TriggeredEndstop()
if hit is not None:
    _TriggerEmergencyStop(hit); targetRpm = 0.0; ...
elif estopped:
    time.sleep(0.005); continue    # a switch clearing on its own never auto-resumes
```

End-stops always win — **except** during calibration, because `Calibrate()` drives into switches on purpose and handles that itself. Reacting here as well would force the step pins low from the main thread while `_StepAxis()` on the listener thread is mid-pulse on those same pins.

`GCODE` mode likewise idles the main loop at 10 ms, because the listener thread drives those moves directly; `RPM` mode pulses both axes together at `stepDelayS` as a motor test; `CIRCLE` mode walks `theta` and issues `BresenhamMove` deltas to trace a circle of `CircleRadiusMm`.

CLI commands: `calibrate`, a bare number for RPM (0 stops), `c [radius]` for the circle demo, `g <path>` to run a file, `u`/`d` for the pen, `r` to clear an e-stop, `q` to quit.

---

## D.2 `Server.py` — the pipeline over HTTP

Flask plus flask-cors. The front end is a separate static site that can be hosted anywhere, since it only makes `fetch()` calls.

### D.2.1 Startup

Every model is loaded **once** at import, not per request:

```python
TEXT_MODEL         = PaperCRNNNumpy()                    # text recogniser
SHAPE_AUTHOR_MODEL = AuthorClassifierCNNNumpy()          # stroke-normalised: judges synthesis
INK_AUTHOR_MODEL   = AuthorClassifierCNNNumpy(checkpoint_path=INK_WEIGHTS_PATH)   # classifies real photos
PROFILES           = SY.LoadAllProfiles()
```

**Two writer-ID classifiers, for two different jobs.** The ink-based one classifies *real photographed handwriting*, where the author's own pen weight is genuine evidence. The stroke-normalised one judges *synthesised* output, where pen weight is an artefact of the renderer and the only valid evidence is shape. `_ClassifyInk` exists specifically to call the ink model with `apply_stroke_normalize=False`:

```python
#runs the INK-based classifier on one real line image (NOT the
#stroke-normalised one -- that one is only valid for judging
#synthesised/machine-rendered output)
```

Both forward and reverse index mappings are built at startup from each checkpoint's own `author_mapping`.

**Cache pre-warming runs in a background thread**, not before `app.run()`. The first pre-warm can take minutes for personal authors, because it segments their real photographs, and the server would otherwise refuse every connection until it finished. Later starts are fast, since results are cached to disk. Failures per author are caught and logged rather than aborting the warm-up.

Authentication is one `@app.before_request` hook: if `WEBAPP_API_KEY` is set, every request needs a matching `X-API-Key` header. The module header is explicit that this is a deterrent for local-network use, not real security.

### D.2.2 The endpoints

| Endpoint | What it does |
|---|---|
| `GET /api/authors` | the dropdown list, with each profile's summary |
| `POST /api/camera/capture` | triggers the camera, returns an `image_id` |
| `POST /api/camera/preview/start`/`stop` | opens and releases the camera for live framing |
| `GET /api/camera/preview/stream` | `multipart/x-mixed-replace` MJPEG, only while active |
| `GET /api/images/<image_id>` | serves a captured or uploaded image |
| `POST /api/classify` | photo → per-line text and writer identity |
| `POST /api/generate` | text + author → G-code and preview |
| `POST /api/gcode/upload` | an arbitrary G-code file through the same preview and send path |
| `POST /api/write/<job_id>` | sends a job's G-code to the physical gantry |
| `GET /api/gantry/status` | `{connected, calibrated, calibration}` |
| `POST /api/gantry/calibrate` | runs the homing sequence |
| `GET /api/gcode/<job_id>` | the raw `.gcode` file |
| `GET /api/model_info` | model specs and measured accuracy |
| `GET /api/glyphs/<author>` | the glyph-grid PNG |
| `GET /api/samples/<author>` | real sample crops plus transcriptions |

Every path-parameter id is filtered with `"".join(c for c in image_id if c.isalnum())` before being used to build a filesystem path, so a job or image id cannot escape its directory.

### D.2.3 `POST /api/classify` — the read path

1. Accept either an uploaded file or a previously captured `image_id`, saving uploads as `<uuid>.jpg`.
2. Convert to greyscale, write a temporary PNG, and run `SegmentLean.SegmentLines` on it — the same segmenter used everywhere else. The temp file is removed in a `finally`.
3. Return **422** with a readable message if no lines were detected.
4. Per line: `ReadText` with the text model, and `_ClassifyInk` for writer identity.
5. Join the per-line texts with spaces for the page's transcript, and **average the author probability vectors across lines**, taking the arg-max of the average as the page's predicted author.

Averaging the full probability vectors rather than voting on per-line arg-maxes is the right aggregation: a line the model is unsure about contributes its uncertainty instead of a hard, possibly wrong vote.

Accuracy against `expected_text` and a boolean against `expected_author` are included when the caller supplies them, and the full per-author confidence table is always returned.

### D.2.4 `POST /api/generate` — the write path

```python
traj = SY.SynthesizeJointBestOf(
    author,text,prof,nTries=nTries,mmPerXh=4.0,
    lineWidthMm=cfg.boundsMaxXmm-cfg.originXmm-5,
    reader=TEXT_MODEL,authorModel=SHAPE_AUTHOR_MODEL,authorMapping=SHAPE_MAPPING)
```

The joint harmonic-mean selector of C.3.6, with `nTries` defaulting to 20. The already-loaded models are passed in, so best-of-N does not reload a checkpoint per candidate. `lineWidthMm` is derived from the machine's own work area with a 5 mm margin, so wrapping happens at a width the gantry can actually draw.

Each job gets a UUID directory under `NOGIT/WebJobs/<job_id>/` holding `job.gcode` and `preview.png`.

`_JobResponse` is shared with the upload route and does something worth noting: it **reads the rendered preview back** with the text recogniser and reports that transcript plus its accuracy against the requested text. So the API's own response tells the caller how legible the thing it is about to draw actually is.

### D.2.5 The gantry routes

`GET /api/gantry/status` attempts a `Connect()` if not already connected, and reports `connected: false` with a detail message rather than failing.

`POST /api/write/<job_id>` catches both hardware exception types and returns them as **200 with an explanatory message**, not a 500 — because "not on the Odroid" and "not calibrated yet" are expected everyday states while developing on a laptop, not server errors. It also surfaces a mid-run end-stop abort:

```python
if result["estopped"]:
    return jsonify({"sent": False, ...
        "message": f"Job aborted by end-stop '{result['tripped_switch']}' mid-run. "
                   "Clear it on the gantry before sending another job."}),200
```

The front end greys out Generate, Upload and Send until `/api/gantry/calibrate` has been called once per server session, which mirrors `RunGcodeFile`'s own refusal.

`POST /api/gcode/upload` saves the file, counts non-blank lines and `M3`/`M5` occurrences for the pen-pulse estimate, renders it with `RenderGcodePreview`, and returns **422** with the parse error if the file cannot be read. It then goes through the identical `_JobResponse` and `/api/write` path as a generated job.

---

## D.3 `Camera.py`

A plain USB/UVC webcam through OpenCV's `VideoCapture`, which goes through the OS's normal camera driver on **both** Windows and Linux — so the same code path runs on the development laptop and on the Odroid, with no platform branching.

`cv2` is imported **inside** each function rather than at module level, so importing the module costs nothing and does not require OpenCV to be present.

Two modes share one handle, protected by a lazily created lock:

- **Preview.** `PreviewStart` opens the camera and keeps it open; `PreviewFrameJpeg` grabs one frame and JPEG-encodes it for the MJPEG endpoint; `PreviewStop` releases it. These are kept separate from capture so the sensor is not left powered on between uses — it only runs while someone is actually framing the shot and letting autofocus settle.
- **Capture.** `CaptureImage` has an important special case:

  ```python
  with _GetLock():
      if _previewCap is not None:
          ok,frame = _previewCap.read()     # use the already-open handle
          ...
  ```

  If a preview is open, the still is grabbed from **that** handle rather than opening a second one. Most webcams allow only one handle on the device at a time, and this is also the intended workflow: watch the live preview until autofocus settles on the page, then capture what is already showing.

With no preview open it opens the device itself and discards `WARMUP_FRAMES = 5` frames first, because a UVC webcam's first frame after opening is often dark or unfocused. A `timeout` (default 10 s) bounds the warm-up, `cap.release()` is in a `finally`, and every failure raises `RuntimeError` with a message aimed at the user — "is it plugged in, and not already in use by another program" — so `Server.py` can report it cleanly. The write is verified (`if not outPath.exists()`) before the path is returned.

Captures land in `NOGIT/CameraCaptures/capture_<ms>.jpg`, and `CAMERA_INDEX` is overridable by environment variable.

---

## D.4 `WebRenderHelpers.py`

Pure rendering, no Flask, so it also runs standalone. Results are cached to disk under `NOGIT/WebCache/` the first time each is requested, since glyph grids are mildly expensive to render and never change unless the profiles do.

**`GlyphGridPath(author)`** renders the author's whole glyph library as one sheet: one row per character, one column per stored variant, with the character labelled at the left. `_RenderOneGlyph` draws one stored glyph into a 90 px cell, scaling by `(cell-2*pad)/max(w,h)` so the glyph's **aspect ratio is preserved** and flipping y, since glyph space has y up and image space has y down. These are the same manual-inspection sheets used during development, which makes them the natural figure for showing what a profile actually contains.

**`SampleCrops(author, n=2)`** returns up to `n` `(image_path, transcription)` pairs of **real, non-holdout** handwriting for the author, so the UI can show what the hand actually looks like beside what the robot produced. IAM authors are filtered out of `IAMLineDatasetRaw`; personal authors come through `AddPersonalSamples`. Lines are filtered to `20 ≤ len(text) ≤ 70`, which picks a representative line rather than a fragment or a wrapped monster. Transcriptions are cached beside the images in a `_meta.txt`, and the cache is only trusted when it holds at least `n` images *and* the metadata file.

`_font` tries Consolas and falls back to PIL's default, so a missing font degrades the label rather than raising.
