# Handwriting Robot — Final Pipeline (curated)

This folder contains **only the files that make up the current, working
pipeline** for: (1) text recognition training/reading, (2) writer
identification training/classifying, and (3) the current best
handwriting-reproduction method (style extraction → synthesis → G-code).
Everything experimental, superseded, or exploratory that lives elsewhere
in the repo (old label generators, the ESP32 gantry path, the
`ReproduceV2` diffusion-based alternative, one-off diagnostic/comparison
scripts, etc.) is deliberately **left out** — see
`../FILE_INVENTORY.md` at the repo root if you want the full picture of
what every file in the project does and why it is/isn't here.

## IMPORTANT — where this folder must live to actually run

These scripts locate their data (`Data/Datasets/...`) and trained
weights/caches/profiles (`NOGIT/...`) using paths computed *relative to
their own location on disk*, exactly the way they do in the main repo.
The folder structure here (`Software/CNN/...`) intentionally mirrors the
real repo so those relative paths still resolve correctly.

**Two ways to actually run these:**

1. **Recommended — don't move this folder at all.** Everything in here
   is an exact copy of files that also exist in your main repo, at
   `Software/CNN/...`. Just run the *original* files where they already
   are (this folder is a clean, organised reference/submission copy, not
   a second working copy). All the commands below work as-is from the
   repo root.
2. **If you do want this folder to run standalone** (e.g. you copied it
   somewhere else, or you're using it as your Submission Item 2 code
   upload and want to test it in isolation first): copy your existing
   `Software/CNN/NOGIT/` folder to `FinalPipeline/Software/CNN/NOGIT/`,
   and your existing `Data/Datasets/` folder to
   `FinalPipeline/Data/Datasets/`. Both are large (weights + the IAM
   page scans), which is why they aren't duplicated here automatically.

All commands below assume you're running from inside
`FinalPipeline/Software/CNN/` (or the equivalent real location,
`Software/CNN/`), using whichever Python environment already has
`torch`, `numpy`, `PIL`, and `pytesseract` installed (the same one you've
been using all along).

---

## Stage 1 — Dataset preparation (fixing/labelling scanned pages)

**File:** `DatasetPrep/regenerate_labels_with_alignment.py`

**What it does:** reads each author's scanned/photographed page images
under `Data/Datasets/IAMpages10/<authorId>/*.png`, OCRs each page's own
clean printed header text, and aligns it word-by-word (by pixel width,
not by guessing) against the detected handwriting lines, writing a
`<page>_labels.txt` file next to each image. This is the **correct**
labelling method — an earlier, simpler method (proportional word
splitting) is not included here because it was found to systematically
mis-label pages where different authors wrapped a shared source
paragraph into a different number of lines (see
`../method2-judge-ceiling` write-up / the final report's Discussion
section for the full story).

**Input:** page images already sitting in `Data/Datasets/IAMpages10/<authorId>/`
(these are not included in this folder — they're your existing dataset).

**Output:** a `<page>_labels.txt` file written next to each page image,
in the same folder.

**Command:**
```bash
cd DatasetPrep
python regenerate_labels_with_alignment.py --num-folders 10 --data-dir "../../../Data/Datasets/IAMpages10" --dry-run
```
Check the dry-run output first (it reports how many pages would change
and by how much), then re-run without `--dry-run` to actually write the
labels. `--num-folders 10` covers all ten author folders currently in
that directory.

You only need to run this once per dataset change — the line *images*
are cached separately (see Stage 2) and don't need to be rebuilt when
only the label text changes.

---

## Stage 2 — Text recognizer (handwriting → text)

**Files:** `TrainText.py` (core model/dataset code, imported by
everything else — never run directly), `ExtractIAMLines.py` (line
segmentation, imported by `TrainText.py`), `TrainTextHF.py` (stage 2a),
`TrainTextJoint.py` (stage 2b, produces the final checkpoint),
`ClassifyText.py` (reading/inference on a new page).

### 2a. Train the general-purpose recognizer
```bash
python TrainTextHF.py
```
Trains a CNN-BiLSTM-CTC recognizer on 6,480 clean, professionally
segmented handwriting lines (Teklia/IAM-line). Writes
`NOGIT/weights/paper_cnn_bilstm_ctc_hf_best.pt`. Run this once; it
doesn't need to be repeated unless you want to retrain from scratch.

### 2b. Fine-tune on your own handwriting, without forgetting the general case
```bash
python TrainTextJoint.py --init-from NOGIT/weights/paper_cnn_bilstm_ctc_hf_best.pt
```
This is the **current best** text recognizer training method: it mixes
your own (personal) handwriting lines into every training epoch
alongside the general-purpose data (via a weighted sampler), instead of
fine-tuning on your handwriting alone afterward — plain sequential
fine-tuning was measured to cause the model to forget general
handwriting (a real regression that was found and fixed this project).
Writes `NOGIT/weights/paper_cnn_bilstm_ctc_joint_best.pt` — this is the
final checkpoint every other script in this folder automatically prefers
when it exists.

**Input for your own handwriting:** photographed pages under
`NOGIT/<yourname>/*.jpg`, with matching `_labels.txt` transcriptions
(same format as Stage 1's output) — set up once per person you want the
recognizer/system to know. **Note:** the personal author names
(currently `yeukita` and `dylan`) are hardcoded as `PERSONAL_AUTHORS` at
the top of `TrainAuthor10.py` and
`AuthorReproductionStuff/BuildStyleProfile10Authors.py` — if you add a
third person, add their folder name to that list in both files.

### 2c. Read a new page (inference / "classifying" text)
```bash
python ClassifyText.py
```
Interactive — just run it. It asks for: the image (or folder of images)
to transcribe, whether it's an IAM-style scan or your own photographed
page, and which weights file to use (blank = its own sensible default).
Prints the predicted transcription line by line, and a character-error
rate if a matching `_labels.txt` exists next to the image for
comparison.

**Input:** any page image (a path you type in when prompted).
**Output:** printed to the terminal; no file is written.

---

## Stage 3 — Writer identification (whose handwriting is this?)

**Files:** `TrainAuthor.py` (core classifier model, imported by the
others — never run directly), `TrainAuthor10.py` (ink-based 10-author
classifier), `AuthorReproductionStuff/TrainAuthorShape.py`
(**stroke-normalised** classifier — this is the one actually used to
judge synthesised/machine-drawn output, see below).

### 3a. Train the ink-based classifier
```bash
python TrainAuthor10.py 30
```
(the number is how many epochs to train; 30 is a reasonable default —
it typically converges within 10-15). Trains a CNN classifier to tell
apart the ten authors (eight from the IAM dataset plus your two personal
ones) from their real, photographed handwriting. Reaches ~100% on real
ink. Writes `NOGIT/weights/author_classifier_10new_weights.pt`, saving a
new best checkpoint every time validation improves (safe to stop early).

**Important limitation, by design:** this classifier leans on ink
density/pressure, which a pen-plotter physically cannot reproduce — it
is NOT used to judge synthesised output (see 3b). It's included because
it's still the right tool for classifying **real** photographed pages
(e.g. "whose handwriting is this scan?").

### 3b. Train the stroke-normalised classifier (for judging synthesis)
```bash
cd AuthorReproductionStuff
python TrainAuthorShape.py
```
Trains a second classifier on the same ten authors, but every training
line is binarised, skeletonised, and re-inked at one constant width
first — removing ink-density as a usable cue, so the classifier is
forced to learn slant, proportions, spacing and connection habits
instead: the things a gantry with a single pen *can* actually reproduce.
Reaches ~97% on real ink and, critically, does not collapse when judging
machine-rendered output the way the ink-based classifier does (~100%
real ink but only ~20% on synthesis). Writes
`NOGIT/weights/author_shape_10new_weights.pt`. **This is the checkpoint
`EvaluateStyle.py` actually loads.**

---

## Stage 4 — Handwriting reproduction (the actual "write like author X" pipeline)

**Files:** `AuthorReproductionStuff/BuildStyleProfile.py` +
`BuildStyleProfile10Authors.py` (style/glyph extraction),
`SynthesizeHandwriting.py` (the synthesis engine — the core of the whole
project), `EvaluateStyle.py` (writer-ID judging for synthesised output),
`VerifyRewrite.py` (full end-to-end accuracy measurement),
`WriteGCode.py` (G-code emission/parsing), `WriteAsAuthor.py` (the
practical "just write this text as this author" entry point — **this is
the one you actually run day-to-day**).

### 4a. Build each author's style profile (run once, or after data/label changes)
```bash
cd AuthorReproductionStuff
python -c "import BuildStyleProfile as BSP; BSP.BuildAll()"
python BuildStyleProfile10Authors.py
```
The first command extracts individual character glyphs (via
forced-alignment against the text recognizer) for every author folder
under `Data/Datasets/IAMpages10/`, caching the raw result in
`NOGIT/GlyphCache10/<author>.pkl`, and writes a style profile per author
to `NOGIT/StyleProfiles10/<author>.json`. The second command redoes the
final pooling/profile step correctly across the real 10-author set (the
eight kept IAM authors + your two personal ones), overwriting the
profiles from the first command with the correctly-pooled versions —
**always run both, in this order.**

If you've added or re-labelled pages for an author, delete that
author's `NOGIT/GlyphCache10/<author>.pkl` first so it gets re-extracted
rather than reusing the stale cache.

**A per-author tuning parameter you may want to adjust:** each profile
JSON has a `legibilityLambda` field (0.0-1.0) controlling how much a
character blends toward a cleaned-up reference letterform versus the
author's own raw handwriting when a letter is ambiguous. Higher =
more legible, less authentic; this was tuned per author against
`VerifyRewrite.py`'s output (current values: see the profiles
themselves). If you rebuild profiles from scratch, this field is not
regenerated automatically — you'll need to re-tune it for best results
(see `VerifyRewrite.py` below to check its effect), or just copy the
tuned values back in from your existing profile files.

### 4b. Write something (the actual point of the project)
```bash
python WriteAsAuthor.py "Hello, this is a test of the handwriting robot" dylan
```
(replace `dylan` with any author id, or a number 1-10 as listed when you
run the script with no arguments). Generates:
- `NOGIT/WriteJobs/<author>/job.gcode` — the G-code file to send to the gantry
- `NOGIT/WriteJobs/<author>/steps.csv` — a step/direction schedule (alternative to G-code, if your controller wants raw steps)
- `NOGIT/WriteJobs/<author>/synth_preview.png` — what the trajectory looks like
- `NOGIT/WriteJobs/<author>/plotter_preview.png` — what the emitted G-code, re-parsed, would actually draw
- Console output reporting the text/writer-ID confidence of the result, stroke count, and a simulation check (does re-parsing the G-code reproduce the original trajectory?)

Internally this uses `SynthesizeHandwriting.SynthesizeJointBestOf`: it
draws several candidate versions of each line and keeps whichever one
scores best on *both* text legibility (via the Stage 2 recognizer) and
writer-ID confidence (via the Stage 3b classifier) together — this is
the single biggest improvement made to reproduction quality this
project, and is what took writer-ID from ~79% to ~99% and text accuracy
from ~76% to ~86% on the full evaluation (see 4c). It costs one
recognizer pass + one classifier pass per candidate, so it's slower than
a single plain draw — the `legibilityTries` parameter in `WriteAsAuthor.Run()`
(default 20) trades speed for quality; lower it for faster interactive
use.

### 4c. Measure how well the whole pipeline is doing
```bash
python VerifyRewrite.py
```
The authoritative, full end-to-end evaluation: synthesises six novel
sentences (never seen anywhere in training) for all ten authors, reads
each one back with the Stage 2 recognizer and classifies it with the
Stage 3b classifier, and reports mean text accuracy and mean writer-ID
accuracy — both directly and through a full G-code emit → re-parse →
re-render round trip (i.e. what the physical gantry would actually
produce). Also reports the same recognizer's accuracy on each author's
*real* held-out handwriting, as an honest ceiling to judge the
synthesised numbers against. This is the script to re-run after any
change to the synthesis pipeline, to confirm it actually helped — **do
not trust a smaller/custom test to predict this script's result; several
changes this project looked good on a smaller test and then regressed
here.** Samples are saved to `NOGIT/EndToEnd/<author>_{synth,gcode}.png`
and `<author>_read.txt` for spot-checking by eye.

---

## Quick reference: full pipeline in order, from nothing

```bash
# 1. Label your scanned/photographed pages correctly
cd Software/CNN/DatasetPrep
python regenerate_labels_with_alignment.py --num-folders 10 --data-dir "../../../Data/Datasets/IAMpages10"

# 2. Train the text recognizer
cd ../
python TrainTextHF.py
python TrainTextJoint.py --init-from NOGIT/weights/paper_cnn_bilstm_ctc_hf_best.pt

# 3. Train both writer-ID classifiers
python TrainAuthor10.py 30
cd AuthorReproductionStuff
python TrainAuthorShape.py

# 4. Build style profiles and verify
python -c "import BuildStyleProfile as BSP; BSP.BuildAll()"
python BuildStyleProfile10Authors.py
python VerifyRewrite.py

# 5. Actually write something
python WriteAsAuthor.py "Your text here" <author>
```

## Stage 5 — Web app (camera-capture read page, write page, model/style dashboard)

**Backend files:** `AuthorReproductionStuff/server.py` (Flask API),
`AuthorReproductionStuff/web_render_helpers.py` (glyph-grid/sample-crop
image generation, cached to disk), `AuthorReproductionStuff/camera_capture.py`
(camera adapter — see note below).

**Frontend files:** `WebFrontend/` — three static pages (`index.html`
= Read, `write.html` = Write, `stats.html` = Model & Style), plus
shared `style.css` and `app.js`. Pure HTML/CSS/JS, no build step.

### Running it

**1. On the Odroid — start the backend** (this is where all the actual
processing happens; it needs the trained weights and datasets, so it
must run from its real location with `NOGIT/`/`Data/` accessible, same
as every other script in this folder):
```bash
cd Software/CNN/AuthorReproductionStuff
pip install flask flask-cors
python server.py
```
Listens on `0.0.0.0:5000` by default (change with the `PORT` env var).
The first time you start it, it will take a little while in the
background to pre-warm the glyph/sample-crop caches for the stats page
(especially for your two personal authors, since that runs page
segmentation on their real photos) — the server itself is usable
immediately, and requesting a personal author's samples/glyphs before
that finishes just returns a "still preparing" message rather than
hanging. Every start after the first is fast, since results are cached
to `NOGIT/WebCache/`.

**2. Anywhere else — open the frontend.** Since you said you'd rather
not run things from the Odroid itself: copy the `WebFrontend/` folder
to your laptop/phone/another machine and either just open `index.html`
directly, or (recommended, since some browsers restrict `fetch()` from
plain local files) serve it with any static file server, e.g.:
```bash
cd WebFrontend
python -m http.server 8080
```
then open `http://localhost:8080/index.html` in a browser.

**3. Point it at your Odroid.** Click the connection indicator (top
right of any page) and enter your Odroid's address, e.g.
`http://192.168.1.50:5000` (find its LAN IP with `hostname -I` on the
Odroid). This is saved in the browser's local storage, so you only set
it once per device. The dot next to it turns green when connected.

### Security note

This server has no login and minimal input validation — it's meant for
your own local network. If you ever want it reachable from the wider
internet (not just your LAN), set the `WEBAPP_API_KEY` environment
variable before starting `server.py`, and enter the same key in the
frontend's connection settings; every request then needs a matching
`X-API-Key` header. This is a basic deterrent, not real security — put
a proper reverse proxy with TLS and/or a VPN in front of it before
exposing it beyond your LAN, and never do so with a predictable/default
API key.

### Camera capture — you'll need to check this matches your hardware

`camera_capture.py` shells out to whichever of `libcamera-still`,
`raspistill`, or `fswebcam` it finds installed, in that order — I don't
know your Odroid's exact camera module/library, so this is a
best-effort default for common Linux camera setups, not a guarantee.
If your camera needs a different tool or a Python SDK (e.g.
`picamera2`), edit `capture_image()` in that file to call it directly;
everything else (the `/api/camera/capture` endpoint, the Read page's
"Capture from camera" button) will keep working unchanged as long as
that one function still returns a path to a saved image.

### What each page does

- **Read** (`index.html`): capture or upload a photographed page,
  type the paragraph of text and pick the author you expect it to be,
  click Classify. Runs the Stage 2 text recognizer (line by line) and
  the **ink-based** writer-ID classifier (Stage 3a — the correct one for
  real photos, not the synthesis judge) on the detected lines, and
  reports the full recognized text, a text accuracy percentage against
  what you typed, the classified author, and a confidence percentage
  (plus whether it matched your expected author).
- **Write** (`write.html`): type text, pick an author, generate. Runs
  the full Stage 4 reproduction pipeline (`SynthesizeJointBestOf` +
  `WriteGCode`) and displays the rendered preview and the actual G-code
  (downloadable), plus how the frozen recognizer reads the result back.
  A "Quality" field controls `nTries` — how many candidate lines are
  drawn before picking the best one; higher is slower but more accurate
  (see Stage 4b above). Gantry live-progress reporting is a placeholder
  for later — there's nowhere for it to come from yet since the gantry
  doesn't report status back over the network at the moment.
- **Model & Style** (`stats.html`): live specs (architecture, parameter
  count, file size) and measured accuracy for all three models, the
  whole reproduction pipeline's measured numbers, and, per author, two
  real handwriting sample crops plus every extracted character glyph
  variant currently stored for them (the same glyph-grid visualisation
  used during development to find and fix extraction bugs this project).

## What's deliberately NOT in this folder

- `Software/StyleSynthesis/` — an early prototype, no longer functional (imports a module that was later removed)
- `Software/GantryControl/motion_planner.py` / `motion_executor.ino` — the old ESP32/serial gantry control path, superseded by the direct-GPIO driver
- `AuthorReproductionStuff/ReproduceV2/` — a parked, image-output-only alternative pipeline using a pretrained diffusion model instead of this project's own glyph-library synthesiser; not connected to G-code/the gantry
- `TrainTextPersonal.py`, older label generators, `MyHandwriting/`'s comparative-method scripts, `RNNHandwriting.py`/`StyleTransferRNN.py`, `Diagnostics/`, `TestsForReport/` — one-off experiments, superseded attempts, or report-analysis scripts, not part of the delivered pipeline

See `../FILE_INVENTORY.md` for the full breakdown of every file in the
project, including all of the above, with dates and reasoning.
