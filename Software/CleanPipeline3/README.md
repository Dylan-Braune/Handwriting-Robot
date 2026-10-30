# CleanPipeline3

The runnable system: the pipeline, the web server, and both trainers. No test
scripts and no test data — those stay in CleanPipeline2.

## Layout

```
paths.py          where the data lives and who the writers are
segment.py        photograph -> conditioned page -> one image per text line
recognise.py      inference: the reader, the two writer namers, CTC decoding
style.py          glyph extraction and the measured style parameters
synthesise.py     glyphs + style -> a pen trajectory
reproduce.py      trajectory -> G-code, and G-code -> gantry motion
timing.py         stage timings and the per-character figures
train.py          training from first principles, numpy only
trainlibrary.py   the same two networks trained with torch, for speed
web/
  server.py       the HTTP API, with timings on every request
  camera.py       the page-scanning camera
  render.py       glyph sheets for the stats page
  frontend/       the web app itself (index, write, stats)
```

Data is **not** duplicated in here. `paths.py` resolves everything to
`Software/CNN`: weights in `weights/`, style profiles and caches in `NOGIT/`.

## Running it

On the single-board computer:

```sh
pip install flask flask-cors pillow numpy
python web/server.py
```

It listens on `0.0.0.0:5000` and serves the web app at `/`, so a browser on the
same network can open `http://<board-address>:5000` and nothing else needs
hosting. `PORT` overrides the port, `WEBAPP_API_KEY` requires a matching
`X-API-Key` header on every request, and `WEBAPP_TRIES` sets how many
candidates the synthesis search draws (default 14).

`opencv-python` is needed only for the camera endpoints; everything else runs
without it.

## What the server loads

| role | checkpoint | why |
|---|---|---|
| reader | `paper_cnn_bilstm_ctc_joint_best.npz` | the one the reported 96.5–99.1% was measured on |
| ink namer | `author_classifier_10new_weights.npz` | names the writer of a real photograph |
| shape namer | `author_shape_10new_weights.npz` | judges drawn output, which has no ink evidence |

The two namers are **not** interchangeable. A drawn line has one pen width
everywhere, so the ink namer has nothing to go on; the shape namer is trained
on stroke-normalised crops and is the one that scores synthesis candidates.

## Which way up the page is

`StandUp` in `server.py` settles the orientation from the page itself, on both
the upload and the camera path, so the two cannot disagree. It used to be only
`CAPTURE_ROTATE_DEG`, applied to camera captures alone, because an upload was
assumed to be a phone photo and already upright. The gantry webcam writes no
EXIF orientation tag, so the *same photograph* gave:

| path | before | after |
|---|---|---|
| camera button | dylan, 78.4% | dylan, 94.1% |
| upload of that saved file | **owen, 28.7%** | dylan, 94.1% |

The quarter turn is decided by `segment.LineAxisScore`, which counts the
fraction of rows that are nearly empty: the white bands between lines only
exist on the correct axis.

The turn has to be *earned* by `AXIS_TURN_MARGIN`, because leaving a page alone
is the safe answer. A plain "is sideways better" test turned `Abhinav05.jpg`,
an upright page of 30 tightly spaced lines, on 0.167 against 0.164 — its gaps
are narrow because it is dense, so the axes nearly tie and noise decides. It
then segmented into 16 slices across the writing and was named wrongly. Over 30
dataset pages and 3 webcam pages the ratio `s90/s0` is at most **1.02** on a
page that should not turn and at least **1.85** on one that should, so the
threshold sits at 1.35.

Autocorrelation of the row profile was tried instead of counting gaps, on the
theory that line spacing is regular whatever the density. It is worse: 6 of
those 33 pages wrong against 1.

Which *end* is up cannot be seen from row structure at all, because a page and
the same page upside down have the same one, so it is settled by reading
`ORIENT_PROBE_LINES` lines both ways and keeping whichever the reader is more
confident of.

## Naming the writer, and the per-line report

Every line votes, weighted by how much ink it carries, and `/api/classify`
returns a `lines` array with each line's writer, confidence, reading confidence
and character accuracy. The browser shows it as a table and offers it as
`/api/reports/<image_id>.csv` or `.json`.

Per-line character accuracy needs the expected text entered **one written line
per line**. A pasted paragraph gives the page figure only; the column is left
empty rather than guessed at.

Two figures are reported for the writer, and they answer different questions:
`author_confidence_pct` is one pooled posterior over the whole page, and
`author_line_agreement_pct` is how many lines would have named that writer on
their own with no pooling at all.

In CleanPipeline2 the reader's candidate list had the scratch checkpoint first,
which reads about a quarter of the characters correctly. That order is
corrected here: joint, then hf, then personal, then scratch.

## Timings

Every endpoint that does real work returns a `timing` object, and the web app
draws it as a table. Stages are reported separately because the requirements
are written per character, and a total alone cannot be checked against them.

```
timing for classify: 26.01 s total, 136 characters, 24 lines
  conditioning    8.006 s   30.8%
  segmenting      9.703 s   37.3%
  cropping        5.888 s   22.6%
  reading         1.688 s    6.5%  0.0124 s/char
  naming          0.430 s    1.7%  0.0032 s/char
  requirement 30 s: met (87% of the allowance)
  0.1913 s per character, 5.23 characters per second
```

`conditioning`, `segmenting` and `cropping` are recorded inside
`SegmentLines`, which takes an optional `timer` and is otherwise unchanged.
Drawing is timed end to end: `WriteGcodeFile` returns once the machine has run
the whole file and parked itself, so the wall time of that call is the real
drawing time.

`GET /api/timings` returns every run this server has measured, with means per
kind, for collecting figures without watching the browser.

The limits checked against are in `timing.py`: 30 s to read a page (R4) and
120 s to produce and draw a line (R3).

## The C1 join

`_Ligature` in `synthesise.py` shapes the join between two letters as a
quadratic curve whose control height follows **both** end heights. The earlier
version always dipped below the lower of the two, so a join that had to climb
first dived under the letter and hooked back up.

This is the `c1_base` arm of the four-arm comparison: it had the best character
accuracy of the four at 81.5%, and unlike C2, C3 and C4 it is a synthesis-time
change only, so it works with the glyph libraries already on disk and needs no
profile rebuild.

Synthesis goes through `SynthesizeJointBestOf`, which draws `nTries`
candidates and keeps whichever scores best on the harmonic mean of how
readable it is and how much it looks like the requested writer. The harmonic
mean rather than the average, so a candidate weak on either axis is penalised
instead of being carried by the other.

## Training

Two routes to the same networks, with the same architecture and the same
recipe. Parameter names match, so a checkpoint from either loads into the
other and into `recognise.py` without renaming.

```sh
python train.py                       # from first principles, numpy only
python trainlibrary.py text --epochs 40
python trainlibrary.py author --epochs 40
```

`train.py` carries its own PNG and JPEG decoders, so it needs nothing outside
numpy and the standard library to read the database pages and the photographed
ones. Measured: about 1.1 s per line for the reader and 1.2 s for the namer,
single-threaded. `trainlibrary.py` is roughly five times faster on the same
machine and writes its weights as `.npz` in the layout `recognise.py` reads.

## Known limits

- A full 24-line page takes about 0.56 s per line to read, so reading alone
  adds roughly 13 s to the 24 s that conditioning and segmentation cost. A
  dense page will therefore land near or above the 30 s allowance; the stage
  table shows exactly where it goes.
- `recognise.py`'s `Sigmoid` overflows on large negative inputs and prints a
  warning. The result is correct — it underflows to zero — but the warning is
  noisy. Run with `-W ignore` or rewrite it in the stable form.
- `accelMmS2` in `reproduce.py` is declared and never read; there is no
  acceleration limiting, which is the likeliest source of lost steps on a fast
  travel move.
- Naming every line instead of six, and probing the orientation, cost time.
  Measured on the same 16-line page, camera path: **17.9 s before, 24.1 s
  after** (reading 6.3 s to 8.1 s, naming 1.6 s to 5.0 s). Still inside the
  30 s page allowance, but the per-character allowance in `timing.py`
  (2 s + 0.025 s/char) was already exceeded before this change and is now
  exceeded by more. Naming is the bigger half and is the price of the per-line
  table; if it has to come back, sample the *vote* but keep reporting every
  line, and never with `linspace`, which always picks the last crop.
