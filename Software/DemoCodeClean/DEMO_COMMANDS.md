# Running the demo from the command line

Everything here runs on the ODROID N2+, from inside the `DemoCode` folder.
Set that once and the rest can be pasted as it stands:

```bash
cd ~/DemoCode            # wherever the folder sits on the board
export DEMO=$PWD
python3 -c "import paths; print(paths.Weights)"   # proves the tree is found
```

Every script finds its siblings itself, so they are run from this folder and
nothing needs `PYTHONPATH`. `python3`, not `python`.

Four environment knobs, all optional:

| variable | default | what it does |
|---|---|---|
| `PORT` | 5000 | the port the website listens on |
| `LINE_WORKERS` | 0 (one per core) | processes used to read and name lines; `1` forces one |
| `CROP_WORKERS` | 0 (one per core) | processes used to cut line crops out |
| `READER_WEIGHTS` | the scratch reader | a different reader checkpoint by file name |

---

## 1. The website

```bash
cd $DEMO && PORT=5000 python3 web/server.py
```

It prints the models it loaded and the eleven writers it found before it
serves anything — worth letting the room see:

```
[server] 11 style profiles: 150, 152, 153, 384, 552, 588, abhinav, dylan, owen, thiya, yeukita
```

Then `http://<board address>:5000` for Classify, Reproduce, Authors, Stats and
Training. Stop it with Ctrl-C.

---

## 2. One page segmented, with the results to look at

```bash
cd $DEMO
python3 examine.py --list                       # the pages on disk
python3 examine.py --writer dylan --page 1      # segment, read and name it
python3 examine.py --writer owen --page 3 --no-read     # segmentation only, no networks
python3 examine.py /path/to/photo.jpg --labels /path/to/truth.txt
```

It prints how the page was conditioned, every line the segmenter kept, what
the reader made of each, which writer each line votes for, and the page
verdict:

```
segment  24 lines, skew +1.0 deg, line pitch 48 px, paper coverage 98%
line  ink%   writer      conf   acc   text
   1   4.0  dylan       98.9% 100.0%  The input to this stage is a raw photo of
...
writer   dylan at 59.6% over 24 lines
         15/24 lines name that writer on their own
text     98.5% mean per line
page     98.6% reading the page as one string
```

The Classify page reports the same writer, the same 59.6% and the same 15/24
on this page. Its *page* text figure differs slightly (94.8% here) because it
measures per-line against its own alignment rather than joining the page into
one string — close, but not the same measurement, so quote one or the other.

and writes `overlay.png` (the page with a box round every line it cut) plus
one PNG per line into `Data/CNN/NOGIT/Examined/<page>/`. `--out DIR` puts them
somewhere else.

These are the same calls the website makes, through the same worker pool, so
the numbers match the Classify page. The same thing over HTTP, if the question
is about the web path specifically:

```bash
curl -s -X POST http://127.0.0.1:5000/api/classify \
     -F image=@Data/CNN/PersonalDataset/Dylan/Pages/Dylan01.jpg \
     -F expected_author=dylan | python3 -m json.tool | head -30
```

---

## 3. Training from scratch

**Read this before running it at a demonstration.** Both trainings write
their checkpoint under a fixed name, and that name is the one the website
loads. An interrupted run leaves a half-trained checkpoint where the shipped
one was. So either back the weights up:

```bash
cp -r $DEMO/Data/CNN/weights $DEMO/Data/CNN/weights_backup_$(date +%F)
```

or, better, give the run a tag so it is written *beside* the shipped
checkpoint instead of over it:

```bash
python3 train.py author --epochs 1 --tag demo     # -> author_shape_11w_pw170_scratch_demo_weights.npz
```

Tagged or not, the two networks and all their conditions:

**The reader** (CNN + BiLSTM + CTC, the text model):

```bash
python3 train.py text --epochs 40 --batch 8 --lr 1e-3 --seed 7 \
                      --source joint --personal-fraction 0.20
```

Those are the defaults, and they are the conditions behind the shipped
`paper_cnn_bilstm_ctc_scratch_best.npz`. `--source` is the thing worth
explaining:

| `--source` | what it trains on |
|---|---|
| `joint` (default) | the pre-cut public lines **and** the photographed writers, `--personal-fraction` of every batch drawn from the latter |
| `teklia` | the public line corpus only |
| `pages` | segments this repository's own database pages and trains on those |

**The namer** (the writer classifier). The shipped model has **eleven**
classes, and the default writer list is thirteen — 151 and 551 have pages on
disk but were dropped before it was trained. To reproduce what ships, name
the eleven:

```bash
python3 train.py author --epochs 40 --batch 8 --lr 1e-3 --seed 7 \
    --writers 150 152 153 384 552 588 abhinav dylan owen thiya yeukita
```

Leave `--writers` off and you get a thirteen-class model instead, which is a
different experiment, not the shipped one. The eleven are also in
`paths.ModelWriters`.

Both tasks also take:

```bash
--resume                     # continue from this task's own checkpoint
--warm-start FILE.npz        # start from a named checkpoint instead of noise
--pages N                    # text, --source pages: cut up only N pages
--data DIR                   # text: read the lines or pages from elsewhere
```

Checkpoints, the best-so-far copy and the epoch log all land in
`Data/CNN/weights/`. Watch it with:

```bash
tail -f $DEMO/Data/CNN/weights/author_shape_11w_pw170_scratch_log.txt
```

The same two runs are on the website's **Training** tab, which streams the log
into the page and holds one job at a time. Note the tab does **not** pass a
tag — a run started there writes the shipped name, so back the folder up first
if the examiner wants to watch training start.

**How long it takes.** One epoch of the namer over two writers (231 training
lines) took **7.7 minutes on a Windows host**, not on the board — I have not
timed either training on the ODROID, and the board is slower. Eleven writers
for forty epochs is therefore hours, not minutes. Show `--epochs 1` live and
talk over it; a full run is an overnight job.

---

## 4. Adding a new author

A writer needs photographs of their pages and a transcript of each: either a
`<page>_labels.txt` beside the image (or in a `Labels` folder next to
`Pages`, as the dataset keeps them), one written line per text line and in
order, or `--transcript FILE` for a single page.

**Always dry-run first.** It reports which written line was matched to which
transcript line and writes nothing, which is where a bad transcript or a
missed line shows up:

```bash
cd $DEMO
python3 enrol.py newwriter --folder /path/to/their/Pages --dry-run
```

```
--- page 1: 14 written lines, 9 transcript lines, 9 paired (floor 0.15)
  line  1 -> transcript  1   26.7%  '.Scinau). 15 18..'
  line  2 -> transcript  2   76.5%  ' Dauid was usefust'
  line  3 -> NOTHING            '.'  <- no transcript line matched this crop
```

A crop paired with NOTHING is usually a fragment the segmenter cut, and is
meant to be dropped. A *transcript* line matched to nothing means a written
line was missed — if several are, the photograph or the transcript is wrong.

Then build it:

```bash
python3 enrol.py newwriter --folder /path/to/their/Pages
python3 enrol.py newwriter page1.jpg page2.jpg page3.jpg      # or name the pages
python3 enrol.py newwriter page.jpg --transcript lines.txt    # a single page
```

It prints the letters it cut and the hand it measured, and writes
`Data/CNN/NOGIT/StyleProfiles10/newwriter.json`:

```
letters  43 characters, 148 variants, from 44 usable lines
hand     slant -7.5 deg, connectedness 0.74, x-height 31.0 px
namer    has NO class for this writer: they can be written, not judged,
         until it is retrained
```

Restart the server and the new writer appears in the Reproduce list. Useful
flags: `--out FILE` to write the profile elsewhere, `--force` to replace an
existing one.

**The honest limit, worth saying out loud at a demo:** a new writer can be
*written* but not *judged*. The namer's eleven classes are fixed by its
checkpoint, so a page in the new hand cannot be attributed until the namer is
retrained (§3) with their pages added to the dataset.

The website's **Training → Add author** does the same build from the same code
— a profile made either way is interchangeable. Over HTTP:

```bash
curl -s -X POST http://127.0.0.1:5000/api/authors/add \
     -F name=newwriter \
     -F images=@p1.jpg -F labels=@p1_labels.txt \
     -F images=@p2.jpg -F labels=@p2_labels.txt
curl -s http://127.0.0.1:5000/api/train/status | python3 -m json.tool
```

---

## 5. Other single runs an examiner might ask for

**Trace a page's own ink straight to G-code** — no recognition, no glyph
library, no synthesis, so whatever the gantry draws differently is mechanical.
This is the measuring instrument for the machine itself:

```bash
python3 retrace.py Data/CNN/PersonalDataset/Dylan/Pages/Dylan01.jpg \
        -o /tmp/dylan01.gcode --preview /tmp/dylan01.png
```

**Write a page in a writer's hand** (the Reproduce page, over HTTP):

```bash
curl -s -X POST http://127.0.0.1:5000/api/generate -H 'Content-Type: application/json' \
     -d '{"text":"the quick brown fox","author":"owen","nTries":14}' \
     | python3 -m json.tool | head -40
```

The reply carries the job id, the G-code line and pen-pulse counts, the
read-back of the written file and the writer the namer finds in it.

**Prove the G-code is what was measured** — send a file back in and have it
placed on a page and read again. A generated job and its own file read back
the same is the end-to-end check:

```bash
curl -s -X POST http://127.0.0.1:5000/api/gcode/upload \
     -F gcode=@/tmp/dylan01.gcode -F expected_text="the quick brown fox" \
     | python3 -m json.tool
```

**The gantry.** Calibrate first — until the axes are homed a millimetre is a
nominal figure and the pen will not go where it is asked:

```bash
curl -s http://127.0.0.1:5000/api/gantry/status   | python3 -m json.tool
curl -s -X POST http://127.0.0.1:5000/api/gantry/calibrate | python3 -m json.tool
curl -s http://127.0.0.1:5000/api/gantry/position | python3 -m json.tool

# send the pen to a coordinate (clamped to the work area server-side)
curl -s -X POST http://127.0.0.1:5000/api/gantry/jog -H 'Content-Type: application/json' \
     -d '{"x":100,"y":120}' | python3 -m json.tool
curl -s -X POST http://127.0.0.1:5000/api/gantry/pen -H 'Content-Type: application/json' \
     -d '{"down":true}'  | python3 -m json.tool
curl -s -X POST http://127.0.0.1:5000/api/gantry/pen -H 'Content-Type: application/json' \
     -d '{"down":false}' | python3 -m json.tool

# draw a generated job, and the stop that works while it is drawing
curl -s -X POST http://127.0.0.1:5000/api/write/<jobId> | python3 -m json.tool
curl -s -X POST http://127.0.0.1:5000/api/gantry/stop   | python3 -m json.tool
```

The same jog and pen controls are the **Pen** panel on the Reproduce page.

**What the build is running**, if asked which checkpoint is live:

```bash
curl -s http://127.0.0.1:5000/api/model_info | python3 -m json.tool
curl -s http://127.0.0.1:5000/api/authors    | python3 -m json.tool
curl -s http://127.0.0.1:5000/api/timings    | python3 -m json.tool   # the session's stage times
```

**A writer's letters as a picture** (the Authors page shows this):

```bash
curl -s http://127.0.0.1:5000/api/glyphs/dylan -o /tmp/dylan_glyphs.png
```

---

## 6. What will not work, and why

- **The gantry endpoints need the board.** They talk to the pins through
  `gpiod`. Anywhere else they answer `Gantry not connected: gpiod not
  available` and still report the work-area bounds.
- **Jog and pen refuse before calibration**, and refuse while an end-stop is
  tripped. That is deliberate.
- **One background job at a time.** A second training or add-author request is
  refused with what is already running, rather than queued.
- **A page is never read as one image.** The reader and the namer are both
  *line* models; the page is segmented first and the page verdict is the
  ink-weighted vote over its lines.
