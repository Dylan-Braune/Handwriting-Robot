"""Web server for the handwriting robot.  Runs on the single-board computer.

    pip install flask flask-cors pillow
    python web/server.py

It listens on 0.0.0.0:5000, serves the pages in web/frontend, and exposes the
pipeline over json.  Every request that does real work returns a `timing`
object alongside its result: the wall time of each stage, and that time per
character, because the requirements are written per character and a bare
total cannot be checked against them.

Stages reported
    classify     conditioning, segmenting, cropping, reading, naming
    reproduce    synthesis, rendering, gcode, drawing

Security: no accounts and almost no input checking.  Meant for a local
network.  If it must be reachable from outside, set WEBAPP_API_KEY and every
request then needs a matching X-API-Key header, which is a deterrent and not
real security.

Endpoints
    GET  /                            the web app itself
    GET  /api/authors                 writers with a style profile on disk
    POST /api/camera/capture          take a photograph, returns an imageId
    POST /api/camera/preview/start    open the camera for live framing
    POST /api/camera/preview/stop     release it
    GET  /api/camera/preview/stream   mjpeg stream while open
    GET  /api/images/<imageId>        a captured, uploaded or rendered image
    POST /api/classify                read a page and name its writer
    POST /api/generate                write text in a writer's hand, make gcode
    POST /api/gcode/upload            send a hand made gcode file through the same flow
    POST /api/write/<jobId>           draw a job on the gantry
    GET  /api/gantry/status           connected and calibrated or not
    POST /api/gantry/calibrate        home the axes and measure steps per mm
    GET  /api/gcode/<jobId>           the raw gcode of a job
    GET  /api/model_info              which checkpoints are loaded
    GET  /api/timings                 every timing this server has measured
    GET  /api/gantry/position         where the carriage is, and the bounds
    POST /api/gantry/jog              send the pen to one point, in millimetres
    POST /api/gantry/pen              put the pen down or lift it
    POST /api/train/start             train the reader or the namer
    GET  /api/train/status            how that run is going, with its log
    GET  /api/train/log               that log as plain text
    POST /api/train/stop              end it
    POST /api/authors/add             build a glyph profile from page images
"""
import csv
import dataclasses
import io
import json
import os
import re
import sys
import threading
import uuid
import time
from pathlib import Path

import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
CP3 = HERE.parent
#the pipeline modules sit one level up, camera and render sit beside this
#file.  python does not always add a script's own folder to the import path,
#so both are named explicitly rather than relied on
sys.path.insert(0, str(CP3))
sys.path.insert(0, str(HERE))

import paths
import recognise as REC
import reproduce as REP
import segment as SEG
import synthesise as SYN
import retrace as RETRACE
import timing as TIMING

import camera as CAM
import lineworkers as LW
import render as RENDER

FRONTEND = HERE/"frontend"
UPLOADS = paths.Cache/"WebUploads"
JOBS = paths.Cache/"WebJobs"
REPORTS = paths.Cache/"WebReports"
UPLOADS.mkdir(parents=True, exist_ok=True)
JOBS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

#----------------------------------------------------------------------
#A CAPTURED PAGE IS TURNED A QUARTER TURN LEFT AS IT IS SAVED
#the camera is mounted across the gantry, so a page photographed on the
#bed arrives on its side.  turning it here means the copy on disk is the
#right way up: the browser shows that file and the segmenter reads that
#same file, and the two cannot drift apart.
#
#this is now only a HINT, not the thing that makes reading work.  it used
#to be the only turn in the system, and an uploaded file got none at all,
#on the assumption that an upload is a phone photo and already upright.
#the webcam writes no EXIF orientation tag, so the same photograph read
#as dylan at 78.4 per cent through the camera button and as owen at 28.7
#through upload.  StandUp now settles the axis from the page itself on
#both paths, and is idempotent, so whatever this is set to the page ends
#up the right way round.  it is kept because turning the capture as it is
#saved still makes the saved thumbnail look right in the browser
#
#0 switches it off.  only quarter turns are meaningful here
#----------------------------------------------------------------------
CAPTURE_ROTATE_DEG = 90   #90 is a quarter turn anticlockwise, ie to the left

#---NAME_FROM_LINES: WITHDRAWN, every line now names the writer---
#this used to take 6 lines spread down the page with linspace.  linspace
#always includes the LAST index, so a junk crop at the foot of a page was
#in every pool and at 2 lines was half of it.  on the page that exposed
#this all 16 real lines named the writer right on their own, and the
#pooled answer still came back at 78.4 per cent because the 17th crop was
#a blot.  dropping blots in the segmenter and letting every line vote
#puts the same page at 94.1 per cent.  naming all 16 lines rather than 6
#costs about 4.3 s, which is what the per line table needs anyway
NAME_FROM_LINES = 0   #kept so an older caller setting it is not a crash

#---ORIENT_PROBE_LINES: lines read both ways round to settle which end is up---
#3 is enough: the gap between upright and inverted is large and consistent,
#and probing every line doubled the read stage for the same answer
ORIENT_PROBE_LINES = 3


def OpenUpright(src):
    """Open an image with its EXIF orientation applied to the pixels.

    A phone or webcam does not usually turn the sensor frame round when the
    camera is held sideways.  It writes the pixels as the sensor read them
    and records a separate orientation tag saying how far to turn them.
    Every viewer honours that tag, so the file LOOKS upright, but PIL hands
    back the raw pixels and ignores it, so the segmenter was reading a page
    lying on its side while the same file looked correct on screen.

    That is not a small error.  A sideways page makes the horizontal
    projection cut ACROSS the writing instead of between the lines, so the
    segmenter returned about forty narrow vertical slices of text and the
    recogniser read gibberish from them.  Measured on the nine photographed
    gantry pages: 5.7 per cent mean character accuracy before this call,
    65.1 per cent after it, best page 90.2 per cent.

    exif_transpose is a no-op on a file with no orientation tag, which is
    what cv2.imwrite produces for the webcam captures, so this is safe to
    use on every path into the system.
    """
    return ImageOps.exif_transpose(Image.open(src))


def IngestImage(src):
    """Open a page just taken by the camera and stand it the right way up."""
    img = OpenUpright(src)
    deg = int(CAPTURE_ROTATE_DEG) % 360
    if deg:
        #expand keeps the whole page after a quarter turn instead of
        #cropping it back to the old frame
        img = img.rotate(deg, expand=True)
    return img

API_KEY = os.environ.get("WEBAPP_API_KEY", "")
PORT = int(os.environ.get("PORT", "5000"))

#how many candidates the joint search draws per line.  14 is the tuned value:
#raising it from 6 lifted the reported writer identification to 97.5 per cent
#and text accuracy to 81.6 per cent, and it costs time linearly
DEFAULT_TRIES = int(os.environ.get("WEBAPP_TRIES", "14"))

#---how finely a job is drawn, in pixels per millimetre---
#the tight preview is the picture of the pen path on its own, cropped to the
#ink, and 18 is what the synthesiser's own scoring uses.  the PAGE render is
#the whole sheet with the writing where the machine will actually put it, so
#it is several thousand pixels across and 10 is enough: at a 4 mm x-height
#that is a 40 px x-height, which is the size the reader was trained on
PREVIEW_PX_PER_MM = 18.0
PAGE_PX_PER_MM = 10.0

app = Flask(__name__, static_folder=None)
CORS(app)


#======================================================================
#MODELS, LOADED ONCE
#the reader is the joint checkpoint, which is what the reported accuracy
#was measured on.  there are two writer namers and they are not
#interchangeable: the ink one reads real photographs, the shape one
#judges drawn output, which has one pen width everywhere and so carries
#no ink evidence at all
#======================================================================
#which reader checkpoint to load.  recognise.py names the scratch one, which
#is the weakest of the four on disk; READER_WEIGHTS picks another by name for
#a run without editing recognise.py itself
_reader = os.environ.get("READER_WEIGHTS")
READER_PATH = (paths.Weights/_reader) if _reader else REC.DEFAULT_TEXT_WEIGHTS
print("[server] loading the reader from %s" % READER_PATH.name)
TEXT_MODEL = REC.PaperCRNNNumpy(checkpointPath=READER_PATH)

print("[server] loading the ink namer from %s" % REC.DEFAULT_AUTHOR_WEIGHTS.name)
INK_MODEL = REC.AuthorClassifierCNNNumpy(checkpointPath=REC.DEFAULT_AUTHOR_WEIGHTS)
INK_MAP = INK_MODEL.authorMapping
INK_BY_INDEX = {v: k for k, v in INK_MAP.items()}

#this tree ships one namer, the shape one, so the ink handle above and the
#shape handle here are the same loaded model rather than two checkpoints
print("[server] loading the shape namer from %s" % REC.DEFAULT_AUTHOR_WEIGHTS.name)
SHAPE_MODEL = INK_MODEL
SHAPE_MAP = SHAPE_MODEL.authorMapping

#======================================================================
#WHICH NAMER NAMES A PHOTOGRAPHED PAGE
#the ink namer used to do this, on raw crops.  it reads ink statistics,
#pen width and darkness, as much as it reads handwriting, and a page
#photographed on a different day with a different pen is then a different
#writer.  one of yeukita's pages came back as owen at 84 per cent, with
#0 of its 13 lines naming her.
#
#measured four ways over 15 labelled pages, with the crops perturbed to
#imitate exactly that change:
#
#   config       none   thicker  thinner  faint   gamma   overall
#   ink_raw      15/15   15/15    12/15   12/15   15/15    92%
#   ink_norm     13/15   13/15    15/15   13/15   14/15    91%
#   shape_raw    15/15   15/15    13/15   15/15   15/15    97%
#   shape_norm   15/15   15/15    15/15   15/15   15/15   100%
#
#the ink namer fails precisely where the theory says it should, on a
#finer pen and on fainter ink.  over the 30 labelled pages plus the one
#misnamed photograph, shape_raw is the only setting that gets all 31
#right; shape_norm is steadier still under the simulated change but gets
#that real photograph wrong, and ink_norm loses 3 of the 30.
#
#this decides the label for a PHOTOGRAPH only.  synthesis still scores
#its candidates with the shape namer on stroke normalised crops, which
#is a different question and is not touched
#
#----------------------------------------------------------------------
#SET BACK TO TRUE, 18 November 2026, and the sweep above is kept rather
#than deleted because it is real: on PHOTOGRAPHED pages raw crops did
#win, 31 of 31 against 30 of 31.
#
#it was the wrong setting all the same, because the namer only ever saw
#crops that StrokeNormalize had REDRAWN -- skeletonised to a centreline
#and re-inked at 0.17 x-heights, which is what "pw170" in its file name
#records.  raw ink is therefore outside the distribution it was trained
#on whatever the page, and the 31-of-31 says the margin on those
#photographs was wide enough to absorb that, not that raw ink is sound.
#
#measured on the 185 stored line crops of the fourteen pages the gantry
#drew, one checkpoint, this flag the only difference:
#
#   strokeNormalise=True    159/185 = 85.95 per cent
#   strokeNormalise=False   147/185 = 79.46 per cent
#
#and the loss is not spread evenly.  per writer, normalised then raw:
#yeukita 12/13 -> 2/13, dylan 15/16 -> 9/16, 552 and 152 -2 each, while
#153 and 150 actually gain.  the True figure also reproduces the
#recorded table for these pages exactly, cell for cell, which the False
#figure cannot.
#
#NOT the explanation, though it looks like one: distance from the 0.17
#training width does not predict who suffers.  measured median pen width
#per writer, dylan sits 0.010 x-heights from 0.17 and loses six lines,
#while thiya sits 0.600 away and loses none.  so this is not "writers
#with an unusual pen" -- it is that thickness carries writer information
#the model was never shown varying, and which writers that helps or
#hurts is not predictable from the width alone
#----------------------------------------------------------------------
#======================================================================
PAGE_NAMER = SHAPE_MODEL
PAGE_NAMER_STROKE_NORM = True

#the per-line pool uses exactly these, so a pooled page and a sequential
#page are the same computation
LW.Configure(TEXT_MODEL, PAGE_NAMER, PAGE_NAMER_STROKE_NORM)
PAGE_BY_INDEX = {v: k for k, v in PAGE_NAMER.authorMapping.items()}
print("[server] naming photographed pages with %s, strokeNormalise=%s"
      % ("the shape namer" if PAGE_NAMER is SHAPE_MODEL else "the ink namer",
         PAGE_NAMER_STROKE_NORM))


#---LoadProfiles: every style profile on disk, by writer---
#one file per writer, named after the writer and nothing else.  the curated
#libraries replaced the raw ones in place, so there is no longer a second
#file per writer with a suffix the models know nothing about
def LoadProfiles():
    out = {}
    for f in sorted(paths.Profiles.glob("*.json")):
        try:
            out[f.stem] = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print("[server] skipping profile %s: %s" % (f.name, e))
    return out


PROFILES = LoadProfiles()
print("[server] %d style profiles: %s" % (len(PROFILES), ", ".join(sorted(PROFILES))))

#every timing this server has measured, newest last, for the stats page
TIMINGS = []
TIMINGS_LOCK = threading.Lock()


def Remember2(d):
    """record an already-built timing dict, for work timed outside a Run"""
    with TIMINGS_LOCK:
        TIMINGS.append(d)
        del TIMINGS[:-200]
    return d


def Remember(run):
    d = run.AsDict()
    with TIMINGS_LOCK:
        TIMINGS.append(d)
        del TIMINGS[:-200]        #keep the last two hundred, no more
    print(run.Report(), flush=True)
    return d


@app.before_request
def CheckApiKey():
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        abort(401)


#======================================================================
#THE WEB APP ITSELF
#======================================================================
@app.route("/")
def Index():
    return send_from_directory(FRONTEND, "index.html")


@app.route("/<path:name>")
def Static(name):
    if (FRONTEND/name).exists():
        return send_from_directory(FRONTEND, name)
    abort(404)


#======================================================================
#WRITERS AND IMAGES
#======================================================================
@app.route("/api/authors")
def ApiAuthors():
    out = []
    for a in sorted(PROFILES):
        p = PROFILES[a]
        out.append(dict(id=a,
                        glyphs=len(p.get("glyphs") or {}),
                        variants=sum(len(v) for v in (p.get("glyphs") or {}).values()),
                        lines=p.get("nLines"),
                        #the style page prints these, and was showing
                        #"undefined" for both because they were measured into
                        #the profile and then never sent
                        slantDeg=p.get("slantDeg"),
                        connectedness=p.get("connectedness"),
                        xHeightPx=p.get("xHeightPx"),
                        wordSpaceXh=p.get("wordSpaceXh"),
                        ascender=p.get("ascender"),
                        descender=p.get("descender"),
                        legibility=p.get("legibilityLambda"),
                        inGantryModels=a in SHAPE_MAP))
    return jsonify(out)


@app.route("/api/images/<imageId>")
def ApiGetImage(imageId):
    safe = "".join(c for c in imageId if c.isalnum() or c == "_")
    if safe.startswith("page_"):
        #the whole sheet, with the writing where the machine will put it
        p = JOBS/safe[5:]/"page.png"
    elif safe.startswith("job_"):
        p = JOBS/safe[4:]/"preview.png"
    else:
        p = UPLOADS/("%s.jpg" % safe)
    if not p.exists():
        abort(404)
    return send_file(str(p))


@app.route("/api/camera/capture", methods=["POST"])
def ApiCameraCapture():
    try:
        captured = CAM.CaptureImage()
    except Exception as e:
        return jsonify({"error": "Camera failed: %s" % e}), 200
    imageId = uuid.uuid4().hex
    IngestImage(captured).convert("RGB").save(UPLOADS/("%s.jpg" % imageId))
    return jsonify({"image_id": imageId, "image_url": "/api/images/%s" % imageId})


@app.route("/api/camera/preview/start", methods=["POST"])
def ApiPreviewStart():
    try:
        CAM.PreviewStart()
    except Exception as e:
        return jsonify({"active": False, "error": str(e)}), 200
    return jsonify({"active": True})


@app.route("/api/camera/preview/stop", methods=["POST"])
def ApiPreviewStop():
    CAM.PreviewStop()
    return jsonify({"active": False})


@app.route("/api/camera/preview/stream")
def ApiPreviewStream():
    if not CAM.PreviewActive():
        abort(409)
    def Frames():
        #one jpeg per part, which is what an img tag in the page expects
        while CAM.PreviewActive():
            try:
                jpeg = CAM.PreviewFrameJpeg()
            except RuntimeError:
                break
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+jpeg+b"\r\n")
    return Response(Frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


#======================================================================
#READING A PAGE
#conditioning, segmenting and cropping are timed inside the segmenter,
#======================================================================
#ORIENTATION, DECIDED FROM THE PAGE AND NOT FROM WHERE IT ARRIVED
#a capture used to be turned by a fixed CAPTURE_ROTATE_DEG while an
#upload was not turned at all, on the assumption that an uploaded file
#is a phone photograph and so already upright.  the webcam on the gantry
#writes no EXIF orientation tag, so the SAME photograph read correctly
#through the camera button and as nonsense when its saved file was
#uploaded: 91.1 per cent of characters against 19.8.
#
#both paths go through this now, so they cannot disagree.  it is also
#idempotent, which is why the camera path can go on standing its capture
#up as it saves it: a page that is already upright is left alone
#======================================================================
def StandUp(gray):
    """Put a page on its correct axis.  Returns (page, degrees, scores)."""
    up, deg, s0, s90 = SEG.UprightPage(gray)
    return up, deg, {"axis_0": round(s0, 3), "axis_90": round(s90, 3)}


def ReadAllLines(crops):
    """Read every crop, and settle which end of the page is up.

    The axis test cannot tell a page from the same page turned through
    180 degrees, because the rows of writing fall in the same places
    either way.  The reader can.  Turning the page over is the same as
    turning every crop over and reversing their order, so it is read both
    ways here and whichever the reader is more confident of wins.  That
    needs no second segmentation.

    Only a SAMPLE of lines is read both ways.  Reading every line twice
    settled it too, and identically on the test page, but it put the read
    stage up from 5.9 s to 11.9 s on a sixteen line page, which is most
    of the per character allowance spent proving something three lines
    already show beyond doubt.  The sampled lines are not read again.
    """
    def Read(c, flip):
        img = np.rot90(c, 2) if flip else c
        return REC.ReadTextScored(Image.fromarray(img).convert("L"), TEXT_MODEL)

    n = len(crops)
    probe = list(np.unique(np.linspace(0, n-1, min(ORIENT_PROBE_LINES, n))
                           .round().astype(int)))
    fwd = {i: Read(crops[i], False) for i in probe}
    rev = {i: Read(crops[i], True) for i in probe}
    sF = float(np.mean([fwd[i][1] for i in probe]))
    sR = float(np.mean([rev[i][1] for i in probe]))

    flip = sR > sF
    cache = rev if flip else fwd
    #the probed lines are already read; the rest are independent of one
    #another and go out to the pool
    todo = [(i, crops[i], flip) for i in range(n) if i not in cache]
    done = {i: (t, sc) for i, t, sc in LW.Map(LW.ReadOne, todo)}
    out = [cache[i] if i in cache else done[i] for i in range(n)]
    if flip:
        #the order the lines come back in changes too, so the caller is
        #given it and can put everything else it holds per line in step
        idx = list(range(n))[::-1]
        return ([np.rot90(crops[i], 2) for i in idx], [out[i][0] for i in idx],
                [out[i][1] for i in idx], idx, 180, sF, sR)
    idx = list(range(n))
    return (list(crops), [t for t, _s in out], [s for _t, s in out], idx, 0, sF, sR)


def SplitExpected(expectedText, n):
    """Per line truth, but only when the expected text really is per line.

    If the user pasted a paragraph, there is no honest way to say which
    written line each word belongs to, so this returns None and the per
    line accuracy columns are left empty rather than guessed at.
    """
    got = [l.strip() for l in (expectedText or "").splitlines() if l.strip()]
    return got if len(got) == n and n > 0 else None


def WriteReport(reportId, payload):
    """Save the per line table as CSV and JSON, and return where it went."""
    (REPORTS/("%s.json" % reportId)).write_text(json.dumps(payload, indent=2),
                                                encoding="utf-8")
    cols = ["line", "author_named", "author_confidence_pct",
            "page_author_probability_pct", "char_accuracy_pct",
            "word_accuracy_pct", "read_confidence", "ink_pct",
            "text_read", "text_expected"]
    with (REPORTS/("%s.csv" % reportId)).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for k in ("image_id", "predicted_author", "author_confidence_pct",
                  "author_line_agreement_pct", "author_line_correct_pct",
                  "text_accuracy_pct", "word_accuracy_pct",
                  "mean_line_char_accuracy_pct", "lines_detected",
                  "lines_dropped_as_blots", "orientation_applied_deg"):
            w.writerow(["# %s" % k, payload.get(k)])
        w.writerow([])
        w.writerow(cols)
        for r in payload["lines"]:
            w.writerow([r.get(c) for c in cols])
    return "/api/reports/%s" % reportId


@app.route("/api/reports/<reportId>")
@app.route("/api/reports/<reportId>.<ext>")
def ApiReport(reportId, ext="csv"):
    safe = "".join(c for c in reportId if c.isalnum())
    if ext not in ("csv", "json"):
        abort(404)
    p = REPORTS/("%s.%s" % (safe, ext))
    if not p.exists():
        abort(404)
    return send_file(str(p), as_attachment=True,
                     download_name="per_line_report_%s.%s" % (safe, ext))


#---WordAcc: fraction of whole words that came back exactly right---
#kept here because the recognise.py in this tree exports CharAcc only
def WordAcc(pred,truth):
    tw,pw = truth.split(),pred.split()
    if not tw:
        return 1.0
    m,n = len(tw),len(pw)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            dp[i+1][j+1] = (dp[i][j]+1 if tw[i].lower()==pw[j].lower()
                            else max(dp[i][j+1],dp[i+1][j]))
    return dp[m][n]/m


#======================================================================
#then reading and naming are timed here, so every stage of the classify
#path appears separately with its own per character cost
#======================================================================
@app.route("/api/classify", methods=["POST"])
def ApiClassify():
    expectedText = (request.form.get("expected_text") or "").strip()
    expectedAuthor = request.form.get("expected_author") or ""

    run = TIMING.Run("classify")
    with run.Step("loading", chars=0):
        if "image" in request.files and request.files["image"].filename:
            #OpenUpright, not Image.open: an uploaded photograph carries its
            #orientation in EXIF, and the saved copy below is what the page
            #and the segmenter both read afterwards, so it is turned ONCE
            #here and the two can never disagree
            page = OpenUpright(request.files["image"].stream).convert("L")
            imageId = uuid.uuid4().hex
        elif request.form.get("image_id"):
            imageId = "".join(c for c in request.form["image_id"] if c.isalnum())
            p = UPLOADS/("%s.jpg" % imageId)
            if not p.exists():
                return jsonify({"error": "Unknown image_id"}), 400
            page = Image.open(p).convert("L")
        else:
            return jsonify({"error": "Provide either 'image' or 'image_id'"}), 400
        #stood on its correct axis HERE, however it arrived, and the copy
        #saved to disk is the stood up one, so the picture the browser
        #shows and the one the segmenter reads stay the same picture.  an
        #already upright page scores higher on its own axis and is left be
        gray, axisDeg, axisScores = StandUp(np.asarray(page, dtype=np.uint8))
        Image.fromarray(gray).convert("RGB").save(UPLOADS/("%s.jpg" % imageId))

    #this tree's SegmentLines takes no timer, so conditioning, segmenting and
    #cropping cannot be reported apart here and the whole of it is one stage
    with run.Step("segmenting", chars=0):
        lines, _dbg, info = SEG.SegmentLines(gray)
    crops = [np.asarray(ln["image"]) for ln in lines]
    if not crops:
        return jsonify({"error": "No handwriting lines were detected.",
                        "timing": Remember(run)}), 422

    with run.Step("reading", chars=0) as step:
        crops, readLines, readScores, order, flipDeg, sF, sR = ReadAllLines(crops)
        lines = [lines[i] for i in order]   #keep the crop metadata in step
        step.chars = sum(len(t) for t in readLines)
    predictedText = " ".join(readLines)

    with run.Step("naming", chars=len(predictedText)):
        perLineProbs, avg, inkWeights = LW.NameWriter(crops)
        bestIdx = int(np.argmax(avg))
        predictedAuthor = PAGE_BY_INDEX[bestIdx]

    #---the per line table, which is what the downloadable report holds---
    truthLines = SplitExpected(expectedText, len(crops))
    named = [PAGE_BY_INDEX[int(np.argmax(p))] for p in perLineProbs]
    agree = sum(1 for n in named if n == predictedAuthor)
    correct = (sum(1 for n in named if n == expectedAuthor)
               if expectedAuthor else None)

    lineRows = []
    for i, p in enumerate(perLineProbs):
        t = truthLines[i] if truthLines else None
        lineRows.append({
            "line": i+1,
            "author_named": named[i],
            "author_confidence_pct": round(100*float(p[int(np.argmax(p))]), 1),
            "page_author_probability_pct": round(100*float(p[bestIdx]), 1),
            "author_matches_page": named[i] == predictedAuthor,
            "author_correct": (named[i] == expectedAuthor) if expectedAuthor else None,
            "char_accuracy_pct": (round(100*REC.CharAcc(readLines[i], t), 1)
                                  if t is not None else None),
            "word_accuracy_pct": (round(100*WordAcc(readLines[i], t), 1)
                                  if t is not None else None),
            "read_confidence": round(readScores[i], 3),
            "ink_pct": round(float(lines[i].get("inkPct", 0.0)), 2),
            "text_read": readLines[i],
            "text_expected": t,
        })

    meanLineChar = ([r["char_accuracy_pct"] for r in lineRows] if truthLines else None)
    run.chars = len(predictedText)
    run.lines = len(crops)

    payload = {
        "image_id": imageId,
        "predicted_text": predictedText,
        "predicted_lines": readLines,
        "text_accuracy_pct": (round(100*REC.CharAcc(predictedText, expectedText), 1)
                              if expectedText else None),
        "word_accuracy_pct": (round(100*WordAcc(predictedText, expectedText), 1)
                              if expectedText else None),
        #the mean of the per line figures, which is the one quoted per page
        "mean_line_char_accuracy_pct": (round(float(np.mean(meanLineChar)), 1)
                                        if meanLineChar else None),
        "predicted_author": predictedAuthor,
        #the pooled posterior over every line, weighted by ink
        "author_confidence_pct": round(100*float(avg[bestIdx]), 1),
        #and how many lines would have said so entirely on their own
        "author_line_agreement_pct": round(100.0*agree/len(crops), 1),
        "author_lines_agreeing": "%d/%d" % (agree, len(crops)),
        "author_line_correct_pct": (round(100.0*correct/len(crops), 1)
                                    if correct is not None else None),
        "author_correct": (predictedAuthor == expectedAuthor) if expectedAuthor else None,
        "lines_detected": len(crops),
        "lines_dropped_as_blots": info.get("dropped", 0),
        "orientation_applied_deg": (axisDeg+flipDeg) % 360,
        "orientation_scores": dict(axisScores, read_upright=round(sF, 3),
                                   read_inverted=round(sR, 3)),
        "page_angle_deg": info.get("angle"),
        "line_pitch_px": info.get("pitch"),
        "per_author_confidence_pct": {PAGE_BY_INDEX[i]: round(100*float(v), 1)
                                      for i, v in enumerate(avg)},
        "lines": lineRows,
        "timing": Remember(run),
    }
    payload["report_url"] = WriteReport(imageId, payload)
    return jsonify(payload)


#======================================================================
#WRITING TEXT IN A WRITER'S HAND
#the joint search draws nTries candidates and keeps whichever scores best
#on the harmonic mean of how readable it is and how much it looks like
#the requested writer, which is the selection that lifted both figures
#======================================================================
def ReadBackPage(previewImg, pxPerMm):
    """Read a written page back the way a photographed one is read.

    Returns (text, lineCrops).

    REC.ReadText is a LINE recogniser: it runs ResizeLineImageFixed, which
    squashes whatever it is given into a single strip a few dozen pixels
    tall.  Handing it a whole multi-line page crushes the lines on top of
    one another and it returns gibberish -- measured at 24.1 per cent on a
    three line job that reads 83.0 per cent when the lines are taken one at
    a time.  So the page goes through the same segmenter a photograph does,
    padded first, because the segmenter needs paper around the writing:
    with no margin it clipped the first and last letters and scored
    64.7 per cent against 94.1 once padded.

    The crops are handed back because NAMING the writer has exactly the
    same problem.  REC.ClassifyImage is a LINE classifier too, and
    /api/generate was calling it on the whole rendered page.  A dylan page
    that reads back at 95.4 per cent came back named as owen with dylan at
    4.6 per cent -- not a judgement about the handwriting, but the namer
    being shown four lines of writing stacked on top of each other.

    So the page is segmented ONCE here and the crops are reused: the reader
    reads them and NameWriter votes over them, ink-weighted, which is
    precisely how a photographed page is named on the classify path.  The
    two paths now agree because they are the same computation.
    """
    arr = np.asarray(previewImg, dtype=np.uint8)
    pad = max(8, int(8.0*float(pxPerMm or 8.0)))   #8 mm of paper, as the writers' own sheets have
    padded = np.full((arr.shape[0]+2*pad, arr.shape[1]+2*pad), 255, np.uint8)
    padded[pad:pad+arr.shape[0], pad:pad+arr.shape[1]] = arr
    try:
        #mergeValley is asked for HERE and nowhere else.  it was added when
        #this read the tight trajectory render, which is drawn at one pen
        #width with no paper behind it and on which a single line's descender
        #band is readily taken for a second line.  now that the input is a
        #whole sheet rendered from the G-code it makes no difference either
        #way: measured on four-line pages from two writers it gave the same
        #4 lines and the same 95.4 and 97.7 per cent with it and without it.
        #it is kept because it costs nothing and still covers the tight case
        lines, _dbg, _info = SEG.SegmentLines(padded, mergeValley=0.72)
    except Exception:
        #a read-back is a report, never the job: if segmenting throws, say
        #what the single-line path says rather than failing the request
        return REC.ReadText(previewImg, TEXT_MODEL), []
    if not lines:
        #nothing was found to read, so the whole image is the last resort
        return REC.ReadText(previewImg, TEXT_MODEL), []
    #a one-line result used to be read as the WHOLE image here, on the
    #reasoning that for a single line the crop can only lose edges the full
    #image still has.  that held when the input was the trajectory render,
    #cropped tight to the ink, where the image WAS the line.  it is false
    #for a page: a one-line job on a 190 by 250 mm sheet is a strip of
    #writing across the top of 2500 rows of blank paper, and squashing all
    #of it into the reader's fixed-height strip returned "." for a line the
    #segmenter had already cut correctly and which reads at 85 per cent.
    #the crop the segmenter found is the line, however many it found
    crops = [np.asarray(ln["image"]) for ln in lines]
    text = " ".join(filter(None, (
        REC.ReadText(Image.fromarray(c).convert("L"), TEXT_MODEL) for c in crops)))
    return text, crops


def JobResponse(jobId, written, previewImg, run, expectedText=None,
                pxPerMm=8.0, nameWriter=False, askedFor=None, **extra):
    with run.Step("reading back", chars=len(expectedText or "")):
        readBack, crops = ReadBackPage(previewImg, pxPerMm)
    named = conf = None
    linesNaming = None
    if nameWriter and crops:
        #one writer for the page, from every line, ink-weighted: the same
        #NameWriter a photographed page goes through
        with run.Step("naming", chars=len(readBack or "")):
            perLine, avg, _w = LW.NameWriter(crops)
        named = PAGE_BY_INDEX[int(np.argmax(avg))]
        if askedFor in SHAPE_MAP:
            conf = round(100*float(avg[SHAPE_MAP[askedFor]]), 1)
            agree = sum(1 for p in perLine
                        if PAGE_BY_INDEX[int(np.argmax(p))] == askedFor)
            linesNaming = "%d/%d" % (agree, len(perLine))
    body = {
        "job_id": jobId,
        "gcode_url": "/api/gcode/%s" % jobId,
        "preview_url": "/api/images/job_%s" % jobId,
        "gcode_line_count": written.get("lines"),
        "pen_pulses": written.get("penPulses"),
        "read_back_text": readBack,
        "read_back_lines": len(crops),
        "writer_named": named,
        "writer_confidence_pct": conf,
        "writer_lines_naming_asked_for": linesNaming,
        "text_accuracy_pct": (round(100*REC.CharAcc(readBack, expectedText), 1)
                              if expectedText else None),
    }
    body.update(extra)
    body["timing"] = Remember(run)
    return jsonify(body)


def _Clamp(value, default, lo, hi):
    """A number from the request, held inside a range, falling back if absent."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    #a slider that has been dragged to zero should not silently become a
    #machine that never moves, so the floor is a real feed rate
    return float(min(max(v, lo), hi))


@app.route("/api/generate", methods=["POST"])
def ApiGenerate():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    author = data.get("author")
    nTries = int(data.get("nTries", DEFAULT_TRIES))
    if not text:
        return jsonify({"error": "text is required"}), 400
    if author not in PROFILES:
        return jsonify({"error": "Unknown author %r" % author}), 400
    #a writer the namer has no class for can still be WRITTEN, it just
    #cannot be scored: the joint search ranks candidates partly on how much
    #they look like the requested writer, and there is no such score for a
    #hand the namer has never seen.  an author added from the training page
    #is exactly that case, so it falls back to one unscored draw rather
    #than being refused outright
    judged = author if author in SHAPE_MAP else None

    profile = PROFILES[author]
    #which ruled line to write on.  without it every job lands at the same
    #height and the second one is drawn on top of the first
    cfg = REP.GantryConfig()
    lineH = float(data.get("lineHeightMm") or cfg.lineHeightMm)
    lineIdx = max(0, int(data.get("line", 0)))
    #the two feeds are separate because they buy different things.  the
    #pen-down feed is a quality control: a tight curve drawn too fast
    #overshoots, which is what rounds off the small loops of an 'e'.  the
    #pen-up feed is pure time, since nothing is being drawn while the
    #carriage crosses the page, so it can be raised to pay for a slower
    #draw without costing any accuracy.  clamped to what the runner will
    #accept at the bottom and to a rate the belts hold at the top
    drawFeed = _Clamp(data.get("drawFeedMmMin"), cfg.drawFeedMmMin, 60.0, 3000.0)
    travelFeed = _Clamp(data.get("travelFeedMmMin"), cfg.travelFeedMmMin, 60.0, 6000.0)
    #the two pen dwells are tunable from the page because the only way to find
    #the real throw time is to cut them until the pen starts missing, and that
    #has to be done against the machine rather than guessed at.  zero is a
    #legal setting: it emits no G4 at all
    penUpDwell = _Clamp(data.get("penUpDwellMs"), cfg.penUpDwellMs, 0.0, 1000.0)
    penDownDwell = _Clamp(data.get("penDownDwellMs"), cfg.penDownDwellMs, 0.0, 1000.0)
    #a page that will be photographed and read back wants the laid-out
    #pitch, not the ruling: on the ruling the ascenders and descenders of
    #neighbouring lines meet, and ink that touches is cut as one line
    ignore = data.get("ignoreRuling")
    ignore = cfg.ignoreRuling if ignore is None else bool(ignore)
    #best-prior glyphs: take the best-formed shape the writer produced for
    #each letter instead of drawing from their top few variants.  the hand
    #loses its letter-to-letter variation and reads more cleanly; measured
    #on the weakest writer it gained 1.1 points of character accuracy on its
    #own, and 3.4 with the weak-character repair pass that follows it, with
    #writer identification unchanged at 100 per cent
    bestPrio = bool(data.get("bestPrio", False))
    cfg = dataclasses.replace(cfg, lineHeightMm=lineH, lineOffsetMm=lineIdx*lineH,
                              ignoreRuling=ignore,
                              drawFeedMmMin=drawFeed, travelFeedMmMin=travelFeed,
                              penUpDwellMs=penUpDwell, penDownDwellMs=penDownDwell)
    run = TIMING.Run("reproduce", chars=len(text), lines=1)

    with run.Step("synthesis"):
        #size the letters to the ruling BEFORE laying out, so the line breaks
        #are chosen for the size that will actually be written
        xh = REP.XhForRuling(profile, cfg, text)
        if judged is None:
            traj = SYN.SynthesizeText(text, profile, mmPerXh=xh,
                                      lineWidthMm=cfg.usableWidthMm,
                                      bestPrio=bestPrio)
        else:
            traj = SYN.SynthesizeJointBestOf(
                judged, text, profile, nTries=nTries,
                mmPerXh=xh, lineWidthMm=cfg.usableWidthMm, bestPrio=bestPrio,
                reader=TEXT_MODEL, authorModel=SHAPE_MODEL, authorMapping=SHAPE_MAP)

    jobId = uuid.uuid4().hex
    jobDir = JOBS/jobId
    jobDir.mkdir(parents=True, exist_ok=True)

    with run.Step("rendering"):
        #RenderTrajectory takes a trajectory and a scale and nothing else in
        #this tree.  it was being called with profile= and uniformInk=, which
        #this build's signature has never had, so every generate raised
        #TypeError before it reached the G-code stage
        preview = SYN.RenderTrajectory(traj, pxPerMm=PREVIEW_PX_PER_MM)
        Image.fromarray(np.asarray(preview, dtype=np.uint8)).save(jobDir/"preview.png")

    with run.Step("gcode"):
        written = REP.WriteGcode(traj, cfg, str(jobDir/"job.gcode"),
                                 title="web job %s" % author)

    #the read-back is taken off the G-CODE, on a whole sheet of paper, not
    #off the trajectory render: see RenderGcodePage
    with run.Step("placing on the page"):
        #the pen width is a fraction of the x-height, so the size the text
        #was actually laid out at has to be handed over: at the default 4 mm
        #a job sized down to fit a tighter ruling would be drawn with too
        #thick a pen and read back as a blur
        pageImg = RenderGcodePage(jobDir/"job.gcode", cfg,
                                  mmPerXh=traj.meta.get("mmPerXh", 4.0))
        Image.fromarray(np.asarray(pageImg, dtype=np.uint8)).save(jobDir/"page.png")

    #the generation stages are stored with the job so that drawing it later can
    #report the whole reproduction, generation included, against one allowance
    (jobDir/"meta.json").write_text(json.dumps({"text": text, "author": author,
                                                "nTries": nTries, "line": lineIdx,
                                                "bestPrio": bestPrio,
                                                "lineHeightMm": lineH,
                                                "ignoreRuling": ignore,
                                                "drawFeedMmMin": drawFeed,
                                                "travelFeedMmMin": travelFeed,
                                                "penUpDwellMs": penUpDwell,
                                                "penDownDwellMs": penDownDwell,
                                                "generateStages": run.stages}),
                                    encoding="utf-8")
    return JobResponse(jobId, written, pageImg, run, expectedText=text,
                       pxPerMm=PAGE_PX_PER_MM, nameWriter=True, askedFor=judged,
                       author=author,
                       n_tries=(nTries if judged is not None else 1),
                       judged_by_namer=judged is not None,
                       placement=written.get("placement"),
                       draw_feed_mm_min=drawFeed, travel_feed_mm_min=travelFeed,
                       pen_up_dwell_ms=penUpDwell, pen_down_dwell_ms=penDownDwell,
                       ignore_ruling=ignore,
                       page_url="/api/images/page_%s" % jobId,
                       #measured off the job that was just written, so a tweak
                       #can be judged from the page without running the gantry
                       budget=dict(drawMm=written.get("drawMm"),
                                   travelMm=written.get("travelMm"),
                                   penPulses=written.get("penPulses"),
                                   drawSeconds=written.get("drawSeconds"),
                                   travelSeconds=written.get("travelSeconds"),
                                   dwellSeconds=written.get("dwellSeconds"),
                                   estimatedSeconds=written.get("estimatedSeconds")),
                       )


#----------------------------------------------------------------------
#POST /api/retrace   multipart: image (file) or image_id
#the page's own ink traced straight into G-code, with no recognition and
#no synthesis in between.  drawing this and comparing it against the
#original page measures the gantry by itself, since nothing along the way
#made a decision that could have changed the shapes
#----------------------------------------------------------------------
@app.route("/api/retrace", methods=["POST"])
def ApiRetrace():
    if "image" in request.files and request.files["image"].filename:
        imageId = uuid.uuid4().hex
        srcPath = UPLOADS/("%s.jpg" % imageId)
        OpenUpright(request.files["image"].stream).convert("RGB").save(srcPath)
    elif request.form.get("image_id"):
        imageId = "".join(c for c in request.form["image_id"] if c.isalnum())
        srcPath = UPLOADS/("%s.jpg" % imageId)
        if not srcPath.exists():
            return jsonify({"error": "Unknown image_id"}), 400
    else:
        return jsonify({"error": "Provide either 'image' or 'image_id'"}), 400

    jobId = uuid.uuid4().hex
    jobDir = JOBS/jobId
    jobDir.mkdir(parents=True, exist_ok=True)
    try:
        result = RETRACE.RetracePage(srcPath, jobDir/"job.gcode",
                                     cfg=REP.GantryConfig(),
                                     previewPath=jobDir/"preview.png")
    except SystemExit as e:
        return jsonify({"error": str(e)}), 422

    (jobDir/"meta.json").write_text(json.dumps(
        {"text": "", "author": "retrace", "source": str(srcPath),
         "generateStages": result["timing"]["stages"]}), encoding="utf-8")
    Remember2(result["timing"])
    return jsonify({
        "job_id": jobId,
        "gcode_url": "/api/gcode/%s" % jobId,
        "preview_url": "/api/images/job_%s" % jobId,
        "source_url": "/api/images/%s" % imageId,
        "lines_traced": result["timing"]["lines"],
        "strokes_traced": result["tracedStrokes"],
        "gcode_line_count": result["lines"],
        "pen_pulses": result["penPulses"],
        "scale": round(result["scale"], 4),
        "clamped_points": result["clipped"],
        "timing": result["timing"],
    })


#======================================================================
#GANTRY GLUE
#the reproduce.py in this tree drives the machine a line and a file at a
#time.  the three helpers below sit on top of that and give the routes the
#shapes they ask for, and they live here rather than in reproduce.py so
#that file stays exactly as it was handed in
#======================================================================
#---IsConnected: whether the pins are currently claimed---
def IsConnected():
    return getattr(REP,"req",None) is not None


#---WriteGcodeFile: claim the pins, draw the file, say how the job ended---
#the three flags are read back off the module after the run, because
#RunGcodeFile sets them as it goes and returns nothing itself
def WriteGcodeFile(path):
    REP.Connect()
    REP.RunGcodeFile(path)
    return {"estopped": REP.estopped,
            "tripped_switch": REP.TrippedSwitch,
            "aborted": REP.lastJobAborted}


#----------------------------------------------------------------------
#ParseGcode: read a g-code file back into strokes
#written as its own reader rather than sharing code with the writer, so
#that comparing the two actually tests the file instead of testing one
#shared assumption twice over
#----------------------------------------------------------------------
def ParseGcode(path):
    strokes = []
    cur = []
    penDown = False
    x = y = 0.0
    with open(path,encoding="utf-8") as f:
        for raw in f:
            line = raw.split(";")[0].strip()
            if not line:
                continue
            up = line.upper()
            code = up.split()[0]
            if code=="M5":
                if penDown and len(cur)>=2:
                    strokes.append(cur)
                cur,penDown = [],False
                continue
            if code=="M3":
                penDown = True
                cur = [(x,y)]
                continue
            if code in ("G0","G1"):
                nx = re.search(r"X(-?\d+\.?\d*)",up)
                ny = re.search(r"Y(-?\d+\.?\d*)",up)
                if nx:
                    x = float(nx.group(1))
                if ny:
                    y = float(ny.group(1))
                if (nx or ny) and penDown:
                    cur.append((x,y))
    if penDown and len(cur)>=2:
        strokes.append(cur)
    return strokes


#----------------------------------------------------------------------
#RenderGcodePage: the written file, drawn on a whole sheet of paper
#
#this replaces reading the TRAJECTORY render back.  the two differ in ways
#that matter for a reported accuracy figure.
#
#the trajectory render is cropped to the ink and carries no paper, so the
#read-back never saw the margins, the line pitch in millimetres, or the
#place on the sheet the writing was put; and it never saw the G-code at
#all, so a fault in WriteGcode -- a clamped point, a dropped pen pulse, a
#collapsed short move -- could not show up in it.  what was being measured
#was the synthesiser, with the file it wrote taken on trust.
#
#this parses the file back with ParseGcode, which shares no code with the
#writer, and draws those strokes at their own machine coordinates on a
#sheet the size of the work area.  so the picture that is read is the
#picture the machine will draw, and the number that comes out is a figure
#for the whole chain down to the file.
#----------------------------------------------------------------------
def RenderGcodePage(gcodePath, cfg, pxPerMm=None, mmPerXh=4.0):
    pxPerMm = float(pxPerMm or PAGE_PX_PER_MM)
    strokes = ParseGcode(str(gcodePath))
    W = max(8, int(round(float(cfg.boundsMaxXmm)*pxPerMm)))
    H = max(8, int(round(float(cfg.boundsMaxYmm)*pxPerMm)))
    arr = np.full((H, W), 255, np.uint8)
    #one pen width everywhere, the same width the synthesiser draws with, so
    #the namer is not being handed an ink cue that the real pen would not give
    sp = max(1, int(round(SYN.UNIFORM_PEN_WIDTH_XH*mmPerXh*pxPerMm)))
    for s in strokes:
        #machine millimetres with y measured UP, image pixels with y down
        pts = [(x*pxPerMm, (float(cfg.boundsMaxYmm)-y)*pxPerMm) for (x, y) in s]
        SYN.DrawPolyline(arr, pts, 0, sp)
    soft = SYN.UNIFORM_SOFT_FRAC*max(1.0, sp)
    if soft > 0.05:
        arr = SYN.Blur(arr, soft)
    return np.asarray(arr, dtype=np.uint8)


@app.route("/api/gcode/upload", methods=["POST"])
def ApiGcodeUpload():
    if "gcode" not in request.files or not request.files["gcode"].filename:
        return jsonify({"error": "Provide a 'gcode' file"}), 400
    expectedText = (request.form.get("expected_text") or "").strip()

    jobId = uuid.uuid4().hex
    jobDir = JOBS/jobId
    jobDir.mkdir(parents=True, exist_ok=True)
    gcodePath = jobDir/"job.gcode"
    request.files["gcode"].save(str(gcodePath))

    run = TIMING.Run("reproduce", chars=len(expectedText), lines=1)
    with run.Step("placing on the page"):
        #the same path a generated job takes, so an uploaded file and a
        #generated one are read back the same way and can be compared
        pageImg = RenderGcodePage(gcodePath, REP.GantryConfig())
        Image.fromarray(np.asarray(pageImg, dtype=np.uint8)).save(jobDir/"page.png")
        Image.fromarray(np.asarray(pageImg, dtype=np.uint8)).save(jobDir/"preview.png")
    written = dict(lines=sum(1 for _ in open(gcodePath, encoding="utf-8", errors="replace")),
                   penPulses=None)
    return JobResponse(jobId, written, pageImg, run, pxPerMm=PAGE_PX_PER_MM,
                       nameWriter=True, askedFor=None,
                       page_url="/api/images/page_%s" % jobId,
                       expectedText=expectedText or None)


@app.route("/api/gcode/<jobId>")
def ApiGetGcode(jobId):
    safe = "".join(c for c in jobId if c.isalnum())
    p = JOBS/safe/"job.gcode"
    if not p.exists():
        abort(404)
    return send_file(str(p), mimetype="text/plain",
                     as_attachment=True, download_name="%s.gcode" % safe)


#======================================================================
#THE GANTRY
#drawing is timed end to end: the call returns when the machine has run
#the whole file and parked itself again, so the wall time of that call is
#the real drawing time
#======================================================================
@app.route("/api/gantry/status")
def ApiGantryStatus():
    if not IsConnected():
        try:
            REP.Connect()
        except REP.GantryNotConnectedError as e:
            return jsonify({"connected": False, "calibrated": False, "detail": str(e)})
    return jsonify({"connected": True,
                    "calibrated": REP.Calibration["done"],
                    "calibration": REP.Calibration if REP.Calibration["done"] else None})


@app.route("/api/gantry/calibrate", methods=["POST"])
def ApiGantryCalibrate():
    run = TIMING.Run("calibrate")
    try:
        REP.Connect()
        with run.Step("homing"):
            result = REP.Calibrate()
    except REP.GantryNotConnectedError as e:
        return jsonify({"calibrated": False, "message": "Gantry not connected: %s" % e})
    except RuntimeError as e:
        return jsonify({"calibrated": False, "message": "Calibration failed: %s" % e})
    return jsonify({"calibrated": True, "calibration": result, "timing": Remember(run)})


#----------------------------------------------------------------------
#the stop has to be reachable while a job is drawing, which it is because
#the server runs threaded: the drawing request is still blocked inside
#RunGcodeFile on its own thread while this one is served.  on a
#single-threaded server this request would queue behind the very job it is
#meant to interrupt and arrive only after it had finished
#----------------------------------------------------------------------
@app.route("/api/gantry/stop", methods=["POST"])
def ApiGantryStop():
    try:
        result = REP.EmergencyStop()
    except REP.GantryNotConnectedError as e:
        return jsonify({"stopped": False, "gantry_connected": False,
                        "message": "Gantry not connected: %s" % e})
    except RuntimeError as e:
        return jsonify({"stopped": False, "gantry_connected": True,
                        "message": "Stop failed: %s" % e})
    bits = ["pen up" if result["pen_up"] else "PEN DID NOT LIFT"]
    bits.append("parked at the corner" if result["parked"] else "not parked")
    msg = "Stopped -- " + ", ".join(bits) + "."
    if result.get("detail"):
        msg += " " + result["detail"][0].upper() + result["detail"][1:] + "."
    result.update({"gantry_connected": True, "message": msg})
    return jsonify(result)


#======================================================================
#DRIVING THE PEN BY HAND
#one point and one pen state, which is what setting a sheet up needs: the
#G-code path can only be exercised as a whole file, and finding where the
#paper actually is means putting the pen somewhere and looking at it.
#
#every target is held inside the homed area by the same clamp the G-code
#runner uses, so a typed coordinate cannot drive the carriage into a
#switch, and both calls refuse outright until the machine has been homed,
#because before that the steps per millimetre are a nominal figure and a
#millimetre is not yet a known distance.
#======================================================================
#---PenPositionMm: where the carriage is, in the same frame the G-code uses---
def PenPositionMm():
    spmmX = REP.Calibration["steps_per_mm_x"]
    spmmY = REP.Calibration["steps_per_mm_y"]
    return (REP.CurrentXSteps/spmmX-REP.EDGE_TOLERANCE_MM,
            REP.CurrentYSteps/spmmY-REP.EDGE_TOLERANCE_MM)


#---JogBounds: the rectangle a typed coordinate is allowed to land in---
def JogBounds():
    return {"x_min_mm": 0.0, "y_min_mm": 0.0,
            "x_max_mm": round(float(REP.Calibration["usable_width_mm"]), 1),
            "y_max_mm": round(float(REP.Calibration["usable_height_mm"]), 1)}


@app.route("/api/gantry/position")
def ApiGantryPosition():
    if not IsConnected():
        try:
            REP.Connect()
        except REP.GantryNotConnectedError as e:
            return jsonify({"connected": False, "calibrated": False,
                            "bounds": JogBounds(), "detail": str(e)})
    x, y = PenPositionMm()
    return jsonify({"connected": True,
                    "calibrated": bool(REP.Calibration["done"]),
                    "x_mm": round(x, 2), "y_mm": round(y, 2),
                    "pen_up": bool(REP.PenIsUp()),
                    "bounds": JogBounds()})


@app.route("/api/gantry/jog", methods=["POST"])
def ApiGantryJog():
    data = request.get_json(silent=True) or {}
    b = JogBounds()
    try:
        REP.Connect()
    except REP.GantryNotConnectedError as e:
        return jsonify({"moved": False, "gantry_connected": False,
                        "message": "Gantry not connected: %s" % e,
                        "bounds": b})
    if not REP.Calibration["done"]:
        return jsonify({"moved": False, "gantry_connected": True,
                        "calibrated": False, "bounds": b,
                        "message": "Calibrate first -- until the axes have been "
                                   "homed a millimetre is a nominal figure and "
                                   "the pen would not go where you asked."})
    #an absent coordinate means "leave this axis alone", so one box can be
    #used on its own without the other silently jumping to zero
    curX, curY = PenPositionMm()
    x = _Clamp(data.get("x", curX), curX, b["x_min_mm"], b["x_max_mm"])
    y = _Clamp(data.get("y", curY), curY, b["y_min_mm"], b["y_max_mm"])
    feed = _Clamp(data.get("feedMmMin"), REP.GantryConfig().travelFeedMmMin,
                  60.0, 6000.0)
    hit = REP.TriggeredEndstop()
    if hit is not None:
        return jsonify({"moved": False, "gantry_connected": True, "bounds": b,
                        "message": "End-stop %s is tripped. Clear it first." % hit})
    run = TIMING.Run("jog")
    try:
        with run.Step("moving", chars=0):
            REP._GcodeLinearMove(x, y, feed)
    except RuntimeError as e:
        return jsonify({"moved": False, "gantry_connected": True, "bounds": b,
                        "message": "Move failed: %s" % e})
    nx, ny = PenPositionMm()
    return jsonify({"moved": True, "gantry_connected": True, "bounds": b,
                    "x_mm": round(nx, 2), "y_mm": round(ny, 2),
                    "requested_x_mm": round(x, 2), "requested_y_mm": round(y, 2),
                    "clamped": bool(abs(x-float(data.get("x", curX) or curX)) > 0.01
                                    or abs(y-float(data.get("y", curY) or curY)) > 0.01),
                    "pen_up": bool(REP.PenIsUp()),
                    "message": "At X%.1f Y%.1f mm." % (nx, ny),
                    "timing": Remember(run)})


@app.route("/api/gantry/pen", methods=["POST"])
def ApiGantryPen():
    data = request.get_json(silent=True) or {}
    #"down" is the word the page uses, because that is the thing being asked
    #for; SetPen is written the other way round and takes targetUp
    wantUp = not bool(data.get("down", False))
    try:
        REP.Connect()
    except REP.GantryNotConnectedError as e:
        return jsonify({"ok": False, "gantry_connected": False,
                        "message": "Gantry not connected: %s" % e})
    reached = REP.SetPen(wantUp)
    return jsonify({"ok": bool(reached), "gantry_connected": True,
                    "pen_up": bool(REP.PenIsUp()),
                    "message": ("Pen %s." % ("up" if wantUp else "down")) if reached
                               else ("Pen did NOT confirm %s on its switch -- check "
                                     "the motor and the wiring."
                                     % ("up" if wantUp else "down"))})


@app.route("/api/write/<jobId>", methods=["POST"])
def ApiWriteGcode(jobId):
    safe = "".join(c for c in jobId if c.isalnum())
    jobDir = JOBS/safe
    gcodePath = jobDir/"job.gcode"
    if not gcodePath.exists():
        return jsonify({"error": "Unknown job_id"}), 404

    meta = {}
    if (jobDir/"meta.json").exists():
        meta = json.loads((jobDir/"meta.json").read_text(encoding="utf-8"))
    chars = len(meta.get("text", ""))

    #replay what generating this job cost, so the figure reported here is the
    #whole reproduction: synthesis, rendering, gcode and then the pen
    priorStages = meta.get("generateStages") or []
    run = TIMING.Run("reproduce", chars=chars, lines=1)
    for s in priorStages:
        run.Carry(s.get("stage", "earlier"), s.get("seconds", 0.0), s.get("characters", 0) or 0)
    try:
        with run.Step("drawing", chars=chars):
            result = WriteGcodeFile(str(gcodePath))
    except REP.GantryNotConnectedError as e:
        return jsonify({"sent": False, "gantry_connected": False,
                        "message": "Gantry not connected: %s" % e})
    except REP.NotCalibratedError as e:
        return jsonify({"sent": False, "gantry_connected": True, "calibrated": False,
                        "message": "Calibration required: %s" % e})

    #a job the stop button interrupted did not finish, and reporting it as
    #sent would put an accuracy figure against a half-drawn page
    if result.get("aborted"):
        return jsonify({"sent": False, "gantry_connected": True, "aborted": True,
                        "message": "Job stopped before it finished. The page is "
                                   "incomplete -- do not measure it.",
                        "timing": Remember(run)})
    if result.get("estopped"):
        return jsonify({"sent": False, "gantry_connected": True,
                        "message": "Job aborted by end-stop %r mid-run. Clear it on the "
                                   "gantry before sending another job."
                                   % result.get("tripped_switch"),
                        "timing": Remember(run)})
    return jsonify({"sent": True, "gantry_connected": True, "message": "Job completed.",
                    "timing": Remember(run)})


#======================================================================
#INFORMATION
#======================================================================
#the figures below are this project's own measurements, weighted by the
#number of crops, from testing/results/RESULTS.md: 669 real line crops from
#the five photographed writers, and 13 writers for the synthesis row
@app.route("/api/model_info")
def ApiModelInfo():
    def Params(model):
        return int(sum(arr.size for arr in model.sd.values()))

    def Mb(path):
        return round(path.stat().st_size/(1024*1024), 2) if path and path.exists() else None

    return jsonify({
        "text_recognizer": {
            "architecture": "CNN-BiLSTM-CTC, 12 convolutions then 2 bidirectional LSTM layers",
            "weights_file": TEXT_MODEL.checkpointPath.name,
            "parameters": Params(TEXT_MODEL),
            "file_size_mb": Mb(TEXT_MODEL.checkpointPath),
            "measured_accuracy": {
                "real_ink_characters_pct": 98.61,
                "real_ink_words_pct": 94.17,
                "note": "669 real line crops from five photographed writers. The "
                        "same measurement on the from-scratch checkpoint gives 24.1%, "
                        "which is why the joint checkpoint is the one loaded.",
            },
        },
        "writer_id_ink_based": {
            "architecture": "CNN classifier on raw grey crops",
            "weights_file": INK_MODEL.checkpointPath.name,
            "parameters": Params(INK_MODEL),
            "file_size_mb": Mb(INK_MODEL.checkpointPath),
            "measured_accuracy": {
                "real_photographed_pages_pct": 99.70,
                "note": "Use only on real photographed pages. A drawn line has one "
                        "pen width everywhere, so there is no ink evidence to judge.",
            },
        },
        "writer_id_stroke_normalised": {
            "architecture": "CNN classifier on stroke-normalised crops, ink-density invariant",
            "weights_file": SHAPE_MODEL.checkpointPath.name,
            "parameters": Params(SHAPE_MODEL),
            "file_size_mb": Mb(SHAPE_MODEL.checkpointPath),
            "measured_accuracy": {
                "real_ink_pct": 98.20,
                "note": "This is the judge used for drawn and synthesised output.",
            },
        },
        "reproduction_pipeline": {
            "authors": sorted(PROFILES),
            "n_authors": len(PROFILES),
            "measured_on_novel_sentences": {
                "writer_id_direct_pct": 77.86,
                "writer_id_gcode_pct": 77.86,
                "text_char_direct_pct": 80.07,
                "text_char_gcode_pct": 80.07,
                "text_word_pct": None,
                "writers_identified": "12 of 13",
                "seconds_per_line": 10.9,
                "method": "SynthesizeJointBestOf: %d candidates per line, scored by the "
                          "harmonic mean of text accuracy and writer-ID confidence "
                          "together, with the C1 join" % DEFAULT_TRIES,
                "measured_via": "testing/results/RESULTS.md, 13 writers on novel sentences",
            },
        },
        "gantry": {
            "gcode_round_trip_iou": 0.9974,
            "gcode_max_error_mm": 0.0007,
        },
        "charset_size": len(REC.CHARSET)+1,
        "input_canvas": [REC.INPUT_HEIGHT, REC.INPUT_WIDTH],
        "timing_allowances": TIMING.ALLOWANCE_TEXT,
        "device": "cpu (numpy inference, no machine-learning library at runtime)",
    })


@app.route("/api/timings")
def ApiTimings():
    with TIMINGS_LOCK:
        rows = list(TIMINGS)
    summary = {}
    for r in rows:
        s = summary.setdefault(r["kind"], dict(runs=0, totalSeconds=0.0, characters=0))
        s["runs"] += 1
        s["totalSeconds"] += r["totalSeconds"]
        s["characters"] += r["characters"] or 0
    for kind, s in summary.items():
        s["meanSeconds"] = round(s["totalSeconds"]/s["runs"], 3)
        s["meanSecondsPerChar"] = (round(s["totalSeconds"]/s["characters"], 4)
                                   if s["characters"] else None)
        #the allowance scales with characters, so the mean is marked against
        #the allowance for the mean number of characters
        formula = TIMING.ALLOWANCE.get(kind)
        meanChars = s["characters"]/s["runs"] if s["runs"] else 0
        s["meanAllowanceSeconds"] = round(formula(meanChars), 2) if formula and meanChars else None
        s["allowanceFormula"] = TIMING.ALLOWANCE_TEXT.get(kind)
        s["meanWithinAllowance"] = (bool(s["meanSeconds"] <= s["meanAllowanceSeconds"])
                                    if s["meanAllowanceSeconds"] else None)
    return jsonify({"summary": summary, "runs": rows})


@app.route("/api/samples/<author>")
def ApiSamples(author):
    if author not in PROFILES:
        abort(404)
    try:
        pairs = RENDER.SampleCrops(author, n=2)
    except Exception as e:
        return jsonify({"error": str(e)}), 200
    #a bare list, which is what the page maps over
    return jsonify([{"image_url": "/api/samples/%s/%s" % (author, p.name), "text": t}
                    for p, t in pairs])


@app.route("/api/samples/<author>/<filename>")
def ApiSampleFile(author, filename):
    safe = "".join(c for c in filename if c.isalnum() or c in "._-")
    p = RENDER.CACHE_DIR/safe
    if not safe.startswith("sample_%s_" % author) or not p.exists():
        abort(404)
    return send_file(str(p))


@app.route("/api/glyphs/<author>")
def ApiGlyphs(author):
    if author not in PROFILES:
        abort(404)
    try:
        return send_file(str(RENDER.GlyphGridPath(author, PROFILES[author])))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


#======================================================================
#ONE BACKGROUND JOB AT A TIME
#training and building a writer's glyph library are both long and both
#hungry: extraction holds every cut letter of every line in memory, and a
#training run holds the batches.  the board has run out of memory and had
#jobs killed under it before, so this is deliberately ONE slot.  a second
#request is refused with what is already running rather than queued, so
#nothing can start a run the asker did not watch begin.
#
#a training run is a SUBPROCESS, so that it cannot take the server down
#with it and so that killing it is a kill rather than a flag another
#thread has to agree to notice.  building a profile is a thread, because
#it has to hand its result back through the loaded reader.
#======================================================================
BG = {"kind": None, "label": "", "started": 0.0, "proc": None,
      "thread": None, "logPath": None, "done": True, "ok": None,
      "message": "", "result": None}
BG_LOCK = threading.Lock()
TRAIN_LOGS = paths.Cache/"WebTraining"
TRAIN_LOGS.mkdir(parents=True, exist_ok=True)


#---BgRunning: whether the slot is occupied, reaping a finished subprocess---
def BgRunning():
    p = BG.get("proc")
    if p is not None and p.poll() is not None:
        BG["done"] = True
        BG["ok"] = (p.returncode == 0)
        BG["message"] = ("Finished." if p.returncode == 0
                         else "Exited with code %d." % p.returncode)
        BG["proc"] = None
    t = BG.get("thread")
    if t is not None and not t.is_alive():
        BG["thread"] = None
    return not BG["done"]


#---BgLogTail: the last few hundred lines of whatever is running---
def BgLogTail(limit=400):
    p = BG.get("logPath")
    if not p or not Path(p).exists():
        return ""
    try:
        lines = Path(p).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-int(limit):])


#---BgStatus: the shape both the training page and the add-author form read---
def BgStatus():
    running = BgRunning()
    return {"running": running,
            "kind": BG["kind"], "label": BG["label"],
            "seconds": round(time.time()-BG["started"], 1) if BG["started"] else None,
            "ok": BG["ok"], "message": BG["message"],
            "result": BG["result"],
            "log": BgLogTail()}


@app.route("/api/train/status")
def ApiTrainStatus():
    with BG_LOCK:
        return jsonify(BgStatus())


@app.route("/api/train/stop", methods=["POST"])
def ApiTrainStop():
    with BG_LOCK:
        if not BgRunning():
            return jsonify(dict(BgStatus(), message="Nothing is running."))
        p = BG.get("proc")
        if p is None:
            #a profile build is a thread inside this process and there is no
            #safe way to tear one down part way through, so it is left to
            #finish rather than half-written over a profile on disk
            return jsonify(dict(BgStatus(),
                                message="A profile build cannot be interrupted; "
                                        "it will finish on its own."))
        p.terminate()
        try:
            p.wait(timeout=10)
        except Exception:
            p.kill()
        BG["done"] = True
        BG["ok"] = False
        BG["message"] = "Stopped."
        BG["proc"] = None
        return jsonify(BgStatus())


#----------------------------------------------------------------------
#POST /api/train/start   json: task, epochs, batch, lr, seed, source,
#                              personalFraction, writers, resume, warmStart
#runs train.py, which is the same command line the measured checkpoints
#were trained with, so a run started here and a run started in a terminal
#are the same run.  every number is bounded, because these go onto a
#command line and an hour of an ODROID's time is easy to ask for by
#mistyping a digit
#----------------------------------------------------------------------
@app.route("/api/train/start", methods=["POST"])
def ApiTrainStart():
    import subprocess

    data = request.get_json(silent=True) or {}
    task = data.get("task")
    if task not in ("text", "author"):
        return jsonify({"started": False,
                        "error": "task must be 'text' or 'author'"}), 400

    epochs = int(_Clamp(data.get("epochs"), 40, 1, 400))
    batch = int(_Clamp(data.get("batch"), 8, 1, 64))
    lr = _Clamp(data.get("lr"), 1e-3, 1e-6, 1.0)
    seed = int(_Clamp(data.get("seed"), 7, 0, 999999))
    argv = [sys.executable, "-u", str(CP3/"train.py"), task,
            "--epochs", str(epochs), "--batch", str(batch),
            "--lr", repr(lr), "--seed", str(seed)]

    #the tag goes into the checkpoint's filename, so a run started here lands
    #BESIDE the shipped weights rather than over them.  a blank tag is still
    #allowed -- that is how the shipped checkpoint would be retrained on
    #purpose -- but the page defaults it to something, because the first epoch
    #to beat its starting score writes the file and a demonstration run would
    #otherwise leave the website loading a one-epoch model
    tag = "".join(c for c in str(data.get("tag") or "").strip()
                  if c.isalnum() or c in "-_")[:40]
    if tag:
        argv += ["--tag", tag]

    if task == "text":
        source = data.get("source") or "joint"
        if source not in ("joint", "teklia", "pages"):
            return jsonify({"started": False,
                            "error": "source must be joint, teklia or pages"}), 400
        argv += ["--source", source]
        if source == "joint":
            argv += ["--personal-fraction",
                     repr(_Clamp(data.get("personalFraction"), 0.20, 0.0, 1.0))]
        if data.get("pages"):
            argv += ["--pages", str(int(_Clamp(data.get("pages"), 1, 1, 100000)))]
    else:
        #only the writers this build has data for, checked here rather than
        #letting train.py refuse after the dataset has been read in
        known = set(paths.DatabaseWriters)|set(paths.PersonalWriters)
        writers = [w for w in (data.get("writers") or []) if w in known]
        if data.get("writers") and not writers:
            return jsonify({"started": False,
                            "error": "none of those writers have data here; "
                                     "known: %s" % ", ".join(sorted(known))}), 400
        if writers:
            argv += ["--writers"]+writers

    if data.get("resume"):
        argv += ["--resume"]
    warm = (data.get("warmStart") or "").strip()
    if warm:
        #a name out of the weights folder, never a path from the request
        safe = "".join(c for c in warm if c.isalnum() or c in "._-")
        if not (paths.Weights/safe).exists():
            return jsonify({"started": False,
                            "error": "no checkpoint named %r in the weights "
                                     "folder" % safe}), 400
        argv += ["--warm-start", str(paths.Weights/safe)]

    with BG_LOCK:
        if BgRunning():
            return jsonify({"started": False,
                            "error": "%s is already running (%s). Stop it first."
                                     % (BG["kind"], BG["label"])}), 409
        logPath = TRAIN_LOGS/("train_%s_%s.log" % (task, uuid.uuid4().hex[:8]))
        fh = open(logPath, "w", encoding="utf-8", buffering=1)
        fh.write("$ %s\n\n" % " ".join(argv))
        try:
            proc = subprocess.Popen(argv, cwd=str(CP3), stdout=fh,
                                    stderr=subprocess.STDOUT)
        except OSError as e:
            fh.close()
            return jsonify({"started": False,
                            "error": "could not start train.py: %s" % e}), 500
        BG.update(kind="training", label="%s model" % task, started=time.time(),
                  proc=proc, thread=None, logPath=logPath, done=False, ok=None,
                  message="Running.", result=None)
    return jsonify({"started": True, "command": " ".join(argv),
                    "log_file": logPath.name, "status": BgStatus()})


@app.route("/api/train/log")
def ApiTrainLog():
    return Response(BgLogTail(limit=4000), mimetype="text/plain")


#======================================================================
#ADDING A WRITER FROM PHOTOGRAPHS
#the eleven shipped writers were built from pages plus a transcript per
#page, and this is the same build: segment each page into lines, pair each
#line with its transcript line, cut the letters out by aligning the
#reader's own output against that transcript, and keep the most typical
#variants.  nothing is trained here -- a glyph library is measured, not
#fitted -- so this takes a minute or two rather than an afternoon.
#
#the cross-writer letter prior is the one part that cannot be rebuilt from
#one writer: it exists to reject a writer's OWN bad cuts by comparing them
#against what the letter looks like across every hand, and pooling only
#the new writer's cuts would compare them against themselves.  so it is
#pooled from the eleven stored libraries plus the new writer's raw cuts,
#which is the same pooling the shipped profiles were built with, minus the
#cuts that were already filtered out of those libraries.
#
#the new writer will NOT be one of the namer's classes, because those are
#fixed by the checkpoint.  the page says so: their hand can be written,
#and the joint search cannot score it.
#======================================================================
@app.route("/api/authors/add", methods=["POST"])
def ApiAddAuthor():
    raw = (request.form.get("name") or "").strip().lower()
    name = "".join(c for c in raw if c.isalnum() or c in "_-")
    if not name or name != raw:
        return jsonify({"error": "Give the writer a name made of letters, digits, "
                                 "'-' or '_' only."}), 400
    if name in PROFILES:
        return jsonify({"error": "There is already a profile for %r. Pick another "
                                 "name, or delete that file first." % name}), 400

    images = [f for f in request.files.getlist("images") if f.filename]
    labels = {Path(f.filename).stem.replace("_labels", ""): f
              for f in request.files.getlist("labels") if f.filename}
    if not images:
        return jsonify({"error": "Attach at least one photograph of a written page."}), 400

    #a single page may instead carry its transcript typed into the box, one
    #written line per text line, which is how one page is quickest to add
    typed = [l.strip() for l in (request.form.get("transcript") or "").splitlines()
             if l.strip()]
    if not labels and not typed:
        return jsonify({"error": "Attach a transcript: either one '<page>_labels.txt' "
                                 "per image, or type the lines into the box."}), 400

    #the uploaded pages are kept, in the same shape the shipped writers' pages
    #are kept in.  a glyph library can be rebuilt from them, and they are what
    #a later training run would need: the namer's classes are fixed by its
    #checkpoint, so putting this writer INTO the namer means retraining it,
    #and retraining it means still having their pages.  they are kept here
    #rather than in PersonalDataset so that the measured dataset stays exactly
    #what it was when the reported figures were taken
    keepDir = paths.Cache/"AddedWriters"/name
    (keepDir/"Pages").mkdir(parents=True, exist_ok=True)
    (keepDir/"Labels").mkdir(parents=True, exist_ok=True)

    pages = []
    for f in images:
        key = Path(f.filename).stem
        stem = "".join(c for c in key if c.isalnum() or c in "._-") or "page"
        try:
            gray = np.asarray(OpenUpright(f.stream).convert("L"), dtype=np.uint8)
        except Exception as e:
            return jsonify({"error": "could not read %s: %s" % (f.filename, e)}), 400
        gray, _deg, _sc = StandUp(gray)
        if key in labels:
            text = labels[key].read().decode("utf-8", "replace")
        elif len(images) == 1 and typed:
            text = "\n".join(typed)
        else:
            return jsonify({"error": "no transcript for %s -- attach "
                                     "%s_labels.txt" % (f.filename, stem)}), 400
        #the same leading "7<tab>" index the shipped label files carry
        lines = [re.sub(r"^\d+\t", "", l.strip())
                 for l in text.splitlines() if l.strip()]
        #the STOOD UP page is what is kept, because that is the page the
        #letters were actually cut out of
        Image.fromarray(gray).convert("L").save(keepDir/"Pages"/("%s.png" % stem))
        (keepDir/"Labels"/("%s_labels.txt" % stem)).write_text(
            "\n".join(lines), encoding="utf-8")
        pages.append((gray, lines))

    outPath = paths.Profiles/("%s.json" % name)
    logPath = TRAIN_LOGS/("addauthor_%s_%s.log" % (name, uuid.uuid4().hex[:8]))

    def Work():
        import contextlib
        import style as STY
        with open(logPath, "w", encoding="utf-8", buffering=1) as fh:
            with contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
                try:
                    res = STY.BuildProfileFromPages(
                        name, pages, outPath, TEXT_MODEL,
                        priorLibs=PROFILES, inNamer=name in SHAPE_MAP)
                except Exception as e:
                    print("[add] FAILED: %s" % e, flush=True)
                    with BG_LOCK:
                        BG.update(done=True, ok=False,
                                  message="Could not build a profile: %s" % e)
                    return
        #picked up straight away, so the writer can be written with without
        #restarting the server
        PROFILES.clear()
        PROFILES.update(LoadProfiles())
        res["authorsNow"] = sorted(PROFILES)
        res["pages_kept_in"] = str(keepDir)
        with BG_LOCK:
            BG.update(done=True, ok=True, result=res,
                      message="Built a profile for %s: %d characters, %d variants."
                              % (name, res["characters"], res["variants"]))

    with BG_LOCK:
        if BgRunning():
            return jsonify({"error": "%s is already running (%s). Wait for it or "
                                     "stop it first." % (BG["kind"], BG["label"])}), 409
        t = threading.Thread(target=Work, name="addauthor", daemon=True)
        BG.update(kind="profile", label=name, started=time.time(), proc=None,
                  thread=t, logPath=logPath, done=False, ok=None,
                  message="Segmenting the pages and cutting the letters out.",
                  result=None)
        t.start()
    return jsonify({"started": True, "author": name, "pages": len(pages),
                    "status": BgStatus()})


if __name__ == "__main__":
    print("[server] listening on 0.0.0.0:%d, serving %s" % (PORT, FRONTEND))
    app.run(host="0.0.0.0", port=PORT, threaded=True)
