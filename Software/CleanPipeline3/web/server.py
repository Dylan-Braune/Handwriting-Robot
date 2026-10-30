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
    GET  /api/curate/<writer>         every letterform variant, as strokes
    POST /api/curate/<writer>         remove the marked ones, rebuild, save
    DEL  /api/curate/<writer>         throw the curated profile away
"""
import csv
import dataclasses
import io
import json
import os
import sys
import threading
import uuid
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
import curate as CUR
import recognise as REC
import reproduce as REP
import segment as SEG
import synthesise as SYN
import retrace as RETRACE
import timing as TIMING

import camera as CAM
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
print("[server] loading the reader from %s" % REC.DEFAULT_TEXT_WEIGHTS.name)
TEXT_MODEL = REC.PaperCRNNNumpy(checkpointPath=REC.DEFAULT_TEXT_WEIGHTS)

print("[server] loading the ink namer from %s" % REC.DEFAULT_AUTHOR_WEIGHTS.name)
INK_MODEL = REC.AuthorClassifierCNNNumpy(checkpointPath=REC.DEFAULT_AUTHOR_WEIGHTS)
INK_MAP = INK_MODEL.authorMapping
INK_BY_INDEX = {v: k for k, v in INK_MAP.items()}

print("[server] loading the shape namer from %s" % REC.SHAPE_AUTHOR_WEIGHTS.name)
SHAPE_MODEL = REC.AuthorClassifierCNNNumpy(checkpointPath=REC.SHAPE_AUTHOR_WEIGHTS)
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
#======================================================================
PAGE_NAMER = SHAPE_MODEL
PAGE_NAMER_STROKE_NORM = False
PAGE_BY_INDEX = {v: k for k, v in PAGE_NAMER.authorMapping.items()}
print("[server] naming photographed pages with %s, strokeNormalise=%s"
      % ("the shape namer" if PAGE_NAMER is SHAPE_MODEL else "the ink namer",
         PAGE_NAMER_STROKE_NORM))


#---BaseAuthor: the writer a profile belongs to, curated or not---
#curate.py writes <writer>_curated.json beside the original, and the server
#lists every profile it finds, so these names reach the endpoints.  the
#writer models were trained on the plain names and know nothing of the
#suffix, so anything asking them has to ask about the writer
def BaseAuthor(name):
    return name[:-8] if name.endswith("_curated") else name


#---LoadProfiles: every style profile on disk, by writer---
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
                        legibility=p.get("legibilityLambda"),
                        inGantryModels=a in SHAPE_MAP))
    return jsonify(out)


@app.route("/api/images/<imageId>")
def ApiGetImage(imageId):
    safe = "".join(c for c in imageId if c.isalnum() or c == "_")
    if safe.startswith("job_"):
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
    out = [cache[i] if i in cache else Read(crops[i], flip) for i in range(n)]
    if flip:
        #the order the lines come back in changes too, so the caller is
        #given it and can put everything else it holds per line in step
        idx = list(range(n))[::-1]
        return ([np.rot90(crops[i], 2) for i in idx], [out[i][0] for i in idx],
                [out[i][1] for i in idx], idx, 180, sF, sR)
    idx = list(range(n))
    return (list(crops), [t for t, _s in out], [s for _t, s in out], idx, 0, sF, sR)


def NameWriter(crops):
    """One writer for the page, from every line, weighted by ink.

    Two changes from naming off six lines spread down the page.

    The six were picked with linspace, which always includes the LAST
    index, so a junk crop at the foot of the page got a vote in every
    pool and at two lines was half of it.  On the page that prompted
    this, every one of the sixteen real lines named the writer correctly
    on its own, yet the pooled confidence came back at 78.4 per cent
    because the seventeenth crop was a blot.  Dropping the blots is the
    first half of the fix; letting every surviving line vote, so that no
    single crop can carry that much weight, is the second.

    The vote is then weighted by how much ink the line carries.  On that
    same page the per line confidence ran from 48.4 to 100 per cent, and
    the three weakest lines were the three shortest ones: two words of
    writing is less evidence than a full line and should not count the
    same as one.
    """
    per = []
    for c in crops:
        _i, p = REC.ClassifyImage(Image.fromarray(c).convert("L"), PAGE_NAMER,
                                  strokeNormalise=PAGE_NAMER_STROKE_NORM)
        per.append(np.asarray(p, dtype=np.float64))
    w = np.array([float((c < 128).sum()) for c in crops], dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones(len(crops), dtype=np.float64)
    avg = (np.vstack(per)*w[:, None]).sum(axis=0)/w.sum()
    return per, avg, w


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

    #SegmentLines records conditioning, segmenting and cropping into the run
    lines, _dbg, info = SEG.SegmentLines(gray, timer=run)
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
        perLineProbs, avg, inkWeights = NameWriter(crops)
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
            "word_accuracy_pct": (round(100*REC.WordAcc(readLines[i], t), 1)
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
        "word_accuracy_pct": (round(100*REC.WordAcc(predictedText, expectedText), 1)
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
def ReadBackText(previewImg, pxPerMm):
    """Read a rendered page back the way a photographed one is read.

    REC.ReadText is a LINE recogniser: it runs ResizeLineImageFixed, which
    squashes whatever it is given into a single strip a few dozen pixels
    tall.  Handing it a whole multi-line page crushes three lines of
    writing on top of each other, and it returns gibberish -- measured at
    24.1 per cent on a three line job that reads 83.0 per cent when the
    lines are taken one at a time.

    So the page goes through the same segmenter a photograph does.  Two
    details matter.  The render is cropped tight to the ink, and the
    segmenter needs paper around the writing to work: with no margin it
    clipped the first and last letters and scored 64.7 per cent against
    94.1 per cent once padded.  And a page that segments to a single line
    is read whole, because for one line the crop can only lose edges that
    the full image still has.
    """
    arr = np.asarray(previewImg, dtype=np.uint8)
    pad = max(8, int(8.0*float(pxPerMm or 8.0)))   #8 mm of paper, as the writers' own sheets have
    padded = np.full((arr.shape[0]+2*pad, arr.shape[1]+2*pad), 255, np.uint8)
    padded[pad:pad+arr.shape[0], pad:pad+arr.shape[1]] = arr
    try:
        #mergeValley is asked for HERE and nowhere else.  a rendered page is
        #drawn at one pen width with no paper behind it, and on that input a
        #single line's descender band is readily taken for a second line; the
        #merge fixes exactly that.  it is left off for photographs, where it
        #was measured to change nothing and cost 0.4 points
        lines, _dbg, _info = SEG.SegmentLines(padded, mergeValley=0.72)
    except Exception:
        #a read-back is a report, never the job: if segmenting throws, say
        #what the single-line path says rather than failing the request
        return REC.ReadText(previewImg, TEXT_MODEL)
    if len(lines) <= 1:
        return REC.ReadText(previewImg, TEXT_MODEL)
    return " ".join(filter(None, (
        REC.ReadText(Image.fromarray(np.asarray(ln["image"])).convert("L"), TEXT_MODEL)
        for ln in lines)))


def JobResponse(jobId, written, previewImg, run, expectedText=None,
                pxPerMm=8.0, **extra):
    with run.Step("reading back", chars=len(expectedText or "")):
        readBack = ReadBackText(previewImg, pxPerMm)
    body = {
        "job_id": jobId,
        "gcode_url": "/api/gcode/%s" % jobId,
        "preview_url": "/api/images/job_%s" % jobId,
        "gcode_line_count": written.get("lines"),
        "pen_pulses": written.get("penPulses"),
        "read_back_text": readBack,
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
    #a curated profile is the same hand with a tidier library, so the writer
    #model is asked about the writer, not about the file.  without this the
    #curated profiles appear in the author list and every one of them is
    #refused, because the classifier's classes are the plain names
    judged = BaseAuthor(author)
    if judged not in SHAPE_MAP:
        return jsonify({"error": "No writer model for %r, so its output cannot be "
                                 "judged during the search" % author}), 400

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
    cfg = dataclasses.replace(cfg, lineHeightMm=lineH, lineOffsetMm=lineIdx*lineH,
                              ignoreRuling=ignore,
                              drawFeedMmMin=drawFeed, travelFeedMmMin=travelFeed,
                              penUpDwellMs=penUpDwell, penDownDwellMs=penDownDwell)
    run = TIMING.Run("reproduce", chars=len(text), lines=1)

    with run.Step("synthesis"):
        #size the letters to the ruling BEFORE laying out, so the line breaks
        #are chosen for the size that will actually be written
        traj = SYN.SynthesizeJointBestOf(
            judged, text, profile, nTries=nTries,
            mmPerXh=REP.XhForRuling(profile, cfg, text),
            lineWidthMm=cfg.usableWidthMm,
            reader=TEXT_MODEL, authorModel=SHAPE_MODEL, authorMapping=SHAPE_MAP)

    jobId = uuid.uuid4().hex
    jobDir = JOBS/jobId
    jobDir.mkdir(parents=True, exist_ok=True)

    with run.Step("rendering"):
        #RenderTrajectory returns a grey array; the recogniser takes it as is,
        #only writing it to disk needs it wrapped
        preview = SYN.RenderTrajectory(traj, pxPerMm=18.0, profile=profile, uniformInk=True)
        Image.fromarray(np.asarray(preview, dtype=np.uint8)).save(jobDir/"preview.png")

    with run.Step("gcode"):
        written = REP.WriteGcode(traj, cfg, str(jobDir/"job.gcode"),
                                 title="web job %s" % author)

    #the generation stages are stored with the job so that drawing it later can
    #report the whole reproduction, generation included, against one allowance
    (jobDir/"meta.json").write_text(json.dumps({"text": text, "author": author,
                                                "nTries": nTries, "line": lineIdx,
                                                "lineHeightMm": lineH,
                                                "ignoreRuling": ignore,
                                                "drawFeedMmMin": drawFeed,
                                                "travelFeedMmMin": travelFeed,
                                                "penUpDwellMs": penUpDwell,
                                                "penDownDwellMs": penDownDwell,
                                                "generateStages": run.stages}),
                                    encoding="utf-8")
    _idx, probs = REC.ClassifyImage(preview, SHAPE_MODEL)
    return JobResponse(jobId, written, preview, run, expectedText=text,
                       pxPerMm=18.0, author=author, n_tries=nTries,
                       placement=written.get("placement"),
                       draw_feed_mm_min=drawFeed, travel_feed_mm_min=travelFeed,
                       pen_up_dwell_ms=penUpDwell, pen_down_dwell_ms=penDownDwell,
                       ignore_ruling=ignore,
                       #measured off the job that was just written, so a tweak
                       #can be judged from the page without running the gantry
                       budget=dict(drawMm=written.get("drawMm"),
                                   travelMm=written.get("travelMm"),
                                   penPulses=written.get("penPulses"),
                                   drawSeconds=written.get("drawSeconds"),
                                   travelSeconds=written.get("travelSeconds"),
                                   dwellSeconds=written.get("dwellSeconds"),
                                   estimatedSeconds=written.get("estimatedSeconds")),
                       writer_confidence_pct=round(100*float(probs[SHAPE_MAP[judged]]), 1))


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
    with run.Step("parsing gcode"):
        #ParseGcode hands back a list of strokes, which the renderer cannot take
        #on its own: it reads the writer and the size off a trajectory's meta
        strokes = REP.ParseGcode(str(gcodePath))
        traj = SYN.Trajectory(strokes, dict(author="uploaded",
                                            source=request.files["gcode"].filename))
    with run.Step("rendering"):
        #a whole page of traced ink at 18 px/mm is several thousand pixels square
        #and slow to draw, and this preview only has to be recognisable
        preview = SYN.RenderTrajectory(traj, pxPerMm=8.0)
        Image.fromarray(np.asarray(preview, dtype=np.uint8)).save(jobDir/"preview.png")
    written = dict(lines=sum(1 for _ in open(gcodePath, encoding="utf-8", errors="replace")),
                   penPulses=None)
    return JobResponse(jobId, written, preview, run, pxPerMm=8.0,
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
    if not REP.IsConnected():
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
            result = REP.WriteGcodeFile(str(gcodePath))
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
#CURATING A WRITER'S LETTERFORMS
#the filters that pick variants score how typical a shape is, which
#cannot tell a badly cut letter from an unusual one, so a wrong glyph
#can survive.  these three endpoints let a person look at every variant
#and say which to keep.  all the real work, including rebuilding the
#prototype, the medoid and the advances, is curate.py's, so the page and
#the command line cannot disagree about what a removal means
#======================================================================
@app.route("/api/curate/<author>")
def ApiCurateGet(author):
    base = author[:-8] if author.endswith("_curated") else author
    if base not in PROFILES and author not in PROFILES:
        abort(404)
    curated = request.args.get("curated") in ("1", "true", "yes")
    try:
        return jsonify(CUR.Variants(base, curated))
    except SystemExit as e:
        return jsonify({"error": str(e)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/curate/<author>", methods=["POST"])
def ApiCuratePost(author):
    if author not in PROFILES:
        abort(404)
    body = request.get_json(silent=True) or {}
    drop = {k: [int(i) for i in v] for k, v in (body.get("drop") or {}).items()}
    keep = {k: [int(i) for i in v] for k, v in (body.get("keep") or {}).items()}
    if not drop and not keep:
        return jsonify({"error": "nothing was marked for removal"}), 200
    try:
        summary = CUR.Apply(author, drop, keep,
                            curated=bool(body.get("curated")),
                            outName=body.get("out") or None)
    except SystemExit as e:
        return jsonify({"error": str(e)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    #the curated file is a profile like any other, so picking it up here
    #means it can be written with straight away instead of after a restart
    PROFILES.clear()
    PROFILES.update(LoadProfiles())
    summary["authorsNow"] = sorted(PROFILES)
    return jsonify(summary)


@app.route("/api/curate/<author>", methods=["DELETE"])
def ApiCurateDelete(author):
    #start again: the original was never written over, so throwing the
    #curated file away is the whole of the undo
    p = paths.Profiles/("%s_curated.json" % author)
    existed = p.exists()
    if existed:
        p.unlink()
    PROFILES.clear()
    PROFILES.update(LoadProfiles())
    return jsonify({"deleted": existed, "file": p.name, "authorsNow": sorted(PROFILES)})


if __name__ == "__main__":
    print("[server] listening on 0.0.0.0:%d, serving %s" % (PORT, FRONTEND))
    app.run(host="0.0.0.0", port=PORT, threaded=True)
