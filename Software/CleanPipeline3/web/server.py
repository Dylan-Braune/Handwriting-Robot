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
from PIL import Image

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
UPLOADS.mkdir(parents=True, exist_ok=True)
JOBS.mkdir(parents=True, exist_ok=True)

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
    Image.open(captured).convert("RGB").save(UPLOADS/("%s.jpg" % imageId))
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
            page = Image.open(request.files["image"].stream).convert("L")
            imageId = uuid.uuid4().hex
            page.convert("RGB").save(UPLOADS/("%s.jpg" % imageId))
        elif request.form.get("image_id"):
            imageId = "".join(c for c in request.form["image_id"] if c.isalnum())
            p = UPLOADS/("%s.jpg" % imageId)
            if not p.exists():
                return jsonify({"error": "Unknown image_id"}), 400
            page = Image.open(p).convert("L")
        else:
            return jsonify({"error": "Provide either 'image' or 'image_id'"}), 400
        gray = np.asarray(page, dtype=np.uint8)

    #SegmentLines records conditioning, segmenting and cropping into the run
    lines, _dbg, info = SEG.SegmentLines(gray, timer=run)
    crops = [np.asarray(ln["image"]) for ln in lines]
    if not crops:
        return jsonify({"error": "No handwriting lines were detected.",
                        "timing": Remember(run)}), 422

    readLines = []
    with run.Step("reading", chars=0) as step:
        for crop in crops:
            readLines.append(REC.ReadText(Image.fromarray(crop).convert("L"), TEXT_MODEL))
        step.chars = sum(len(t) for t in readLines)
    predictedText = " ".join(readLines)

    with run.Step("naming", chars=len(predictedText)):
        totals = None
        for crop in crops:
            _idx, probs = REC.ClassifyImage(Image.fromarray(crop).convert("L"),
                                            INK_MODEL, strokeNormalise=False)
            totals = np.asarray(probs, dtype=np.float64) if totals is None else totals+probs
        avg = totals/len(crops)
        bestIdx = int(np.argmax(avg))
        predictedAuthor = INK_BY_INDEX[bestIdx]

    run.chars = len(predictedText)
    run.lines = len(crops)
    return jsonify({
        "image_id": imageId,
        "predicted_text": predictedText,
        "predicted_lines": readLines,
        "text_accuracy_pct": (round(100*REC.CharAcc(predictedText, expectedText), 1)
                              if expectedText else None),
        "word_accuracy_pct": (round(100*REC.WordAcc(predictedText, expectedText), 1)
                              if expectedText else None),
        "predicted_author": predictedAuthor,
        "author_confidence_pct": round(100*float(avg[bestIdx]), 1),
        "author_correct": (predictedAuthor == expectedAuthor) if expectedAuthor else None,
        "lines_detected": len(crops),
        "page_angle_deg": info.get("angle"),
        "line_pitch_px": info.get("pitch"),
        "per_author_confidence_pct": {INK_BY_INDEX[i]: round(100*float(v), 1)
                                      for i, v in enumerate(avg)},
        "timing": Remember(run),
    })


#======================================================================
#WRITING TEXT IN A WRITER'S HAND
#the joint search draws nTries candidates and keeps whichever scores best
#on the harmonic mean of how readable it is and how much it looks like
#the requested writer, which is the selection that lifted both figures
#======================================================================
def JobResponse(jobId, written, previewImg, run, expectedText=None, **extra):
    with run.Step("reading back", chars=len(expectedText or "")):
        readBack = REC.ReadText(previewImg, TEXT_MODEL)
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
    cfg = dataclasses.replace(cfg, lineHeightMm=lineH, lineOffsetMm=lineIdx*lineH)
    run = TIMING.Run("reproduce", chars=len(text), lines=1)

    with run.Step("synthesis"):
        traj = SYN.SynthesizeJointBestOf(
            judged, text, profile, nTries=nTries, mmPerXh=4.0,
            lineWidthMm=cfg.boundsMaxXmm-cfg.originXmm-5,
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
                                                "generateStages": run.stages}),
                                    encoding="utf-8")
    _idx, probs = REC.ClassifyImage(preview, SHAPE_MODEL)
    return JobResponse(jobId, written, preview, run, expectedText=text,
                       author=author, n_tries=nTries,
                       placement=written.get("placement"),
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
        Image.open(request.files["image"].stream).convert("RGB").save(srcPath)
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
    return JobResponse(jobId, written, preview, run, expectedText=expectedText or None)


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
