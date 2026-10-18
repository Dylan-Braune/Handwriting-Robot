#Flask backend for the Handwriting Robot web app.
#
#runs on the Odroid (where the trained models and the physical pipeline
#already live). The frontend (a separate, static HTML/CSS/JS site) talks
#to this over HTTP/JSON and can be hosted anywhere since it's just making
#fetch() calls to whatever address this server is reachable at.
#
#SECURITY NOTE (please read): this server has no user accounts and does
#essentially no input sanitisation beyond what's noted below. It is meant
#for use on your own local network. If you expose it to the open
#internet (e.g. via port forwarding), set the WEBAPP_API_KEY environment
#variable before starting it -- every request will then need an
#X-API-Key header matching it, which is a basic deterrent, not real
#security. Do not expose this to the internet without at least that, and
#ideally a VPN/reverse-proxy with TLS in front of it instead.
#
#run:
#    pip install flask flask-cors
#    python Server.py
#    (listens on 0.0.0.0:5000 by default -- change PORT below, or set the
#    PORT environment variable)
#
#endpoints (all JSON in/out unless noted):
#    GET  /api/authors                 -- list of authors for the dropdown
#    POST /api/camera/capture          -- trigger the camera, returns an image_id
#    POST /api/camera/preview/start    -- open the camera for live framing (toggle ON)
#    POST /api/camera/preview/stop     -- release the camera (toggle OFF)
#    GET  /api/camera/preview/stream   -- multipart/x-mixed-replace MJPEG stream (only while active)
#    GET  /api/images/<image_id>       -- serves a previously captured/uploaded image
#    POST /api/classify                -- multipart: image (file) or image_id, expected_text, expected_author
#    POST /api/generate                -- json: text, author, [nTries] -> gcode + preview
#    POST /api/gcode/upload            -- multipart: gcode (file), [expected_text] -> same details as /api/generate
#    POST /api/write/<job_id>          -- sends a generated/uploaded job's G-code to the physical gantry
#    GET  /api/gantry/status           -- {"connected", "calibrated", "calibration"} -- gantry state
#    POST /api/gantry/calibrate        -- runs the homing/calibration sequence, required before /api/write
#    GET  /api/gcode/<job_id>          -- serves the raw .gcode file for a generate/upload job
#    GET  /api/model_info              -- model specs + measured accuracy
#    GET  /api/glyphs/<author>         -- glyph-grid PNG for one author
#    GET  /api/samples/<author>        -- list of real sample crops (image URLs + transcriptions)
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask,request,jsonify,send_file,abort,Response
from flask_cors import CORS
from PIL import Image

import SynthesizeHandwriting as SY
import SegmentLean as SL
import WebRenderHelpers as WRH
import Camera as CAM
from Classify import PaperCRNNNumpy,ReadText,CharAcc,AuthorClassifierCNNNumpy,ClassifyImage
from AuthorsConfig import AuthorClassifierWeightsFilename

#Gantry.py is safe to import anywhere (see its own comments) -- gpiod
#access only happens inside Connect(), not at import time, so this
#doesn't crash on a laptop with no GPIO chip
import Gantry as GANTRY

SCRIPT_DIR = Path(__file__).resolve().parent
#this file lives in Software/CleanPipeline, a sibling of Software/CNN
#where the trained weights/NOGIT data actually live -- data is not
#duplicated into CleanPipeline, only code moves
CNN_DATA_DIR = SCRIPT_DIR.parent/"CNN"
NOGIT_DIR = CNN_DATA_DIR/"NOGIT"
UPLOAD_DIR = NOGIT_DIR/"WebUploads"
JOBS_DIR = NOGIT_DIR/"WebJobs"
UPLOAD_DIR.mkdir(parents=True,exist_ok=True)
JOBS_DIR.mkdir(parents=True,exist_ok=True)

API_KEY = os.environ.get("WEBAPP_API_KEY","")
PORT = int(os.environ.get("PORT","5000"))
DEBUG = os.environ.get("WEBAPP_DEBUG","0")=="1"

app = Flask(__name__)
CORS(app)

#load every model ONCE at startup, not per-request. All three run through
#Classify.py (pure numpy, no torch) -- these are the same trained .pt
#weights as before, just no longer executed by a library at inference time
print("[server] loading text recognizer...")
TEXT_MODEL = PaperCRNNNumpy()

print("[server] loading stroke-normalised writer-ID classifier (judges synthesis)...")
SHAPE_AUTHOR_MODEL = AuthorClassifierCNNNumpy()
SHAPE_MAPPING = SHAPE_AUTHOR_MODEL.author_mapping
SHAPE_IDX2AUTHOR = {v: k for k,v in SHAPE_MAPPING.items()}

print("[server] loading ink-based writer-ID classifier (classifies REAL photos)...")
_INK_WEIGHTS_CANDIDATES = [
    NOGIT_DIR.parent/"weights"/AuthorClassifierWeightsFilename(),
    NOGIT_DIR.parent/"weights"/"author_classifier_10_weights.pt",
]
INK_WEIGHTS_PATH = next((p for p in _INK_WEIGHTS_CANDIDATES if p.exists()), None)
if INK_WEIGHTS_PATH is None:
    raise SystemExit("No ink-based author classifier weights found -- run "
                      "`python TrainAuthor.py 10authors` first.")
INK_AUTHOR_MODEL = AuthorClassifierCNNNumpy(checkpoint_path=INK_WEIGHTS_PATH)
INK_MAPPING = INK_AUTHOR_MODEL.author_mapping
INK_IDX2AUTHOR = {v: k for k,v in INK_MAPPING.items()}

print("[server] loading style profiles...")
PROFILES = SY.LoadAllProfiles()

#pre-warm the stats page's glyph-grid/sample-crop caches in a BACKGROUND
#thread, not before app.run() -- the first pre-warm can take a couple of
#minutes for personal authors (page segmentation on their real photos),
#and the server would otherwise refuse every connection until that
#finished. Later starts are fast since results are cached to disk.
import threading

CACHE_WARM_DONE = {a: False for a in PROFILES}


def _prewarm():
    for _a in sorted(PROFILES):
        try:
            WRH.GlyphGridPath(_a)
            WRH.SampleCrops(_a,n=2)
        except Exception as _e:
            print(f"[server] WARNING: pre-warm failed for {_a}: {_e}")
        CACHE_WARM_DONE[_a] = True
    print("[server] cache pre-warm done (glyphs/samples pages are now instant).")

threading.Thread(target=_prewarm,daemon=True).start()

print(f"[server] ready. authors: {sorted(PROFILES)}")


def _CheckApiKey():
    if not API_KEY:
        return
    if request.headers.get("X-API-Key")!=API_KEY:
        abort(401)

@app.before_request
def _Auth():
    _CheckApiKey()


#runs the INK-based classifier on one real line image (NOT the
#stroke-normalised one -- that one is only valid for judging
#synthesised/machine-rendered output). Returns (author_id, probs_dict).
def _ClassifyInk(pil_line_img):
    idx,probs = ClassifyImage(pil_line_img,INK_AUTHOR_MODEL,apply_stroke_normalize=False)
    return INK_IDX2AUTHOR[idx],{INK_IDX2AUTHOR[i]: float(p) for i,p in enumerate(probs)}

#---------------------------------------------------------------------------
#GET /api/authors
#---------------------------------------------------------------------------
@app.route("/api/authors",methods=["GET"])
def ApiAuthors():
    out = []
    for a in sorted(PROFILES):
        p = PROFILES[a]
        out.append({
            "id": a,
            "label": a,
            "slantDeg": p.get("slantDeg"),
            "connectedness": p.get("connectedness"),
        })
    return jsonify(out)

#---------------------------------------------------------------------------
#POST /api/camera/capture
#---------------------------------------------------------------------------
@app.route("/api/camera/capture",methods=["POST"])
def ApiCameraCapture():
    try:
        path = CAM.CaptureImage()
    except Exception as e:
        return jsonify({"error": f"Camera capture failed: {e}"}),500
    image_id = uuid.uuid4().hex
    dest = UPLOAD_DIR/f"{image_id}.jpg"
    Image.open(path).convert("RGB").save(dest)
    return jsonify({"image_id": image_id,"image_url": f"/api/images/{image_id}"})

#---------------------------------------------------------------------------
#camera LIVE PREVIEW -- a toggle, not an always-on stream. Frame the shot
#and let autofocus settle while watching this, then hit /api/camera/capture
#to grab the good frame, then /api/camera/preview/stop. Leaving a USB
#webcam running 24/7 for no reason is needless heat/wear -- this keeps it
#off except while someone's actively using the Read page's capture flow.
#---------------------------------------------------------------------------
@app.route("/api/camera/preview/start",methods=["POST"])
def ApiCameraPreviewStart():
    try:
        CAM.PreviewStart()
    except Exception as e:
        return jsonify({"error": f"Could not start camera preview: {e}"}),500
    return jsonify({"preview_active": True})

@app.route("/api/camera/preview/stop",methods=["POST"])
def ApiCameraPreviewStop():
    CAM.PreviewStop()
    return jsonify({"preview_active": False})

@app.route("/api/camera/preview/stream",methods=["GET"])
def ApiCameraPreviewStream():
    if not CAM.PreviewActive():
        abort(409)   #not started -- call /api/camera/preview/start first

    def _Frames():
        while CAM.PreviewActive():
            try:
                jpeg = CAM.PreviewFrameJpeg()
            except Exception:
                break
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+jpeg+b"\r\n")
    return Response(_Frames(),mimetype="multipart/x-mixed-replace; boundary=frame")

#---------------------------------------------------------------------------
#GET /api/images/<image_id>
#---------------------------------------------------------------------------
@app.route("/api/images/<image_id>",methods=["GET"])
def ApiGetImage(image_id):
    safeId = "".join(c for c in image_id if c.isalnum())
    path = UPLOAD_DIR/f"{safeId}.jpg"
    if not path.exists():
        abort(404)
    return send_file(path,mimetype="image/jpeg")

#---------------------------------------------------------------------------
#POST /api/classify
#multipart/form-data: image=<file>  OR  image_id=<id from capture>
#                     expected_text=<str>  expected_author=<id>
#---------------------------------------------------------------------------
@app.route("/api/classify",methods=["POST"])
def ApiClassify():
    expectedText = request.form.get("expected_text","")
    expectedAuthor = request.form.get("expected_author","")

    if "image" in request.files and request.files["image"].filename:
        pilPage = Image.open(request.files["image"].stream).convert("RGB")
        image_id = uuid.uuid4().hex
        pilPage.save(UPLOAD_DIR/f"{image_id}.jpg")
    elif request.form.get("image_id"):
        image_id = "".join(c for c in request.form["image_id"] if c.isalnum())
        imgPath = UPLOAD_DIR/f"{image_id}.jpg"
        if not imgPath.exists():
            return jsonify({"error": "Unknown image_id"}),400
        pilPage = Image.open(imgPath).convert("RGB")
    else:
        return jsonify({"error": "Provide either 'image' (file) or 'image_id'"}),400

    #segment the photographed page into individual handwriting lines (same
    #personal-page segmenter used everywhere else in this project)
    import numpy as np
    gray = np.array(pilPage.convert("L"))
    tmpPath = UPLOAD_DIR/f"_classify_tmp_{image_id}.png"
    Image.fromarray(gray).save(tmpPath)
    try:
        leanLines,_overlay,_info = SL.SegmentLines(str(tmpPath))
    finally:
        tmpPath.unlink(missing_ok=True)
    lineCrops = [np.asarray(ln["image"]) for ln in leanLines]
    if not lineCrops:
        return jsonify({"error": "No handwriting lines were detected in this image."}),422

    predictedLines = []
    authorProbsSum = None
    for crop in lineCrops:
        pilLine = Image.fromarray(crop).convert("L")
        predictedLines.append(ReadText(pilLine,TEXT_MODEL))
        _predAuthor,probs = _ClassifyInk(pilLine)
        if authorProbsSum is None:
            authorProbsSum = dict(probs)
        else:
            for k,v in probs.items():
                authorProbsSum[k]+=v

    predictedText = " ".join(predictedLines)
    n = len(lineCrops)
    authorProbsAvg = {k: v/n for k,v in authorProbsSum.items()}
    predictedAuthor = max(authorProbsAvg,key=authorProbsAvg.get)
    author_confidence_pct = round(authorProbsAvg[predictedAuthor]*100,1)

    textAccuracyPct = None
    if expectedText.strip():
        textAccuracyPct = round(CharAcc(predictedText,expectedText)*100,1)

    authorCorrect = None
    if expectedAuthor:
        authorCorrect = (predictedAuthor==expectedAuthor)

    return jsonify({
        "predicted_text": predictedText,
        "text_accuracy_pct": textAccuracyPct,
        "predicted_author": predictedAuthor,
        "author_confidence_pct": author_confidence_pct,
        "author_correct": authorCorrect,
        "expected_author": expectedAuthor or None,
        "lines_detected": n,
        "per_author_confidence_pct": {k: round(v*100,1) for k,v in authorProbsAvg.items()},
    })


#shared response shape for /api/generate and /api/gcode/upload: reads the
#rendered preview back with the text recognizer and reports accuracy
#against expected_text when one was given
def _JobResponse(job_id,lineCount,penPulses,previewImg,expectedText=None,**extra):
    readBack = ReadText(previewImg,TEXT_MODEL)
    return jsonify({
        "job_id": job_id,
        "gcode_url": f"/api/gcode/{job_id}",
        "preview_url": f"/api/images/job_{job_id}",
        "gcode_line_count": lineCount,
        "pen_pulses": penPulses,
        "read_back_text": readBack,
        "text_accuracy_pct": (round(CharAcc(readBack,expectedText)*100,1)
                              if expectedText else None),
        **extra,
    })

#---------------------------------------------------------------------------
#POST /api/generate   json: {text, author, nTries?}
#---------------------------------------------------------------------------
@app.route("/api/generate",methods=["POST"])
def ApiGenerate():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    author = data.get("author")
    nTries = int(data.get("nTries",20))
    if not text:
        return jsonify({"error": "text is required"}),400
    if author not in PROFILES:
        return jsonify({"error": f"Unknown author {author!r}"}),400

    prof = PROFILES[author]
    cfg = SY.GantryConfig()

    t0 = time.time()
    traj = SY.SynthesizeJointBestOf(
        author,text,prof,nTries=nTries,mmPerXh=4.0,
        lineWidthMm=cfg.boundsMaxXmm-cfg.originXmm-5,
        reader=TEXT_MODEL,authorModel=SHAPE_AUTHOR_MODEL,
        authorMapping=SHAPE_MAPPING,
    )
    synthSeconds = time.time()-t0

    job_id = uuid.uuid4().hex
    jobDir = JOBS_DIR/job_id
    jobDir.mkdir(parents=True,exist_ok=True)

    gcodePath = jobDir/"job.gcode"
    gRes = SY.WriteGcode(traj,cfg,gcodePath,title=f"web job {author}")
    previewImg = SY.RenderTrajectory(traj,pxPerMm=18.0,profile=prof)
    previewImg.save(jobDir/"preview.png")

    return _JobResponse(job_id,gRes["lines"],gRes["penPulses"],previewImg,
                         expectedText=text,synthSeconds=round(synthSeconds,1))

@app.route("/api/gantry/status",methods=["GET"])
def ApiGantryStatus():
    connected = GANTRY.IsConnected()
    if not connected:
        try:
            GANTRY.Connect()
            connected = True
        except GANTRY.GantryNotConnectedError as e:
            return jsonify({"connected": False,"calibrated": False,"detail": str(e)})
    return jsonify({
        "connected": connected,
        "calibrated": GANTRY.Calibration["done"],
        "calibration": GANTRY.Calibration if GANTRY.Calibration["done"] else None,
    })

#runs the full homing/calibration sequence (see Calibrate() in Gantry.py):
#finds the real end-stop positions, measures the actual steps-per-mm for
#each axis, and parks the gantry at the safe starting corner. Must
#succeed before /api/write will run anything -- the frontend greys out
#Generate/Upload/Send until this has been called once per server session.
@app.route("/api/gantry/calibrate",methods=["POST"])
def ApiGantryCalibrate():
    try:
        GANTRY.Connect()
        result = GANTRY.Calibrate()
    except GANTRY.GantryNotConnectedError as e:
        return jsonify({"calibrated": False,"message": f"Gantry not connected: {e}"}),200
    except RuntimeError as e:
        return jsonify({"calibrated": False,"message": f"Calibration failed: {e}"}),200
    return jsonify({"calibrated": True,"calibration": result})

#sends a previously-generated (or uploaded) job's G-code to the physical
#gantry. On the Odroid with the hardware actually wired up, this really
#draws. Anywhere else (your laptop, testing), or if calibration hasn't
#been run yet this session, that's caught here and reported as a normal
#JSON response -- not a 500 -- since both are expected, everyday states.
@app.route("/api/write/<job_id>",methods=["POST"])
def ApiWriteGcode(job_id):
    safeId = "".join(c for c in job_id if c.isalnum())
    gcodePath = JOBS_DIR/safeId/"job.gcode"
    if not gcodePath.exists():
        return jsonify({"error": "Unknown job_id"}),404

    try:
        result = GANTRY.WriteGcodeFile(str(gcodePath))
    except GANTRY.GantryNotConnectedError as e:
        return jsonify({
            "sent": False,
            "gantry_connected": False,
            "message": f"Gantry not connected: {e}",
        }),200
    except GANTRY.NotCalibratedError as e:
        return jsonify({
            "sent": False,
            "gantry_connected": True,
            "calibrated": False,
            "message": f"Calibration required: {e}",
        }),200

    if result["estopped"]:
        return jsonify({
            "sent": False,
            "gantry_connected": True,
            "message": f"Job aborted by end-stop '{result['tripped_switch']}' mid-run. "
                       "Clear it on the gantry before sending another job.",
        }),200

    return jsonify({"sent": True,"gantry_connected": True,"message": "Job completed."})

#---------------------------------------------------------------------------
#POST /api/gcode/upload   multipart: gcode (file), [expected_text]
#lets you test drawing an arbitrary hand-made/downloaded G-code file
#through the exact same preview/read-back/send-to-gantry flow as a
#text-reproduction job
#---------------------------------------------------------------------------
@app.route("/api/gcode/upload",methods=["POST"])
def ApiGcodeUpload():
    if "gcode" not in request.files or not request.files["gcode"].filename:
        return jsonify({"error": "Provide a 'gcode' file"}),400
    expectedText = (request.form.get("expected_text") or "").strip()

    job_id = uuid.uuid4().hex
    jobDir = JOBS_DIR/job_id
    jobDir.mkdir(parents=True,exist_ok=True)
    gcodePath = jobDir/"job.gcode"
    request.files["gcode"].save(gcodePath)

    rawText = gcodePath.read_text(encoding="utf-8",errors="ignore")
    lineCount = sum(1 for line in rawText.splitlines() if line.strip())
    penPulses = sum(rawText.count(cmd) for cmd in ("M3","M5"))

    cfg = SY.GantryConfig()
    previewPath = jobDir/"preview.png"
    try:
        previewImg,_strokes = SY.RenderGcodePreview(str(gcodePath),cfg,path=str(previewPath))
    except Exception as e:
        return jsonify({"error": f"Could not parse/render that G-code file: {e}"}),422

    return _JobResponse(job_id,lineCount,penPulses,previewImg,expectedText=expectedText)

@app.route("/api/gcode/<job_id>",methods=["GET"])
def ApiGetGcode(job_id):
    safeId = "".join(c for c in job_id if c.isalnum())
    path = JOBS_DIR/safeId/"job.gcode"
    if not path.exists():
        abort(404)
    return send_file(path,mimetype="text/plain")

@app.route("/api/images/job_<job_id>",methods=["GET"])
def ApiGetJobPreview(job_id):
    safeId = "".join(c for c in job_id if c.isalnum())
    path = JOBS_DIR/safeId/"preview.png"
    if not path.exists():
        abort(404)
    return send_file(path,mimetype="image/png")

#---------------------------------------------------------------------------
#GET /api/model_info
#---------------------------------------------------------------------------
@app.route("/api/model_info",methods=["GET"])
def ApiModelInfo():
    def CountParams(model):
        return sum(arr.size for arr in model.sd.values())

    def FileMb(path):
        return round(path.stat().st_size/(1024*1024),2) if path and path.exists() else None

    return jsonify({
        "text_recognizer": {
            "architecture": "CNN-BiLSTM-CTC (PaperCRNN)",
            "weights_file": TEXT_MODEL.checkpoint_path.name,
            "parameters": CountParams(TEXT_MODEL),
            "file_size_mb": FileMb(TEXT_MODEL.checkpoint_path),
            "measured_accuracy": {
                "general_benchmark_val_pct": 94.03,
                "general_benchmark_test_pct": 91.82,
                "personal_val_pct": 89.64,
                "real_ink_10authors_pct": 89.4,
                "note": "Teklia/IAM-line general benchmark + this project's own 10-author real-ink measurement",
            },
        },
        "writer_id_ink_based": {
            "architecture": "CNN classifier (AuthorClassifierCNN)",
            "weights_file": INK_WEIGHTS_PATH.name,
            "parameters": CountParams(INK_AUTHOR_MODEL),
            "file_size_mb": FileMb(INK_WEIGHTS_PATH),
            "measured_accuracy": {
                "real_photographed_pages_pct": 100.0,
                "synthesised_output_pct": 20.0,
                "note": "Use ONLY on real photographed pages -- collapses on machine-rendered output because it relies on ink density/pressure a pen-plotter can't reproduce.",
            },
        },
        "writer_id_stroke_normalised": {
            "architecture": "CNN classifier, trained on skeletonised/re-inked (ink-density-invariant) input",
            "weights_file": SHAPE_AUTHOR_MODEL.checkpoint_path.name,
            "parameters": CountParams(SHAPE_AUTHOR_MODEL),
            "file_size_mb": FileMb(SHAPE_AUTHOR_MODEL.checkpoint_path),
            "measured_accuracy": {
                "real_ink_pct": 96.7,
                "synthesised_output_pct": 99.2,
                "note": "This is the judge used for synthesised/machine-rendered output.",
            },
        },
        "reproduction_pipeline": {
            "authors": sorted(PROFILES),
            "n_authors": len(PROFILES),
            "measured_on_novel_sentences": {
                "writer_id_direct_pct": 99.2,
                "writer_id_gcode_pct": 99.2,
                "text_char_direct_pct": 85.8,
                "text_char_gcode_pct": 84.1,
                "text_word_pct": 54.3,
                "method": "SynthesizeJointBestOf: best-of-N candidate draws scored by the harmonic mean of text accuracy and writer-ID confidence together, plus a joint-aware repair pass",
                "measured_via": "Evaluate.py rewrite, full official evaluation script",
            },
        },
        "device": "cpu (numpy inference, no ML library at runtime)",
    })

#---------------------------------------------------------------------------
#GET /api/glyphs/<author>
#---------------------------------------------------------------------------
@app.route("/api/glyphs/<author>",methods=["GET"])
def ApiGlyphs(author):
    if author not in PROFILES:
        abort(404)
    if not CACHE_WARM_DONE.get(author):
        return jsonify({"error": "Still preparing this author's data on the server "
                                  "(first-run cache warm-up) -- try again shortly."}),202
    try:
        path = WRH.GlyphGridPath(author)
    except Exception as e:
        return jsonify({"error": str(e)}),500
    return send_file(path,mimetype="image/png")

#---------------------------------------------------------------------------
#GET /api/samples/<author>
#---------------------------------------------------------------------------
@app.route("/api/samples/<author>",methods=["GET"])
def ApiSamples(author):
    if author not in PROFILES:
        abort(404)
    if not CACHE_WARM_DONE.get(author):
        return jsonify({"error": "Still preparing this author's data on the server "
                                  "(first-run cache warm-up) -- try again shortly."}),202
    try:
        crops = WRH.SampleCrops(author,n=2)
    except Exception as e:
        return jsonify({"error": str(e)}),500
    out = []
    for i,(path,text) in enumerate(crops):
        rel = path.name
        out.append({"image_url": f"/api/samples/{author}/{rel}","text": text})
    return jsonify(out)

@app.route("/api/samples/<author>/<filename>",methods=["GET"])
def ApiSampleFile(author,filename):
    safe = "".join(c for c in filename if c.isalnum() or c in "._-")
    path = WRH.CACHE_DIR/safe
    if not path.exists() or not path.name.startswith(f"sample_{author}_"):
        abort(404)
    return send_file(path,mimetype="image/png")

if __name__=="__main__":
    if not API_KEY:
        print("[server] WARNING: WEBAPP_API_KEY is not set -- this server has NO access "
              "control. Fine on a trusted local network; do not expose it to the open "
              "internet without setting this and putting a proper proxy/TLS in front of it.")
    if DEBUG:
        print("[server] WEBAPP_DEBUG=1 -- Flask debug mode is ON (auto-reload + interactive "
              "debugger). Only use this on a trusted network; the debugger allows arbitrary "
              "code execution from any request that triggers an unhandled exception.")
    app.run(host="0.0.0.0",port=PORT,debug=DEBUG)
