"""
server.py -- Flask backend for the Handwriting Robot web app.

Runs on the Odroid (where the trained models and the physical pipeline
already live). The frontend (a separate, static HTML/CSS/JS site) talks
to this over HTTP/JSON and can be hosted anywhere -- another machine on
your network, or opened as a local file -- since it's just making fetch()
calls to whatever address this server is reachable at.

SECURITY NOTE (please read): this server has no user accounts and does
essentially no input sanitisation beyond what's noted below. It is meant
for use on your own local network. If you expose it to the open
internet (e.g. via port forwarding), set the WEBAPP_API_KEY environment
variable before starting it -- every request will then need an
`X-API-Key` header matching it, which is a basic deterrent, not real
security. Do not expose this to the internet without at least that, and
ideally a VPN/reverse-proxy with TLS in front of it instead.

Run:
    pip install flask flask-cors
    python server.py
    (listens on 0.0.0.0:5000 by default -- change PORT below, or set the
    PORT environment variable)

Endpoints (all JSON in/out unless noted):
    GET  /api/authors                 -- list of authors for the dropdown
    POST /api/camera/capture          -- trigger the camera, returns an image_id
    POST /api/camera/preview/start    -- open the camera for live framing (toggle ON)
    POST /api/camera/preview/stop     -- release the camera (toggle OFF)
    GET  /api/camera/preview/stream   -- multipart/x-mixed-replace MJPEG stream (only while active)
    GET  /api/images/<image_id>       -- serves a previously captured/uploaded image
    POST /api/classify                -- multipart: image (file) or image_id, expected_text, expected_author
    POST /api/generate                -- json: text, author, [nTries] -> gcode + preview
    POST /api/gcode/upload            -- multipart: gcode (file), [expected_text] -> same details as /api/generate
    POST /api/write/<job_id>          -- sends a generated/uploaded job's G-code to the physical gantry
    GET  /api/gantry/status           -- {"connected", "calibrated", "calibration"} -- gantry state
    POST /api/gantry/calibrate        -- runs the homing/calibration sequence, required before /api/write
    GET  /api/gcode/<job_id>          -- serves the raw .gcode file for a generate/upload job
    GET  /api/model_info              -- model specs + measured accuracy
    GET  /api/glyphs/<author>         -- glyph-grid PNG for one author
    GET  /api/samples/<author>        -- list of real sample crops (image URLs + transcriptions)
"""
import io
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# odroid_direct_drive.py lives in a sibling top-level folder, not under
# Software/CNN -- add it explicitly rather than moving the gantry code
# into the CNN tree just for import convenience.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "GantryControl"))

from flask import Flask, request, jsonify, send_file, abort, Response
from flask_cors import CORS
from PIL import Image
import torch

import SynthesizeHandwriting as SY
import EvaluateStyle as ES
import VerifyRewrite as VR
import WriteGCode as GW
import SegmentPage as PS
import web_render_helpers as WRH
import camera_capture as CAM
from TrainAuthor import AuthorClassifierCNN
from TrainText import resize_line_image_fixed, tensor_from_resized

# odroid_direct_drive.py is safe to import anywhere now (see its own
# comments) -- gpiod access only happens inside connect(), not at import
# time, so this doesn't crash on a laptop with no GPIO chip.
import odroid_direct_drive as GANTRY

SCRIPT_DIR = Path(__file__).resolve().parent
NOGIT_DIR = SCRIPT_DIR.parent / "NOGIT"
UPLOAD_DIR = NOGIT_DIR / "WebUploads"
JOBS_DIR = NOGIT_DIR / "WebJobs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("WEBAPP_API_KEY", "")
PORT = int(os.environ.get("PORT", "5000"))
DEBUG = os.environ.get("WEBAPP_DEBUG", "0") == "1"

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Load every model ONCE at startup, not per-request.
# ---------------------------------------------------------------------------
print(f"[server] device: {DEVICE}")
print("[server] loading text recognizer...")
TEXT_MODEL = VR.LoadTextModel(DEVICE)

print("[server] loading stroke-normalised writer-ID classifier (judges synthesis)...")
SHAPE_AUTHOR_MODEL, SHAPE_MAPPING, SHAPE_IDX2AUTHOR = ES.LoadAuthorModel(DEVICE)

print("[server] loading ink-based writer-ID classifier (classifies REAL photos)...")
_INK_WEIGHTS_CANDIDATES = [
    NOGIT_DIR / "weights" / "author_classifier_10new_weights.pt",
    NOGIT_DIR / "weights" / "author_classifier_10_weights.pt",
]
INK_WEIGHTS_PATH = next((p for p in _INK_WEIGHTS_CANDIDATES if p.exists()), None)
if INK_WEIGHTS_PATH is None:
    raise SystemExit("No ink-based author classifier weights found -- run TrainAuthor10.py first.")
_ck = torch.load(INK_WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
INK_MAPPING = _ck["author_mapping"]
INK_IDX2AUTHOR = {v: k for k, v in INK_MAPPING.items()}
INK_AUTHOR_MODEL = AuthorClassifierCNN(num_authors=len(INK_MAPPING)).to(DEVICE)
INK_AUTHOR_MODEL.load_state_dict(_ck["model_state_dict"])
INK_AUTHOR_MODEL.eval()

print("[server] loading style profiles...")
PROFILES = SY.LoadAllProfiles()

# Pre-warm the stats page's glyph-grid/sample-crop caches in a BACKGROUND
# thread, not before app.run() -- the first pre-warm can take a couple of
# minutes for personal authors (it runs page segmentation on their real
# photos), and the server would otherwise refuse every connection,
# including for the Read/Write pages, until that finished. Subsequent
# server starts are fast since results are cached to disk; only the very
# first run after a fresh checkout pays this cost, and now it pays it
# without blocking anything else.
import threading

CACHE_WARM_DONE = {a: False for a in PROFILES}


def _prewarm():
    for _a in sorted(PROFILES):
        try:
            WRH.glyph_grid_path(_a)
            WRH.sample_crops(_a, n=2)
        except Exception as _e:
            print(f"[server] WARNING: pre-warm failed for {_a}: {_e}")
        CACHE_WARM_DONE[_a] = True
    print("[server] cache pre-warm done (glyphs/samples pages are now instant).")


threading.Thread(target=_prewarm, daemon=True).start()

print(f"[server] ready. authors: {sorted(PROFILES)}")


def _check_api_key():
    if not API_KEY:
        return
    if request.headers.get("X-API-Key") != API_KEY:
        abort(401)


@app.before_request
def _auth():
    _check_api_key()


def _classify_ink(pil_line_img):
    """Runs the INK-based classifier on one real line image (NOT the
    stroke-normalised one -- that one is only valid for judging
    synthesised/machine-rendered output, see EvaluateStyle.py's
    docstring). Returns (author_id, probs_dict)."""
    t = tensor_from_resized(resize_line_image_fixed(pil_line_img)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(INK_AUTHOR_MODEL(t), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return INK_IDX2AUTHOR[idx], {INK_IDX2AUTHOR[i]: float(p) for i, p in enumerate(probs)}


# ---------------------------------------------------------------------------
# GET /api/authors
# ---------------------------------------------------------------------------
@app.route("/api/authors", methods=["GET"])
def api_authors():
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


# ---------------------------------------------------------------------------
# POST /api/camera/capture
# ---------------------------------------------------------------------------
@app.route("/api/camera/capture", methods=["POST"])
def api_camera_capture():
    try:
        path = CAM.capture_image()
    except Exception as e:
        return jsonify({"error": f"Camera capture failed: {e}"}), 500
    image_id = uuid.uuid4().hex
    dest = UPLOAD_DIR / f"{image_id}.jpg"
    Image.open(path).convert("RGB").save(dest)
    return jsonify({"image_id": image_id, "image_url": f"/api/images/{image_id}"})


# ---------------------------------------------------------------------------
# Camera LIVE PREVIEW -- a toggle, not an always-on stream. Frame the shot
# and let autofocus settle while watching this, then hit /api/camera/capture
# to grab the good frame, then /api/camera/preview/stop. Leaving a USB
# webcam's sensor running 24/7 for no reason is needless heat/wear even
# with a heatsink -- this keeps it off except while someone's actively
# using the Read page's capture flow.
# ---------------------------------------------------------------------------
@app.route("/api/camera/preview/start", methods=["POST"])
def api_camera_preview_start():
    try:
        CAM.preview_start()
    except Exception as e:
        return jsonify({"error": f"Could not start camera preview: {e}"}), 500
    return jsonify({"preview_active": True})


@app.route("/api/camera/preview/stop", methods=["POST"])
def api_camera_preview_stop():
    CAM.preview_stop()
    return jsonify({"preview_active": False})


@app.route("/api/camera/preview/stream", methods=["GET"])
def api_camera_preview_stream():
    if not CAM.preview_active():
        abort(409)  # not started -- call /api/camera/preview/start first

    def _frames():
        while CAM.preview_active():
            try:
                jpeg = CAM.preview_frame_jpeg()
            except Exception:
                break
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
    return Response(_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# GET /api/images/<image_id>
# ---------------------------------------------------------------------------
@app.route("/api/images/<image_id>", methods=["GET"])
def api_get_image(image_id):
    safe_id = "".join(c for c in image_id if c.isalnum())
    path = UPLOAD_DIR / f"{safe_id}.jpg"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/jpeg")


# ---------------------------------------------------------------------------
# POST /api/classify
# multipart/form-data: image=<file>  OR  image_id=<id from capture>
#                      expected_text=<str>  expected_author=<id>
# ---------------------------------------------------------------------------
@app.route("/api/classify", methods=["POST"])
def api_classify():
    expected_text = request.form.get("expected_text", "")
    expected_author = request.form.get("expected_author", "")

    if "image" in request.files and request.files["image"].filename:
        pil_page = Image.open(request.files["image"].stream).convert("RGB")
        image_id = uuid.uuid4().hex
        pil_page.save(UPLOAD_DIR / f"{image_id}.jpg")
    elif request.form.get("image_id"):
        image_id = "".join(c for c in request.form["image_id"] if c.isalnum())
        img_path = UPLOAD_DIR / f"{image_id}.jpg"
        if not img_path.exists():
            return jsonify({"error": "Unknown image_id"}), 400
        pil_page = Image.open(img_path).convert("RGB")
    else:
        return jsonify({"error": "Provide either 'image' (file) or 'image_id'"}), 400

    # Segment the photographed page into individual handwriting lines
    # (same personal-page segmenter used everywhere else in this project).
    import numpy as np
    gray = np.array(pil_page.convert("L"))
    tmp_path = UPLOAD_DIR / f"_classify_tmp_{image_id}.png"
    Image.fromarray(gray).save(tmp_path)
    try:
        results, _preview, _meta = PS.ProcessPage(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)
    line_crops = [r["raw_crop"] for r in results if r.get("tag") == "TEXT"]
    if not line_crops:
        return jsonify({"error": "No handwriting lines were detected in this image."}), 422

    predicted_lines = []
    author_probs_sum = None
    for crop in line_crops:
        pil_line = Image.fromarray(crop).convert("L")
        predicted_lines.append(VR.ReadText(TEXT_MODEL, pil_line, DEVICE))
        _pred_author, probs = _classify_ink(pil_line)
        if author_probs_sum is None:
            author_probs_sum = dict(probs)
        else:
            for k, v in probs.items():
                author_probs_sum[k] += v

    predicted_text = " ".join(predicted_lines)
    n = len(line_crops)
    author_probs_avg = {k: v / n for k, v in author_probs_sum.items()}
    predicted_author = max(author_probs_avg, key=author_probs_avg.get)
    author_confidence_pct = round(author_probs_avg[predicted_author] * 100, 1)

    text_accuracy_pct = None
    if expected_text.strip():
        text_accuracy_pct = round(VR.CharAcc(predicted_text, expected_text) * 100, 1)

    author_correct = None
    if expected_author:
        author_correct = (predicted_author == expected_author)

    return jsonify({
        "predicted_text": predicted_text,
        "text_accuracy_pct": text_accuracy_pct,
        "predicted_author": predicted_author,
        "author_confidence_pct": author_confidence_pct,
        "author_correct": author_correct,
        "expected_author": expected_author or None,
        "lines_detected": n,
        "per_author_confidence_pct": {k: round(v * 100, 1) for k, v in author_probs_avg.items()},
    })


# ---------------------------------------------------------------------------
# POST /api/generate   json: {text, author, nTries?}
# ---------------------------------------------------------------------------
@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    author = data.get("author")
    n_tries = int(data.get("nTries", 20))
    if not text:
        return jsonify({"error": "text is required"}), 400
    if author not in PROFILES:
        return jsonify({"error": f"Unknown author {author!r}"}), 400

    prof = PROFILES[author]
    cfg = GW.GantryConfig()

    t0 = time.time()
    traj = SY.SynthesizeJointBestOf(
        author, text, prof, nTries=n_tries, mmPerXh=4.0,
        lineWidthMm=cfg.boundsMaxXmm - cfg.originXmm - 5,
        reader=TEXT_MODEL, authorModel=SHAPE_AUTHOR_MODEL,
        authorMapping=SHAPE_MAPPING, device=DEVICE,
    )
    synth_seconds = time.time() - t0

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    gcode_path = job_dir / "job.gcode"
    g_res = GW.WriteGcode(traj, cfg, gcode_path, title=f"web job {author}")
    preview_path = job_dir / "preview.png"
    SY.RenderTrajectory(traj, pxPerMm=18.0, profile=prof).save(preview_path)

    read_back = VR.ReadText(TEXT_MODEL, Image.open(preview_path), DEVICE)

    return jsonify({
        "job_id": job_id,
        "gcode_url": f"/api/gcode/{job_id}",
        "preview_url": f"/api/images/job_{job_id}",
        "gcode_line_count": g_res["lines"],
        "pen_pulses": g_res["penPulses"],
        "read_back_text": read_back,
        "text_accuracy_pct": round(VR.CharAcc(read_back, text) * 100, 1),
        "synth_seconds": round(synth_seconds, 1),
    })


@app.route("/api/gantry/status", methods=["GET"])
def api_gantry_status():
    connected = GANTRY.is_connected()
    if not connected:
        try:
            GANTRY.connect()
            connected = True
        except GANTRY.GantryNotConnectedError as e:
            return jsonify({"connected": False, "calibrated": False, "detail": str(e)})
    return jsonify({
        "connected": connected,
        "calibrated": GANTRY.calibration["done"],
        "calibration": GANTRY.calibration if GANTRY.calibration["done"] else None,
    })


@app.route("/api/gantry/calibrate", methods=["POST"])
def api_gantry_calibrate():
    """Runs the full homing/calibration sequence (see calibrate() in
    odroid_direct_drive.py): finds the real end-stop positions, measures
    the actual steps-per-mm for each axis, and parks the gantry at the
    safe starting corner. Must succeed before /api/write will run
    anything -- the frontend greys out Generate/Upload/Send until this
    has been called once per server session."""
    try:
        GANTRY.connect()
        result = GANTRY.calibrate()
    except GANTRY.GantryNotConnectedError as e:
        return jsonify({"calibrated": False, "message": f"Gantry not connected: {e}"}), 200
    except RuntimeError as e:
        return jsonify({"calibrated": False, "message": f"Calibration failed: {e}"}), 200
    return jsonify({"calibrated": True, "calibration": result})


@app.route("/api/write/<job_id>", methods=["POST"])
def api_write_gcode(job_id):
    """Sends a previously-generated (or uploaded) job's G-code to the
    physical gantry. On the Odroid with the hardware actually wired up,
    this really draws. Anywhere else (your laptop, testing), or if
    calibration hasn't been run yet this session, that's caught here and
    reported as a normal JSON response -- not a 500 -- since both are
    expected, everyday states, not server bugs."""
    safe_id = "".join(c for c in job_id if c.isalnum())
    gcode_path = JOBS_DIR / safe_id / "job.gcode"
    if not gcode_path.exists():
        return jsonify({"error": "Unknown job_id"}), 404

    try:
        result = GANTRY.write_gcode_file(str(gcode_path))
    except GANTRY.GantryNotConnectedError as e:
        return jsonify({
            "sent": False,
            "gantry_connected": False,
            "message": f"Gantry not connected: {e}",
        }), 200
    except GANTRY.NotCalibratedError as e:
        return jsonify({
            "sent": False,
            "gantry_connected": True,
            "calibrated": False,
            "message": f"Calibration required: {e}",
        }), 200

    if result["estopped"]:
        return jsonify({
            "sent": False,
            "gantry_connected": True,
            "message": f"Job aborted by end-stop '{result['tripped_switch']}' mid-run. "
                       "Clear it on the gantry before sending another job.",
        }), 200

    return jsonify({"sent": True, "gantry_connected": True, "message": "Job completed."})


# ---------------------------------------------------------------------------
# POST /api/gcode/upload   multipart: gcode (file), [expected_text]
# Lets you test drawing an arbitrary hand-made/downloaded G-code file
# (pictures, shapes, whatever) through the exact same preview/read-back/
# send-to-gantry flow as a text-reproduction job.
# ---------------------------------------------------------------------------
@app.route("/api/gcode/upload", methods=["POST"])
def api_gcode_upload():
    if "gcode" not in request.files or not request.files["gcode"].filename:
        return jsonify({"error": "Provide a 'gcode' file"}), 400
    expected_text = (request.form.get("expected_text") or "").strip()

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    gcode_path = job_dir / "job.gcode"
    request.files["gcode"].save(gcode_path)

    raw_text = gcode_path.read_text(encoding="utf-8", errors="ignore")
    line_count = sum(1 for line in raw_text.splitlines() if line.strip())
    pen_pulses = sum(raw_text.count(cmd) for cmd in ("M3", "M5"))

    cfg = GW.GantryConfig()
    preview_path = job_dir / "preview.png"
    try:
        preview_img, _strokes = GW.RenderGcodePreview(str(gcode_path), cfg, path=str(preview_path))
    except Exception as e:
        return jsonify({"error": f"Could not parse/render that G-code file: {e}"}), 422

    read_back = VR.ReadText(TEXT_MODEL, preview_img, DEVICE)

    response = {
        "job_id": job_id,
        "gcode_url": f"/api/gcode/{job_id}",
        "preview_url": f"/api/images/job_{job_id}",
        "gcode_line_count": line_count,
        "pen_pulses": pen_pulses,
        "read_back_text": read_back,
        "text_accuracy_pct": None,
    }
    if expected_text:
        response["text_accuracy_pct"] = round(VR.CharAcc(read_back, expected_text) * 100, 1)
    return jsonify(response)


@app.route("/api/gcode/<job_id>", methods=["GET"])
def api_get_gcode(job_id):
    safe_id = "".join(c for c in job_id if c.isalnum())
    path = JOBS_DIR / safe_id / "job.gcode"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="text/plain")


@app.route("/api/images/job_<job_id>", methods=["GET"])
def api_get_job_preview(job_id):
    safe_id = "".join(c for c in job_id if c.isalnum())
    path = JOBS_DIR / safe_id / "preview.png"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="image/png")


# ---------------------------------------------------------------------------
# GET /api/model_info
# ---------------------------------------------------------------------------
@app.route("/api/model_info", methods=["GET"])
def api_model_info():
    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    def file_mb(path):
        return round(path.stat().st_size / (1024 * 1024), 2) if path and path.exists() else None

    return jsonify({
        "text_recognizer": {
            "architecture": "CNN-BiLSTM-CTC (PaperCRNN)",
            "weights_file": VR.TEXT_WEIGHTS.name,
            "parameters": count_params(TEXT_MODEL),
            "file_size_mb": file_mb(VR.TEXT_WEIGHTS),
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
            "parameters": count_params(INK_AUTHOR_MODEL),
            "file_size_mb": file_mb(INK_WEIGHTS_PATH),
            "measured_accuracy": {
                "real_photographed_pages_pct": 100.0,
                "synthesised_output_pct": 20.0,
                "note": "Use ONLY on real photographed pages -- collapses on machine-rendered output because it relies on ink density/pressure a pen-plotter can't reproduce.",
            },
        },
        "writer_id_stroke_normalised": {
            "architecture": "CNN classifier, trained on skeletonised/re-inked (ink-density-invariant) input",
            "weights_file": ES.AUTHOR_WEIGHTS.name,
            "parameters": count_params(SHAPE_AUTHOR_MODEL),
            "file_size_mb": file_mb(ES.AUTHOR_WEIGHTS),
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
                "measured_via": "VerifyRewrite.py, full official evaluation script",
            },
        },
        "device": str(DEVICE),
    })


# ---------------------------------------------------------------------------
# GET /api/glyphs/<author>
# ---------------------------------------------------------------------------
@app.route("/api/glyphs/<author>", methods=["GET"])
def api_glyphs(author):
    if author not in PROFILES:
        abort(404)
    if not CACHE_WARM_DONE.get(author):
        return jsonify({"error": "Still preparing this author's data on the server "
                                  "(first-run cache warm-up) -- try again shortly."}), 202
    try:
        path = WRH.glyph_grid_path(author)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return send_file(path, mimetype="image/png")


# ---------------------------------------------------------------------------
# GET /api/samples/<author>
# ---------------------------------------------------------------------------
@app.route("/api/samples/<author>", methods=["GET"])
def api_samples(author):
    if author not in PROFILES:
        abort(404)
    if not CACHE_WARM_DONE.get(author):
        return jsonify({"error": "Still preparing this author's data on the server "
                                  "(first-run cache warm-up) -- try again shortly."}), 202
    try:
        crops = WRH.sample_crops(author, n=2)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    out = []
    for i, (path, text) in enumerate(crops):
        rel = path.name
        out.append({"image_url": f"/api/samples/{author}/{rel}", "text": text})
    return jsonify(out)


@app.route("/api/samples/<author>/<filename>", methods=["GET"])
def api_sample_file(author, filename):
    safe = "".join(c for c in filename if c.isalnum() or c in "._-")
    path = WRH.CACHE_DIR / safe
    if not path.exists() or not path.name.startswith(f"sample_{author}_"):
        abort(404)
    return send_file(path, mimetype="image/png")


if __name__ == "__main__":
    if not API_KEY:
        print("[server] WARNING: WEBAPP_API_KEY is not set -- this server has NO access "
              "control. Fine on a trusted local network; do not expose it to the open "
              "internet without setting this and putting a proper proxy/TLS in front of it.")
    if DEBUG:
        print("[server] WEBAPP_DEBUG=1 -- Flask debug mode is ON (auto-reload + interactive "
              "debugger). Only use this on a trusted network; the debugger allows arbitrary "
              "code execution from any request that triggers an unhandled exception.")
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)
