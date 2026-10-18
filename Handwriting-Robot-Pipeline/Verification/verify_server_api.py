"""Stage 6: boot the server and exercise its API, and verify the library-free
inference claim from inside the process that actually made it.

Two independent checks:

  1. IMPORT CLOSURE. server.py's whole selling point is that the production
     path runs a from-scratch numpy forward pass and never imports torch. This
     replicates server.py's exact import statements in the same order and then
     reports which heavyweight modules ended up in sys.modules -- measured,
     not assumed.

  2. API. The Flask app is loaded in-process with its test client, so no port
     is bound and nothing is left running: /api/authors (all 13 authors?),
     /api/classify on a real crop, and a full /api/generate round trip whose
     read_back_text is scored against the requested text.

Run from Software/CNN:
    python NOGIT/full_pipeline_run/verify_server_api.py
"""
import json
import sys
import time
from pathlib import Path

# path setup for the clean package layout
import sys as _sys
from pathlib import Path as _p
_root = _p(__file__).resolve().parent.parent
_cnn = _root / 'Software' / 'CNN'
_ar = _cnn / 'AuthorReproductionStuff'
_gantry = _root / 'Software' / 'GantryControl'
for _q in (_cnn, _ar, _ar / 'np_inference', _gantry):
    if str(_q) not in _sys.path:
        _sys.path.insert(0, str(_q))


THIS_DIR = Path(__file__).resolve().parent
CNN_DIR = _cnn
REPO = _root
AR = _ar

RESULTS = {}


def check_import_closure():
    print("=" * 78)
    print("CHECK 1: server import closure (is torch really absent?)")
    print("=" * 78)
    watched = ["torch", "torchvision", "TrainText", "ExtractIAMLines",
               "SegmentPage", "matplotlib", "scipy", "cv2", "tensorflow"]
    # These are exactly server.py's own top-level imports, in its order.
    from flask import Flask, request, jsonify, send_file, abort, Response  # noqa
    from flask_cors import CORS  # noqa
    from PIL import Image  # noqa
    import SynthesizeHandwriting as SY  # noqa
    import SegmentLean as SL  # noqa
    import web_render_helpers as WRH  # noqa
    import camera_capture as CAM  # noqa
    from np_inference.text_model import PaperCRNNNumpy, ReadText, CharAcc  # noqa
    from np_inference.author_model import AuthorClassifierCNNNumpy, ClassifyImage  # noqa
    from authors_config import author_classifier_weights_filename  # noqa
    import odroid_direct_drive as GANTRY  # noqa

    loaded = {}
    for m in watched:
        loaded[m] = m in sys.modules
    for m, v in loaded.items():
        print(f"  {'LOADED ' if v else 'absent '} {m}")
    RESULTS["import_closure"] = loaded
    RESULTS["torch_imported"] = loaded["torch"]
    RESULTS["gantry_on_windows"] = dict(
        gpiod_available=GANTRY.gpiod is not None,
        connected=GANTRY.is_connected(),
    )
    print(f"  gantry: gpiod={'yes' if GANTRY.gpiod else 'no'}, "
          f"connected={GANTRY.is_connected()}")
    return loaded


def check_api():
    print()
    print("=" * 78)
    print("CHECK 2: API endpoints (Flask test client, in-process)")
    print("=" * 78)
    import server as S
    client = S.app.test_client()

    # --- /api/authors -------------------------------------------------------
    t0 = time.time()
    r = client.get("/api/authors")
    dt = time.time() - t0
    authors = r.get_json() if r.status_code == 200 else None
    ids = [a["id"] for a in authors] if isinstance(authors, list) else []
    print(f"  GET /api/authors -> {r.status_code} in {dt*1000:.0f} ms, {len(ids)} authors")
    print(f"     {ids}")
    RESULTS["authors"] = dict(status=r.status_code, n=len(ids), ids=ids, ms=round(dt * 1000))

    # --- /api/classify on a real crop ---------------------------------------
    crop = _cnn / "Crops" / "Owen" / "Owen01" / "classifier" / "00.png"
    if not crop.exists():
        cands = sorted((_cnn / "Crops" / "Owen").rglob("*.png"))
        crop = cands[0] if cands else None
    if crop and crop.exists():
        # The file handle must stay open until the test client has streamed the
        # multipart body -- handing werkzeug a `with open(...)` result closes it
        # before the request is built ("read of closed file").
        fh = open(crop, "rb")
        try:
            t0 = time.time()
            r = client.post("/api/classify", data={"image": (fh, crop.name)},
                            content_type="multipart/form-data")
            dt = time.time() - t0
        finally:
            fh.close()
        body = r.get_json()
        print(f"  POST /api/classify ({crop.name}, true author Owen) -> {r.status_code} "
              f"in {dt*1000:.0f} ms")
        if isinstance(body, dict):
            print(f"     author={body.get('author')} confidence={body.get('confidence')}")
        RESULTS["classify"] = dict(status=r.status_code, crop=str(crop.relative_to(REPO)),
                                   truth="owen", response=body, ms=round(dt * 1000))

    # --- /api/generate round trip -------------------------------------------
    text = "test"
    author = "owen"
    print(f"  POST /api/generate text={text!r} author={author!r} (nTries=6, CPU)...")
    t0 = time.time()
    r = client.post("/api/generate", json={"text": text, "author": author, "nTries": 6})
    dt = time.time() - t0
    body = r.get_json() if r.is_json else {}
    print(f"     -> {r.status_code} in {dt:.1f}s")
    got = body.get("read_back_text")
    acc = body.get("text_accuracy_pct")
    print(f"     read_back_text={got!r}  text_accuracy_pct={acc}")
    RESULTS["generate"] = dict(status=r.status_code, requested=text, author=author,
                               response=body, seconds=round(dt, 1))

    # --- gcode sanity on every produced job ---------------------------------
    gcode_checks = []
    for p in sorted((CNN_DIR / "NOGIT" / "WebJobs").rglob("*.gcode")):
        txt = p.read_text(encoding="utf-8", errors="replace")
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        m3 = sum(1 for l in lines if l.startswith("M3"))
        m5 = sum(1 for l in lines if l.startswith("M5"))
        last_codes = [l for l in lines if l.startswith("M") or l.startswith("G")]
        gcode_checks.append(dict(
            file=str(p.relative_to(REPO)),
            m3=m3, m5=m5, balanced=(m3 == m5),
            ends_m2=bool(last_codes) and last_codes[-1].split()[0] == "M2",
            last_commands=last_codes[-4:],
            n_lines=len(lines),
        ))
    for g in gcode_checks[-6:]:
        print(f"  gcode {Path(g['file']).name}: M3={g['m3']} M5={g['m5']} "
              f"balanced={g['balanced']} ends_M2={g['ends_m2']}")
    RESULTS["gcode"] = gcode_checks

    # also sanity-check the Evaluate.py round-trip temp gcode
    tmpg = CNN_DIR / "NOGIT" / "EndToEnd" / "_tmp.gcode"
    if tmpg.exists():
        lines = [l.strip() for l in tmpg.read_text(errors="replace").splitlines() if l.strip()]
        cmds = [l for l in lines if l[:2] in ("M3", "M5", "M2", "G0", "G1")]
        RESULTS["rewrite_tmp_gcode"] = dict(
            file=str(tmpg.relative_to(REPO)),
            m3=sum(1 for l in cmds if l.startswith("M3")),
            m5=sum(1 for l in cmds if l.startswith("M5")),
            last=cmds[-1] if cmds else None,
        )

    RESULTS["server_boot_print"] = "see stage6 console output"
    return RESULTS


def main():
    check_import_closure()
    check_api()
    out = THIS_DIR / "stage6_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\n[Stage6] results -> {out}")


if __name__ == "__main__":
    main()
