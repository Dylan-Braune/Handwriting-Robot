"""
qtp6_storage.py -- QTP6 / requirement R6: >= 10 writing styles trained and permanently stored.

 1. list the stored style profiles (Software/CNN/NOGIT/StyleProfiles10/*.json): count, file size,
    modification time, content summary
 2. persistence across a fresh process: load every profile in a NEW python process (no shared
    state) and compare content hashes with the files on disk
 3. start server.py (the real web back end) with PORT=5099 as a child process, wait until it answers,
    GET /api/authors, pick a RANDOM writer (random.SystemRandom, recorded), POST /api/generate with
    a small nTries, record the response summary, fetch the G-code and the preview image, GET
    /api/model_info; then terminate the server process this script started
 4. stored model weight file sizes (recogniser + both writer classifiers)
"""
import os
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtp_common as C

PORT = 5099
BASE = f"http://127.0.0.1:{PORT}"
TEXT = "Hello robot"
N_TRIES = 3


def main():
    out = dict(machine=C.machine_info())
    # ---------------- 1 ----------------
    files = sorted(C.PROFILE_DIR.glob("*.json"))
    profs = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        profs.append(dict(author=f.stem, file=f.name, size_bytes=f.stat().st_size, size_kB=f.stat().st_size / 1024,
                          modified=datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"),
                          sha256_12=hashlib.sha256(open(f, "rb").read()).hexdigest()[:12],
                          n_top_level_keys=len(d), slantDeg=d.get("slantDeg"), connectedness=d.get("connectedness"),
                          legibilityLambda=d.get("legibilityLambda")))
    out["profiles"] = profs
    out["n_profiles"] = len(profs)
    out["total_size_kB"] = sum(p["size_kB"] for p in profs)
    print(f"{len(profs)} profiles, total {out['total_size_kB']:.0f} kB:", [p["author"] for p in profs], flush=True)

    # ---------------- 2 ----------------
    code = ("import sys,json,hashlib,time;"
            f"sys.path.insert(0,r'{C.ARS}');sys.path.insert(0,r'{C.CNN}');"
            "t=time.perf_counter();import SynthesizeHandwriting as SY;P=SY.LoadAllProfiles();"
            "print(json.dumps(dict(n=len(P),authors=sorted(P),load_s=time.perf_counter()-t,"
            "hash={a:hashlib.sha256(json.dumps(P[a],sort_keys=True).encode()).hexdigest()[:12] for a in P})))")
    t = time.perf_counter()
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    fresh = json.loads(r.stdout.strip().splitlines()[-1])
    # compare with an in-process load of the same files
    here = {f.stem: hashlib.sha256(json.dumps(json.load(open(f, encoding="utf-8")), sort_keys=True).encode()).hexdigest()[:12]
            for f in files}
    out["fresh_process_load"] = dict(n_profiles=fresh["n"], authors=fresh["authors"], load_s_in_fresh_process=fresh["load_s"],
                                     process_wall_s=time.perf_counter() - t, hashes_match_files=(fresh["hash"] == here))
    print("fresh process loaded", fresh["n"], "profiles; hashes match:", fresh["hash"] == here, flush=True)

    # ---------------- 4 ----------------
    w = {}
    for name in ("paper_cnn_bilstm_ctc_joint_best.pt", "author_classifier_10new_weights.pt", "author_shape_10new_weights.pt",
                 "paper_cnn_bilstm_ctc_hf_best.pt", "paper_cnn_bilstm_ctc_personal_best.pt"):
        p = C.WEIGHTS / name
        w[name] = dict(size_bytes=p.stat().st_size, size_MB=p.stat().st_size / 1024 / 1024,
                       modified=datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"))
    out["weights"] = w

    # ---------------- 3 ----------------
    env = dict(os.environ, PORT=str(PORT))
    env.pop("WEBAPP_API_KEY", None)
    logf = open(C.DATA_OUT / "qtp6_server_log.txt", "w", encoding="utf-8")
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-u", str(C.ARS / "server.py")], cwd=str(C.ARS), env=env,
                            stdout=logf, stderr=subprocess.STDOUT)
    srv = dict(pid=proc.pid, port=PORT)
    try:
        ready = False
        while time.perf_counter() - t0 < 600:
            if proc.poll() is not None:
                raise RuntimeError("server exited early, see qtp6_server_log.txt")
            try:
                r = requests.get(BASE + "/api/authors", timeout=3)
                if r.status_code == 200:
                    ready = True
                    break
            except requests.RequestException:
                pass
            time.sleep(1.0)
        srv["startup_s_until_api_answers"] = time.perf_counter() - t0
        srv["ready"] = ready
        if not ready:
            raise RuntimeError("server did not answer in 600 s")
        t = time.perf_counter()
        r = requests.get(BASE + "/api/authors", timeout=30)
        authors = r.json()
        srv["GET_api_authors"] = dict(status=r.status_code, seconds=time.perf_counter() - t, n_authors=len(authors),
                                      authors=authors)
        print("GET /api/authors ->", r.status_code, len(authors), "authors", flush=True)
        rnd = random.SystemRandom()
        chosen = rnd.choice([a["id"] for a in authors])
        srv["random_author_chosen"] = chosen
        t = time.perf_counter()
        r = requests.post(BASE + "/api/generate", json=dict(text=TEXT, author=chosen, nTries=N_TRIES), timeout=900)
        gen = r.json()
        srv["POST_api_generate"] = dict(status=r.status_code, request_wall_s=time.perf_counter() - t, text=TEXT,
                                        author=chosen, nTries=N_TRIES, response={k: v for k, v in gen.items()})
        print("POST /api/generate ->", r.status_code, {k: v for k, v in gen.items() if k not in ("preview_url", "gcode_url")}, flush=True)
        if r.status_code == 200:
            g = requests.get(BASE + gen["gcode_url"], timeout=30)
            srv["gcode_fetch"] = dict(status=g.status_code, bytes=len(g.content), lines=len(g.text.splitlines()),
                                      first_lines=g.text.splitlines()[:8])
            pimg = requests.get(BASE + gen["preview_url"], timeout=30)
            srv["preview_fetch"] = dict(status=pimg.status_code, bytes=len(pimg.content))
            if pimg.status_code == 200:
                (C.DATA_OUT / f"qtp6_generate_preview_{chosen}.png").write_bytes(pimg.content)
                srv["preview_file"] = f"qtp6_generate_preview_{chosen}.png"
        mi = requests.get(BASE + "/api/model_info", timeout=30)
        if mi.status_code == 200:
            m = mi.json()
            srv["model_info_summary"] = dict(
                recogniser=dict(file=m["text_recognizer"]["weights_file"], MB=m["text_recognizer"]["file_size_mb"], params=m["text_recognizer"]["parameters"]),
                writer_ink=dict(file=m["writer_id_ink_based"]["weights_file"], MB=m["writer_id_ink_based"]["file_size_mb"], params=m["writer_id_ink_based"]["parameters"]),
                writer_shape=dict(file=m["writer_id_stroke_normalised"]["weights_file"], MB=m["writer_id_stroke_normalised"]["file_size_mb"], params=m["writer_id_stroke_normalised"]["parameters"]),
                n_authors=m["reproduction_pipeline"]["n_authors"])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
            srv["stopped"] = "terminated, exit code %s" % proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            srv["stopped"] = "killed"
        logf.close()
    out["server"] = srv
    C.save_json(out, "qtp6_results.json")

    # ---------------- figure ----------------
    plt = C.mpl_style()
    from PIL import Image
    fig = plt.figure(figsize=(16 * C.CM, 7.5 * C.CM))
    ax = fig.add_axes([0.08, 0.15, 0.36, 0.78])
    names = [p["author"] for p in profs]
    ax.barh(names[::-1], [p["size_kB"] for p in profs][::-1], color=C.PALETTE[0])
    ax.set_xlabel("stored style profile [kB]")
    ax.set_title(f"{len(profs)} stored writers (JSON, {out['total_size_kB']/1024:.2f} MB)", fontsize=9)
    ax2 = fig.add_axes([0.50, 0.15, 0.48, 0.78])
    pf = srv.get("preview_file")
    if pf:
        im = Image.open(C.DATA_OUT / pf)
        ax2.imshow(im, cmap="gray"); ax2.axis("off")
        ro = srv["POST_api_generate"]["response"]
        ax2.set_title(f"POST /api/generate, writer '{srv['random_author_chosen']}' (random pick)\n"
                      f"read-back: '{ro.get('read_back_text')}', char acc {ro.get('text_accuracy_pct')} %", fontsize=8)
    else:
        ax2.axis("off")
    C.savefig(fig, "qtp6_storage.png"); plt.close(fig)


if __name__ == "__main__":
    main()
