"""
qtp1_synth.py -- generate (and cache) all synthesised lines needed by QTP1.

PRODUCTION SYNTHESIS: SynthesizeHandwriting.SynthesizeJointBestOf exactly as server.py
/api/generate calls it (production judges: joint recogniser + stroke-normalised writer
classifier, mmPerXh=4.0, jitter=0.5 default, nTries=20 = server and web-UI default), then
RenderTrajectory(uniformInk=True, pxPerMm=18).  The ONLY difference from server.py:
lineWidthMm=10000 (no wrapping) so that every sentence is a single-line image, as
Evaluate.py does for its judges (server wraps at 185 mm).

Job set (all cached one-file-per-job under qtp_data/synth/, restartable):
  A  : 10 writers x 6 Evaluate.NOVEL_SENTENCES, seed = 100*sentence_index   (QTP1a, QTP1c)
  B  : for every real held-out line text T of every writer (3 per writer, picked by
       qtp_lines.pick_overlay_texts) T synthesised in ALL 10 writers' profiles, seed 7,
       nTries=10 (REDUCED from 20: 300 syntheses on a 4-core laptop)
       (QTP1b text-controlled overlay test, QTP1c matched real reference)
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # single-thread BLAS is faster per process here
os.environ.setdefault("OMP_NUM_THREADS", "1")
import io
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtp_common as C

N_TRIES = 20          # production quality (server + web-UI default) for the A set
N_TRIES_B = 10        # reduced for the 300-synthesis overlay set B (compute: 4-core laptop)
OUT = C.DATA_OUT / "synth"
_G = {}


def _init():
    import SynthesizeHandwriting as SY
    from np_inference.text_model import PaperCRNNNumpy
    from np_inference.author_model import AuthorClassifierCNNNumpy
    _G["SY"] = SY
    _G["reader"] = PaperCRNNNumpy()
    _G["auth"] = AuthorClassifierCNNNumpy()
    _G["profiles"] = SY.LoadAllProfiles()


def _run(job):
    name, writer, text, seed = job
    n_tries = N_TRIES if name.startswith("A_") else N_TRIES_B
    path = OUT / (name + ".pkl")
    if path.exists():
        return name, 0.0, "cached"
    SY = _G["SY"]
    prof = _G["profiles"][writer]
    t0 = time.perf_counter()
    traj = SY.SynthesizeJointBestOf(writer, text, prof, nTries=n_tries, mmPerXh=4.0,
                                    lineWidthMm=10000.0, reader=_G["reader"],
                                    authorModel=_G["auth"], authorMapping=_G["auth"].author_mapping,
                                    seed=seed)
    dt = time.perf_counter() - t0
    img = SY.RenderTrajectory(traj, pxPerMm=18.0, profile=prof, uniformInk=True)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    rec = dict(name=name, writer=writer, text=text, seed=seed, nTries=n_tries, seconds=dt,
               strokes=[[(float(x), float(y)) for (x, y) in s] for s in traj.strokes],
               meta={k: (v if not isinstance(v, dict) else dict(v)) for k, v in traj.meta.items()},
               png=buf.getvalue())
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(rec, f)
    os.replace(tmp, path)
    return name, dt, "done"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # NOVEL_SENTENCES copied verbatim from Evaluate.py (importing Evaluate drags in torch+TrainText);
    # verified identical below.
    sentences = [
        "The gantry writes this sentence for the first time today",
        "My robot copies handwriting from ten different authors",
        "Please bring the blue notebook and a sharp pencil",
        "Every careful measurement makes the next result better",
        "We tested the machine on Monday and it worked quietly",
        "Nothing about this line appears in the training pages",
    ]
    import re
    src = open(C.ARS / "Evaluate.py", encoding="utf-8").read()
    for s in sentences:
        assert f'"{s}"' in src, s
    import qtp_lines as L
    lines = L.load_real_lines()
    picks = L.pick_overlay_texts(lines, k=3)
    jobs = []
    for si, s in enumerate(sentences):
        for w in C.ALL_AUTHORS:
            jobs.append((f"A_{w}_s{si}", w, s, 100 * si))
    texts = []
    rank_of = {}
    for pi, l in enumerate(picks):
        rank_of[pi] = sum(1 for q in picks[:pi] if q["writer"] == l["writer"])   # 0,1,2 within its writer
        texts.append(dict(tid=pi, real_writer=l["writer"], page=l["page"], line_idx=l["line_idx"],
                          text=l["text"].strip(), rank=rank_of[pi]))
    # round-robin by rank so that stopping early still leaves a balanced set of writers
    for rk in range(3):
        for pi, l in enumerate(picks):
            if rank_of[pi] != rk:
                continue
            for w in C.ALL_AUTHORS:
                jobs.append((f"B_t{pi:02d}_{w}", w, l["text"].strip(), 7))
    C.save_json(dict(sentences=sentences, overlay_texts=texts, nTries=N_TRIES, nTries_B=N_TRIES_B,
                     n_jobs=len(jobs)), "qtp1_synth_jobs.json")
    todo = [j for j in jobs if not (OUT / (j[0] + ".pkl")).exists()]
    print(f"{len(jobs)} jobs ({len(todo)} to do), nTries={N_TRIES}, workers=6", flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=6, initializer=_init) as ex:
        for name, dt, st in ex.map(_run, todo, chunksize=1):
            done += 1
            if done % 5 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"[{done}/{len(todo)}] {name} {dt:.0f}s ({st}) elapsed {el/60:.1f} min, "
                      f"eta {(el/done)*(len(todo)-done)/60:.1f} min", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
