#u23539195 D.R. Braune.  Spreading the per-line network passes over the
#board's cores.
#
#reading a line and naming its writer are independent of every other line,
#so the two loops that dominate a classify request are mapped over a pool
#of processes instead of run one after another.  measured on the board:
#one line costs 0.99 s read and 0.90 s name with the maths library using
#all six cores, and 1.385 s and 1.282 s with it held to one.  six cores
#therefore buy only 1.4x inside a single pass, which leaves most of the
#machine idle; six passes at once use it.
#
#the pool is only taken when the platform forks.  a forked child inherits
#the already-loaded models, so nothing is re-read from disk and nothing
#large is copied.  anywhere else, and for a short page, the ordinary
#sequential loop runs instead and the result is identical either way.
import os
import sys

#how many workers; 0 asks for one per core, capped at the line count
LINE_WORKERS = int(os.environ.get("LINE_WORKERS", "0"))

#below this many lines the pool costs more to start than it saves
MIN_LINES_FOR_POOL = 4

#filled by the parent before any work is handed out, inherited on fork
_CFG = {}


#---Configure: hand the workers the models and flags the server is using---
def Configure(reader, namer, strokeNormalise):
    _CFG["reader"] = reader
    _CFG["namer"] = namer
    _CFG["norm"] = bool(strokeNormalise)


#---CanPool: whether this platform and this page justify a pool---
def CanPool(n):
    return (LINE_WORKERS != 1 and n >= MIN_LINES_FOR_POOL
            and hasattr(os, "fork") and bool(_CFG))


#---_Workers: how many processes to run---
def _Workers(n):
    if LINE_WORKERS > 0:
        return min(LINE_WORKERS, n)
    return max(1, min(os.cpu_count() or 1, n))


#----------------------------------------------------------------------
#_CapThreads: one maths-library thread per worker
#six workers each asking the library for six threads would put 36 threads
#on six cores and spend the gain on contention.  the environment variables
#are read when the library loads, which has already happened in the
#parent, so the live setting is changed through the library itself where
#that entry point can be found, and the variables are set as well for any
#library loaded later
#----------------------------------------------------------------------
def _CapThreads():
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import ctypes
        import glob
        import numpy
        root = os.path.dirname(os.path.dirname(os.path.abspath(numpy.__file__)))
        pats = [os.path.join(root, "numpy.libs", "*openblas*.so*"),
                os.path.join(root, "numpy", ".libs", "*openblas*.so*")]
        #the wheel renames its symbols, so the plain name is not there: the
        #board carries libscipy_openblas64_, whose entry point is
        #scipy_openblas_set_num_threads64_.  every spelling seen in the
        #numpy and scipy builds is tried in turn
        names = ("scipy_openblas_set_num_threads64_",
                 "scipy_openblas_set_num_threads_64_",
                 "scipy_openblas_set_num_threads",
                 "openblas_set_num_threads64_",
                 "openblas_set_num_threads",
                 "scipy_goto_set_num_threads64_",
                 "goto_set_num_threads")
        for pat in pats:
            for so in glob.glob(pat):
                lib = ctypes.CDLL(so)
                for fn in names:
                    if hasattr(lib, fn):
                        getattr(lib, fn)(1)
                        return True
    except Exception:
        pass
    return False


def _Init():
    _CapThreads()


#---ReadOne: one crop read, as ReadAllLines reads it---
def ReadOne(job):
    import numpy as np
    from PIL import Image
    import recognise as REC
    i, crop, flip = job
    img = np.rot90(crop, 2) if flip else crop
    text, score = REC.ReadTextScored(Image.fromarray(img).convert("L"),
                                     _CFG["reader"])
    return i, text, score


#---NameOne: one crop's writer posterior, as NameWriter computes it---
def NameOne(job):
    import numpy as np
    from PIL import Image
    import recognise as REC
    i, crop = job
    _idx, p = REC.ClassifyImage(Image.fromarray(crop).convert("L"),
                                _CFG["namer"], strokeNormalise=_CFG["norm"])
    return i, np.asarray(p, dtype=np.float64)


#----------------------------------------------------------------------
#Map: run one of the two workers over a list of jobs
#returns the results in the order the jobs were given.  any failure in
#the pool falls back to running the same function in this process, so a
#page is never lost to a parallelism problem
#----------------------------------------------------------------------
def Map(fn, jobs):
    jobs = list(jobs)
    if not jobs:
        return []
    if not CanPool(len(jobs)):
        return [fn(j) for j in jobs]
    try:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=_Workers(len(jobs)), initializer=_Init) as pool:
            out = pool.map(fn, jobs, chunksize=1)
        return out
    except Exception as e:
        print("[lineworkers] pool unavailable (%s), running in one process"
              % type(e).__name__, file=sys.stderr, flush=True)
        return [fn(j) for j in jobs]
