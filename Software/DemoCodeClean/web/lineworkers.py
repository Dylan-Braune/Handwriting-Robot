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

import numpy as np

#how many workers; 0 asks for one per core, capped at the line count
LINE_WORKERS = int(os.environ.get("LINE_WORKERS", "0"))    #zero asks for one worker per core

#below this many lines the pool costs more to start than it saves
MIN_LINES_FOR_POOL = 4   #under four lines the pool costs more than it saves

#filled by the parent before any work is handed out, inherited on fork
_CFG = {}


#---Configure: hand the workers the models and flags the server is using---
def Configure(reader, namer, strokeNormalise):
    _CFG["reader"] = reader
    _CFG["namer"] = namer
    _CFG["norm"] = bool(strokeNormalise)


#---CanPool: whether this platform and this page justify a pool---
def CanPool(n):
    return (LINE_WORKERS != 1 and n >= MIN_LINES_FOR_POOL   # a pool needs fork the line count and loaded models
            and hasattr(os, "fork") and bool(_CFG))


#---_Workers: how many processes to run---
def _Workers(n):
    if LINE_WORKERS > 0:
        return min(LINE_WORKERS, n)
    return max(1, min(os.cpu_count() or 1, n))  # one worker a core capped at the number of lines


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
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   #the variables are read when a library loads
        os.environ[v] = "1"
    try:
        import ctypes
        import glob
        import numpy
        root = os.path.dirname(os.path.dirname(os.path.abspath(numpy.__file__)))   #the folder the numpy wheel ships its own libraries in
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
                        getattr(lib, fn)(1)   #the live thread count set to one through the library itself
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
    img = np.rot90(crop, 2) if flip else crop  #a half turn is two quarter turns
    text, score = REC.ReadTextScored(Image.fromarray(img).convert("L"),   # the reading and its mean column score
                                     _CFG["reader"])
    return i, text, score


#---NameOne: one crop's writer posterior, as NameWriter computes it---
def NameOne(job):
    import numpy as np
    from PIL import Image
    import recognise as REC
    i, crop = job
    _idx, p = REC.ClassifyImage(Image.fromarray(crop).convert("L"),   #the probability over the writers for this crop
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
        return [fn(j) for j in jobs]   #the sequential path gives the same answers
    try:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=_Workers(len(jobs)), initializer=_Init) as pool:   #a forked worker inherits the models already in memory
            out = pool.map(fn, jobs, chunksize=1)   # one line a worker at a time keeps the order
        return out
    except Exception as e:
        print("[lineworkers] pool unavailable (%s), running in one process"
              % type(e).__name__, file=sys.stderr, flush=True)
        return [fn(j) for j in jobs]


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
    got = Map(NameOne, [(i, c) for i, c in enumerate(crops)])   # every line votes rather than a spread sample
    per = [p for _i, p in sorted(got, key=lambda r: r[0])]   #the results put back into line order
    w = np.array([float((c < 128).sum()) for c in crops], dtype=np.float64)   #the dark pixel count is how much ink each line carries
    if w.sum() <= 0:
        w = np.ones(len(crops), dtype=np.float64)
    avg = (np.vstack(per)*w[:, None]).sum(axis=0)/w.sum()   #each line vote weighted by that ink then normalised
    return per, avg, w
