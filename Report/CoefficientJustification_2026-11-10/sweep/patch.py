#u23539195 coefficient justification: builds a patched copy of the submitted code
#the submitted modules are never written to.  one coefficient is rewritten in a
#throwaway copy, which is what the sweep imports, so every run differs from the
#submission by exactly one token
import hashlib
import os
import shutil
from pathlib import Path

#the submitted code, byte-identical to 23539195_BrauneDR_CODE.zip
HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parent.parent
SUBMITTED = REPO/"Software"/"u23539195_BrauneDR_Code"
DATA = REPO/"Software"/"CNN"

MODULES = ["paths","recognise","reproduce","segment","style","synthesise","train"]

#where the throwaway copies go
SCRATCH = Path(os.environ.get("HWR_SWEEP_SCRATCH",
    r"C:/Users/braun/AppData/Local/Temp/claude/C--Users-braun-OneDrive-Documents-GitHub-Handwriting-Robot/685927e6-23ae-495d-95f3-1824d845cbe0/scratchpad/variants"))


#---SourceOf: one submitted module's text---
def SourceOf(module):
    return (SUBMITTED/(module+".py")).read_text(encoding="utf-8")


#---Sha256: the digest of a file, so the report can show what was run---
def Sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


#---SubmittedDigests: the digest of every submitted module---
def SubmittedDigests():
    return {m: Sha256(SUBMITTED/(m+".py")) for m in MODULES}


#----------------------------------------------------------------------
#Substitute: rewrites one occurrence of old with new in a module's text, and
#refuses unless old appears exactly once, so a patch can never land on the
#wrong line
#----------------------------------------------------------------------
def Substitute(text,old,new):
    n = text.count(old)
    if n != 1:
        raise ValueError("pattern %r occurs %d times, need exactly 1" % (old,n))
    return text.replace(old,new)


#----------------------------------------------------------------------
#BuildVariant: a directory holding the seven modules with the given substitutions
#applied.  paths.py is also rewritten to point at the real data folder by
#absolute path, because the copy no longer sits beside it
#----------------------------------------------------------------------
def BuildVariant(tag,subs):
    #subs: {module: [(old,new),...]}
    out = SCRATCH/tag
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for m in MODULES:
        text = SourceOf(m)
        for (old,new) in subs.get(m,[]):
            text = Substitute(text,old,new)
        if m == "paths":
            text = Substitute(
                text,
                'Data = Path(__file__).resolve().parent.parent/"CNN"',
                'Data = Path(r"%s")' % str(DATA))
        (out/(m+".py")).write_text(text,encoding="utf-8")
    return out


#---BaselineVariant: the submitted code with no coefficient changed, only the data path---
def BaselineVariant():
    return BuildVariant("baseline",{})


#----------------------------------------------------------------------
#VerifySubmittedUntouched: the submitted folder must still match the digests recorded
#when this study started, checked after every batch of runs
#----------------------------------------------------------------------
def VerifySubmittedUntouched(expected):
    now = SubmittedDigests()
    bad = {m:(expected[m],now[m]) for m in expected if expected[m]!=now[m]}
    if bad:
        raise RuntimeError("submitted code changed: %s" % bad)
    return True
