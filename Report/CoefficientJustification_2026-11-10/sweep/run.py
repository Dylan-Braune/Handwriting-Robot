#u23539195 coefficient justification: runs one coefficient's sweep end to end
#for each of the five values it patches a throwaway copy of the submitted code,
#runs the stage in a separate process, then draws the five-panel figure
#   python run.py                 every coefficient
#   python run.py core_band_frac  just that one
#   python run.py --list          the keys
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import patch
import registry

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
FIGURES = STUDY/"figures"
RESULTS = STUDY/"results"
OUT = Path(os.environ.get("HWR_SWEEP_OUT",
    r"C:/Users/braun/AppData/Local/Temp/claude/C--Users-braun-OneDrive-Documents-GitHub-Handwriting-Robot/685927e6-23ae-495d-95f3-1824d845cbe0/scratchpad/out"))

#how many values of one coefficient are run at once
WORKERS = int(os.environ.get("HWR_SWEEP_WORKERS","5"))


#---Tag: a filesystem-safe name for one run---
def Tag(key,value):
    return "%s__%s" % (key,str(value).replace(".","p").replace("-","m")
                       .replace(" ","").replace("/","_"))


#----------------------------------------------------------------------
#RunOne: one value of one coefficient.  builds the patched copy, runs the task in
#its own process so no module state is shared, and returns the metrics
#----------------------------------------------------------------------
def RunOne(entry,value):
    tag = Tag(entry["key"],value)
    subs = {}
    #at the submitted value nothing is rewritten at all, so that panel is run
    #against the submitted source itself and not a re-formatted copy of it
    if value != entry["default"]:
        for (mod,old,new) in entry["patches"](value):
            subs.setdefault(mod,[]).append((old,new))
    vdir = patch.BuildVariant(tag,subs)
    shutil.copy(HERE/"worker.py",vdir/"worker.py")
    prefix = OUT/tag
    params = dict(entry["params"])
    #a task that needs a profile file gets one per run, so a style coefficient
    #really does change the profile the synthesis uses
    if params.get("_profileFrom"):
        params["profile"] = str(OUT/(tag+"_profile.json"))
    if entry["task"] == "gcode":
        params["gcodeOut"] = str(OUT/(tag+".gcode"))
    cmd = [sys.executable,"worker.py",entry["task"],json.dumps(params),str(prefix)]
    t0 = time.time()
    p = subprocess.run(cmd,cwd=str(vdir),capture_output=True,text=True,timeout=3600)
    secs = time.time()-t0
    jsonPath = Path(str(prefix)+".json")
    npzPath = Path(str(prefix)+".npz")
    if p.returncode != 0 or not jsonPath.exists():
        return dict(value=value,ok=False,
                    error=(p.stderr or "")[-1200:],seconds=round(secs,1),
                    npzPath=None,metrics={})
    metrics = json.loads(jsonPath.read_text(encoding="utf-8"))
    return dict(value=value,ok=True,seconds=round(secs,1),
                npzPath=(str(npzPath) if npzPath.exists() else None),
                metrics=metrics,variant=str(vdir))


#----------------------------------------------------------------------
#RunEntry: all five values of one coefficient, then both figures
#----------------------------------------------------------------------
def RunEntry(entry,workers=WORKERS):
    import compose
    key = entry["key"]
    print("[%s] %s" % (key,entry["title"]))
    if entry.get("noSweep"):
        #documented but not run: it needs hardware this study did not drive
        print("   not swept: %s" % (entry.get("notes","") or "")[:90])
        rec = dict(key=key,title=entry["title"],symbol=entry.get("symbol"),
                   subsystem=entry.get("subsystem"),module=entry["module"],
                   where=entry.get("where"),default=entry["default"],
                   values=entry["values"],task=None,params={},figure=None,
                   trend=None,noSweep=True,
                   submittedDigests=patch.SubmittedDigests(),runs=[])
        RESULTS.mkdir(parents=True,exist_ok=True)
        (RESULTS/("%s.json" % key)).write_text(json.dumps(rec,indent=1,default=str),
                                               encoding="utf-8")
        return rec
    values = entry["values"]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        runs = list(ex.map(lambda v: RunOne(entry,v),values))
    for r in runs:
        r["chosen"] = (r["value"] == entry["default"])
        flag = "ok " if r["ok"] else "FAIL"
        print("   %s %-10s %5.1fs %s" % (flag,r["value"],r["seconds"],
              (r.get("error","") or "").replace("\n"," ")[-110:] if not r["ok"] else ""))
    if not any(r["chosen"] for r in runs):
        print("   WARNING: no run matches the submitted default %r" % entry["default"])
    figPath = FIGURES/("%s.png" % key)
    compose.Figure(entry,runs,figPath)
    trendPath = FIGURES/("%s_trend.png" % key)
    got = compose.TrendFigure(entry,runs,trendPath)
    rec = dict(
        key=key,title=entry["title"],symbol=entry.get("symbol"),
        subsystem=entry.get("subsystem"),module=entry["module"],
        where=entry.get("where"),default=entry["default"],values=values,
        task=entry["task"],params=entry["params"],
        figure="figures/%s.png" % key,
        trend=("figures/%s_trend.png" % key) if got else None,
        submittedDigests=patch.SubmittedDigests(),
        runs=[{k:v for k,v in r.items() if k != "npzPath"} for r in runs])
    RESULTS.mkdir(parents=True,exist_ok=True)
    (RESULTS/("%s.json" % key)).write_text(json.dumps(rec,indent=1,default=str),
                                           encoding="utf-8")
    return rec


#----------------------------------------------------------------------
#RefigureEntry: redraws one coefficient's figures from the arrays a previous run
#left behind, so the look of the study can be changed without running the
#pipeline again
#----------------------------------------------------------------------
def RefigureEntry(entry):
    import compose
    key = entry["key"]
    recPath = RESULTS/("%s.json" % key)
    if not recPath.exists():
        return None
    rec = json.loads(recPath.read_text(encoding="utf-8"))
    if rec.get("noSweep") or not rec.get("runs"):
        return None
    runs = []
    for r in rec["runs"]:
        npz = OUT/(Tag(key,r["value"])+".npz")
        runs.append(dict(r,npzPath=(str(npz) if npz.exists() else None),
                         chosen=(r["value"] == rec["default"])))
    compose.Figure(entry,runs,FIGURES/("%s.png" % key))
    compose.TrendFigure(entry,runs,FIGURES/("%s_trend.png" % key))
    return key


def Main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--list" in sys.argv:
        for k in registry.Keys():
            print(k)
        return 0
    if "--refigure" in sys.argv:
        entries = registry.All()
        if args:
            entries = [e for e in entries if e["key"] in args]
        done = [RefigureEntry(e) for e in entries]
        done = [d for d in done if d]
        print("redrew %d figure set(s) from stored runs" % len(done))
        return 0
    expected = patch.SubmittedDigests()
    entries = registry.All()
    if args:
        entries = [e for e in entries if e["key"] in args]
        if not entries:
            print("no coefficient matches %s" % args)
            return 2
    FIGURES.mkdir(parents=True,exist_ok=True)
    t0 = time.time()
    done = []
    for e in entries:
        try:
            done.append(RunEntry(e))
        except Exception as ex:
            print("   ERROR in %s: %s" % (e["key"],ex))
    patch.VerifySubmittedUntouched(expected)
    print("\n%d coefficient(s) in %.1f min; submitted code verified unchanged"
          % (len(done),(time.time()-t0)/60))
    return 0


if __name__ == "__main__":
    sys.exit(Main())
