#u23539195 coefficient justification: the live demonstration
#type a coefficient and the values you want to see, and it runs them and opens
#the picture.  nothing in the submitted code is touched
#
#   python demo.py                          list every coefficient
#   python demo.py darkness_gamma           its five recorded values
#   python demo.py darkness_gamma 0.2 0.9   the submitted value plus these
#   python demo.py darkness_gamma --info    what it is and why, no run
import json
import os
import subprocess
import sys
from pathlib import Path

import registry
import run as R

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent


#---Parse: turns a typed value into the type the coefficient uses---
def Parse(text,like):
    if isinstance(like,bool):
        return text.strip().lower() in ("1","true","yes","on")
    if isinstance(like,int) and not isinstance(like,bool):
        try:
            return int(text)
        except ValueError:
            return float(text)
    try:
        return float(text)
    except ValueError:
        return text


#----------------------------------------------------------------------
#Plain: the stored justification is LaTeX, because the document is built from the
#same text.  this renders it readably in a terminal
#----------------------------------------------------------------------
def Plain(t):
    import re
    #\cite[p. 63]{otsu1979} -> [otsu1979, p. 63]
    t = re.sub(r"\\cite\[([^\]]*)\]\{([^}]*)\}",r"[\2, \1]",t)
    t = re.sub(r"\\cite\{([^}]*)\}",r"[\1]",t)
    #\texttt{x}, \emph{x}, \textbf{x}, \textit{x} -> x ; \enquote{x} -> "x"
    t = re.sub(r"\\enquote\{([^{}]*)\}",r'"\1"',t)
    for cmd in ("texttt","emph","textbf","textit","mathrm","text"):
        t = re.sub(r"\\%s\{([^{}]*)\}" % cmd,r"\1",t)
    #simple maths
    t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}",r"(\1)/(\2)",t)
    t = re.sub(r"\\sqrt\{([^{}]*)\}",r"sqrt(\1)",t)
    t = re.sub(r"\\mathrm\{([^{}]*)\}",r"\1",t)
    for a,b in ((r"\times"," x "),(r"\approx","~"),(r"\le","<="),(r"\ge",">="),
                (r"\pm","+/-"),(r"\cdot","."),(r"\ldots","..."),(r"\dots","..."),
                (r"\,"," "),(r"\;"," "),(r"\ "," "),(r"\%","%"),(r"\&","&"),
                (r"\_","_"),(r"\$","$"),(r"\#","#"),(r"\\","\n"),
                (r"\gamma","gamma"),(r"\sigma","sigma"),(r"\phi","phi"),
                (r"\tau","tau"),(r"\theta","theta"),(r"\lambda","lambda"),
                (r"\alpha","alpha"),(r"\varepsilon","eps"),(r"\epsilon","eps"),
                (r"\min","min"),(r"\max","max"),(r"\circ","deg"),
                (r"\textbackslash{}","\\")):
        t = t.replace(a,b)
    #anything left: drop the backslash and the braces, keep the words
    t = re.sub(r"\\[a-zA-Z]+","",t)
    t = t.replace("{","").replace("}","").replace("$","")
    return re.sub(r"\s+"," ",t).strip()


#---ShowInfo: the written justification, without running anything---
def ShowInfo(e):
    print("=" * 72)
    print("%s  (%s)" % (e["title"],e["key"]))
    print("=" * 72)
    print("value in the submitted code : %s" % e["default"])
    print("lives in                    : %s.py, %s" % (e["module"],e.get("where")))
    print("recorded sweep values       : %s" % ", ".join(str(v) for v in e["values"]))
    print("pipeline stage run          : %s" % e["task"])
    for label,field in (("BASIS","source"),("WHY THIS VALUE","reason"),
                        ("WHAT IT DOES","understanding"),
                        ("ALTERNATIVES","alternatives"),
                        ("WORKED EXAMPLE","worked"),
                        ("LIMITS OF THE TEST","notes")):
        if e.get(field):
            print("\n%s" % label)
            t = Plain(e[field])
            words,line = t.split(),""
            for w in words:
                if len(line)+len(w) > 76:
                    print("  "+line)
                    line = w
                else:
                    line = (line+" "+w).strip()
            if line:
                print("  "+line)
    res = STUDY/"results"/("%s.json" % e["key"])
    if res.exists():
        rec = json.loads(res.read_text(encoding="utf-8"))
        print("\nLAST RECORDED RUN")
        for r in rec.get("runs",[]):
            mark = " <- submitted" if r.get("chosen") else ""
            ms = {k:v for k,v in (r.get("metrics") or {}).items()
                  if k in (e.get("metrics") or []) and not isinstance(v,(list,dict))}
            print("  %-10s %s%s" % (r["value"],ms,mark))


def Main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("coefficients you can demonstrate:\n")
        last = None
        for e in registry.All():
            if e.get("subsystem") != last:
                last = e.get("subsystem")
                print("  -- %s" % last)
            print("     %-22s %-10s %s" % (e["key"],e["default"],e["title"][:46]))
        print("\n  python demo.py <key>                 run its recorded values")
        print("  python demo.py <key> 0.2 0.9         run the submitted value plus these")
        print("  python demo.py <key> --info          explain it, run nothing")
        return 0

    key = args[0]
    try:
        entry = dict(registry.ByKey(key))
    except KeyError:
        print("no coefficient called %r; run with no arguments to list them" % key)
        return 2

    if "--info" in sys.argv:
        ShowInfo(entry)
        return 0

    if entry.get("noSweep"):
        print("%s is documented but cannot be swept offline:" % key)
        print(entry.get("notes",""))
        return 0

    if len(args) > 1:
        wanted = [Parse(a,entry["default"]) for a in args[1:]]
        if entry["default"] not in wanted:
            wanted.append(entry["default"])
        entry["values"] = sorted(
            wanted,key=lambda v: (isinstance(v,str),v))
        print("running %s at %s" % (key,", ".join(str(v) for v in entry["values"])))

    ShowInfo(entry)
    print("\nrunning...")
    rec = R.RunEntry(entry)
    fig = STUDY/("figures/%s.png" % key)
    print("\nfigure: %s" % fig)
    if fig.exists() and os.name == "nt" and "--no-open" not in sys.argv:
        try:
            os.startfile(str(fig))                      # noqa: S606
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(Main())
