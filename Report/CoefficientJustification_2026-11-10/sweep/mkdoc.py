#u23539195 coefficient justification: builds the LaTeX document from the registry
#and the recorded sweep results.  nothing here invents a number: every figure,
#table and statistic comes from results/*.json, which came from a real run
import json
from pathlib import Path

import registry

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
RESULTS = STUDY/"results"
LATEX = STUDY/"latex"


#---Tex: escapes the characters LaTeX treats specially, for text read from JSON---
def Tex(s):
    if s is None:
        return ""
    s = str(s)
    for a,b in (("\\",r"\textbackslash{}"),("&",r"\&"),("%",r"\%"),
                ("$",r"\$"),("#",r"\#"),("_",r"\_"),("{",r"\{"),("}",r"\}"),
                ("~",r"\textasciitilde{}"),("^",r"\textasciicircum{}")):
        s = s.replace(a,b)
    return s


#---Num: a value as it should appear in a results table---
def Num(v):
    if v is None:
        return "--"
    if isinstance(v,bool):
        return "yes" if v else "no"
    if isinstance(v,float):
        return ("%.4g" % v)
    if isinstance(v,(list,dict)):
        return Tex(json.dumps(v))[:40]
    return Tex(v)


#---LoadResult: one coefficient's recorded run---
def LoadResult(key):
    p = RESULTS/("%s.json" % key)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


#----------------------------------------------------------------------
#ResultTable: the five runs of one coefficient as a table, with the metrics the
#entry nominated as its columns
#----------------------------------------------------------------------
def ResultTable(entry,res):
    keys = [k for k in entry.get("metrics",[])
            if any(r["metrics"].get(k) is not None for r in res["runs"])]
    if not keys or not res["runs"]:
        return ""
    head = " & ".join([Tex(entry.get("symbol",entry["key"]))]
                      + [r"\textsf{%s}" % Tex(k) for k in keys])
    rows = []
    for r in res["runs"]:
        mark = r"\bfseries " if r["chosen"] else ""
        cells = [mark+Num(r["value"])]
        for k in keys:
            cells.append(mark+Num(r["metrics"].get(k)))
        rows.append(" & ".join(cells)+r" \\")
    return ("\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{%s: measured effect of each swept value. The submitted value is in bold.}"
        % Tex(entry["title"]),
        r"\begin{tabular}{l"+"r"*len(keys)+"}",
        r"\toprule",
        head+r" \\",
        r"\midrule",
        "\n".join(rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"]))


#----------------------------------------------------------------------
#EntrySection: one coefficient: what it is, why, what else was possible, a worked
#number, the sweep figure and the measured table
#----------------------------------------------------------------------
def EntrySection(entry):
    res = LoadResult(entry["key"])
    L = []
    L.append(r"\subsection{%s}" % Tex(entry["title"]))
    L.append(r"\label{coef:%s}" % entry["key"].replace("_","-"))

    #the identity block
    sym = entry.get("symbol",entry["key"])
    L.append(r"\begin{description}[leftmargin=5.2em,style=sameline]")
    L.append(r"\item[Symbol] %s" % sym)
    L.append(r"\item[Value] \texttt{%s}" % Tex(entry["default"]))
    L.append(r"\item[Source] \texttt{%s.py}, \texttt{%s}"
             % (Tex(entry["module"]),Tex(entry.get("where","--"))))
    L.append(r"\item[Swept] %s"
             % ", ".join(r"\texttt{%s}" % Tex(v) for v in entry["values"]))
    L.append(r"\end{description}")

    for label,field in (("Basis","source"),
                        ("Why this value","reason"),
                        ("What the coefficient does","understanding"),
                        ("Alternatives considered","alternatives"),
                        ("Worked example","worked")):
        if entry.get(field):
            L.append(r"\paragraph{%s}" % label)
            L.append(entry[field])

    if entry.get("prior"):
        L.append(r"\paragraph{Earlier evidence}")
        L.append(entry["prior"])

    if entry.get("observed"):
        L.append(r"\paragraph{What the sweep actually showed}")
        L.append(entry["observed"])

    if entry.get("notes"):
        L.append(r"\paragraph{Limits of this test}")
        L.append(entry["notes"])

    if res and res.get("noSweep"):
        L.append(r"\paragraph{Not swept} This coefficient was not swept; see "
                 r"the paragraph above for why, and what test would settle it.")
    elif res:
        failed = [r for r in res["runs"] if not r.get("ok")]
        if failed and len(failed) == len(res["runs"]):
            L.append(r"\paragraph{Every run failed} No value of this "
                     r"coefficient produced a result; the reason is recorded in "
                     r"\texttt{results/%s.json}." % entry["key"].replace("_","\\_"))
        else:
            if failed:
                L.append(r"\paragraph{Partial result} %d of the %d values failed "
                         r"to run; the failures are recorded in the results file "
                         r"and the panels for them are left blank."
                         % (len(failed),len(res["runs"])))
            fig = STUDY/("figures/%s.png" % entry["key"])
            if fig.exists() and not entry.get("skipFigure"):
                L.append(r"\begin{figure}[htbp]")
                L.append(r"\centering")
                L.append(r"\includegraphics[width=\linewidth]{../figures/%s.png}"
                         % entry["key"])
                L.append(r"\caption{%s. Each panel is the same input processed by "
                         r"the submitted code with only %s changed; the panel "
                         r"outlined in dark blue is the submitted value.}"
                         % (Tex(entry["title"]),sym))
                L.append(r"\label{fig:%s}" % entry["key"].replace("_","-"))
                L.append(r"\end{figure}")
            trend = STUDY/("figures/%s_trend.png" % entry["key"])
            if trend.exists():
                L.append(r"\begin{figure}[htbp]")
                L.append(r"\centering")
                L.append(r"\includegraphics[width=\linewidth]{../figures/%s_trend.png}"
                         % entry["key"])
                L.append(r"\caption{%s: measured quantities against the swept "
                         r"value. The circled point is the submitted value.}"
                         % Tex(entry["title"]))
                L.append(r"\end{figure}")
            t = ResultTable(entry,res)
            if t:
                L.append(t)
    else:
        L.append(r"\paragraph{No recorded run} This coefficient is documented "
                 r"but its sweep has not been run in this build of the study.")
    return "\n\n".join(L)


#---Bysubsystem: the entries grouped in the order they appear in the pipeline---
def BySubsystem(entries):
    order,groups = [],{}
    for e in entries:
        s = e.get("subsystem","Other")
        if s not in groups:
            groups[s] = []
            order.append(s)
        groups[s].append(e)
    return [(s,groups[s]) for s in order]


#----------------------------------------------------------------------
#Sensitivity: how far each coefficient moved its own headline metric across the
#sweep, so the document can say which constants matter on this probe and which
#did nothing measurable
#----------------------------------------------------------------------
def Sensitivity(entries):
    rows = []
    for e in entries:
        res = LoadResult(e["key"])
        if not res or res.get("noSweep") or not res.get("runs"):
            continue
        ok = [r for r in res["runs"] if r.get("ok")]
        if len(ok) < 2:
            continue
        #the first nominated metric that is numeric everywhere
        for k in e.get("metrics",[]):
            vals = [r["metrics"].get(k) for r in ok]
            if all(isinstance(v,(int,float)) and not isinstance(v,bool)
                   for v in vals):
                chosen = next((r["metrics"].get(k) for r in ok if r["chosen"]),None)
                lo,hi = min(vals),max(vals)
                if chosen is None:
                    break
                span = hi-lo
                rel = (span/abs(chosen)) if abs(chosen) > 1e-12 else (
                    0.0 if span == 0 else float("inf"))
                rows.append((e,k,lo,hi,chosen,rel))
                break
    rows.sort(key=lambda r: -r[5])
    return rows


#---SensitivityTable: that ranking as a LaTeX table---
def SensitivityTable(rows):
    out = [r"\begin{longtable}{p{0.23\linewidth}p{0.17\linewidth}rrrr}",
           r"\caption{How far each coefficient moved its headline metric across "
           r"its five values, on the fixed probe input. A span of zero means the "
           r"coefficient changed nothing measurable on this input, which is "
           r"reported rather than hidden.}\\",
           r"\toprule",
           r"\textsf{coefficient} & \textsf{metric} & \textsf{min} & "
           r"\textsf{submitted} & \textsf{max} & \textsf{span/submitted} \\",
           r"\midrule",r"\endfirsthead",
           r"\toprule",
           r"\textsf{coefficient} & \textsf{metric} & \textsf{min} & "
           r"\textsf{submitted} & \textsf{max} & \textsf{span/submitted} \\",
           r"\midrule",r"\endhead"]
    for (e,k,lo,hi,chosen,rel) in rows:
        relStr = "--" if rel == float("inf") else ("%.2f" % rel)
        out.append(r"\texttt{%s} & \textsf{%s} & %s & %s & %s & %s \\"
                   % (Tex(e["key"]),Tex(k),Num(lo),Num(chosen),Num(hi),relStr))
    out += [r"\bottomrule",r"\end{longtable}"]
    return "\n".join(out)


def Main():
    entries = registry.All()
    arch = json.loads((RESULTS/"architecture.json").read_text(encoding="utf-8"))

    body = []
    for subsystem,group in BySubsystem(entries):
        body.append(r"\section{%s}" % Tex(subsystem))
        for e in group:
            body.append(EntrySection(e))

    LATEX.mkdir(parents=True,exist_ok=True)
    (LATEX/"coefficients.tex").write_text("\n\n".join(body),encoding="utf-8")

    #a one-line-per-coefficient summary table for the front of the document
    rows = []
    for e in entries:
        res = LoadResult(e["key"])
        if res is None:
            state = "not run"
        elif res.get("noSweep"):
            state = "not swept"
        else:
            ok = sum(1 for r in res["runs"] if r.get("ok"))
            state = "%d/%d" % (ok,len(res["runs"]))
        #\path, not \texttt: a key like seam_tail_radius_dense is one
        #unbreakable word in typewriter and will run out of its column
        rows.append(r"\path{%s} & %s & \texttt{%s} & \texttt{%s} & %s \\"
                    % (e["key"],Tex(e.get("subsystem","")),
                       Tex(e["module"]),Tex(e["default"]),state))
    summary = "\n".join([
        r"\small",
        r"\begin{longtable}{p{0.27\linewidth}p{0.24\linewidth}p{0.11\linewidth}p{0.10\linewidth}l}",
        r"\caption{Every coefficient covered by this study.}\\",
        r"\toprule",
        r"\textsf{key} & \textsf{subsystem} & \textsf{module} & \textsf{value} & \textsf{runs} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textsf{key} & \textsf{subsystem} & \textsf{module} & \textsf{value} & \textsf{runs} \\",
        r"\midrule",
        r"\endhead",
        "\n".join(rows),
        r"\bottomrule",
        r"\end{longtable}"])
    (LATEX/"summary.tex").write_text(summary,encoding="utf-8")

    (LATEX/"sensitivity.tex").write_text(SensitivityTable(Sensitivity(entries)),
                                         encoding="utf-8")

    #the architecture numbers, as a file of macros so the prose stays readable
    a = []
    bb = arch["backbone"]
    a.append(r"\newcommand{\ArchBackboneParams}{%s}" % "{:,}".format(bb["params"]))
    a.append(r"\newcommand{\ArchConvLayers}{%d}" % bb["convLayers"])
    a.append(r"\newcommand{\ArchRF}{%d}" % arch["receptiveField"]["pixels"])
    a.append(r"\newcommand{\ArchStride}{%d}" % arch["receptiveField"]["stride"])
    a.append(r"\newcommand{\ArchReaderTotal}{%s}" % "{:,}".format(arch["reader"]["total"]))
    a.append(r"\newcommand{\ArchLstmParams}{%s}" % "{:,}".format(arch["reader"]["lstmParams"]))
    a.append(r"\newcommand{\ArchNamerTotal}{%s}" % "{:,}".format(arch["namer"]["total"]))
    a.append(r"\newcommand{\ArchFrames}{%d}" % arch["ctc"]["frames"])
    a.append(r"\newcommand{\ArchMaxLabel}{%d}" % arch["ctc"]["maxSafeLabelChars"])
    a.append(r"\newcommand{\ArchClasses}{%d}" % arch["reader"]["numClasses"])
    a.append(r"\newcommand{\ArchPNoAug}{%.2f}" % (100*arch["augmentation"]["pNoAugment"]))
    a.append(r"\newcommand{\ArchPEachAug}{%.2f}" % (100*arch["augmentation"]["pGivenAugmentApplied"]))
    (LATEX/"archmacros.tex").write_text("\n".join(a),encoding="utf-8")

    print("wrote coefficients.tex (%d entries), summary.tex, archmacros.tex"
          % len(entries))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(Main())
