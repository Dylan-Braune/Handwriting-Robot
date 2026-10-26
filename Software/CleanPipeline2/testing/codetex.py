"""Build the theory manual as LaTeX, then compile it to PDF.

python codetex.py            build the .tex files and compile
python codetex.py --notex    build the .tex files only

Signatures and argument names come from the code itself via ast, so a
declaration printed in the manual is the real one.  The theory is written by
hand, one python file per module, in the format documented in
codetheory_common.py
"""
import ast
import importlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CP2 = HERE.parent
sys.path.insert(0, str(HERE))

OUT = HERE / "results" / "CodeTheory"
MODULES = [
 ("paths.py", "codetheory_paths"),
 ("segment.py", "codetheory_segment"),
 ("recognise.py", "codetheory_recognise"),
 ("style.py", "codetheory_style"),
 ("synthesise.py", "codetheory_synth"),
 ("reproduce.py", "codetheory_reproduce"),
 ("train.py", "codetheory_train"),
]

SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}", "|": r"$|$",
           #typographic characters prose is written with, spelt for LaTeX so
           #the source stays readable and pdflatex still accepts it
           "−": "$-$", "≈": r"$\approx$", "·": r"$\cdot$",
           "×": r"$\times$", "≤": r"$\le$", "≥": r"$\ge$",
           "°": r"$^\circ$", "²": r"\textsuperscript{2}",
           "³": r"\textsuperscript{3}", "–": "--",
           "—": "---", "’": "'", "“": "``", "”": "''",
           "α": r"$\alpha$", "β": r"$\beta$",
           "γ": r"$\gamma$", "λ": r"$\lambda$",
           "σ": r"$\sigma$", "θ": r"$\theta$",
           "μ": r"$\mu$", "π": r"$\pi$"}


#the only LaTeX commands prose is allowed to contain; anything else that looks
#like a command is escaped so it appears literally
PASSTHROUGH = ("\\cite{", "\\emph{", "\\textbf{", "\\texttt{", "\\ref{")


def Esc(s):
    """escape plain prose for LaTeX, leaving the allowed commands alone"""
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            hit = next((m for m in PASSTHROUGH if s.startswith(m, i)), None)
            if hit and "}" in s[i:]:
                j = s.index("}", i)
                out.append(s[i:j + 1])
                i = j + 1
                continue
        out.append(SPECIAL.get(c, c))
        i += 1
    return "".join(out)


def Rich(s):
    """**bold**, *emphasis* and `code` in prose, then escaped"""
    import re
    parts = []
    for piece in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", s):
        if not piece:
            continue
        if piece.startswith("**"):
            parts.append(r"\textbf{%s}" % Esc(piece[2:-2]))
        elif piece.startswith("*"):
            parts.append(r"\emph{%s}" % Esc(piece[1:-1]))
        elif piece.startswith("`"):
            parts.append(r"\texttt{%s}" % Esc(piece[1:-1]))
        else:
            parts.append(Esc(piece))
    return "".join(parts)


def Signature(node):
    a = node.args
    parts = []
    defaults = [None] * (len(a.args) - len(a.defaults)) + list(a.defaults)
    for arg, d in zip(a.args, defaults):
        parts.append(arg.arg + ("=" + ast.unparse(d) if d is not None else ""))
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        parts.append(arg.arg + ("=" + ast.unparse(d) if d is not None else ""))
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return "def %s(%s)" % (node.name, ", ".join(parts))


def Scan(fileName):
    """{name: (kind, signature, argnames, lineno)} in source order"""
    src = (CP2 / fileName).read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            out[node.name] = ("function", Signature(node), args, node.lineno)
        elif isinstance(node, ast.ClassDef):
            out[node.name] = ("class", "class %s" % node.name, [], node.lineno)
        elif isinstance(node, ast.Assign) and node.targets and \
                isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            out[name] = ("value", "%s = %s" % (name, ast.unparse(node.value)),
                         [], node.lineno)
    return out


def Block(kind, payload):
    """one theory block -> LaTeX"""
    if kind == "h":
        return "\n\\paragraph{%s}\n" % Rich(payload)
    if kind == "p":
        return Rich(payload) + "\n\n"
    if kind == "note":
        return ("\n\\begin{quote}\\small\\itshape %s\\end{quote}\n"
                % Rich(payload))
    if kind == "items":
        s = "\n\\begin{itemize}\\setlength\\itemsep{0pt}\n"
        for it in payload:
            s += "  \\item %s\n" % Rich(it)
        return s + "\\end{itemize}\n"
    if kind == "code":
        s = "\n\\begin{lstlisting}[style=plain]\n"
        for l in payload:
            s += l + "\n"
        return s + "\\end{lstlisting}\n"
    raise SystemExit("unknown block kind %r" % kind)


def Equation(eq):
    """(latex, [(symbol, meaning)], citeKey or None) -> equation + nomenclature"""
    body, where, cite = eq
    s = "\n\\begin{equation}\n%s\n\\end{equation}\n" % body
    if cite:
        s = s.replace("\\end{equation}",
                      "\\end{equation}\n\\noindent\\emph{Source: }\\cite{%s}\n"
                      % cite)
    if where:
        s += "\n\\noindent where\n\\begin{description}[leftmargin=2.4em,style=nextline,font=\\normalfont\\ttfamily]\\setlength\\itemsep{1pt}\n"
        for sym, mean in where:
            #braces round the label, else a ] inside the symbol ends it early
            s += "  \\item[{$%s$}] \\normalfont %s\n" % (sym, Rich(mean))
        s += "\\end{description}\n"
    return s


def WrapSig(sig, width=84):
    """hard wrap a long declaration; lstlisting cannot break some lines and
    a very long one overflows TeX's dimension limit"""
    if len(sig) <= width:
        return sig
    if len(sig) > 320:                       # a huge literal says nothing
        sig = sig[:300].rsplit(",", 1)[0] + ", ...}"
    out, line = [], ""
    for part in sig.replace(", ", ",\v").split("\v"):
        if len(line) + len(part) > width and line:
            out.append(line)
            line = "        " + part
        else:
            line += part
    out.append(line)
    return "\n".join(out)


def Entry(name, info, txt):
    kind, sig, args, _ln = info
    s = "\n\\section{%s}\n" % Esc(name)
    s += ("\\begin{lstlisting}[style=sig]\n%s\n\\end{lstlisting}\n"
          % WrapSig(sig))
    if txt is None:
        return s + "\\emph{Theory not yet written for this item.}\n"
    ins = txt.get("inputs")
    if ins:
        s += "\n\\noindent\\textbf{Inputs}\n"
        s += "\\begin{description}[leftmargin=2.2em,style=nextline,font=\\normalfont\\ttfamily]\\setlength\\itemsep{1pt}\n"
        for a, meaning in ins:
            s += "  \\item[{%s}] \\normalfont %s\n" % (Esc(a), Rich(meaning))
        s += "\\end{description}\n"
    elif kind == "function" and args:
        s += ("\n\\noindent\\textbf{Inputs:} %s\n\n"
              % Esc(", ".join(args)))
    if txt.get("output"):
        s += "\n\\noindent\\textbf{Output.} %s\n\n" % Rich(txt["output"])
    for b in txt.get("theory", []):
        if b[0] == "eq":
            s += Equation(b[1:])
        else:
            s += Block(b[0], b[1])
    return s


def Main(compile=True):
    OUT.mkdir(parents=True, exist_ok=True)
    common = importlib.import_module("codetheory_common")
    (OUT / "refs.bib").write_text(common.BIB, encoding="utf-8")

    bodies = []
    missing, total = [], 0
    for fileName, modName in MODULES:
        try:
            m = importlib.import_module(modName)
        except ModuleNotFoundError:
            m = None
        info = Scan(fileName)
        funcs = getattr(m, "FUNCS", {}) if m else {}
        chap = "\n\\chapter{%s}\n" % Esc(fileName)
        for b in (getattr(m, "INTRO", []) if m else []):
            chap += Block(b[0], b[1]) if b[0] != "eq" else Equation(b[1:])
        for name in sorted(info, key=lambda n: info[n][3]):
            total += 1
            t = funcs.get(name)
            if t is None:
                missing.append("%s.%s" % (fileName, name))
            chap += Entry(name, info[name], t)
        path = OUT / (modName + ".tex")
        path.write_text(chap, encoding="utf-8")
        bodies.append(modName + ".tex")

    main = common.PREAMBLE
    main += common.FRONT
    for b in bodies:
        main += "\n\\input{%s}\n" % b.replace(".tex", "")
    main += common.BACK
    (OUT / "main.tex").write_text(main, encoding="utf-8")
    print("wrote %s" % (OUT / "main.tex"))
    print("theory written for %d of %d items" % (total - len(missing), total))
    if missing:
        print("still to write (%d):" % len(missing))
        for x in missing[:40]:
            print("   ", x)
        if len(missing) > 40:
            print("    ... and %d more" % (len(missing) - 40))

    if compile:
        subprocess.run(["pdflatex", "-disable-installer",
                        "-interaction=nonstopmode", "main.tex"],
                       cwd=OUT, capture_output=True, text=True)
        b = subprocess.run(["bibtex", "main"], cwd=OUT,
                           capture_output=True, text=True)
        if "error" in (b.stdout or "").lower():
            print("bibtex reported a problem:")
            print(b.stdout[-800:])
        for i in range(2):
            r = subprocess.run(["pdflatex", "-disable-installer",
                                "-interaction=nonstopmode",
                                "-halt-on-error", "main.tex"],
                               cwd=OUT, capture_output=True, text=True)
            if r.returncode != 0:
                tail = [l for l in r.stdout.splitlines()
                        if l.startswith("!") or "Error" in l][:12]
                print("pdflatex failed on pass %d:" % (i + 1))
                print("\n".join(tail) or r.stdout[-1500:])
                return
        pdf = OUT / "main.pdf"
        if pdf.exists():
            print("compiled %s (%.0f KB)" % (pdf, pdf.stat().st_size / 1024))


if __name__ == "__main__":
    Main(compile="--notex" not in sys.argv)
