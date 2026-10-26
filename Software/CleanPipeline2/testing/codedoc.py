"""Build the full technical reference for every function in the pipeline.

python codedoc.py

The skeleton is read straight out of the code with ast: every function, its
real signature, its defaults, what it returns, who calls it and what it calls.
The prose for each function lives in codedoc_text.py and is written by hand.
Anything in the code but missing from the prose is listed at the end of the
run, so the document cannot quietly drift out of date.

Writes testing/results/CodeReference.docx
"""
import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CP2 = HERE.parent
sys.path.insert(0, str(HERE))

from docx import Document                                        # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH                    # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT                   # noqa: E402
from docx.shared import Inches, Pt, RGBColor                     # noqa: E402

import codedoc_text as T                                         # noqa: E402

OUT = HERE / "results" / "CodeReference.docx"
FILES = ["paths.py", "segment.py", "recognise.py", "style.py",
         "synthesise.py", "reproduce.py", "train.py"]
MONO = "Consolas"


# ----------------------------------------------------------------- reading
def Signature(node):
    """the def line exactly as python sees it, defaults included"""
    a = node.args
    parts = []
    defaults = [None] * (len(a.args) - len(a.defaults)) + list(a.defaults)
    for arg, d in zip(a.args, defaults):
        s = arg.arg
        if d is not None:
            s += "=" + ast.unparse(d)
        parts.append(s)
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        parts.append(arg.arg + ("=" + ast.unparse(d) if d is not None else ""))
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return "%s(%s)" % (node.name, ", ".join(parts))


def Returns(node):
    """every distinct return expression, as written"""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Return):
            out.append("None" if n.value is None else ast.unparse(n.value))
    seen, uniq = set(), []
    for r in out:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def Scan():
    """{file: {name: info}} plus a project wide call graph"""
    mods, defined = {}, {}
    for f in FILES:
        src = (CP2 / f).read_text(encoding="utf-8")
        tree = ast.parse(src)
        items = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                methods = []
                if kind == "class":
                    methods = [(m.name, Signature(m), Returns(m))
                               for m in node.body
                               if isinstance(m, ast.FunctionDef)]
                items[node.name] = dict(
                    kind=kind, file=f, lineno=node.lineno,
                    end=node.end_lineno,
                    sig=Signature(node) if kind == "function" else node.name,
                    returns=Returns(node) if kind == "function" else [],
                    methods=methods,
                    calls=sorted({c.func.id for c in ast.walk(node)
                                  if isinstance(c, ast.Call)
                                  and isinstance(c.func, ast.Name)}))
                defined[node.name] = f
        mods[f] = items
    for f, items in mods.items():                 # who calls me
        for name, info in items.items():
            info["callers"] = sorted(
                "%s.%s" % (g.replace(".py", ""), other)
                for g, its in mods.items() for other, oi in its.items()
                if name in oi["calls"] and other != name)
            info["calls"] = [c for c in info["calls"] if c in defined]
    return mods


# ----------------------------------------------------------------- writing
def Mono(par, text, size=8.5):
    r = par.add_run(text)
    r.font.name, r.font.size = MONO, Pt(size)
    return r


def Rich(par, text, size=None):
    """**bold**, `code`, and $formula$ shown as its own monospace run"""
    import re
    for piece in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not piece:
            continue
        if piece.startswith("**"):
            r = par.add_run(piece[2:-2]); r.bold = True
        elif piece.startswith("`"):
            r = par.add_run(piece[1:-1]); r.font.name = MONO
            r.font.size = Pt(size or 9)
        else:
            r = par.add_run(piece)
        if size:
            r.font.size = Pt(size)


def Formula(doc, lines):
    for l in lines:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.space_after = Pt(2)
        Mono(p, l, 9.5)


def KeyValue(doc, rows):
    t = doc.add_table(rows=len(rows), cols=2)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    for i, (k, v) in enumerate(rows):
        c = t.cell(i, 0); c.text = ""
        r = c.paragraphs[0].add_run(k); r.bold = True; r.font.size = Pt(8)
        c2 = t.cell(i, 1); c2.text = ""
        p = c2.paragraphs[0]; p.paragraph_format.space_after = Pt(0)
        Mono(p, v, 8) if k in ("Signature", "Returns", "Defined") else Rich(p, v, 8.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def Main():
    mods = Scan()
    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.8)
        s.top_margin = s.bottom_margin = Inches(0.7)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10)

    doc.add_heading("Handwriting robot: code reference", 0)
    p = doc.add_paragraph()
    Rich(p, T.PREAMBLE)

    doc.add_heading("How to read this document", 1)
    for para in T.HOWTO:
        Rich(doc.add_paragraph(), para)

    missing, total = [], 0
    for f in FILES:
        doc.add_page_break()
        head = T.MODULES.get(f, {})
        doc.add_heading("%s" % f, 1)
        for para in head.get("intro", ["(no description written yet)"]):
            Rich(doc.add_paragraph(), para)
        order = sorted(mods[f], key=lambda n: mods[f][n]["lineno"])
        for name in order:
            info = mods[f][name]
            total += 1
            txt = T.FUNCS.get(name)
            if txt is None:
                missing.append("%s.%s" % (f, name))
            doc.add_heading("%s  %s" % (name, "(class)" if info["kind"] == "class" else ""), 2)
            rows = [("Signature", info["sig"]),
                    ("Defined", "%s lines %d-%d" % (f, info["lineno"], info["end"]))]
            if txt and txt.get("inputs"):
                rows.append(("Inputs", txt["inputs"]))
            if info["returns"]:
                rows.append(("Returns", "   |   ".join(info["returns"][:6])))
            if txt and txt.get("output"):
                rows.append(("Meaning of output", txt["output"]))
            if info["calls"]:
                rows.append(("Calls", ", ".join(info["calls"])))
            if info["callers"]:
                rows.append(("Called by", ", ".join(info["callers"][:10])))
            if txt and txt.get("cite"):
                rows.append(("Based on", txt["cite"]))
            KeyValue(doc, rows)
            if info["methods"]:
                doc.add_paragraph().add_run("Methods").bold = True
                for mn, ms, mr in info["methods"]:
                    pp = doc.add_paragraph()
                    pp.paragraph_format.left_indent = Inches(0.25)
                    pp.paragraph_format.space_after = Pt(1)
                    Mono(pp, ms, 8.5)
                    if mr:
                        Mono(pp, "   ->  " + " | ".join(mr[:3]), 8)
            if txt:
                for block in txt.get("body", []):
                    if isinstance(block, tuple):
                        kind, payload = block
                        if kind == "formula":
                            Formula(doc, payload)
                        elif kind == "sub":
                            r = doc.add_paragraph().add_run(payload)
                            r.bold = True
                            r.font.size = Pt(9.5)
                    else:
                        Rich(doc.add_paragraph(), block)
            else:
                r = doc.add_paragraph().add_run(
                    "Not yet written up in this document.")
                r.italic = True
                r.font.color.rgb = RGBColor(0x90, 0x90, 0x90)

    doc.add_page_break()
    doc.add_heading("References", 1)
    for key, ref in sorted(T.REFERENCES.items()):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3)
        r = p.add_run("[%s]  " % key); r.bold = True
        Rich(p, ref)

    doc.save(OUT)
    print("wrote %s" % OUT)
    print("documented %d of %d top level items" % (total - len(missing), total))
    if missing:
        print("still to write (%d):" % len(missing))
        for m in missing:
            print("   ", m)


if __name__ == "__main__":
    Main()
