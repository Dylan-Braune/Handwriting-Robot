"""Structural checks on the LaTeX sources of the final report.

Run from the FinalReport_v4 directory:  python verify_report.py
Each check that found something real during the v3 work is kept here so that a
rewrite cannot silently reintroduce it.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# The 33 keys the report is allowed to cite.  See REFERENCE_KEEPLIST.md.
STUDENT_SOURCES = {
    "chen2023", "teikari2016", "ma2023", "kumar2024", "mamun2024",
    "hemanth2022", "yahya2021", "ji2021", "iqs2024",
}
METHOD_SOURCES = {
    # The published approaches the system is built from.  Appendix 4 gives credit
    # for own first-principles IMPLEMENTATION; the approaches themselves come
    # from the literature and must be credited where they are used, never
    # presented as derived here.
    "kizilirmak2023",   # the CNN-BiLSTM-CTC recogniser architecture
    "graves2006",       # CTC loss and forward-backward
    "hochreiter1997",   # LSTM
    "ioffe2015",        # batch normalisation
    "srivastava2014",   # dropout
    "nair2010",         # rectified linear unit
    "kingma2015",       # Adam
    "loshchilov2019",   # AdamW decoupled weight decay
    "tieleman2012",     # RMSprop
    "otsu1979",         # global thresholding
    "crow1984",         # summed-area tables
    "zhang1984",        # Zhang-Suen thinning
    "viterbi1967",      # Viterbi / forced alignment
    "bresenham1965",    # integer line interpolation
    "marti2002",        # IAM database
    "bulacu2007",       # writer identification features
    "acarnley2002",     # stepper motor drive
    "iso6983",          # G-code
    # --- added for the expanded Section 1 survey (2026-10-21) ---------------
    "plamondon2000",    # online/offline handwriting survey
    "niblack1986", "sauvola2000", "bradley2007", "shafait2008",   # binarisation
    "postl1986", "vinciarelli2001",          # skew and slant normalisation
    "likforman2007", "wong1982", "avidan2007", "arvanitopoulos2014",  # line separation
    "lam1992",          # thinning survey
    "shi2017", "schuster1997", "graves2009", "puigcerver2017",   # line recognisers
    "michael2019", "li2021", "graves2014",   # seq2seq, transformer, beam search
    "he2015", "loshchilov2017", "pascanu2013", "simard2003",
    "wigington2017", "chellapilla2006", "sze2017",               # training techniques
    "levenshtein1966", "wagner1974",         # edit distance / error rates
    "brink2012", "christlein2017", "he2020", # writer identification
    "graves2013", "kang2020", "fogel2020", "bhunia2021", "dai2024",
    "haines2016",       # the four handwriting-generation families
    "bhunia2018",       # pen-trajectory recovery
    "douglas1973",      # Douglas-Peucker, original
    "tresset2013",      # drawing robot
    "biagiotti2008",    # velocity profiling
    "a4988ds", "stepperds", "bc337ds", "odroidn2ds",
}
OFF_LIMITS_SOURCES = {
    "barrow1977", "dubuisson1994", "efron1979", "fawcett2006", "fleiss1971",
    "huang2024", "jcgm2008", "kizilirmak2023", "levenshtein1966",
    "pearson1896", "wagner1974", "wilson1927",
}
ALLOWED_KEYS = STUDENT_SOURCES | METHOD_SOURCES | OFF_LIMITS_SOURCES

# Sections the student is writing himself; citations there are not our business.
OFF_LIMITS_FILES = ("4_Results", "5_Discussion", "6_Conclusion", "results/")


def tex_files():
    skip = {"epr400.cls"}
    for p in sorted(ROOT.rglob("*.tex")):
        if p.name in skip:
            continue
        yield p


def strip_comments(text):
    """Remove comment tails so that checks do not fire on commented-out code."""
    out = []
    for line in text.split("\n"):
        out.append(re.sub(r"(?<!\\)%.*$", "", line))
    return "\n".join(out)


def rel(p):
    return p.relative_to(ROOT).as_posix()


def check_duplicate_labels(sources):
    seen, dupes = {}, []
    for path, text in sources:
        for m in re.finditer(r"\\label\{([^}]+)\}", text):
            key = m.group(1)
            line = text[: m.start()].count("\n") + 1
            if key in seen:
                dupes.append(f"{key}: {seen[key]} and {rel(path)}:{line}")
            else:
                seen[key] = f"{rel(path)}:{line}"
    return dupes, set(seen)


def check_broken_refs(sources, labels):
    bad = []
    for path, text in sources:
        for m in re.finditer(r"\\(?:ref|eqref|autoref|pageref)\{([^}]+)\}", text):
            for key in m.group(1).split(","):
                key = key.strip()
                if key and key not in labels:
                    line = text[: m.start()].count("\n") + 1
                    bad.append(f"{rel(path)}:{line}: \\ref to undefined '{key}'")
    return bad


def check_unreferenced_equations(sources, texts_joined):
    """Appendix 4: every equation must be referred to in the text."""
    bad = []
    for path, text in sources:
        for m in re.finditer(r"\\label\{(eq:[^}]+)\}", text):
            key = m.group(1)
            if not re.search(r"\\(?:ref|eqref|autoref)\{[^}]*\b" + re.escape(key) + r"\b",
                             texts_joined):
                line = text[: m.start()].count("\n") + 1
                bad.append(f"{rel(path)}:{line}: equation '{key}' is never referred to")
    return bad


def check_unreferenced_floats(sources, texts_joined):
    bad = []
    for path, text in sources:
        for m in re.finditer(r"\\label\{((?:fig|tab):[^}]+)\}", text):
            key = m.group(1)
            if not re.search(r"\\(?:ref|autoref)\{[^}]*\b" + re.escape(key) + r"\b",
                             texts_joined):
                line = text[: m.start()].count("\n") + 1
                bad.append(f"{rel(path)}:{line}: float '{key}' is never referred to")
    return bad


def check_caption_outside_float(sources):
    """A \\caption outside a float environment breaks the build."""
    bad = []
    envs = ("figure", "table", "figure*", "table*", "subfigure", "wrapfigure",
            "longtable", "sidewaysfigure", "sidewaystable")
    for path, text in sources:
        depth = 0
        for n, line in enumerate(text.split("\n"), 1):
            for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", line):
                if m.group(2) in envs:
                    depth += 1 if m.group(1) == "begin" else -1
            if re.search(r"\\caption\b", line) and depth <= 0:
                bad.append(f"{rel(path)}:{n}: \\caption outside a float")
    return bad


def check_corrupt_row_breaks(sources):
    """A single backslash where a row break belongs collapses aligned maths."""
    bad = []
    math_envs = ("align", "align*", "aligned", "split", "gather", "gather*",
                 "cases", "array", "matrix", "pmatrix", "bmatrix",
                 "tabular", "tabularx", "eqnarray")
    for path, text in sources:
        depth = 0
        for n, line in enumerate(text.split("\n"), 1):
            opened = False
            for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", line):
                if m.group(2).rstrip("*") in [e.rstrip("*") for e in math_envs]:
                    depth += 1 if m.group(1) == "begin" else -1
                    opened = True
            if depth > 0 and not opened:
                # a lone trailing backslash, i.e. "\" not "\\"
                if re.search(r"(?<!\\)\\\s*$", line.rstrip()) and not line.rstrip().endswith("\\\\"):
                    bad.append(f"{rel(path)}:{n}: lone trailing backslash inside aligned maths")
    return bad


def check_unescaped_percent(sources):
    """An unescaped % silently comments out the rest of the line."""
    bad = []
    for path, text in sources:
        for n, line in enumerate(text.split("\n"), 1):
            stripped = line.lstrip()
            if stripped.startswith("%"):
                continue
            for m in re.finditer(r"(?<!\\)%", line):
                tail = line[m.end():].strip()
                # A real comment is usually empty or prose; a swallowed sentence
                # is the dangerous case.  Flag digits or a space then a word.
                before = line[max(0, m.start() - 3):m.start()]
                if re.search(r"[\d,.]\s*$", before) and tail:
                    bad.append(f"{rel(path)}:{n}: '%' after a number swallows: {tail[:60]!r}")
                break
    return bad


def check_citations(sources):
    bad = []
    for path, text in sources:
        if any(tag in rel(path) for tag in OFF_LIMITS_FILES):
            continue
        for m in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", text):
            for key in m.group(1).split(","):
                key = key.strip()
                if key and key not in ALLOWED_KEYS:
                    line = text[: m.start()].count("\n") + 1
                    bad.append(f"{rel(path)}:{line}: cites '{key}', which is not on the keep-list")
    return bad


def check_first_person(sources):
    """Appendix 4: the preamble "is the only section of the report written in the
    first person.  No other section of the report may be written in the first
    person."  So Part 3 must be impersonal."""
    bad = []
    # The trailing exclusion of "/" keeps "file I/O" from matching.
    pat = re.compile(r"(?<![\w\\])(I|my|mine|me|we|our|us)(?![\w'/])")
    math_envs = ("equation", "align", "gather", "multline", "aligned", "split",
                 "array", "cases", "tikzpicture", "verbatim", "lstlisting")
    for path, text in sources:
        if not rel(path).startswith("3_Main_Report"):
            continue
        depth = 0
        for n, line in enumerate(text.split("\n"), 1):
            s = line.strip()
            opened = False
            for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", line):
                if m.group(2).rstrip("*") in math_envs:
                    depth += 1 if m.group(1) == "begin" else -1
                    opened = True
            if depth > 0 or opened or s.startswith("%"):
                continue
            if s.startswith("\\node") or s.startswith("\\draw"):
                continue
            # The red markers are notes to the student, deleted before
            # submission, so first person inside them is not a violation.
            stripped = re.sub(r"\\must(?:do|complete)\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
                              " ", line)
            stripped = re.sub(r"\$[^$]*\$", " ", stripped)
            for m in pat.finditer(stripped):
                bad.append(f"{rel(path)}:{n}: first person '{m.group(1)}': {s[:70]}")
                break
    return bad


def check_heading_then_float(sources):
    """Appendix 4: never commence a section with a figure, and use at least one
    introductory sentence after a heading."""
    bad = []
    head = re.compile(r"\\(?:sub)*section\*?\{")
    for path, text in sources:
        lines = text.split("\n")
        for n, line in enumerate(lines):
            if not head.match(line.strip()):
                continue
            # look at what follows, skipping blank lines and \label
            for look in lines[n + 1:n + 8]:
                s = look.strip()
                if not s or s.startswith("\\label"):
                    continue
                if s.startswith("\\begin{figure") or s.startswith("\\begin{table"):
                    bad.append(f"{rel(path)}:{n+1}: heading is followed straight by a float: "
                               f"{line.strip()[:60]}")
                break
    return bad


def check_bold(sources):
    """The student asked for no bold in running prose.

    Captions, table headings and the abbreviation list are legitimate, so only
    body text in Sections 1 to 3 is reported.
    """
    hits = []
    for path, text in sources:
        if not rel(path).startswith("3_Main_Report"):
            continue
        if any(tag in rel(path) for tag in OFF_LIMITS_FILES):
            continue
        for n, line in enumerate(text.split("\n"), 1):
            stripped = line.strip()
            if stripped.startswith("%") or "\\caption" in line:
                continue
            if "&" in line or "\\hline" in line:  # a table row
                continue
            if "\\node" in line or "\\draw" in line:  # a label inside a diagram
                continue
            if "\\textbf{" in line:
                hits.append(f"{rel(path)}:{n}: {stripped[:90]}")
    return hits


def main():
    sources = [(p, strip_comments(p.read_text(encoding="utf-8", errors="replace")))
               for p in tex_files()]
    joined = "\n".join(t for _, t in sources)

    dupes, labels = check_duplicate_labels(sources)

    checks = [
        ("duplicate labels", dupes, True),
        ("undefined \\ref targets", check_broken_refs(sources, labels), True),
        ("\\caption outside a float", check_caption_outside_float(sources), True),
        ("corrupted row breaks in maths", check_corrupt_row_breaks(sources), True),
        ("unescaped % swallowing text", check_unescaped_percent(sources), True),
        ("equations never referred to", check_unreferenced_equations(sources, joined), True),
        ("floats never referred to", check_unreferenced_floats(sources, joined), True),
        ("citations off the keep-list", check_citations(sources), True),
        ("heading followed straight by a float", check_heading_then_float(sources), True),
        ("first person in Part 3", check_first_person(sources), True),
        ("\\textbf outside captions (advisory)", check_bold(sources), False),
    ]

    failed = 0
    for name, items, fatal in checks:
        if not items:
            print(f"  ok    {name}")
            continue
        mark = "FAIL " if fatal else "note "
        if fatal:
            failed += 1
        print(f"  {mark} {name}: {len(items)}")
        for item in items[:25]:
            print(f"          {item}")
        if len(items) > 25:
            print(f"          ... and {len(items) - 25} more")

    print()
    print(f"{len(sources)} .tex files checked, {len(labels)} labels.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
