"""Non-invasive size estimate per source file, for use while the PDF is being rebuilt.

Counts prose words, equations, figures, tables and code listings, and estimates
printed pages from them.  Run from the FinalReport_v4 directory.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Page equivalents, calibrated against the real printed spans of the build of
# 19 Oct 22:00 (3.3 = 23 pp, 3.4 = 19, 3.5 = 8, 3.6 = 7, 3.7 = 6, 3.8 = 3, 3.9 = 8).
#
# Accuracy: good in aggregate (it put Section 1 at 5.9 against a real 6, and
# Section 3 at 89 against a real ~86), but noisy per file, by up to about two
# pages.  A least-squares fit over those seven subsections produced a NEGATIVE
# cost per equation, i.e. five parameters overfitting seven points, so no better
# model is justified from this data.  Treat the output as a progress indicator
# only and settle every real question with `latexmk -pdf main.tex` followed by
# `python measure_pages.py`, which reads the printed folios.
#
# 2026-10-20: refitted against all eight measured subsection spans (3.3 to 3.10
# from main.toc) rather than guessed.  645 reproduces 3.3 at 13.8 against a real
# 14, 3.4 at 16.4 against 16, 3.8 at 3.75 against 4 and 3.9 at 8.5 against 9,
# rms 0.41 pages.  The previous 450 overstated every file by roughly a third.
WORDS_PER_PAGE = 645
EQ_PAGE = 0.06           # a displayed equation with its surrounding space
FIG_PAGE = 0.30          # a TikZ or included figure, caption included
TAB_PAGE = 0.26
LST_LINE_PAGE = 0.021    # one line of a verbatim/lstlisting block

ORDER = [
    ("1  Literature study", ["3_Main_Report/1_Literature_Study.tex"], 4, 8),
    ("2  Approach", ["3_Main_Report/2_Approach.tex"], 1, 2),
    ("3  Design and implementation", ["3_Main_Report/3_Design_and_Implementation.tex",
                                      "3_Main_Report/design/"], 30, 40),
    ("4  Results", ["3_Main_Report/4_Results.tex", "3_Main_Report/results/"], 10, 15),
    ("5  Discussion", ["3_Main_Report/5_Discussion.tex"], 6, 8),
    ("6  Conclusion", ["3_Main_Report/6_Conclusion.tex"], 1, 4),
]


def strip_comments(text):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in text.split("\n"))


def prose_words(text):
    """Words outside maths, floats, listings and TikZ."""
    t = text
    # drop whole float and verbatim environments
    for env in ("figure", "table", "lstlisting", "verbatim", "tikzpicture",
                "equation", "align", "gather", "tabularx", "tabular"):
        t = re.sub(r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}",
                   " ", t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", " ", t)          # inline maths
    t = re.sub(r"\\[a-zA-Z@]+\*?", " ", t)    # commands
    t = re.sub(r"[{}\[\]&~\\^_]", " ", t)
    return len([w for w in t.split() if any(c.isalpha() for c in w)])


def counts(text):
    eq = len(re.findall(r"\\begin\{(?:equation|align|gather|multline)\*?\}", text))
    fig = len(re.findall(r"\\begin\{figure\*?\}", text))
    tab = len(re.findall(r"\\begin\{table\*?\}", text))
    lst = 0
    for m in re.finditer(r"\\begin\{(lstlisting|verbatim)\}(.*?)\\end\{\1\}", text, re.S):
        lst += m.group(2).count("\n")
    return eq, fig, tab, lst


def collect(patterns):
    files = []
    for pat in patterns:
        p = ROOT / pat
        if pat.endswith("/"):
            if p.is_dir():
                files.extend(sorted(p.glob("*.tex")))
        elif p.exists():
            files.append(p)
    return files


def main():
    print(f"{'section / file':<46}{'words':>7}{'eq':>5}{'fig':>5}{'tab':>5}"
          f"{'code':>6}{'~pages':>8}{'target':>10}")
    print("-" * 92)
    grand = 0.0
    for name, patterns, lo, hi in ORDER:
        files = collect(patterns)
        tw = te = tf = tt = tl = 0
        rows = []
        for f in files:
            text = strip_comments(f.read_text(encoding="utf-8", errors="replace"))
            w = prose_words(text)
            eq, fig, tab, lst = counts(text)
            pages = (w / WORDS_PER_PAGE + eq * EQ_PAGE + fig * FIG_PAGE
                     + tab * TAB_PAGE + lst * LST_LINE_PAGE)
            rows.append((f.name, w, eq, fig, tab, lst, pages))
            tw += w; te += eq; tf += fig; tt += tab; tl += lst
        total = (tw / WORDS_PER_PAGE + te * EQ_PAGE + tf * FIG_PAGE
                 + tt * TAB_PAGE + tl * LST_LINE_PAGE)
        grand += total
        flag = "" if lo <= total <= hi else "  <-- OUT"
        print(f"{name:<46}{tw:>7}{te:>5}{tf:>5}{tt:>5}{tl:>6}{total:>8.1f}"
              f"{f'{lo}-{hi}':>10}{flag}")
        if len(rows) > 1:
            for n, w, eq, fig, tab, lst, pages in rows:
                print(f"    {n:<42}{w:>7}{eq:>5}{fig:>5}{tab:>5}{lst:>6}{pages:>8.1f}")
    print("-" * 92)
    print(f"{'main report total':<46}{'':>7}{'':>5}{'':>5}{'':>5}{'':>6}{grand:>8.1f}"
          f"{'50-70':>10}")
    print("\nEstimates only; run measure_pages.py against a fresh build for the real figure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
