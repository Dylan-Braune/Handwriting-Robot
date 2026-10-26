"""Measure the main-report page budget from the built PDF.

Appendix 4: the main report (Part 3) must be 50-70 pages.  The count starts at
page 1 of Part 3 and EXCLUDES the References section and the Appendix.

The previous version of this script text-searched the PDF for the words
"Part 3. Main Report" and "7. References" and found them in the table of
contents instead of in the body, so it reported a 2-page main report.  This
version instead reads the PRINTED page numbers off each physical page and maps
them to physical pages, then takes the section start pages from main.toc, which
is what the typeset document actually says.
"""
import re
import sys
from pathlib import Path

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
PDF = HERE / "main.pdf"
TOC = HERE / "main.toc"

MIN_PAGES, MAX_PAGES = 50, 70

reader = PdfReader(str(PDF))
n_phys = len(reader.pages)
texts = [(reader.pages[i].extract_text() or "") for i in range(n_phys)]


def printed_number(i):
    """The page number printed on physical page i, or None."""
    lines = [l.strip() for l in texts[i].strip().split("\n") if l.strip()]
    for line in reversed(lines[-4:]):
        if re.fullmatch(r"[0-9]{1,3}", line):
            return int(line)
    return None


# printed arabic page -> first physical page carrying it
arabic = {}
for i in range(n_phys):
    p = printed_number(i)
    if p is not None:
        arabic.setdefault(p, i + 1)

# section start pages, from the table-of-contents file the document wrote
toc = TOC.read_text(encoding="utf8", errors="replace")
sections = re.findall(
    r"\\contentsline\s*\{section\}\s*\{\\numberline\s*\{(\d+)\}([^}]*)\}\s*\{(\d+)\}",
    toc,
)

print(f"main.pdf: {n_phys} physical pages")
if arabic:
    print(f"printed page 1 is physical page {arabic.get(1)}; "
          f"highest printed page is {max(arabic)}")
print()

if not sections:
    sys.exit("could not read section start pages from main.toc - build the document first")

print(f"{'section':<34}{'starts on printed page':>24}")
print("-" * 58)
refs_start = None
for num, title, page in sections:
    title = re.sub(r"\\[a-zA-Z]+\s*", "", title).strip()
    print(f"{num + '. ' + title:<34}{page:>24}")
    if title.lower().startswith("reference"):
        refs_start = int(page)

# per-section lengths
print()
starts = [(int(p), re.sub(r"\\[a-zA-Z]+\s*", "", t).strip()) for _, t, p in sections]
starts.sort()
GUIDE = {
    "Literature study": (4, 8),
    "Approach": (1, 2),
    "Design and implementation": (30, 40),
    "Results": (10, 15),
    "Discussion": (6, 8),
    "Conclusion": (1, 4),
}
print(f"{'section':<30}{'pages':>7}{'guideline':>13}{'':>4}")
print("-" * 58)
for (p0, t0), (p1, _t1) in zip(starts, starts[1:]):
    span = p1 - p0
    lo_hi = GUIDE.get(t0)
    if lo_hi:
        lo, hi = lo_hi
        flag = "ok" if lo <= span <= hi else ("+%d" % (span - hi) if span > hi else "-%d" % (lo - span))
        print(f"{t0:<30}{span:>7}{f'{lo}-{hi}':>13}{flag:>4}")
    else:
        print(f"{t0:<30}{span:>7}{'-':>13}")

if refs_start is None:
    sys.exit("\ncould not find the References section in main.toc")

main_pages = refs_start - 1
print()
print(f"MAIN REPORT (Part 3, sections 1-6, excluding References and Appendix): "
      f"{main_pages} pages  (printed 1..{main_pages})")
print(f"  Appendix 4 limit: {MIN_PAGES} (min) to {MAX_PAGES} (max)")
if main_pages < MIN_PAGES:
    print(f"  -> {MIN_PAGES - main_pages} pages SHORT of the minimum")
    sys.exit(1)
elif main_pages > MAX_PAGES:
    print(f"  -> {main_pages - MAX_PAGES} pages OVER the maximum "
          f"(not allowed under any circumstances)")
    sys.exit(1)
else:
    print(f"  -> within limits; {MAX_PAGES - main_pages} pages of headroom")
