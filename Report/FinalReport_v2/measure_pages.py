"""Measure the main-report page budget from the built PDF.

Appendix 4: the main report (Part 3) must be 50-70 pages, and the count starts
at page 1 of Part 3, excluding the References section and the Appendix.
This finds where each section starts in the PDF so the budget can be measured
rather than guessed.
"""
import re
import sys
from pathlib import Path

from pypdf import PdfReader

PDF = Path(__file__).resolve().parent / "main.pdf"
TOC = Path(__file__).resolve().parent / "main.toc"

r = PdfReader(str(PDF))
print(f"total PDF pages: {len(r.pages)}\n")

# Map section titles -> printed page number from the .toc file
toc_text = TOC.read_text(encoding="utf-8", errors="replace")

# epr400.cls writes \contentsline{section}{...}{page}{...}
entries = re.findall(r"\\contentsline\s*\{(\w+)\}\{[^}]*?([^{}]+)\}\{(\d+)\}", toc_text)
print("Table-of-contents entries (printed page numbers, roman = preamble):")
for kind, title, page in entries:
    title_clean = re.sub(r"\\[a-zA-Z]+\s*", "", title).strip()
    print(f"  [{kind:<12}] p.{page:<5} {title_clean[:66]}")

# Find the PDF page index where the body starts and where references start
def find_page(needle, start=0):
    for i in range(start, len(r.pages)):
        t = r.pages[i].extract_text() or ""
        if needle.lower() in t.lower():
            return i + 1
    return None

p_part3 = find_page("Part 3. Main Report")
p_refs = None
for i in range(len(r.pages) - 1, 0, -1):
    t = r.pages[i].extract_text() or ""
    if re.search(r"^\s*7\.\s*References", t, re.M):
        p_refs = i + 1
p_appendix = None
for i in range(len(r.pages) - 1, 0, -1):
    t = r.pages[i].extract_text() or ""
    if re.match(r"^\s*Appendix\s*$", t, re.M):
        p_appendix = i + 1

print()
print(f"PDF page where 'Part 3. Main Report' appears : {p_part3}")
print(f"PDF page where '7. References' appears      : {p_refs}")
print(f"PDF page where 'Appendix' appears            : {p_appendix}")

# printed page 1 of Part 3 == the page after the Part 3 divider
if p_part3 and p_refs:
    body_start = p_part3 + 1
    main_pages = p_refs - body_start
    print()
    print(f"MAIN REPORT (Part 3, excluding References and Appendix): "
          f"{main_pages} pages  (physical {body_start}..{p_refs - 1})")
    print("  APPENDIX4 limit: 50 (min) to 70 (max)")
    if main_pages < 50:
        print(f"  -> {50 - main_pages} pages SHORT of the minimum")
    elif main_pages > 70:
        print(f"  -> {main_pages - 70} pages OVER the maximum")
    else:
        print(f"  -> within limits; {70 - main_pages} pages of headroom")
