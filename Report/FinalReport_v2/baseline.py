"""Baseline section map for Report/FinalReport_v2 (built main.pdf, 123 pages).

Appendix 4: the main report (Part 3) = 50-70 pages, starting at page 1 of
Part 3, excluding References and Appendix. This records the measured baseline
so improvements can be tracked against it.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
from pypdf import PdfReader  # noqa: E402

r = PdfReader(str(HERE / "main.pdf"))
marks = {}
for i, pg in enumerate(r.pages, 1):
    head = (pg.extract_text() or "")[:300]
    for pat in ("1. Literature study", "2. Approach", "3. Design and implementation",
                "4. Results", "5. Discussion", "6. Conclusion", "7. References"):
        if pat in head and pat not in marks:
            marks[pat] = i

DIVIDER = 20          # PDF page holding the "Part 3. Main Report" divider
body = {k: v - DIVIDER for k, v in marks.items()}   # printed page numbers

order = ["1. Literature study", "2. Approach", "3. Design and implementation",
         "4. Results", "5. Discussion", "6. Conclusion", "7. References"]
print(f"PDF total pages: {len(r.pages)}")
print(f"Part 3 divider at PDF page {DIVIDER} (= printed page 0)\n")
print(f"{'section':<32}{'starts (printed)':>18}{'pages':>8}")
print("-" * 60)
for i, k in enumerate(order):
    start = body[k]
    end = (body[order[i + 1]] - 1) if i + 1 < len(order) else 70
    print(f"{k:<32}{start:>18}{end - start + 1:>8}")

main_pages = body["7. References"] - body["1. Literature study"]
print("-" * 60)
print(f"{'MAIN REPORT (Lit study .. Conclusion)':<32}{body['1. Literature study']:>18}"
      f"{main_pages:>8}")
print()
print("APPENDIX4 limits: total 50-70; Literature study 4-8; Approach 1-2;")
print("                  Design 30-40; Results 10-15; Discussion 6-8; Conclusion 1-4")
print(f"baseline main report = {main_pages} pages -> {70 - main_pages} pages of headroom")

json.dump(dict(total_pdf_pages=len(r.pages), divider_pdf_page=DIVIDER,
               sections={k: body[k] for k in order},
               main_report_pages=main_pages),
          open(HERE / "baseline_pages.json", "w", encoding="utf-8"), indent=2)
print("\nwrote baseline_pages.json")
