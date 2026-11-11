from pathlib import Path
from pypdf import PdfReader,PdfWriter
p=Path('Report/SubmissionRevision_2026-11-02/Figures/design-owen-first-features.pdf');page=PdfReader(p).pages[0]
page.mediabox.lower_left=(page.mediabox.left,350);page.cropbox.lower_left=(page.cropbox.left,350)
w=PdfWriter();w.add_page(page);t=p.with_name('first-features-crop.tmp.pdf');w.write(t);t.replace(p)
