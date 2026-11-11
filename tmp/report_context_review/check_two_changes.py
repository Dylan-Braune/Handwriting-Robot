from pathlib import Path
from pypdf import PdfReader
import pypdfium2 as pdfium
b=Path('Report/IntegratedReport_2026-11-01'); r=PdfReader(b/'report_preview.pdf');d=pdfium.PdfDocument(str(b/'report_preview.pdf'))
for i,p in enumerate(r.pages):
 t=p.extract_text() or ''
 if 'Architecture of the text reader and writer classifier.' in t or 'Controller pin, signal-level and supply map' in t:
  out=Path('tmp/report_context_review')/f'changed-page-{i+1}.png';d[i].render(scale=1.2).to_pil().save(out);print(out,flush=True)
