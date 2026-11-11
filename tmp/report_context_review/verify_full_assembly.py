from pathlib import Path
from pypdf import PdfReader
import pypdfium2 as pdfium
b=Path('Report/IntegratedReport_2026-11-01');o=b.parent/'PrismSnapshot_2026-10-31'
for folder in ['1_Preamble','2_Project_Definition','4_Appendices']:
 for p in (o/folder).rglob('*'):
  if p.is_file():assert p.read_bytes()==(b/p.relative_to(o)).read_bytes()
assert (o/'3_Main_Report/6_Conclusion.tex').read_bytes()==(b/'3_Main_Report/6_Conclusion.tex').read_bytes()
r=PdfReader(b/'full_report.pdf');d=pdfium.PdfDocument(str(b/'full_report.pdf'));print('Pages:',len(r.pages));inds={0,1,2}
for i,p in enumerate(r.pages):
 t=p.extract_text() or ''
 if '??' in t:print('Unresolved marker',i+1)
 if '6. Conclusion' in t or 'Project definition: approved' in t or 'LIST OF ABBREVIATIONS' in t:inds.add(i)
for i in sorted(inds):
 out=Path('tmp/report_context_review')/f'full-assembly-{i+1}.png';d[i].render(scale=1).to_pil().save(out);print(out)
print('Original added sections match byte for byte.')
