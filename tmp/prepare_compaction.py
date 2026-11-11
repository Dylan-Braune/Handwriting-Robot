from pathlib import Path
from pypdf import PdfReader,PdfWriter
import shutil,json,re
r=Path.cwd();b=r/'Report/SubmissionRevision_2026-11-02';a=r/'tmp/final_report_review/baseline';a.mkdir(exist_ok=True)
for p in b.rglob('*.tex'):
 if '.build' not in p.parts:
  q=a/p.relative_to(b);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
shutil.copy2(b/'.build/full_report.pdf',r/'tmp/final_report_review/baseline.pdf')
for stem,new,lo,hi in [('T25_seam_cost_field','design-seam-two-panels',240,440.337125),('T32_feature_stack','design-owen-first-features',343,455)]:
 p=PdfReader(b/'Figures/VisualTests'/(stem+'.pdf')).pages[0];p.mediabox.lower_left=(0,lo);p.mediabox.upper_right=(float(p.mediabox.right),hi);p.cropbox.lower_left=p.mediabox.lower_left;p.cropbox.upper_right=p.mediabox.upper_right
 w=PdfWriter();w.add_page(p);w.write(b/'Figures'/(new+'.pdf'))
print('Baseline saved. Partial vector figures prepared.')
