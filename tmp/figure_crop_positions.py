from pathlib import Path
from pypdf import PdfReader
for f in ['T25_seam_cost_field','T32_feature_stack']:
 p=PdfReader('Report/SubmissionRevision_2026-11-02/Figures/VisualTests/'+f+'.pdf').pages[0];print(f,list(p.mediabox))
 rows=[]
 def v(t,cm,tm,fo,size):
  if t.strip():rows.append((round(cm[5]+tm[5],1),t[:95].replace('\n',' ')))
 p.extract_text(visitor_text=v)
 print('\n'.join(str(z) for z in rows if not z[1].startswith('/uni'))[:5000])
