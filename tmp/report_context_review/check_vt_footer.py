from pathlib import Path
from pypdf import PdfReader
for n in ['T25_seam_cost_field','T32_feature_stack','T38_gradient_check','T45_variant_pool_medoid']:
 p=PdfReader(Path('Report/DesignVisualTests_2026-11-01/figures')/(n+'.pdf')).pages[0];a=[]
 def v(t,cm,tm,font,size):
  if t.strip():a.append((t[:65].replace('\n',' '),round(tm[5],1),round(cm[5],1),round(size,1)))
 p.extract_text(visitor_text=v);print(n,list(p.mediabox),a[-7:])
