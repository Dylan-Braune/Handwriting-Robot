from pathlib import Path
import shutil
r=Path.cwd();src=r/'Report/IntegratedReport_2026-11-01';dst=r/'tmp/report_context_review/network_section_compilecheck';assert not dst.exists();shutil.copytree(src,dst)
shutil.copy2(r/'output/Neural_Network_Section_Revised_2026-11-02.tex',dst/'3_Main_Report/design/3_5_classification.tex')
print(dst)
