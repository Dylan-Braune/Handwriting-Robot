from pathlib import Path
import zipfile
p=Path('output/Handwriting_Robot_Supporting_Evidence_2026-11-02.zip');t=p.with_suffix('.new.zip')
with zipfile.ZipFile(p) as src,zipfile.ZipFile(t,'w',zipfile.ZIP_DEFLATED) as dst:
 for item in src.infolist():
  if item.filename=='ReportValidation/FINAL_VALIDATION.json':continue
  dst.writestr(item,src.read(item))
 dst.write('tmp/final_report_review/FINAL_VALIDATION.json','ReportValidation/FINAL_VALIDATION.json')
t.replace(p)
for f in ['output/Handwriting_Robot_Concise_Source_2026-11-02.zip',str(p)]:
 with zipfile.ZipFile(f) as z:assert z.testzip() is None
print('Both ZIP archives verified')
