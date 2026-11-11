from pathlib import Path
import zipfile
p=Path('output/Handwriting_Robot_Supporting_Evidence_2026-11-02.zip');t=p.with_suffix('.new.zip')
with zipfile.ZipFile(p) as src,zipfile.ZipFile(t,'w',zipfile.ZIP_DEFLATED) as dst:
 for item in src.infolist():
  name=item.filename.replace('DesignEvidence/','Evidence/',1) if item.filename.startswith('DesignEvidence/') else item.filename
  dst.writestr(name,src.read(item))
 dst.write('tmp/final_report_review/FINAL_VALIDATION.json','ReportValidation/FINAL_VALIDATION.json')
 dst.write('tmp/final_report_review/source-cleanup.json','ReportValidation/source-cleanup.json')
t.replace(p)
print('Evidence package updated')
