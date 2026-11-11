from pathlib import Path
import zipfile,json
r=Path.cwd();dest=(r/'Report/SubmissionRevision_2026-11-02').resolve();assert not dest.exists()
with zipfile.ZipFile('C:/Users/braun/Downloads/IntegratedReport_2026-11-01.zip') as z:
 for i in z.infolist():
  q=(dest/i.filename).resolve();assert q.is_relative_to(dest)
 z.extractall(dest)
a=json.loads((r/'tmp/final_report_review/revision1_annotations.json').read_text())
print('\n'.join(f"p{x['page']}: {x['text']}" for x in a if x['text'] and x['page']>=65))
