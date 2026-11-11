from pathlib import Path
import shutil,json,hashlib
r=Path.cwd();a=r/'Report/ReportWorkingArchive_2026-11-01_VisualTests/VisualTestEvidence';b=r/'Report/IntegratedReport_2026-11-01'
shutil.copy2(r/'tmp/report_context_review/plot_vt_time.py',a/'scripts/plot_time_budget_report.py')
records=[]
for p in sorted((b/'Figures/VisualTests').glob('*.pdf')):
 q=r/'Report/DesignVisualTests_2026-11-01/figures'/p.name
 records.append({'figure':p.name,'original_sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'included_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'treatment':'replotted unchanged recorded data' if p.name.startswith('T56') else 'cropped footer in vector PDF copy'})
(a/'ASSET_RECORD.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
