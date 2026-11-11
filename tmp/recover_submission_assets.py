from pathlib import Path
import re,shutil,json,hashlib
r=Path.cwd();b=r/'Report/SubmissionRevision_2026-11-02';recovered=[]
for p in b.rglob('*.tex'):
 for name in re.findall(r'\\(?:includegraphics|includepdf)(?:\[[^\]]*\])?\{([^}]+)\}',p.read_text(encoding='utf-8-sig')):
  dst=b/name
  if dst.exists():continue
  for root in [r/'Report/PrismSnapshot_2026-10-31',r/'Report/IntegratedReport_2026-11-01',r/'Report/ReportWorkingArchive_2026-11-01_VisualTests']:
   src=root/name
   if src.exists():dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);recovered.append(name);break
  else:raise FileNotFoundError(name)
protected={q:hashlib.sha256((b/q).read_bytes()).hexdigest() for q in ['3_Main_Report/1_Literature_Study.tex','3_Main_Report/2_Approach.tex','3_Main_Report/design/3_3_hardware.tex','1_Preamble/1_Preamble.tex','2_Project_Definition/project_proposal.pdf']}
(r/'tmp/final_report_review/baseline_protected.json').write_text(json.dumps(protected,indent=2));print('Recovered missing assets:',recovered)
