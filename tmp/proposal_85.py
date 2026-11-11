from pathlib import Path
import json
x=json.loads(Path('Report/ReportWorkingArchive_2026-11-01_VisualTests/ResultsEvidence/proposal_form_values.json').read_text())
for k,v in x.items():
 if isinstance(v,str) and any(q in v.lower() for q in ['85','correlation','overlay']):print(k.split('.')[-1],v[:4000])
