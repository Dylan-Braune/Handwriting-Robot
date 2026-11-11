import json
from pathlib import Path
E=Path('Report/ReportWorkingArchive_2026-11-01_VisualTests/ResultsEvidence')
for f in ['REVIEWED_METRICS.json','measurements/outputs/blocks/lines.json','measurements/outputs/holdout_namer/lines.json']:
 x=json.loads((E/f).read_text());print(f,type(x).__name__)
 if isinstance(x,list):print(len(x),x[0])
 else:
  print(list(x));r=x.get('reproduction_timing');print(str(r)[:2000])
print(list(Path('Software/CleanPipeline3/ReportResults/FullReproducedPages').glob('*'))[:4])
