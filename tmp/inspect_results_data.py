from pathlib import Path
import json,collections
r=Path('Report/ReportWorkingArchive_2026-11-01_VisualTests/ResultsEvidence')
for n in ['measurements/outputs/blocks/lines.json','measurements/outputs/holdout_namer/lines.json','measurements/outputs/timing/timing_blocks.json','measurements/outputs/timing/sessions.json','measurements/outputs/timing/sentence_match.json','REVIEWED_METRICS.json']:
 p=r/n;x=json.loads(p.read_text());print('\n',n,type(x).__name__);print(str(x[:1] if isinstance(x,list) else {k:str(v)[:170] for k,v in list(x.items())[:8]})[:1300])
x=json.loads((r/'measurements/outputs/blocks/lines.json').read_text());print('pages',collections.Counter(z['page'] for z in x));print('totals',sum(z['refChars'] for z in x),sum(z['dist'] for z in x),sum(z['writer']==z['named'] for z in x))
