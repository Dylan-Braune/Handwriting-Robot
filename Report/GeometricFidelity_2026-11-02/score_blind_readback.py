"""Score a completed blind human read-back CSV. No neural model is involved."""
from pathlib import Path
import csv,json,sys,unicodedata
B=Path(__file__).resolve().parent

def normalise(s):return ' '.join(unicodedata.normalize('NFC',s).split())
def edit(a,b):
 prev=list(range(len(b)+1))
 for i,ca in enumerate(a,1):
  row=[i]
  for j,cb in enumerate(b,1):row.append(min(row[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
  prev=row
 return prev[-1]
assert edit('kitten','sitting')==3 and edit('','abc')==3 and edit('abc','abc')==0
path=Path(sys.argv[1]) if len(sys.argv)>1 else B/'blind_readback/responses.csv'
key=json.loads((B/'blind_readback/answer_key.json').read_text());answers={r['id']:r['transcription'] for r in csv.DictReader(path.open(encoding='utf-8-sig',newline=''))}
missing=[r['id'] for r in key if not answers.get(r['id'],'').strip()]
if missing:raise SystemExit('Incomplete study. Supply transcriptions for '+', '.join(missing)+'. No accuracy is claimed.')
rows=[];by={}
for r in key:
 target=normalise(r['expected']);got=normalise(answers[r['id']]);d=edit(got,target);rows.append(dict(**r,transcription=got,edits=d,reference_characters=len(target)));z=by.setdefault(r['writer'],{'edits':0,'characters':0,'lines':0});z['edits']+=d;z['characters']+=len(target);z['lines']+=1
for z in by.values():z['character_accuracy_percent']=100*(1-z['edits']/z['characters'])
out={'normalisation':'NFC and whitespace collapsing. Case and punctuation retained. All supplied physical source-text lines must be answered.','writers':by,'lines':rows}
output=path.with_suffix('.scored.json');output.write_text(json.dumps(out,indent=2));print(json.dumps(by,indent=2));print(output)
