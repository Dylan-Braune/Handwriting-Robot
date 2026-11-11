from pathlib import Path
import shutil,json,hashlib,ast
from docx import Document
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
SNAP=BASE/'snapshot';SNAP.mkdir(exist_ok=True)
manifest=[]
for p in (ROOT/'Software/DemoCode').rglob('*'):
    if p.is_file() and p.suffix in ['.py','.sh','.html','.css','.js']:
        rel=p.relative_to(ROOT/'Software/DemoCode');q=SNAP/rel;q.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,q)
        manifest.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
guide=ROOT/'output/Live_System_Evaluator_Guide_2026-11-11/Handwriting_Robot_Live_System_Evaluation_Guide.docx'
rows=[]
for ti,t in enumerate(Document(guide).tables):
    for ri,row in enumerate(t.rows):
        cells=[c.text for c in row.cells]
        if ri and cells[0].strip():rows.append({'id':len(rows)+1,'table':ti,'control':cells[0],'question':cells[1] if len(cells)>2 else '', 'expected':cells[-1]})
(BASE/'guide_controls.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
(BASE/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Copied source files',len(manifest),'guide controls',len(rows))
for x in rows:print(x['id'],x['control'].encode('ascii','replace').decode())
PY=BASE/'snapshot/paths.py'
# No source edits: paths.Data is overridden only in the test process.
