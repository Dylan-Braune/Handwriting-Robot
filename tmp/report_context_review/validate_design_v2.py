from pathlib import Path
import ast
from collections import Counter
import hashlib
import json
import re
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'Report/DesignPolished_2026-11-01'
texfiles=[BASE/'3_Main_Report/3_Design_and_Implementation.tex',*sorted((BASE/'3_Main_Report/design').glob('*.tex')),BASE/'Appendices/design_implementation_details.tex']
text='\n'.join(p.read_text(encoding='utf-8') for p in texfiles)
labels=re.findall(r'\\label\{([^}]+)\}',text)
refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',text)
keys=set(re.findall(r'@\w+\{\s*([^,]+),',(BASE/'references.bib').read_text(encoding='utf-8')))
cites=[k.strip() for group in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}',text) for k in group.split(',')]
assets=re.findall(r'\\designasset\{([^}]+)\}',text)
log=(BASE/'design_preview.log').read_text(errors='replace')
pdf=PdfReader(BASE/'design_preview.pdf')
pages=[p.extract_text() or '' for p in pdf.pages]
refs_page=next((i+1 for i,p in enumerate(pages) if re.search(r'(?:^|\n)4\.\s*References',p)),None)
app_page=next((i+1 for i,p in enumerate(pages) if 'A. Supporting implementation calculations' in p),None)
if refs_page is None:
    refs_page=next((i+1 for i,p in enumerate(pages) if re.search(r'(?:^|\n)References',p)),None)
source=ROOT/'Software/CleanPipeline3'
files=['segment.py','recognise.py','train.py','trainlibrary.py','style.py','synthesise.py','reproduce.py','curate.py','retrace.py','extractiam.py','strokenorm_prepare.py','strokenorm_train.py','paths.py','timing.py','web/server.py','web/camera.py','web/render.py']
inventory={}
for name in files:
    p=source/name
    tree=ast.parse(p.read_text(encoding='utf-8-sig'))
    inventory[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'top_level_definitions':[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]}
(BASE/'CODE_COVERAGE_INVENTORY.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
baseline=ROOT/'Report/PrismSnapshot_2026-10-31/epr400.cls'
out={
    'revision':'Hardware diagram and pin-map update, 1 November 2026',
    'pages':len(pages),'references_start_pdf_page':refs_page,'appendix_start_pdf_page':app_page,
    'design_pages':refs_page-1 if refs_page else None,
    'appendix_pages':len(pages)-app_page+1 if app_page else None,
    'algorithm_blocks':text.count('\\begin{designalgorithm}'),
    'equations':text.count('\\begin{equation}'),
    'figures':text.count('\\begin{figure}'),'tables':text.count('\\begin{table}'),
    'duplicate_labels':[k for k,v in Counter(labels).items() if v>1],
    'missing_labels':sorted(set(refs)-set(labels)),
    'unreferenced_labels':sorted(set(labels)-set(refs)),
    'missing_bibliography_keys':sorted(set(cites)-keys),
    'missing_figure_assets':[a for a in assets if not (BASE/'Figures'/a).exists()],
    'completion_notes':text.count('\\designgap{'),
    'unresolved_pdf_markers':any('??' in p for p in pages),
    'overfull_warnings':re.findall(r'Overfull[^\n]+',log),
    'undefined_warnings':re.findall(r'[^\n]*(?:undefined|multiply defined)[^\n]*',log),
    'class_unchanged':baseline.exists() and baseline.read_bytes()==(BASE/'epr400.cls').read_bytes(),
    'visual_review':'pending final rendered-page inspection',
    'scope':'Report coverage and build checks; no training, learned-model inference or hardware tests.'
}
(BASE/'VALIDATION.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
assert not out['duplicate_labels'] and not out['missing_labels']
assert not out['missing_bibliography_keys'] and not out['unresolved_pdf_markers']
assert not out['overfull_warnings'] and not out['undefined_warnings']
assert out['class_unchanged']
