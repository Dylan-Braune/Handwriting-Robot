from pathlib import Path
import json,hashlib
here=Path(__file__).resolve().parent
s=(here/'validate_design_v2.py').read_text(encoding='utf-8')
s=s.replace("Report/DesignRevision_2026-11-01","Report/DesignPolished_2026-11-01")
s=s.replace("BASE/'Appendices/design_implementation_details.tex'","*sorted((BASE/'Appendices').glob('*.tex'))")
s=s.replace('second Design pass, 1 November 2026','Full Design review and polish, 1 November 2026')
s=s.replace('Report coverage and build checks; no training, learned-model inference or hardware tests.','Offline subsystem checks and selected NumPy model inference; no training or hardware tests. See Evidence manifests.')
exec(compile(s,str(here/'validate_design_v2.py'),'exec'))
# Check source modules against prior inspected versions, rather than merely recording new hashes.
old=json.loads((ROOT/'Report/DesignRevision_2026-11-01/CODE_COVERAGE_INVENTORY.json').read_text(encoding='utf-8'))
assert all(inventory[n]['sha256']==v['sha256'] for n,v in old.items())
out['code_unchanged_from_previous_pass']=True
out['evidence_appendix_start_pdf_page']=next(i+1 for i,p in enumerate(pages) if 'B. Subsystem visual evidence' in p)
(BASE/'VALIDATION.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
