from pathlib import Path
import hashlib,json,shutil,zipfile
from pypdf import PdfReader
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
BUILD=BASE/'build'
source=ROOT/'Software/DemoCode'
manifest=json.loads((BASE/'SOURCE_MANIFEST.json').read_text())
differences=[]
for item in manifest:
    p=source/item['path']
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
        differences.append(item['path'])
if differences:raise SystemExit('Source changed since capture: '+repr(differences))
log=(BUILD/'main.log').read_text(encoding='utf-8',errors='replace')
assert 'Output written on main.pdf' in log
assert not any(x in log for x in ['Overfull','undefined','multiply defined','Rerun to get','Fatal error'])
qa=json.loads((BASE/'visual_qa/qa_summary.json').read_text())
pages=len(PdfReader(BUILD/'main.pdf').pages)
assert pages==qa['pages']
report=Path('C:/Users/braun/Downloads/23539195_BrauneDR_FINALREPORT.pdf')
assert hashlib.sha256(report.read_bytes()).hexdigest().upper()=='A7C44C245CD34032CE6B57FF54C35221B8E85D8C318BA27D45863E02DB9E52E0'
out=ROOT/'output/pdf';out.mkdir(exist_ok=True,parents=True)
dest=out/'Handwriting_Robot_Code_Theory_Demo_2026-11-12.pdf'
shutil.copy2(BUILD/'main.pdf',dest)
canonical=ROOT/'Software/CleanPipeline2/testing/results/CodeTheory'
edition=canonical/'DemoEdition_2026-11-12';edition.mkdir(exist_ok=True)
for p in list(BUILD.glob('*.tex'))+list(BUILD.glob('*.bib')):
    shutil.copy2(p,edition/p.name)
for n in ['FUNCTION_COVERAGE.csv','SOURCE_MANIFEST.json']:
    shutil.copy2(BASE/n,edition/n)
readme=f'''# Code theory manual: current demo edition

Updated 12 November 2026. {pages} pages.

The current PDF is ../main.pdf. Its standalone editable LaTeX source is this
directory's main.tex. The parent directory's old main.tex and codetheory_*.tex
are the historical source, preserved rather than overwritten. The previous
373-page PDF is preserved under output/Code_Theory_Demo_2026-11-11/Previous_373_page_manual.pdf.

Build from this directory: pdflatex main.tex; bibtex main; pdflatex main.tex;
pdflatex main.tex. These commands build documentation only.

Scope: static inspection of fourteen Software/DemoCode Python modules and the
startup script, including web/server.py and web/lineworkers.py. The latter was
refreshed during final review after a concurrent external edit. Exact captured
file hashes are in SOURCE_MANIFEST.json. The source ZIP includes the captured
source so the reference can be checked independently of later code edits.
346 top-level functions/classes and 57 class methods are inventoried.

The report-to-code index links all 35 numbered report equations to their
appropriate code-stage explanations. Module chapters follow input-to-output
stages, with relevant report theory, methods, alternatives and library
comparisons beside them. Shared methods are cross-referenced. Training has its
own chapter outside live inference. The function index and coverage CSV locate
current names and source lines.

Checks completed: citation-key resolution; LaTeX cross-reference resolution;
complete callable inventory; all 35 equation homes; current source hash match;
PDF rendering and visual sampling/contact-sheet review. No robot code was
executed, no inference/training or hardware tests were run, and neither the
report nor the implementation was edited by this documentation task.

The source audit flags unsupported ignoreRuling use in ApiGenerate, unsupported
profile/uniformInk keywords to RenderTrajectory, and SegmentLines(timer=...)
in retracing. These are static findings, not repaired or runtime-certified paths.
Published method sources are distinguished from project heuristics, derived
arithmetic and historical measurements. Unverified torque-curve points and
component/current assumptions remain explicitly qualified. Performance
comparisons state semantic and implementation differences without invented
timing or accuracy measurements. This is not a certification of deployed
ODROID files, checkpoint arrays, or every historical report reference.
'''
(edition/'README.md').write_text(readme,encoding='utf-8')
shutil.copy2(dest,canonical/'main.pdf')
(canonical/'CURRENT_EDITION.md').write_text('The current full PDF is **main.pdf**. Its updated source and build instructions are in **DemoEdition_2026-11-12/README.md**. Parent-level main.tex and codetheory files are preserved historical source.\n',encoding='utf-8')
archive=ROOT/'output/Code_Theory_Demo_2026-11-11/Handwriting_Robot_Code_Theory_Demo_Source.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in edition.iterdir():
        if p.is_file():z.write(p,'latex/'+p.name)
    for p in (BASE/'source_snapshot').rglob('*'):
        if p.is_file():z.write(p,'source_snapshot/'+p.relative_to(BASE/'source_snapshot').as_posix())
    z.write(BASE/'visual_qa/qa_summary.json','review/qa_summary.json')
    z.write(edition/'README.md','README.md')
print(json.dumps({'pages':pages,'pdf':str(dest),'canonical_pdf':str(canonical/'main.pdf'),'source_zip':str(archive),'source_files_checked':len(manifest)},indent=2))
