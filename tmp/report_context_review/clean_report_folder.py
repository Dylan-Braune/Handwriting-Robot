from pathlib import Path
import re,shutil,json,hashlib
root=Path.cwd().resolve();b=(root/'Report/IntegratedReport_2026-11-01').resolve();archive=(root/'Report/ReportWorkingArchive_2026-11-01_VisualTests').resolve()
assert b.is_relative_to(root) and archive.is_relative_to(root/'Report') and archive!=b
assert not archive.exists(),'Archive already exists; do not overwrite'
keep=set()
def add(rel):
 p=(b/rel).resolve();assert p.is_relative_to(b) and p.is_file(),str(p)
 if p in keep:return
 keep.add(p)
 if p.suffix in ['.tex','.cls','.sty']:
  t=p.read_text(encoding='utf-8-sig')
  for x in re.findall(r'\\(?:input|include)\{([^}]+)\}',t):
   add(x if Path(x).suffix else x+'.tex')
  for x in re.findall(r'\\(?:includegraphics|includepdf)(?:\[[^\]]*\])?\{([^}]+)\}',t):
   if '#' not in x:add(x)
  for x in re.findall(r'\\bibliography\{([^}]+)\}',t):
   for y in x.split(','):add(y+'.bib')
add('full_report.tex');add('epr400.cls')
# Preserve the completed PDF outside the source folder before archiving build outputs.
pdf=root/'output/pdf/Handwriting_Robot_Full_Report_Visuals_2026-11-01.pdf';shutil.copy2(b/'full_report.pdf',pdf)
for p in keep:assert p.is_relative_to(b)
archive.mkdir()
moved=[]
for p in sorted(b.rglob('*')):
 if not p.is_file() or p.resolve() in keep:continue
 rel=p.relative_to(b);target=(archive/rel).resolve();assert target.is_relative_to(archive)
 target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(p),str(target));moved.append(str(rel))
# Empty directories only; no recursive deletion.
for p in sorted(b.rglob('*'),key=lambda p:len(p.parts),reverse=True):
 if p.is_dir() and not any(p.iterdir()):p.rmdir()
script='''from pathlib import Path
import subprocess,sys
base=Path(__file__).resolve().parent
out=base/".build"
out.mkdir(exist_ok=True)
commands=[['pdflatex','-disable-installer','-interaction=nonstopmode','-halt-on-error','-recorder','-output-directory=.build','full_report.tex'],['bibtex','.build/full_report'],*([['pdflatex','-disable-installer','-interaction=nonstopmode','-halt-on-error','-recorder','-output-directory=.build','full_report.tex']]*2)]
for cmd in commands:
 result=subprocess.run(cmd,cwd=base,capture_output=True,text=True)
 print(cmd[0],result.returncode,flush=True)
 if result.returncode:print(result.stdout[-6000:],result.stderr);sys.exit(result.returncode)
print('PDF: '+str(out/'full_report.pdf'))
'''
(b/'build_report.py').write_text(script,encoding='utf-8')
(b/'README.md').write_text('''# Compile the full report

Entry point: `full_report.tex`.

With Python and MiKTeX (or TeX Live with the required packages) installed:

    python build_report.py

The PDF and temporary LaTeX files are written to `.build/`, keeping this source folder clean. Alternatively compile `full_report.tex` with pdfLaTeX, BibTeX, pdfLaTeX, pdfLaTeX in your LaTeX editor.

Only referenced source, bibliography, class and graphics are included. Evidence, generation scripts, audits, old builds and unused figures were moved to the sibling `ReportWorkingArchive_2026-11-01_VisualTests` folder. Existing project data are preserved there. Figures are already generated; compiling does not run tests, train models or operate hardware.
''',encoding='utf-8')
manifest={'source_folder':str(b),'archive_folder':str(archive),'kept_compile_dependencies':len(keep),'archived_files':len(moved),'moved':moved,'compiled_pdf':str(pdf)}
(archive/'CLEANUP_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k!='moved'},indent=2))
