from pathlib import Path
import re,subprocess
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'Report/DesignPolished_2026-11-01'
for p in [p for f in ['3_Main_Report','Appendices'] for p in (B/f).rglob('*.tex')]:
 t=p.read_text(encoding='utf-8')
 t=re.sub(r'\n\\designgap\{([^}]+)\}(~\\cite(?:\[[^\]]*\])?\{[^}]+\})',r'\2\n\\designgap{\1}',t)
 p.write_text(t,encoding='utf-8')
args=['pdflatex','-disable-installer','-interaction=batchmode','-halt-on-error','-file-line-error','-no-shell-escape','design_preview.tex']
for i in range(3):
 r=subprocess.run(args,cwd=B,capture_output=True,text=True)
 print('LaTeX pass',i+1,'exit',r.returncode,flush=True)
 if r.returncode:print(r.stdout,r.stderr);raise SystemExit(r.returncode)
 if i==0:
  r=subprocess.run(['bibtex','design_preview'],cwd=B,capture_output=True,text=True)
  print('BibTeX exit',r.returncode,flush=True)
  if r.returncode:print(r.stdout,r.stderr);raise SystemExit(r.returncode)
p=ROOT/'tmp/report_context_review/render_polished_revision.py'
p.with_name('render_polished_verified.py').write_text(p.read_text().replace('polished_render','polished_verified_render'),encoding='utf-8')
