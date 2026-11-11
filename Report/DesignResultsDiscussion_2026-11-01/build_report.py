"""Build the integrated Design/Results/Discussion review; requires MiKTeX."""
from pathlib import Path
import subprocess,sys,re,json
B=Path(__file__).resolve().parent
for cmd in [[sys.executable,'make_results_figures.py'],['pdflatex','-disable-installer','-interaction=batchmode','-halt-on-error','-file-line-error','-no-shell-escape','report_preview.tex'],['bibtex','report_preview'],*([['pdflatex','-disable-installer','-interaction=batchmode','-halt-on-error','-file-line-error','-no-shell-escape','report_preview.tex']]*2)]:
 r=subprocess.run(cmd,cwd=B,capture_output=True,text=True)
 print(' '.join(cmd[:2]),'exit',r.returncode,flush=True)
 if r.returncode:print(r.stdout,r.stderr);raise SystemExit(r.returncode)
log=(B/'report_preview.log').read_text(errors='replace')
bad=re.findall(r'[^\n]*(?:Overfull|undefined|multiply defined|ignored error|Float too large)[^\n]*',log)
if bad:print('\n'.join(bad));raise SystemExit(1)
print('Clean LaTeX build, including citations and table layout.')
