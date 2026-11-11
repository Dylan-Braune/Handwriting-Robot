from pathlib import Path
import subprocess,sys,re
b=Path('Report/IntegratedReport_2026-11-01').resolve()
for cmd in [['pdflatex','-disable-installer','-interaction=batchmode','-halt-on-error','-file-line-error','-no-shell-escape','full_report.tex'],['bibtex','full_report'],*([['pdflatex','-disable-installer','-interaction=batchmode','-halt-on-error','-file-line-error','-no-shell-escape','full_report.tex']]*2)]:
 r=subprocess.run(cmd,cwd=b,capture_output=True,text=True);print(cmd[0],r.returncode,flush=True)
 if r.returncode:print(r.stdout[-6000:]);sys.exit(r.returncode)
log=(b/'full_report.log').read_text(errors='replace');bad=re.findall(r'[^\n]*(?:Overfull|undefined|multiply defined|Float too large)[^\n]*',log)
print('\n'.join(bad) if bad else 'Clean build and references.');sys.exit(bool(bad))
