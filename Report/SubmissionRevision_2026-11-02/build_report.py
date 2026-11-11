from pathlib import Path
import subprocess,sys,os
base=Path(__file__).resolve().parent
out=base/".build"
out.mkdir(exist_ok=True)
commands=[['pdflatex','-disable-installer','-interaction=nonstopmode','-halt-on-error','-recorder','-output-directory=.build','full_report.tex'],['bibtex','full_report'],*([['pdflatex','-disable-installer','-interaction=nonstopmode','-halt-on-error','-recorder','-output-directory=.build','full_report.tex']]*2)]
env=os.environ.copy()
env['BIBINPUTS']=str(base)+os.pathsep+env.get('BIBINPUTS','')
for cmd in commands:
 result=subprocess.run(cmd,cwd=out if cmd[0]=='bibtex' else base,env=env,capture_output=True,text=True,errors='replace')
 print(cmd[0],result.returncode,flush=True)
 if result.returncode:print(result.stdout[-6000:],result.stderr);sys.exit(result.returncode)
print('PDF: '+str(out/'full_report.pdf'))
