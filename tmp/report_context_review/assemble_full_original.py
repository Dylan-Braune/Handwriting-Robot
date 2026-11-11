from pathlib import Path
import shutil,subprocess,sys
b=Path('Report/IntegratedReport_2026-11-01').resolve();o=b.parent/'PrismSnapshot_2026-10-31'
for folder in ['1_Preamble','2_Project_Definition','4_Appendices']:
 shutil.copytree(o/folder,b/folder,dirs_exist_ok=True)
shutil.copy2(o/'3_Main_Report/6_Conclusion.tex',b/'3_Main_Report/6_Conclusion.tex')
s=(b/'report_preview.tex').read_text(encoding='utf-8')
front=r'''\maketitlepage
\pagestyle{plain}
\pagenumbering{roman}
\eprsecwithout{Part 1. Preamble}
\vspace*{0.5cm}
\input{1_Preamble/1_Preamble}
\newpage
\tableofcontents
\newpage
\input{1_Preamble/2_Abbreviations}
\newpage
\input{2_Project_Definition/project_definition}
\eprsec{Part 3. Main Report}
\newpage
\pagenumbering{arabic}
\setcounter{page}{1}
'''
s=s.replace('\\begin{document}','\\begin{document}\n'+front)
s=s.replace('\\renewcommand{\\eprthepartname}{References}',r'''\renewcommand{\eprthepartname}{Conclusion}
\section{Conclusion}
\input{3_Main_Report/6_Conclusion}
\clearpage
\renewcommand{\eprthepartname}{References}''')
(b/'full_report.tex').write_text(s,encoding='utf-8')
for cmd in [['pdflatex','-disable-installer','-interaction=batchmode','-halt-on-error','-file-line-error','-no-shell-escape','full_report.tex'],['bibtex','full_report'],*([['pdflatex','-disable-installer','-interaction=batchmode','-halt-on-error','-file-line-error','-no-shell-escape','full_report.tex']]*2)]:
 r=subprocess.run(cmd,cwd=b,capture_output=True,text=True);print(cmd[0],r.returncode,flush=True)
 if r.returncode:print(r.stdout[-5000:],r.stderr[-1000:]);sys.exit(r.returncode)
shutil.copy2(b/'full_report.pdf',Path('output/pdf/Handwriting_Robot_Full_Report_2026-11-01.pdf'))
