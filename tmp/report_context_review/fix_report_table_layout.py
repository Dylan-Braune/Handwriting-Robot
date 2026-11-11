from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'Report/DesignResultsDiscussion_2026-11-01'
p=B/'3_Main_Report/4_Results.tex';s=p.read_text().replace('Section~\\ref{sec:res-conditions}','Sec.~\\ref{sec:res-conditions}')
a=s.index('\\begingroup\\small');z=s.index('\\end{longtable}\\endgroup',a)+len('\\end{longtable}\\endgroup')
block=s[a:z];rows=[ln for ln in block.splitlines() if re.match(r'\\textbf\{(?:R[1-6]|FC[12])\.',ln)]
out=[]
for i,part in enumerate([rows[:3],rows[3:]]):
 out.append('\\begin{table}[!htbp]'+('\\ContinuedFloat' if i else '')+'\\centering\\small\n\\setlength{\\tabcolsep}{5pt}\n\\begin{tabular}{p{0.32\\linewidth}p{0.48\\linewidth}p{0.12\\linewidth}}\\toprule\n\\textbf{Intended outcome} & \\textbf{Actual outcome} & \\textbf{Location}\\\\\\midrule\n'+'\n'.join(part)+'\n\\bottomrule\\end{tabular}\n\\caption{Summary of results achieved against the Project Proposal'+(' (continued)' if i else '')+'.}'+('' if i else '\\label{tab:res-summary}')+'\n\\end{table}\n')
s=s[:a]+'\n'.join(out)+'\\FloatBarrier\n'+s[z:];p.write_text(s,encoding='utf-8')
p=B/'3_Main_Report/design/3_3_hardware.tex';s=p.read_text();a=s.index('\\begingroup\n\\small\n\\setlength{\\tabcolsep}{3pt}');z=s.index('\\endgroup',a)+len('\\endgroup');block=s[a:z]
rows=[ln for ln in block.splitlines() if ' & ' in ln and not '\\textbf{' in ln]
split=next(i for i,ln in enumerate(rows) if ln.startswith('7 / 64'))
out=[]
for i,part in enumerate([rows[:split],rows[split:]]):
 out.append('\\begin{table}[!htbp]'+('\\ContinuedFloat' if i else '')+'\\centering\\small\n\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.03}\n\\begin{tabular}{p{0.18\\linewidth}p{0.22\\linewidth}p{0.23\\linewidth}p{0.29\\linewidth}}\\toprule\n\\textbf{Header pin / GPIO line} & \\textbf{Connection} & \\textbf{Direction and nominal level} & \\textbf{Notes / verification}\\\\\\midrule\n'+'\n'.join(part)+'\n\\bottomrule\\end{tabular}\n\\caption{Controller pin, signal-level and supply map'+(' (continued)' if i else ' for the installed system')+'.}'+('' if i else '\\label{tab:des-pin-map}')+'\n\\end{table}\n')
s=s[:a]+'\n'.join(out)+'\\FloatBarrier\n'+s[z:];p.write_text(s,encoding='utf-8')
print('Converted two long tables into continued table floats; all rows preserved.')
