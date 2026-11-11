from pathlib import Path
import shutil
b=Path('Report/IntegratedReport_2026-11-01')
p=b/'3_Main_Report/design/3_3_hardware.tex'
s=p.read_text(encoding='utf-8'); start=s.index('\\begin{table}',s.index('Table~\\ref{tab:des-pin-map} maps')); end=s.index('\\FloatBarrier',start)+len('\\FloatBarrier')
block=s[start:end];p.write_text(s[:start]+s[end:],encoding='utf-8')
a=b/'Appendices/design_implementation_details.tex'
a.write_text(a.read_text(encoding='utf-8')+'\n\\subsection{Controller pin and supply planning}\n\n'+block+'\n',encoding='utf-8')
shutil.copy2(Path('C:/Users/braun/Downloads/BiLSTMCRNN.jpeg'),b/'Figures/BiLSTMCRNN.jpeg')
p=b/'3_Main_Report/design/3_5_classification.tex';s=p.read_text(encoding='utf-8');s=s.replace('\\includegraphics[width=0.96\\linewidth]{Figures/design-network-architecture.pdf}','\\includegraphics[width=0.96\\linewidth,trim=0 570 0 490,clip]{Figures/BiLSTMCRNN.jpeg}')
p.write_text(s,encoding='utf-8')
