from pathlib import Path
p=Path('Report/SubmissionRevision_2026-11-02/3_Main_Report/design/3_3_hardware.tex');s=p.read_text()
s=s.replace('The supplied circuit image does not show a flyback diode.','Figure~\\ref{fig:des-pencircuit} shows the low-side transistor connection. The supplied circuit image does not show a flyback diode.')
# Introduce switch drawing where the description starts.
needle='\\includegraphics[width=0.62\\linewidth]{Figures/endstop-switch-pulldown-circuit.png}'
if needle not in s:
 import re
 m=re.search(r'\\begin\{figure\}\[H\].*?\\label\{fig:endstop-switch-pulldown\}',s,re.S)
 # use the nearest begin rather than matching previous figures
pos=s.index('\\label{fig:endstop-switch-pulldown}');begin=s.rfind('\\begin{figure}',0,pos)
s=s[:begin]+r'Figure~\ref{fig:des-switch} shows the switch contact and pull-down that define the GPIO input state.'+'\n\n'+s[begin:]
p.write_text(s)
