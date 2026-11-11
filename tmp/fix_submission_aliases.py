from pathlib import Path
import re
s=Path('tmp/final_report_review/revision1_text.txt').read_text(encoding='utf-8')
for n in [13,18]:
 for m in re.finditer(r'Figure '+str(n)+r'\.',s):print(n,s[m.start()-60:m.start()+360].replace('\n',' '))
b=Path('Report/SubmissionRevision_2026-11-02');mapping={}
for p in b.rglob('*.tex'):
 s=p.read_text(encoding='utf-8-sig')
 for m in re.finditer(r'(?:\\label\{[^}]+\}){2,}',s):
  labels=re.findall(r'\\label\{([^}]+)\}',m.group())
  if labels[0].startswith('eq:'):
   canonical=next((x for x in labels if x.startswith('eq:des-')),labels[0]);mapping.update({x:canonical for x in labels if x!=canonical})
for p in b.rglob('*.tex'):
 s=p.read_text(encoding='utf-8-sig');t=s
 for a,c in mapping.items():t=t.replace('\\label{'+a+'}','').replace('\\ref{'+a+'}','\\ref{'+c+'}').replace('\\eqref{'+a+'}','\\eqref{'+c+'}')
 if t!=s:p.write_text(t,encoding='utf-8')
print('Equation aliases repaired',mapping)
