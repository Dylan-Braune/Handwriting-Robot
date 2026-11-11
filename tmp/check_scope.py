from pathlib import Path
import re,json
P=Path('Report/SubmissionRevision_2026-11-02')
# Check every figure is introduced before its source occurrence.
for p in [*list((P/'3_Main_Report/design').glob('*.tex')),P/'3_Main_Report/4_Results.tex',*list((P/'Appendices').glob('*.tex'))]:
 s=p.read_text()
 for m in re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}',s,re.S):
  labs=re.findall(r'\\label\{([^}]+)\}',m[0])
  if labs and not any('\\ref{'+lab+'}' in s[:m.start()] for lab in labs):print('Missing figure introduction',p.name,labs)
# Check protected subsection headings and math labels are all retained.
base=Path('tmp/final_report_review/baseline')
for p in (P/'3_Main_Report').rglob('*.tex'):
 b=base/p.relative_to(P)
 if not b.exists():continue
 h=lambda s:re.findall(r'\\(?:subsection|subsubsection|paragraph)\{[^}]+\}',s)
 if h(p.read_text())!=h(b.read_text()):print('Heading diff',p.name)
alltext='\n'.join(f.read_text() for f in P.rglob('*.tex') if '.build' not in f.parts)
for b in (base/'3_Main_Report/design').glob('*.tex'):
 for lab in re.findall(r'\\label\{([^}]+)\}',b.read_text()):
  if '\\label{'+lab+'}' not in alltext:print('Label omitted',lab)
print('Final source checks complete')
