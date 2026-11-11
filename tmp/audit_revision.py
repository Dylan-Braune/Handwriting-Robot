from pathlib import Path
import re,json
P=Path('Report/SubmissionRevision_2026-11-02');log=(P/'.build/full_report.log').read_text(errors='replace')
print('\n'.join(x for x in log.splitlines() if any(k in x for k in ['undefined','multiply defined','Overfull','Too many','Error'])))
a=(P/'.build/full_report.aux').read_text()
for line in a.splitlines():
 if any(x in line for x in [r'contentsline {section}',r'contentsline {subsection}']):print(line[:230])
# Main hierarchy and preserved source audit.
for name in ['1_Literature_Study.tex','2_Approach.tex']:
 b=Path('tmp/final_report_review/baseline/3_Main_Report')/name;c=P/'3_Main_Report'/name;print('UNCHANGED',name,b.read_bytes()==c.read_bytes())
# citations without bibliography records
alltex='\n'.join(x.read_text() for x in P.rglob('*.tex') if '.build' not in x.parts);bib=(P/'references.bib').read_text()
used=set(k.strip() for m in re.finditer(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}',alltex) for k in m[1].split(','));known=set(re.findall(r'@\w+\s*\{\s*([^,]+)',bib));print('MISSING BIB',used-known)
for f in (P/'3_Main_Report/design').glob('*.tex'):
 base=Path('tmp/final_report_review/baseline/3_Main_Report/design')/f.name
 if base.exists():
  h=lambda t:re.findall(r'\\(?:subsection|subsubsection|paragraph)\{[^}]+\}',t)
  if h(f.read_text())!=h(base.read_text()): print('HEADINGS CHANGED',f.name,h(base.read_text()),h(f.read_text()))
