from pathlib import Path
import re,json
P=Path('Report/SubmissionRevision_2026-11-02')
a=(P/'.build/full_report.aux').read_text(errors='replace')
print('\n'.join(x for x in a.splitlines() if 'contentsline {section}' in x))
print('Bib entries',len(re.findall(r'\\bibitem', (P/'.build/full_report.bbl').read_text())))
