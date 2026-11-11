from pathlib import Path
import re
P=Path('Report/SubmissionRevision_2026-11-02')
for p in [*list((P/'3_Main_Report/design').glob('*.tex')),*list((P/'Appendices').glob('*.tex'))]:
 for m in re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}',p.read_text(),re.S):
  labs=re.findall(r'\\label\{([^}]+)\}',m[0]);cap=re.search(r'\\caption\{(.*?)\}\s*(?:\\label|\n)',m[0],re.S)
  print(p.name,','.join(labs), cap[1][:170] if cap else 'NO CAP')
