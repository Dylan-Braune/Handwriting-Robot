from pathlib import Path
import re,json
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'Report/DesignPolished_2026-11-01'
# Associate implementation claims with their unpaginated module, not a fabricated page.
modules={'3_1_summary.tex':'project implementation overview','3_2_system.tex':'web/server.py; paths.py',
'3_3_hardware.tex':'reproduce.py','3_4_segmentation.tex':'segment.py',
'3_5_classification.tex':'recognise.py; train.py; trainlibrary.py',
'3_6_reproduction.tex':'style.py; synthesise.py; reproduce.py',
'3_7_integration.tex':'web/server.py; reproduce.py; timing.py',
'design_implementation_details.tex':'train.py; segment.py; style.py',
'design_subsystem_evidence.tex':'Evidence manifests'}
counts={}
for p in [*sorted((B/'3_Main_Report/design').glob('*.tex')),*sorted((B/'Appendices').glob('*.tex'))]:
 t=p.read_text(encoding='utf-8').replace('\x0barnothing',r'\varnothing')
 t=t.replace('targeting ``a,','targeting the symbol $a$,').replace('occupancy of ``a at','occupancy of $a$ at')
 t=t.replace('0.5404$ and $dc_{t-1}=0.4323','0.5403$ and $dc_{t-1}=0.4322')
 t=t.replace('segment.py, ZhangSuenThin','segment.py, Skeletonize').replace(r'motor\textquotesingle s',"motor's")
 t=t.replace(r'p_t(\ell_s\prime)',r'p_t(\ell\prime_s)')
 paragraphs=re.split(r'(\n\s*\n)',t);n=0
 for i,block in enumerate(paragraphs):
  stripped=block.lstrip()
  # Whole prose paragraphs only. Never put a citation into math, tables or pseudocode.
  if not re.match(r'[A-Za-z]',stripped):continue
  if '\\begin{' in block or '\\end{' in block or '\\caption{' in block:continue
  if re.search(r'\\cite(?:\[[^\]]*\])?\{[^}]*braune(?:pipeline|evidence)2026',block):continue
  if stripped.startswith('Calculated example:') or stripped.startswith('Implementation/input basis:'):continue
  # Pure cross-reference directions do not make a new factual claim.
  if len(re.sub(r'\\[A-Za-z]+','',stripped).split())<18:continue
  evidence=('design_subsystem_evidence' in p.name or re.search(r'\b(1186|1031|155|669|517|0\.9935|99\.35|50-epoch|recorded|archive|archived|newly run|photographs of the same|supplied eleven|0\.35\\,A)\b',block))
  if p.name=='3_3_hardware.tex':
   # Hardware specifications are user records; control details additionally use code.
   cite=r'\cite[hardware clarifications; '+modules[p.name]+r']{brauneevidence2026,braunepipeline2026}'
  elif evidence:cite=r'\cite[training records and subsystem manifests]{brauneevidence2026}'
  else:cite=r'\cite['+modules[p.name]+r']{braunepipeline2026}'
  trailing='\n' if block.endswith('\n') else ''
  paragraphs[i]=block.rstrip()+'~'+cite+trailing;n+=1
 t=''.join(paragraphs)
 # Caption source statements make provenance visible beside diagrams, tables and algorithms.
 # Brace-aware to preserve captions containing nested TeX.
 starts=list(re.finditer(r'\\caption\{',t))
 for m in reversed(starts):
  j=m.end();depth=1
  while depth:
   depth+=(t[j]=='{')-(t[j]=='}');j+=1
  cap=t[m.end():j-1]
  if '\\cite' not in cap:
   source=r'~\cite{brauneevidence2026,braunepipeline2026}'
   t=t[:j-1]+source+t[j-1:]
 p.write_text(t,encoding='utf-8');counts[p.name]=n
(B/'PARAGRAPH_CITATION_AUDIT.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
print(counts)
