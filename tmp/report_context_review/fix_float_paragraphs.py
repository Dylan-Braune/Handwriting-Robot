from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'Report/DesignPolished_2026-11-01'
for p in [p for f in ['3_Main_Report','Appendices'] for p in (B/f).rglob('*.tex')]:
 t=p.read_text(encoding='utf-8')
 t=re.sub(r'(?<!\n)\n(\\begin\{(?:figure|table|designalgorithm)\})',r'\n\n\1',t)
 t=re.sub(r'(\\end\{(?:figure|table|designalgorithm)\})\n(?!\n)',r'\1\n\n',t)
 t=t.replace(r'\cite[training records and subsystem manifests]{brauneevidence2026}',r'\cite[project records; implementation]{brauneevidence2026,braunepipeline2026}')
 # These formerly adjacent prose blocks were only supported by their caption.
 blocks=re.split(r'(\n\s*\n)',t)
 for i,s in enumerate(blocks):
  if not re.match(r'[A-Za-z]',s.lstrip()) or '\\cite' in s:continue
  if '\\begin{' in s or '\\end{' in s:continue
  if len(s.split())<22:continue
  if p.name.startswith('3_3'):cite=r'~\cite[hardware clarifications]{brauneevidence2026}'
  elif 'subsystem_evidence' in p.name:cite=r'~\cite[subsystem manifests]{brauneevidence2026}'
  elif p.name.startswith('3_5') and ('checkpoint' in s or 'sweep' in s):cite=r'~\cite[training records]{brauneevidence2026}'
  else:
   mod='segment.py' if p.name.startswith('3_4') else ('recognise.py; train.py' if p.name.startswith('3_5') else ('style.py; synthesise.py' if p.name.startswith('3_6') else 'project implementation'))
   cite=r'~\cite['+mod+r']{braunepipeline2026}'
  blocks[i]=s.rstrip()+cite+'\n'
 t=''.join(blocks)
 if p.name=='3_2_system.tex': t=t.replace('[web/server.py; paths.py]','[web/server.py]')
 if p.name=='3_3_hardware.tex':t=t.replace(r'\cite[p.~9]{a4988ds}.'+'\n'+r'\cite[p.~9]{a4988ds}',r'\cite[p.~9]{a4988ds}.')
 if p.name=='3_6_reproduction.tex':t=t.replace('Table~\\ref{tab:des-styleparameters} relates the main quantities to their effect on reproduction.\n','').replace('Table~\\ref{tab:des-styleparameters} relates the main quantities to their effect on reproduction.','')
 if p.name=='3_5_classification.tex':t=t.replace('Table~\\ref{tab:des-implementations} distinguishes the operations supplied by each route. The same exported parameters can be evaluated by the numerical inference module independently of the training engine.\n','')
 p.write_text(t,encoding='utf-8')
