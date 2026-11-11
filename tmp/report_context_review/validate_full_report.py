from pathlib import Path
from collections import Counter
import re,json,hashlib
from pypdf import PdfReader
R=Path(__file__).resolve().parents[2];B=R/'Report/IntegratedReport_2026-11-01'
seen=[]
def visit(p):
 if p in seen:return
 seen.append(p)
 for x in re.findall(r'\\input\{([^}]+)\}',p.read_text(encoding='utf-8')):
  q=B/(x if x.endswith('.tex') else x+'.tex');assert q.exists(),q;visit(q)
visit(B/'report_preview.tex')
t='\n'.join(p.read_text(encoding='utf-8') for p in seen)
labels=re.findall(r'\\label\{([^}]+)\}',t)
assert not [k for k,v in Counter(labels).items() if v>1]
assert not set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',t))-set(labels)
keys=set(re.findall(r'@\w+\{\s*([^,]+)',(B/'references.bib').read_text(encoding='utf-8')))
cites={k for s in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}',t) for k in s.split(',')}
assert cites==keys,(cites-keys,keys-cites)
opts=re.findall(r'\\cite\[([^\]]+)\]',t)
assert all(re.fullmatch(r'pp?\.~[0-9]+(?:(?:--|, )\d+)*',o) for o in opts),[o for o in opts if not re.fullmatch(r'pp?\.~[0-9]+(?:(?:--|, )\d+)*',o)]
assert not re.search(r'\\(?:designgap|designsourceflag|designasset|mustdo|new)\b|To complete:|Figure requested:|page to verify|Intended contribution',t,re.I)
assert not re.search(r'(?<!\\)%[^\n]*\S',t)
for im in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',t):
 assert (B/im).exists(),im
for name in ['hardware','software']:
 assert (B/f'Evidence/user_{name}_diagram_input.png').read_bytes()==(R/f'Report/DesignResultsDiscussion_2026-11-01/Evidence/user_{name}_diagram_input.png').read_bytes()
assert (B/'epr400.cls').read_bytes()==(R/'Report/PrismSnapshot_2026-10-31/epr400.cls').read_bytes()
log=(B/'report_preview.log').read_text(errors='replace')
bad=re.findall(r'[^\n]*(?:Overfull|undefined|multiply defined|ignored error|Float too large)[^\n]*',log);assert not bad,bad
pdf=PdfReader(B/'report_preview.pdf');pages=[p.extract_text() or '' for p in pdf.pages]
assert not any('??' in p for p in pages)
starts={}
for i,p in enumerate(pages):
 for key,title in [('literature','1. Literature study\n'),('approach','2. Approach\n'),('design','3. Design and implementation\n'),('results','4. Results\n'),('discussion','5. Discussion\n'),('references','6. References\n')]:
  if title in p:starts[key]=i+1
order=list(starts);counts={key:starts[order[i+1]]-value for i,(key,value) in enumerate(starts.items()) if i+1<len(starts)}
assert counts['literature']<=8 and counts['approach']<=2,counts
assert len(re.findall(r'\\reportqt\{qt:r\d\}',t))==6
assert t.count(r'\resstep{')==42
for h in ['Interpretation of results','Critical evaluation','Unsolved problems','Strong points of the design','Expected failure conditions','Ergonomics','Health and safety','Environmental impact','Social and legal impact','Ethics clearance']:
 assert '\\subsubsection{'+h+'}' in t
eqs=re.findall(r'\\label\{(eq:[^}]+)\}',t);audit=(B/'EQUATION_SOURCE_AUDIT.md').read_text()
assert len(eqs)==42 and all('`'+eq+'`' in audit for eq in eqs)
for row in json.loads((B/'ResultsEvidence/checkpoints/manifest.json').read_text()):
 assert hashlib.sha256((B/'ResultsEvidence/checkpoints'/row['file']).read_bytes()).hexdigest()==row['sha256']
ps=json.loads((B/'ResultsEvidence/physical_style/results.json').read_text());assert ps['counts']['original']['__total__']==156 and ps['counts']['physical']['__total__']==184
for typ in ['inputs','code']:
 for path,digest in ps[typ].items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==digest
out=dict(pdf_pages=len(pages),section_starts=starts,section_pages=counts,main_pages=starts['references']-1,full_report_scope='Sections 1-5, references and appendices; front matter/proposal/Conclusion not included',overall_70_page_cap_met=starts['references']-1<=70,qualification_tests=6,qualification_subheadings=42,required_discussion_subheadings=10,numbered_equations=42,cited_sources=len(cites),journal_sources_in_literature=len(set(re.findall(r'@article\{([^,]+)',(B/'references.bib').read_text())) & {k for s in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}',(B/'3_Main_Report/1_Literature_Study.tex').read_text()) for k in s.split(',')}),citation_options='source and page only; source-only where unverified',inline_editorial_notes=0,missing_figure_placeholders=0,missing_references=[],build_warnings=bad,original_hardware_and_software_diagrams_unchanged=True,original_class_unchanged=True,physical_style_counts=dict(original=156,physical=184,writers=11),checkpoints_archived=4,visual_review='pending latest rendered-page inspection')
(B/'INTEGRATED_REVIEW.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
