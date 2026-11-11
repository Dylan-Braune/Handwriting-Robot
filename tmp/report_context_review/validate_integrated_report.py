from pathlib import Path
from collections import Counter
import json,re,hashlib
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'Report/DesignResultsDiscussion_2026-11-01'
seen=[]
def visit(p):
 if p in seen:return
 seen.append(p)
 for n in re.findall(r'\\input\{([^}]+)\}',p.read_text(encoding='utf-8')):
  q=B/(n if n.endswith('.tex') else n+'.tex');assert q.exists(),q;visit(q)
visit(B/'report_preview.tex');text='\n'.join(p.read_text(encoding='utf-8') for p in seen)
labels=re.findall(r'\\label\{([^}]+)\}',text);refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',text)
assert not [n for n,c in Counter(labels).items() if c>1]
assert not set(refs)-set(labels),set(refs)-set(labels)
bib=(B/'references.bib').read_text();keys=set(re.findall(r'@\w+\{\s*([^,]+)',bib));cites=[k for x in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}',text) for k in x.split(',')]
assert not set(cites)-keys,set(cites)-keys
for img in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text):
 if '#' not in img:assert (B/img).exists(),img
log=(B/'report_preview.log').read_text(errors='replace');bad=re.findall(r'[^\n]*(?:Overfull|undefined|multiply defined|ignored error|Float too large)[^\n]*',log);assert not bad,bad
pdf=PdfReader(B/'report_preview.pdf');pages=[p.extract_text() or '' for p in pdf.pages];assert not any('??' in p for p in pages)
starts={}
for i,t in enumerate(pages):
 for k,title in [('results','4. Results\n'),('discussion','5. Discussion\n'),('references','6. References\n')]:
  if title in t:starts[k]=i+1
r=(B/'3_Main_Report/4_Results.tex').read_text();d=(B/'3_Main_Report/5_Discussion.tex').read_text()
assert len(re.findall(r'\\reportqt\{qt:r\d\}',r))==6
assert r.count('\\resstep{')==42
for h in ['Interpretation of results','Critical evaluation','Unsolved problems','Strong points of the design','Expected failure conditions','Ergonomics','Health and safety','Environmental impact','Social and legal impact','Ethics clearance']:assert '\\subsubsection{'+h+'}' in d,h
ps=json.loads((B/'ResultsEvidence/physical_style/results.json').read_text());assert ps['counts']['original']['__total__']==156 and ps['counts']['physical']['__total__']==184 and len(ps['writers_analysed'])==11 and not ps['exclusions']
for typ in ['inputs','code']:
 for name,digest in ps[typ].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
for row in json.loads((B/'ResultsEvidence/checkpoints/manifest.json').read_text()):assert hashlib.sha256((B/'ResultsEvidence/checkpoints'/row['file']).read_bytes()).hexdigest()==row['sha256']
inv=json.loads((B/'ResultsEvidence/profile_inventory.json').read_text());assert inv['writers']==11
for row in inv['profiles']:assert hashlib.sha256((ROOT/row['source']).read_bytes()).hexdigest()==row['sha256']
assert (B/'Evidence/user_hardware_diagram_input.png').read_bytes()==(ROOT/'Report/DesignPolished_2026-11-01/Evidence/user_hardware_diagram_input.png').read_bytes()
assert (B/'epr400.cls').read_bytes()==(ROOT/'Report/PrismSnapshot_2026-10-31/epr400.cls').read_bytes()
out=dict(pdf_pages=len(pages),design_pages=starts['results']-1,results_pages=starts['discussion']-starts['results'],discussion_pages=starts['references']-starts['discussion'],starts=starts,main_sections_included='3 Design, 4 Results, 5 Discussion only; full report assembly still required',qualification_tests=6,qualification_subheadings=42,required_discussion_subheadings=10,cited_sources=len(set(cites)),missing_references=[],build_warnings=bad,original_hardware_diagram_unchanged=True,class_unchanged=True,physical_style_counts={'original':156,'physical':184,'writers':11},physical_style_correlations=ps['correlations'],checkpoints_archived=4,nonempty_style_profiles=11,intentional_figure_placeholders=[n for n in re.findall(r'\\designasset\{([^}]+)\}',text) if not (B/'Figures'/n).exists()],visual_review='pending latest rendered-page inspection')
(B/'INTEGRATED_REVIEW.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
