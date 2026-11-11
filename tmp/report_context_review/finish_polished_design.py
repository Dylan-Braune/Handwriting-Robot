from pathlib import Path
import re,json
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'Report/DesignPolished_2026-11-01'
p=B/'3_Main_Report/design/3_6_reproduction.tex'
t=p.read_text(encoding='utf-8')
anchor='\\subsubsection{Character alignment and glyph extraction}'
rationale=r'''\subsubsection{Choice of reproduction representation}
The available writer samples are a few labelled pages photographed after writing. A glyph library can reuse these observations without fitting a separate generative network for each writer. Haines et al. provide a relevant precedent: their standard experiments used approximately four A4 pages per writer~\cite[PDF p.~4, Sec.~4]{haines2016}. This is evidence that a small-page collection can support a glyph-based approach, not a guarantee that four pages cover every character or writing context. The present library likewise depends on the characters and variants actually observed.

The offline data also influenced the choice. Graves's recurrent synthesis model learns from ordered pen positions and end-of-stroke markers~\cite[p.~18]{graves2013}; these are not recorded in this project's photographs. A skeleton supplies geometric paths, but cannot recover their true order, speed or pressure. Using it as measured online training data would introduce an unsupported assumption.

Learned image generators remain a valid alternative, including methods that need very few references from a new writer. For example, One-DM uses one style reference at generation, but its IAM experiment trains with 339 writers and its implementation uses four RTX 3090 GPUs~\cite[p.~9]{dai2024}. A small reference set must therefore be distinguished from the corpus and computation needed to train the generator. Developing and training an additional generator from first principles on the ODROID was outside the chosen computational scope; no training-time comparison was measured. The glyph approach instead retains explicit trajectories that can be scaled, joined and converted directly to machine coordinates. Its trade-off is weaker modelling of connected, context-dependent letterforms, examined at the end of this subsection.

'''
assert anchor in t and 'Choice of reproduction representation' not in t
t=t.replace(anchor,rationale+anchor)
t=t.replace(r'\designgap{Add the actual reason for retaining the present glyph representation and default tracer after these alternatives were considered. Do not infer that reason from the existence of an experimental function.}',
'The retained design therefore prioritises extraction from the available offline pages and explicit control of the plotted geometry. Recovering natural pen dynamics would require new trajectory data, while replacing the library with a learned generator would add a separate training and validation problem. The alternative tracer remains an implementation option; its presence alone does not establish an improvement over the default.')
p.write_text(t,encoding='utf-8')
# Preserve bibliography identities; identify the exact reading copies used.
p=B/'references.bib';bib=p.read_text(encoding='utf-8')
for key,url in {'haines2016':'https://reality.cs.ucl.ac.uk/projects/haines/haines16handwriting.pdf','dai2024':'https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07598.pdf'}.items():
 m=re.search(r'@\w+\{'+key+',',bib);start=bib.index('{',m.start());depth=1;i=start+1
 while depth:
  depth+=(bib[i]=='{')-(bib[i]=='}');i+=1
 entry=bib[m.start():i]
 if 'url=' not in entry: bib=bib[:m.start()]+entry[:-1]+',\n  url={'+url+'}}'+bib[i:]
p.write_text(bib,encoding='utf-8')
p=B/'SOURCE_AUDIT.md';t=p.read_text(encoding='utf-8')
t+='''
## Additional representation-rationale checks in the full polish pass

- `haines2016`: inspected complete author-preprint PDF page 4, section 4,
  including the approximate four-A4-page sample statement. This reading copy is
  a 19-page preprint with inconsistent journal-header metadata; the citation
  explicitly uses **PDF p. 4**, not an inferred page in the final 18-page article.
  The present project does not inherit that paper's manual annotation procedure,
  scan resolution or collection time merely by citing its sample size.
- `graves2013`: additionally inspected complete printed page 18, which identifies
  the data as online pen positions and end-of-stroke markers. It supports the
  missing-data distinction, not an assertion that every generator needs online data.
- `dai2024`: inspected complete printed page 9: 339 training writers, one style
  reference during generation, and training on four RTX 3090 GPUs. These describe
  that study, not measured ODROID performance or a universal data minimum.

Both newly cited keys were already in the user's Literature Review. The corpus
versus per-writer-reference distinction is intentional. Few photographed pages,
no online trajectories and avoiding a separate ODROID generative training task
are the user's supplied reasons; explicit geometric control is a design argument
grounded in the current stored-path and conversion implementation.
'''
p.write_text(t,encoding='utf-8')
def cites(t): return {k.strip() for a in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}',t) for k in a.split(',')}
lit=(ROOT/'Report/PrismSnapshot_2026-10-31/3_Main_Report/1_Literature_Study.tex').read_text(encoding='utf-8')
des='\n'.join(p.read_text(encoding='utf-8') for folder in ['3_Main_Report','Appendices'] for p in (B/folder).rglob('*.tex'))
dk=cites(des);lk=cites(lit)
baseline='\n'.join(p.read_text(encoding='utf-8') for p in (ROOT/'Report/PrismSnapshot_2026-10-31/3_Main_Report').glob('*.tex') if not p.name.startswith('3_'))
audit=json.loads((B/'SOURCE_AUDIT_COUNTS.json').read_text(encoding='utf-8'))
audit.update(design_and_appendix_sources=len(dk),shared_with_literature=sorted(dk&lk),union_literature_design=len(dk|lk),union_current_baseline_sections_and_revised_design=len(dk|cites(baseline)),external_citation_sites=len(re.findall(r'\\cite',des)),citations_with_locators=len(re.findall(r'\\cite\[',des)))
(B/'SOURCE_AUDIT_COUNTS.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))
