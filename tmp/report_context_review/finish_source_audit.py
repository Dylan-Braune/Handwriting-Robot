from pathlib import Path
import re,json
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'Report/DesignRevision_2026-11-01'
p=B/'3_Main_Report/design/3_5_classification.tex';t=p.read_text(encoding='utf-8')
t=t.replace('Public IAM handwriting and the photographed personal contributions provide examples with known text and writer identity.',
 r'Public IAM handwriting~\cite{marti2002}\designsourceflag{marti2002} and the photographed personal contributions provide examples with known text and writer identity.')
p.write_text(t,encoding='utf-8')
# Record the exact unpaginated/read-copy locations without creating new keys.
bib=B/'references.bib';s=bib.read_text(encoding='utf-8')
urls={
'ioffe2015':'https://proceedings.mlr.press/v37/ioffe15.pdf',
'graves2006':'https://www.cs.toronto.edu/~graves/icml_2006.pdf',
'graves2013':'https://arxiv.org/pdf/1308.0850',
'kizilirmak2023':'https://arxiv.org/pdf/2307.00664',
'avidan2007':'https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Avidan07.pdf',
'zhang1984':'https://agcggs680.pbworks.com/f/Zhan-Suen_algorithm.pdf',
'rosenfeld1966':'https://www.cs.virginia.edu/~jlp/66.sequential.op.pdf',
'srivastava2014':'https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf',
'bresenham1965':'https://janmr.com/files/papers/bresenham65.pdf',
'kingma2015':'https://arxiv.org/pdf/1412.6980',
'loshchilov2019':'https://arxiv.org/pdf/1711.05101',
'tieleman2012':'https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf',
}
for key,url in urls.items():
 m=re.search(r'@\w+\{'+re.escape(key)+r',',s)
 if not m: raise ValueError(key)
 start=s.index('{',m.start());depth=1;i=start+1
 while depth:
  if s[i]=='{':depth+=1
  elif s[i]=='}':depth-=1
  i+=1
 entry=s[m.start():i]
 if not re.search(r'\burl\s*=',entry):
  entry=entry[:-1]+',\n  url={'+url+'}}'
  s=s[:m.start()]+entry+s[i:]
bib.write_text(s,encoding='utf-8')
def cites(t):return {k.strip() for a in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}',t) for k in a.split(',')}
lit=ROOT/'Report/PrismSnapshot_2026-10-31/3_Main_Report/1_Literature_Study.tex'
litkeys=cites(lit.read_text(encoding='utf-8'))
texts=[p.read_text(encoding='utf-8') for p in (B/'3_Main_Report').rglob('*.tex')]+[(B/'Appendices/design_implementation_details.tex').read_text(encoding='utf-8')]
des='\n'.join(texts);dk=cites(des)
baseline='\n'.join(p.read_text(encoding='utf-8',errors='replace') for p in (ROOT/'Report/PrismSnapshot_2026-10-31/3_Main_Report').glob('*.tex') if not p.name.startswith('3_'))
basekeys=cites(baseline)
pending=sorted(set(re.findall(r'\\designsourceflag\{([^}]+)\}',des)))
audit={'literature_sources':len(litkeys),'design_and_appendix_sources':len(dk),
       'shared_with_literature':sorted(dk&litkeys),
       'union_literature_design':len(dk|litkeys),
       'union_current_baseline_sections_and_revised_design':len(dk|basekeys),
       'pending_manual_source_checks':pending,
       'external_citation_sites':len(re.findall(r'\\cite',des)),
       'citations_with_locators':len(re.findall(r'\\cite\[',des)),
       'new_external_keys_vs_baseline_bib':sorted(dk-set(re.findall(r'@\w+\{([^,]+),',(ROOT/'Report/PrismSnapshot_2026-10-31/references.bib').read_text(encoding='utf-8')))-{'braunepipeline2026'})}
(B/'SOURCE_AUDIT_COUNTS.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit,indent=2))
