from pathlib import Path
import json,zipfile,hashlib,shutil
R=Path(__file__).resolve().parents[2];B=R/'Report/IntegratedReport_2026-11-01'
p=B/'INTEGRATED_REVIEW.json';review=json.loads(p.read_text())
assert not review['missing_references'] and not review['build_warnings']
assert review['inline_editorial_notes']==0 and review['missing_figure_placeholders']==0
review['visual_review']=dict(status='complete',rendered_pages=review['pdf_pages'],inspected='New Literature and Approach pages plus contact sheets covering all rendered pages; hardware/software diagrams and equation/citation layout checked',findings='No clipped or overlapping text, broken tables, missing graphics or unresolved reference markers observed.')
p.write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
out=R/'output';(out/'pdf').mkdir(exist_ok=True)
pdf=out/'pdf/Handwriting_Robot_Report_Sections_1-5_2026-11-01.pdf'
shutil.copy2(B/'report_preview.pdf',pdf)
completion=out/'Handwriting_Robot_Author_Completion_List_2026-11-01.md'
shutil.copy2(B/'AUTHOR_COMPLETION_LIST.md',completion)
archive=out/'Handwriting_Robot_Report_Sections_1-5_Source_2026-11-01.zip'
skip={'.aux','.bbl','.blg','.out','.toc','.pyc'};files=[]
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(B.rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts:continue
  rel=p.relative_to(B)
  if len(rel.parts)==1 and p.name.startswith('report_preview.') and p.suffix!='.tex':continue
  if rel.parts[0]=='SourceChecks' and p.suffix in {'.pdf','.txt'}:continue
  if p.suffix in skip:continue
  z.write(p,rel.as_posix());files.append(rel.as_posix())
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for name in ['AUTHOR_COMPLETION_LIST.md','EQUATION_SOURCE_AUDIT.md','3_Main_Report/1_Literature_Study.tex','3_Main_Report/2_Approach.tex','ResultsEvidence/measurements/raw/odroid_server_log.txt']:
  assert name in z.namelist(),name
assert pdf.read_bytes()==(B/'report_preview.pdf').read_bytes()
context=R/'Report/REPORT_COLLABORATION_CONTEXT.md'
text=context.read_text(encoding='utf-8')
note=f'''## Current revision: Literature and Approach integrated, 1 November 2026

The latest working report is `Report/IntegratedReport_2026-11-01/`, built from
the previous Design/Results/Discussion revision and the author's original
Prism Literature/Approach. Sections 1-5 are now included, with references and
appendices. Literature is {review['section_pages']['literature']} pages and Approach {review['section_pages']['approach']} pages. Main text is {review['main_pages']} pages;
the final whole-report 70-page cap still requires compression and Conclusion
and front matter remain outside this assembly. Do not claim full submission readiness.

The Literature preserves the author's explanatory voice where accurate, removes
Intended contribution, adds topic-level project applications, and retains a brief
Application of the literature summary. Approach follows the proposal's functional
units and the final hardware/segmentation/classification/reproduction arrangement.
Unsupported multiplied-accuracy and all-no-library claims were removed.

User's latest citation preference: only source and page number(s) in text; no
chapter, section, algorithm, equation or PDF qualifiers. Where a page is not
verified, cite the source alone. One bibliography has {review['cited_sources']} cited entries. Existing
Ma and Hemanth bibliographic records were corrected, not duplicated. All {review['numbered_equations']} numbered
equations have a source/derivation basis in EQUATION_SOURCE_AUDIT.md. Own formulas
are not falsely attributed to papers. No code citations were introduced.

All inline editorial comments/blue notes and absent-figure placeholders were
removed from the manuscript and preserved in AUTHOR_COMPLETION_LIST.md. Unknown
measurements remain factual limitations, not invented completions. Both user
hardware/software diagrams remain byte-for-byte unchanged and retain their
numbered explanations underneath. Measurements and model choices remain as
previously recorded. No training, gantry operation, deployment, commit or push.

Delivery: `output/pdf/Handwriting_Robot_Report_Sections_1-5_2026-11-01.pdf`,
`output/Handwriting_Robot_Report_Sections_1-5_Source_2026-11-01.zip`, and
`output/Handwriting_Robot_Author_Completion_List_2026-11-01.md`.

'''
if '## Current revision: Literature and Approach integrated' not in text:
 context.write_text(text.replace('# Report collaboration context\n','# Report collaboration context\n\n'+note,1),encoding='utf-8')
print(json.dumps(dict(pdf=str(pdf),source_zip=str(archive),completion_list=str(completion),files=len(files),zip_bytes=archive.stat().st_size,pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest()),indent=2))
