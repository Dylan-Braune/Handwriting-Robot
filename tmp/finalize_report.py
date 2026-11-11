from pathlib import Path
import re,shutil,json,hashlib,zipfile
from pypdf import PdfReader
import pypdfium2 as pdfium
P=Path('Report/SubmissionRevision_2026-11-02').resolve();O=Path('output').resolve();T=Path('tmp/final_report_review');R=PdfReader(P/'.build/full_report.pdf');a=(P/'.build/full_report.aux').read_text();log=(P/'.build/full_report.log').read_text(errors='replace')
assert len(R.pages)==133
for pattern in ['undefined','multiply defined','Overfull','! LaTeX Error']:
 assert pattern not in log,pattern
base=Path('tmp/final_report_review/baseline')
for f in ['3_Main_Report/1_Literature_Study.tex','3_Main_Report/2_Approach.tex','2_Project_Definition/project_definition.tex','2_Project_Definition/project_proposal.pdf']:
 if (base/f).exists():assert (P/f).read_bytes()==(base/f).read_bytes(),f
 else:
  with zipfile.ZipFile(r'C:/Users/braun/Downloads/IntegratedReport_2026-11-01.zip') as zz:
   key=next(n for n in zz.namelist() if n.endswith(f));assert (P/f).read_bytes()==zz.read(key),f
# Source has no missing cite records or repeated labels.
texts='\n'.join(p.read_text() for p in P.rglob('*.tex') if '.build' not in p.parts)
labels=re.findall(r'\\label\{([^}]+)\}',texts);assert len(labels)==len(set(labels))
bib=(P/'references.bib').read_text();known=set(re.findall(r'@\w+\s*\{\s*([^,]+)',bib));used={k.strip() for m in re.finditer(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}',texts) for k in m[1].split(',')};assert used<=known
pdf=O/'pdf/Handwriting_Robot_Concise_Revision_2026-11-02.pdf';pdf.parent.mkdir(exist_ok=True);shutil.copy2(P/'.build/full_report.pdf',pdf)
source=O/'Handwriting_Robot_Concise_Source_2026-11-02.zip'
with zipfile.ZipFile(source,'w',zipfile.ZIP_DEFLATED) as z:
 for f in P.rglob('*'):
  if f.is_file() and '.build' not in f.relative_to(P).parts:z.write(f,f.relative_to(P).as_posix())
 with z.open('full_report.tex') as f:assert f.read()==(P/'full_report.tex').read_bytes()
# Capture final QA views, using the actual referenced PDF page number.
render=T/'render';render.mkdir(exist_ok=True);doc=pdfium.PdfDocument(str(pdf));offset=next(i for i in range(1,len(R.pages)) if R.page_labels[i]=='1')
for lab in ['fig:res-owen-page','fig:res-reproduced-confusion','fig:app-physical-pages-1','fig:gantry-cad-components','fig:des-seam-example','fig:des-firstfeatures']:
 m=re.search(r'\\newlabel\{'+re.escape(lab)+r'\}\{\{[^}]*\}\{(\d+)\}',a);n=int(m[1]);idx=offset+n-1
 doc[idx].render(scale=1.1).to_pil().save(render/(lab.replace(':','_')+'.png'))
manifest={'main_pages':63,'design_pages':37,'full_pdf_pages':len(R.pages),'printed_references':len(re.findall(r'\\bibitem',(P/'.build/full_report.bbl').read_text())),'source_files':sum(1 for f in P.rglob('*') if f.is_file() and '.build' not in f.relative_to(P).parts),'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'warnings_checked':['no undefined citations/references','no duplicate labels','no overfull boxes'],'scope':'All main headings retained. Literature, Approach and approved proposal unchanged.'}
(T/'FINAL_VALIDATION.json').write_text(json.dumps(manifest,indent=2));print(manifest);print('Source MB',round(source.stat().st_size/1e6,2));print('PDF MB',round(pdf.stat().st_size/1e6,2))
