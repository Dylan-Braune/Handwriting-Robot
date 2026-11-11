from pathlib import Path
from pypdf import PdfReader
import zipfile,json
r=Path.cwd();o=r/'tmp/final_report_review';o.mkdir(exist_ok=True)
z=zipfile.ZipFile('C:/Users/braun/Downloads/IntegratedReport_2026-11-01.zip');names=z.namelist();(o/'zip_files.json').write_text(json.dumps(names,indent=2));print('ZIP files',len(names));print('\n'.join(n for n in names if n.endswith(('.tex','.bib','.cls','.py'))))
p=PdfReader('C:/Users/braun/Downloads/23539195Revision1.pdf');annotations=[];texts=[]
for i,page in enumerate(p.pages):
 texts.append(f'\n=== PDF PAGE {i+1} ===\n'+(page.extract_text() or ''))
 for ar in page.get('/Annots',[]):
  a=ar.get_object();annotations.append({'page':i+1,'type':str(a.get('/Subtype')),'text':str(a.get('/Contents','')),'subject':str(a.get('/Subj','')),'rect':str(a.get('/Rect'))})
(o/'revision1_text.txt').write_text(''.join(texts),encoding='utf-8');(o/'revision1_annotations.json').write_text(json.dumps(annotations,indent=2),encoding='utf-8')
print('PDF pages',len(p.pages),'annotations',len(annotations));print(json.dumps([a for a in annotations if a['text']],indent=2)[:20000])
