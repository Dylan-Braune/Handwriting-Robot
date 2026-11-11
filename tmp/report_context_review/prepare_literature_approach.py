from pathlib import Path
import shutil, re, json, urllib.request, concurrent.futures
root=Path(__file__).resolve().parents[2]
old=root/'Report/DesignResultsDiscussion_2026-11-01'
new=root/'Report/IntegratedReport_2026-11-01'
assert not new.exists(), 'Preserve any already existing revision'
shutil.copytree(old,new,ignore=shutil.ignore_patterns('__pycache__','report_preview.aux','report_preview.log','report_preview.bbl','report_preview.blg','report_preview.out','report_preview.pdf'))
for name in ['1_Literature_Study.tex','2_Approach.tex']:
 shutil.copy2(root/'Report/PrismSnapshot_2026-10-31/3_Main_Report'/name,new/'3_Main_Report'/name)
src=new/'SourceChecks';src.mkdir(exist_ok=True)
urls={
 'chen2023':'https://drpress.org/ojs/index.php/HSET/article/download/5372/5196/5197',
 'ma2023':'https://www.scitepress.org/Papers/2023/128010/128010.pdf',
 'hemanth2022':'https://ictactjournals.in/paper/IJSC_Vol_12_Iss_1_Paper_1_2457_2463.pdf',
 'bulacu2007':'https://www.ai.rug.nl/~mbulacu/tpami2007-bulacu-schomaker.pdf',
 'lecun1998':'http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf',
 'shi2017':'https://arxiv.org/pdf/1507.05717',
}
def fetch(item):
 key,url=item
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
  data=urllib.request.urlopen(req,timeout=25).read()
  assert data.startswith(b'%PDF')
  (src/f'{key}.pdf').write_bytes(data)
  return key,dict(url=url,status='downloaded')
 except Exception as e:return key,dict(url=url,status='unavailable',error=str(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: records=dict(pool.map(fetch,urls.items()))
(src/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
