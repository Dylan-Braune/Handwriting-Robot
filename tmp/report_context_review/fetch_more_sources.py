from pathlib import Path
import concurrent.futures, json, urllib.request
from pypdf import PdfReader
OUT=Path(__file__).parent/'sources'
items={
 'zhang1984':'https://agcggs680.pbworks.com/f/Zhan-Suen_algorithm.pdf',
 'bresenham1965':'https://janmr.com/files/papers/bresenham65.pdf',
 'graves2013':'https://arxiv.org/pdf/1308.0850',
}
def run(item):
 k,u=item
 try:
  with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'}),timeout=20) as r: data=r.read()
  if not data.startswith(b'%PDF'): raise ValueError('not PDF')
  p=OUT/f'{k}.pdf';p.write_bytes(data);reader=PdfReader(p)
  (OUT/f'{k}.txt').write_text('\n'.join(f'\n=== PDF PAGE {i+1} ===\n'+(v.extract_text() or '') for i,v in enumerate(reader.pages)),encoding='utf-8')
  return k,{'url':u,'pages':len(reader.pages),'status':'obtained; locator to verify'}
 except Exception as e: return k,{'url':u,'status':'unavailable','error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex: res=dict(ex.map(run,items.items()))
(OUT/'MORE_MANIFEST.json').write_text(json.dumps(res,indent=2),encoding='utf-8'); print(json.dumps(res,indent=2))
