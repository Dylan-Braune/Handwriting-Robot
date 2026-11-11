from pathlib import Path
import urllib.request,concurrent.futures,json
from pypdf import PdfReader
HERE=Path(__file__).parent/'sources'
items={'haines2016':'https://reality.cs.ucl.ac.uk/projects/haines/haines16handwriting.pdf',
       'kang2020':'https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680273.pdf',
       'dai2024':'https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07598.pdf'}
def fetch(k,u):
    p=HERE/(k+'.pdf')
    if not p.exists():
        with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'}),timeout=40) as r:p.write_bytes(r.read())
    doc=PdfReader(p)
    (HERE/(k+'.txt')).write_text('\n'.join(f'\n=== PDF PAGE {i+1} ===\n'+(p.extract_text() or '') for i,p in enumerate(doc.pages)),encoding='utf-8')
    return k,{'url':u,'pages':len(doc.pages),'status':'downloaded; relevant pages to inspect'}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    results=dict(pool.map(lambda x:fetch(*x),items.items()))
(HERE/'SYNTHESIS_MANIFEST.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))
