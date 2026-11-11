from pathlib import Path
import json,pickle,io,zlib
import numpy as np
from PIL import Image,ImageDraw
B=Path('Report/GeometricFidelity_AllAuthors_2026-11-02');x=json.loads((B/'candidates.json').read_text());ids=['R00638','R00656','R00767','R00775','R00776','R00778','R00779','R00781','R00786','R00806','R00160','R00405']
iam=pickle.loads(Path('Software/CNN/NOGIT/qtp_cache/iam_lines.pkl').read_bytes());per=pickle.loads(Path('Software/CNN/NOGIT/personal_lines_cache.pkl').read_bytes())['samples'];out=Image.new('RGB',(2000,len(ids)*140),'white');dr=ImageDraw.Draw(out)
for i,uid in enumerate(ids):
 r=next(r for r in x['references'] if r['id']==uid)
 if r['writer'].isdigit():
  s=next(q for q in iam if q['author']==r['writer'] and Path(q['page_key']).stem==r['page'] and q['line_idx']==r['line']);im=Image.open(io.BytesIO(s['png'])).convert('RGB')
 else:
  s=next(q for q in per if q['page_key'].startswith(r['writer']+'/'+r['page']+'_') and q['line_idx']==r['line'])
  try:im=Image.fromarray(np.load(io.BytesIO(zlib.decompress(s['imageBytes'])))).convert('RGB')
  except:im=Image.open(io.BytesIO(s['imageBytes'])).convert('RGB')
 dr.text((5,i*140+2),f"{uid} expected {r['token']} at word {r['word_index']}, current crop x={r['box'][0]}..{r['box'][2]}. Transcript {s['text']}",fill='black');im.thumbnail((1980,100));out.paste(im,(5,i*140+30))
out.save(B/'review/questionable_contexts.jpg')
print('Context sheet saved')
