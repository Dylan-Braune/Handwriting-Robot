from pathlib import Path
import json,hashlib
B=Path('Report/GeometricFidelity_2026-11-02');raw=json.loads((B/'raw_results.json').read_text());old="sys.path.insert(0,str(ROOT/'Software/CleanPipeline3'))";new="sys.path.insert(0,str(ROOT/'Software/CleanPipeline3' if (ROOT/'Software/CleanPipeline3/segment.py').exists() else B/'source'))"
for fn,expected in raw['code_sha256'].items():
 text=(B/fn).read_text().replace(new,old);base=text.rstrip('\r\n');found=False
 for line in ['\n','\r\n']:
  for end in ['','\n','\r\n','\n\n','\r\n\r\n','\n\r\n','\r\n\n']:
   for bom in [b'',b'\xef\xbb\xbf']:
    data=bom+(base.replace('\n',line)+end).encode()
    if hashlib.sha256(data).hexdigest()==expected:(B/'source'/(Path(fn).stem+'_as_run.py')).write_bytes(data);print(fn,'recovered',repr(line),repr(end));found=True;break
   if found:break
  if found:break
 if not found:print('NOT FOUND',fn,expected,hashlib.sha256((B/fn).read_bytes()).hexdigest())
