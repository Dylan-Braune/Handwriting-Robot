from pathlib import Path
import json,hashlib,shutil,sys,platform,zipfile
import numpy,scipy,PIL,cv2,matplotlib
B=Path('Report/GeometricFidelity_2026-11-02');data=json.loads((B/'input_manifest.json').read_text());index=[]
for r in data['physical']:
 src=Path(r['file']);dst=B/'inputs'/f"physical_{r['writer']}_{r['line']:02}.png";shutil.copy2(src,dst);index.append({'source':str(src),'snapshot':str(dst.relative_to(B)),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
(B/'input_snapshot_index.json').write_text(json.dumps(index,indent=2))
(B/'environment.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'numpy':numpy.__version__,'scipy':scipy.__version__,'pillow':PIL.__version__,'opencv':cv2.__version__,'matplotlib':matplotlib.__version__},indent=2))
(B/'requirements.txt').write_text('numpy\nscipy\npillow\nmatplotlib\nopencv-python\n')
# Verify the preserved source snapshots against the primary input manifest.
for r in index:
 assert r['sha256']==data['inputs'][r['source']],r['source']
manifest={str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in B.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ARCHIVE_MANIFEST.json'}
(B/'ARCHIVE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
out=Path('output/Handwriting_Geometric_Comparison_2026-11-02.zip')
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for p in B.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:z.write(p,'GeometricFidelity_2026-11-02/'+p.relative_to(B).as_posix())
with zipfile.ZipFile(out) as z:assert z.testzip() is None
shutil.copy2(B/'FINDINGS.md','output/Handwriting_Geometric_Findings_2026-11-02.md')
print('Final archive',round(out.stat().st_size/1e6,2),'MB. Preserved physical input snapshots',len(index))
