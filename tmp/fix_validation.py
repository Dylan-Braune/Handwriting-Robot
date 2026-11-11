from pathlib import Path
p=Path('tmp/finalize_report.py');s=p.read_text();s=s.replace(' assert (P/f).read_bytes()==(base/f).read_bytes(),f', ''' if (base/f).exists():assert (P/f).read_bytes()==(base/f).read_bytes(),f
 else:
  with zipfile.ZipFile(r'C:/Users/braun/Downloads/IntegratedReport_2026-11-01.zip') as zz:
   key=next(n for n in zz.namelist() if n.endswith(f));assert (P/f).read_bytes()==zz.read(key),f''');p.write_text(s)
