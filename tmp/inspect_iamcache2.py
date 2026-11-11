import pickle
from pathlib import Path
x=pickle.loads(Path('Software/CNN/NOGIT/qtp_cache/iam_lines.pkl').read_bytes());r=x[0];print(type(r));print(list(r) if isinstance(r,dict) else str(r)[:200]);
if isinstance(r,dict):print({k:(type(v).__name__,getattr(v,'shape',None),str(v)[:100] if not hasattr(v,'shape') else '') for k,v in r.items()})
