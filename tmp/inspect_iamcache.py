from pathlib import Path
import pickle
p=Path('Software/CNN/NOGIT/qtp_cache/iam_lines.pkl');x=pickle.loads(p.read_bytes());print(type(x),str(list(x)[:4])[:600] if isinstance(x,dict) else len(x));
if isinstance(x,dict):
 k=next(iter(x));v=x[k];print('first',k,type(v),str(v)[:500])
