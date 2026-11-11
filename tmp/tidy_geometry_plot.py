from pathlib import Path
p=Path('Report/GeometricFidelity_2026-11-02/make_evidence.py');s=p.read_text();s=s.replace("  ax.text(i-.16,means[i]+1.5,f'{means[i]:.1f}',ha='center',fontsize=9)","  label_y=(100*st['group_bootstrap_95'][1]+1.5) if n>=10 else means[i]+1.5\n  ax.text(i-.16,label_y,f'{means[i]:.1f}',ha='center',fontsize=9)")
p.write_text(s)
