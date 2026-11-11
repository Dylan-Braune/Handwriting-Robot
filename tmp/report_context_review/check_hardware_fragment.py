from pathlib import Path
import re,collections
b=Path('output/Hardware_Revision_2026-11-02');s=(b/'3_3_hardware.tex').read_text(encoding='utf-8');bib=(b/'references.bib').read_text(encoding='utf-8')
a=collections.Counter(re.findall(r'\\begin\{([^}]+)\}',s));c=collections.Counter(re.findall(r'\\end\{([^}]+)\}',s));assert a==c,(a,c)
keys=set(re.findall(r'@\w+\{([^,]+)',bib));assert all(k in keys for g in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}',s) for k in g.split(','))
for q in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',s):assert (b/q).exists(),q
assert s.index('\\label{alg:des-stepping}')>s.index('\\subsubsection{Gcode generation and motion mapping}')
print('Environment balance, citations, all images and algorithm placement checked.')
