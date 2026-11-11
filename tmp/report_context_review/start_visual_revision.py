from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[2]
src=root/'Report/DesignRevision_2026-11-01'
dst=root/'Report/DesignVisualRevision_2026-11-01'
dst.mkdir(exist_ok=True)
for p in src.rglob('*'):
    rel=p.relative_to(src)
    if p.is_file() and p.suffix.lower() in {'.tex','.cls','.bib','.md','.json','.py','.png','.jpg','.pdf'} and p.name!='design_preview.pdf':
        target=dst/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
support=dst/'design_support.tex'
s=support.read_text(encoding='utf-8')
s+=r'''
% Line-numbered, indented pseudocode; no external algorithm package needed.
\newcounter{designline}
\newenvironment{designpseudo}{\small\setcounter{designline}{0}\begin{list}{}{\setlength{\leftmargin}{2.1em}\setlength{\labelwidth}{1.6em}\setlength{\labelsep}{0.5em}\setlength{\itemsep}{0pt}\setlength{\parsep}{0pt}\setlength{\topsep}{3pt}}}{\end{list}}
\newcommand{\pstate}[2]{\stepcounter{designline}\item[\arabic{designline}:]\hspace*{#1em}#2}
\newcommand{\prequire}[1]{\par\small\noindent\textbf{Require:} #1\par}
\newcommand{\pensure}[1]{\par\small\noindent\textbf{Ensure:} #1\par}
'''
support.write_text(s,encoding='utf-8')
print(dst)
