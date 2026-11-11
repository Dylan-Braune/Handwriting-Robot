import ast, re, json, hashlib, csv, shutil
from pathlib import Path
from methods import F, classify, EQ
from current_notes import NOTES
from reading_flow import STAGES, FLOW, route_report
from theory_corrections import correct

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
BUILD=BASE/'build'
SNAP=BASE/'source_snapshot'
OLD=ROOT/'Software/CleanPipeline2/testing/results/CodeTheory'
A=json.loads((BASE/'audit.json').read_text(encoding='utf-8'))
BUILD.mkdir(exist_ok=True)
for handfile in ['report_priority.tex','current_pipeline.tex']:
    hp=BUILD/handfile
    raw=hp.read_text(encoding='utf-8')
    tokens=re.split(r'(\$[^$]*\$|\\path\{[^}]*\})',raw,flags=re.S)
    hp.write_text(''.join(t if t.startswith('$') or t.startswith('\\path{') else re.sub(r'(?<!\\)_',r'\\_',t) for t in tokens),encoding='utf-8')
def esc(s):
    table={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(table.get(c,c) for c in str(s))
def norm(node):
    class N(ast.NodeTransformer):
        def visit_FunctionDef(self,n):
            self.generic_visit(n); n.name=n.name.lower();return n
        def visit_Name(self,n):n.id=n.id.lower();return n
        def visit_arg(self,n):n.arg=n.arg.lower();return n
        def visit_Attribute(self,n):self.generic_visit(n);n.attr=n.attr.lower();return n
    return ast.dump(N().visit(ast.parse(ast.unparse(node))),include_attributes=False)
def q(kind,title,body):return '\\Q{'+kind+'}{'+esc(title)+'}\n'+body+'\n\n'
def label(e):return 'fn:'+e['module'].replace('/','-').replace('.py','')+':'+e['name'].replace('_','-')
def group_label(k):return 'method:'+k
def card(e):
    fam=F[classify(e)]
    key=e['module']+':'+e['name']
    text=NOTES.get(key)
    if not text:
        text=e['comment'].split('---')[-1].strip()
        text=re.sub(r'^'+re.escape(e['name'])+r'\s*:\s*','',text)
        if len(text)>650:text=text[-650:]
        if not text:text=fam['purpose']
    if text and text[0].islower():text=text[0].upper()+text[1:]
    text=re.sub(r'={3,}','',text).strip()
    if text and text[-1] not in '.!?':text+='.'
    return text
def questionize(tex):
    def sub(m):
        title=m.group(1); lo=title.lower()
        kind='Method'
        if any(x in lo for x in ['what it is for','purpose','problem','why it matters']):kind='Purpose'
        elif any(x in lo for x in ['why','reason','choice','advantage']):kind='Justification'
        elif any(x in lo for x in ['alternative','rather than']):kind='Alternatives'
        elif any(x in lo for x in ['cost','speed','memory','performance']):kind='Library'
        elif any(x in lo for x in ['guard','failure','subtle','limit','caveat','risk']):kind='Evidence'
        return r'\Q{'+kind+'}{'+title+'}'
    return re.sub(r'\\paragraph\{([^{}]*)\}',sub,tex)

# Bibliography source metadata is revised in this build only.
shutil.copy2(OLD/'refs.bib',BUILD/'refs.bib')
bib=(OLD/'expansion_sources.bib').read_text(encoding='utf-8-sig')
bib=bib.replace('CleanPipeline3 source snapshot inspected 10 November 2026','DemoCode source snapshot captured for the November 2026 documentation revision').replace('Local Software/CleanPipeline3: paths.py, segment.py, recognise.py, style.py, synthesise.py, reproduce.py and train.py. Read-only implementation evidence, not an as-run experimental log','Local Software/DemoCode including core, training, web server, lineworkers, camera, curation, retracing and timing. Exact source hashes and static coverage manifest accompany this edition. No inference or hardware execution')
(BUILD/'expansion_sources.bib').write_text(bib,encoding='utf-8')

# Reuse the long worked report explanations, with current-tree corrections.
for name in ['segment','recognise','style','synth','hardware','train','verification']:
    tex=(OLD/f'expansion_{name}.tex').read_text(encoding='utf-8-sig')
    tex=tex.replace('CleanPipeline3','DemoCode')
    tex=tex.replace('does not certify all forty-four report references or every claim in the\nunchanged original 335 pages.', 'does not certify all forty-four report references or every historical claim.\nCurrent implementation explanations have been revised against the captured DemoCode tree.')
    tex=tex.replace('The inspected \\texttt{WordAcc}', 'The inspected server \\texttt{WordAcc}')
    tex=tex.replace('an older legibility-only routine has', 'an older legibility-only routine had')
    tex=tex.replace('An older legibility-only routine has','An older legibility-only routine had')
    tex=tex.replace('An older legibility-only routine','The removed older legibility-only routine')
    tex=tex.replace('The inspected joint defaults use six tries','The direct joint function defaults use six tries')
    if name=='synth':
        tex=tex.replace('An older legibility-only routine has','The removed older legibility-only routine had')
        tex=tex.replace("A writer's medoid, a prototype, a case-related glyph and a generic anchor", "A writer's medoid, a separately stored prototype, a case-related glyph and a generic anchor")
    if name=='hardware':
        tex=tex.replace('The source assumes this setting; it does not','The code assumes this setting; it does not')
        tex=tex.replace('A mean of job ratios','A mean of job ratios')
    # Colour existing subhead labels without changing their mathematical content.
    tex=questionize(tex)
    # Every displayed formula receives purpose/use/source context from its subsection.
    parts=re.split(r'(?=\\(?:subsection|section)\{)',tex)
    changed=[]
    for part in parts:
        title=re.search(r'\\(?:subsection|section)\{([^}]*)\}',part)
        title=title.group(1) if title else name
        refs={'segment':'otsu1979,crow1984,code2026','recognise':'graves2006,shi2017,code2026','style':'douglaspeucker1973,zhangsuen1984,code2026','synth':'code2026','hardware':'code2026,a4988datasheet','train':'gravesbook,code2026','verification':'finalreport2026,code2026'}[name]
        pat=r'\\begin\{(equation\*?|align\*?|gather\*?)\}.*?\\end\{\1\}|\\\[.*?\\\]'
        def annotate(m):
            return r'\FormulaNote{'+esc('Purpose in this explanation: '+title)+r'}{'+refs+'}\n'+m.group(0)
        part=re.sub(pat,annotate,part,flags=re.S)
        changed.append(part)
    tex=''.join(changed)
    (BUILD/f'worked_{name}.tex').write_text(tex,encoding='utf-8')

# A replacement opening for the old unqualified scope statements.
preamble=r'''\documentclass[11pt,a4paper,oneside]{report}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=2.1cm,headheight=15pt]{geometry}
\usepackage{amsmath,amssymb,enumitem,booktabs,longtable,array,listings,xcolor,titlesec,fancyhdr,microtype}
\usepackage[colorlinks=true,linkcolor=navy,urlcolor=navy,citecolor=source]{hyperref}
\usepackage{url}
\urlstyle{same}
\definecolor{navy}{HTML}{17344A}
\definecolor{purpose}{HTML}{126B91}
\definecolor{method}{HTML}{495AA1}
\definecolor{reason}{HTML}{176B50}
\definecolor{alternative}{HTML}{80508F}
\definecolor{library}{HTML}{8B5814}
\definecolor{source}{HTML}{7B3C42}
\definecolor{sigbg}{HTML}{EFF4F7}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5.5pt plus 1pt minus 1pt}
\setlength{\emergencystretch}{4em}
\setcounter{secnumdepth}{2}
\setcounter{tocdepth}{0}
\titleformat{\chapter}[hang]{\sffamily\LARGE\bfseries\color{navy}}{\thechapter}{0.6em}{}
\titlespacing*{\chapter}{0pt}{0pt}{16pt}
\titleformat{\section}{\sffamily\Large\bfseries\color{navy}}{\thesection}{0.6em}{}
\titleformat{\subsection}{\sffamily\large\bfseries\color{navy}}{\thesubsection}{0.6em}{}
\newcommand{\Q}[2]{\par\addvspace{7pt}\begingroup\sffamily\bfseries\small
\def\qt{#1}\def\pp{Purpose}\def\jj{Justification}\def\aa{Alternatives}\def\ll{Library}\def\ee{Evidence}
\ifx\qt\pp\color{purpose}\else\ifx\qt\jj\color{reason}\else\ifx\qt\aa\color{alternative}\else\ifx\qt\ll\color{library}\else\ifx\qt\ee\color{source}\else\color{method}\fi\fi\fi\fi\fi
#1\enspace\textbar\enspace #2\par\endgroup\nobreak\smallskip}
\newcommand{\FormulaNote}[2]{\par\smallskip{\small\color{purpose}\textbf{Formula purpose and system use.} #1\enspace\textcolor{source}{\textbf{Source basis:} \cite{#2}. Exact project constants remain implementation choices unless evidence is stated.}}\par\nobreak}
\newcommand{\ReportEq}[6]{\subsection*{Report equation #1\quad #2}\phantomsection\label{report-eq:#1}\addcontentsline{toc}{subsection}{Report equation #1: #2}
\Q{Method}{Formula}#3\par\Q{Purpose}{Why the system needs it}#4\par
\Q{Evidence}{Current location and source}#5\enspace\cite{#6}.\par}
\lstdefinestyle{sig}{basicstyle=\ttfamily\small,backgroundcolor=\color{sigbg},frame=none,breaklines=true,columns=fullflexible,keepspaces=true,aboveskip=5pt,belowskip=7pt,literate={_}{{\_}}1}
\lstdefinestyle{plain}{basicstyle=\ttfamily\footnotesize,breaklines=true,columns=fullflexible,keepspaces=true,literate={_}{{\_}}1}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small\sffamily\color{navy}Handwriting Robot\quad Code theory}
\fancyhead[R]{\small\sffamily\color{navy}DemoCode edition}\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\begin{document}
\begin{titlepage}
\vspace*{1.4cm}
{\sffamily\Huge\bfseries\color{navy}Handwriting Robot\par Code Theory Manual\par}
\vspace{0.7cm}{\sffamily\Large Report methods and the current demo pipeline\par}
\vspace{1cm}D. R. Braune\quad u23539195\par University of Pretoria\par
\vspace{0.4cm}Revised study edition\quad November 2026\par
\vspace{1.1cm}
\Q{Purpose}{What problem does this solve} Blue labels explain the role in the robot.
\Q{Method}{How does it work} Indigo labels unpack mathematics and implementation.
\Q{Justification}{Why use this approach} Green labels state the rationale and assumptions.
\Q{Alternatives}{What else could have been used} Purple labels compare relevant methods.
\Q{Library}{What could a library change} Amber labels distinguish implementation, cost and equivalence.
\Q{Evidence}{Where does the claim come from} Burgundy labels identify sources, records and limits.
\vfill The report's named methods and all 35 numbered equations take priority.
Current signatures and source hashes identify the implementation being explained.
\end{titlepage}
\tableofcontents
\input{report_index}
\input{current_pipeline}
'''

method_tex=[r'\chapter{Method choices alternatives and libraries}\label{ch:methods}',
r'Each comparison below distinguishes an algorithmic trade-off from a documented historical choice. Library semantics are sourced; no new timing or accuracy benchmark was run. Identical weights and preparation should be compared before comparing training routes.']
for key,f in F.items():
    method_tex += ['\\section{'+esc(f['title'])+'}\\label{'+group_label(key)+'}',q('Purpose','The role in this system',esc(f['purpose'])),q('Justification','The reason for this construction',esc(f['why'])),q('Alternatives','What changes if another method is used',esc(f['alternatives'])),q('Library','Implementation speed and accuracy',esc(f['library'])+' \\cite{'+f['refs']+'}.')]
(BUILD/'methods.tex').write_text('\n'.join(method_tex),encoding='utf-8')
report_index,report_routes=route_report((BUILD/'report_priority.tex').read_text(encoding='utf-8'))
(BUILD/'report_index.tex').write_text(report_index,encoding='utf-8')
seen_families=set()
def method_comparison(key):
    f=F[key]
    seen_families.add(key)
    return '\n'.join(['\\section{Method background: '+esc(f['title'])+'}\\label{'+group_label(key)+'}',q('Purpose','The role in this system',esc(f['purpose'])),q('Justification','The reason for this construction',esc(f['why'])),q('Alternatives','What changes if another method is used',esc(f['alternatives'])),q('Library','Implementation speed and accuracy',esc(f['library'])+' \\cite{'+f['refs']+'}.')])

order=['paths.py','web/camera.py','segment.py','recognise.py','style.py','curate.py','synthesise.py','retrace.py','reproduce.py','web/server.py','web/lineworkers.py','web/render.py','train.py','timing.py']
titlemap={'paths.py':'Paths and datasets','segment.py':'Image processing and line segmentation','recognise.py':'Text recognition and writer identification','style.py':'Glyph extraction and writer profiles','synthesise.py':'Handwriting synthesis and candidate selection','reproduce.py':'Gantry mapping and control','train.py':'Training and numerical differentiation','web/server.py':'Web orchestration and evaluation','web/lineworkers.py':'Independent line workers','web/camera.py':'Camera acquisition','curate.py':'Glyph curation','retrace.py':'Direct page retracing','timing.py':'Timing and allowance accounting','web/render.py':'Inspection images and glyph sheets'}
worked={'segment.py':'segment','recognise.py':'recognise','style.py':'style','synthesise.py':'synth','reproduce.py':'hardware','train.py':'train'}
preserved=annotated=0; coverage=[]
for module in order:
    entries=[e for e in A['inventory'] if e['module']==module]
    if not entries:continue
    stages=STAGES[module]
    entries.sort(key=lambda e:(stages.index(classify(e)) if classify(e) in stages else len(stages),e['line']))
    out=['\\chapter{'+titlemap[module]+'}\\label{module:'+module.replace('/','-').replace('.py','')+'}',
         '\\Q{Evidence}{Current implementation boundary}\n\\path{Software/DemoCode/'+module+'}. Exact file hashes are in the accompanying source manifest. Names, declarations and source locations below come from the captured source, not a runtime import.']
    out.append(q('Purpose','Input-to-output reading flow',esc(FLOW[module])))
    if module in worked:
        out.append('\\input{worked_'+worked[module]+'}')
        out.append(r'\section*{Current function reference}')
    tree=ast.parse((SNAP/module).read_text(encoding='utf-8-sig'))
    # Short current settings; huge designed glyph dictionaries are source data.
    settings=[]
    for node in tree.body:
        if isinstance(node,(ast.Assign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            value=ast.unparse(node.value) if node.value else ''
            for target in targets:
                names=[n.id for n in ast.walk(target) if isinstance(n,ast.Name)]
                if names and len(value)<190 and (module=='paths.py' or any(n.isupper() or n in ['PenWidthXh','ValFraction','SplitSeed','InputHeight','InputWidth'] for n in names)):
                    settings.append(', '.join(names)+' = '+value)
    if settings:
        out += [r'\Q{Evidence}{Selected current module settings}',r'These are declarations or expressions, not recovered environment values. Source comments do not establish a measured optimum. Larger glyph coordinate tables are designed data in the bundled source.',r'\begin{lstlisting}[style=plain]','\n'.join(settings),r'\end{lstlisting}']
    previous_path=ROOT/'Software/CleanPipeline2'/module
    previous_defs={n.name:n for n in ast.parse(previous_path.read_text(encoding='utf-8-sig')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))} if previous_path.exists() else {}
    current_defs={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    last_family=None
    for e in entries:
        famkey=classify(e); fam=F[famkey]
        if famkey!=last_family:
            out.append('\\section*{Stage: '+esc(fam['title'])+'}')
            if (module,famkey) in report_routes:
                out.append(report_routes.pop((module,famkey)))
            if famkey not in seen_families:
                out.append(method_comparison(famkey))
            else:
                out.append('Method rationale, alternatives and library comparison: Section~\\ref{'+group_label(famkey)+'}.')
            last_family=famkey
        out += ['\\section{'+esc(e['name'])+'}\\label{'+label(e)+'}',r'\begin{lstlisting}[style=sig]',e['signature'],r'\end{lstlisting}',
                q('Evidence','Source and method link',r'\path{Software/DemoCode/'+module+'} line '+str(e['line'])+'. Implementation evidence: \\cite{code2026}. Related method, alternatives and library discussion: Section~\\ref{'+group_label(famkey)+'}.'),
                q('Purpose','What this contributes',esc(card(e)))]
        if e['methods']:
            out += [r'\Q{Method}{Methods exposed by this current class}',r'\begin{lstlisting}[style=plain]','\n'.join(m['signature'] for m in e['methods']),r'\end{lstlisting}']
        old=e.get('doc')
        same=e['same_old']
        if not same and e['module']+':'+e['name'] not in NOTES and e['name'] in previous_defs:
            same=norm(current_defs[e['name']])==norm(previous_defs[e['name']])
        if same and old:
            old=re.sub(r'^\\section\{[^}]*\}\s*','',old,count=1)
            old=re.sub(r'\\begin\{lstlisting\}\[style=sig\].*?\\end\{lstlisting\}','',old,count=1,flags=re.S)
            old=old.replace('.forward(','.Forward(').replace('author_mapping','authorMapping')
            old=old.replace('would eat twelve pixels off the height, which is a fifth of a 64-pixel crop','would remove twenty-four pixels in total across twelve convolutions, before accounting for any intervening pooling')
            old=old.replace('runs perhaps two orders of magnitude faster than the equivalent Python loops','can substantially reduce interpreter overhead relative to Python loops, with the actual factor depending on shape, backend and hardware')
            old=old.replace('Building the columns without copying','The strided view and reshape storage')
            old=old.replace('with no data copied','as a view without an initial patch copy; a later reshape can still materialise storage')
            old=old.replace('A product of a hundred numbers each around 0.1 is around ten to the power minus one hundred, far below the smallest number a double can represent, so it becomes exactly zero and all information is lost.','A product of a hundred values near 0.1 is about $10^{-100}$: representable in float64, but below the ordinary float32 range. Longer or lower-probability paths can also underflow in float64. Direct products therefore have a precision-dependent numerical risk.')
            old=old.replace('Working with logarithms removes it entirely','Working with logarithms avoids these direct tiny products')
            old=old.replace('the fit places the crop at the top left of the canvas and pads to the right and below, so the origins coincide','the fit is left aligned horizontally, so the horizontal origins coincide; its vertical placement is centred')
            old=old.replace('A character whose state is never occupied gets no span, which happens when the aligner had to pass through it in a single column shared with its neighbours; the caller handles that by skipping the glyph rather than cutting a zero-width one.','A character without a usable recovered span is handled as an alignment/extraction failure. A legal CTC transition cannot simply skip an obligatory character or assign two different target states to one frame.')
            old=old.replace("and since the derivative of ReLU is zero there, nothing ever pushes it back", "and its local derivative is zero there; it receives no direct gradient through that inactive response, though upstream changes can alter future inputs")
            old=old.replace("and it is ReLU's one real drawback", "and it is one training limitation of ReLU")
            old=old.replace('which needs some explanation','which needs some explanation')
            old=questionize(correct(old))
            index=[0]
            pat=r'\\begin\{(equation\*?|align\*?|gather\*?)\}.*?\\end\{\1\}|\\\[.*?\\\]'
            def ann(m):
                global annotated
                i=index[0];index[0]+=1;annotated+=1
                purposes=EQ.get(e['name'],[])
                purpose=purposes[i] if i<len(purposes) else fam['purpose']
                return r'\FormulaNote{'+esc(purpose)+'}{'+fam['refs']+'}\n'+m.group(0)
            old=re.sub(pat,ann,old,flags=re.S)
            out.append(old)
            preserved+=1
        else:
            out += [q('Method','Current implementation',esc(fam['purpose'])+' '+esc('The excerpt below is from the current definition; the complete function is in the source snapshot. Its constants and branches, rather than an older signature, govern this path.'))]
            # Current source excerpts are documentation only, never executed.
            lines=[s for s in e['source'].splitlines() if not s.lstrip().startswith('#')]
            if len(lines)>58:lines=lines[:58]+['# Excerpt ends; full definition continues in bundled source.']
            # Avoid copied narrative docstrings inside listings.
            txt='\n'.join(lines)
            txt=re.sub(r'""".*?"""','"""See source snapshot for the original docstring."""',txt,flags=re.S)
            txt=txt.encode('ascii','replace').decode()
            out += [r'\begin{lstlisting}[style=plain]',txt,r'\end{lstlisting}',q('Justification','Interpret the method rather than infer a benchmark',esc(fam['why'])+' \\cite{'+fam['refs']+'}.')]
        coverage.append(dict(module=module,name=e['name'],line=e['line'],methods=len(e['methods']),retained_theory=bool(same and old),family=famkey))
    filename='reference_'+module.replace('/','_').replace('.py','.tex')
    (BUILD/filename).write_text('\n'.join(out),encoding='utf-8')
    preamble+='\\input{'+filename[:-4]+'}\n'

remaining=[k for k in F if k not in seen_families]
(BUILD/'methods.tex').write_text('\\chapter{Additional method comparisons}\\label{ch:methods}\n'+'\n'.join(method_comparison(k) for k in remaining),encoding='utf-8')
if remaining:preamble+='\\input{methods}\n'
verification=(BUILD/'worked_verification.tex').read_text(encoding='utf-8')
verification=verification.replace('\\label{ch:expand-verification}','\\label{ch:expand-verification}\n'+report_routes.pop(('verification','metrics'),''))
(BUILD/'worked_verification.tex').write_text(verification,encoding='utf-8')
assert not report_routes,report_routes.keys()
preamble+='\\input{worked_verification}\n'
old_main=(OLD/'main.tex').read_text(encoding='utf-8-sig')
gloss=old_main.split('\\chapter{Glossary}',1)[1].split('\\input{codetheory_paths}',1)[0]
(BUILD/'glossary.tex').write_text('\\chapter{Glossary}'+gloss,encoding='utf-8')
preamble+='\\input{glossary}\n'
preamble+=r'''\chapter{Function index and source provenance}
\Q{Evidence}{Coverage and scope}
Every top-level function/class in the captured fourteen Python modules is listed
below; methods are listed inside their class entries. Nested local helpers belong
to their enclosing function. The source ZIP includes the exact captured Python
and startup files for cross-checking; the manual does not execute them.
The report equation cards cover numbered equations 1 through 35. A library
comparison is grouped by method family so small helpers need not repeat it.
\small\begin{longtable}{p{0.28\linewidth}p{0.48\linewidth}r}
\toprule Module & Function or class & Source line\\\midrule\endhead
'''
for e in coverage:preamble+=esc(e['module'])+' & \\hyperref['+label(e)+']{'+esc(e['name'])+'} & '+str(e['line'])+r'\\'+'\n'
preamble+=r'''\bottomrule\end{longtable}\normalsize
\Q{Evidence}{Unresolved sources and comparison limits}
The exact motor torque-curve points, the assumed friction coefficient, installed
sense resistance/reference voltage and GPIO current-limit provenance remain
conditional. Published algorithms do not supply optima for the project seam,
glyph-cut, drift or joining constants. Historical sweeps are identified as such;
this edition does not absorb another agent's unreviewed new experiments.
Some publisher/manufacturer pages blocked automated full-text access; those
limitations are retained in bibliography notes rather than described as a
completed reference certification. The reader should use the exact component
edition and operating conditions before adopting a hardware numerical claim.
\bibliographystyle{ieeetr}
\bibliography{refs,expansion_sources,library_sources}
\end{document}
'''
(BUILD/'main.tex').write_text(preamble,encoding='utf-8')
with (BASE/'FUNCTION_COVERAGE.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(coverage[0]));w.writeheader();w.writerows(coverage)
manifest=[]
for p in sorted(SNAP.rglob('*')):
    if p.is_file() and p.suffix in ['.py','.sh','.js','.html','.css']:
        manifest.append(dict(path=str(p.relative_to(SNAP)).replace('\\','/'),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(BASE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
# All cited keys must resolve before compilation.
alltex='\n'.join(p.read_text(encoding='utf-8') for p in BUILD.glob('*.tex'))
allbib='\n'.join(p.read_text(encoding='utf-8') for p in BUILD.glob('*.bib'))
defined=set(re.findall(r'@\w+\{([^,]+),',allbib))
cited={k.strip() for m in re.findall(r'\\cite\{([^}]+)\}',alltex) for k in m.split(',')}
# FormulaNote/ReportEq macro arguments contain citations without literal cite.
cited|={k.strip() for f in F.values() for k in f['refs'].split(',')}
missing={k for k in cited-defined if not k.startswith('#')}
print('Coverage',len(coverage),'class methods',sum(e['methods'] for e in coverage),'retained current theory',preserved,'annotated retained formulas',annotated)
print('Report equation cards',len(re.findall(r'\\ReportEq\{\d+\}',(BUILD/'report_priority.tex').read_text(encoding='utf-8'))))
print('Missing bibliography keys',sorted(missing))
if missing:raise SystemExit(1)
