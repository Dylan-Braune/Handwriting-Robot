from pathlib import Path
import re,json,shutil
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'Report/DesignPolished_2026-11-01'
backup=ROOT/'tmp/report_context_review/before_research_citation_revision'
assert not backup.exists(), 'Preserve the original backup; do not rerun.'
backup.mkdir()
for folder in ['3_Main_Report','Appendices']:
    shutil.copytree(B/folder,backup/folder)
for name in ['references.bib','SOURCE_AUDIT.md','SOURCE_AUDIT_COUNTS.json','EQUATION_AUDIT.json','VALIDATION.json','FULL_REVIEW.md']:
    shutil.copy2(B/name,backup/name)

internal={'braunepipeline2026','brauneevidence2026'}
removed=[]
pattern=re.compile(r'(?P<glue>~| )?\\cite(?:\[[^\]]*\])?\{(?P<keys>[^{}]*)\}')
files=[*sorted((B/'3_Main_Report').rglob('*.tex')),*sorted((B/'Appendices').glob('*.tex'))]
for p in files:
    def strip(m):
        keys=[k.strip() for k in m['keys'].split(',')]
        if set(keys)&internal:
            assert set(keys)<=internal,m.group()
            removed.append({'file':p.relative_to(B).as_posix(),'citation':m.group().strip()})
            return ''
        return m.group()
    s=pattern.sub(strip,p.read_text(encoding='utf-8'))
    s=re.sub(r' +\n','\n',s)
    s=re.sub(r'\n{4,}','\n\n\n',s)
    p.write_text(s,encoding='utf-8')

def edit(name,old,new):
    p=B/name;s=p.read_text(encoding='utf-8')
    assert s.count(old)==1,(name,old,s.count(old))
    p.write_text(s.replace(old,new),encoding='utf-8')

edit('3_Main_Report/3_Design_and_Implementation.tex',
     "Implementation-specific equations, parameter values and pseudocode describe the project's CleanPipeline3 source; borrowed methods are cited at their point of use. Worked numerical examples are derived here unless identified as recorded measurements.",
     "Established methods are referenced where introduced. The coordinate transforms, geometric heuristics and combined selection criteria are design choices developed for this system; their equations and worked examples are derived here. Numerical examples are calculations unless explicitly identified as recorded measurements. Implementation details are available in the submitted CleanPipeline3 source, with occasional module pointers provided below.")

seg='3_Main_Report/design/3_4_segmentation.tex'
edit(seg,'A single threshold applied directly to brightness can classify a shadowed part of the page as ink while losing pale strokes elsewhere. Equation',
     r'A single threshold applied directly to brightness can classify a shadowed part of the page as ink while losing pale strokes elsewhere. Local-mean comparison addresses this illumination problem in adaptive thresholding~\cite[PDF pp.~2--3, Secs.~2.2--3]{bradley2007}. The present design instead forms a local brightness ratio and selects its threshold from the ratio histogram. Equation')
edit(seg,r'The mean is evaluated through a summed-area table~\cite[page to verify]{crow1984}\designsourceflag{crow1984}.',
     r'The mean is evaluated through a summed-area table, using the four-corner rectangle sum described by Bradley and Roth~\cite[PDF p.~2, Eq.~(2)]{bradley2007}.')
edit(seg,'Here $\\widetilde M$ is the mask after a closing operation,',
     r'Here $\widetilde M$ is the mask after morphological closing, a dilation followed by erosion that fills narrow breaks and gaps~\cite[p.~541, Def.~36]{haralick1987},')
edit(seg,'The cost in Equation~\\ref{eq:des-seamcost} strongly discourages crossing ink and weakly attracts the boundary to the space between line centres.',
     r'The project-specific cost in Equation~\ref{eq:des-seamcost} combines an ink-crossing penalty with a weaker attraction to the space between line centres. These terms adapt the minimum-cost seam principle to the detected text-line gaps; the weights are engineering choices examined by the sensitivity check below.')

net='3_Main_Report/design/3_5_classification.tex'
edit(net,'The layer operation in Equation~\\ref{eq:des-conv} shows what the numerical implementation evaluates.',
     r'The convolutional feature extraction follows the reference architecture~\cite[pp.~4--5]{kizilirmak2023}. Equation~\ref{eq:des-conv} writes the layer operation explicitly for the numerical implementation.')
edit(net,'The initial classifier operated on handwriting as photographed and performed well on those samples. However, such images contain more than letter shape: pen width, darkness and the capture process can also distinguish writers.',
     r'The initial classifier operated on handwriting as photographed and performed well on those samples. Ink width can itself carry writer-identification information: Brink et al. explicitly use directional ink-trace width as a feature and relate it to the pen and writing style~\cite[p.~162]{brink2012}. The photographed input also retains darkness and capture variation, which the initial network was not constrained to ignore.')
edit(net,r'then expanded to a controlled width before the fixed-canvas fit. Equation~\ref{eq:des-strokenorm} describes the implemented transformation.',
     r'then expanded using morphological dilation~\cite[p.~533, Def.~1]{haralick1987} before the fixed-canvas fit. Equation~\ref{eq:des-strokenorm} combines these established operations with the project-selected width rule.')
edit(net,r'The implementation produces $T=160$ frames.',
     r'The CTC collapse and best-path decoding rules follow Graves et al.~\cite[PDF pp.~2--3]{graves2006}; the implementation produces $T=160$ frames.')
edit(net,r'The later classifier uses stroke-normalised input.',
     r'The later classifier uses stroke-normalised input to restrict this source of variation.')

rep='3_Main_Report/design/3_6_reproduction.tex'
edit(rep,'The chosen representation is a library of normalised glyph trajectories accompanied by line-level style measurements.',
     r'The chosen representation is a library of normalised glyph trajectories accompanied by line-level style measurements. Profile extraction and trajectory assembly are implemented in \texttt{style.py} and \texttt{synthesise.py}, respectively.')
edit(rep,'The extraction input is a line image and its known transcript. Character boundaries are not supplied as labels.',
     r'The extraction input is a line image and its known transcript. Character boundaries are not supplied as labels. The alignment builds on the blank-interleaved CTC lattice~\cite[PDF pp.~4--5]{graves2006}.')
edit(rep,r'Equation~\ref{eq:des-cutcost} describes the principal column cost in \texttt{ExtractLineGlyphs}.',
     r'Equation~\ref{eq:des-cutcost} defines the project-specific column cost used for this refinement. It combines low ink density with a preference for cutting a shallow connector near the baseline.')
edit(rep,'The coordinate frame separates source-image size, baseline and slant from the geometry to be stored.',
     'The project uses the following coordinate transform to separate source-image size, baseline and slant from the geometry to be stored.')
edit(rep,r'Connectedness is measured using Equation~\ref{eq:des-connectedness}.',
     r'For this profile representation, connectedness is defined as the fraction of eligible adjacent-letter boundaries crossed by a source skeleton, as shown in Equation~\ref{eq:des-connectedness}.')
edit(rep,r'Equation~\ref{eq:des-placement} expresses the main placement operation before machine-coordinate conversion.',
     r'Equation~\ref{eq:des-placement} is the project\textquotesingle s inverse placement transform: it reapplies slant and physical scale to the stored glyph coordinates before machine-coordinate conversion.')
edit(rep,r'The score in Equation~\ref{eq:des-jointscore} combines text recovery and the requested writer\textquotesingle s probability.',
     r'The project-defined score in Equation~\ref{eq:des-jointscore} combines text recovery and the requested writer\textquotesingle s probability.') if False else None
edit(rep,"The score in Equation~\\ref{eq:des-jointscore} combines text recovery and the requested writer's probability.",
     r'The project-defined score in Equation~\ref{eq:des-jointscore} combines text recovery and the requested writer\textquotesingle s probability. It uses the insertion, deletion and substitution error convention of Graves et al.~\cite[PDF p.~2, Sec.~2.1]{graves2006}, then combines the resulting character score with the writer posterior using a harmonic mean.')
edit(rep,r'Equation~\ref{eq:des-timebudget} expresses a nominal job budget from the generated geometry.',
     r'Equation~\ref{eq:des-timebudget} is a dimensional calculation of the nominal job budget: distance divided by feed gives travel time, and pen-state changes add their configured dwells.')

app='Appendices/design_implementation_details.tex'
edit(app,'Closing applies dilation followed by erosion with the same footprint.',
     r'Closing applies dilation followed by erosion with the same footprint~\cite[pp.~533, 541, Defs.~1, 36]{haralick1987}. For a binary mask and symmetric flat footprint, the set operations correspond to neighbourhood maximum and minimum.')
edit(app,r'The following pseudocode records the detailed recurrences and update sequence.',
     r'The following pseudocode records the detailed recurrences and update sequence. Algorithm~\ref{alg:des-ctc} adapts the CTC forward--backward method~\cite[PDF pp.~4--6]{graves2006}; Algorithm~\ref{alg:des-train} assembles the project\textquotesingle s training operations.')
edit(app,r'For one image, let $X_c$ contain flattened input patches, so the convolution becomes $Y=WX_c+b$. With output derivative $G$, Equation~\ref{eq:des-convback} supplies the two matrix derivatives used by the numerical route.',
     r'For one image, let $X_c$ contain flattened input patches, so the matrix formulation of convolution gives $Y=WX_c+b$~\cite[page to verify]{chellapilla2006}\designsourceflag{chellapilla2006}. Applying the chain rule to $Y_{oj}=\sum_k W_{ok}(X_c)_{kj}+b_o$ gives $\partial\mathcal L/\partial W_{ok}=\sum_j G_{oj}(X_c)_{kj}$ and $\partial\mathcal L/\partial(X_c)_{kj}=\sum_o W_{ok}G_{oj}$. Equation~\ref{eq:des-convback} collects these derived sums in matrix form.')
edit(app,'The finite-difference reference uses Equation~\\ref{eq:des-gradcheck}, with one logit or parameter perturbed at a time.',
     r'The finite-difference reference is derived by subtracting the two Taylor expansions $\mathcal L(\theta\pm\varepsilon e_i)=\mathcal L(\theta)\pm\varepsilon\partial_i\mathcal L+\tfrac12\varepsilon^2\partial_{ii}\mathcal L+O(\varepsilon^3)$. Division by $2\varepsilon$ cancels the constant and even terms, giving Equation~\ref{eq:des-gradcheck} with truncation error $O(\varepsilon^2)$ where the loss is sufficiently smooth. One logit or parameter is perturbed at a time.')

# Project evidence needs provenance, not a repeated self-citation after each result.
evidence='Appendices/design_subsystem_evidence.tex'
edit(evidence,'Diagnostic settings are stated explicitly rather than inherited silently from server defaults.',
     r'Diagnostic settings are stated explicitly rather than inherited silently from server defaults. The results in this appendix are project experiments. Their inputs, settings and numerical outputs are recorded in \texttt{Evidence/bench.json}, \texttt{Evidence/camera\_bench.json} and \texttt{Evidence/coefficient\_sensitivity.json}; these manifests and the evidence scripts accompany the report source.')

# Remove the two self-reference bibliography entries. Keep raw provenance files in the package.
p=B/'references.bib';s=p.read_text(encoding='utf-8')
for key in sorted(internal):
    start=s.index('@misc{'+key+',');m=re.search(r'\n@\w+\{',s[start+1:])
    end=start+1+m.start() if m else len(s)
    s=s[:start]+s[end:]
for key,url in [('bradley2007','https://people.scs.carleton.ca/~roth/iit-publications-iti/docs/gerh-50002.pdf'),('brink2012','https://www.ai.rug.nl/~mbulacu/brink-quill-pr2012.pdf'),('haralick1987','https://www.haralick.org/journals/04767941.pdf')]:
    pattern_entry=re.compile(r'(@\w+\{'+key+r',[\s\S]*?)(\}\s*\n(?=\s*@))')
    m=pattern_entry.search(s);assert m,key
    s=s[:m.start()]+m[1].rstrip()+',\n  url={'+url+'}'+m[2]+s[m.end():]
p.write_text(s,encoding='utf-8')
(B/'CITATION_REVISION.json').write_text(json.dumps({'removed_internal_citation_sites':len(removed),'removed':removed,'policy':'Research supports borrowed methods. Own algorithms, transforms, parameter choices and measurements are identified as project work; optional module pointers are prose, not numbered references.'},indent=2)+'\n',encoding='utf-8')
print('Removed',len(removed),'internal citation sites; retained and strengthened research citations.')
