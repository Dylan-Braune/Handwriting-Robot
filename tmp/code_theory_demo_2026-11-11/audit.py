import ast, re, json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
OLD=ROOT/'Software/CleanPipeline2/testing/results/CodeTheory'
SNAP=BASE/'source_snapshot'
modules={str(p.relative_to(SNAP)).replace('\\','/'):ast.parse(p.read_text(encoding='utf-8-sig')) for p in SNAP.rglob('*.py')}
def defs(t): return {n.name:n for n in t.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
inventory=[]; oldblocks={}; changes=[]; formulae=[]
for p,t in modules.items():
    text=(SNAP/p).read_text(encoding='utf-8-sig'); lines=text.splitlines()
    core=Path(p).stem
    prev=ROOT/'Software/CleanPipeline2'/p
    previous=defs(ast.parse(prev.read_text(encoding='utf-8-sig'))) if prev.exists() else {}
    docname='synth' if core=='synthesise' else core
    doc=OLD/f'codetheory_{docname}.tex'
    blocks={}
    if doc.exists():
        parts=re.split(r'(?=\\section\{)',doc.read_text(encoding='utf-8-sig'))
        for b in parts[1:]:
            name=re.match(r'\\section\{([^}]*)\}',b).group(1).replace(r'\_','_')
            blocks[name]=b
    oldblocks[p]=blocks
    for name,n in defs(t).items():
        sig=('class '+name) if isinstance(n,ast.ClassDef) else ('def '+name+'('+ast.unparse(n.args)+')')
        current_source='\n'.join(lines[n.lineno-1:n.end_lineno])
        same=name in previous and ast.dump(n,include_attributes=False)==ast.dump(previous[name],include_attributes=False)
        comment=[]; k=n.lineno-2
        while k>=0 and (not lines[k].strip() or lines[k].lstrip().startswith('#')):
            if lines[k].lstrip().startswith('#') and not set(lines[k].strip())<=set('#- '):comment.append(lines[k].lstrip('# ').strip())
            k-=1
        entry=dict(module=p,name=name,line=n.lineno,end=n.end_lineno,signature=sig,source=current_source,same_old=same,comment=' '.join(reversed(comment)),doc=blocks.get(name),methods=[])
        if isinstance(n,ast.ClassDef):
            entry['methods']=[dict(name=m.name,line=m.lineno,signature='def '+m.name+'('+ast.unparse(m.args)+')') for m in n.body if isinstance(m,ast.FunctionDef)]
        inventory.append(entry)
        if not same: changes.append(f'{p}:{n.lineno} {name}')
        if name in blocks:
            for j,m in enumerate(re.finditer(r'\\begin\{(?:equation\*?|align\*?|gather\*?)\}.*?\\end\{(?:equation\*?|align\*?|gather\*?)\}|\\\[.*?\\\]',blocks[name],re.S)):
                formulae.append(dict(module=p,name=name,index=j+1,same=same,formula=m.group(0),context=blocks[name][max(0,m.start()-240):m.start()]))
conflicts=[]
for module,t in modules.items():
    aliases={}
    for n in t.body:
        if isinstance(n,ast.Import):
            for a in n.names:
                possible=a.name.replace('.','/')+'.py'
                if possible not in modules: possible='web/'+possible
                if possible in modules:aliases[a.asname or a.name]=possible
    for n in ast.walk(t):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id in aliases:
            target_module=aliases[n.func.value.id]; target=defs(modules[target_module]).get(n.func.attr)
            if isinstance(target,ast.FunctionDef) and target.args.kwarg is None:
                allowed={a.arg for a in target.args.args+target.args.kwonlyargs}
                bad=[k.arg for k in n.keywords if k.arg is not None and k.arg not in allowed]
                if bad:conflicts.append(dict(module=module,line=n.lineno,target=target_module+':'+target.name,unsupported=bad))
data=dict(inventory=inventory,conflicts=conflicts,formulae=formulae)
(BASE/'audit.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
(BASE/'formula_review.txt').write_text('\n\n'.join(f"{f['module']} {f['name']} [{f['index']}] same={f['same']}\n{f['context']}\n{f['formula']}" for f in formulae),encoding='utf-8')
print('Top-level callables',len(inventory),'class methods',sum(len(e['methods']) for e in inventory),'unchanged',sum(e['same_old'] for e in inventory),'display formulas',len(formulae))
print('STATIC KEYWORD CONFLICTS',json.dumps(conflicts,indent=2))
print('Changed core functions:', '\n'.join(c for c in changes if not c.startswith(('web/','train.py','curate.py','timing.py','retrace.py'))))
