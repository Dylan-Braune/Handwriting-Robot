from pathlib import Path
import re,json
R=Path(__file__).resolve().parents[2];B=R/'Report/IntegratedReport_2026-11-01'
changes={
'3_Main_Report/design/3_3_hardware.tex':{
 'For the 20-tooth GT2 pulleys and 2\\,mm belt pitch, the travel per revolution and pitch radius follow from Equation~\\ref{eq:des-pulley}.':'Using the stepper transmission relationship~\\cite{acarnley2002}, Equation~\\ref{eq:des-pulley} derives the travel per revolution and pitch radius for the installed 20-tooth GT2 pulleys and 2\\,mm belt pitch.',
 'Here $N_t$ is the tooth count and $p$ is belt pitch. With 200 full steps per revolution and a microstep subdivision $\\mu_s=16$, the nominal commanded displacement is given by Equation~\\ref{eq:des-stepres}.':'Here $N_t$ is the tooth count and $p$ is belt pitch. The A4988 supports a microstep subdivision of $\\mu_s=16$~\\cite[p.~6]{a4988ds}. Combining this with 200 full steps per revolution gives the nominal displacement in Equation~\\ref{eq:des-stepres}.',
 'The load model in Equation~\\ref{eq:des-torque} separates translation and rotation.':'The load model in Equation~\\ref{eq:des-torque} combines the standard linear-force and rotary-inertia relationships for a stepper-driven load~\\cite{acarnley2002}, separating translation and rotation.',
 'its first-order contribution to position error at commanded distance $x$ is given by Equation~\\ref{eq:des-scaleerror}.':'its first-order contribution to position error at commanded distance $x$ follows from the sensitivity-coefficient approach~\\cite[pp.~18--19]{jcgm2008}, giving Equation~\\ref{eq:des-scaleerror}.',
 'the physical connections should be checked against the finished schematic before wiring or submission. The 3.3\\,V header pin and ground pins actually used, and the common return between the board and motor drivers, remain to be confirmed.':'the table therefore identifies the configured signals. The retained record does not identify the physical 3.3\\,V and ground header connections or the common-return routing.',
 'Reported 10\\,k$\\Omega$ pull-down; verify its fitted location and return.':'Reported 10\\,k$\\Omega$ pull-down.',
 'Confirm the switch model and pull-down arrangement from the installed circuit.':'Switch model and pull-down location are not recorded.',
 'A 10\\,k$\\Omega$ base pull-down was reported; verify its connection.':'A 10\\,k$\\Omega$ base pull-down was reported; its connection is not documented.',
 'Actual physical 3.3\\,V header pin and total rail loading must be confirmed on the installed harness.':'Physical header pin and total rail loading are not recorded.',
 'A common signal reference is required; exact ground topology and physical GND pin are to be checked against the circuit.':'A common signal reference is required; exact return topology and physical GND pin are not documented.',
 'The reported resistor marking is tentatively R10 and must be checked before applying that bound.':'The bound is conditional on the reported R10 resistor marking.',
 'its DC output rating remains to be confirmed separately from the 24\\,V stepper supply.':'its DC output rating is not recorded and is distinct from the 24\\,V stepper supply.',
 'so the saved frame dimensions must be checked rather than equating sensor specification with operating resolution.':'so the sensor specification alone does not establish operating resolution.',
},
'3_Main_Report/design/3_6_reproduction.tex':{
 ' Figure~\\ref{fig:des-joins} relates endpoint selection to the resulting paths, including a pair for which a connector is unsuitable.':'',
 'Equation~\\ref{eq:des-glyphcoords} first expresses each path point':'Slant normalisation uses the shear principle described for cursive writing~\\cite{vinciarelli2001}; the following coordinate convention is defined for this project. Equation~\\ref{eq:des-glyphcoords} first expresses each path point',
 'For this profile representation, connectedness is defined as':'Building on the use of textural and shape features for writer characterisation~\\cite{bulacu2007}, this project defines connectedness as',
},
'3_Main_Report/design/3_5_classification.tex':{
 'Equation~\\ref{eq:des-conv} writes the layer operation explicitly for the numerical implementation.':'The local weighted-sum operation follows the CNN formulation~\\cite[pp.~2280--2283]{lecun1998}; Equation~\\ref{eq:des-conv} writes it explicitly for the numerical implementation.',
},
'Appendices/design_implementation_details.tex':{
 'For a linear layer $Y=XW^T+b$ with incoming derivative $G$,':'Applying the backpropagation chain rule~\\cite{lecun1998} to a linear layer $Y=XW^T+b$ with incoming derivative $G$,',
 'Before the writer update, all gradient tensors are scaled together':'Following global gradient-norm clipping~\\cite{pascanu2013}, before the writer update all gradient tensors are scaled together',
 'The finite-difference reference is derived by subtracting':'Numerical perturbation provides a finite-difference sensitivity check~\\cite[pp.~18--19]{jcgm2008}. For the loss function here, the central-difference expression is derived by subtracting',
},
'3_Main_Report/4_Results.tex':{
 'whose make, model and least count remain to be recorded':'whose make, model and least count are not present in the retained record',
 'ground-truth counts and any acquisition latency outside the timer must be checked when repeating the full request-to-result test specified in the proposal.':'independent ground-truth counts and any acquisition latency outside the timer are not established by those records.',
},
}
for name,replacements in changes.items():
 p=B/name;t=p.read_text(encoding='utf-8')
 for old,new in replacements.items():
  if old not in t:print('UNMATCHED:',name,old[:100])
  t=t.replace(old,new)
 p.write_text(t,encoding='utf-8')

# Adjacent citations with identical locators become one simple citation.
for p in [*B.glob('3_Main_Report/**/*.tex'),*B.glob('Appendices/*.tex')]:
 t=p.read_text(encoding='utf-8')
 t=re.sub(r'\\cite(\[[^\]]*\])?\{([^}]+)\}\\cite\1\{([^}]+)\}',lambda m:'\\cite'+(m[1] or '')+'{'+m[2]+','+m[3]+'}',t)
 p.write_text(t,encoding='utf-8')

# One bibliography record per cited source; no duplicate title/DOI aliases.
p=B/'references.bib';bib=p.read_text(encoding='utf-8')
entries=[]
for m in re.finditer(r'@\w+\{\s*([^,]+),',bib):
 start=m.start();pos=bib.index('{',start)+1;depth=1
 while depth:
  if bib[pos]=='{' and bib[pos-1]!='\\':depth+=1
  if bib[pos]=='}' and bib[pos-1]!='\\':depth-=1
  pos+=1
 entries.append((m[1],bib[start:pos]))
alltext='\n'.join(p.read_text(encoding='utf-8') for p in [*B.glob('3_Main_Report/**/*.tex'),*B.glob('Appendices/*.tex')])
cited={key for s in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}',alltext) for key in s.split(',')}
assert not cited-{key for key,_ in entries},cited-{key for key,_ in entries}
kept=[(key,t) for key,t in entries if key in cited]
assert len({key for key,_ in kept})==len(kept)
p.write_text('\n\n'.join(t for _,t in kept)+'\n',encoding='utf-8')
print('Cited bibliography records:',len(kept))
