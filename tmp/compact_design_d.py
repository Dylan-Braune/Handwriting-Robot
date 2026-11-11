from pathlib import Path
exec(Path('tmp/compact_design_a.py').read_text().split('# Segmentation.')[0])
o=source('3_7_integration.tex');collect(o,['Applying this vote'],'System integration calculation')
s=r'''\subsection{System integration and embedded operation}
\label{sec:des-integration}
The server connects page processing, stored models, writer profiles and the motion interpreter. The browser supplies requests and displays results.

\subsubsection{Classification and reproduction workflows}
Captured and uploaded pages use the same preparation and recognition path. For each writer $k$, page pooling combines the line probabilities $p_{i,k}$ with ink weights $w_i$.
'''+eq(o,'eq:des-pagevote')+r'''Each weight counts pixels below grey value 128. Empty weights fall back to equal weighting. Crop rejection precedes pooling so that a dark artifact does not gain excessive influence. The two-line Owen calculation is retained in Appendix~\ref{app:worked-design-calculations}.

A writing request stores its trajectory, preview, G-code and metadata as one generated job. Drawing is a separate action requiring connection and calibration. This allows the operator to inspect the path before movement. Table~\ref{tab:des-contracts} records the interfaces and coordinate conventions.

\subsubsection{Embedded execution, interfaces and job control}
The ODROID runs numerical inference and motion control. NumPy provides array operations, Flask handles requests, image libraries load files, OpenCV captures camera frames and gpiod controls GPIO. The source code is supplied separately with the project.

The pulse loop checks an abort event and confirmed end stops. A separate request thread allows a stop during drawing. End-stop trips latch a fault and invoke recovery under the configured conditions. Generated, refused, completed and interrupted jobs are recorded separately. The software stop does not independently disconnect motor power.

\subsubsection{Integration verification and remaining constraints}
Interface checks cover upload and camera consistency, writer-to-profile mapping, bounded G-code and interrupted-job reporting. The diagnostic retracing path bypasses recognition and synthesis, using thinning and tracing followed by ordinary machine conversion. Comparing its preview and drawing helps locate whether an error enters before or during execution.

Stage records separate image processing, inference, generation and drawing. Generation timing is carried into the reproduction total. Accuracy and time requirements are evaluated in Results using the relevant image populations and ODROID records.
\FloatBarrier
'''
(B/'3_Main_Report/design/3_7_integration.tex').write_text(s,encoding='utf-8')
x=json.loads((R/'tmp/final_report_review/moved_calculations.json').read_text());x.extend(moved);(R/'tmp/final_report_review/moved_calculations.json').write_text(json.dumps(x,indent=2),encoding='utf-8')
# Hardware wording remains from the submitted draft. Shorten only worked calculations and restore/explain figures.
p=B/'3_Main_Report/design/3_3_hardware.tex';s=p.read_text(encoding='utf-8')
# Keep a full copy of calculation expansions before replacing their prose.
calc=[]
for start in ['where $T_{\\mathrm{req}}$','This keeps the GPIO current','Using the nominal 80 microsteps','The hand measurement of travel']:
 i=s.find(start)
 if i>=0:
  j=s.find('\n\n',i);calc.append(('Hardware calculation detail',s[i:j]))
i=s.find('This keeps the GPIO current');j=s.find('\n\n',i)
if i>=0:s=s[:i]+r'''The base current is much lower than the motor current. A forced gain of 10, as used in the BC337 saturation test~\cite{bc337ds}, gives a conservative 26\,mA collector-current check. This is below the listed 30\,mA variant rating, so saturation at startup is not established by the base-resistor calculation alone. The full rated- and stall-current checks are retained in Appendix~\ref{app:worked-design-calculations}.'''+s[j:]
# The scale-error allocation belongs with extended calculations rather than the core calibration explanation.
a=s.find('The hand measurement of travel');z=s.find('A mechanical switch does not change state cleanly',a)
if a>=0 and z>=0:
 x=s[a:z].strip();calc.append(('Hardware scale-error allocation',x));s=s[:a]+r'''Travel measurement error also affects the calibrated scale. The numerical allocation of the positional-error allowance is retained in Appendix~\ref{app:worked-design-calculations}.

'''+s[z:]
s=s.replace('The following diagram explains axes of the gantry:',r'Figure~\ref{fig:gantry-axis-diagram} shows the fixed rails guiding $y$ movement, the moving $x$ rail and the separate pen lift.')
# Explain CAD panels with identifiers and a short functional table. No dimensions are invented.
needle='\\begin{figure}[!htbp]';idx=s.index('\\label{fig:gantry-cad-components}');beg=s.rfind(needle,0,idx)
intro=r'''Figure~\ref{fig:gantry-cad-components} shows the custom CAD parts. Their shapes locate the pen, drive components and rail bearings while leaving mounting access. Table~\ref{tab:des-cad-functions} relates each panel group to its mechanical purpose.

'''
s=s[:beg]+intro+s[beg:]
for k in range(1,9):
 pat=r'(\\includegraphics\[[^\]]+\]\{Figures/cad-component-'+f'{k:02d}'+r'\.png\})'
 s=re.sub(pat,lambda m:m.group(1)+'\\par('+chr(96+k)+')',s,count=1)
idx=s.index('\\end{figure}',s.index('\\label{fig:gantry-cad-components}'))+len('\\end{figure}')
table=r'''

\begin{table}[!htbp]\centering\small
\begin{tabularx}{\linewidth}{lX}\toprule
\textbf{CAD panels}&\textbf{Mechanical purpose}\\\midrule
(a) to (c)&Cam and pen mounting pieces provide lift and hold the pen relative to its actuator.\\
(d) to (e)&Support brackets locate the rotating drive components and their mounting fasteners.\\
(f)&The toothed belt grip secures the belt to the moving assembly.\\
(g) to (h)&Bearing and rod supports locate the rails and guide the moving parts.\\\bottomrule
\end{tabularx}
\caption{Functions of the CAD parts.}\label{tab:des-cad-functions}
\end{table}
'''
s=s[:idx]+table+s[idx:]
for label,text in [('fig:stepper-motor-and-driver',r'Figure~\ref{fig:stepper-motor-and-driver} shows the selected motor and its A4988 driver.'),('fig:pen-up-down-mechanism',r'Figure~\ref{fig:pen-up-down-mechanism} shows the cam holding the pen in its two positions.')]:
 i=s.index('\\label{'+label+'}');j=s.rfind('\\begin{figure}',0,i);s=s[:j]+text+'\n\n'+s[j:]
p.write_text(s,encoding='utf-8')
x=json.loads((R/'tmp/final_report_review/moved_calculations.json').read_text());x.extend(calc);(R/'tmp/final_report_review/moved_calculations.json').write_text(json.dumps(x,indent=2),encoding='utf-8')
# Two concise section introductions, no heading changes.
p=B/'3_Main_Report/3_Design_and_Implementation.tex';s=p.read_text(encoding='utf-8');s=s[s.index('\\label{sec:design}'):];first=s.index('\\input{');p.write_text('\\label{sec:design}\nThe design follows the hardware, page processing, classification and reproduction stages, then their integration.\n\n'+s[first:],encoding='utf-8')
p=B/'3_Main_Report/design/3_1_summary.tex';s=p.read_text(encoding='utf-8');s=s.replace('NumPy calculations and framework training; selected weights are framework-trained.','The numerical and framework routes train the same network architectures.');s=s.replace('Completion refers to the implementation of a task; compliance with a numerical system specification is established separately by the qualification tests in the Results section.','The qualification tests in Results assess numerical compliance.');p.write_text(s,encoding='utf-8')
print('Integration shortened. Hardware changes limited to calculations, CAD explanation and figure references.')
