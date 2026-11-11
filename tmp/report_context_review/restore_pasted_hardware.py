from pathlib import Path
import re,shutil,zipfile,json
r=Path.cwd();p=Path('C:/Users/braun/.codex/attachments/439b6e90-13d1-4e86-8b97-8780e1a2ae55/Pasted text.txt');raw=p.read_text(encoding='utf-8-sig');s=raw[raw.index('\\subsection{Hardware Design}'):raw.index('\\subsubsection{Theoretical Background}')]
cur=r/'Report/IntegratedReport_2026-11-01';old=r/'Report/PrismSnapshot_2026-10-31';out=r/'output/Hardware_Restored_From_Your_Draft_2026-11-02';out.mkdir(exist_ok=True)
s=s.replace('\\subsection{Hardware Design}','\\subsection{Hardware Design}\n\\label{sec:des-hardware}',1)
s=s.replace('concepts -> Plotter Mechanics, Stepper Motor and Driver, Pen up/down mechanics and circuitry, Calibration and Endstops circuitry, Gcode reading and pin outs etc.','')
s=re.sub(r'\\begin\{table\}\[!htbp\].*?\\label\{tab:gantry-component-dimensions\}.*?\\end\{table\}', '',s,count=1,flags=re.S)
# Small factual corrections; otherwise preserve the supplied paragraph wording.
s=s.replace('It does this through a process called chopping, which switches','The A4988 regulates the winding current through a process called chopping, which switches',1)
s=s.replace('with no encoders or feedback loops required.','with position tracked from commanded steps rather than an encoder. This assumes that the motors do not miss steps.',1)
s=s.replace('Nema 17 bipolar stepper motor, with a resolution of 400 steps per revolution, a current rating of 1.68A per phase, and a holding torque of 0.44 N.m. The pull-out torque for this motor is shown in (Reference manual and show picture for pull out torque).','42BYGH40-01 Nema 17 bipolar stepper motor, with a resolution of 200 full steps per revolution and a current rating of 1.68A per phase~\\cite{stepperds}.',1)
s=s.replace('\\subsubsection{Stepper motor and driver}','\\subsubsection{Stepper motor and driver}\n\\label{sec:des-stepper}',1)
s=s.replace('Desired speed source: Dziedzic, Sen Yilmaz & Szafrańska, "How fast is fluent handwriting?", Forensic Science International, vol. 383, art. 112896, June 2026, doi:10.1016/j.forsciint.2026.112896.',r'''Dziedzic et al. examined samples written at 10--60\,mm/s~\cite{dziedzic2026}. The original sizing check uses the upper value, rounded from 3600 to 4000\,mm/min for headroom; this is a design benchmark, not the selected operating feed. Current generated jobs use 900\,mm/min for drawing and 2400\,mm/min for travel.

For the installed 20-tooth GT2 pulleys with 2\,mm belt pitch, the transmission geometry gives~\cite{acarnley2002}
\begin{equation}
\ell_{\mathrm{rev}}=N_t p=20(2)=40\ \mathrm{mm/rev},\qquad
r_p=\frac{N_t p}{2\pi}=6.37\ \mathrm{mm}.
\label{eq:gan-radius}\label{eq:des-pulley}
\end{equation}''')
s=s.replace("The commanded linear feed is converted to a motor speed, read against the manufacturer's measured pull-out torque curve, and compared with the torque the moving mass actually demands.","The design linear feed is converted to a motor speed and compared with the torque required to move the assembly. The earlier curve readings are retained as a conditional calculation: their applicability to the installed motor and driver settings has not yet been verified.")
s=s.replace(r'n = \frac{v}{\ell_{\mathrm{rev}}}\times 60',r'n = \frac{v}{\ell_{\mathrm{rev}}}')
s=s.replace('is the commanded feed (60\\,mm/s rounded up for margin)','is the sizing feed (60\\,mm/s rounded up for margin)')
s=s.replace("where $T_{\\mathrm{po}}(n)$ is the pull-out torque available at speed $n$, found by linear interpolation between the two bracketing points read off the manufacturer's pull-out torque curve~\\cite{stepperds}:","where $T_{\\mathrm{po}}(n)$ is the conditional pull-out torque estimate at speed $n$, found by linear interpolation between the original two curve readings:")
s=re.sub(r'\\mustdo\{[^{}]*\\texttt\{stepperds\}[^{}]*\}', '',s)
s=re.sub(r'\\mustdo\{[^{}]*\}', '',s)
s=s.replace(r'0.00625 \times 0.004 \times 0.950 \times 9.81 \approx 0.029',r'0.006366 \times 0.004 \times 0.950 \times 9.81 \approx 0.0237')
a=s.index('where $T_{\\mathrm{req}}$');b=s.index('\\begin{equation}',a)
s=s[:a]+r'''where $T_{\mathrm{req}}$ is the torque needed at constant velocity, $r=6.37$\,mm is the pulley pitch radius from Equation~\ref{eq:gan-radius}, $\mu=0.004$ is the original assumed bearing-friction coefficient, $m=0.950$\,kg is the complete moving assembly and $g=9.81$\,m/s$^2$. The bearings support the weight, so the motor overcomes horizontal resistance rather than lifting the assembly. The relations $F_f=\mu mg$ and $T=rF_f$ give this steady-speed estimate~\cite{acarnley2002}.

'''+s[b:]
s=s.replace(r'\frac{16.3}{0.029} \approx 560',r'\frac{16.33}{0.02373} \approx 688')
s=s.replace('The pull-out torque available at the highest commanded speed exceeds the torque the carriage actually demands by a factor of about 560, so the drive is not torque-limited at any feed used by this machine.','Under the assumed friction and original curve values, the estimated torque ratio is about 688. This retains the sizing argument for one motor driving both sides through a shared shaft. It is not a measured operating margin and does not include acceleration, preload or misalignment.')
s=s.replace('\\label{eq:gan-treq}','\\label{eq:gan-treq}\\label{eq:des-torque}')
# Move existing pulse timing explanation into the original G-code/motor-motion subsection.
a=s.index('Equation~\\ref{eq:gan-tau}');b=s.index('\\paragraph{Coil current',a);timing=s[a:b]
s=s[:a]+s[b:]
timing=re.sub(r'There are \\emph\{no acceleration ramps\}.*?\\cite\{acarnley2002\}\.',r'There are no controlled acceleration ramps; the steady-speed torque estimate alone does not establish a safe starting rate~\\cite{acarnley2002}.',timing,flags=re.S)
timing=timing.replace('\\label{eq:gan-tau}','\\label{eq:gan-tau}\\label{eq:des-pulsetime}')
# Remove unrecorded installed-current claims while retaining the original current calculation.
a=s.index('for the 1.68\\,A rating');b=s.index('\\subsubsection{Pen up/down mechanism}',a)
s=s[:a]+r'''for the 1.68\,A motor rating~\cite{stepperds}, conditional on the reported R10 sense-resistor marking meaning $0.1\,\Omega$. This gives a rating-based ceiling, rather than the measured adjustment. For comparison, reducing the phase-current limit to 1\,A would give $V_{\mathrm{ref}}=8(0.1)(1)=0.8$\,V. Carrier cooling can require a lower current, but the actual fitted $V_{\mathrm{ref}}$ and thermal margin were not recorded.

'''+s[b:]
s=s.replace('\\subsubsection{Pen up/down mechanism}','\\subsubsection{Pen up/down mechanism}\n\\label{sec:des-pen}\\label{sec:gan-circuits}',1)
s=s.replace("The pen-lift motor's drive pulse width was found by direct testing to be 260\\,ms of rotation plus 90\\,ms of settling, 350\\,ms in total, rather than by calculation, since the cam geometry and the gearbox reduction were not modelled.","The controller runs the motor until the requested switch state is confirmed or a timeout occurs, with a 40\\,ms overrun after pen-up confirmation. The generator's separate settling dwell is 350\\,ms; this is not a measured fixed motor-rotation time.")
s=s.replace('these are supplier figures for a product variant and should be checked against the installed motor and its actual load','these are supplier figures for that variant, rather than measured currents under the installed cam load')
s=s.replace('Treat 1\\,k$\\Omega$ as the prototype\'s GPIO-current-limiting choice, and verify the installed motor\'s current and transistor temperature under the actual cam load; use a suitable logic-level MOSFET or motor driver if the motor approaches its rated or stall current.','The 1\\,k$\\Omega$ resistor is the prototype\'s GPIO-current-limiting choice. A suitable logic-level MOSFET or motor driver would avoid relying on this base-current margin if the installed motor approaches those loads.')
s=s.replace('\\cite{ucc3626ds}','\\cite{rohmlowside2026}')
a=s.index('During tuning the clicker');b=s.index('\\subsubsection{End stop switches and calibration}',a)
s=s[:a]+r'''During tuning the clicker sometimes over-rotated and the pen did not push down fully before the switch read as triggered, requiring the switch seating and timing to be adjusted experimentally. Differing pen pressure on a book or an exam pad also introduced inconsistent ink, so a rigid clipboard with a single page clamped flat was adopted. The 0.35\,s settling dwell contributes to the predicted writing time, but is separate from switch-confirmed actuation. Missed steps cannot be detected from the step counter, so physical accuracy must be assessed from the drawn output rather than nominal resolution alone.

'''+s[b:]
s=s.replace('\\subsubsection{End stop switches and calibration}','\\subsubsection{End stop switches and calibration}\n\\label{sec:des-calibration}',1)
a=s.index('All four end-stop switches');b=s.index('The DML31P was selected',a)
s=s[:a]+r'''All four axis end-stop switches are checked before every major-axis step and every G-code line. A potentially triggered switch is accepted only after three consecutive high reads 6\,ms apart (Algorithm~\ref{alg:gan-endstop}). The pen switch is read separately by the pen-control routine. The confirmation interval is
\begin{equation}
t_{\mathrm{confirm}}=(3-1)(6\,\mathrm{ms})=12\,\mathrm{ms}.
\label{eq:gan-latency}
\end{equation}
The stepping loop does not continue pulsing during this check. The interval is therefore not a measured trip-to-stop latency and cannot simply be multiplied by the requested feed to infer stopping distance.

'''+s[b:]
s=s.replace('and limits current when the switch closes.','and draws a small current when the switch closes.')
s=s.replace('no endpoint aborts a running job, and no lock prevents two simultaneous jobs.','the web stop request can interrupt a running job, but it is not an independent hardware power cut.')
s=s.replace('$194\\times252$\\,mm','$190.6\\times251.3$\\,mm')
s=s.replace('The two traverses need $(204+262)/7.8\\approx60$\\,s, derived and not measured.','Using the nominal 80 microsteps/mm, a 1.6\\,ms homing-step period corresponds to 7.8125\\,mm/s. The counted switch-to-switch traverses alone would then need $(200.6+261.3)/7.8125\\approx59.1$\\,s; this excludes the initial searches, retractions and parking, and is not a measured calibration duration.')
s=s.replace('and an abort refuses all jobs.','and a failed calibration leaves movement unavailable until calibration succeeds.')
s=s.replace('The tightest tolerance in the whole hardware design falls on that hand measurement of travel.','The hand measurement of travel also sets the scale of every subsequent movement.')
s=s.replace('over the 194\\,mm of usable X travel','over the 190.6\\,mm of usable X travel')
s=s.replace(r'\frac{0.25\ \mathrm{mm}}{194\ \mathrm{mm}}\,L=\frac{0.25}{194}\cdot204',r'\frac{0.25\ \mathrm{mm}}{190.6\ \mathrm{mm}}\,L=\frac{0.25}{190.6}\cdot200.6')
s=s.replace('Equation~\\ref{eq:gan-budget} is the tightest tolerance anywhere in the hardware design, and it falls on a measurement taken with a hand-held rule.','This is an allocation of half the positional-error allowance to scale error, not a measured uncertainty budget. Switch repeatability and mechanical compliance use part of the remaining allowance~\\cite[pp.~18--19]{jcgm2008}.')
s=s.replace('The two traverses','The counted traverses')
s=s.replace('\\subsubsection{Gcode generation and motion mapping}','\\subsubsection{Gcode generation and motion mapping}\n\\label{sec:des-drive}\nG-code stores movement and pen-state instructions in a text file. The hardware controller reads each command and its parameters, then converts the target coordinates to motor pulses. The instructions used by this plotter are explained below before the square-and-triangle commissioning example.\n',1)
s=s.replace('which is within one step of the ideal $k\\,a_y/a_x$, about 6\\,\\textmu m','which is within half a microstep of the ideal $k\\,a_y/a_x$, at most 6.25\\,\\textmu m at the nominal 80 microsteps/mm')
needle='Figure~\\ref{fig:gcode-square-triangle-comparison} compares'
index=s.index(needle)
algorithm=r'''\begin{algorithm}[!htbp]
\caption{Coordinated line movement with an integer error accumulator}
\label{alg:gan-step}
\begin{algorithmic}[1]
\REQUIRE signed target changes in microsteps
\STATE set axis directions; choose major count $N_M$ and minor count $N_m$
\STATE $e\leftarrow\lfloor N_M/2\rfloor$
\FOR{each of the $N_M$ major-axis steps}
    \IF{abort requested or an end-stop is confirmed}
        \STATE \textbf{return} interrupted
    \ENDIF
    \STATE pulse major axis; update its count; $e\leftarrow e-N_m$
    \IF{$e<0$}
        \STATE pulse minor axis; update its count; $e\leftarrow e+N_M$
    \ENDIF
\ENDFOR
\end{algorithmic}
\end{algorithm}

'''
newfig=r'''Figure~\ref{fig:vt-stepping} illustrates the commanded microsteps for a separate generated move of $(+6,-28)$ microsteps. Its maximum deviation from the ideal line is 0.5 microsteps, or $0.5/80=0.00625$\,mm at the nominal scale. This checks command-space interpolation; it does not measure the pen's physical path.
\begin{figure}[!htbp]\centering
\includegraphics[width=0.96\linewidth]{Figures/VisualTests/T52_bresenham_staircase.pdf}
\caption{Integer axis stepping and command-space deviation for a generated move.}
\label{fig:vt-stepping}
\end{figure}

'''
s=s[:index]+algorithm+newfig+timing+s[index:]
# The excerpt is the square portion of a longer job; M2 ends the full job, not that portion.
s=s.replace('G4 P0.350\nM2\n\\end{verbatim}','G4 P0.350\n\\end{verbatim}',1)
s=s.replace('The following extract shows the commands used to draw the square.','The following extract shows the commands used to draw the square; the full file then continues with the triangle before its final \\texttt{M2}.',1)
s=s.replace('The meanings of \\texttt{M3} and \\texttt{M5} are specific','Travel commands do not themselves raise the pen, and the travel feed is restated after drawing because \\texttt{F} remains active until changed. The meanings of \\texttt{M3} and \\texttt{M5} are specific',1)
# Keep external references used by the existing assembled report compatible.
for oldlabel,newlabel in [('fig:gantry-axis-diagram','fig:des-gantry-axes'),('fig:pen-up-down-mechanism','fig:des-penmechanism'),('fig:pen-clicker-bc337-circuit','fig:des-pencircuit'),('fig:endstop-switch-pulldown','fig:des-switch'),('fig:gcode-square-triangle-comparison','fig:des-shapetest'),('eq:gan-spm','eq:des-stepres'),('eq:pen-bc337-base-current','eq:des-basecurrent'),('eq:gan-current','eq:des-current'),('eq:gan-cal','eq:des-calibration'),('alg:gan-step','alg:des-stepping')]:
 s=s.replace('\\label{'+oldlabel+'}','\\label{'+oldlabel+'}\\label{'+newlabel+'}',1)
s=s.replace('\\label{sec:gan-circuits}','\\label{sec:gan-circuits}\\label{sec:des-power}',1)
s=s.replace('\\subsubsection{Plotter designs and mechanics}','\\subsubsection{Plotter designs and mechanics}\n\\label{sec:des-camera}',1)
s+='\n\\FloatBarrier\n'
# Only remove TeX editorial comments, not the user's prose or pseudocode comments.
s=re.sub(r'(?<!\\)%[^\n]*','',s);s=re.sub(r'\n{4,}','\n\n\n',s)
(out/'3_3_hardware_restored.tex').write_text(s,encoding='utf-8')
# Build an accompanying image/bibliography bundle; the assembled report is untouched.
for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',s):
 src=cur/name
 if not src.exists():src=old/name
 assert src.exists(),name
 dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
bib=(r/'output/Hardware_Revision_2026-11-02/references.bib').read_text(encoding='utf-8');oldbib=(old/'references.bib').read_text(encoding='utf-8')
used={k for g in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}',s) for k in g.split(',')};keys=set(re.findall(r'@\w+\{([^,]+)',bib))
for k in sorted(used-keys):
 m=re.search(r'@\w+\{'+re.escape(k)+r',.*?(?=\n@|\Z)',oldbib,re.S);assert m,k;entry=m.group(0)
 if k=='stepperds':entry='''@misc{stepperds,
 author={{Communica}},
 title={BMT stepper motor 40mm NEMA 17, model 42BYGH40-01},
 howpublished={Product specification},
 url={https://www.communica.co.za/products/bmt-stepper-motor-40mm-nema-17}}
'''
 bib+='\n'+entry
(out/'references.bib').write_text(bib,encoding='utf-8')
refs=set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',s));labels=set(re.findall(r'\\label\{([^}]+)\}',s));assert refs<=labels,refs-labels
from collections import Counter
assert Counter(re.findall(r'\\begin\{([^}]+)\}',s))==Counter(re.findall(r'\\end\{([^}]+)\}',s))
assert '\\mustdo' not in s and 'eq:gan-margin-eq' not in s
notes='''# Restoration notes

Baseline: the author's newly pasted hardware section, from Hardware Design through the G-code command table. The six original subsection headings, CAD panels, motor photographs, speed-to-torque sequence, pen and switch circuits, pen/end-stop/calibration algorithms, calibration explanation, G-code extract and command table are restored. The newer command-space stepping figure is retained in the G-code subsection. The assembled report has not been modified.

Corrections retained: 200 rather than 400 full steps; chopping distinguished from stepping; dimensional speed conversion; correct 6.37 mm pulley pitch radius, 0.0237 N cm estimated load and conditional ratio approximately 688; unverified curve and friction identified as assumptions; actual Vref and driver derating not invented; switch-confirmed pen actuation distinguished from generator dwell; axis switches distinguished from the pen switch; debounce interval distinguished from stopping distance; actual workspace 190.6 x 251.3 mm; counted homing legs distinguished from total calibration time; software stop endpoint acknowledged; broken labels repaired and missing stepping algorithm supplied. Removed the blank CAD dimensions table and unfinished drafting notes; no dimensions were invented.

The matching torque-speed curve is still needed to verify the inherited 7 and 21 N cm curve readings. No assertion that the motor is never torque-limited is made. The new diagram in the report's separate hardware overview remains unchanged.
'''
(out/'RESTORATION_NOTES.md').write_text(notes,encoding='utf-8')
with zipfile.ZipFile(r/'output/Hardware_Restored_From_Your_Draft_2026-11-02.zip','w',zipfile.ZIP_DEFLATED) as z:
 for q in sorted(out.rglob('*')):
  if q.is_file():z.write(q,q.relative_to(out).as_posix())
print('Restored source:',out/'3_3_hardware_restored.tex');print('All internal references and environments checked.');print('Original headings:',re.findall(r'\\subsubsection\{([^}]+)\}',s))
