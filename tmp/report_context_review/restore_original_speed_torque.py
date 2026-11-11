from pathlib import Path
import shutil,zipfile,re,math,json
r=Path.cwd();o=r/'output/Hardware_Revision_2026-11-02';p=o/'3_3_hardware.tex';s=p.read_text(encoding='utf-8')
start=s.index('\\paragraph{Driving the 950');end=s.index('\\subsubsection{Step resolution, microstepping and calibration}',start)
block=r'''\paragraph{Speed-to-torque margin check.}
The following calculations retain the original sizing check for one motor driving the 950\,g moving assembly. Dziedzic et al. studied handwritten samples produced at 10--60\,mm/s~\cite{dziedzic2026}. The upper value provides the original speed benchmark: $60(60)=3600$\,mm/min, rounded upwards to 4000\,mm/min for the motor-sizing check. This is design headroom, rather than the selected drawing feed. The current generated jobs use 900\,mm/min for drawing and 2400\,mm/min for travel.

The installed 20-tooth GT2 pulley advances the belt by $20(2)=40$\,mm per revolution. Converting the design feed to shaft speed gives~\cite{acarnley2002}
\begin{equation}
n=\frac{F}{\ell_{\mathrm{rev}}}
=\frac{4000\ \mathrm{mm/min}}{40\ \mathrm{mm/rev}}
=100\ \mathrm{rev/min}.
\label{eq:gan-rpm}
\end{equation}
At the selected travel and drawing feeds, the corresponding speeds are $2400/40=60$\,rev/min and $900/40=22.5$\,rev/min. The torque comparison is therefore made at the higher 100\,rev/min sizing point, rather than using holding torque alone.

The original check used the curve points $(n_1,T_1)=(60\,\mathrm{rev/min},7\,\mathrm{N\,cm})$ and $(n_2,T_2)=(120\,\mathrm{rev/min},21\,\mathrm{N\,cm})$. Retaining those earlier curve readings provisionally, linear interpolation gives
\begin{equation}
T_{\mathrm{po}}(100)=T_1+\frac{100-n_1}{n_2-n_1}(T_2-T_1)
=7+\frac{100-60}{120-60}(21-7)
=16.33\ \mathrm{N\,cm}.
\label{eq:gan-pullout}
\end{equation}
The matching motor, supply and phase-current conditions for these curve readings are not established in the retained source record. Consequently, this is a conditional sizing value, rather than a verified pull-out torque for the installed motor.

The bearings support the moving assembly, so the motor moves its 950\,g mass horizontally rather than lifting its full weight. Retaining the original assumed rolling-friction coefficient $\mu=0.004$, the steady-speed load follows from $F_f=\mu mg$ and $T=r_pF_f$~\cite{acarnley2002}. The correct GT2 pitch radius is $r_p=20(0.002)/(2\pi)=0.006366$\,m, giving
\begin{equation}
T_{\mathrm{req}}=r_p\mu mg
=0.006366(0.004)(0.950)(9.81)
=2.37\times10^{-4}\ \mathrm{N\,m}
=0.0237\ \mathrm{N\,cm}.
\label{eq:des-torque}
\label{eq:gan-treq}
\end{equation}
Here the 950\,g is the complete moving load; supporting it on two rails does not double its mass. The resulting conditional torque ratio is
\begin{equation}
\frac{T_{\mathrm{po}}}{T_{\mathrm{req}}}
=\frac{16.33}{0.02373}\approx688.
\label{eq:gan-margin}
\end{equation}
This reproduces the reasoning behind choosing one motor and a shared shaft for the two belt drives: under the assumed bearing resistance and earlier curve values, the steady-speed load is small compared with the available torque estimate. The ratio is not a measured safety factor. It excludes acceleration, belt preload, shaft and bearing misalignment, and depends on the applicability of the earlier torque curve. It therefore supports the original sizing argument without establishing that every starting rate is safe.

\begin{figure}[!htbp]\centering
\begin{minipage}{0.48\linewidth}\centering
\includegraphics[width=\linewidth,height=0.18\textheight,keepaspectratio]{Figures/nema-17-stepper-motor.png}
\end{minipage}\hfill
\begin{minipage}{0.48\linewidth}\centering
\includegraphics[width=\linewidth,height=0.18\textheight,keepaspectratio]{Figures/a4988-stepper-driver.png}
\end{minipage}
\caption{NEMA 17 stepper motor and A4988 driver used in the gantry.}
\label{fig:stepper-motor-and-driver}
\end{figure}

'''
s=s[:start]+block+s[end:]
s=s.replace('They use active-high 3.3\\,V inputs with 10\\,k$\\Omega$ pull-down resistors','The axis switches use active-high 3.3\\,V inputs with 10\\,k$\\Omega$ pull-down resistors')
p.write_text(s,encoding='utf-8')
for name in ['nema-17-stepper-motor.png','a4988-stepper-driver.png']:
 shutil.copy2(r/'Report/PrismSnapshot_2026-10-31/Figures'/name,o/'Figures'/name)
bib=(r/'Report/IntegratedReport_2026-11-01/references.bib').read_text(encoding='utf-8')
entry=r'''@article{dziedzic2026,
 author={Dziedzic, Tomasz and {Sen Yilmaz}, Aysegul and Szafranska, Anna},
 title={How fast is fluent handwriting?},
 journal={Forensic Science International},
 volume={383}, pages={112896}, year={2026},
 doi={10.1016/j.forsciint.2026.112896}}
'''
(o/'references.bib').write_text(bib+'\n'+entry,encoding='utf-8')
(o/'README.md').write_text('''# Standalone hardware revision

Replace only the hardware section with `3_3_hardware.tex`. The complete report has not been edited. The Figures directory contains the existing images plus the restored original CAD panels and motor/driver photographs. The existing report class and design_support macros are still required.

`references.bib` is a copy of the current report bibliography with one added entry, dziedzic2026, for the speed source already named in the original draft. Alternatively copy just that entry to your own bibliography.

The original speed-to-torque calculation sequence is restored: handwriting-speed benchmark, 4000 mm/min sizing feed, 100 rpm, interpolation, 950 g load and torque ratio. Pulley radius and arithmetic are corrected. The earlier 7/21 N cm curve points and 0.004 friction assumption are explicitly conditional. A matching source curve is still needed to present the available torque and margin as verified. The published abstract verifies the 10–60 mm/s sample range; full-paper page numbers were unavailable.
''',encoding='utf-8')
with zipfile.ZipFile(r/'output/Hardware_Revision_2026-11-02.zip','w',zipfile.ZIP_DEFLATED) as z:
 for q in sorted(o.rglob('*')):
  if q.is_file():z.write(q,q.relative_to(o).as_posix())
assert abs(0.006366197723675814*0.004*0.95*9.81*100-0.023732870674366696)<1e-6
print('Restored speed rationale, design feed, rpm, pull-out interpolation, 950 g torque and conditional ratio; report untouched.')
