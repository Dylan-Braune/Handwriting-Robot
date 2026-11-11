from pathlib import Path
import zipfile
r=Path.cwd();o=r/'output/Hardware_Revision_2026-11-02';p=o/'3_3_hardware.tex';s=p.read_text(encoding='utf-8')
needle='\\subsubsection{Step resolution, microstepping and calibration}'
block=r'''\paragraph{Driving the 950\,g moving assembly.}
The shared $y$-axis motor drives both belt pulleys through the coupled shaft. The bearings support the assembly's weight, so the motor moves the 950\,g load horizontally rather than lifting it. For a steady-speed estimate, the resisting force and pulley torque are related by $F_f=\mu mg$ and $T=r_pF_f$~\cite{acarnley2002}. Retaining the original assumed friction coefficient $\mu=0.004$, with $m=0.950$\,kg and the 20-tooth GT2 pitch radius $r_p=0.020(2)/(2\pi)=0.006366$\,m,
\begin{equation}
F_f=0.004(0.950)(9.81)=0.03728\ \mathrm{N},\qquad
T_{\mathrm{req}}=0.006366(0.03728)
=2.37\times10^{-4}\ \mathrm{N\,m}
=0.0237\ \mathrm{N\,cm}.
\label{eq:des-torque}
\end{equation}
The mass is counted once for the complete moving assembly: the two supporting sides do not double that total load. The calculation supports the choice of a mechanically coupled single-motor drive under the assumed low bearing resistance. It is a steady-speed estimate, not a measured operating margin; startup, belt preload and misalignment add load. At the configured 2400\,mm/min travel feed, the transmission calls for $2400/40=60$\,rev/min, so available torque must be assessed at that speed rather than from holding torque alone~\cite{acarnley2002}.

'''
assert needle in s and '\\label{eq:des-torque}' not in s;s=s.replace(needle,block+needle,1);p.write_text(s,encoding='utf-8')
z=r/'output/Hardware_Revision_2026-11-02.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for q in sorted(o.rglob('*')):
  if q.is_file():f.write(q,q.relative_to(o).as_posix())
print(p)
