from pathlib import Path
import re,zipfile
r=Path.cwd();o=r/'output/Hardware_Restored_From_Your_Draft_2026-11-02';p=o/'3_3_hardware_restored.tex';s=p.read_text(encoding='utf-8')
s=s.replace('\\subsubsection{Plotter designs and mechanics}\n\\label{sec:des-camera}','\\subsubsection{Plotter designs and mechanics}',1)
needle='\\subsubsection{Stepper motor and driver}'
s=s.replace(needle,'\\label{sec:des-camera}\nThe camera support holds the USB camera over the restrained page so that the image-processing stage receives a consistent view of the writing area. The IMX179 autofocus webcam and page restraint connect to the hardware arrangement shown in the system block diagram.\n\n'+needle,1)
s=s.replace(r'\tau=\max\Bigl(\frac{d/v_f}{N_{\mathrm{maj}}},\ 2\,\mu\mathrm s\Bigr),\qquad v_f=\max(F/60,\,0.1),',r'\tau=\max\Bigl(\frac{d/v_f}{N_{\mathrm{maj}}},\ 2\,\mu\mathrm s\Bigr),\qquad v_f=\max(F/60,\,0.1),')
p.write_text(s,encoding='utf-8')
# Confirm compatibility with labels referenced outside the replaced section.
cur=r/'Report/IntegratedReport_2026-11-01';whole='\n'.join(q.read_text(encoding='utf-8-sig') for q in cur.rglob('*.tex') if q.name!='3_3_hardware.tex')+'\n'+s
missing=set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',whole))-set(re.findall(r'\\label\{([^}]+)\}',whole));assert not missing,missing
with zipfile.ZipFile(r/'output/Hardware_Restored_From_Your_Draft_2026-11-02.zip','w',zipfile.ZIP_DEFLATED) as z:
 for q in sorted(o.rglob('*')):
  if q.is_file():z.write(q,q.relative_to(o).as_posix())
p=r/'Report/REPORT_COLLABORATION_CONTEXT.md';s=p.read_text(encoding='utf-8');note='''## Latest hardware preference and standalone replacement, 2 November 2026

The user rejected the heavily expanded hardware wording and structure. For hardware,
prefer the newly pasted original draft in attachment 439b6e90-13d1-4e86-8b97-8780e1a2ae55.
Restore its six substantive headings, mechanics voice, CAD, speed rationale, 950 g
load/torque sequence, pen/switch/calibration algorithms and G-code explanation. Keep
new command-space figure in the G-code subsection. Preserve the user's own A4988
wording where technically valid; do not expand into a new generic engineering review.
Latest delivered standalone: output/Hardware_Restored_From_Your_Draft_2026-11-02/3_3_hardware_restored.tex
and matching ZIP. The assembled report is NOT changed; insert only when asked.
The inherited pull-out curve (7 N cm at 60 rpm; 21 N cm at 120 rpm) and friction 0.004
remain conditional, not verified. Correct pulley radius gives 0.0237 N cm steady-load
estimate and conditional ratio about 688. 4000 mm/min is sizing headroom, not actual
selected operating feed. Never restore the false unconditional 560-fold guarantee.

'''
header=s.split('\n',1)[0]+'\n';s=s.replace(header,header+'\n'+note,1);p.write_text(s,encoding='utf-8')
print('Standalone replacement checked; assembled report unchanged.')
