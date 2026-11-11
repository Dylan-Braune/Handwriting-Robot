from pathlib import Path
import re
P=Path('Report/SubmissionRevision_2026-11-02'); R=P/'3_Main_Report'
def entry(file,key):
 s=Path(file).read_text();start=re.search(r'@\w+\{'+key+r',',s).start();dep=0
 for i in range(s.index('{',start),len(s)):
  dep+=(s[i]=='{')-(s[i]=='}')
  if dep==0:return s[start:i+1]
b=P/'references.bib';s=b.read_text()
for key,file in [('dziedzic2026','Report/FinalReport_v6/references.bib'),('stepperds','Report/FinalReport_v6/references.bib'),('dml31p','Report/DesignPolished_2026-11-01/references.bib'),('ga12n20ds','Report/DesignPolished_2026-11-01/references.bib')]:s+='\n\n'+entry(file,key)
b.write_text(s)
p=P/'Appendices/design_implementation_details.tex';s=p.read_text();marker=r'\subsubsection{Hardware scale-error allocation}';i=s.index(marker);s=s[:i];s=re.sub(r'\\subsubsection\{Hardware calculation detail\}\s*where .*?\n\n',r'\\subsubsection{Hardware calculation detail}\n',s,flags=re.S);p.write_text(s)
# Accurate assistance declarations, retain front-matter headings and unsigned signature fields.
p=P/'1_Preamble/1_Preamble.tex';s=p.read_text();s=s.replace('The author confirms that no artificial intelligence tools, including language models or code-generation tools, were used in the preparation and writing, or any analysis that appears in this report. All content, design, explanations, data analysis, and code were developed solely by the author.','Artificial intelligence tools assisted with code development, analysis support and report editing. The author is responsible for checking the submitted work, understanding the implementation and acknowledging assistance in accordance with the University\'s requirements.')
s=s.replace('Unless explicitly stated otherwise in the next paragraph, the author confirms that no software library functions, packages, or toolboxes were used in the design, implementation, or analysis presented in this report. All code and computations were developed entirely by the author','The project includes first-principles implementations of the network operations, training and image-processing algorithms described in Design. NumPy provides array operations. Supporting libraries provide image handling, the web interface and hardware access. A PyTorch route provides reference training and comparisons. The evaluated checkpoint provenance is recorded in Appendix~\\ref{app:res-evidence}.')
p.write_text(s)
# Concise system explanation, in diagram order.
p=R/'design/3_2_system.tex';s=p.read_text();a=s.index('The system is split');b=s.index('\\begin{figure}',a)
s=s[:a]+r'''Figure~\ref{fig:des-hardware-system} groups the hardware into five parts. External power (1) supplies the ODROID and motor drivers. External interfaces (2) provide page restraint, camera images and operator access. Stepper control (3) converts GPIO commands into winding currents. The belt-driven gantry (4) moves the pen in $x$ and $y$. The pen mechanism (5) switches contact with the page and reports its state. Table~\ref{tab:des-pin-map} in the appendix gives the individual connections.

'''+s[b:]
a=s.index('Figure~\\ref{fig:des-processing-system}');b=s.index('\\begin{figure}',a)
s=s[:a]+r'''Figure~\ref{fig:des-processing-system} shows four software groups. Segmentation (1) conditions a page and returns line crops. Reproduction (2) assembles the selected profile into trajectories and G-code, then executes the path after calibration. Training (3) uses labelled crops to fit the text or writer model and builds stored glyph profiles from labelled handwriting. Classification (4) recognises text or identifies the writer, applying stroke normalisation before writer inference. The diagram groups these functions, while their data dependencies and separate training stages are detailed below.

'''+s[b:]
s=re.sub(r'\\caption\{Hardware system block.*?\}\s*\\label',r'\\caption{Hardware system block diagram.}\n\\label',s,flags=re.S)
s=re.sub(r'\\caption\{Software system block.*?\}\s*\\label',r'\\caption{Software system block diagram.}\n\\label',s,flags=re.S);p.write_text(s)
# Table implementation cells as sentences, preserve its headings and rows.
p=R/'design/3_1_summary.tex';s=p.read_text();repls={
'Camera arrangement, illumination normalisation, ink extraction and orientation correction.':'The camera supplies pages for illumination correction, ink extraction and orientation correction.',
'Line detection, curved boundaries and ink ownership; later transcript-constrained glyph localisation.':'The segmenter separates lines and assigns their ink. Known transcripts then guide glyph localisation.',
'Convolutional features, sequence recognition and writer identification.':'Separate networks recognise line text and identify the writer from convolutional features.',
'Aligned glyph extraction, skeleton tracing, variant filtering and persistent style profiles.':'Aligned glyphs are traced, filtered and stored with each writer\'s style measurements.',
'Text layout, glyph assembly, candidate selection and conversion to machine coordinates.':'The synthesiser lays out glyphs, selects a candidate and converts its paths to machine coordinates.',
'Custom mechanical assembly, two-axis drive, pen switching, end stops and calibration.':'The gantry combines a two-axis belt drive, switched pen lift, end stops and calibration.',
'Numerical checks and offline image/path experiments; formal integrated qualification is reported in Results.':'Numerical checks and offline image and path experiments verify the stages. Results reports the integrated qualification.'}
for a,b in repls.items():s=s.replace(a,b)
p.write_text(s)
# Figure captions and positions. Keep the protected Literature and Approach untouched.
short={
'fig:des-hardware-system':'Hardware system block diagram.', 'fig:des-processing-system':'Software system block diagram.',
'fig:stepper-motor-and-driver':'NEMA 17 motor and A4988 driver.', 'fig:pen-up-down-mechanism':'Pen lowered (left) and raised (right).',
'fig:pen-clicker-bc337-circuit':'BC337 pen-motor switching circuit.', 'fig:endstop-switch-pulldown':'End-stop input and pull-down resistor.',
'fig:des-style-files':'Four profiles reconstructed from emitted G-code.', 'fig:des-physical-page':'Photographed output and extracted line crops.',
'fig:vt-grad':'Analytic and finite-difference gradients.', 'fig:vt-width':'Stroke-width kernels and validation sweep.',
'fig:vt-coordinates':'Image and stored glyph coordinate frames.', 'fig:vt-connectedness':'Observed connections between neighbouring characters.',
'fig:app-physical-style':'Original and reproduced writer descriptor means.', 'fig:res-r3':'Completed reproduction times and allowances.',
'fig:res-r4':'ODROID classification stages and allowances.', 'fig:res-geometry':'Signed errors in the measured feature lengths.'}
def capreplace(block,new):
 i=block.index('\\caption{')+9;level=1
 for j in range(i,len(block)):
  level+=(block[j]=='{')-(block[j]=='}')
  if level==0:return block[:i]+new+block[j:]
 return block
paths=[*list((R/'design').glob('*.tex')),*list((P/'Appendices').glob('*.tex')),R/'4_Results.tex']
for p in paths:
 s=p.read_text()
 def fix(m):
  t=m[0];labels=re.findall(r'\\label\{([^}]+)\}',t)
  for lab in labels:
   if lab in short:t=capreplace(t,short[lab]);break
  # Strictly preserve source order for figures in main Design and Results.
  if p.parent.name=='design' or p.name=='4_Results.tex':t=re.sub(r'\\begin\{figure\}(?:\[[^]]*\])?',r'\\begin{figure}[H]',t,1)
  return t
 s=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',fix,s,flags=re.S)
 p.write_text(s)
# Small summary-table width fix.
p=R/'4_Results.tex';s=p.read_text().replace(r'\begin{tabular}{p{.27',r'\setlength{\tabcolsep}{5pt}\begin{tabular}{p{.27');p.write_text(s)
# Appendix evidence descriptions refer to preserved archives, not invented reruns.
p=P/'Appendices/design_subsystem_evidence.tex';s=p.read_text();a=s.index('This appendix records');b=s.index('\\subsection',a);s=s[:a]+r'''This appendix retains the offline checks and figures that extend the Design explanations. No training or gantry motion was performed during report revision. The original inputs, scripts and recorded outputs are preserved in the separate supporting-evidence package. The integrated qualification tests remain in Results.

'''+s[b:];p.write_text(s)
p=P/'Appendices/results_evidence.tex';s=p.read_text().replace('The source package contains the original v6 measurement archive, the reviewed aggregate records and the scripts that generate the revised tables and figures.','The separate supporting-evidence package contains the original v6 measurement archive, reviewed records and figure-generation scripts.')
# Explain appendix figures before them.
intro={
'fig:res-r3':r'Figure~\ref{fig:res-r3} compares the worst completed reproduction total at each tested length with its allowance. Labels retain the number of records.',
'fig:res-r4':r'Figure~\ref{fig:res-r4} separates image and network processing for the two ODROID requests. The dashed markers show their complete allowances.',
'fig:res-geometry':r'Figure~\ref{fig:res-geometry} compares each measured feature length with its nominal value. The dashed limits apply to length differences and do not establish absolute bed coordinates.'}
for lab,text in intro.items():
 pat=r'(\\begin\{figure\}.*?\\label\{'+re.escape(lab)+r'\}.*?\\end\{figure\})'
 # locate only that block, not earlier figures
 for m in list(re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}',s,re.S)):
  if '\\label{'+lab+'}' in m[0]:s=s[:m.start()]+text+'\n'+s[m.start():];break
p.write_text(s)
print('Polish complete')
