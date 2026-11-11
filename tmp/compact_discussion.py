from pathlib import Path
import re,json
P=Path('Report/SubmissionRevision_2026-11-02');R=P/'3_Main_Report'
p=R/'5_Discussion.tex';s=p.read_text()
# Retain every heading, replace only its explanatory body.
def body(title,text,kind='paragraph'):
 global s
 pat=r'(\\'+kind+r'\{'+re.escape(title)+r'\}\s*(?:\\label\{[^}]+\}\s*)?)(.*?)(?=\\(?:paragraph|subsubsection|subsection)\{|\\FloatBarrier|\Z)'
 s,n=re.subn(pat,lambda m:m[1]+text.strip()+'\n\n',s,flags=re.S)
 assert n==1,(title,n)
body('Useful reproduction is demonstrated, with a clear operating range.',r'''Seven writers exceed the 85\% character target, with four above 93\%. The pooled 84.41\% score shows that the main challenge is consistency across styles. Figure~\ref{fig:res-physical} makes the difference visible in actual output. The association between separated original characters and better reproduction ($\rho=0.845$) supports investigating connected handwriting, but does not isolate joining from input quality or glyph coverage.

Writer identity survives on 86.41\% of physical lines and all fourteen page pools. Geometric shape correlation is lower, at 0.697. Slant is strongly retained ($r=0.991$), while word spacing is weaker ($r=0.486$). These results favour improving context and layout while preserving the useful glyph geometry. The writer classifier also helps select candidates, so its score alone is not an independent judgement of style. The geometric comparison adds a separate measure, while human reading and style assessment would extend both automated tests.''')
body('The classifier results support the chosen architecture.',r'''The joint reader's 94.06\% character accuracy is 0.94 percentage points below target. Its public and personal subsets score 94.12\% and 93.33\%. This supports line recognition across both inputs, although the experiment does not isolate errors introduced by segmentation.

Writer identification reaches 98.08\%, with every writer group correct after pooling. Stroke normalisation makes the test less dependent on original pen thickness. However, an error at 83.8\% confidence shows that confidence is not a reliable identity guarantee. The personal line split also shares capture conditions with training. A new-session evaluation would test deployment more strongly.''')
body('The physical timing result is a practical success.',r'''All 25 completed ODROID jobs meet 4 s per character, including generation and drawing. This is stronger evidence than nominal feed calculations because it includes pen changes and execution overhead. The worst short request leaves 5.39 s of margin. The measured range starts at 43 characters, so shorter requests need their own test. Repeated 378-character jobs also vary substantially, consistent with different path lengths and pen transitions.''')
body('The classification-time gap is measurable and actionable.',r'''The two classification requests exceed their allowances by 31.76 and 28.94 s. Networks contribute about two thirds of the combined time. The host comparison demonstrates a 2.76-fold network improvement with unchanged reader text, but transferring that factor still leaves both projected totals above target. Network acceleration and image-stage optimisation therefore need to be addressed together. Halving image resolution is unsuitable because it changes the detected lines and decoded text. A useful optimisation must retain the reference output as well as reduce runtime.''')
body('Mechanical accuracy is promising, with a precise distinction in what was measured.',r'''All seven measured lengths are within 0.5 mm of nominal, supporting the calibrated belt drive. The unequal triangle slopes suggest a possible alignment effect. The apparent $0.706^\circ$ diagnostic in Appendix~\ref{app:res-geometry-calculation} is conditional on the assumed height and does not independently measure rail squareness.

A translated drawing can retain correct side lengths. Repeated bed-coordinate points and word overlays would therefore extend the present dimensional check to absolute placement and repeatability. This builds on the functioning motion system rather than requiring an unsupported redesign.''')
body('Relationship to published approaches.',r'''The reader follows the CNN, BiLSTM and CTC approach discussed by K{\i}z{\i}l{\i}rmak and Yan{\i}ko{\u g}lu~\cite[pp.~4--5]{kizilirmak2023}. Its 5.94\% character error supports the architecture, but different partitions and decoding prevent a controlled ranking against published scores.

Haines et al. support example-based synthesis from a small collection of pages~\cite[p.~4]{haines2016}. Graves's sequence generator instead requires ordered pen coordinates~\cite[p.~18]{graves2013}, which are absent from the project's photographs. One-DM uses a single reference at generation time, but still requires a separate training corpus~\cite[p.~9]{dai2024}. The glyph approach suits the available offline data and makes geometry directly controllable, with limited contextual variation as its main compromise.''')
body('Separating the implementation contribution from checkpoint selection.',r'''The first-principles implementation covers the forward passes, derivatives, CTC and optimiser updates. Table~\ref{tab:dis-training} compares its trained models with the reference route on common validation data. The first-principles writer reaches 95.51\% and the reader reaches 94.39\%, supporting the numerical implementation independently of the selected system configuration.

The supplied logs in Appendix~\ref{app:training-records} record approximately 19 to 23 minutes per writer epoch and 111 to 122 minutes per reader epoch. Float32 reduced the numerical training cost, but no controlled experiment attributes the model-score differences to precision. The 155-line training validation and 156-line re-extraction are also kept separate.''')
body('Line recognition addresses a real weakness in character isolation.',r'''The proposal described classifying separated characters or connected groups. The final reader recognises whole lines, then uses known-transcript CTC alignment to recover character boundaries for profiles. This avoids making uncertain character cuts a prerequisite for reading~\cite[pp.~2--3]{graves2006}. The compromise is a larger sequence model and continued dependence on correct line ownership.''')
body('The glyph library fits the data and exposes its own limitations.',r'''Stored glyphs allow size, slant, spacing and machine coordinates to be inspected and changed directly. Eleven profiles demonstrate the approach on real output. A finite library, however, cannot reproduce every change caused by neighbouring characters. Multi-character units or context-specific variants would address this more directly than simply increasing candidate count. Their extraction would need enough reliable examples.

Off-device training and embedded inference keep model preparation separate from operation. Batched numerical kernels or compiled hot loops are plausible speed improvements compatible with the numerical contribution. Their benefit and full timing compliance still require embedded measurements.''')
body('Economic feasibility.',r'''The current build estimate is approximately R4\,000. The ODROID was borrowed at no purchase cost and will be returned. A repeat build must include its replacement cost. Later purchases and estimated training electricity total R750, but may overlap the approximate estimate, so they are not added as an exact revised total. The software speed experiment gives a measured improvement opportunity before committing to a more expensive computer.''')
body('Unsolved problems',r'''The main reproduction problem is variation between writers. A controlled comparison of current glyphs and multi-character units, using fixed texts and models, would test the context explanation. Matching original and reproduced sentences would also strengthen the style comparison.

The timing priority is to preserve the reference output while profiling and optimising ODROID hot loops. Mechanical work should concentrate on repeated coordinates, frame alignment and pen-holder stiffness. A shared pulse scheduler and controlled acceleration are reasonable extensions, with their value assessed on completed drawings.''','subsubsection')
body('Strong points of the design',r'''The complete chain captures handwriting, reads text, identifies writers, stores paths and produces physical writing. Eleven styles are exercised, seven exceed the reproduction character target and every validation writer group is identified correctly.

A path can be followed from source crop through glyph extraction, placement and G-code. This makes faults easier to locate and allows individual stages to be improved. Numerical operation checks and the trained-model comparison support the first-principles contribution alongside the physical prototype.''','subsubsection')
body('Expected failure conditions',r'''Strong cursive connections, missing character variants and text beyond the available alphabet can weaken reproduction. Large layouts can exceed the bed, making preview inspection important before drawing.

Touching lines, page curvature and perspective can defeat preparation even when the reader works on clean crops. The writer classifier always selects a known identity, so an unfamiliar writer can receive a confident label. Page pooling adds evidence but cannot recover information missing from every line.

The gantry tracks open-loop steps. A missed step changes the real position without changing its counter. Limit switches and abort handling reduce operational risk, but depend on working electronics and software. The missing pen-motor flyback diode remains a circuit limitation described in Design. Requests outside the tested lengths have no demonstrated timing envelope.''','subsubsection')
body('Ergonomics',r'''The browser combines capture review, text entry, writer selection and preview. Calibration and the visible writing area assist setup. Paper and pen positioning remain manual, so the operator must check restraint and clearance before starting. Clear generated, running, stopped and completed states help distinguish a prepared path from physical execution.''','subsubsection')
body('Health and safety',r'''Belts, pulleys and the carriage create pinch points. The working area must remain clear during motion. End stops and software aborts assist normal operation, while independent motor-power removal would protect against an unresponsive controller. Driver temperature, wiring restraint and supply enclosures also require attention.

The 24 V motor and 3.3 V logic rails are separated from the purchased supplies' mains inputs. A reverse-biased diode across the pen motor would provide an inductive-current path when switching off~\cite{rohmlowside2026}. It is a recommended improvement, not a fitted component.''','subsubsection')
body('Environmental impact',r'''The borrowed ODROID will be returned. Motors, rails and drivers can be reused, and individual printed parts can be replaced. The printed material is not recorded, so no specific recycling claim is made. Digital previews reduce wasted paper during development, while physical tests remain necessary. Training electricity is estimated as a cost, without an energy-meter record or calculated environmental saving.''','subsubsection')
body('Social and legal impact',r'''Personalised writing could assist a person who has difficulty writing, although this benefit has not been evaluated with users. Style reproduction also permits impersonation. Samples should be used with permission and machine-produced material identified where its origin matters.

The five personal contributors signed consent forms, retained securely rather than publishing their signatures. Handwriting and profiles require appropriate collection, retention and access controls under the Protection of Personal Information Act~\cite[pp.~8--9, 11]{popia2013}. IAM data must also be used under its applicable terms. The prototype is intended for a controlled local network.''','subsubsection')
body('Ethics clearance','Signed consent was obtained from all five personal contributors. An institutional clearance or exemption reference is not present in the retained records, so consent is not presented as institutional approval.','subsubsection')
s=s.replace('The project connects photographed handwriting, numerical recognition, persistent writer profiles and a working Cartesian gantry. The results support this central function across eleven writers, with quality depending on the handwriting and embedded classification time remaining the principal performance constraint. This evaluation relates those outcomes to the decisions explained in Design.','The evaluation relates the measured performance to the Design choices. The main trade-offs concern variation between writing styles and the time needed for embedded classification.')
p.write_text(s)
(R/'6_Conclusion.tex').write_text(r'''\subsection{Summary of the work completed}
A handwriting robot was designed and built to recognise photographed text, identify a known writer and reproduce new digital text in that writer's style. The system combines page conditioning, line segmentation, neural recognition, stored glyph trajectories and a calibrated Cartesian gantry. Eleven writer profiles are stored and exercised in physical reproduction.

\subsection{Summary of observations and findings}
The selected reader achieves 94.06\% character accuracy and writer identification reaches 98.08\% on the recorded validation set. Physical reproduction gives 84.41\% read-back character accuracy, with seven writers above 85\%. Writer identity is retained on 86.41\% of reproduced lines and all fourteen page pools. The measured shape correlation is 0.697, showing that recognisable identity does not imply complete geometric similarity.

All 25 completion-confirmed ODROID reproduction jobs satisfy 4 s per character. The two measured classification requests exceed their time allowances. Every measured square and triangle length is within 0.5 mm of nominal, with a maximum difference of 0.37 mm. These results establish dimensional performance on the specimens, while absolute placement and repeatability need further qualification.

\subsection{Contribution}
The project demonstrates a complete path from offline handwriting samples to physical multi-writer output. Its first-principles network implementation is supported by numerical checks and trained models with scores close to the reference route. Explicit glyphs and G-code make the relationship between image features and pen movement visible, allowing each stage to be inspected and improved.

\subsection{Suggestions for future work}
The first priority is faster embedded classification without changing the detected lines or decoded text. Batching and compiled numerical kernels should be measured on the ODROID. Context-specific glyphs and multi-character units should then be tested on the more connected writers. Repeated bed-coordinate measurements, word overlays and matched original-to-reproduced sentences would strengthen the mechanical and stylistic evaluation. Improvements to motor protection and pulse scheduling would complete the next hardware revision.
''')
p=P/'1_Preamble/2_Abbreviations.tex';p.write_text(r'''\section*{LIST OF ABBREVIATIONS}
\begin{tabular}{p{3cm}p{10cm}}
ADC & analogue-to-digital converter\\
BiLSTM & bidirectional long short-term memory\\
CAD & computer-aided design\\
CER & character error rate\\
CNN & convolutional neural network\\
CPU & central processing unit\\
CTC & connectionist temporal classification\\
DC & direct current\\
GPIO & general-purpose input/output\\
GPU & graphics processing unit\\
GT2 & 2 mm pitch timing-belt profile\\
IAM & Institute of Informatics and Applied Mathematics handwriting database\\
I/O & input/output\\
LSTM & long short-term memory\\
NEMA & National Electrical Manufacturers Association\\
NLL & negative log-likelihood\\
PWM & pulse-width modulation\\
QT & qualification test\\
ReLU & rectified linear unit\\
RNN & recurrent neural network\\
USB & universal serial bus\\
VNC & virtual network computing\\
WER & word error rate\\
\end{tabular}
''')
# Insert moved calculations and figures.
p=P/'Appendices/design_implementation_details.tex';t=p.read_text();t+='\n'+r'\subsection{Worked values from the implemented system}\label{app:worked-design-calculations}'+'\n'
group=None
for name,para in json.loads(Path('tmp/final_report_review/moved_calculations.json').read_text()):
 if name!=group:t+='\n\\subsubsection{'+name+'}\n';group=name
 t+=para+'\n\n'
p.write_text(t)
p=P/'Appendices/design_subsystem_evidence.tex';p.write_text(p.read_text()+'\n'+Path('tmp/final_report_review/moved_figures.tex').read_text())
p=P/'Appendices/results_evidence.tex';t=p.read_text();t+='\n'+r'\subsection{Triangle geometry diagnostic}\label{app:res-geometry-calculation}'+'\n'+Path('tmp/final_report_review/apex_calculation.tex').read_text();p.write_text(t)
print('Discussion words',len(s.split()))
