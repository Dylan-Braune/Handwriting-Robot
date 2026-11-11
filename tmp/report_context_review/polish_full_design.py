from pathlib import Path
import shutil,re
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'Report/DesignVisualRevision_2026-11-01'
DST=ROOT/'Report/DesignPolished_2026-11-01'
for p in SRC.rglob('*'):
    if p.is_file() and p.suffix.lower() in {'.tex','.bib','.cls','.md','.json','.py','.png','.jpg','.jpeg','.pdf','.gcode'} and p.name!='design_preview.pdf':
        q=DST/p.relative_to(SRC);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def edit(name,changes):
    p=DST/name;s=p.read_text(encoding='utf-8')
    for old,new in changes:
        if old not in s:raise ValueError((name,old[:100]))
        s=s.replace(old,new)
    p.write_text(s,encoding='utf-8')

edit('3_Main_Report/design/3_1_summary.tex',[
('Network training and verification&Numerical training implementation and a corresponding framework implementation; labelled public and personal handwriting data.',
 'Network training implementations&Explicit NumPy training calculations and a framework reference; final selected weights were framework-trained.'),
('Display and system communication&Web interface and server connect camera acquisition, classification, generation and physical writing.',
 'Processing and motion integration&Server connects capture, numerical processing, stored profiles and physical execution; browser provides the operating interface.'),
('The final implementation separates text into lines rather than requiring isolated letters before recognition.',
 'The recognition design uses complete text lines rather than requiring isolated letters as proposed for FU~2.2.'),
('These choices retain the proposal\'s intended functions while changing how those functions are implemented.',
 'The resulting architecture implements the proposal\'s recognition and reproduction functions through line-based recognition and later transcript-constrained character localisation. Off-the-shelf motors, drivers, camera and interface libraries support these functions; the claimed design work lies in their integration, the numerical processing and the pen-path and motion algorithms.'),
])

edit('3_Main_Report/design/3_2_system.tex',[
('The complete system has a physical interface and a processing interface.',
 'The complete system connects page capture, handwriting interpretation and physical reproduction.'),
('The interface between software and mechanics is G-code.',
 r'''The proposal's functional units remain identifiable in this architecture. Camera acquisition and request handling implement FU~1; conditioning and line preparation implement FU~2.1--2.2; the two trained networks implement classification FU~2.3. Offline weight training corresponds to FU~2.4, while glyph extraction, profile storage and glyph retrieval implement FU~2.5--2.7. Trajectory conversion and the motion interpreter implement FU~2.8, followed by the axis drives, pen holder and lift in FU~3.1--3.4. Training and profile preparation produce stored artefacts before request-time recognition or writing; they are not repeated for each page.

The interface between software and mechanics is G-code.'''),
])

edit('3_Main_Report/design/3_3_hardware.tex',[
('The observed 24\\,V and 0.35\\,A correspond to 8.4\\,W at that supply output; this is not a peak-power rating or the total robot consumption unless all loads and the measurement condition are included.',
 'At the recorded operating condition, that supply delivers approximately $24(0.35)=8.4$\\,W. This reading does not include a verified peak-load test or establish total consumption from both supplies.'),
('its DC output rating is recorded separately from the 24\\,V stepper supply.',
 'its DC output rating remains to be confirmed separately from the 24\\,V stepper supply.'),
('Figure~\\ref{fig:des-complete-circuit} provides the place for the final circuit with those connections and component values visible.',
 'Figure~\\ref{fig:des-complete-circuit} shows the complete supply and signal connections, with component values and the common reference identified.'),
('``Why NPN Devices Are Used in Low-Side Switches]',"``Why NPN Devices Are Used in Low-Side Switches'' ]"),
('``Protection and Wiring Techniques]',"``Protection and Wiring Techniques'' ]"),
('The formal dimensional and repeated-drawing measurements belong in the Results section.',
 'The formal dimensional and repeated-drawing measurements in Results assess the resulting mechanical accuracy.'),
])

edit('3_Main_Report/design/3_4_segmentation.tex',[
('The chosen design therefore first locates the lines, then finds curved boundaries between them, and finally assigns complete ink pieces to a line.',
 'The chosen design first locates the lines, finds curved boundaries between them, then uses component geometry to refine which line owns each ink piece.'),
('The constants express the chosen preference, while the examples in Section~\\ref{sec:des-segverify} must establish whether that preference produces useful boundaries.',
 'The examples in Section~\\ref{sec:des-segverify} examine the resulting boundaries and the cases that still defeat these geometric preferences.'),
('It also prevents a shadow surviving as a confident writer vote in Section~\\ref{sec:des-integration}.',
 'This reduces the chance of a solid shadow receiving a strong writer vote in Section~\\ref{sec:des-integration}. If every crop would be rejected, the implementation retains the input crop list; crop rejection is therefore a guard rather than a guarantee of valid text.'),
('its two gap scores were 0.379 and 0.400, below the 1.35 turn-ratio requirement,',
 'its gap scores were 0.379 and 0.400, giving a ratio of only 1.054 against the required 1.35,'),
])

p=DST/'3_Main_Report/design/3_5_classification.tex';s=p.read_text(encoding='utf-8')
start=s.index('Preparing labelled public forms')
end=s.index('\\designgap{Record reader split counts',start)
s=s[:start]+r'''Preparing public forms also requires excluding printed prompts and signatures before matching detected lines to their transcripts. The eleven-writer classifier corpus contains 1186 labelled lines: 517 IAM and 669 photographed lines, split into 1031 training and 155 held-out lines. The final reader instead uses a joint corpus of Teklia pre-cut IAM lines and the five photographed writers; its exact training and validation counts are not preserved in the available record.

Evaluation separation concerns three inputs: the writer classifier's training lines, the reader's training lines and the pages used to build glyph profiles. Holding a line out of one classifier does not exclude it from the other corpus, and constructing a profile from an evaluation page already exposes its letterforms. Cross-corpus source identifiers and profile-page manifests are therefore needed before describing combined performance as independent of those inputs.
'''+s[end:]
s=s.replace('Both networks are trained through their complete convolutional backbones and task-specific heads.',
            'The implemented training routes update the convolutional backbone and task-specific head together.')
s=s.replace('Reusing the split and initialisation isolates width more effectively than changing width, population and sampling together.',
            'A fixed split and controlled initialisation make the widths comparable without changing the writer population at the same time.')
s=s.replace('The framework route in \\texttt{trainlibrary.py} expresses the corresponding networks using PyTorch layers and automatic differentiation. It uses the project\'s data preparation and parameter naming, and exports weights in the same archive layout for numerical inference. It provides a reference implementation and an alternative training route; claims about training speed need a matched timing comparison because the framework may use a different processor or accelerator.',
            'The framework route in \\texttt{trainlibrary.py} expresses the corresponding networks using PyTorch layers and automatic differentiation. Both routes use the project\'s data preparation, parameter naming and named-array export layout. This makes the framework implementation useful as a numerical reference while allowing either training route to supply weights to the NumPy inference code.')
s=s.replace('The final reader weights and the supplied writer-model weights were trained using PyTorch and exported to named arrays. The selected eleven-writer stroke-normalised model also follows the framework route. The first-principles trainer is an implemented and numerically checked contribution, while the operational forward path is numerical inference. These are separate claims: the report does not attribute production of the supplied final weights to the NumPy trainer.',
            'The selected reader and eleven-writer weights were trained using PyTorch and exported to named arrays. The first-principles contribution comprises the implemented numerical training operations and the operational NumPy forward path. Fixed-parameter comparisons verify the numerical calculations; the provenance of the selected weights remains the framework training route.')
s=s.replace('The latter also enables zeroing of impossible-alignment losses. A controlled training comparison must match these conventions if learning-curve equivalence is claimed: target-length weighting changes the relative influence of short and long lines. This does not prevent common weight export or matched fixed-parameter verification.',
            'The framework loop also zeroes impossible-alignment losses. These conventions must be matched for a controlled training comparison because target-length weighting changes the influence of short and long lines. Named-array export remains compatible with both routes.')
s=s.replace('``Parameters]',"``Parameters'' ]")
s=s.replace(' Older classifier archives do not establish the selected eleven-class configuration merely because their filenames also contain ``shape\'\'.','')
s=s.replace('The two routes are summarised in Table~\\ref{tab:des-implementations}. Both export named arrays for numerical inference, so the deployed forward calculation is independent of the training engine. Training provenance remains a separate property of each checkpoint: executing weights with the numerical reader does not establish which implementation produced them.',
            'Table~\\ref{tab:des-implementations} distinguishes the operations supplied by each route. The same exported parameters can be evaluated by the numerical inference module independently of the training engine.')
s=s.replace('\\textbf{Writer head}', '\\textbf{Writer classifier}')
s=s.replace('An additional check can compare an implemented derivative with the central difference in Equation~\\ref{eq:des-gradcheck}.',
            'The finite-difference reference uses Equation~\\ref{eq:des-gradcheck}, with one logit or parameter perturbed at a time.')
p.write_text(s,encoding='utf-8')

p=DST/'3_Main_Report/design/3_6_reproduction.tex';s=p.read_text(encoding='utf-8')
old=r'''Each glyph is stored relative to its left boundary and the line baseline. Equation~\ref{eq:des-glyphcoords} defines the normalised coordinate frame.
\begin{equation}
v=\frac{b-y}{h_x},\qquad
u=\frac{x-x_L}{h_x}-v\tan\phi.
\label{eq:des-glyphcoords}
\end{equation}
Here $x_L$ is the cell's left edge, $h_x$ is x-height and $\phi$ is the estimated writing slant. In Equation~\ref{eq:des-glyphcoords}, image rows increasing downwards become glyph height increasing upwards. Removing slant at storage allows the writer's slant to be reapplied during placement without making every spacing measurement depend on shear. The glyph can then be drawn at a chosen physical x-height instead of a fixed number of source pixels.'''
new=r'''The coordinate frame separates source-image size, baseline and slant from the geometry to be stored. Equation~\ref{eq:des-glyphcoords} first expresses each path point relative to the cut boundary $x_L$ and baseline $b$, using the estimated x-height $h_x$ and slant $\phi$.
\begin{equation}
v=\frac{b-y}{h_x},\qquad
u_0=\frac{x-x_L}{h_x}-v\tan\phi,\qquad u=u_0-\delta.
\label{eq:des-glyphcoords}
\end{equation}
Image rows increasing downwards become glyph height increasing upwards. After trimming eligible ligature tails, the code sets $\delta=\min(u_0+v\tan\phi)$ over the remaining points, subtracts it from the stored horizontal coordinates and records lead $l=\operatorname{clip}(\delta,0,1.5)$. Thus the stored path and its lead are separate quantities: they preserve the relation between the character cell and the ink rather than forcing every cell to start at its first stroke. Slant is removed for storage and reapplied during placement. The glyph can then be drawn at a chosen physical x-height instead of a fixed number of source pixels.'''
if old not in s:raise ValueError('glyph coordinate paragraph')
s=s.replace(old,new)
needle='The worked example in Figure~\\ref{fig:des-alignment}'
idx=s.index(needle)
s=s[:idx]+r'''The baseline and x-height are operational estimates from the line's horizontal ink profile, not annotated typographic measurements. With $p(y)=\sum_x M(y,x)$, the implementation takes the first and last rows satisfying $p(y)\geq0.45\max p$ as the body-band top and baseline; their difference gives $h_x$. Bands below three pixels are rejected. This common scale makes glyph dimensions comparable across source resolutions, although unusual capitals, short lines or a large descender can bias the estimated body band.

'''+s[idx:]
s=s.replace('Up to twelve variants per character are retained.',
            'The ordinary variant pool is limited to twelve per character; an accepted denoised prototype can be prepended as an additional candidate.')
s=s.replace('Slant&Projection concentration under trial shears&Reapplied to stored upright glyphs\\\\\\hline',
            'Slant&Projection concentration under trial shears&Reapplied to stored upright glyphs\\\\\\hline\nX-height&Height of the estimated line body band&Normalises stored paths; physical x-height sets drawing size\\\\\\hline')
s=s.replace('Rendering every writer with one pen-width convention avoids deliberately supplying ink differences that the physical pen cannot preserve.',
            'Uniform rendering keeps candidate selection focused on reproducible geometry, while the shape classifier applies the selected stroke-normalisation convention.')
s=s.replace('The joint route does not promise that every proposed repair is accepted, nor that weak characters will all be corrected. Its repair state is committed only on improvement. A separate reader-only synthesis helper exists, but its more aggressive print-escalation behaviour is not the joint procedure described in Algorithm~\\ref{alg:des-joint}.',
            'A repair is retained only when the complete trial improves the joint score. This bounds the search and protects the best candidate already found. A separate reader-only helper uses a different escalation policy; it is not part of this joint selection procedure.')
s=s.replace('The most informative design example follows one source line through alignment, extraction, storage, assembly, candidate selection and machine mapping. Each representation can then be checked against the preceding one.',
            'The reproduction checks follow the representations established above: aligned source ink, stored glyph paths, assembled trajectories and emitted machine commands.')
p.write_text(s,encoding='utf-8')

edit('3_Main_Report/design/3_7_integration.tex',[
('The final integration connects the representations described above without moving the handwriting algorithms into the user interface. The browser supplies an image or a writing request and displays the returned information. The server loads the models and profiles, invokes the processing modules and communicates with the gantry. The interface is therefore an operating tool for the design rather than the source of its classification or synthesis calculations.',
 'The server connects the processing modules, stored models, writer profiles and motion interpreter. The browser supplies images or writing requests and displays the outputs; the handwriting calculations remain in the numerical processing modules.'),
('For reproduction, the selected writer profile and digital text determine a generated job.',
 'For reproduction, the selected writer profile and digital text determine a generated job. New text is supported within the configured character set; characters absent from a writer\'s profile may require the fallback paths described in Section~\\ref{sec:des-layout}.'),
('The complete source code has been submitted separately on the AMS.',
 'The complete source code is supplied separately from this report for the AMS submission.'),
] if 'The complete source code has been submitted separately on the AMS.' in (DST/'3_Main_Report/design/3_7_integration.tex').read_text() else [
('The final integration connects the representations described above without moving the handwriting algorithms into the user interface. The browser supplies an image or a writing request and displays the returned information. The server loads the models and profiles, invokes the processing modules and communicates with the gantry. The interface is therefore an operating tool for the design rather than the source of its classification or synthesis calculations.',
 'The server connects the processing modules, stored models, writer profiles and motion interpreter. The browser supplies images or writing requests and displays the outputs; the handwriting calculations remain in the numerical processing modules.'),
('For reproduction, the selected writer profile and digital text determine a generated job.',
 'For reproduction, the selected writer profile and digital text determine a generated job. New text is supported within the configured character set; characters absent from a writer\'s profile may require the fallback paths described in Section~\\ref{sec:des-layout}.'),
('Library usage is separated by function.',
 'The complete source code is supplied separately from this report for the AMS submission. The algorithm blocks expose the important control flow, while the executable modules retain the full implementation.\n\nLibrary usage is separated by function.'),
])
print(DST)
