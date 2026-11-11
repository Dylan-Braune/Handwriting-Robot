"""Documentation ordering only; never imports project modules."""
import re

STAGES={
'paths.py':['checkpoint'],
'web/camera.py':['camera'],
'segment.py':['integral','morph','threshold','components','filter','resize','pitch','deskew','seam','ownership','skeleton','simplify'],
'recognise.py':['checkpoint','resize','skeleton','conv','bn','activation','pool','softmax','lstm','ctc','metrics'],
'style.py':['trace','simplify','glyph','descriptor','medoid'],
'curate.py':['curate','medoid','render'],
'synthesise.py':['synth','render','ctc'],
'retrace.py':['retrace'],
'reproduce.py':['motion','calibrate','pen','bresenham','gcode'],
'web/server.py':['deskew','metrics','motion','server'],
'web/lineworkers.py':['workers'],
'train.py':['codec','dataset','augment','init','conv','bn','activation','pool','dropout','lstm','softmax','ctc','crossentropy','backward','optim','checkpoint'],
'timing.py':['timing'],
'web/render.py':['render'],
}
FLOW={
'paths.py':'Resolve datasets, personal writer pages and checkpoint locations before loading images or networks. Paths establish provenance; they do not establish which model is deployed.',
'web/camera.py':'Acquire a still image or preview frame under a shared camera lock, then pass captured pixels to the page-processing stage. Preview lifecycle and still capture share the device.',
'segment.py':'Start with pixel sums and neighbourhood operations; estimate paper and foreground; infer text scale and orientation; find separator seams; repair ink ownership; return ordered line crops. Resampling, drawing and thinning helpers support several of those stages and later glyph extraction.',
'recognise.py':'Load compatible arrays; fit and optionally normalise a line; form convolutional features; apply the writer or recurrent text head; decode or align CTC; score the result. Writer pooling and recurrent text decoding are parallel heads, not successive compulsory stages.',
'style.py':'Thin and trace labelled line ink; align transcript frames and cut characters; normalise glyph coordinates; reject implausible shapes; select medoids and optional prototypes; assemble the writer profile consumed by synthesis.',
'curate.py':'Inspect the stored profile and its variants, choose or remove entries, then persist the curated profile. This is profile preparation rather than inference-time model training.',
'synthesise.py':'Read a writer profile and text; choose eligible source glyphs and calibration; lay out words and joins; render candidates; use text/writer judges and accepted local repair to select a trajectory. The text and writer judges score a candidate rather than train during the request.',
'retrace.py':'Take page pixels through line segmentation, skeleton tracing and pruning, then map recovered trajectories for reproduction. The current SegmentLines call mismatch is described in the pipeline audit; this is not a verified successful run.',
'reproduce.py':'Establish units, workspace and axis calibration; map and order trajectories; describe pen state and switches; generate coordinated integer pulses and G-code; interpret the job with stop and connection checks. Hardware sizing explains assumptions before the motion equations.',
'web/server.py':'Load the networks and profiles; ingest and orient an image; segment/read/name its lines; orchestrate classification, generation or retracing; expose reports, G-code, calibration and writing routes. API endpoints are alternative request paths, not a single sequential call chain.',
'web/lineworkers.py':'Configure inherited readers/classifiers; decide whether pooling is eligible; initialise per-process thread limits; map independent line jobs and preserve order, with serial fallback. The server remains responsible for page-level orchestration.',
'train.py':'Decode source images and labels; construct page-separated data views and batches; augment and initialise; run train-mode forward passes; calculate losses and reverse derivatives; clip/update parameters; validate and save checkpoints. This chapter explains the report training methods and is outside live demo inference.',
'timing.py':'Measure elapsed stage intervals and aggregate a request record. Compare those measurements with nominal motion allowances only after checking the stage boundary, host and completion status.',
'web/render.py':'Render stored glyphs and sample crops into inspection sheets after profiles or datasets exist. These outputs support inspection rather than replace recognition or physical validation.',
}

# Every numbered report equation has exactly one explanatory home.
EQ_HOME={**{n:('reproduce.py','motion') for n in range(1,7)},
         **{n:('reproduce.py','pen') for n in range(7,10)},
         10:('reproduce.py','calibrate'),11:('reproduce.py','bresenham'),
         12:('segment.py','integral'),13:('segment.py','threshold'),14:('segment.py','threshold'),
         15:('segment.py','deskew'),16:('segment.py','seam'),17:('segment.py','seam'),
         18:('recognise.py','resize'),19:('recognise.py','conv'),20:('recognise.py','bn'),
         21:('recognise.py','lstm'),22:('recognise.py','ctc'),23:('train.py','ctc'),
         24:('recognise.py','skeleton'),25:('style.py','glyph'),26:('style.py','glyph'),
         27:('style.py','glyph'),28:('style.py','glyph'),29:('synthesise.py','synth'),
         30:('synthesise.py','synth'),31:('synthesise.py','synth'),
         32:('reproduce.py','motion'),33:('reproduce.py','gcode'),34:('recognise.py','metrics'),
         35:('verification','metrics')}

def route_report(raw):
    routed={}
    cards=re.findall(r'\\ReportEq\{(\d+)\}.*?(?=\\ReportEq\{|\\section\{|\Z)',raw,flags=re.S)
    blocks=re.findall(r'(\\ReportEq\{\d+\}.*?)(?=\\ReportEq\{|\\section\{|\Z)',raw,flags=re.S)
    assert len(cards)==len(blocks)==35
    index=[r'\chapter{Report-to-code reading map}\label{ch:report-first}',
           'The report is the priority map. Its explanations are placed beside the relevant code stages, rather than collected in a detached report chapter. All 35 numbered equations have a home below. Each module opens with its input-to-output flow; its detailed stages group dependent calculations and their supporting functions. Training remains outside the live path.',
           r'\Q{Evidence}{Reading the evidence}',
           'The reference report is the unsuffixed '+r'\path{23539195_BrauneDR_FINALREPORT.pdf}'+' in Downloads (108 PDF pages). Published sources establish methods; captured code establishes implementation; worked arithmetic is a derivation. Trade-off explanations are not invented historical decision records. Unsupported coefficient choices and missing sources are identified explicitly.',
           r'\begin{longtable}{rp{0.78\linewidth}}\toprule Report equation & Explanation in code context\\\midrule\endhead']
    for n,b in zip(cards,blocks):
        num=int(n);home=EQ_HOME[num]
        routed[home]=routed.get(home,'')+b+'\n'
        title=re.search(r'\\ReportEq\{\d+\}\{([^}]+)\}',b).group(1)
        index.append(n+' & '+r'\hyperref[report-eq:'+n+']{'+title+'} ('+home[0].replace('_',r'\_')+r', page~\pageref{report-eq:'+n+r'})\\')
    index.append(r'\bottomrule\end{longtable}')
    tail=raw.split(r'\section{Other named report methods that must not disappear}',1)[1]
    extras=re.split(r'(?=\\Q\{)',tail)
    homes=[('segment.py','threshold'),('recognise.py','conv'),('synthesise.py','synth'),('style.py','descriptor'),('reproduce.py','motion'),('verification','metrics'),('verification','metrics'),('verification','metrics'),('reproduce.py','motion')]
    chunks=[x for x in extras if x.startswith(r'\Q{')]
    assert len(chunks)==len(homes),(len(chunks),len(homes))
    for home,chunk in zip(homes,chunks):
        chunk=re.sub(r'\\section\{Report algorithm and figure reading map\}', '',chunk)
        routed[home]=routed.get(home,'')+chunk
    return '\n'.join(index),routed
