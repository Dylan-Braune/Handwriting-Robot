from pathlib import Path
import re,json
R=Path(__file__).resolve().parents[2];B=R/'Report/IntegratedReport_2026-11-01'
pending=[]; missing_labels=[]; source_checks=set(); citation_changes=[]
def args_at(t,start,n):
 pos=start; out=[]
 for _ in range(n):
  while pos<len(t) and t[pos].isspace():pos+=1
  assert t[pos]=='{',(t[start:start+100],pos)
  first=pos+1; depth=1;pos+=1
  while depth:
   if t[pos]=='{' and (pos==0 or t[pos-1]!='\\'):depth+=1
   elif t[pos]=='}' and (pos==0 or t[pos-1]!='\\'):depth-=1
   pos+=1
  out.append(t[first:pos-1])
 return out,pos
def strip_macro(t,cmd,n,file):
 while (m:=re.search(r'\\'+cmd+r'\b',t)):
  args,end=args_at(t,m.end(),n)
  if cmd=='designgap':pending.append(dict(file=file,kind='Author record or check',text=args[0]))
  else:source_checks.add(args[0])
  t=t[:m.start()]+t[end:]
 return t
def clean_cite(m):
 opt=m.group(1);key=m.group(2)
 if not opt:return m.group(0)
 page=re.search(r'\b(pp?\.\s*~?\s*\d+(?:(?:--|[–-]|,\s*)\d+)*)',opt)
 result='\\cite'+('['+page.group(1)+']' if page else '')+'{'+key+'}'
 citation_changes.append(dict(before=m.group(0),after=result))
 return result
files=[*B.glob('3_Main_Report/**/*.tex'),*B.glob('Appendices/*.tex')]
for p in files:
 rel=str(p.relative_to(B)); t=p.read_text(encoding='utf-8')
 if p.name=='results_evidence.tex':
  a=t.index(r'\subsection{Planned repeated-word');b=t.index(r'\subsection{Cost, consent',a)
  pending.append(dict(file=rel,kind='Physical overlay experiment',text=t[a:b]))
  t=t[:a]+t[b:]
 def figure(m):
  s=m.group(0); a=re.search(r'\\designasset\{([^}]+)\}',s)
  if not a:return s
  if (B/'Figures'/a.group(1)).exists():return s
  pending.append(dict(file=rel,kind='Figure',text=s))
  missing_labels.extend(re.findall(r'\\label\{([^}]+)\}',s));return ''
 t=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',figure,t,flags=re.S)
 t=strip_macro(t,'designgap',1,rel)
 t=strip_macro(t,'designsourceflag',1,rel)
 t=re.sub(r'\\cite(?:\[([^\]]*)\])?\{([^}]+)\}',clean_cite,t)
 for pat,rep in {
  ' Their relationship within the completed assembly is identified in Figure~\\ref{fig:des-assembly}.':'',
  ' Figure~\\ref{fig:des-complete-circuit} shows the complete supply and signal connections, with component values and the common reference identified.':' Table~\\ref{tab:des-pin-map} records the supply and signal connections.',
  ' The installed camera position, page restraint and illumination arrangement are identified in Figure~\\ref{fig:des-capture}.':'',
  ' The accepted and rejected variants in Figure~\\ref{fig:des-glyphquality} illustrate the distinction between retaining handwriting variation and retaining extraction errors.':' The filtering distinguishes handwriting variation from implausible extraction geometry.',
  'Figure~\\ref{fig:des-joining}':'The joining rules',
  'The printed material must be recorded before a particular recycling route can be specified.':'The printed-part material is not identified in the retained record, so no material-specific recycling claim is made.',
  "The user has confirmed signed consent from the five contributors; institutional ethics status and the final material and expenditure records still need to be completed. These are documentation items separate from the measured engineering shortfalls. ":'',
  "Signed consent from the five contributors has been confirmed by the author. The institution's clearance or exemption decision is a separate matter and must be stated with its reference, where one exists.":"Signed consent was obtained from the five personal contributors. No institutional clearance or exemption reference is included in the retained records; participant consent is not presented as evidence of institutional approval.",
  'A redacted consent template may be supplied where institutionally required. The institutional clearance or exemption reference, printed-part material, caliper identification and measured lighting record remain author-supplied completion items.':'Participant consent is recorded separately from institutional clearance or exemption.',
  'The final account must establish which are already included in the estimate; no exact total is produced by adding overlapping entries.':'Their overlap with the approximate build estimate is not established, so they are not added to produce an unsupported exact total.',
  'Extended Modified National Institute of Standards and Technology':'Extended MNIST',
 }.items():t=t.replace(pat,rep)
 t=re.sub(r'(?<!\\)%[^\n]*','',t)
 t=re.sub(r'\n{3,}','\n\n',t)
 p.write_text(t,encoding='utf-8')

# Remove dormant blue-note/placeholder definitions from the typeset-support file.
p=B/'design_support.tex';t=p.read_text()
t=re.sub(r'\\newcommand\{\\designsourceflag\}.*?\n','',t)
t=re.sub(r'\\newcommand\{\\designgap\}.*?\n','',t)
a=t.index(r'\newcommand{\designasset}'); b=t.index(r'\lstdefinestyle',a)
t=t[:a]+r'\newcommand{\designasset}[3]{\includegraphics[width=#2\linewidth]{Figures/#1}}'+'\n'+t[b:]
t=re.sub(r'(?<!\\)%[^\n]*','',t)
p.write_text(t,encoding='utf-8')

# Correct existing bibliographic records rather than introducing aliases.
p=B/'references.bib'; t=p.read_text(encoding='utf-8')
t=t.replace('Unpaginated; section headings used as locators','')
t=t.replace('Official consolidated text, sections 10--11, 14 and 19. ','Official consolidated text. ')
a=t.index('@article{hemanth2022');b=t.index('\n@',a+1);s=t[a:b].replace('year={2022}, month={Jan}','year={2021}, month={Oct}');t=t[:a]+s+t[b:]
a=t.index('@inproceedings{ma2023');b=t.index('\n@',a+1);s=t[a:b].replace('pages={275--280}','pages={268--273}').replace('year={2023}}','year={2023}, doi={10.5220/0012801000003885}}');t=t[:a]+s+t[b:]
t=re.sub(r'(?<!\\)%[^\n]*','',t)
p.write_text(t,encoding='utf-8')

notes=['# Author completion list and source checks','',
 'These items were removed from the typeset report at the author\'s request. Removing a note does not establish the missing measurement or fact. Existing limitations and the scope of the reported evidence remain explicit.','',
 '## Records and figures removed from the manuscript','']
for i,item in enumerate(pending,1):notes.extend([f'### {i}. {item["kind"]}: {item["file"]}','',item['text'],''])
notes.extend(['## Sources without verified page locators','',
 'The report cites these sources without a page number where no verified locator was available. Check the relevant edition or paper manually; no page has been invented.',''])
notes.extend('- `'+key+'`' for key in sorted(source_checks))
notes.extend(['','## Other final records','',
 '- Institutional ethics-clearance or exemption reference, if applicable.',
 '- Reconciled budget, printed-part material, caliper make/model/least count, and measured lighting record.',
 '- Final whole-report assembly, including front matter and Conclusion, and its overall 70-page main-text budget.',
 '- Missing figures are excluded rather than presented as empty or completed figures. Their labels and instructions are preserved above.',''])
(B/'AUTHOR_COMPLETION_LIST.md').write_text('\n'.join(notes),encoding='utf-8')
(B/'SourceChecks/citation_cleanup.json').write_text(json.dumps(dict(citation_changes=citation_changes,removed_figure_labels=missing_labels,removed_notes=len(pending)),indent=2)+'\n')
print(json.dumps(dict(notes=len(pending),removed_figure_labels=missing_labels,citation_changes=len(citation_changes)),indent=2))
