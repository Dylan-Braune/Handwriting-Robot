from pathlib import Path
import re,json
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'Report/DesignResultsDiscussion_2026-11-01'
p=B/'3_Main_Report/4_Results.tex';s=p.read_text();moved=[]
for label in ['fig:res-r3','fig:res-r4','fig:res-geometry']:
 pat=r'\\begin\{figure\}\[!htbp\].*?\\label\{'+re.escape(label)+r'\}\\end\{figure\}'
 # Select the containing float without crossing another begin marker.
 for match in re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}',s,re.S):
  if '\\label{'+label+'}' in match.group():
   moved.append(match.group());s=s[:match.start()]+s[match.end():];break
p.write_text(s,encoding='utf-8')
p=B/'Appendices/results_evidence.tex';a=p.read_text();a+='\n\\subsection{Supporting timing and dimensional figures}\n'+'\n'.join(moved)+'\n';p.write_text(a,encoding='utf-8')
# A portable aggregate review copied from the preparation harness.
src=(ROOT/'tmp/report_context_review/prepare_results_revision.py').read_text();tail=src[src.index('def load(n):'):]
tail=tail.replace("print('style',m['archive_style']['all_nine'],m['archive_style']['shape_seven'])", "print('style',m['archive_style']['all_nine'],m['archive_style']['shape_seven'])")
prefix="from pathlib import Path\nimport json,re,hashlib,math\nimport numpy as np\nfrom collections import Counter,defaultdict\nB=Path(__file__).resolve().parent\nROOT=B.parents[1]\nE=B/'ResultsEvidence'\narchive=E/'archive'\nproposal=ROOT/'ProjectProposal2026DylanFinalRev0 - Dylan Braune-1.pdf'\n"
(B/'review_results.py').write_text(prefix+tail,encoding='utf-8')
print('Moved three supporting plots to the qualification appendix and preserved the review harness.')
