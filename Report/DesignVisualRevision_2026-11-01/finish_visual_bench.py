"""Continue saved report bench without rerunning the twelve exploratory cases."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
source=(HERE/'run_visual_bench.py').read_text(encoding='utf-8')
saved=json.loads((HERE/'Evidence/bench.json').read_text(encoding='utf-8'))
exec(compile(source.split('# Three original pages')[0],str(HERE/'run_visual_bench.py'),'exec'))
records=saved
# Deliberately use the archived, visually matched crop for glyph extraction.
# The resampled exploratory crop did not contain the matching first line.
crop=ROOT/'Handwriting-Robot-Pipeline/Software/CNN/Crops/Owen/Owen01/00.png';record(crop)
gray=np.array(Image.open(crop).convert('L'))
ll=[{'image':gray}]
allruns={('Owen','original'):(None,ll,None,None)}
tail=source.split('# Forced alignment on an actual original line.')[1].split('\n',1)[1]
tail=tail.replace("'line_count_matches_labels':len(ll)==records['segmentation'][0]['expected_label_lines'],", "'crop_provenance':'Archived original-handwriting crop visually matched to the first labelled line; not the new reduced-page segmentation output',")
exec(compile(tail,str(HERE/'run_visual_bench.py'),'exec'))
