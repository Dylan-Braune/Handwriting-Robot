from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
s=(HERE/'run_visual_bench.py').read_text(encoding='utf-8')
exec(compile(s.split('# Three original pages')[0],str(HERE/'run_visual_bench.py'),'exec'))
records=json.loads((OUT/'bench.json').read_text(encoding='utf-8'))
model=rec.PaperCRNNNumpy(ROOT/'Software/CNN/weights/paper_cnn_bilstm_ctc_joint_best.npz')
p=ROOT/'Software/CNN/NOGIT/StyleProfiles10/owen_curated.json';record(p)
profiles={'owen':json.loads(p.read_text(encoding='utf-8'))}
exec(compile(s.split('# Candidate scores are diagnostic selection evidence, explicitly normalised at .17.')[1],str(HERE/'run_visual_bench.py'),'exec'))
