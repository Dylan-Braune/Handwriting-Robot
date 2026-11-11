from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
s=(HERE/'run_visual_bench.py').read_text(encoding='utf-8')
exec(compile(s.split('# Three original pages')[0],str(HERE/'run_visual_bench.py'),'exec'))
p=ROOT/'Software/CNN/PersonalDataset/Owen/Pages/Owen01.jpg'
a=np.array(Image.open(p).convert('L'));upright=seg.UprightPage(a)[0]
full,half,mask,ratio,clean,overlay,angle=stages(upright)
ll=[np.array(Image.open(p).convert('L')) for p in sorted((OUT/'native_Owen01').glob('line_*.png'))]
c=pdf('design-segmentation-stages',720,475)
for j,(im,title) in enumerate([(upright,'(a) Original Owen01'),(np.uint8(np.clip(ratio/1.2,0,1)*255),'(b) Local brightness ratio'),
                               (np.uint8(~mask)*255,'(c) Thresholded ink'),(overlay,'(d) Actual seam locations')]):
    panel(c,im,15+j*176,130,165,310,title)
text(c,15,110,f'Native input: {a.shape[1]} x {a.shape[0]}; {len(ll)} crops; deskew {angle:+.1f} deg.',10)
for i in range(2):panel(c,ll[i],15,58-i*48,690,32,f'Output line {i+1}')
finish(c,'design-segmentation-stages')
