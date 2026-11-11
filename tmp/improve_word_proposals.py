from pathlib import Path
p=Path('Report/GeometricFidelity_2026-11-02/annotate_words.py');s=p.read_text()
s=s.replace('def clean(a):return SEG.BinarizeLine(a,stripRules=True,closeGaps=False)', '''def clean(a):
 m=SEG.BinarizeLine(a,stripRules=True,closeGaps=False);band=SEG.CoreBand(m)
 if band:
  t,b=band;xh=b-t;nn,lab,stats,cents=cv2.connectedComponentsWithStats(m.astype('uint8'),8)
  for i in range(1,nn):
   x,y,w,h,area=stats[i];cy=cents[i,1]
   if cy<t-.75*xh or cy>b+.75*xh or (h>3*xh and w<.12*xh):m[lab==i]=False
 return m''')
s=s.replace('  chosen=sorted(sorted(gg,key=lambda q:-(q[1]-q[0]))[:n-1]);cuts=[0]+[(s+e)//2 for s,e in chosen]+[mask.shape[1]]','''  # Transcript-length priors select word gaps, without looking at the matching image.
  from scipy.ndimage import gaussian_filter1d
  tokens=r['text'].split();prof=gaussian_filter1d(mask.sum(0).astype(float),1.0)
  cand={x0:0.,x1:0.}
  for st,en in gg:cand[(st+en)//2]=-(2*np.log1p((en-st)/max(3,(y1-y0)/2)))
  for xx in range(x0+2,x1-2):
   if prof[xx]<=prof[xx-1] and prof[xx]<prof[xx+1]:cand.setdefault(xx,5*prof[xx]/max(1,y1-y0))
  xx=np.array(sorted(cand));cost=np.array([cand[q] for q in xx]);xx=xx[(xx>=x0)&(xx<=x1)];cost=np.array([cand[q] for q in xx]);N=len(xx)
  def letters(t):return sum(.45 if c in 'ilI1.,:;' else .65 if c in 'ftjr' else 1.35 if c in 'mwMW' else 1.0 for c in t)
  lens=np.array([letters(t) for t in tokens]);unit=(x1-x0)/(lens.sum()+1.3*(n-1));want=(lens+1.3)*unit;want[[0,-1]]-=.65*unit
  dp=np.full((n+1,N),np.inf);prev=np.full((n+1,N),-1,int);dp[0,0]=0
  for j in range(1,n+1):
   for q in range(1,N):
    widths=xx[q]-xx[:q];sigma=.35*want[j-1]+.6*unit
    losses=dp[j-1,:q]+.5*((widths-want[j-1])/sigma)**2+cost[q]
    losses[widths<.2*unit]=np.inf
    k=int(np.argmin(losses));dp[j,q]=losses[k];prev[j,q]=k
  at=N-1;cuts=[int(xx[at])]
  for j in range(n,0,-1):at=prev[j,at];cuts.append(int(xx[at]))
  cuts=cuts[::-1]''')
s=s.replace("  if key in ann:cuts=ann[key]['cuts']", "  if key in ann:cuts=ann[key]['cuts']")
p.write_text(s)
