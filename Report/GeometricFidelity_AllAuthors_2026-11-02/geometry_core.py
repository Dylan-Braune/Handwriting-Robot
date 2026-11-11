from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import distance_transform_edt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent/'source'))
import segment as SEG
TOLS=[.05,.10,.15,.20]

def canvas(a,b):
 h=max(a.shape[0],b.shape[0])+50;w=max(a.shape[1],b.shape[1])+50;out=[]
 for m in [a,b]:
  z=np.zeros((h,w),bool);y=(h-m.shape[0])//2;x=(w-m.shape[1])//2;z[y:y+m.shape[0],x:x+m.shape[1]]=m;out.append(z)
 return out

def compare(a,b,return_masks=False):
 a,b=canvas(a,b);sa=SEG.Skeletonize(a);sb=SEG.Skeletonize(b)
 ya,xa=np.where(sa);yb,xb=np.where(sb)
 if not(len(xa) and len(xb)):return {'error':'empty skeleton'}
 da=distance_transform_edt(~sa);db=distance_transform_edt(~sb)
 best=(np.inf,0,0)
 for dy in range(-5,6):
  for dx in range(-5,6):
   dist=.5*(da[yb+dy,xb+dx].mean()+db[ya-dy,xa-dx].mean())
   if (dist,dy*dy+dx*dx,dy,dx)<(best[0],best[1]*best[1]+best[2]*best[2],best[1],best[2]):best=(float(dist),dy,dx)
 _,dy,dx=best;bb=np.roll(b,(dy,dx),axis=(0,1));bs=np.roll(sb,(dy,dx),axis=(0,1));dbs=distance_transform_edt(~bs)
 def overlap(x,y):
  inter=int((x&y).sum());union=int((x|y).sum());return inter/union if union else 0.,2*inter/(x.sum()+y.sum()) if x.sum()+y.sum() else 0.
 iou,dice=overlap(a,bb);an=da<=.085*48;bn=dbs<=.085*48;ni,nd=overlap(an,bn)
 dba=da[bs];dab=dbs[sa];out={'iou':float(iou),'dice':float(dice),'normalised_iou':float(ni),'normalised_dice':float(nd),'mean_distance_xh':best[0]/48,'dx':dx,'dy':dy,'original_points':len(xa),'physical_points':len(xb)}
 for t in TOLS:
  p=float((dba<=48*t).mean());r=float((dab<=48*t).mean());out[f'precision_{t:.2f}']=p;out[f'recall_{t:.2f}']=r;out[f'f1_{t:.2f}']=2*p*r/(p+r) if p+r else 0.
 vals=[overlap(a,dbs<=rad)[0] for rad in range(13)];k=int(np.argmax(vals));out.update(adaptive_iou=float(vals[k]),adaptive_radius=k)
 if return_masks:return out,dict(original=a,physical=bb,original_skeleton=sa,physical_skeleton=bs)
 return out
