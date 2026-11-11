from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from geometry_core import SEG

class FastGeometry:
 def __init__(self, arrays):
  self.h=max(a.shape[0] for a in arrays.values())+54;self.w=max(a.shape[1] for a in arrays.values())+54;self.items={}
  for uid,a in arrays.items():
   h,w=a.shape;oy=(self.h-h)//2;ox=(self.w-w)//2;s=SEG.Skeletonize(a);y,x=np.where(s);yy=y+oy;xx=x+ox;mask=np.zeros((self.h,self.w),bool);mask[yy,xx]=1
   self.items[uid]={'h':h,'w':w,'oy':oy,'ox':ox,'y':yy,'x':xx,'dt':distance_transform_edt(~mask)}
  dy,dx=np.meshgrid(np.arange(-5,6),np.arange(-5,6),indexing='ij');self.dy=dy.ravel();self.dx=dx.ravel()
 def score(self,aid,bid):
  a=self.items[aid];b=self.items[bid];ph=max(a['h'],b['h'])+50;pw=max(a['w'],b['w'])+50
  cy=((ph-b['h'])//2-(ph-a['h'])//2)-(b['oy']-a['oy']);cx=((pw-b['w'])//2-(pw-a['w'])//2)-(b['ox']-a['ox'])
  yy=self.dy+cy;xx=self.dx+cx
  ba=a['dt'][b['y'][None,:]+yy[:,None],b['x'][None,:]+xx[:,None]];ab=b['dt'][a['y'][None,:]-yy[:,None],a['x'][None,:]-xx[:,None]]
  means=.5*(ba.mean(1)+ab.mean(1));i=int(np.lexsort((self.dx,self.dy,self.dy**2+self.dx**2,means))[0]);p=float((ba[i]<=4.8).mean());r=float((ab[i]<=4.8).mean());f=2*p*r/(p+r) if p+r else 0.
  return {'f1':f,'precision':p,'recall':r,'mean_distance_xh':float(means[i])/48,'dx':int(self.dx[i]),'dy':int(self.dy[i]),'physical_points':len(b['x']),'original_points':len(a['x'])}
