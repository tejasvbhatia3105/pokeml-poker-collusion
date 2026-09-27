\
\
\
\
\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl,torch
import session89_crop_list_training as engine
from session11_conditional_family import features,template,list_nll,log_count_at_least
from session88_crop_audit import refresh,WINDOWS
ROOT=Path('artifacts/evidence_session90_partial_crop_list');C=pl.col;ARM='all'
def configure(arm):
 global ARM
 ROOT.mkdir(exist_ok=True)
 ARM=arm;engine.ROOT=ROOT/arm;engine.CONFIG=dict(engine.CONFIG,method=__doc__,arm=arm,objective='original full-list conditional NLL with omitted-hand teacher probabilities frozen; corrections and regularization only on visible rows');engine.pack=pack;engine.objective=objective
def pack(d,f,augment=True):
 q=d.join(pl.read_parquet(Path(engine.CONFIG['input_root'])/f'nested_outer{f}.parquet').drop('fold','time'),on=['pair_id','hand_id'],validate='1:1');groups=[];originals=[];weights=[];complete=[]
 for (pid,),g in sorted(q.group_by('pair_id'),key=lambda t:t[0][0]):
  g=g.sort('time','hand_id').with_row_index('full_position');views=[('full',g)];ti=np.rint(g['time'].to_numpy()*5000).astype(int)
  if augment:
   for window,(lo,hi) in WINDOWS.items():
    if window=='full':continue
    z=g.filter(pl.Series((ti>=round(lo*5000))&(ti<round(hi*5000))))
    if len(z) and (ARM=='all' or z['evidence'].sum()==g['evidence'].sum()):views.append((window,refresh(z.with_columns(((C('time')-lo)/(hi-lo)).alias('relative_time')))))
  total=sum(engine.CONFIG['view_weights'][w] for w,z in views)
  for w,z in views:groups.append((w,z));originals.append(g);weights.append(engine.CONFIG['view_weights'][w]/total);complete.append(z['evidence'].sum()==g['evidence'].sum())
 N=len(groups);n=max(map(len,originals));P=np.zeros((N,n,2),np.float32);M=np.zeros((N,n),bool);F=np.zeros_like(M);T=np.zeros((N,6,n),np.int64);V=np.zeros((N,6),bool);D=np.zeros(N,np.float32);fv=np.zeros(N,int);parts=[];bid=[];pos=[]
 for i,((w,g),original) in enumerate(zip(groups,originals)):
  x,p=features(g);_,op=features(original);k=len(original);positions=g['full_position'].to_numpy();P[i,:k]=op;P[i,positions]=p;F[i,:k]=True;M[i,positions]=True;e=np.flatnonzero(original['evidence_rank'].is_not_null());e=e[np.argsort(original['evidence_rank'].to_numpy()[e])];t=template(k,e);T[i,:len(t),:k]=t;V[i,:len(t)]=True;D[i]=len(e);fv[i]=g['fold'][0];parts.append(np.column_stack([x,g.select(engine.GROUND_COLUMNS).to_numpy()]));bid.extend([i]*len(g));pos.extend(positions.tolist())
 X=np.nan_to_num(np.concatenate(parts),nan=0,posinf=1e6,neginf=-1e6);families=np.array([g['behavior_family'][0] for w,g in groups]);tr=(fv!=f)&V.any(1);minimums={fam:int(D[tr&(families==fam)].min()) for fam in set(families)};K=torch.tensor([minimums[v] for v in families]);return {'groups':groups,'originals':originals,'X':X,'P':torch.tensor(P),'M':torch.tensor(M),'F':torch.tensor(F),'T':torch.tensor(T),'V':torch.tensor(V),'D':torch.tensor(D),'W':torch.tensor(weights,dtype=torch.float32),'fv':fv,'tr':tr,'bid':np.array(bid),'pos':np.array(pos),'K':K,'minimums':minimums,'predict_views':np.array(complete)}
def objective(z,A):
 tr=torch.tensor(z['tr']);visible=A[tr]*z['M'][tr,:,None];P=z['P'][tr]+visible;loss=list_nll(P,z['T'][tr],z['V'][tr],z['D'][tr])+log_count_at_least(P,z['F'][tr],z['K'][tr])/z['D'][tr];reg=visible.square().sum((1,2))/z['M'][tr].sum(1);return ((loss+engine.CONFIG['ridge']*reg)*z['W'][tr]).sum()
def main():
 import sys
 for arm in (sys.argv[1].split(',') if len(sys.argv)>1 else ['complete','all']):
  assert arm in ['complete','all'];configure(arm);engine.main()
if __name__=='__main__':main()
