\
\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl,torch,joblib
from sklearn.tree import DecisionTreeRegressor
from session8_data import hand_data
from session11_conditional_family import features,template,list_nll,log_count_at_least
from session65_list_pressure_inference import GROUND_COLUMNS,tree_delta
from session8_count_conditioning import conditioned
from session88_crop_audit import refresh,WINDOWS
ROOT=Path('artifacts/evidence_session89_crop_list_training');C=pl.col
CONFIG={'iterations':120,'learning_rate':.15,'max_leaf_nodes':10,'min_samples_leaf':80,'max_features':.7,'ridge':.02,'bound':1.5,'seed':1111,'view_weights':{'full':1.,'first_2000':.5,'last_2000':.5},'input_root':'artifacts/evidence_session55_current_nested','objective':'conditional exact list NLL / truth count + mean squared correction, weighted views summing1 per relationship','method':__doc__}
def data():
 d=hand_data();x=np.load('artifacts/evidence_session62_grounded_list_boost/grounded_features.npz')['x'];return d.with_columns(*[pl.Series(n,x[:,i]) for i,n in enumerate(GROUND_COLUMNS)])
def pack(d,f,augment=True):
 q=d.join(pl.read_parquet(Path(CONFIG['input_root'])/f'nested_outer{f}.parquet').drop('fold','time'),on=['pair_id','hand_id'],validate='1:1');groups=[];weights=[]
 for (pid,),g in sorted(q.group_by('pair_id'),key=lambda t:t[0][0]):
  g=g.sort('time','hand_id');views=[('full',g)];ti=np.rint(g['time'].to_numpy()*5000).astype(int)
  if augment:
   for window,(lo,hi) in WINDOWS.items():
    if window=='full':continue
    z=g.filter(pl.Series((ti>=round(lo*5000))&(ti<round(hi*5000))))
    if len(z) and z['evidence'].sum()==g['evidence'].sum():views.append((window,refresh(z.with_columns(((C('time')-lo)/(hi-lo)).alias('relative_time')))))
  total=sum(CONFIG['view_weights'][w] for w,z in views)
  for w,z in views:groups.append((w,z));weights.append(CONFIG['view_weights'][w]/total)
 N=len(groups);n=max(len(g) for w,g in groups);P=np.zeros((N,n,2),np.float32);M=np.zeros((N,n),bool);T=np.zeros((N,6,n),np.int64);V=np.zeros((N,6),bool);D=np.zeros(N,np.float32);fv=np.zeros(N,int);parts=[];bid=[];pos=[]
 for i,(w,g) in enumerate(groups):
  x,p=features(g);k=len(g);P[i,:k]=p;M[i,:k]=True;e=np.flatnonzero(g['evidence_rank'].is_not_null());e=e[np.argsort(g['evidence_rank'].to_numpy()[e])];t=template(k,e);T[i,:len(t),:k]=t;V[i,:len(t)]=True;D[i]=len(e);fv[i]=g['fold'][0];parts.append(np.column_stack([x,g.select(GROUND_COLUMNS).to_numpy()]));bid.extend([i]*k);pos.extend(range(k))
 X=np.nan_to_num(np.concatenate(parts),nan=0,posinf=1e6,neginf=-1e6);families=np.array([g['behavior_family'][0] for w,g in groups]);tr=(fv!=f)&V.any(1);minimums={fam:int(D[tr&(families==fam)].min()) for fam in set(families)};K=torch.tensor([minimums[v] for v in families]);return {'groups':groups,'X':X,'P':torch.tensor(P),'M':torch.tensor(M),'T':torch.tensor(T),'V':torch.tensor(V),'D':torch.tensor(D),'W':torch.tensor(weights,dtype=torch.float32),'fv':fv,'tr':tr,'bid':np.array(bid),'pos':np.array(pos),'K':K,'minimums':minimums}
def objective(z,A):
 tr=torch.tensor(z['tr']);P=z['P'][tr]+A[tr];loss=list_nll(P,z['T'][tr],z['V'][tr],z['D'][tr])+log_count_at_least(P,z['M'][tr],z['K'][tr])/z['D'][tr];reg=(A[tr].square().sum(2)*z['M'][tr]).sum(1)/z['M'][tr].sum(1);return ((loss+CONFIG['ridge']*reg)*z['W'][tr]).sum()
def scores(g,p,model,x):
 z=p+tree_delta(model,x);pr=torch.softmax(torch.tensor(np.column_stack([np.zeros(len(g),np.float32),z])),1).numpy();inc=conditioned(pr[:,1:],model['minimums'][g['behavior_family'][0]]);return .25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));d=data();parts=[];audit=[];start=time.time();torch.set_num_threads(3)
 for f in range(4):
  z=pack(d,f);X=z['X'];bid=z['bid'];pos=z['pos'];rowtrain=z['tr'][bid];rowvalid=z['fv'][bid]==f;P=z['P'];M=z['M'];A=torch.zeros_like(P);rw=z['W'].numpy()[bid]/z['D'].numpy()[bid];model={'steps':[],'fold':f,'kind':'crop_augmented','config':CONFIG,'minimums':z['minimums'],'grounded_columns':GROUND_COLUMNS,'feature_count':X.shape[1]};trace=[];initial=float(objective(z,A));checkpoint=ROOT/f'list_boost_fold{f}.joblib';resume=joblib.load(checkpoint) if checkpoint.exists() else None
  if resume is not None:
   assert resume['config']==CONFIG and resume['fold']==f and len(resume['steps'])==120;model=resume
  for it in range(CONFIG['iterations']):
   if resume is not None:
    tree,step=resume['steps'][it];a=A.numpy()[bid,pos];update=tree.predict(X).astype(np.float32);A=torch.zeros_like(P);A[bid,pos]=torch.tensor(np.clip(a+step*update,-1.5,1.5));trace.append(float(objective(z,A)));continue
   A.requires_grad_(True);loss=objective(z,A);loss.backward();grad=A.grad.detach().numpy()[bid,pos];a=A.detach().numpy()[bid,pos];pr=torch.softmax(torch.cat([torch.zeros_like(P[:,:,:1]),P+A.detach()],2),2).numpy()[bid,pos,1:];h=np.maximum(pr*(1-pr),.02)*rw[:,None]+2*CONFIG['ridge']*z['W'].numpy()[bid,None]/M.sum(1).numpy()[bid,None];target=np.clip(-grad/h,-5,5);tree=DecisionTreeRegressor(max_leaf_nodes=10,min_samples_leaf=80,max_features=.7,random_state=1111+1000*f+it);tree.fit(X[rowtrain],target[rowtrain],sample_weight=h[rowtrain].mean(1));update=tree.predict(X).astype(np.float32);old=float(loss.detach());A=A.detach();step=.15
   for attempt in range(7):
    candidate=torch.zeros_like(P);candidate[bid,pos]=torch.tensor(np.clip(a+step*update,-1.5,1.5));value=float(objective(z,candidate))
    if value<=old+1e-6:break
    step*=.5
   else:step=0;candidate=A;value=old
   A=candidate;model['steps'].append((tree,step));trace.append(value)
  joblib.dump(model,ROOT/f'list_boost_fold{f}.joblib',compress=3);replay_error=float(abs(tree_delta(model,X[rowvalid])-A.numpy()[bid[rowvalid],pos[rowvalid]]).max());assert replay_error<1e-6;control=joblib.load(f'artifacts/evidence_session62_grounded_list_boost/list_boost_full_fold{f}.joblib')
  for i in np.flatnonzero((z['fv']==f)&z.get('predict_views',np.ones(len(z['fv']),bool))):
   window,g=z['groups'][i];mask=bid==i;pp=P.numpy()[i,pos[mask]];parts.append(g.select('pair_id','hand_id','fold','evidence').with_columns(pl.lit(window).alias('window'),pl.Series('tree_augmented',scores(g,pp,model,X[mask])),pl.Series('tree_control',scores(g,pp,control,X[mask]))))
  weights={}
  for (w,g),wt in zip(z['groups'],z['W'].tolist()):weights[g['pair_id'][0]]=weights.get(g['pair_id'][0],0)+wt
  assert max(abs(v-1) for v in weights.values())<1e-6;audit.append({'fold':f,'train_views':int(z['tr'].sum()),'validation_views':int((z['fv']==f).sum()),'training_rows':int(rowtrain.sum()),'per_relationship_weight_sum_error':max(abs(v-1) for v in weights.values()),'training_vs_inference_float_logit_error':replay_error,'resumed_saved_model':resume is not None,'initial_loss':initial,'final_loss':trace[-1],'training_loss':trace,'heldout_row_overlap':int((rowtrain&rowvalid).sum())});print('crop list fit',f,round(time.time()-start,1),flush=True)
 pl.concat(parts).write_parquet(ROOT/'oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
