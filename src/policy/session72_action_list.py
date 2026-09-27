\
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
from session57_isolation_pressure import data
from session67_isolation_types import design
from session59_pressure_equity import ROOT as PRECISE
from session11_conditional_family import template,list_nll,log_count_at_least
from session8_count_conditioning import conditioned
from session6_priority import inclusion
from session12_compare import compare
ROOT=Path('artifacts/evidence_session72_action_list');C=pl.col
CONFIG={'iterations':200,'learning_rate':.15,'max_leaf_nodes':10,'min_samples_leaf':20,'max_features':.7,'bound':6.,'ridge':.002,'initial_action_probability':.15,'seed':7200}
torch.set_num_threads(2)
def pack():
 _,d,a,ac=data();hc=json.load(open(PRECISE/'config.json'))['hand_columns'];g=a['row'].to_numpy();x=np.column_stack([a.select(ac).to_numpy(),d.select(hc).to_numpy()[g],np.load(PRECISE/'features.npz')['x']]).astype(np.float64);x=np.nan_to_num(x,nan=0,posinf=1e6,neginf=-1e6);tx,tc=design(d,a,ac);groups=[q.sort('time','hand_id') for _,q in d.group_by('pair_id')];groups.sort(key=lambda q:q['pair_id'][0]);n=max(map(len,groups));N=len(groups);T=np.zeros((N,6,n),np.int64);V=np.zeros((N,6),bool);M=np.zeros((N,n),bool);D=np.zeros(N);fv=np.zeros(N,int);bypair={};pos=np.zeros(len(d),int);bid=np.zeros(len(d),int)
 for i,q in enumerate(groups):
  rows=q['row'].to_numpy();pos[rows]=np.arange(len(q));bid[rows]=i;e=np.flatnonzero(q['evidence_rank'].is_not_null());e=e[np.argsort(q['evidence_rank'].to_numpy()[e])];t=template(len(q),e);T[i,:len(t),:len(q)]=t;V[i,:len(t)]=True;M[i,:len(q)]=True;D[i]=len(e);fv[i]=q['fold'][0]
 return d,a,x,groups,torch.tensor(T),torch.tensor(V),torch.tensor(M),torch.tensor(D),fv,bid,pos,tx[:,tc.index('players_active_max')]
def hand_logits(A,indices,N,n):
                                                                             
                                                                                
 logs=torch.zeros(N*n*2,dtype=A.dtype).index_add(0,indices,torch.nn.functional.logsigmoid(-A));p=-torch.expm1(logs.reshape(N,n,2));p1=p[:,:,0];p2=(1-p1)*p[:,:,1];p0=((1-p1)*(1-p[:,:,1])).clamp_min(1e-14);return torch.stack([torch.log(p1.clamp_min(1e-14)/p0),torch.log(p2.clamp_min(1e-14)/p0)],2)
def objective(A,indices,N,n,T,V,M,D,train,K,rowtrain,rowbid):
 z=hand_logits(A,indices,N,n);loss=(list_nll(z[train],T[train],V[train],D[train])+log_count_at_least(z[train],M[train],K[train])/D[train]).sum();counts=torch.bincount(rowbid,minlength=N).clamp_min(1);reg=((A[rowtrain]-BIAS).square()/counts[rowbid[rowtrain]]).sum();return loss+CONFIG['ridge']*reg
BIAS=float(np.log(.15/.85))
def infer(model,x):
 a=np.full(len(x),model['bias'],float)
 for tree,step in model['steps']:a=np.clip(a+step*tree.predict(x),-model['config']['bound'],model['config']['bound'])
 return a
def main():
 ROOT.mkdir(exist_ok=True);d,a,x,groups,T,V,M,D,fv,bid,pos,amax=pack();N,n=M.shape;arow=a['row'].to_numpy();active=a['players_active'].to_numpy();sub=d['subtype'].to_numpy();handfv=d['fold'].to_numpy();outputs=[];audit=[];start=time.time()
 for f in range(4):
  levels=[int(np.unique(amax[(handfv!=f)&(sub==k)])[0]) for k in [1,2]];support=np.isin(active,levels);X=x[support];rows=arow[support];rb=bid[rows];rp=pos[rows];head=(active[support]==levels[1]).astype(int);indices=torch.tensor((rb*n+rp)*2+head);train=torch.tensor(np.flatnonzero(fv!=f));rt=fv[rb]!=f;rowtrain=torch.tensor(rt);rowbid=torch.tensor(rb);K=torch.full((N,),int(D[train].min()));A=torch.full((len(X),),BIAS,dtype=torch.float64);model={'steps':[],'config':CONFIG,'bias':BIAS,'fold':f,'levels':levels,'minimum':int(K[0]),'feature_count':X.shape[1]};history=[];initial=float(objective(A,indices,N,n,T,V,M,D,train,K,rowtrain,rowbid));count=np.bincount(rb,minlength=N);weight=1/np.maximum(count[rb],1);gradvalid=0
  for it in range(CONFIG['iterations']):
   A.requires_grad_(True);loss=objective(A,indices,N,n,T,V,M,D,train,K,rowtrain,rowbid);loss.backward();grad=A.grad.detach().numpy();gradvalid=max(gradvalid,float(abs(grad[~rt]).max(initial=0)));p=torch.sigmoid(A.detach()).numpy();h=np.maximum(p*(1-p),.02)/D.numpy()[rb]+2*CONFIG['ridge']*weight;target=np.clip(-grad/h,-5,5);tree=DecisionTreeRegressor(max_leaf_nodes=CONFIG['max_leaf_nodes'],min_samples_leaf=CONFIG['min_samples_leaf'],max_features=CONFIG['max_features'],random_state=CONFIG['seed']+f*1000+it);tree.fit(X[rt],target[rt],sample_weight=h[rt]);update=tree.predict(X);old=float(loss.detach());aa=A.detach().numpy();step=CONFIG['learning_rate']
   for back in range(8):
    candidate=torch.tensor(np.clip(aa+step*update,-CONFIG['bound'],CONFIG['bound']));new=float(objective(candidate,indices,N,n,T,V,M,D,train,K,rowtrain,rowbid))
    if new<=old+1e-8:break
    step*=.5
   else:step=0;candidate=A.detach();new=old
   A=candidate;model['steps'].append((tree,step));history.append(new)
  assert gradvalid==0;joblib.dump(model,ROOT/f'action_list_fold{f}.joblib',compress=3);np.testing.assert_array_equal(infer(model,X),A.numpy());z=hand_logits(A,indices,N,n);p=torch.softmax(torch.cat([torch.zeros_like(z[:,:,:1]),z],2),2).numpy()
  for i in np.flatnonzero(fv==f):
   q=groups[i];v=p[i,:len(q),1:];inc=conditioned(v,model['minimum']);ci=inclusion(*v.T);outputs.append(q.select('pair_id','hand_id').with_columns(pl.Series('list_inclusion',inc),pl.Series('raw_inclusion',ci),pl.Series('event_primary',v[:,0]),pl.Series('event_secondary',v[:,1])))
  audit.append({'fold':f,'levels':levels,'initial_loss':initial,'final_loss':history[-1],'training_loss':history,'heldout_gradient_max':gradvalid,'train_pairs':len(train),'training_actions':int(rt.sum()),'model_replay_error':0});print('action-list',f,initial,history[-1],round(time.time()-start,1),flush=True)
 z=pl.concat(outputs);z.write_parquet(ROOT/'isolation_oof.parquet');base=pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id',C('equal').alias('r33'),'pressure59','full');rawbase=pl.concat([pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').filter(C('fold')==f).select('pair_id','hand_id','base') for f in range(4)]);out=base.join(z,on=['pair_id','hand_id'],how='left',validate='1:1').join(rawbase,on=['pair_id','hand_id'],validate='1:1').with_columns(pl.when(C('list_inclusion').is_not_null()).then(.25*C('base')+.25*C('raw_inclusion')+.5*C('list_inclusion')).otherwise(C('pressure59')).alias('action_list'));out=out.with_columns(((C('action_list')+C('full'))*.5).alias('action_list_r33_recipe'));out.write_parquet(ROOT/'oof.parquet');names=['r33','action_list','action_list_r33_recipe'];r,_=compare(ROOT/'oof.parquet',names,'action_list');report={'method':__doc__,'config':CONFIG,'results':{k:{'MAP':r[k].mean(),'folds':r.group_by('fold').agg(C(k).mean()).sort('fold')[k].to_list(),'families':dict(r.group_by('family').agg(C(k).mean()).iter_rows())} for k in names}};(ROOT/'report.json').write_text(json.dumps(report,indent=2));(ROOT/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
