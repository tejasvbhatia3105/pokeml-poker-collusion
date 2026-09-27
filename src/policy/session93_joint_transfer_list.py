\
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
from session55_current_targets import state
from session39_bet_call import data as calls
from session51_matchup_inference import current_from_design
from session11_conditional_family import template,list_nll,log_count_at_least
from session8_count_conditioning import conditioned
from session6_priority import inclusion
ROOT=Path('artifacts/evidence_session93_joint_transfer_list');C=pl.col
CONFIG={'iterations':200,'learning_rate':.15,'max_leaf_nodes':10,'min_samples_leaf':20,'max_features':.7,'bound':6.,'ridge':.002,'initial_action_probability':.15,'seed':9300,'heads':'separate trees for fold and call, joint list objective','training_donor_prior':[.5,.5],'inference_donor':'fixed held-out session25 posterior; separate per-actor conditioning','features':'original matched action+hand+paired decisions+current2; no92 chip fields','targets':'only published hand list and order; no fitted teacher or pseudo action/type/donor labels'}
BIAS=float(np.log(.15/.85));torch.set_num_threads(2)
def pack():
 v=state()['directed_transfer'];d=v['d'];a=v['a'];dc,ca,cfg=calls();np.testing.assert_array_equal(d['hand_id'].to_numpy(),dc['hand_id'].to_numpy());cc=json.load(open('artifacts/evidence_session37_bet_fold/config.json'));ec=pl.read_parquet('artifacts/evidence_session39_bet_call/action_features.parquet').sort('action_row');ec=ec.rename({c:c.replace('call_minus_bet_','fold_minus_bet_') for c in ec.columns});cx=np.column_stack([ca.select(cc['fold_columns']).to_numpy(),d.select(cc['hand_columns']).to_numpy()[ca['row'].to_numpy()],ec.select(cc['paired_columns']).to_numpy()]);cx=np.column_stack([cx,current_from_design(ca,cx,cc)]);assert cx.shape[1]==v['x'].shape[1]
 x=np.nan_to_num(np.vstack([v['x'],cx]),nan=0,posinf=1e6,neginf=-1e6);act=pl.concat([a.select('pair_id','hand_id','row','actor','action_no').with_columns(pl.lit(0).alias('head')),ca.select('pair_id','hand_id','row','actor','action_no').with_columns(pl.lit(1).alias('head'))],how='vertical_relaxed');groups=[g.sort('time','hand_id') for _,g in d.group_by('pair_id')];groups.sort(key=lambda g:g['pair_id'][0]);N=len(groups);n=max(map(len,groups));T=np.zeros((N,6,n),np.int64);V=np.zeros((N,6),bool);M=np.zeros((N,n),bool);D=np.zeros(N);fv=np.zeros(N,int);pos=np.zeros(len(d),int);bid=np.zeros(len(d),int)
 for i,q in enumerate(groups):
  rows=q['row'].to_numpy();pos[rows]=np.arange(len(q));bid[rows]=i;e=np.flatnonzero(q['evidence_rank'].is_not_null());e=e[np.argsort(q['evidence_rank'].to_numpy()[e])];t=template(len(q),e);T[i,:len(t),:len(q)]=t;V[i,:len(t)]=True;M[i,:len(q)]=True;D[i]=len(e);fv[i]=q['fold'][0]
 rows=act['row'].to_numpy();rb=bid[rows];head=act['head'].to_numpy();indices=((rb*2+act['actor'].to_numpy())*n+pos[rows])*2+head
 return d,act,x,groups,torch.tensor(T),torch.tensor(V),torch.tensor(M),torch.tensor(D),fv,rb,head,torch.tensor(indices,dtype=torch.int64)
def hand_logits(A,indices,N,n):
 logs=torch.zeros(N*2*n*2,dtype=A.dtype).index_add(0,indices,torch.nn.functional.logsigmoid(-A));p=-torch.expm1(logs.reshape(N,2,n,2));p1=p[:,:,:,0];p2=(1-p1)*p[:,:,:,1];p0=((1-p1)*(1-p[:,:,:,1])).clamp_min(1e-14);return torch.stack([torch.log(p1.clamp_min(1e-14)/p0),torch.log(p2.clamp_min(1e-14)/p0)],3)
def list_loss(z,T,V,M,D,K):
 loglist=torch.stack([-list_nll(z[:,a],T,V,D)*D for a in [0,1]],1);logmass=torch.stack([log_count_at_least(z[:,a],M,K) for a in [0,1]],1)
 return (-torch.logsumexp(loglist,1)+torch.logsumexp(logmass,1))/D
def objective(A,indices,T,V,M,D,train,K,rb,rt):
 z=hand_logits(A,indices,*M.shape);loss=list_loss(z[train],T[train],V[train],M[train],D[train],K[train]).sum();count=torch.bincount(rb,minlength=len(M)).clamp_min(1);reg=((A[rt]-BIAS).square()/count[rb[rt]]).sum();return loss+CONFIG['ridge']*reg
def infer(model,x,head):
 a=np.full(len(x),model['bias'],float)
 for trees,step in model['steps']:
  u=np.zeros(len(x))
  for k,t in enumerate(trees):u[head==k]=t.predict(x[head==k])
  a=np.clip(a+step*u,-model['config']['bound'],model['config']['bound'])
 return a
def predict(model,x,head,indices,groups,M,fv):
 A=torch.tensor(infer(model,x,head));z=hand_logits(A,indices,*M.shape);p=torch.softmax(torch.cat([torch.zeros_like(z[:,:,:,:1]),z],3),3).numpy();donors=pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet');parts=[]
 for i in np.flatnonzero(fv==model['fold']):
  q=groups[i];w=donors.filter(C('pair_id')==q['pair_id'][0]).select('actor0','actor1').to_numpy()[0];pp=p[i,:,:len(q),1:];ci=sum(w[k]*conditioned(pp[k],model['minimum']) for k in [0,1]);raw=sum(w[k]*inclusion(*pp[k].T) for k in [0,1]);parts.append(q.select('pair_id','hand_id').with_columns(pl.Series('list_inclusion',ci),pl.Series('raw_inclusion',raw)))
 return pl.concat(parts)
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));d,a,x,groups,T,V,M,D,fv,rb,head,indices=pack();audit=[];parts=[];start=time.time();rr=torch.tensor(rb)
 for f in range(4):
  train=torch.tensor(np.flatnonzero(fv!=f));rt=fv[rb]!=f;rtensor=torch.tensor(rt);K=torch.full((len(M),),int(D[train].min()));A=torch.full((len(x),),BIAS,dtype=torch.float64);model={'steps':[],'config':CONFIG,'bias':BIAS,'fold':f,'minimum':int(K[0]),'feature_count':x.shape[1]};trace=[];initial=float(objective(A,indices,T,V,M,D,train,K,rr,rtensor));weight=1/np.maximum(np.bincount(rb,minlength=len(M))[rb],1);valid_grad=0
  for it in range(CONFIG['iterations']):
   A.requires_grad_(True);loss=objective(A,indices,T,V,M,D,train,K,rr,rtensor);loss.backward();grad=A.grad.detach().numpy();valid_grad=max(valid_grad,float(abs(grad[~rt]).max(initial=0)));p=torch.sigmoid(A.detach()).numpy();h=np.maximum(p*(1-p),.02)/D.numpy()[rb]+2*CONFIG['ridge']*weight;target=np.clip(-grad/h,-5,5);u=np.zeros(len(x));trees=[]
   for k in [0,1]:
    mask=rt&(head==k);tree=DecisionTreeRegressor(max_leaf_nodes=CONFIG['max_leaf_nodes'],min_samples_leaf=CONFIG['min_samples_leaf'],max_features=CONFIG['max_features'],random_state=CONFIG['seed']+1000*f+2*it+k);tree.fit(x[mask],target[mask],sample_weight=h[mask]);u[head==k]=tree.predict(x[head==k]);trees.append(tree)
   old=float(loss.detach());aa=A.detach().numpy();step=CONFIG['learning_rate']
   for back in range(8):
    candidate=torch.tensor(np.clip(aa+step*u,-CONFIG['bound'],CONFIG['bound']));new=float(objective(candidate,indices,T,V,M,D,train,K,rr,rtensor))
    if new<=old+1e-8:break
    step*=.5
   else:step=0;candidate=A.detach();new=old
   A=candidate;model['steps'].append((trees,step));trace.append(new)
  assert valid_grad==0;joblib.dump(model,ROOT/f'joint_fold{f}.joblib',compress=3);np.testing.assert_array_equal(infer(model,x,head),A.numpy());parts.append(predict(model,x,head,indices,groups,M,fv));audit.append({'fold':f,'training_pairs':len(train),'training_actions':int(rt.sum()),'initial_loss':initial,'final_loss':trace[-1],'training_loss':trace,'heldout_gradient':valid_grad});print('joint transfer list',f,initial,trace[-1],round(time.time()-start,1),flush=True)
 pl.concat(parts).write_parquet(ROOT/'direct_oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
