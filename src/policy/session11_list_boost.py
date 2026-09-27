\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl,torch,joblib
from sklearn.tree import DecisionTreeRegressor
from session8_data import hand_data
from session10_list_learning import features,template,list_nll
from session6_priority import inclusion

ROOT=Path('artifacts/evidence_session11');C=pl.col
CONFIG={'iterations':120,'learning_rate':.15,'max_leaf_nodes':10,
        'min_samples_leaf':80,'max_features':.7,'ridge':.02,'bound':1.5,
        'seed':1111,'input_root':'artifacts/evidence_session10/nested6',
        'objective':'exact observed list NLL / truth count plus mean residual square',
        'selection':'fixed schedule; training-only backtracking; compact/full matched controls'}

def pack(d,f,cols):
    q=d.join(pl.read_parquet(Path(CONFIG['input_root'])/f'nested_outer{f}.parquet').drop('fold','time'),on=['pair_id','hand_id'],validate='1:1')
    groups=[g.sort('time','hand_id') for _,g in q.group_by('pair_id')];groups.sort(key=lambda g:g['pair_id'][0])
    n=max(map(len,groups));N=len(groups);P=np.zeros((N,n,2),np.float32);M=np.zeros((N,n),bool);T=np.zeros((N,6,n),np.int64);V=np.zeros((N,6),bool);D=np.zeros(N,np.float32);fv=np.zeros(N,np.int64);small=[];big=[];bid=[];pos=[]
    for i,g in enumerate(groups):
        x,p=features(g);k=len(g);P[i,:k]=p;M[i,:k]=True
        e=np.flatnonzero(g['evidence_rank'].is_not_null());e=e[np.argsort(g['evidence_rank'].to_numpy()[e])];t=template(k,e);T[i,:len(t),:k]=t;V[i,:len(t)]=True;D[i]=len(e);fv[i]=g['fold'][0]
        small.append(x);big.append(g.select(cols).to_numpy().astype('float32'));bid.extend([i]*k);pos.extend(range(k))
    return groups,np.concatenate(small),np.concatenate(big),torch.tensor(P),torch.tensor(M),torch.tensor(T),torch.tensor(V),torch.tensor(D),fv,np.array(bid),np.array(pos)

def objective(P,A,M,T,V,D,train):
    z=list_nll(P[train]+A[train],T[train],V[train],D[train]).sum()
                                                                            
    reg=(A[train].square().sum(2)*M[train]).sum(1)/M[train].sum(1)
    return z+CONFIG['ridge']*reg.sum()

def infer(model,X):
    a=np.zeros((len(X),2),np.float32)
    for tree,step in model['steps']:a=np.clip(a+step*tree.predict(X),-CONFIG['bound'],CONFIG['bound']).astype('float32')
    return a

def main():
    ROOT.mkdir(exist_ok=True);(ROOT/'list_boost_config.json').write_text(json.dumps(CONFIG,indent=2))
    cols=json.loads(Path('artifacts/evidence_session6/priority_ordered_columns.json').read_text())['event'];d=hand_data();outputs=[];audit=[];start=time.time()
    for f in range(4):
        groups,x,big,P,M,T,V,D,fv,bid,pos=pack(d,f,cols);train=np.flatnonzero((fv!=f)&V.any(1).numpy());valid=np.flatnonzero(fv==f);rowtrain=np.isin(bid,train);rowvalid=np.isin(bid,valid);ti=torch.tensor(train);rowweight=1/D.numpy()[bid];pred={}
        assert not np.any(rowtrain&rowvalid)
        for kind in os.environ.get('BOOST_KINDS','compact,full').split(','):
            X=x if kind=='compact' else np.column_stack([x,big]);X=np.nan_to_num(X,nan=0,posinf=1e6,neginf=-1e6);model={'steps':[],'fold':f,'kind':kind,'columns':cols,'config':CONFIG};A=torch.zeros_like(P);initial=float(objective(P,A,M,T,V,D,ti));history=[]
            for it in range(CONFIG['iterations']):
                A.requires_grad_(True);loss=objective(P,A,M,T,V,D,ti);loss.backward();grad=A.grad.detach().numpy()[bid,pos];a=A.detach().numpy()[bid,pos];pr=torch.softmax(torch.cat([torch.zeros_like(P[:,:,:1]),P+A.detach()],2),2).numpy()[bid,pos,1:];h=np.maximum(pr*(1-pr),.02)*rowweight[:,None]+2*CONFIG['ridge']/M.sum(1).numpy()[bid,None]
                target=np.clip(-grad/h,-5,5);tree=DecisionTreeRegressor(max_leaf_nodes=CONFIG['max_leaf_nodes'],min_samples_leaf=CONFIG['min_samples_leaf'],max_features=CONFIG['max_features'],random_state=CONFIG['seed']+f*1000+it);tree.fit(X[rowtrain],target[rowtrain],sample_weight=h[rowtrain].mean(1));update=tree.predict(X).astype('float32');old=float(loss.detach());A=A.detach();step=CONFIG['learning_rate'];accepted=False
                for _ in range(7):
                    candidate=torch.zeros_like(P);candidate[bid,pos]=torch.tensor(np.clip(a+step*update,-CONFIG['bound'],CONFIG['bound']));new=float(objective(P,candidate,M,T,V,D,ti))
                    if new<=old+1e-6:accepted=True;break
                    step*=.5
                if not accepted:step=0;candidate=A;new=old
                A=candidate;model['steps'].append((tree,step));history.append(new)
            pred[kind]=A.numpy();joblib.dump(model,ROOT/f'list_boost_{kind}_fold{f}.joblib',compress=3);replayed=infer(model,X[rowvalid]);assert np.max(abs(replayed-A.numpy()[bid[rowvalid],pos[rowvalid]]))<1e-6
            audit.append({'fold':f,'kind':kind,'initial_loss':initial,'final_loss':history[-1],'training_loss':history,'train_pairs':len(train),'validation_pairs':len(valid),'features':X.shape[1]});print('list boost',f,kind,'loss',initial,history[-1],'seconds',round(time.time()-start,1),flush=True)
        for i in valid:
            g=groups[i];k=len(g);z=g.select('pair_id','hand_id','fold','evidence','r29')
            for kind,A in pred.items():
                logits=P[i,:k]+torch.tensor(A[i,:k]);p=torch.softmax(torch.cat([torch.zeros_like(logits[:,:1]),logits],1),1).numpy();inc=inclusion(p[:,1],p[:,2]);score=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc;z=z.with_columns(pl.Series(kind,score))
            outputs.append(z)
    pl.concat(outputs).write_parquet(ROOT/'list_boost_oof.parquet');(ROOT/'list_boost_audit.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':main()
