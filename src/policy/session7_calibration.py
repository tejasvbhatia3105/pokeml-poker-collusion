\
\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl,torch
from catboost import CatBoostClassifier
from scipy.special import softmax
from session7_em import data
from session6_priority import training_targets,inclusion
from session6_priority_model import load_models
from session7_likelihood import masks
ROOT=Path('artifacts/evidence_session7');C=pl.col;torch.set_num_threads(2)
def odds(a,b):
    scale=np.maximum(1.000001,a+b);a=a/scale;b=b/scale;none=np.maximum(1e-8,1-a-b)
    return np.log(np.maximum(1e-8,np.stack([a,b],1)))-np.log(none)[:,None]
def fit_calibration(q):
    bags=[]
    for _,g in q.sort('pair_id','time','hand_id').group_by('pair_id',maintain_order=True):
        ranks=g['evidence_rank'].fill_null(0).to_numpy();pos=np.flatnonzero(ranks>0);e=pos[np.argsort(ranks[pos])];h=masks(len(g),e)
        if h:bags.append((odds(g['primary'].to_numpy(),g['secondary'].to_numpy()),[z for _,z in h]))
    n=max(len(x) for x,_ in bags);h=max(len(z) for _,z in bags);x=np.zeros((len(bags),n,2));allowed=np.ones((len(bags),h,n,3),bool);valid=np.zeros((len(bags),h),bool)
    for i,(xx,zz) in enumerate(bags):x[i,:len(xx)]=xx;allowed[i,:len(zz),:len(xx)]=zz;valid[i,:len(zz)]=True
    x=torch.tensor(x,dtype=torch.float64);allowed=torch.tensor(allowed);valid=torch.tensor(valid);initial=torch.tensor([1.,1.,0.,0.],dtype=torch.float64);w=torch.nn.Parameter(initial.clone());opt=torch.optim.LBFGS([w],lr=.5,max_iter=80,line_search_fn='strong_wolfe')
    def loss():
        z=x*w[:2]+w[2:];z=torch.cat([torch.zeros((*z.shape[:-1],1),dtype=z.dtype),z],-1);logp=torch.log_softmax(z,-1);ll=torch.logsumexp(logp[:,None,:,:].masked_fill(~allowed,-1e30),-1).sum(-1).masked_fill(~valid,-1e30)
        return -torch.logsumexp(ll,-1).mean()+.1*((w-initial)**2).sum()
    before=float(loss().detach())
    def closure():opt.zero_grad();v=loss();v.backward();return v
    opt.step(closure);return w.detach().numpy(),{'loss_before':before,'loss_after':float(loss().detach()),'bags':len(bags)}
def main():
    d=data().join(pl.read_parquet('artifacts/evidence_session6/order_subtype_labels.parquet'),on=['pair_id','hand_id'],how='left',maintain_order='left',validate='1:1').with_columns(C('subtype').fill_null(0));models,cols=load_models('priority_ordered');tc=[c for c in cols if not any(k in c for k in ['relationship_','preceding_','near5','time'])];X=d.select(cols).to_numpy();XT=d.select(tc).to_numpy();fv=d['fold'].to_numpy();fam=d['behavior_family'].to_numpy();sub=d['subtype'].to_numpy();y=d['evidence'].to_numpy();rank=d['evidence_rank'].fill_null(0).to_numpy();t=d['time'].to_numpy();groups=[g['row'].to_numpy() for _,g in d.group_by('pair_id',maintain_order=True)];start=time.time();parts=[];audit=[]
    for f in range(4):
        path=ROOT/f'calibration_nested_outer{f}.parquet'
        if not path.exists():
            pred=np.full((len(d),2),np.nan)
            for j in range(4):
                if j==f:continue
                for b in models:
                    tr=(fv!=f)&(fv!=j)&(fam==b);va=(fv==j)&(fam==b);known=tr&(sub>0)
                    typ=CatBoostClassifier(iterations=200,depth=3,learning_rate=.04,l2_leaf_reg=10,thread_count=4,random_seed=6210+j,verbose=False,allow_writing_files=False);typ.fit(XT[known],(sub[known]==1).astype(int));tp=typ.predict_proba(XT,thread_count=4)[:,1];a,btarget,ea,eb=training_targets(y,sub,tp,tr,t,groups,rank)
                    for k,target,eligible in [(1,a,ea),(2,btarget,eb)]:
                        assert not np.any(eligible&((fv==f)|(fv==j)))
                        m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=4,random_seed=6310+11*j+k,verbose=False,allow_writing_files=False);m.fit(X[eligible],target[eligible]);pred[va,k-1]=m.predict_proba(X[va],thread_count=4)[:,1]
                print('nested outer',f,'inner',j,'seconds',round(time.time()-start,1),flush=True)
            d.select('pair_id','hand_id','fold').with_columns(pl.Series('primary',pred[:,0]),pl.Series('secondary',pred[:,1])).filter(C('fold')!=f).write_parquet(path)
        nested=pl.read_parquet(path);assert (nested['fold']!=f).all()
        q=d.join(nested.drop('fold'),on=['pair_id','hand_id'],validate='1:1')
        for b in models:
            w,stats=fit_calibration(q.filter(C('behavior_family')==b));np.save(ROOT/f'calibration_{b}_fold{f}.npy',w);stats.update({'fold':f,'family':b,'weights':w.tolist()});audit.append(stats)
            z=d.filter((C('fold')==f)&(C('behavior_family')==b));x=z.select(cols).to_numpy();a=models[b][f][0].predict_proba(x,thread_count=4)[:,1];bb=models[b][f][1].predict_proba(x,thread_count=4)[:,1];logits=odds(a,bb)*w[:2]+w[2:];p=softmax(np.column_stack([np.zeros(len(z)),logits]),axis=1);z=z.select('pair_id','hand_id','time').with_columns(pl.Series('primary',p[:,1]),pl.Series('secondary',p[:,2]))
            for _,g in z.group_by('pair_id',maintain_order=True):g=g.sort('time','hand_id');parts.append(g.with_columns(pl.Series('score',inclusion(g['primary'].to_numpy(),g['secondary'].to_numpy()))))
    pl.concat(parts).write_parquet(ROOT/'calibration_oof.parquet');(ROOT/'calibration_audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
