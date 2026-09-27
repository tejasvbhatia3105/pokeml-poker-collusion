\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
os.environ.setdefault('BOOST_KINDS','full')
from pathlib import Path
import numpy as np,torch
import session11_list_boost as b
from session11_conditional_family import log_count_at_least
from session8_count_conditioning import conditioned

ROOT=Path('artifacts/evidence_session12/conditional_boost')
MINIMUM=None
original_pack=b.pack
def pack(d,f,cols):
    global MINIMUM
    values=original_pack(d,f,cols);groups,x,big,P,M,T,V,D,fv,bid,pos=values
    fam=np.array([g['behavior_family'][0] for g in groups]);tr=(fv!=f)&V.any(1).numpy();mins={k:int(D.numpy()[tr&(fam==k)].min()) for k in set(fam)}
    MINIMUM=torch.tensor([mins[k] for k in fam]);(ROOT/f'minimums_fold{f}.json').write_text(b.json.dumps(mins,indent=2))
    return values
def objective(P,A,M,T,V,D,train):
    z=b.list_nll(P[train]+A[train],T[train],V[train],D[train])
    z=z+log_count_at_least(P[train]+A[train],M[train],MINIMUM[train])/D[train]
    reg=(A[train].square().sum(2)*M[train]).sum(1)/M[train].sum(1)
    return z.sum()+b.CONFIG['ridge']*reg.sum()
def main():
                                                                              
                                                                  
    ROOT.mkdir(parents=True,exist_ok=True);b.ROOT=ROOT;b.CONFIG=dict(b.CONFIG)
    b.CONFIG['objective']='family-conditional exact observed list NLL / truth count plus mean residual square'
    b.pack=pack;b.objective=objective;b.main()
    d=b.hand_data();cols=b.json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];parts=[]
    for f in range(4):
        groups,x,big,P,M,T,V,D,fv,bid,pos=pack(d,f,cols);minimums=b.json.load(open(ROOT/f'minimums_fold{f}.json'));X=np.nan_to_num(np.column_stack([x,big]),nan=0,posinf=1e6,neginf=-1e6);model=b.joblib.load(ROOT/f'list_boost_full_fold{f}.joblib');model['minimums']=minimums;b.joblib.dump(model,ROOT/f'list_boost_full_fold{f}.joblib',compress=3);a=b.infer(model,X);z=P.numpy().copy();z[bid,pos]+=a
        for i in np.flatnonzero(fv==f):
            g=groups[i];p=torch.softmax(torch.tensor(np.column_stack([np.zeros(len(g),np.float32),z[i,:len(g)]])),1).numpy();inc=conditioned(p[:,1:],minimums[g['behavior_family'][0]]);score=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc;parts.append(g.select('pair_id','hand_id','fold','evidence','r29').with_columns(b.pl.Series('conditional_boost',score)))
    b.pl.concat(parts).write_parquet(ROOT/'conditional_boost_oof.parquet')
if __name__=='__main__':main()
