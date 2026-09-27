import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json
import numpy as np,polars as pl,torch,joblib
import session130_hierarchical_list as h

def main():
    d=h.data();records=[]
    for f in range(4):
        (groups,x,big,P,M,T,V,D,fv,bid,pos),fam,K,tr=h.pack(d,f)
        altered=d.with_columns(pl.when(pl.col('fold')==f).then(None).otherwise(pl.col('evidence_rank')).alias('evidence_rank'),
                               pl.when(pl.col('fold')==f).then(0).otherwise(pl.col('evidence')).alias('evidence'))
        alt,af,ak,at=h.pack(altered,f)
        assert np.array_equal(x,alt[1]) and np.array_equal(big,alt[2]) and torch.equal(P,alt[3])
        assert np.array_equal(tr,at) and torch.equal(K,ak)
        assert torch.equal(T[tr],alt[5][at]) and torch.equal(V[tr],alt[6][at]) and torch.equal(D[tr],alt[7][at])
        model=joblib.load(h.ROOT/f'model_random_rate_fold{f}.joblib');X=np.nan_to_num(np.column_stack([x,big]),nan=0,posinf=1e6,neginf=-1e6)
        a=h.infer(model,X);A=torch.zeros_like(P,dtype=torch.float64);A[bid,pos]=torch.tensor(a);P=P.double();D=D.double()
        slopes={}
        for j,name in enumerate(h.FAMILIES):
            ii=tr[fam[tr]==j];z=P[ii]+A[ii]
                                                                             
                                                                               
            losses=[]
            for variance in [0.,1e-4]:
                with torch.no_grad():
                    loss=h.mixture_nll(z,M[ii],T[ii],V[ii],D[ii],K[ii],torch.full((len(ii),),np.sqrt(variance),dtype=P.dtype)).mean()
                losses.append(float(loss))
            slopes[name]=(losses[1]-losses[0])/1e-4
        records.append(dict(fold=f,heldout_evidence_mutation=True,features_unchanged=True,training_targets_and_count_minima_unchanged=True,
                            data_only_NLL_slope_at_zero_variance=slopes))
        print(records[-1],flush=True)
    (h.ROOT/'training_isolation_and_boundary.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
