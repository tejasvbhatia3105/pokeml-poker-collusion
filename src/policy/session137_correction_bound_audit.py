\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json
from pathlib import Path
import numpy as np,torch,polars as pl,joblib
import session130_hierarchical_list as h
ROOT=Path('artifacts/evidence_session137_bound_audit')

def main():
    ROOT.mkdir(exist_ok=True);d=h.data();records=[];parts=[]
    for f in range(4):
        (groups,x,big,P,M,T,V,D,fv,bid,pos),fam,K,tr=h.pack(d,f)
        X=np.nan_to_num(np.column_stack([x,big]),nan=0,posinf=1e6,neginf=-1e6)
        model=joblib.load(h.ROOT/f'model_point_fold{f}.joblib');a=h.infer(model,X);sat=abs(a)>=h.CONFIG['bound']-1e-10
        A=torch.zeros_like(P,dtype=torch.float64);A[bid,pos]=torch.tensor(a);A.requires_grad_(True)
        loss=h.objective(P.double(),A,M,T,V,D.double(),K,fam,tr,torch.zeros(3,dtype=torch.float64),'point');grad,=torch.autograd.grad(loss,A);g=grad.numpy()[bid,pos]
        rt=np.isin(bid,tr);rv=fv[bid]==f;outward=sat&(g*a<0)
        records.append(dict(fold=f,training_hands=int(rt.sum()),training_saturated_hands=int(sat[rt].any(1).sum()),training_outward_gradient_hands=int(outward[rt].any(1).sum()),
                            training_total_abs_gradient=float(abs(g[rt]).sum()),training_outward_bound_abs_gradient=float(abs(g[rt])[outward[rt]].sum()),
                            heldout_hands=int(rv.sum()),heldout_saturated_hands=int(sat[rv].any(1).sum())))
        for i in np.flatnonzero(fv==f):
            gg=groups[i];ii=np.flatnonzero(bid==i)
            parts.append(gg.select('pair_id','hand_id','fold','evidence').with_columns(pl.Series('delta_primary',a[ii,0]),pl.Series('delta_secondary',a[ii,1]),pl.Series('saturated',sat[ii].any(1))))
        print(records[-1],flush=True)
    pl.concat(parts).write_parquet(ROOT/'heldout_corrections.parquet');(ROOT/'report.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()
