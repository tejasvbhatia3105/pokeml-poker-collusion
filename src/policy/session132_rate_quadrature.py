import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json
import numpy as np,polars as pl,joblib
import session130_hierarchical_list as h

def nodes(n):
    x,w=np.polynomial.hermite.hermgauss(n);h.QX=x*np.sqrt(2);h.QW=w/np.sqrt(np.pi)

def main():
    d=h.data();records=[];parts=[]
    for f in range(4):
        (groups,x,big,P,M,T,V,D,fv,bid,pos),fam,K,tr=h.pack(d,f)
        X=np.nan_to_num(np.column_stack([x,big]),nan=0,posinf=1e6,neginf=-1e6)
        model=joblib.load(h.ROOT/f'model_random_rate_fold{f}.joblib');a=h.infer(model,X)
        for i in np.flatnonzero(fv==f):
            g=groups[i];z=P[i,:len(g)].double().numpy()+a[bid==i];sd=float(model['sigma'][fam[i]]);out={}
            for n in [9,17,33]:
                nodes(n);out[n]=h.inclusion_mixture(z,int(K[i]),sd)
            records.append(dict(pair_id=g['pair_id'][0],fold=f,sigma=sd,max9to33=float(abs(out[9]-out[33]).max()),max17to33=float(abs(out[17]-out[33]).max())))
            parts.append(g.select('pair_id','hand_id').with_columns(*[pl.Series('inclusion_q'+str(n),out[n]) for n in out]))
        print('quadrature fold',f,'sigma',model['sigma'].tolist(),flush=True)
    nodes(9);pl.DataFrame(records).write_parquet(h.ROOT/'quadrature_pairs.parquet');pl.concat(parts).write_parquet(h.ROOT/'quadrature_inclusion.parquet')
    report=dict(heldout_labels_used=False,pairs=len(records),max9to33=max(r['max9to33'] for r in records),max17to33=max(r['max17to33'] for r in records),
                max_sigma=max(r['sigma'] for r in records),criteria='If max9to33 exceeds0.001, numerical convergence remains unresolved; no candidate claim.')
    report['criterion_passed']=report['max9to33']<=.001
    (h.ROOT/'quadrature.json').write_text(json.dumps(report,indent=2));print(report)

if __name__=='__main__':main()
