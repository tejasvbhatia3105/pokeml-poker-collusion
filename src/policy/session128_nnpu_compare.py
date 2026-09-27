import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
import json
import numpy as np
import polars as pl
from session127_nnpu import ROOT, WINDOWS, metrics

def main():
    meta=pl.read_parquet(ROOT/'metadata.parquet');result={}
    for w in WINDOWS:
        m=meta.filter(pl.col('window')==w).sort('row');parts={}
        for v in ['r4s','r4k']:
            z=pl.read_parquet(f'artifacts/pair_session74_complete_rescore/{v}_{w}.parquet').select('pair_id',pl.col('risk_orig').alias(v))
            m=m.join(z,on='pair_id',how='left',validate='1:1',maintain_order='left')
            assert m[v].null_count()==0
        parts['raw_gbdt']=np.sqrt(m['r4s'].to_numpy().clip(1e-6,1)*m['r4k'].to_numpy().clip(1e-6,1))
        for mult in [2,4]:
            for arm in ['naive','nnpu']:
                k=f'{arm}_prior{mult}'
                p=pl.concat([pl.read_parquet(ROOT/f'{k}_fold{f}.parquet') for f in range(4)]).filter(pl.col('window')==w).sort('row')
                assert np.array_equal(p['row'].to_numpy(),m['row'].to_numpy())
                parts[k]=p['logit'].to_numpy()
        for k,s in parts.items():
            result[f'{k}/{w}']=dict(all=metrics(m,s),folds=[metrics(m.filter(pl.col('fold')==f),s[m['fold'].to_numpy()==f]) for f in range(4)])
    (ROOT/'raw_comparison.json').write_text(json.dumps(result,indent=2))
    for k,v in result.items():
        s=v['all'];print(k,{n:round(s[n],5) for n in ['trusted_AP','recall_at_100_other','weak_recall_at_100_other','recall_at_300_other','weak_recall_at_300_other']})

if __name__=='__main__':main()
