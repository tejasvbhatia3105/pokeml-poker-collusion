\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json,hashlib
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session170_negative_action_data as data
ROOT=Path('artifacts/evidence_session176_negative_action_diagnostics');C=pl.col
SOURCES={'positive_only':'artifacts/evidence_session167_action_em_extended','negative_control':'artifacts/evidence_session172_negative_action_em'}

def main(arm):
    root=Path(SOURCES[arm]);v=json.load(open(root/'verification.json'));assert v['models']==24 and v['final_prediction_error']==0
    meta=pl.read_parquet(data.ROOT/'actions.parquet');x=np.load(data.ROOT/'x.npy',mmap_mode='r');parts=[];audit=[]
    for fold in range(4):
        path=root/f'fold{fold}_em6.cbm';r=json.load(open(path.with_suffix('.json')));assert r['model_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        ix=np.flatnonzero(meta['fold'].to_numpy()==fold);assert not set(meta[ix]['table_id'])&set(r['training_tables'])
        m=CatBoostClassifier();m.load_model(str(path));p=m.predict_proba(x[ix],thread_count=2);assert np.array_equal(m.classes_,[0,1,2])
        np.testing.assert_array_equal(p[::103],m.predict_proba(x[ix[::103][::-1]],thread_count=2)[::-1])
        parts.append(meta[ix].select('pair_id','table_id','fold').with_columns(pl.Series('primary',p[:,1]),pl.Series('secondary',p[:,2]),pl.Series('none_nll',-np.log(p[:,0].clip(1e-12)))))
        audit.append(dict(fold=fold,actions=len(ix),model_sha256=r['model_sha256'],pool_exclusions=True))
    q=pl.concat(parts).group_by('pair_id').agg(C('table_id').first(),C('fold').first(),C('primary','secondary','none_nll').mean(),pl.len().alias('actions')).sort('pair_id');assert len(q)==1488
    ROOT.mkdir(exist_ok=True);q.write_parquet(ROOT/(arm+'.parquet'));report=dict(arm=arm,negative_pairs=len(q),actions=int(q['actions'].sum()),equal_pair_means=q.select(C('primary','secondary','none_nll').mean()).to_dicts()[0],folds=q.group_by('fold').agg(C('primary','secondary','none_nll').mean()).sort('fold').to_dicts(),audit=audit,limitations=__doc__)
    (ROOT/(arm+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    import sys
    main(sys.argv[1])
