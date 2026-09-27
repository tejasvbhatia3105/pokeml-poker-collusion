\
\
\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json,hashlib,time
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session123_family_holdout as s
ROOT=Path('artifacts/evidence_session174_direct_list_membership');C=pl.col
CONFIG=dict(method=__doc__,iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,seed=12301,
    weighting='Equal aggregate weight per training relationship, no class reweighting',selection='All16 fixed fits; no validation stopping or selection')

def targets(d,groups,tr):
    rows=np.flatnonzero(tr);y=d['evidence'].to_numpy()[rows];weight=np.zeros(len(d))
    for ix in groups:
        if tr[ix[0]]:weight[ix]=1/len(ix)
    w=weight[rows];w/=w.mean();assert np.all(w>0)
    return rows,y,w

def main(verify=False):
    ROOT.mkdir(exist_ok=True);d,x,groups=s.load();cfg=dict(CONFIG,data_config=json.load(open(s.data.ROOT/'config.json')),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    assert cfg['data_config']['x_sha256']==hashlib.sha256((s.data.ROOT/'x.npy').read_bytes()).hexdigest()
    cp=ROOT/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==cfg
    else:assert not verify;cp.write_text(json.dumps(cfg,indent=2))
    start=time.time();records=[]
    for held in ['all']+s.FAMILIES:
        folder=ROOT/held;folder.mkdir(exist_ok=True);parts=[]
        for fold in range(4):
            tr,va=s.masks(d,fold,held);rows,y,w=targets(d,groups,tr)
            mutation=d.with_columns(pl.when(pl.Series(~tr)).then(1-C('evidence')).otherwise(C('evidence')).alias('evidence'))
            for a,b in zip((rows,y,w),targets(mutation,groups,tr)):np.testing.assert_array_equal(a,b)
            tables=sorted(d[rows]['table_id'].unique());families=sorted(d[rows]['behavior_family'].unique())
            assert not set(tables)&set(d.filter(C('fold')==fold)['table_id']);assert held=='all' or held not in families
            rec=dict(fold=fold,held=held,training_tables=tables,training_families=families,training_pairs=d[rows]['pair_id'].n_unique(),training_hands=len(rows),target_sha256=s.checksum(rows,y,w),protected_membership_mutation_exact=True)
            path=folder/f'fold{fold}.cbm'
            if path.exists():
                old=json.load(open(path.with_suffix('.json')));assert all(old[k]==v for k,v in rec.items());assert old['model_sha256']==hashlib.sha256(path.read_bytes()).hexdigest();m=CatBoostClassifier();m.load_model(str(path))
            else:
                assert not verify;m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,loss_function='Logloss',random_seed=12301+fold,thread_count=4,verbose=False,allow_writing_files=False)
                m.fit(x[rows],y,sample_weight=w);m.save_model(str(path));rec['model_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();path.with_suffix('.json').write_text(json.dumps(rec,indent=2))
            assert m.tree_count_==400 and np.array_equal(m.classes_,[0,1]);ix=np.flatnonzero(va);p=m.predict_proba(x[ix],thread_count=2)[:,1]
            np.testing.assert_array_equal(p[::97],m.predict_proba(x[ix[::97][::-1]],thread_count=2)[:,1][::-1])
            out=d[ix].select('hand_index','pair_id','hand_id','table_id','fold','behavior_family','evidence').with_columns(pl.Series('score',p))
            if verify:assert out.equals(pl.read_parquet(folder/f'fold{fold}.parquet'))
            else:out.write_parquet(folder/f'fold{fold}.parquet')
            parts.append(out);records.append(rec);print('DIRECT_MEMBERSHIP',held,fold,len(rows),round(time.time()-start,1),flush=True)
        if verify:assert pl.concat(parts).equals(pl.read_parquet(folder/'oof.parquet'))
        else:pl.concat(parts).write_parquet(folder/'oof.parquet')
    if verify:(ROOT/'verification.json').write_text(json.dumps(dict(models=16,final_prediction_error=0,protected_membership_mutations_exact=True,pool_exclusions=True,family_exclusions=True,records=records),indent=2))
    else:(ROOT/'audit.json').write_text(json.dumps(records,indent=2))
    print('COMPLETE',verify,len(records),round(time.time()-start,1),flush=True)

if __name__=='__main__':
    import sys
    assert sys.argv[1] in ['train','verify'];main(sys.argv[1]=='verify')
