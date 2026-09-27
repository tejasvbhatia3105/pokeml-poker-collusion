import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data,targets,ROOT,C
from session6_priority import inclusion
def main():
    path=ROOT/'ledger_hand_features.parquet'
    d=hand_data().join(pl.read_parquet(path),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');cols=json.loads(Path('artifacts/evidence_session6/priority_ordered_columns.json').read_text())['event']+[c for c in d.columns if c.startswith('ledger_')];X=d.select(cols).to_numpy();assert np.isfinite(X).all();pred=np.zeros((len(d),2));start=time.time()
    for f in range(4):
        for b in ['directed_transfer','soft_play','coordinated_isolation']:
            t1,t2,e1,e2,va=targets(d,f,b)
            for k,y,tr in [(1,t1,e1),(2,t2,e2)]:
                m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=4,random_seed=6310+11*f+k,verbose=False,allow_writing_files=False);m.fit(X[tr],y[tr]);pred[va,k-1]=m.predict_proba(X[va],thread_count=4)[:,1];m.save_model(str(ROOT/f'ledger_event{k}_{b}_fold{f}.cbm'))
        print('ledger heads fold',f,'seconds',round(time.time()-start,1),flush=True)
    q=d.select('pair_id','hand_id','time').with_columns(pl.Series('primary',pred[:,0]),pl.Series('secondary',pred[:,1]));parts=[]
    for _,g in q.group_by('pair_id',maintain_order=True):parts.append(g.with_columns(pl.Series('score',inclusion(g['primary'].to_numpy(),g['secondary'].to_numpy()))))
    pl.concat(parts).write_parquet(ROOT/'ledger_oof.parquet');(ROOT/'ledger_columns.json').write_text(json.dumps(cols,indent=2))
if __name__=='__main__':main()
