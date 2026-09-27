import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data,targets,ROOT,C
from session6_priority import inclusion
def aggregate(a,neighbors):
    keys=['pair_id','hand_id','action_no'];d=a.select(*keys,'action_class','facing_partner','mw_alive','players_active').join(neighbors,on=keys,validate='1:1',maintain_order='left');alive=C('mw_alive');facing=C('facing_partner')&alive
    gates={'fold_partner':facing&(C('action_class')==0),'call_partner':facing&(C('action_class')==2),'raise_partner':facing&(C('action_class')==3),'check_hu':alive&(C('players_active')==2)&(C('action_class')==1),'raise_outside':alive&~facing&(C('action_class')==3),'fold_outside':alive&~facing&(C('action_class')==0)}
    cols=[c for c in neighbors.columns if c.startswith('neighbor_') and 'table_' not in c];expr=[]
    for name,gate in gates.items():
        expr.append(gate.sum().cast(pl.Float32).alias('context_'+name+'_count'))
        for c in cols:expr.append(C(c).filter(gate).mean().fill_null(-2).cast(pl.Float32).alias('context_'+name+'_'+c))
    return d.group_by('pair_id','hand_id').agg(expr)
def main():
    path=ROOT/'context_hand_features.parquet'
    if not path.exists():
        a=pl.read_parquet('artifacts/evidence_session5/mil_actions.parquet');neighbors=pl.read_parquet(ROOT/'neighbor_actions.parquet');z=aggregate(a,neighbors);z.write_parquet(path)
        shuffled=aggregate(a.reverse(),neighbors.reverse()).sort('pair_id','hand_id');original=z.sort('pair_id','hand_id');assert np.allclose(original.select(pl.selectors.numeric()).to_numpy(),shuffled.select(pl.selectors.numeric()).to_numpy(),atol=1e-5)
    d=hand_data().join(pl.read_parquet(path),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');cols=json.loads(Path('artifacts/evidence_session6/priority_ordered_columns.json').read_text())['event']+[c for c in d.columns if c.startswith('context_')];X=d.select(cols).to_numpy();assert np.isfinite(X).all();pred=np.zeros((len(d),2));start=time.time()
    for f in range(4):
        for b in ['directed_transfer','soft_play','coordinated_isolation']:
            t1,t2,e1,e2,va=targets(d,f,b)
            for k,y,tr in [(1,t1,e1),(2,t2,e2)]:
                m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=4,random_seed=6310+11*f+k,verbose=False,allow_writing_files=False);m.fit(X[tr],y[tr]);pred[va,k-1]=m.predict_proba(X[va],thread_count=4)[:,1];m.save_model(str(ROOT/f'context_event{k}_{b}_fold{f}.cbm'))
        print('context heads fold',f,'seconds',round(time.time()-start,1),flush=True)
    q=d.select('pair_id','hand_id','time').with_columns(pl.Series('primary',pred[:,0]),pl.Series('secondary',pred[:,1]));parts=[]
    for _,g in q.group_by('pair_id',maintain_order=True):parts.append(g.with_columns(pl.Series('score',inclusion(g['primary'].to_numpy(),g['secondary'].to_numpy()))))
    pl.concat(parts).write_parquet(ROOT/'context_oof.parquet');(ROOT/'context_columns.json').write_text(json.dumps(cols,indent=2))
if __name__=='__main__':main()
