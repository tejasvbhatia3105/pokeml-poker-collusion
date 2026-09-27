\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data,targets,ROOT,C
from session6_priority import inclusion
FAMILIES=['directed_transfer','soft_play','coordinated_isolation']
def main():
    mode=os.environ.get('EVENT_MODE','shared_events');d=hand_data();cols=json.loads(Path('artifacts/evidence_session6/priority_ordered_columns.json').read_text())['event']
    assert mode in ['shared_events','raw_events','no_clock']
    if mode=='raw_events':
        z=pl.read_parquet('artifacts/evidence_session4/raw_action_features.parquet');d=d.join(z,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');cols += [c for c in z.columns if c.startswith('raw_action_')]
    elif mode=='no_clock':
        cols=[c for c in cols if c!='relative_time' and not c.startswith('preceding_')]
    else:
        d=d.with_columns(*[(C('behavior_family')==b).cast(pl.Float32).alias('family_'+b) for b in FAMILIES]);cols += ['family_'+b for b in FAMILIES]
    X=d.select(cols).to_numpy();pred=np.zeros((len(d),2));start=time.time()
    for f in range(4):
        ts={b:targets(d,f,b) for b in FAMILIES}
        for b in FAMILIES if mode!='shared_events' else ['shared']:
            if b=='shared':
                t1=np.sum([ts[c][0] for c in FAMILIES],axis=0);t2=np.sum([ts[c][1] for c in FAMILIES],axis=0)
                e1=np.any([ts[c][2] for c in FAMILIES],axis=0);e2=np.any([ts[c][3] for c in FAMILIES],axis=0);va=d['fold'].to_numpy()==f
                                                                                
                                                                              
                t1[:]=0;t2[:]=0
                for c in FAMILIES:
                    mask=d['behavior_family'].to_numpy()==c;t1[mask]=ts[c][0][mask];t2[mask]=ts[c][1][mask]
            else:t1,t2,e1,e2,va=ts[b]
            for k,y,tr in [(1,t1,e1),(2,t2,e2)]:
                assert not np.any(tr & va)
                m=CatBoostClassifier(iterations=500 if b=='shared' else 400,depth=6 if b=='shared' else 5,learning_rate=.035,l2_leaf_reg=8,thread_count=4,random_seed=6310+11*f+k,verbose=False,allow_writing_files=False)
                m.fit(X[tr],y[tr]);pred[va,k-1]=m.predict_proba(X[va],thread_count=4)[:,1];m.save_model(str(ROOT/f'{mode}_event{k}_{b}_fold{f}.cbm'))
        print(mode,'fold',f,'seconds',round(time.time()-start,1),flush=True)
    q=d.select('pair_id','hand_id','time').with_columns(pl.Series('primary',pred[:,0]),pl.Series('secondary',pred[:,1]));parts=[]
    for _,g in q.group_by('pair_id',maintain_order=True):parts.append(g.with_columns(pl.Series('score',inclusion(g['primary'].to_numpy(),g['secondary'].to_numpy()))))
    pl.concat(parts).write_parquet(ROOT/f'{mode}_oof.parquet');(ROOT/f'{mode}_columns.json').write_text(json.dumps(cols,indent=2))
if __name__=='__main__':main()
