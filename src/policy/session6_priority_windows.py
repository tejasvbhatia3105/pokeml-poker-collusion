\
\
\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from threadpoolctl import threadpool_limits
from sequence_features import augment
from session4_evidence_model import load_models,score,correct,COLS,ROOT,FAMILIES
from session6_priority_model import load_models as load_priority,score as priority_score
DEST=Path('artifacts/evidence_session6');C=pl.col
def main():
    prefix=os.environ.get('PRIORITY_PREFIX','priority')
    models=load_models();pri,cols=load_priority(prefix);ix=pl.read_parquet(ROOT/'hand_index.parquet')
    data=pl.read_parquet('artifacts/evidence_windows/window_features.parquet').join(ix.select('pair_id','hand_id','time','fold','net_direction'),on=['pair_id','hand_id'],validate='m:1')
    full=pl.read_parquet(DEST/f'{prefix}_oof.parquet').select('pair_id','hand_id',C('score').alias('full_priority'));rows=[]
    with threadpool_limits(limits=4):
        for window,lo,hi in [('first_2000',0,2000),('last_2000',1000,3000)]:
            z=data.filter(C('window')==window).with_columns(pl.lit('development').alias('phase'),((C('time')*5000-lo)/(hi-lo)).alias('relative_time'));z,_=augment(z)
            for f in range(4):
                weights=np.load(ROOT/f'nested_blend/unary_weights_fold{f}.npy')
                for family in FAMILIES:
                    q=z.filter((C('fold')==f)&(C('behavior_family')==family))
                    if not len(q):continue
                    base=score(models,family,q.select(COLS).to_numpy(),[f]);p=priority_score(pri,cols,family,q,[f])
                    q=q.with_columns(pl.Series('r27',base),pl.Series('priority',p),pl.Series('mixed',.5*base+.5*p)).join(full,on=['pair_id','hand_id'],validate='1:1')
                    q=q.with_columns(((C('r27')+C('full_priority'))*.5).alias('full_history_reference_mix'))
                    for (pid,),g in q.group_by('pair_id'):
                        true=set(g.filter(C('evidence')==1)['hand_id']);den=min(5,len(true))
                        if not den:continue
                        row={'pair_id':pid,'table_id':g['table_id'][0],'fold':f,'window':window,'family':family}
                        for name in ['r27','priority','mixed','full_history_reference_mix']:
                            hands=correct(g.with_columns(C(name).alias('base_score')),weights);y=np.array([h in true for h in hands]);row[name]=float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/den)
                        hands=g.sort(['mixed','hand_id'],descending=[True,False])['hand_id'].to_list()[:5];y=np.array([h in true for h in hands]);row['mixed_raw']=float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/den)
                        rows.append(row)
            print('completed',window,flush=True)
    r=pl.DataFrame(rows);r.write_csv(DEST/f'{prefix}_windows.csv')
    results={'caveat':__doc__,'windows':r.group_by('window').agg(pl.len(),C('r27','priority','mixed','mixed_raw','full_history_reference_mix').mean()).to_dicts(),'families':r.group_by('window','family').agg(pl.len(),C('r27','priority','mixed','mixed_raw','full_history_reference_mix').mean()).to_dicts()}
    (DEST/f'{prefix}_windows.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()
