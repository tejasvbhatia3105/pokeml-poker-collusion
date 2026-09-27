\
\
\
import os; os.environ.setdefault('POLARS_MAX_THREADS','6')
import numpy as np,polars as pl,time
from pathlib import Path
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
rows=[]; t0=time.time()
for i,p in enumerate(sorted(Path('artifacts/policy/actions').glob('*.parquet'))):
    t=p.stem
    a=pl.read_parquet(p,columns=['hand_id','player_id','phase','street_no','action_class','equity','action_no']).filter(C('street_no')==0)
    first=a.sort('action_no').group_by('hand_id','player_id',maintain_order=True).agg(C('action_class').first().alias('a'),C('equity').first().alias('eP'),C('action_no').first().alias('no'),C('phase').first())
    first=first.with_columns(pl.when(C('a')==3).then(2.0).when(C('a')==0).then(0.0).otherwise(1.0).alias('agg'))
    st=pl.read_parquet(f'artifacts/policy/states/{t}.parquet',columns=['hand_id','player_id','street_no','equity']).filter(C('street_no')==0).select('hand_id',C('player_id').alias('q'),C('equity').alias('eQ'))
    qno=first.select('hand_id',C('player_id').alias('q'),C('no').alias('qno'))
    j=first.join(st,on='hand_id').filter(C('q')!=C('player_id')).join(qno,on=['hand_id','q'],how='left').with_columns((C('qno')<C('no')).fill_null(False).alias('q_before')).with_columns(pl.lit(t).alias('table'))
    rows.append(j.select('table','phase','hand_id','player_id','q','agg','eP','eQ','q_before'))
    if i%50==0: print(i,round(time.time()-t0),flush=True)
d=pl.concat(rows); print('rows',d.height,flush=True)
                                                    
d=d.with_columns((C('eP').rank('ordinal').over('phase')*20/pl.len().over('phase')).floor().clip(0,19).alias('bin'))
d=d.with_columns((C('agg')-C('agg').mean().over('phase','bin')).alias('ra'),(C('eQ')-C('eQ').mean().over('phase')).alias('rq'))
def zs(x):
    g=x.group_by('table','phase','player_id','q').agg(pl.len().alias('n'),(C('ra')*C('rq')).mean().alias('cov'),C('ra').std().alias('sa'),C('rq').std().alias('sq'))
    return g.with_columns((C('cov')/(C('sa')*C('sq')+1e-9)*C('n').sqrt()).alias('z'))
allz=zs(d).rename({'n':'n_all','z':'z_all'}).select('table','phase','player_id','q','n_all','z_all')
after=zs(d.filter(~C('q_before'))).rename({'n':'n_after','z':'z_after'}).select('table','phase','player_id','q','n_after','z_after')
before=zs(d.filter(C('q_before'))).rename({'n':'n_before','z':'z_before'}).select('table','phase','player_id','q','n_before','z_before')
out=allz.join(after,on=['table','phase','player_id','q'],how='left').join(before,on=['table','phase','player_id','q'],how='left')
out.write_parquet(S+'card_share_z.parquet'); print('saved',out.shape,round(time.time()-t0),flush=True)
