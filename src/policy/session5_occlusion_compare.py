import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
import json
from pathlib import Path
import numpy as np,polars as pl
from scipy.stats import rankdata
from session4_set_evidence import design,UNAMES,FAMILIES
ROOT=Path('artifacts/evidence_session5');OLD=Path('artifacts/evidence_session4');C=pl.col
d=pl.read_parquet('artifacts/policy/evidence_training.parquet').join(pl.read_parquet(OLD/'hand_index.parquet').select('pair_id','hand_id','fold'),on=['pair_id','hand_id'])
d=d.join(pl.read_parquet(ROOT/'replay_s1/occlusion_oof.parquet'),on=['pair_id','hand_id'],validate='1:1')
base=pl.concat([pl.read_parquet(OLD/f'nested_blend/nested_outer{f}.parquet').join(d.select('pair_id','hand_id','fold'),on=['pair_id','hand_id']).filter(C('fold')==f).select('pair_id','hand_id','base_score') for f in range(4)])
d=d.join(base,on=['pair_id','hand_id'],validate='1:1');rows=[]
for (pid,),g in d.group_by('pair_id'):
    truth=set(g.filter(C('evidence')==1)['hand_id']);den=min(5,len(truth))
    row={'pair_id':pid,'table_id':g['table_id'][0],'family':g['behavior_family'][0],'fold':g['fold'][0]}
    b=design(g.with_columns((C('relative_time')*.6).alias('time')))
    w=np.load(OLD/f'nested_blend/unary_weights_fold{g["fold"][0]}.npy');n=len(UNAMES);f=FAMILIES.index(b['family']);scores=b['U']@(w[:n]+w[n+f*n:n+(f+1)*n])
    def ap(ids):
        y=np.array([h in truth for h in ids[:5]]);return float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/den)
    row['r27']=ap([b['hand'][i] for i in np.lexsort((np.array(b['hand']),-scores))])
    for key in ['occlusion','family_occlusion']:
        row[key]=ap(g.sort([key,'hand_id'],descending=[True,False])['hand_id'].to_list())
        vals=dict(g.select('hand_id',key).iter_rows());v=np.array([vals[h] for h in b['hand']]);z=(rankdata(v)-6.5)/3.5
        for alpha in [.1,.3,1.]:
            order=np.lexsort((np.array(b['hand']),-(scores+alpha*z)))
            row[f'{key}_a{alpha}']=ap([b['hand'][i] for i in order])
    rows.append(row)
r=pl.DataFrame(rows);r.write_csv(ROOT/'occlusion_comparison.csv')
names=[n for n in r.columns if n not in ['pair_id','table_id','family','fold']]
stats={n:{'map5':r[n].mean(),'folds':r.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list(),'families':dict(r.group_by('family').agg(C(n).mean()).iter_rows())} for n in names}
(ROOT/'occlusion_comparison.json').write_text(json.dumps(stats,indent=2));print(json.dumps({k:v['map5'] for k,v in stats.items()},indent=2))
