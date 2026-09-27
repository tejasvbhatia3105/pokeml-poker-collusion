import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import polars as pl,numpy as np
from session8_data import reference
ROOT=Path('artifacts/evidence_session12');C=pl.col
def compare(path,columns,prefix):
    ref=reference();truth={pid:set(g.filter(C('evidence')==1)['hand_id']) for (pid,),g in ref.group_by('pair_id')};meta=ref.group_by('pair_id').agg(C('table_id').first(),C('fold').first(),C('behavior_family').first().alias('family'));r30=pl.read_parquet('artifacts/evidence_session11/conditional_family_deployed_oof.parquet').select('pair_id','hand_id',C('conditional_family').alias('r30'));route=pl.read_csv('artifacts/evidence_session9/routed_evidence.csv').filter(C('window')=='full').select('pair_id','risk_score');assert route['risk_score'].min()>=.01
    d=pl.read_parquet(path).select('pair_id','hand_id',*columns).join(r30,on=['pair_id','hand_id'],validate='1:1').join(route,on='pair_id',validate='m:1').with_columns(*[pl.when(C('risk_score')<.05).then(C('r30')).otherwise(C(n)).alias(n) for n in columns]);assert len(d)==len(r30)
    rows=[]
    for (pid,),g in d.group_by('pair_id'):
        row={'pair_id':pid}
        for name in ['r30']+columns:
            h=g.sort(name,'hand_id',descending=[True,False])['hand_id'].to_list()[:5];y=np.array([v in truth[pid] for v in h]);row[name]=float((y*y.cumsum()/np.arange(1,len(y)+1)).sum()/min(5,len(truth[pid])));row[name+'_hands']=' '.join(h)
        rows.append(row)
    r=meta.join(pl.DataFrame(rows),on='pair_id',validate='1:1').sort('pair_id');names=['r30']+columns;pool=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(1212).integers(0,len(pool),(5000,len(pool)));report={}
    for name in names:
        delta=pool[name].to_numpy()-pool['r30'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);report[name]={'routed_map':r[name].mean(),'gain':r[name].mean()-r['r30'].mean(),'ci95_fixed_predictions':np.quantile(boot,[.025,.975]).tolist(),'folds':r.group_by('fold').agg(C(name).mean()).sort('fold')[name].to_list(),'families':dict(r.group_by('family').agg(C(name).mean()).iter_rows()),'improved':int((r[name]>r['r30']+1e-12).sum()),'worse':int((r[name]<r['r30']-1e-12).sum())}
    r.write_csv(ROOT/f'{prefix}_comparison.csv');(ROOT/f'{prefix}_comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return r,report
if __name__=='__main__':
    import sys
    compare(sys.argv[1],sys.argv[2].split(','),sys.argv[3])
