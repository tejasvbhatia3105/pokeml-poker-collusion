import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import polars as pl,numpy as np
from session8_data import reference
ROOT=Path('artifacts/evidence_session11');C=pl.col
def compare(path,columns,prefix):
    ref=reference();truth={pid:set(g.filter(C('evidence')==1)['hand_id']) for (pid,),g in ref.group_by('pair_id')};meta=ref.group_by('pair_id').agg(C('table_id').first(),C('fold').first(),C('behavior_family').first().alias('family'))
    sources={'r29':(ref,'r29')};d=pl.read_parquet(path)
    for name in columns:sources[name]=(d,name)
    r=meta.sort('pair_id').join(pl.read_csv('artifacts/evidence_session9/routed_evidence.csv').filter(C('window')=='full').select('pair_id','below_gate','risk_score',C('routed_r29').alias('fallback')),on='pair_id',validate='1:1',maintain_order='left');ids=r['pair_id'].to_list()
    for name,(q,col) in sources.items():
        scores={};hands={}
        for (pid,),g in q.group_by('pair_id'):
            h=g.sort(col,'hand_id',descending=[True,False])['hand_id'].to_list()[:5];y=np.array([v in truth[pid] for v in h]);scores[pid]=float((y*y.cumsum()/np.arange(1,len(y)+1)).sum()/min(5,len(truth[pid])));hands[pid]=' '.join(h)
        assert set(scores)==set(ids)
        r=r.with_columns(pl.Series(name,[scores[p] for p in ids]),pl.Series(name+'_hands',[hands[p] for p in ids])).with_columns(pl.when(C('below_gate') if name=='r29' else C('risk_score')<float(os.environ.get('COMPARE_GATE','.05'))).then(C('fallback')).otherwise(C(name)).alias(name))
    names=list(sources);pool=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(1111).integers(0,len(pool),(5000,len(pool)));report={}
    for name in names:
        delta=pool[name].to_numpy()-pool['r29'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);report[name]={'routed_map':r[name].mean(),'gain':r[name].mean()-r['r29'].mean(),'ci95':np.quantile(boot,[.025,.975]).tolist(),'folds':r.group_by('fold').agg(C(name).mean()).sort('fold')[name].to_list(),'families':dict(r.group_by('family').agg(C(name).mean()).iter_rows()),'improved':int((r[name]>r['r29']+1e-12).sum()),'worse':int((r[name]<r['r29']-1e-12).sum())}
    r.write_csv(ROOT/f'{prefix}_comparison.csv');(ROOT/f'{prefix}_comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return r,report
if __name__=='__main__':
    import sys
    compare(ROOT/sys.argv[1],sys.argv[2].split(','),sys.argv[3])
