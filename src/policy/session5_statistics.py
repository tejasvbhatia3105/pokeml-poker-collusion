from pathlib import Path
import json,numpy as np,polars as pl
ROOT=Path('artifacts/evidence_session5');C=pl.col
ix=pl.read_parquet('artifacts/evidence_session4/hand_index.parquet').select('pair_id','table_id').unique();stats={}
for name in ['pairwise','pairwise_hist','pairwise_residual','pairwise_context','nonlinear_context']:
    d=pl.read_csv(ROOT/f'{name}_comparison.csv').join(ix,on='pair_id');names=[c for c in d.columns if c not in ['pair_id','fold','family','table_id','r27']]
    stats[name]={}
    for col in names:
        q=d.group_by('table_id').agg((C(col)-C('r27')).sum().alias('delta'),pl.len().alias('n')).sort('table_id');rng=np.random.default_rng(550);ind=rng.integers(0,len(q),(5000,len(q)));boot=q['delta'].to_numpy()[ind].sum(1)/q['n'].to_numpy()[ind].sum(1)
        stats[name][col]={'map5':d[col].mean(),'gain':d[col].mean()-d['r27'].mean(),'ci95_fixed_predictions':np.quantile(boot,[.025,.975]).tolist(),'fraction_bootstrap_positive':float(np.mean(boot>0))}
(ROOT/'bootstrap.json').write_text(json.dumps(stats,indent=2))
w=pl.read_csv(ROOT/'pairwise_windows.csv');names=[c for c in w.columns if c.endswith('quarter') or c=='cat_r27'];windows=w.group_by('window').agg(pl.len(),C(names).mean()).to_dicts();(ROOT/'window_summary.json').write_text(json.dumps(windows,indent=2))
print('Context comparator:',stats['pairwise_context']['comparator_quarter']);print('Nonlinear unary:',stats['nonlinear_context']['hist_0.25']);print('Windows:',windows)
