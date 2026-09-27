\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json
from pathlib import Path
import numpy as np,polars as pl
from session6_priority import inclusion

ROOT=Path('artifacts/evidence_session156_evaluation_support_bound');C=pl.col

def main():
    ROOT.mkdir(exist_ok=True)
    scores=pl.read_parquet('artifacts/candidate_r33/evidence_scores.parquet')
    selected=scores['pair_id'].unique().to_list()
    caches=[]
    fields=['pair_id','hand_id']+[f'{n}_{f}' for f in range(4) for n in ['base','hist_primary','hist_secondary']]
    for path in sorted(Path('artifacts/evidence_session11/eval_cache').glob('T*.parquet')):
        q=pl.read_parquet(path,columns=fields).filter(C('pair_id').is_in(selected))
        if len(q):caches.append(q.with_columns(pl.lit(path.stem).alias('table_id')))
    q=scores.join(pl.concat(caches),on=['pair_id','hand_id'],validate='1:1')
    assert len(q)==60727 and q['pair_id'].n_unique()==768
    rows=[]
    for (pid,),g in q.group_by('pair_id'):
        g=g.sort('time','hand_id');raw=[]
        for f in range(4):
            cat=g.select(f'new_primary_{f}',f'new_secondary_{f}').to_numpy();cat=cat/np.maximum(1,cat.sum(1))[:,None]
            hist=g.select(f'hist_primary_{f}',f'hist_secondary_{f}').to_numpy();hist=hist/np.maximum(1,hist.sum(1))[:,None]
            joint=.5*(cat+hist)
            raw.append(.25*g[f'base_{f}'].to_numpy()+.25*inclusion(*cat.T)+.5*inclusion(*joint.T))
        mean=np.mean(raw,axis=0);support=float(np.sort(mean)[-5:].mean())
        rows.append(dict(pair_id=pid,table_id=g['table_id'][0],routed_family=g['behavior_family'][0],hands=len(g),native_support=support,can_pass_all_expert_gate=support<.5))
    r=pl.DataFrame(rows).join(pl.read_csv('artifacts/candidate_r33/submission.csv').select('pair_id','risk_score'),on='pair_id',validate='1:1')
    r=r.with_columns(pl.when(C('risk_score')>=.5).then(pl.lit('at_least_.5')).otherwise(pl.lit('.05_to_.5')).alias('risk_band'))
    r.write_parquet(ROOT/'support.parquet')
    report=dict(method=__doc__,rescored_pairs=len(r),necessary_condition_pairs=int(r['can_pass_all_expert_gate'].sum()),
        by_risk=r.group_by('risk_band').agg(pl.len().alias('pairs'),C('can_pass_all_expert_gate').sum().alias('possible_switches')).sort('risk_band').to_dicts(),
        by_family=r.group_by('routed_family').agg(pl.len().alias('pairs'),C('can_pass_all_expert_gate').sum().alias('possible_switches')).sort('routed_family').to_dicts(),
        limitations='Only the 768 pairs already scored by advanced R33 evidence models (risk >=.05). Unknown true-positive counts, private/public partition, evidence accuracy and remaining experts support. This is not a score ceiling or validated routing rule.')
    (ROOT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
