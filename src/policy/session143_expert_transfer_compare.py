import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json
import numpy as np,polars as pl
from session142_family_expert_transfer import ROOT,FAMILIES
C=pl.col

def ap(hands,score,truth):
    ix=np.lexsort((hands,-score))[:5];y=np.isin(hands[ix],list(truth));return float((y*y.cumsum()/np.arange(1,len(y)+1)).sum()/min(5,len(truth)))

def main():
    assert (ROOT/'audit.json').exists();d=pl.read_parquet(sorted(ROOT.glob('T*.parquet')));truth=pl.read_csv('data/development_evidence.csv');tm={pid:set(g['hand_id']) for (pid,),g in truth.group_by('pair_id')}
    assert d.select('pair_id','hand_id','expert').n_unique()==len(d)==45129*3 and set(d['pair_id'])==set(tm)
    r33=pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id','equal');d=d.join(r33,on=['pair_id','hand_id'],validate='m:1');rows=[]
    for (pid,),g in d.group_by('pair_id'):
        row=dict(pair_id=pid,table_id=g['table_id'][0],fold=g['fold'][0],family=g['true_family'][0]);native=FAMILIES.index(row['family']);others=[i for i in range(3) if i!=native]
        frames=[g.filter(C('expert')==fam).sort('hand_id') for fam in FAMILIES];hands=frames[0]['hand_id'].to_numpy()
        assert all(np.array_equal(hands,z['hand_id'].to_numpy()) for z in frames);row['r33_known_reference']=ap(hands,frames[0]['equal'].to_numpy(),tm[pid])
        for score in ['raw_score','conditioned_score']:
            prefix=score.removesuffix('_score');p=np.stack([z[score].to_numpy() for z in frames]);vals=[ap(hands,s,tm[pid]) for s in p]
            for fam,v in zip(FAMILIES,vals):row[prefix+'_'+fam]=v
            row[prefix+'_native']=vals[native];row[prefix+'_other_mean']=ap(hands,p[others].mean(0),tm[pid]);row[prefix+'_other_max']=ap(hands,p[others].max(0),tm[pid]);row[prefix+'_other_best_single_oracle']=max(vals[i] for i in others);row[prefix+'_all_mean']=ap(hands,p.mean(0),tm[pid])
        rows.append(row)
    q=pl.DataFrame(rows).sort('pair_id');q.write_csv(ROOT/'pair_comparison.csv');names=[c for c in q.columns if c not in ['pair_id','table_id','fold','family']];report={}
    for n in names:report[n]=dict(MAP=q[n].mean(),families=dict(q.group_by('family').agg(C(n).mean()).iter_rows()),folds=q.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list())
    pool=q.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(14301).integers(0,len(pool),(5000,len(pool)));ci={}
    for prefix in ['raw','conditioned']:
        for arm in ['other_mean','other_max','other_best_single_oracle','all_mean']:
            a=prefix+'_'+arm;b=prefix+'_native';dd=pool[a].to_numpy()-pool[b].to_numpy();boot=dd[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);ci[a]=dict(delta_vs_native=q[a].mean()-q[b].mean(),pool_CI95_fixed_predictions=np.quantile(boot,[.025,.975]).tolist())
    result=dict(metrics=report,differences=ci,limitations='Conditional evidence retrieval; other-family experts exclude target-family supervision and source pool. Shared R33 corrections absent. R33 reference is not a matched withheld-family comparator. Best-single oracle uses truth and is not deployable; no claim about actual hidden fourth-family score.')
    (ROOT/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
