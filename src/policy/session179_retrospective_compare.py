import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json
from pathlib import Path
import numpy as np,polars as pl
from session152_generic_fallback import ap
import session174_direct_list_membership as baseline
ROOT=Path('artifacts/evidence_session179_retrospective_compare');C=pl.col
SOURCE=Path('artifacts/evidence_session178_retrospective_membership')

def main():
    for p in [SOURCE,baseline.ROOT]:
        v=json.load(open(p/'verification.json'));assert v['models']==16 and v['final_prediction_error']==0 and v['family_exclusions'] and v['pool_exclusions']
    old=pl.read_csv('artifacts/evidence_session175_membership_compare/pair_comparison.csv');rows=[]
    for held in ['all']+baseline.s.FAMILIES:
        d=pl.read_parquet(SOURCE/held/'oof.parquet')
        for (pid,),g in d.group_by('pair_id'):rows.append(dict(held=held,pair_id=pid,outcomes=ap(g)))
    q=old.join(pl.DataFrame(rows),on=['held','pair_id'],validate='1:1');assert len(q)==744
    ROOT.mkdir(exist_ok=True);q.write_csv(ROOT/'pair_comparison.csv');out={}
    for held in ['all','withheld']+baseline.s.FAMILIES:
        z=q.filter(C('held')!='all') if held=='withheld' else q.filter(C('held')==held)
        p=z.group_by('table_id').agg(C('outcomes','direct','cold').sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(17901).integers(0,len(p),(5000,len(p)))
        result=dict(MAP={n:z[n].mean() for n in ['outcomes','direct','cold']},folds=z.group_by('fold').agg(C('outcomes','direct','cold').mean()).sort('fold').to_dicts(),differences={})
        for name in ['direct','cold']:
            dd=p['outcomes'].to_numpy()-p[name].to_numpy();boot=dd[ix].sum(1)/p['n'].to_numpy()[ix].sum(1)
            result['differences'][name]=dict(delta=z['outcomes'].mean()-z[name].mean(),pool_CI95_fixed_predictions=np.quantile(boot,[.025,.975]).tolist())
        out[held]=result
    report=dict(results=out,limitations='Exploratory fixed comparisons on reused public pools. Raw post-hand outcomes are legitimate for retrospective evidence, but not ordinary policy inputs. No hidden-family score or leaderboard gain inferred.')
    (ROOT/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
