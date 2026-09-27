\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import json,time
from pathlib import Path
import numpy as np,polars as pl
from threadpoolctl import threadpool_limits
import session142_family_expert_transfer as expert
from session6_priority import inclusion

ROOT=Path('artifacts/evidence_session157_evaluation_experts');C=pl.col

def main():
    ROOT.mkdir(exist_ok=True);bound=pl.read_parquet('artifacts/evidence_session156_evaluation_support_bound/support.parquet')
    selected=bound.filter(C('can_pass_all_expert_gate'))['pair_id'].to_list();assert len(selected)==280
    reference=pl.read_parquet('artifacts/candidate_r33/evidence_scores.parquet').filter(C('pair_id').is_in(selected))
    players=pl.read_csv('data/evaluation_pairs.csv').select('pair_id','player_1','player_2');models=expert.load_models();audits=[];parts=[];start=time.time()
    with threadpool_limits(limits=2):
        for path in sorted(Path('artifacts/evidence_session11/eval_cache').glob('T*.parquet')):
            d=pl.read_parquet(path).filter(C('pair_id').is_in(selected)).with_columns(pl.lit(path.stem).alias('table_id'))
            if not len(d):continue
            target=ROOT/path.name
            if target.exists():parts.append(pl.read_parquet(target));audits.append(json.load(open(target.with_suffix('.json'))));continue
            truthfree=d.drop('behavior_family','evidence','evidence_rank','subtype',strict=False)
            records=[];error=0.
            for fold in range(4):
                q,pp,info=expert.predictions(truthfree.with_columns(pl.lit(fold).alias('fold')),players,models)
                z=q.select('pair_id','hand_id','time').join(reference.select('pair_id','hand_id','behavior_family',f'new_primary_{fold}',f'new_secondary_{fold}'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
                for fam in expert.FAMILIES:
                    v=pp[fam];ix=z['behavior_family'].to_numpy()==fam
                    if ix.any():
                        refs=[(v['cat'][:,0],z[f'new_primary_{fold}'].to_numpy()),(v['cat'][:,1],z[f'new_secondary_{fold}'].to_numpy()),
                              (v['base'],q[f'base_{fold}'].to_numpy()),(v['hist'][:,0],q[f'hist_primary_{fold}'].to_numpy()),(v['hist'][:,1],q[f'hist_secondary_{fold}'].to_numpy())]
                        error=max(error,max(float(abs(a[ix]-b[ix]).max()) for a,b in refs));assert error<1e-10,(path.stem,fold,fam,error)
                    for (pid,),g in q.with_row_index('ix').group_by('pair_id'):
                        g=g.sort('time','hand_id');i=g['ix'].to_numpy();cat=v['cat'][i];cat=cat/np.maximum(1,cat.sum(1))[:,None]
                        hist=v['hist'][i];hist=hist/np.maximum(1,hist.sum(1))[:,None];joint=.5*(cat+hist)
                        score=.25*v['base'][i]+.25*inclusion(*cat.T)+.5*inclusion(*joint.T)
                        records.append(g.select('pair_id','hand_id').with_columns(pl.lit(fam).alias('expert'),pl.lit(fold).alias('model_fold'),pl.Series('raw_score',score)))
            out=pl.concat(records).group_by('pair_id','hand_id','expert').agg(C('raw_score').mean());out.write_parquet(target)
            audit=dict(table=path.stem,pairs=d['pair_id'].n_unique(),hands=len(d),native_probability_replay_error=error)
            target.with_suffix('.json').write_text(json.dumps(audit,indent=2));audits.append(audit);parts.append(out)
            if len(audits)%20==0:print('EVALUATION_EXPERT_SUPPORT',len(audits),round(time.time()-start,1),flush=True)
    scores=pl.concat(parts);rows=[]
    for (pid,fam),g in scores.group_by('pair_id','expert'):
        rows.append(dict(pair_id=pid,expert=fam,raw_top5_confidence=float(np.sort(g['raw_score'].to_numpy())[-5:].mean())))
    support=pl.DataFrame(rows);support.write_parquet(ROOT/'expert_support.parquet')
    q=support.group_by('pair_id').agg(C('raw_top5_confidence').max().alias('maximum_support')).join(bound,on='pair_id',validate='1:1')
    q=q.with_columns((C('maximum_support')<.5).alias('passes_fixed_gate'));assert len(q)==280
                                                                       
    native=support.join(bound.select('pair_id','routed_family','native_support'),on='pair_id',validate='m:1').filter(C('expert')==C('routed_family'))
    support_error=float((native['raw_top5_confidence']-native['native_support']).abs().max());assert support_error<1e-10
    q.write_parquet(ROOT/'gate_coverage.parquet')
    report=dict(method=__doc__,possible_pairs=len(q),passing_pairs=int(q['passes_fixed_gate'].sum()),
        native_support_replay_error=support_error,maximum_native_probability_replay_error=max(a['native_probability_replay_error'] for a in audits),
        by_risk=q.group_by('risk_band').agg(pl.len().alias('possible_pairs'),C('passes_fixed_gate').sum().alias('passing_pairs')).sort('risk_band').to_dicts(),
        limitations='Predicted support only; no hidden truth, generic inference or submission. R33 risks and behaviors remain unchanged. The fixed .5 gate harmed known-family local retrieval in152; passing this gate alone is not evidence of improved accuracy.')
    (ROOT/'report.json').write_text(json.dumps(report,indent=2));(ROOT/'audit.json').write_text(json.dumps(dict(tables=audits,model_hashes=models['hashes']),indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
