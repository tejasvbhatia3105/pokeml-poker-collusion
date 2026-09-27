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
from session4_evidence_model import load_models as load_base,score as base_score,COLS,FAMILIES
from session6_priority_model import load_models as load_priority,score as priority_score
from session7_model import load_models,score
ROOT=Path('artifacts/evidence_session8');C=pl.col
NAMES=['r29_window','past_context','past_context_normalized','window_magnitude_boost','past_direction_only','past_magnitude_only']
def main():
    ix=pl.read_parquet('artifacts/evidence_session4/hand_index.parquet');data=pl.read_parquet('artifacts/evidence_windows/window_features.parquet').filter(C('window')=='last_2000').join(ix.select('pair_id','hand_id','time','fold','net_direction'),on=['pair_id','hand_id'],validate='1:1').with_columns(pl.lit('development').alias('phase'),((C('time')*5000-1000)/2000).alias('relative_time'));data,_=augment(data)
    full=pl.read_parquet(list(Path('artifacts/policy/relationship_evidence').glob('T*.parquet')));cols=[c for c in full.columns if c.startswith('relationship_') and '_local_' not in c]
    counts=ix.group_by('pair_id').len().rename({'len':'full_n'}).join(data.group_by('pair_id').len().rename({'len':'window_n'}),on='pair_id');z=data.join(full.select('pair_id','hand_id',*[C(c).alias('full_'+c) for c in cols]),on=['pair_id','hand_id'],validate='1:1').join(counts,on='pair_id',validate='m:1');base=load_base();old,pc=load_priority('priority_ordered');hist,_=load_models('hist_eventblend');rows=[]
    with threadpool_limits(limits=4):
        for f in range(4):
            for b in FAMILIES:
                q=z.filter((C('fold')==f)&(C('behavior_family')==b));out=q.select('pair_id','hand_id','evidence','table_id','fold','behavior_family')
                for mode in NAMES:
                    v=q
                    if mode!='r29_window':
                        factor=(C('window_n')/C('full_n')).sqrt() if mode.endswith('normalized') else pl.lit(1.)
                        v=q.with_columns(*[(C('full_'+c)*factor).alias(c) for c in cols])
                    if mode=='window_magnitude_boost':
                        v=q.with_columns(*[(C(c)*(C('full_n')/C('window_n')).sqrt()).alias(c) for c in cols])
                    if mode in ['past_direction_only','past_magnitude_only']:
                        expr=[]
                        for channel in ['call','fold','raise','outside_agg','outside_fold','hu_check']:
                            prefix='relationship_'+channel+'_';st=prefix+'reference_strength';co=prefix+'coherence';pa=prefix+'payoff_alignment'
                            flip=pl.when((C(co)*C('full_'+co)<0)|(C(pa)*C('full_'+pa)<0)).then(-1).otherwise(1)
                            ratio=C('full_'+st)/(C(st)+1e-12)
                            for c in [st,co,pa]:
                                expr.append((C(c)*(ratio if mode=='past_magnitude_only' else (pl.lit(1.) if c==st else flip))).alias(c))
                        v=q.with_columns(*expr)
                    pred=.25*base_score(base,b,v.select(COLS).to_numpy(),[f])+.25*priority_score(old,pc,b,v,[f])+.5*score(hist,pc,b,v,[f]);out=out.with_columns(pl.Series(mode,pred))
                for (pid,),g in out.group_by('pair_id'):
                    truth=set(g.filter(C('evidence')==1)['hand_id']);den=min(5,len(truth))
                    if not den:continue
                    row={'pair_id':pid,'table_id':g['table_id'][0],'fold':f,'family':b}
                    for mode in NAMES:
                        h=g.sort([mode,'hand_id'],descending=[True,False])['hand_id'].to_list()[:5];y=np.array([v in truth for v in h]);row[mode]=float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/den)
                    rows.append(row)
    r=pl.DataFrame(rows);r.write_csv(ROOT/'past_history_probe.csv');names=NAMES;report={'caveat':__doc__,'overall':r.select(C(names).mean()).to_dicts()[0],'family':r.group_by('family').agg(C(names).mean()).to_dicts(),'fold':r.group_by('fold').agg(C(names).mean()).sort('fold').to_dicts()};(ROOT/'past_history_probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
