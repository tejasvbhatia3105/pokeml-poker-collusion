import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl
from session8_data import hand_data
from session6_priority import inclusion
from session10_nested6 import ROOT
C=pl.col
def main():
    d=hand_data();outer_tables={f:set(d.filter(C('fold')==f)['table_id']) for f in range(4)};rows=[];seen=0;replay={}
    for f in range(4):
        old=pl.read_parquet(f'artifacts/evidence_session9/nested_outer{f}.parquet');final=ROOT/f'nested_outer{f}.parquet'
        if final.exists():
            val=pl.read_parquet(final).filter(C('fold')==f).join(old.filter(C('fold')==f).select('pair_id','hand_id',C('r29').alias('expected')),on=['pair_id','hand_id'],validate='1:1');error=float((val['r29']-val['expected']).abs().max());assert error==0;replay[str(f)]=error
        for path in sorted(ROOT.glob(f'teacher_outer{f}_*.parquet')):
            if not path.with_suffix('.json').exists():continue
            audit=json.loads(path.with_suffix('.json').read_text())
            for a in audit:
                assert not set(a['training_tables'])&outer_tables[f];assert not set(a['training_tables'])&set(a['prediction_tables']);seen+=1
            q=pl.read_parquet(path).join(d.select('pair_id','hand_id','evidence'),on=['pair_id','hand_id'],validate='1:1').join(old.select('pair_id','hand_id',C('r29').alias('old_r29')),on=['pair_id','hand_id'],validate='1:1')
            for (pid,),g in q.group_by('pair_id'):
                g=g.sort('time','hand_id');ca=g['cat_primary'].to_numpy();cb=g['cat_secondary'].to_numpy();ha=g['hist_primary'].to_numpy();hb=g['hist_secondary'].to_numpy();cs=np.maximum(1,ca+cb);hs=np.maximum(1,ha+hb);s=.25*g['base'].to_numpy()+.25*inclusion(ca,cb)+.5*inclusion(.5*(ca/cs+ha/hs),.5*(cb/cs+hb/hs));hand=g['hand_id'].to_numpy();y=g['evidence'].to_numpy();den=min(5,y.sum());row={'outer_fold':f,'pair_id':pid,'block':path.stem}
                for name,score in [('old',g['old_r29'].to_numpy()),('six_split',s)]:
                    ix=np.lexsort((hand,-score))[:5];hit=y[ix];row[name]=float((hit*np.cumsum(hit)/np.arange(1,6)).sum()/den)
                rows.append(row)
    r=pl.DataFrame(rows);r.write_csv(ROOT/'input_quality.csv');report={'heads_audited':seen,'validation_replay_max_errors':replay,'completed_training_prediction_pairs':len(rows),'old_teacher_map_on_completed_rows':r['old'].mean(),'six_split_teacher_map_on_completed_rows':r['six_split'].mean(),'caveat':'Training-input quality diagnostic only; includes repeated pairs across outer-training roles, not final model validation.'};(ROOT/'input_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
