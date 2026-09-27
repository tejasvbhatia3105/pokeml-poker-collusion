import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from threadpoolctl import threadpool_limits
from session8_exposure_training import data
from session8_event_model import load_models,score,add_features,ROOT,FAMILIES
from session4_evidence_model import load_models as load_base,score as base_score,COLS
from session6_priority_model import load_models as load_priority,score as priority_score
from session7_model import load_models as load_hist,score as hist_score
C=pl.col
def main():
    mode=os.environ['METHOD'];_,d,_=data();d=d.filter(C('window')!='full');cache=ROOT/'window_r29_predictions.parquet'
    with threadpool_limits(limits=4):
        if not cache.exists():
            base=load_base();old,pc=load_priority('priority_ordered');hist,_=load_hist('hist_eventblend');parts=[]
            for w in ['first_2000','last_2000']:
                for f in range(4):
                    for b in FAMILIES:
                        q=d.filter((C('window')==w)&(C('fold')==f)&(C('behavior_family')==b));p=.25*base_score(base,b,q.select(COLS).to_numpy(),[f])+.25*priority_score(old,pc,b,q,[f])+.5*hist_score(hist,pc,b,q,[f]);parts.append(q.select('pair_id','hand_id','window').with_columns(pl.Series('r29',p)))
            pl.concat(parts).write_parquet(cache)
        d=d.join(pl.read_parquet(cache),on=['pair_id','hand_id','window'],validate='1:1',maintain_order='left');models,cols=load_models(mode);d=add_features(d,mode);parts=[]
        for w in ['first_2000','last_2000']:
            for f in range(4):
                for b in FAMILIES:
                    q=d.filter((C('window')==w)&(C('fold')==f)&(C('behavior_family')==b));p=score(models,cols,b,q,[f]);parts.append(q.select('pair_id','hand_id','window','fold','behavior_family','evidence','r29').with_columns(pl.Series('new',p)))
    d=pl.concat(parts).with_columns((.5*C('new')+.5*C('r29')).alias('half'),(.25*C('new')+.75*C('r29')).alias('quarter'));names=['r29','new','half','quarter'];rows=[]
    d.write_parquet(ROOT/f'{mode}_window_comparison_predictions.parquet')
    for (w,pid),g in d.group_by('window','pair_id'):
        truth=set(g.filter(C('evidence')==1)['hand_id']);den=min(5,len(truth))
        if not den:continue
        row={'window':w,'pair_id':pid,'family':g['behavior_family'][0],'fold':g['fold'][0]}
        for name in names:
            hand=g.sort([name,'hand_id'],descending=[True,False])['hand_id'].to_list()[:5];y=np.array([h in truth for h in hand]);row[name]=float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/den)
        rows.append(row)
    r=pl.DataFrame(rows);r.write_csv(ROOT/f'{mode}_windows.csv');report={'caveat':__doc__,'windows':r.group_by('window').agg(pl.len(),C(names).mean()).to_dicts(),'families':r.group_by('window','family').agg(C(names).mean()).to_dicts(),'folds':r.group_by('window','fold').agg(C(names).mean()).to_dicts()};(ROOT/f'{mode}_windows.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
