import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl
from threadpoolctl import threadpool_limits
from session8_exposure_training import data
from session8_data import ROOT,C
from session8_count_conditioning import categorical,conditioned
from session6_priority import inclusion
from session7_model import load_models
def main():
    _,d,_=data();d=d.filter(C('window')!='full').join(pl.read_parquet(ROOT/'window_r29_predictions.parquet'),on=['pair_id','hand_id','window'],validate='1:1');models,cols=load_models('hist_eventblend');parts=[];rows=[]
    with threadpool_limits(limits=4):
        for (w,f,b),q in d.group_by('window','fold','behavior_family'):
            _,hist,cat=models[b][f];x=q.select(cols).to_numpy();ca=categorical(*[m.predict_proba(x)[:,1] for m in cat]);hp=categorical(*[m.predict_proba(x)[:,1] for m in hist]);q=q.with_columns(pl.Series('ca',ca[:,0]),pl.Series('cb',ca[:,1]),pl.Series('ha',hp[:,0]),pl.Series('hb',hp[:,1]));parts.append(q.select('pair_id','hand_id','window','ca','cb','ha','hb','r29'))
            for (pid,),g in q.group_by('pair_id'):
                g=g.sort('time','hand_id');a=g.select('ca','cb').to_numpy();h=(a+g.select('ha','hb').to_numpy())/2;old=.25*inclusion(*a.T)+.5*inclusion(*h.T);base=g['r29'].to_numpy()-old;g=g.with_columns(*[pl.Series('minimum_'+str(k),base+.25*conditioned(a,k)+.5*conditioned(h,k)) for k in [3,5]])
                truth=set(g.filter(C('evidence')==1)['hand_id']);den=min(5,len(truth))
                if not den:continue
                row={'pair_id':pid,'window':w,'fold':f,'family':b}
                for name in ['r29','minimum_3','minimum_5']:
                    hand=g.sort([name,'hand_id'],descending=[True,False])['hand_id'].to_list()[:5];y=np.array([x in truth for x in hand]);row[name]=float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/den)
                rows.append(row)
    pl.concat(parts).write_parquet(ROOT/'window_event_probabilities.parquet');r=pl.DataFrame(rows);r.write_csv(ROOT/'count_windows.csv');names=['r29','minimum_3','minimum_5'];report={'caveat':__doc__,'windows':r.group_by('window').agg(C(names).mean()).to_dicts(),'families':r.group_by('window','family').agg(C(names).mean()).to_dicts(),'folds':r.group_by('window','fold').agg(C(names).mean()).to_dicts()};(ROOT/'count_windows.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
