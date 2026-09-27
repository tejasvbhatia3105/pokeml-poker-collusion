import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl
from threadpoolctl import threadpool_limits
from sequence_features import augment
from session4_evidence_model import load_models as load_base,score as base_score,COLS,ROOT,FAMILIES
from session6_priority_model import load_models as load_priority,score as priority_score
from session7_model import load_models,score,ROOT as DEST
C=pl.col
def main():
    method=os.environ['METHOD'];base=load_base();old,cols=load_priority('priority_ordered');new,_=load_models(method);ix=pl.read_parquet(ROOT/'hand_index.parquet');data=pl.read_parquet('artifacts/evidence_windows/window_features.parquet').join(ix.select('pair_id','hand_id','time','fold','net_direction'),on=['pair_id','hand_id'],validate='m:1');rows=[]
    with threadpool_limits(limits=4):
        for window,lo,hi in [('first_2000',0,2000),('last_2000',1000,3000)]:
            z=data.filter(C('window')==window).with_columns(pl.lit('development').alias('phase'),((C('time')*5000-lo)/(hi-lo)).alias('relative_time'));z,_=augment(z)
            for f in range(4):
                for b in FAMILIES:
                    q=z.filter((C('fold')==f)&(C('behavior_family')==b))
                    if not len(q):continue
                    a=base_score(base,b,q.select(COLS).to_numpy(),[f]);p=priority_score(old,cols,b,q,[f]);s=score(new,cols,b,q,[f]);q=q.with_columns(pl.Series('r28',.5*a+.5*p),pl.Series('half_base',.5*a+.5*s),pl.Series('quarter_replace',.5*a+.25*p+.25*s),pl.Series('half_r28',.25*a+.25*p+.5*s),pl.Series('standalone',s))
                    for (pid,),g in q.group_by('pair_id'):
                        truth=set(g.filter(C('evidence')==1)['hand_id']);den=min(5,len(truth))
                        if not den:continue
                        row={'pair_id':pid,'table_id':g['table_id'][0],'family':b,'fold':f,'window':window}
                        for name in ['r28','half_base','quarter_replace','half_r28','standalone']:
                            hand=g.sort([name,'hand_id'],descending=[True,False])['hand_id'].to_list()[:5];y=np.array([h in truth for h in hand]);row[name]=float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/den)
                        rows.append(row)
    r=pl.DataFrame(rows);r.write_csv(DEST/f'{method}_windows.csv');report={'caveat':'Partial-context features; original full-phase capped truth truncated to the window, not newly selected truth.','windows':r.group_by('window').agg(pl.len(),C('r28','half_base','quarter_replace','half_r28','standalone').mean()).to_dicts(),'families':r.group_by('window','family').agg(pl.len(),C('r28','half_base','quarter_replace','half_r28','standalone').mean()).to_dicts()};(DEST/f'{method}_windows.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
