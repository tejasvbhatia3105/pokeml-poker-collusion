import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
import polars as pl,numpy as np
from session8_data import ROOT,C
from session8_count_conditioning import categorical
from session6_priority import inclusion
def fuse(a,b,keys):
    d=a.join(b.select(*keys,C('primary').alias('other_primary'),C('secondary').alias('other_secondary')),on=keys,validate='1:1',maintain_order='left');parts=[]
    group=['window','pair_id'] if 'window' in keys else ['pair_id']
    for _,g in d.group_by(group,maintain_order=True):
        g=g.sort('time','hand_id');p=.5*categorical(g['primary'].to_numpy(),g['secondary'].to_numpy())+.5*categorical(g['other_primary'].to_numpy(),g['other_secondary'].to_numpy());parts.append(g.select(*keys,'time').with_columns(pl.Series('primary',p[:,0]),pl.Series('secondary',p[:,1]),pl.Series('score',inclusion(*p.T))))
    return pl.concat(parts)
def main():
    keys=['pair_id','hand_id'];z=fuse(pl.read_parquet(ROOT/'exposure_cat_oof.parquet'),pl.read_parquet(ROOT/'exposure_hist_oof.parquet'),keys);z.write_parquet(ROOT/'exposure_joint_oof.parquet')
    keys+=['window'];z=fuse(pl.read_parquet(ROOT/'exposure_cat_window_predictions.parquet'),pl.read_parquet(ROOT/'exposure_hist_window_predictions.parquet'),keys);z.write_parquet(ROOT/'exposure_joint_window_predictions.parquet')
    ix=pl.read_parquet('artifacts/evidence_session4/hand_index.parquet').select('pair_id','hand_id','evidence','fold',C('behavior_family').alias('family'),'table_id');r29=pl.read_parquet(ROOT/'window_r29_predictions.parquet');rows=[]
    for mode in ['exposure_cat','exposure_hist','exposure_joint']:
        d=pl.read_parquet(ROOT/f'{mode}_window_predictions.parquet').join(r29,on=keys,validate='1:1').join(ix,on=['pair_id','hand_id'],validate='m:1').with_columns((.5*C('score')+.5*C('r29')).alias('half'),(.25*C('score')+.75*C('r29')).alias('quarter'))
        for (w,pid),g in d.group_by('window','pair_id'):
            truth=set(g.filter(C('evidence')==1)['hand_id']);den=min(5,len(truth))
            if not den:continue
            row={'mode':mode,'window':w,'pair_id':pid,'table_id':g['table_id'][0],'fold':g['fold'][0],'family':g['family'][0]}
            for name in ['r29','score','half','quarter']:
                hand=g.sort([name,'hand_id'],descending=[True,False])['hand_id'].to_list()[:5];y=np.array([h in truth for h in hand]);row[name]=float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/den)
            rows.append(row)
    r=pl.DataFrame(rows);r.write_csv(ROOT/'exposure_windows.csv');report={'caveat':'Partial-context windows, original capped truth truncated; not fresh evidence lists. All rows are pool-held-out.','windows':r.group_by('mode','window').agg(pl.len(),C('r29','score','half','quarter').mean()).sort('mode','window').to_dicts(),'families':r.group_by('mode','window','family').agg(C('r29','score','half','quarter').mean()).to_dicts(),'folds':r.group_by('mode','window','fold').agg(C('r29','score','half','quarter').mean()).to_dicts()};(ROOT/'exposure_windows.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['windows'],indent=2))
if __name__=='__main__':main()
