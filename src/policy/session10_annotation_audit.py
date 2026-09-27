import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
import polars as pl
from session8_data import hand_data,reference
from session10_list_learning import ROOT
C=pl.col
def main():
    d=hand_data().join(reference().select('pair_id','hand_id','r29'),on=['pair_id','hand_id']).join(pl.read_parquet('artifacts/evidence_session7/hist_events_oof.parquet').select('pair_id','hand_id','primary','secondary'),on=['pair_id','hand_id']);p=d.group_by('pair_id').agg(C('evidence').sum().alias('truth_count'),pl.len().alias('hands'),C('time').filter(C('evidence')==1).max().alias('last_truth'),(C('primary')+C('secondary')).sum().alias('event_mass'),C('behavior_family').first().alias('family'));rows=[]
    for (pid,),g in d.group_by('pair_id'):
        g=g.sort(['r29','hand_id'],descending=[True,False]).with_row_index('rank',offset=1);truth=g.filter(C('evidence')==1);fp=g.filter((C('rank')<=5)&(C('evidence')==0));fn=g.filter((C('rank')>5)&(C('evidence')==1));rows.append({'pair_id':pid,'family':g['behavior_family'][0],'truth_count':len(truth),'false_selected':len(fp),'missed_truth':len(fn),'false_after_last_truth':fp.filter(C('time')>truth['time'].max()).height,'false_before_first_truth':fp.filter(C('time')<truth['time'].min()).height})
    r=pl.DataFrame(rows);r.write_csv(ROOT/'cutoff_population.csv');p.write_csv(ROOT/'short_list_pairs.csv');report={'short_lists':p.group_by('truth_count').agg(pl.len(),C('hands').mean(),C('event_mass').mean(),C('last_truth').mean()).sort('truth_count').to_dicts(),'short_list_families':p.group_by('truth_count','family').len().sort('truth_count','family').to_dicts(),'cutoff_errors':r.group_by('family').agg(C('false_selected','missed_truth','false_after_last_truth','false_before_first_truth').sum()).to_dicts(),'interpretation':'Short lists have lower exposure and lower OOF event mass; this does not support an independent random-cap explanation. No cap-change model was trained. A nonlisted hand is not necessarily a negative for latent manipulation.'};(ROOT/'annotation_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
