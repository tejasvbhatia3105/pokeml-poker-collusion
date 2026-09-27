import json
import numpy as np,polars as pl,joblib
import session88_crop_audit as old
from session89_crop_list_training import ROOT
C=pl.col
def main():
 d=old.data();pred=pl.read_parquet(ROOT/'oof.parquet');nn=old.correction_models('conditional_family');tree=[joblib.load(f'artifacts/evidence_session62_grounded_list_boost/list_boost_full_fold{f}.joblib') for f in range(4)];rows=[];parts=[];full_control=pl.read_parquet('artifacts/evidence_session62_grounded_list_boost/conditional_boost_oof.parquet').select('pair_id','hand_id','full');err=0
 for (pid,),g in d.group_by('pair_id'):
  g=g.sort('time','hand_id');ti=np.rint(g['time'].to_numpy()*5000).astype(int);f=g['fold'][0]
  for window,(lo,hi) in old.WINDOWS.items():
   z=g.filter(pl.Series((ti>=round(lo*5000))&(ti<round(hi*5000))))
   if not len(z) or z['evidence'].sum()!=g['evidence'].sum():continue
   q=z.join(pred.filter(C('window')==window),on=['pair_id','hand_id'],validate='1:1',maintain_order='left',suffix='_pred');assert q['tree_augmented'].null_count()==0
   if window=='full':
    control=q.select('pair_id','hand_id','tree_control').join(full_control,on=['pair_id','hand_id'],validate='1:1');err=max(err,float(abs(control['tree_control']-control['full']).max()));base=q['r33_full'].to_numpy()
   else:base=old.score(q.with_columns(((C('time')-lo)/(hi-lo)).alias('relative_time')),f,nn,tree,refresh_legacy=True)
   candidate=base+.5*(q['tree_augmented'].to_numpy()-q['tree_control'].to_numpy())
   if g['risk_score'][0]<.05:base=candidate=q['conditional_family'].to_numpy()
   q=q.with_columns(pl.Series('baseline',base),pl.Series('candidate',candidate));rows.append({'pair_id':pid,'table_id':g['table_id'][0],'family':g['behavior_family'][0],'fold':f,'window':window,'baseline':old.ap(q,'baseline'),'candidate':old.ap(q,'candidate')});parts.append(q.select('pair_id','hand_id','baseline','candidate').with_columns(pl.lit(window).alias('window')))
 assert err==0;pl.concat(parts).write_parquet(ROOT/'combined_oof.parquet');r=pl.DataFrame(rows);r.write_parquet(ROOT/'pair_comparison.parquet');report={}
 for (window,),q in r.group_by('window'):
  pool=q.group_by('table_id').agg(C('baseline','candidate').sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(8912).integers(0,len(pool),(5000,len(pool)));delta=pool['candidate'].to_numpy()-pool['baseline'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);report[window]={'pairs':len(q),'baseline_MAP':q['baseline'].mean(),'candidate_MAP':q['candidate'].mean(),'gain':q['candidate'].mean()-q['baseline'].mean(),'CI95':np.quantile(boot,[.025,.975]).tolist(),'folds':q.group_by('fold').agg(C('baseline','candidate').mean()).sort('fold').to_dicts(),'families':q.group_by('family').agg(pl.len(),C('baseline','candidate').mean()).to_dicts(),'better':int((q['candidate']>q['baseline']+1e-12).sum()),'worse':int((q['candidate']<q['baseline']-1e-12).sum())}
 (ROOT/'comparison.json').write_text(json.dumps({'full_control_tree_replay_error':err,'results':report},indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
