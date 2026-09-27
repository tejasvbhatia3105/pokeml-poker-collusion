\
\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data
from session6_priority import training_targets,inclusion
C=pl.col;ROOT=Path('artifacts/evidence_session36_nested_cascade');OLD=Path('artifacts/evidence_session10/nested6')

def main():
 ROOT.mkdir(exist_ok=True);d=hand_data();cfg=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'));x=d.select(cfg['event']).to_numpy();xt=d.select(cfg['type']).to_numpy();fv=d['fold'].to_numpy();fam=d['behavior_family'].to_numpy();y=d['evidence'].to_numpy();sub=d['subtype'].to_numpy();timev=d['time'].to_numpy();ranks=d['evidence_rank'].fill_null(0).to_numpy();groups=[g['row'].to_numpy() for _,g in d.group_by('pair_id',maintain_order=True)];half=json.load(open(OLD/'table_half_split.json'));hv=np.array([half[t] for t in d['table_id']]);start=time.time();(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'inner_splits':'same six disjoint blocks as session10 nested6','type_model':'Cat200 depth3 original parent seed','secondary_model':'Cat400 depth5 lr.035 L2 8 original parent/head seed','outer_predictions':'exact frozen session31 cascade','old_teachers':'same original cached base, primary and HGB; only secondary and derived scores replaced'},indent=2))
 for outer in range(4):
  pieces=[]
  for parent in range(4):
   if parent==outer:continue
   for h in range(2):
    path=ROOT/f'conditional_outer{outer}_parent{parent}_half{h}.parquet';held=(fv==parent)&(hv==h);eligible=(fv!=outer)&~held
    if not path.exists():
     pp=np.zeros(len(d));audit=[]
     for family in ['directed_transfer','soft_play','coordinated_isolation']:
      tr=eligible&(fam==family);va=held&(fam==family);known=tr&(sub>0);typ=CatBoostClassifier(iterations=200,depth=3,learning_rate=.04,l2_leaf_reg=10,random_seed=6210+parent,thread_count=2,verbose=False,allow_writing_files=False);typ.fit(xt[known],(sub[known]==1).astype(int));tp=typ.predict_proba(xt,thread_count=2)[:,1];a,b,ea,eb=training_targets(y,sub,tp,tr,timev,groups,ranks);e=ea&eb&~a;assert not (e&((fv==outer)|held)).any();assert (b&e).sum()==(b&eb).sum();m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6312+11*parent,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[e],b[e]);pp[va]=m.predict_proba(x[va],thread_count=2)[:,1];stem=f'outer{outer}_parent{parent}_half{h}_{family}';m.save_model(str(ROOT/(stem+'.cbm')));typ.save_model(str(ROOT/(stem+'_type.cbm')));audit.append({'outer':outer,'parent':parent,'half':h,'family':family,'training_hands':int(e.sum()),'positive_hands':int(b[e].sum()),'training_tables':sorted(set(d['table_id'].to_numpy()[e])),'prediction_tables':sorted(set(d['table_id'].to_numpy()[va])),'outer_overlap':0,'inner_overlap':0})
     d.filter(pl.Series(held)).select('pair_id','hand_id').with_columns(pl.Series('conditional_secondary',pp[held])).write_parquet(path);path.with_suffix('.json').write_text(json.dumps(audit,indent=2));print('nested cascade',outer,parent,h,round(time.time()-start,1),flush=True)
    pieces.append(pl.read_parquet(path))
  val=pl.read_parquet('artifacts/evidence_session31_conditional_events/event_oof.parquet').filter(C('fold')==outer).select('pair_id','hand_id',C('cat_conditional_secondary').alias('conditional_secondary'));raw=pl.read_parquet(OLD/f'nested_outer{outer}.parquet').join(pl.concat(pieces+[val]),on=['pair_id','hand_id'],validate='1:1').with_columns(((1-C('cat_primary'))*C('conditional_secondary')).alias('cat_secondary'));parts=[]
  for _,g in raw.group_by('pair_id'):
   g=g.sort('time','hand_id');ca=g.select('cat_primary','cat_secondary').to_numpy();hp=g.select('hist_primary','hist_secondary').to_numpy();hp=hp/np.maximum(1,hp.sum(1))[:,None];ci=inclusion(*ca.T);ji=inclusion(*((ca+hp)/2).T);parts.append(g.with_columns(pl.Series('cat_inclusion',ci),pl.Series('joint_inclusion',ji),pl.Series('r29',.25*g['base'].to_numpy()+.25*ci+.5*ji)).drop('conditional_secondary'))
  pl.concat(parts).write_parquet(ROOT/f'nested_outer{outer}.parquet')
 print('nested cascade complete',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
