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
C=pl.col;ROOT=Path('artifacts/evidence_session45_pair_mil');PREV=Path('artifacts/evidence_session43_pair_interactions');NAMES=['none','directed_transfer','soft_play','coordinated_isolation']
def main():
 ROOT.mkdir(exist_ok=True);d=pl.read_parquet(PREV/'window_features.parquet');cfg=json.load(open(PREV/'config.json'));cols=cfg['base_columns'];x=d.select(cols).to_numpy();y=np.array([NAMES.index(f) for f in d['behavior_family']]);fv=d['fold'].to_numpy();wt=np.where(d['window'].to_numpy()=='full',1,.5);tables=d['table_id'].to_numpy();half={}
 for f in range(4):
  ts=sorted(set(tables[fv==f]));order=np.random.default_rng(4510+f).permutation(len(ts));half.update({ts[k]:i%2 for i,k in enumerate(order)})
 (ROOT/'table_half_split.json').write_text(json.dumps(half,indent=2));hv=np.array([half[t] for t in tables]);start=time.time();(ROOT/'nested_config.json').write_text(json.dumps({'method':__doc__,'columns':cols,'schedule':'same600depth5 MultiClass as43, seed991+inner_parent','inner_blocks':6,'source':str(PREV/'window_features.parquet')},indent=2))
 for outer in range(4):
  pieces=[]
  for parent in range(4):
   if parent==outer:continue
   for h in range(2):
    stem=f'outer{outer}_parent{parent}_half{h}';path=ROOT/(stem+'.parquet');va=(fv==parent)&(hv==h);tr=(fv!=outer)&~va
    if not path.exists():
     assert set(tables[tr]).isdisjoint(set(tables[va]));assert not (tr&(fv==outer)).any();m=CatBoostClassifier(iterations=600,depth=5,learning_rate=.04,loss_function='MultiClass',l2_leaf_reg=10,thread_count=2,random_seed=991+parent,verbose=False,allow_writing_files=False);m.fit(x[tr],y[tr],sample_weight=wt[tr]);m.save_model(str(ROOT/(stem+'.cbm')));risk=1-m.predict_proba(x[va],thread_count=2)[:,0];d.filter(pl.Series(va)).select('pair_id','window').with_columns(pl.Series('risk',risk)).write_parquet(path);path.with_suffix('.json').write_text(json.dumps({'outer':outer,'parent':parent,'half':h,'training_rows':int(tr.sum()),'prediction_rows':int(va.sum()),'training_tables':sorted(set(tables[tr])),'prediction_tables':sorted(set(tables[va])),'outer_training_overlap':0},indent=2));print('nested pair',outer,parent,h,round(time.time()-start,1),flush=True)
    pieces.append(pl.read_parquet(path))
  val=pl.read_parquet(PREV/'oof.parquet').filter(C('fold')==outer).select('pair_id','window',C('residual_only').alias('risk'));out=d.select('pair_id','window','table_id','fold').join(pl.concat(pieces+[val]),on=['pair_id','window'],validate='1:1',maintain_order='left');assert out['risk'].null_count()==0;out.write_parquet(ROOT/f'nested_outer{outer}.parquet')
 print('nested pair complete',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
