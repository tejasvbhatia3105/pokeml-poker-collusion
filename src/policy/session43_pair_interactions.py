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
from sklearn.metrics import average_precision_score
from session37_bet_fold import paired_features
C=pl.col;ROOT=Path('artifacts/evidence_session43_pair_interactions');P=Path('artifacts/policy');W={'full':(0,3000),'first_2000':(0,2000),'last_2000':(1000,3000)};NAMES=['none','directed_transfer','soft_play','coordinated_isolation']

def build_actions(save=True):
 labs=pl.read_csv('data/development_labels.csv');pc=json.load(open(P/'feature_columns.json'));cols=[c for c in pc if 'style_' not in c];parts=[];start=time.time();audits=[]
 for i,path in enumerate(sorted((P/'pair_features').glob('T*.parquet'))):
  table=path.stem;query=pl.read_parquet(path).filter(C('phase')=='development').select('pair_id','player_1','player_2').join(labs.select('pair_id'),on='pair_id',validate='1:1');raw=pl.read_parquet(P/'actions'/path.name).filter((C('phase')=='development')&(C('action_class')==0));mapping=pl.concat([query.select('pair_id',C('player_1').alias('player_id'),C('player_2').alias('partner'),pl.lit(0).alias('actor')),query.select('pair_id',C('player_2').alias('player_id'),C('player_1').alias('partner'),pl.lit(1).alias('actor'))]);a=raw.join(mapping,on='player_id').filter((C('last_aggressor')==C('partner')).fill_null(False)).sort('pair_id','hand_id','action_no').with_row_index('action_row')
  if not len(a):continue
  assert a.select('pair_id','hand_id').n_unique()==len(a);d=a.select('pair_id','hand_id').with_columns(pl.lit(table).alias('table_id'));extra,alignment=paired_features(d,a);ec=[c for c in extra.columns if c!='action_row' and 'style_' not in c];assert not any('style_' in c for c in cols+ec);extra=extra.select('action_row',*ec);z=a.select('action_row','pair_id','hand_id','time_index',*[C(c).alias('response_'+c) for c in cols]).join(extra,on='action_row',validate='1:1').drop('action_row').with_columns(pl.lit(table).alias('table_id'));parts.append(z);audits.append({'table':table,'labelled_pairs':len(query),'paired_fold_hands':len(a),'last_aggressor_and_bet_alignment_verified':True})
  if i%70==0:print('pair interaction features',i,round(time.time()-start,1),flush=True)
 out=pl.concat(parts)
 if save:out.write_parquet(ROOT/'action_features.parquet');(ROOT/'feature_audit.json').write_text(json.dumps(audits,indent=2))
 return out

def frames(actions):
 labs=pl.read_csv('data/development_labels.csv');folds=json.load(open(P/'table_folds.json'));basecols=json.load(open(P/'residual_columns.json'));newcols=[c for c in actions.columns if c not in ['pair_id','hand_id','time_index','table_id']];truth=pl.read_csv('data/development_evidence.csv');times=[]
 for path in sorted((P/'actions').glob('T*.parquet')):
  z=pl.read_parquet(path,columns=['hand_id','time_index']).unique().join(truth.select('hand_id').unique(),on='hand_id',how='semi');times.append(z)
 ev=truth.join(pl.concat(times),on='hand_id',validate='m:1');parts=[]
 for window,(lo,hi) in W.items():
  folder=P/'pair_features' if window=='full' else P/'full_window_stress'/window/'pair_features';base=pl.concat([pl.read_parquet(path).filter(C('phase')=='development').join(labs,on=['pair_id','player_1','player_2'],validate='1:1') for path in sorted(folder.glob('T*.parquet'))]);base=base.with_columns(pl.Series('fold',[folds[t] for t in base['table_id']]));local=actions.filter((C('time_index')>=lo)&(C('time_index')<hi));expr=[pl.len().alias('joint_fold_count')]
  for c in newcols:
   expr.extend([C(c).mean().alias('joint_'+c+'_mean'),C(c).std(ddof=0).alias('joint_'+c+'_std'),C(c).min().alias('joint_'+c+'_min'),C(c).max().alias('joint_'+c+'_max')])
  agg=local.group_by('pair_id').agg(expr);nev=ev.filter((C('time_index')>=lo)&(C('time_index')<hi)).group_by('pair_id').len().rename({'len':'n_ev_in'});base=base.join(agg,on='pair_id',how='left',validate='1:1').join(nev,on='pair_id',how='left',validate='1:1').with_columns(C('n_ev_in').fill_null(0),C('joint_fold_count').fill_null(0)).with_columns((C('joint_fold_count')/C('policy_n_hands').clip(1)).alias('joint_fold_fraction'));jointcols=[c for c in base.columns if c.startswith('joint_')];base=base.with_columns(C(jointcols).fill_null(-2)).with_columns(pl.lit(window).alias('window'));base=base.filter((C('label')==0)|(C('n_ev_in')>=1));parts.append(base.select('pair_id','table_id','fold','window','label','behavior_family','n_ev_in',*basecols,*jointcols))
 return pl.concat(parts).sort('window','pair_id'),basecols,jointcols

def main():
 ROOT.mkdir(exist_ok=True);actions=pl.read_parquet(ROOT/'action_features.parquet') if (ROOT/'action_features.parquet').exists() else build_actions();d,bc,jc=frames(actions);d.write_parquet(ROOT/'window_features.parquet');cfg={'method':__doc__,'base_columns':bc,'joint_columns':jc,'windows':W,'schedule':'600 trees depth5 lr.04 L2 10, seed991+fold; full weight1, cropped weights.5','supervision':'confirmed labels only; positive crop must retain listed evidence; no pseudos or unknown negatives','evaluation':'labelled AP and labelled AP with negative weight50 are diagnostics, not population AP or actual R26 reproduction'};(ROOT/'config.json').write_text(json.dumps(cfg,indent=2));fv=d['fold'].to_numpy();y=np.array([NAMES.index(fam) for fam in d['behavior_family']]);wt=np.where(d['window'].to_numpy()=='full',1,.5);pred={k:np.zeros((len(d),4)) for k in ['residual_only','paired_context']};start=time.time()
 for f in range(4):
  tr=fv!=f;va=fv==f
  for kind in pred:
   cols=bc if kind=='residual_only' else bc+jc;x=d.select(cols).to_numpy();m=CatBoostClassifier(iterations=600,depth=5,learning_rate=.04,loss_function='MultiClass',l2_leaf_reg=10,thread_count=2,random_seed=991+f,verbose=False,allow_writing_files=False);m.fit(x[tr],y[tr],sample_weight=wt[tr]);m.save_model(str(ROOT/f'{kind}_fold{f}.cbm'));pred[kind][va]=m.predict_proba(x[va],thread_count=2);print('pair interaction train',f,kind,round(time.time()-start,1),flush=True)
 out=d.select('pair_id','table_id','fold','window','label','behavior_family','n_ev_in').with_columns(*[pl.Series(kind,1-pp[:,0]) for kind,pp in pred.items()]);out.write_parquet(ROOT/'oof.parquet');report=[]
 for (window,),q in out.group_by('window'):
  for kind in pred:
   label=q['label'].to_numpy();score=q[kind].to_numpy();negweights=np.where(label==1,1,50);report.append({'window':window,'model':kind,'labelled_pairs':len(q),'positive_pairs':int(label.sum()),'labelled_AP':average_precision_score(label,score),'labelled_AP_negative_weight50':average_precision_score(label,score,sample_weight=negweights),'positive_below05':int(((label==1)&(score<.5)).sum()),'confirmed_negative_above05':int(((label==0)&(score>.5)).sum()),'fold_weighted_AP':[average_precision_score(z['label'],z[kind],sample_weight=np.where(z['label'].to_numpy()==1,1,50)) for _,z in q.sort('fold').group_by('fold',maintain_order=True)]})
 (ROOT/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
