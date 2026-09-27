\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path('artifacts/pair_session68_exposure_audit');C=pl.col
def main():
 ROOT.mkdir(exist_ok=True);parts=[]
 for path in sorted(Path('artifacts/policy/pair_features').glob('T*.parquet')):parts.append(pl.read_parquet(path,columns=['pair_id','table_id','phase','policy_n_hands']))
 allpairs=pl.concat(parts);labels=pl.read_csv('data/development_labels.csv').select('pair_id','label');dev=allpairs.filter(C('phase')=='development').join(labels,on='pair_id',how='left',validate='1:1');assert dev.filter(C('label').is_not_null()).height==len(labels);groups=dev.with_columns(pl.when(C('label')==1).then(pl.lit('confirmed_target')).when(C('label')==0).then(pl.lit('confirmed_negative')).otherwise(pl.lit('unknown')).alias('group'));summary=groups.group_by('group').agg(pl.len().alias('pairs'),C('policy_n_hands').min().alias('min'),*[C('policy_n_hands').quantile(q).alias('q'+str(q)) for q in [.1,.25,.5,.75,.9,.99]],C('policy_n_hands').max().alias('max')).to_dicts();folds=json.load(open('artifacts/policy/table_folds.json'));known=dev.filter(C('label').is_not_null()).sort('pair_id').with_columns(pl.Series('fold',[folds[t] for t in dev.filter(C('label').is_not_null()).sort('pair_id')['table_id']]));x=np.log1p(known.select('policy_n_hands').to_numpy());y=known['label'].to_numpy();fv=known['fold'].to_numpy();p=np.zeros(len(known))
 for f in range(4):
  tr=fv!=f;va=fv==f;m=CatBoostClassifier(iterations=200,depth=3,learning_rate=.04,l2_leaf_reg=10,random_seed=6800+f,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[tr],y[tr]);m.save_model(str(ROOT/f'exposure_only_fold{f}.cbm'));p[va]=m.predict_proba(x[va],thread_count=2)[:,1];r=CatBoostClassifier();r.load_model(str(ROOT/f'exposure_only_fold{f}.cbm'));np.testing.assert_array_equal(p[va],r.predict_proba(x[va],thread_count=2)[:,1])
 known.with_columns(pl.Series('exposure_only_oof',p)).write_parquet(ROOT/'known_oof.parquet');bins=groups.with_columns(C('policy_n_hands').cut([5,10,20,40,60,80,120,200]).cast(pl.String).alias('exposure_bin')).group_by('exposure_bin','group').len().sort('exposure_bin','group');bins.write_csv(ROOT/'population_exposure_bins.csv');registry=pl.read_csv('artifacts/candidate_r33/submission.csv').select('pair_id','risk_score','predicted_behavior');ev=allpairs.filter(C('phase')=='evaluation').join(registry,on='pair_id',validate='1:1');assert len(ev)==len(registry);evalsummary=ev.with_columns(pl.when(C('risk_score')>=.5).then(pl.lit('risk_ge05')).when(C('risk_score')>=.05).then(pl.lit('risk_005_05')).otherwise(pl.lit('risk_lt005')).alias('group')).group_by('group').agg(pl.len(),*[C('policy_n_hands').quantile(q).alias('q'+str(q)) for q in [.1,.5,.9]]).to_dicts();out={'method':__doc__,'development_distribution':summary,'exposure_only_known_label_AP':average_precision_score(y,p),'exposure_only_known_label_AUC':roc_auc_score(y,p),'exposure_only_negative_weight50_AP':average_precision_score(y,p,sample_weight=np.where(y==1,1,50)),'raw_hand_count_known_AP':average_precision_score(y,known['policy_n_hands'].to_numpy()),'evaluation_distribution':evalsummary,'saved_models_replayed':4,'prediction_error':0,'unknowns_used_as_negative_targets':0};(ROOT/'report.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
