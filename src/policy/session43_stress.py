import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
import session43_pair_interactions as task
ROOT=task.ROOT
def main():
 task.W={'w0_1500':(0,1500),'w1500_3000':(1500,3000),'w500_2500':(500,2500),'w750_2250':(750,2250)};actions=pl.read_parquet(ROOT/'action_features.parquet');d,bc,jc=task.frames(actions);cfg=json.load(open(ROOT/'config.json'));assert bc==cfg['base_columns'] and jc==cfg['joint_columns'];fv=d['fold'].to_numpy();pred={k:np.zeros(len(d)) for k in ['residual_only','paired_context']}
 for kind in pred:
  x=d.select(bc if kind=='residual_only' else bc+jc).to_numpy()
  for f in range(4):
   va=fv==f;m=CatBoostClassifier();m.load_model(str(ROOT/f'{kind}_fold{f}.cbm'));pred[kind][va]=1-m.predict_proba(x[va],thread_count=2)[:,0]
 out=d.select('pair_id','table_id','fold','window','label','behavior_family','n_ev_in').with_columns(*[pl.Series(k,v) for k,v in pred.items()]);out.write_parquet(ROOT/'stress_oof.parquet');report=[]
 for (window,),q in out.group_by('window'):
  for kind in pred:
   label=q['label'].to_numpy();score=q[kind].to_numpy();report.append({'window':window,'model':kind,'positive_pairs':int(label.sum()),'labelled_AP':average_precision_score(label,score),'labelled_AP_negative_weight50':average_precision_score(label,score,sample_weight=np.where(label==1,1,50)),'positive_below05':int(((label==1)&(score<.5)).sum()),'confirmed_negative_above05':int(((label==0)&(score>.5)).sum())})
 (ROOT/'stress_comparison.json').write_text(json.dumps({'windows_not_used_for_training':task.W,'results':report,'limitation':'these are additional history lengths, not new independent labelled pools or unseen competition labels'},indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
