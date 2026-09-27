import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path('artifacts/evidence_session12/pair_events_complete');C=pl.col

def main():
 d=pl.read_parquet(ROOT/'pair_event_scores.parquet');known=d.filter(C('label')>=0);rows=[]
 for score in ['risk_score','joint_event_mass','prob_at_least3','base_top5_sum','joint_top5_sum']:
  rows.append({'score':score,'labelled_AP':average_precision_score(known['label'],known[score]),'labelled_AUC':roc_auc_score(known['label'],known[score]),'positive_median':known.filter(C('label')==1)[score].median(),'negative_median':known.filter(C('label')==0)[score].median()})
 anomalies=d.filter((C('risk_score')>.8)&(C('prob_at_least3')<.05)).sort('risk_score',descending=True);anomalies.write_csv(ROOT/'high_risk_low_events.csv');pos=known.filter(C('label')==1).sort('prob_at_least3').select('pair_id','risk_score','family','shared_hands','joint_event_mass','prob_at_least3','n_truth');pos.head(30).write_csv(ROOT/'lowest_positive_events.csv');tops=[]
 for k in [100,500,1000,2000]:
  z=d.sort('risk_score',descending=True).head(k);tops.append({'top':k,'known_positive':z.filter(C('label')==1).height,'known_negative':z.filter(C('label')==0).height,'unknown':z.filter(C('label')==-1).height,'low_event_tail':z.filter(C('prob_at_least3')<.05).height})
 report={'scores':rows,'top_ranked':tops,'high_risk_low_event_count':len(anomalies),'high_risk_low_event_labels':anomalies.group_by('label').len().to_dicts(),'positive_count_below_tail_005':pos.filter(C('prob_at_least3')<.05).height,'limitations':['Unknown pairs cannot be counted as false positives; unobserved fourth-family signals may escape known-family event heads.','Full-history labelled AP is not an evaluation-population estimate.','No new risk rule fitted; using pooled OOF features to train another outer-fold model would leak teacher-fold information unless rebuilt strictly nested.']};(ROOT/'descriptive_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
