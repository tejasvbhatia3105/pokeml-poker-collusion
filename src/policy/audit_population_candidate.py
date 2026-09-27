import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import importlib.util,json,numpy as np,pandas as pd,polars as pl
from sklearn.metrics import average_precision_score as ap
root=Path('artifacts/policy');dest=Path('artifacts/v5');dest.mkdir(exist_ok=True)
d=pd.read_csv(root/'population_rank_fusion/oof.csv');w=.95
d['fused']=np.maximum(d.base_rarity,w*d.simple_rarity)
tables=pl.scan_parquet(str(root/'pair_features/*.parquet')).filter(pl.col('phase')=='development').select('pair_id','table_id').collect().to_pandas()
d=d.merge(tables,on='pair_id',validate='many_to_one');reports=[]
for family,z in d.groupby('withheld_family'):
    y=z.truth.to_numpy();keep=np.ones(len(z),bool) if family=='all_known' else (y==0)|(z.behavior.to_numpy()==family);sw=np.where(y[keep]>0,1,50)
    reports.append(dict(withheld_family=family,baseline_weighted_AP=ap(y[keep],z.base.to_numpy()[keep],sample_weight=sw),candidate_weighted_AP=ap(y[keep],z.fused.to_numpy()[keep],sample_weight=sw)))
groups=np.sort(d.table_id.unique());index={g:i for i,g in enumerate(groups)};rng=np.random.default_rng(88311);samples=[]
tests=[]
for family,z in d[d.withheld_family!='all_known'].groupby('withheld_family'):
    z=z[(z.truth==0)|(z.behavior==family)];tests.append((z.truth.to_numpy(),z.base.to_numpy(),z.fused.to_numpy(),np.array([index[g] for g in z.table_id])))
for _ in range(500):
    counts=np.bincount(rng.integers(len(groups),size=len(groups)),minlength=len(groups));diff=[]
    for y,base,new,gi in tests:
        weight=np.where(y>0,1,50)*counts[gi];diff.append(ap(y,new,sample_weight=weight)-ap(y,base,sample_weight=weight))
    samples.append(np.mean(diff))

                                                                                 
spec=importlib.util.spec_from_file_location('reference','artifacts/reference_metric/reference_metric.py');metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
p=pd.read_csv(root/'residual_oof.csv').sort_values('pair_id');names=['none','directed_transfer','soft_play','coordinated_isolation'];z=d[d.withheld_family=='all_known'].set_index('pair_id').loc[p.pair_id];truth=pd.read_csv('data/development_labels.csv').set_index('pair_id').loc[p.pair_id].reset_index().rename(columns={'label':'risk_score','behavior_family':'predicted_behavior'})
sub=pd.DataFrame(dict(pair_id=p.pair_id,risk_score=-np.expm1(-z.fused.to_numpy()),predicted_behavior=np.where(1-p.none>=.01,p[names[1:]].idxmax(axis=1),'none')))
ev=pd.read_csv('data/development_evidence.csv').sort_values('evidence_rank').groupby('pair_id').hand_id.apply(list).to_dict();chosen=dict(pl.read_parquet(root/'combined_selected_evidence.parquet').iter_rows())
for j in range(5):
    c=f'evidence_hand_{j+1}';truth[c]=[ev.get(pid,[])[j] if len(ev.get(pid,[]))>j else 'NO_EVIDENCE' for pid in truth.pair_id];sub[c]=[chosen.get(pid,[])[j] if len(chosen.get(pid,[]))>j else 'NO_EVIDENCE' for pid in sub.pair_id]
official=metric.score(truth,sub,'pair_id')
result=dict(family_tests=reports,mean_withheld_baseline=float(np.mean([r['baseline_weighted_AP'] for r in reports if r['withheld_family']!='all_known'])),mean_withheld_candidate=float(np.mean([r['candidate_weighted_AP'] for r in reports if r['withheld_family']!='all_known'])),bootstrap_mean_withheld_gain_95_interval=np.quantile(samples,[.025,.975]).tolist(),bootstrap_replicates=500,known_development_combined_score=official,v4_known_development_combined_score=json.load(open(root/'combined_validation.json'))['combined_score'],limitations=['Exploratory model and weight selection reused these development labels.','Bootstrap resamples pools with fixed predictions; it excludes training and model-selection uncertainty.','Withholding public families is a proxy, not a measurement of the actual undisclosed family.','Development calibration uses each held-out pool set; inference uses the entire evaluation population.','Kaggle performance is unmeasured.'])
(dest/'development_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
