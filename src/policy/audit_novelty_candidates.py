import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl,json
root=Path('artifacts/policy');dest=root/'open_set_floor';C=pl.col
base=pl.read_csv(root/'residual_eval.csv').select('pair_id',(1-C('none')).alias('base_risk'))
d=pl.scan_parquet(str(root/'pair_features/*.parquet')).filter(C('phase')=='evaluation').join(base.lazy(),on='pair_id').collect();pred=[]
for f in range(4):
 z=np.load(dest/f'calibrator_fold{f}.npz');X=d.select(z['columns'].tolist()).to_numpy();s=np.sort(np.log1p(np.abs((X-z['med'])/z['scale'])),axis=1)[:,-3:].mean(1);logit=s*z['coef'].item()+z['intercept'].item();pred.append(1/(1+np.exp(-logit)))
d=d.select('pair_id','table_id','policy_n_hands','base_risk').with_columns(pl.Series('novelty_probability',np.mean(pred,axis=0)));d.write_csv(dest/'eval_novelty.csv');r=[]
for alpha in [.25,.5]:
 z=d.with_columns(pl.max_horizontal(C('base_risk'),alpha*C('novelty_probability')).alias('risk'));raised=z.filter(alpha*C('novelty_probability')>C('base_risk'));r.append({'floor':alpha,'raised_pairs':len(raised),'raised_to_above_0.1':raised.filter(C('risk')>.1).height,'raised_to_above_0.25':raised.filter(C('risk')>.25).height,'novelty_candidates_base_below_0.1_probability_above_0.9':z.filter((C('base_risk')<.1)&(C('novelty_probability')>.9)).height})
print(r);(dest/'eval_audit.json').write_text(json.dumps(r,indent=2));print(d.filter(C('base_risk')<.1).sort('novelty_probability',descending=True).head(15))
