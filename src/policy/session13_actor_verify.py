import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier,CatBoostRegressor
from session13_actor_calibration import ROOT,P,process,check
from session8_data import hand_data
def main():
 d=hand_data();q=pl.read_parquet(ROOT/'hand_features.parquet');z=d.join(q,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');errors={}
 for role in ['lower','higher']:
  for k in range(4):
   expected=z.select(pl.sum_horizontal([f'outcome_{role}_s{s}_partner_{k}_residual' for s in range(4)])).to_numpy().ravel();err=float(abs(expected-z[f'global_{role}_partner_{k}_r'].to_numpy()).max());errors[f'{role}_{k}']=err;assert err<1e-5
 table=d['table_id'][0];players=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');query=d.filter(pl.col('table_id')==table).select('pair_id','hand_id').join(players,on='pair_id');models=[];sizes=[]
 for f in range(4):
  m=CatBoostClassifier();m.load_model(str(P/f'action_fold{f}.cbm'));models.append(m);s=CatBoostRegressor();s.load_model(str(P/f'size_fold{f}.cbm'));sizes.append(s)
 cols=json.load(open(P/'feature_columns.json'));v,audit=process(table,query.rename({'player_1':'player_2','player_2':'player_1'}),models,sizes,cols);orig=q.join(query.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi').sort('pair_id','hand_id');v=v.sort('pair_id','hand_id');err=float(abs(orig.select(pl.selectors.numeric()).to_numpy()-v.select(pl.selectors.numeric()).to_numpy()).max());assert err<1e-6
 out={'mathematical':check(),'global_vs_archived_role_residual_max_errors':errors,'endpoint_swap_max_error':err,'shared_fit_overlap':sum(a['shared_fit_overlap'] for a in json.load(open(ROOT/'fit_audit.json'))),'rows':len(q),'columns':len(q.columns)-2,'calibration_input':'Only raw actions and pair endpoint/shared-hand query; no competition label, subtype, evidence rank or teacher event target'};assert out['shared_fit_overlap']==0;(ROOT/'verification.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
