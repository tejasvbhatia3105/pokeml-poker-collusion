import os,json,sys
os.environ.setdefault('POLARS_MAX_THREADS','3')
import numpy as np,polars as pl,torch
from catboost import CatBoostClassifier
from scipy.special import logit
from session45_pair_mil import ROOT,PREV,pack,Model,CONFIG
C=pl.col
def inputs(d):
 cfg=json.load(open(ROOT/'nested_config.json'));x=d.select(cfg['columns']).to_numpy();fv=d['fold'].to_numpy();tables=d['table_id'].to_numpy();half=json.load(open(ROOT/'table_half_split.json'));hv=np.array([half[t] for t in tables]);err=0.;models=0;valerrs=[]
 for outer in range(4):
  for parent in range(4):
   if parent==outer:continue
   for h in range(2):
    stem=f'outer{outer}_parent{parent}_half{h}';va=(fv==parent)&(hv==h);tr=(fv!=outer)&~va;assert not (tr&va).any();audit=json.load(open(ROOT/(stem+'.json')));assert audit['training_tables']==sorted(set(tables[tr]));assert audit['prediction_tables']==sorted(set(tables[va]));assert set(tables[tr]).isdisjoint(set(tables[fv==outer]));m=CatBoostClassifier();m.load_model(str(ROOT/(stem+'.cbm')));actual=1-m.predict_proba(x[va],thread_count=2)[:,0];q=d.filter(pl.Series(va)).select('pair_id','window').join(pl.read_parquet(ROOT/(stem+'.parquet')),on=['pair_id','window'],validate='1:1',maintain_order='left');err=max(err,float(abs(actual-q['risk'].to_numpy()).max()));models+=1
  q=pl.read_parquet(ROOT/f'nested_outer{outer}.parquet').filter(C('fold')==outer).join(pl.read_parquet(PREV/'oof.parquet').select('pair_id','window','residual_only'),on=['pair_id','window'],validate='1:1');valerrs.append(float((q['risk']-q['residual_only']).abs().max()))
 assert err==0 and max(valerrs)==0;out={'models_replayed':models,'inner_prediction_error':err,'outer_reference_errors':valerrs,'pool_target_separation_passed':True,'scope':'supervised pair targets nested; legacy gameplay features frozen'};(ROOT/'input_verification.json').write_text(json.dumps(out,indent=2));return out
def main():
 d=pl.read_parquet(PREV/'window_features.parquet');inp=inputs(d)
 if '--inputs-only' in sys.argv:print(json.dumps(inp,indent=2));return
 X,A,M,bc,ac=pack(d);fv=d['fold'].to_numpy();saved=pl.read_parquet(ROOT/'oof.parquet');err=0.;permerr=0.;models=0
 for f in range(4):
  q=d.select('pair_id','window').join(pl.read_parquet(ROOT/f'nested_outer{f}.parquet'),on=['pair_id','window'],validate='1:1',maintain_order='left');prior=logit(q['risk'].to_numpy().clip(1e-10,1-1e-10));tr=fv!=f;va=fv==f;Z=np.column_stack([X,prior]).astype(np.float32);mu=Z[tr].mean(0);sd=np.maximum(.05,Z[tr].std(0));am=A[tr][M[tr]].mean(0);astd=np.maximum(.05,A[tr][M[tr]].std(0));xx=torch.tensor(np.clip((Z-mu)/sd,-8,8));aa=torch.tensor(np.clip((A-am)/astd,-8,8));mm=torch.tensor(M);pp=torch.tensor(prior);valid=np.flatnonzero(va)
  for kind in CONFIG['arms']:
   ensemble=[]
   for seed in CONFIG['seeds']:
    st=torch.load(ROOT/f'{kind}_fold{f}_seed{seed}.pt',weights_only=False);assert st['config']==CONFIG and st['base_columns']==bc and st['action_columns']==ac
    for key,v in [('mu',mu),('sd',sd),('action_mu',am),('action_sd',astd)]:np.testing.assert_array_equal(st[key],v)
    m=Model(Z.shape[1],A.shape[2],kind);m.load_state_dict(st['state_dict']);m.eval();chunks=[]
    with torch.no_grad():
     ix=valid[:8];orig=m(xx[ix],aa[ix],mm[ix]);rev=m(xx[ix],aa[ix].flip(1),mm[ix].flip(1));permerr=max(permerr,float(abs(orig-rev).max()))
     for j in range(0,len(valid),128):
      ix=valid[j:j+128];chunks.append(torch.sigmoid(pp[ix]+m(xx[ix],aa[ix],mm[ix])).numpy())
    ensemble.append(np.concatenate(chunks));models+=1
   pred=np.mean(ensemble,0);expected=d.filter(pl.Series(va)).select('pair_id','window').join(saved,on=['pair_id','window'],validate='1:1',maintain_order='left')[kind].to_numpy();err=max(err,float(abs(pred-expected).max()))
 assert err<1e-12 and permerr<2e-6;report={'inputs':inp,'correction_models_replayed':models,'probability_error':err,'bag_permutation_error':permerr,'train_only_normalization_exact':True,'architecture_checks':json.load(open(ROOT/'architecture_checks.json'))};(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
