import json
import numpy as np,polars as pl,torch
from scipy.special import logit
from sklearn.metrics import average_precision_score
import session43_pair_interactions as task
from session45_pair_mil import ROOT,PREV,Model,CONFIG,pack
C=pl.col
def main():
 task.W={'w0_1500':(0,1500),'w1500_3000':(1500,3000),'w500_2500':(500,2500),'w750_2250':(750,2250)};d,bc,_=task.frames(pl.read_parquet(PREV/'action_features.parquet'));X,A,M,_,ac=pack(d);q=d.select('pair_id','window').join(pl.read_parquet(PREV/'stress_oof.parquet'),on=['pair_id','window'],validate='1:1',maintain_order='left');prior=logit(q['residual_only'].to_numpy().clip(1e-10,1-1e-10));Z=np.column_stack([X,prior]).astype(np.float32);fv=d['fold'].to_numpy();pred={k:np.zeros(len(d)) for k in CONFIG['arms']}
 for f in range(4):
  vi=np.flatnonzero(fv==f)
  for kind in CONFIG['arms']:
   ensemble=[]
   for seed in CONFIG['seeds']:
    st=torch.load(ROOT/f'{kind}_fold{f}_seed{seed}.pt',weights_only=False);assert st['base_columns']==bc and st['action_columns']==ac;m=Model(Z.shape[1],A.shape[2],kind);m.load_state_dict(st['state_dict']);m.eval();xx=torch.tensor(np.clip((Z[vi]-st['mu'])/st['sd'],-8,8));aa=torch.tensor(np.clip((A[vi]-st['action_mu'])/st['action_sd'],-8,8));mm=torch.tensor(M[vi]);pp=torch.tensor(prior[vi]);chunks=[]
    with torch.no_grad():
     for j in range(0,len(vi),128):chunks.append(torch.sigmoid(pp[j:j+128]+m(xx[j:j+128],aa[j:j+128],mm[j:j+128])).numpy())
    ensemble.append(np.concatenate(chunks))
   pred[kind][vi]=np.mean(ensemble,0)
 out=d.select('pair_id','table_id','fold','window','label','behavior_family','n_ev_in').with_columns(*[pl.Series(k,v) for k,v in pred.items()]);out.write_parquet(ROOT/'stress_oof.parquet');report=[]
 for (window,),g in out.group_by('window'):
  for kind in pred:report.append({'window':window,'kind':kind,'labelled_AP':average_precision_score(g['label'],g[kind]),'labelled_AP_negative_weight50':average_precision_score(g['label'],g[kind],sample_weight=np.where(g['label'].to_numpy()==1,1,50))})
 (ROOT/'stress_comparison.json').write_text(json.dumps({'no_model_or_normalization_refit':True,'windows':task.W,'results':report},indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
