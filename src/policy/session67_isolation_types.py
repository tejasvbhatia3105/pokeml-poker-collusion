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
from sklearn.metrics import roc_auc_score,log_loss,balanced_accuracy_score
from session57_isolation_pressure import data
from session59_pressure_equity import COLS
from cards import preflop_table
ROOT=Path('artifacts/evidence_session67_isolation_types');C=pl.col
def design(d,a,ac):
 cache=np.load('artifacts/evidence_session59_pressure_equity/equity_audit.npz');pf=preflop_table();v=np.zeros((len(a),3),np.float32)
 for row,before,after,own,partner in cache['meta']:
  s=cache['states'][before];hands=[s[2*k:2*k+2] for k in [own,partner]];eq=[pf[h[0]//4,h[1]//4,int(h[0]%4==h[1]%4)] for h in hands];v[row]=[eq[0],eq[1],eq[0]-eq[1]]
 ex=np.load('artifacts/evidence_session59_pressure_equity/features.npz')['x'];cols=ac+COLS+['own_preflop_equity','partner_preflop_equity','preflop_equity_gap'];q=a.select('pair_id','hand_id',*ac).with_columns(*[pl.Series(c,ex[:,i]) for i,c in enumerate(COLS)],*[pl.Series(c,v[:,i]) for i,c in enumerate(cols[-3:])]);agg=q.group_by('pair_id','hand_id').agg(*[getattr(C(c),stat)().alias(c+'_'+stat) for stat in ['mean','min','max'] for c in cols],pl.len().alias('pressure_count'));pc=[c+'_'+stat for stat in ['mean','min','max'] for c in cols]+['pressure_count'];hx=['fold_partner','call_partner','agg_partner','check_hu'];z=d.select('pair_id','hand_id',*hx).join(agg,on=['pair_id','hand_id'],how='left',validate='1:1',maintain_order='left').with_columns(C('pressure_count').fill_null(0)).select(hx+pc).fill_null(-2);return z.to_numpy().astype(np.float32),hx+pc
def main():
 ROOT.mkdir(exist_ok=True);_,d,a,ac=data();x,cols=design(d,a,ac);np.savez_compressed(ROOT/'features.npz',x=x);(ROOT/'columns.json').write_text(json.dumps(cols,indent=2));cfg=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'));xt=d.select(cfg['type']).to_numpy();fv=d['fold'].to_numpy();sub=d['subtype'].to_numpy();known=sub>0;y=sub==1;pred={k:np.zeros(len(d)) for k in ['original','pressure','augmented']};audit=[];start=time.time()
 for f in range(4):
  tr=(fv!=f)&known;va=fv==f
  for kind in pred:
   X=xt if kind=='original' else x if kind=='pressure' else np.column_stack([xt,x]);m=CatBoostClassifier()
   if kind=='original':m.load_model(f'artifacts/evidence_session6/priority_ordered_type_coordinated_isolation_fold{f}.cbm')
   elif os.environ.get('TYPE_REPORT_ONLY')=='1':m.load_model(str(ROOT/f'{kind}_fold{f}.cbm'))
   else:m=CatBoostClassifier(iterations=200,depth=3,learning_rate=.04,l2_leaf_reg=10,random_seed=6210+f,thread_count=2,verbose=False,allow_writing_files=False);m.fit(X[tr],y[tr]);m.save_model(str(ROOT/f'{kind}_fold{f}.cbm'))
   pred[kind][va]=m.predict_proba(X[va],thread_count=2)[:,1];audit.append({'fold':f,'model':kind,'training_known_hands':int(tr.sum()),'primary_training':int(y[tr].sum()),'heldout_known_hands':int((va&known).sum()),'validation_overlap':0})
  print('isolation type',f,round(time.time()-start,1),flush=True)
 out=d.select('pair_id','hand_id','fold','subtype').with_columns(*[pl.Series(k,p) for k,p in pred.items()]);out.write_parquet(ROOT/'type_oof.parquet');metrics={}
 for k,p in pred.items():metrics[k]={'known_subtype_AUC':roc_auc_score(y[known],p[known]),'known_subtype_logloss':log_loss(y[known],p[known]),'balanced_accuracy':balanced_accuracy_score(y[known],p[known]>=.5),'fold_logloss':[log_loss(y[known&(fv==f)],p[known&(fv==f)]) if np.any(known&(fv==f)) else None for f in range(4)]}
 flags={'partner_fold':d['fold_partner'].to_numpy()>0,'partner_call':d['call_partner'].to_numpy()>0,'partner_raise':d['agg_partner'].to_numpy()>0,'hu_check':d['check_hu'].to_numpy()>0,'multiple_pressure':x[:,cols.index('pressure_count')]>1,'postflop_pressure':x[:,cols.index('street_no_max')]>0,'pressure_facing_bet':x[:,cols.index('call_bb_max')]>0,'any_preflop_weaker_actor':x[:,cols.index('preflop_equity_gap_min')]<0,'all_preflop_weaker_actors':x[:,cols.index('preflop_equity_gap_max')]<0,'any_team_equity_below_half':x[:,cols.index('precise_mw_team_min')]<.5};rules=[]
 for name,v in flags.items():rules.append({'rule':name,'primary_true':int((v&known&y).sum()),'primary_false':int((~v&known&y).sum()),'secondary_true':int((v&known&~y).sum()),'secondary_false':int((~v&known&~y).sum())})
 report={'method':__doc__,'known_hands':int(known.sum()),'known_pairs':d.filter(C('subtype')>0)['pair_id'].n_unique(),'primary_known':int((known&y).sum()),'metrics':metrics,'descriptive_rules':rules,'fit_audit':audit};(ROOT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
