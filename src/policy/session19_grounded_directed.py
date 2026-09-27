\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3');os.environ.setdefault('LOKY_MAX_CPU_COUNT','3')
from pathlib import Path
import numpy as np,polars as pl,joblib
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from session8_data import hand_data
from session17_grounded_tiers import distribution,conditional
from session7_likelihood import multilevel_inclusion
ROOT=Path('artifacts/evidence_session19_directed');C=pl.col
def options(g):
 q=g.filter(C('evidence')==1).sort('evidence_rank');ix=q['local_row'].to_numpy();t=q['time'].to_numpy();support=np.column_stack([q['fold_partner']>0,q['call_partner']>0]);out=[]
 for k in range(len(q)+1):
  c=np.r_[np.zeros(k,int),np.ones(len(q)-k,int)]
  if all(np.all(np.diff(t[c==r])>0) for r in range(2)) and support[np.arange(len(q)),c].all():out.append(c)
 return ix,out
def main():
 ROOT.mkdir(exist_ok=True);d=hand_data().filter(C('behavior_family')=='directed_transfer').drop('row').with_row_index('local_row');cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'));X=d.select(cols['event']).to_numpy();XT=d.select(cols['type']).to_numpy();fv=d['fold'].to_numpy();groups=[g for _,g in d.group_by('pair_id')];predcat=np.zeros((len(d),2));predhist=np.zeros_like(predcat);audit=[];start=time.time();(ROOT/'columns.json').write_text(json.dumps(cols,indent=2))
 with threadpool_limits(limits=2):
  for f in range(4):
   tr=fv!=f;va=~tr;sub=np.full(len(d),-1);pending=[];used=np.zeros(len(d),bool);skipped=[]
   for g in groups:
    if g['fold'][0]==f:continue
    ix,valid=options(g)
    if not valid:skipped.append(g['pair_id'][0]);continue
    pending.append((g,ix,valid));used[g['local_row'].to_numpy()]=True
    if len(valid)==1:sub[ix]=valid[0]
   known=(sub>=0)&tr;typ=CatBoostClassifier(iterations=200,depth=3,learning_rate=.04,l2_leaf_reg=10,thread_count=2,random_seed=6210+f,verbose=False,allow_writing_files=False);typ.fit(XT[known],sub[known]);typ.save_model(str(ROOT/f'type_fold{f}.cbm'));tp=typ.predict_proba(XT,thread_count=2);y=np.zeros((len(d),2));eligible=np.tile(used[:,None],(1,2));records=[]
   for g,ix,valid in pending:
    chosen=valid[int(np.argmax([np.log(tp[ix,c].clip(1e-8,1)).sum() for c in valid]))];y[ix,chosen]=1;rows=g['local_row'].to_numpy();records.append({'pair_id':g['pair_id'][0],'feasible':len(valid),'chosen':chosen.tolist()})
    if len(ix)==5:
     final=int(chosen[-1]);eligible[rows,final+1:]=False;after=rows[d['time'].to_numpy()[rows]>d['time'].to_numpy()[ix[-1]]];eligible[after,final]=False
   assert not np.any(eligible[va]);assert y[va].sum()==0
   for k in range(2):
    e=eligible[:,k];m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=2,random_seed=6310+11*f+k+1,verbose=False,allow_writing_files=False);m.fit(X[e],y[e,k]);m.save_model(str(ROOT/f'event{k}_fold{f}.cbm'));predcat[va,k]=m.predict_proba(X[va],thread_count=2)[:,1];h=HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=10,max_bins=127,early_stopping=False,random_state=4710+f);h.fit(X[e],y[e,k]);joblib.dump(h,ROOT/f'event{k}_fold{f}.joblib',compress=3);predhist[va,k]=h.predict_proba(X[va])[:,1];audit.append({'fold':f,'head':k,'eligible':int(e.sum()),'positive':int(y[e,k].sum()),'unique_type_rows':int(known.sum()),'excluded_incompatible_training_pairs':skipped,'validation_overlap':int((e&va).sum())})
   (ROOT/f'assignments_fold{f}.json').write_text(json.dumps(records,indent=2));print('directed grounded',f,round(time.time()-start,1),flush=True)
 d.select('pair_id','hand_id','time','fold').with_columns(*[pl.Series(f'{name}_{k}',p[:,k]) for name,p in [('cat',predcat),('hist',predhist)] for k in range(2)]).write_parquet(ROOT/'heads_oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2));assemble()
def assemble():
 d=hand_data();new=pl.read_parquet(ROOT/'heads_oof.parquet');r30=pl.read_parquet('artifacts/evidence_session11/conditional_family_deployed_oof.parquet').select('pair_id','hand_id','conditional_family');raw=pl.concat([pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').filter(C('fold')==f).select('pair_id','hand_id','base','cat_inclusion') for f in range(4)]);q=d.join(r30,on=['pair_id','hand_id'],validate='1:1').join(raw,on=['pair_id','hand_id'],validate='1:1').join(new.drop('time','fold'),on=['pair_id','hand_id'],how='left',validate='1:1');parts=[]
 for (pid,),g in q.group_by('pair_id'):
  g=g.sort('time','hand_id');old=g['conditional_family'].to_numpy();values={n:old.copy() for n in ['cat_only','joint','conditional_joint']}
  if g['behavior_family'][0]=='directed_transfer':
   ca=distribution(g.select('cat_0','cat_1').to_numpy());ha=distribution(g.select('hist_0','hist_1').to_numpy());jp=.5*(ca+ha);ci=multilevel_inclusion(ca);values['cat_only']=old+.25*(ci-g['cat_inclusion'].to_numpy());values['joint']=.25*g['base'].to_numpy()+.25*ci+.5*multilevel_inclusion(jp);values['conditional_joint']=.25*g['base'].to_numpy()+.25*ci+.5*conditional(jp,3)
  parts.append(g.select('pair_id','hand_id').with_columns(*[pl.Series(n,v) for n,v in values.items()]))
 pl.concat(parts).write_parquet(ROOT/'directed_oof.parquet')
if __name__=='__main__':main()
