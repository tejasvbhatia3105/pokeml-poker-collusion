\
\
\
\
\
\
import json,time
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session50_matchup import ROOT,load,hand_data,targets,labels,target,OLD,C,assemble
def main():
 full=hand_data();base=pl.read_parquet('artifacts/evidence_session41_isolation_bet_fold/paired_fold/event_oof.parquet');start=time.time()
 for kind,selected in [('outcomes',[0,1,2,4,5]),('current',[3,6])]:
  root=ROOT/kind;root.mkdir(exist_ok=True);(root/'config.json').write_text(json.dumps({'method':__doc__,'selected_columns':selected,'same_model_schedule_as_session50':True},indent=2));parts=[];audit=[]
  for family in ['directed_transfer','soft_play','coordinated_isolation']:
   folder=root/family;folder.mkdir(exist_ok=True);d,a,x,cols=load(family);ex=np.load(ROOT/family/'features.npz')['x'][:,selected];g=a['row'].to_numpy();r=a['actor'].to_numpy();iso=family=='coordinated_isolation';xx=np.full((len(d),len(selected)),-2.) if iso else ex
   if iso:xx[g]=ex
   x=np.column_stack([x,xx]);pp=np.zeros((len(d),2));fv=d['fold'].to_numpy();fam=full['behavior_family'].to_numpy()==family
   for f in range(4):
    if iso:p1,p2,e1,e2,_=targets(full,f,family);ys=np.column_stack([p1,p2])[fam];es=np.column_stack([e1,e2])[fam];va=fv==f
    elif family=='directed_transfer':y,tr,va=labels(d,a,f)
    else:y0,e,_=target(d,f);y=y0[g];tr=e[g];va=fv[g]==f
    for k in range(2 if iso else 1):
     if iso:y=ys[:,k];tr=es[:,k]
     assert not(tr&va).any();m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6311+11*f+k,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[tr],y[tr]);m.save_model(str(folder/f'event{k+1}_fold{f}.cbm'));p=m.predict_proba(x[va],thread_count=2)[:,1]
     if iso:pp[va,k]=p
     else:pp[g[va],r[va]]=p
     audit.append({'family':family,'fold':f,'head':k+1,'training_count':int(tr.sum()),'positive_count':int(y[tr].sum()),'validation_overlap':0})
    print('matchup ablation',kind,family,f,round(time.time()-start,1),flush=True)
   if iso:new=pp
   else:
    dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if family=='directed_transfer' else np.ones_like(pp);new=(pp*dw).sum(1)[:,None]
   parts.append(d.select('pair_id','hand_id').with_columns(pl.Series('new_primary',new[:,0]),pl.Series('new_secondary',new[:,1] if iso else np.full(len(d),np.nan))))
  q=base.join(pl.concat(parts),on=['pair_id','hand_id'],validate='1:1').with_columns(C('new_primary').alias('bg_primary'),pl.when(C('new_secondary').is_nan()).then(C('bg_secondary')).otherwise(C('new_secondary')).alias('bg_secondary')).drop('new_primary','new_secondary');q.write_parquet(root/'event_oof.parquet');(root/'audit.json').write_text(json.dumps(audit,indent=2));assemble(root)
if __name__=='__main__':main()
