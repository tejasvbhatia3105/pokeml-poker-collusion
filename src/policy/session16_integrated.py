\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3');os.environ.setdefault('LOKY_MAX_CPU_COUNT','3')
from pathlib import Path
import numpy as np,polars as pl,torch,joblib
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from session8_data import hand_data,targets
from session11_conditional_family import Model,features
from session6_priority import inclusion
from session8_count_conditioning import conditioned
ROOT=Path('artifacts/evidence_session16_integrated');C=pl.col;torch.set_num_threads(2)
def data():
 d=hand_data();a=pl.read_parquet('artifacts/evidence_session13_actor/hand_features.parquet');b=pl.read_parquet('artifacts/evidence_session14_outcomes/hand_features.parquet');ac=[c for c in a.columns if c.startswith('global_')];bc=[c for c in b.columns if '_prediction_' in c or '_residual_' in c];d=d.join(a.select('pair_id','hand_id',*ac),on=['pair_id','hand_id'],validate='1:1',maintain_order='left').join(b.select('pair_id','hand_id',*bc),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event']+ac+bc;return d,cols
def train():
 ROOT.mkdir(exist_ok=True);d,cols=data();(ROOT/'columns.json').write_text(json.dumps(cols,indent=2));X=d.select(cols).to_numpy();fv=d['fold'].to_numpy();fam=d['behavior_family'].to_numpy();y=d['evidence'].to_numpy();pred=np.zeros((len(d),4));audit=[];start=time.time()
 with threadpool_limits(limits=2):
  for f in range(4):
   for family in ['directed_transfer','soft_play','coordinated_isolation']:
    tr=(fv!=f)&(fam==family);va=(fv==f)&(fam==family);catpath=ROOT/f'base_{family}_fold{f}.cbm';cat=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=2,random_seed=1710+11*f,verbose=False,allow_writing_files=False)
    if catpath.exists():cat.load_model(str(catpath))
    else:cat.fit(X[tr],y[tr]);cat.save_model(str(catpath))
    pred[va,0]=cat.predict_proba(X[va],thread_count=2)[:,1];hp=ROOT/f'base_{family}_fold{f}.joblib'
    if hp.exists():hist=joblib.load(hp)
    else:hist=HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=10,max_bins=127,early_stopping=False,random_state=4710+f);hist.fit(X[tr],y[tr]);joblib.dump(hist,hp,compress=3)
    pred[va,1]=hist.predict_proba(X[va])[:,1];a,b,ea,eb,ev=targets(d,f,family);assert np.array_equal(ev,va)
    for k,(ey,e) in enumerate([(a,ea),(b,eb)],1):
     assert not np.any(e&va);path=ROOT/f'event{k}_{family}_fold{f}.cbm';m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=2,random_seed=6310+11*f+k,verbose=False,allow_writing_files=False)
     if path.exists():m.load_model(str(path))
     else:m.fit(X[e],ey[e].astype(int));m.save_model(str(path))
     pred[va,k+1]=m.predict_proba(X[va],thread_count=2)[:,1];audit.append({'fold':f,'family':family,'head':k,'training_rows':int(e.sum()),'validation_overlap':int((e&va).sum())})
    print('integrated',f,family,round(time.time()-start,1),flush=True)
 d.select('pair_id','hand_id','fold','time','behavior_family').with_columns(*[pl.Series(n,pred[:,k]) for k,n in enumerate(['new_base_cat','new_base_hist','bg_primary','bg_secondary'])]).write_parquet(ROOT/'heads_oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
def assemble():
 d=hand_data();new=pl.read_parquet(ROOT/'heads_oof.parquet').select('pair_id','hand_id','new_base_cat','new_base_hist','bg_primary','bg_secondary');r30=pl.read_parquet('artifacts/evidence_session11/conditional_family_deployed_oof.parquet').select('pair_id','hand_id','conditional_family');parts=[]
 for f in range(4):
  q=d.filter(C('fold')==f).join(pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').filter(C('fold')==f).drop('fold','time'),on=['pair_id','hand_id'],validate='1:1').join(new,on=['pair_id','hand_id'],validate='1:1').join(r30,on=['pair_id','hand_id'],validate='1:1');models=[]
  for seed in [1010,2020]:
   st=torch.load(f'artifacts/evidence_session11/conditional_family/list_independent_fold{f}_seed{seed}.pt',weights_only=False);m=Model(35,'independent');m.load_state_dict(st['state_dict']);models.append((st,m.eval()))
  for (pid,),g in q.group_by('pair_id'):
   g=g.sort('time','hand_id');nb=.5*g['new_base_cat'].to_numpy()+.5*g['new_base_hist'].to_numpy();out=g.select('pair_id','hand_id').with_columns(pl.Series('base_only',g['conditional_family'].to_numpy()+.25*(nb-g['base'].to_numpy())))
   for arm in ['base_propagated','cat_propagated','both_propagated']:
    base=nb if arm!='cat_propagated' else g['base'].to_numpy();ca=g.select('cat_primary','cat_secondary').to_numpy() if arm=='base_propagated' else g.select('bg_primary','bg_secondary').to_numpy();ha=g.select('hist_primary','hist_secondary').to_numpy();nc=ca/np.maximum(1,ca.sum(1))[:,None];nh=ha/np.maximum(1,ha.sum(1))[:,None];jp=.5*(nc+nh);ci=inclusion(nc[:,0],nc[:,1]);ji=inclusion(jp[:,0],jp[:,1]);z=g.with_columns(pl.Series('base',base),pl.Series('cat_primary',ca[:,0]),pl.Series('cat_secondary',ca[:,1]),pl.Series('cat_inclusion',ci),pl.Series('joint_inclusion',ji),pl.Series('r29',.25*base+.25*ci+.5*ji));x,p=features(z);inc=[]
    with torch.no_grad():
     for st,m in models:
      delta=m(torch.tensor(np.clip((x-st['mu'])/st['sd'],-6,6))[None],torch.ones((1,len(g)),dtype=torch.bool))[0];v=torch.tensor(p)+delta;prob=torch.softmax(torch.cat([torch.zeros_like(v[:,:1]),v],1),1).numpy();inc.append(conditioned(prob[:,1:],st['minimums'][g['behavior_family'][0]]))
    out=out.with_columns(pl.Series(arm,.25*base+.25*ci+.5*np.mean(inc,0)))
   parts.append(out)
 pl.concat(parts).write_parquet(ROOT/'integrated_oof.parquet')
if __name__=='__main__':train();assemble()
