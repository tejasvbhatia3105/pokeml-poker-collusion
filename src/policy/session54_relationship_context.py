\
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
from sklearn.metrics import average_precision_score
C=pl.col;ROOT=Path('artifacts/evidence_session54_relationship_context');P=Path('artifacts/policy');BASE=Path('artifacts/evidence_session43_pair_interactions');WINDOWS=['full','first_2000','last_2000'];NAMES=['none','directed_transfer','soft_play','coordinated_isolation']
CHANNELS=[n+'_mean' for n in ['alive_agg','alive_fold','partner_call','weak_partner_call','partner_surrender','hu_passivity','hu_check','outsider_agg','weak_outsider_agg','yield_better','size_partner','size_outsider']]
PEER=[f'peer_{n}_{s}' for n in CHANNELS for s in ['mean_min','mean_max','z_min','z_max']]
GRAPH=[f'graph_{n}_{s}' for n in CHANNELS for s in ['positive_path_mean','positive_path_max']]
def context(q):
                                                                               
 nodes=sorted(set(q['player_1'])|set(q['player_2']));node={p:i for i,p in enumerate(nodes)};a=np.array([node[p] for p in q['player_1']]);b=np.array([node[p] for p in q['player_2']]);n=len(nodes);v=q.select(CHANNELS).to_numpy().astype(np.float64);w=q['policy_n_hands'].to_numpy().astype(np.float64);assert np.isfinite(v).all() and np.all(w>=0);assert len(set(zip(np.minimum(a,b),np.maximum(a,b))))==len(q)
 count=np.zeros(n);total=np.zeros((n,len(CHANNELS)));sq=np.zeros_like(total)
 for ix in [a,b]:np.add.at(count,ix,w);np.add.at(total,ix,w[:,None]*v);np.add.at(sq,ix,w[:,None]*v*v)
 means=[];zs=[]
 for ix in [a,b]:
  den=count[ix]-w;mu=np.divide(total[ix]-w[:,None]*v,den[:,None],out=np.zeros_like(v),where=den[:,None]>0);second=np.divide(sq[ix]-w[:,None]*v*v,den[:,None],out=np.zeros_like(v),where=den[:,None]>0);sd=np.sqrt(np.maximum(0,second-mu*mu));z=(v-mu)/np.maximum(.05,sd);z[den<=0]=0;means.append(mu);zs.append(z)
 peer=np.stack([np.minimum(*means),np.maximum(*means),np.minimum(*zs),np.maximum(*zs)],axis=2).reshape(len(q),-1);edge=(w>0).astype(float);adj=np.zeros((n,n));adj[a,b]=adj[b,a]=edge;paths=[]
 for k in range(len(CHANNELS)):
  strength=np.maximum(v[:,k],0)*edge;mat=np.zeros((n,n));mat[a,b]=mat[b,a]=strength;prod=mat[a,:]*mat[b,:];common=adj[a,:]*adj[b,:];den=common.sum(1);mean=np.divide(prod.sum(1),den,out=np.zeros(len(q)),where=den>0);paths.extend([mean,prod.max(1)])
 graph=np.column_stack(paths);return q.select('pair_id').with_columns(*[pl.Series(c,x) for c,x in zip(PEER+GRAPH,np.column_stack([peer,graph]).astype(np.float32).T)])
def folder(window):return P/'pair_features' if window=='full' else P/'full_window_stress'/window/'pair_features'
def build():
 d=pl.read_parquet(BASE/'window_features.parquet');parts=[];audit=[]
 for window in WINDOWS:
  wanted=set(d.filter(C('window')==window)['pair_id']);seen=set();all_count=0
  for path in sorted(folder(window).glob('T*.parquet')):
   q=pl.read_parquet(path,columns=['pair_id','phase','player_1','player_2','policy_n_hands',*CHANNELS]).filter(C('phase')=='development');all_count+=len(q);z=context(q);ids=set(q['pair_id'])&wanted;seen|=ids
   if ids:parts.append(z.filter(C('pair_id').is_in(list(ids))).with_columns(pl.lit(window).alias('window')))
  assert seen==wanted;audit.append({'window':window,'all_gameplay_pairs_used':all_count,'labelled_training_rows':len(wanted),'known_labels_joined_after_graph_features':True})
 extra=pl.concat(parts);out=d.select('pair_id','table_id','fold','window','label','behavior_family','n_ev_in',*json.load(open(P/'residual_columns.json'))).join(extra,on=['pair_id','window'],validate='1:1',maintain_order='left');assert out.select(PEER+GRAPH).null_count().to_numpy().sum()==0;out.write_parquet(ROOT/'features.parquet');(ROOT/'feature_audit.json').write_text(json.dumps(audit,indent=2));return out
def metrics(out,kinds):
 report=[]
 for (window,),q in out.group_by('window'):
  for kind in kinds:
   y=q['label'].to_numpy();p=q[kind].to_numpy();report.append({'window':window,'model':kind,'labelled_AP':average_precision_score(y,p),'negative_weight50_AP':average_precision_score(y,p,sample_weight=np.where(y==1,1,50)),'fold_weighted_AP':[average_precision_score(z['label'],z[kind],sample_weight=np.where(z['label'].to_numpy()==1,1,50)) for _,z in q.sort('fold').group_by('fold',maintain_order=True)]})
 return report
def main():
 ROOT.mkdir(exist_ok=True);bc=json.load(open(P/'residual_columns.json'));(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'base_columns':bc,'peer_columns':PEER,'graph_columns':GRAPH,'raw_channels':CHANNELS,'peer_sd_floor':.05,'schedule':'same43 Cat600 depth5 lr.04 L2 10 seed991+fold, full weight1/crops.5','caveat':'known-label and weighted-negative AP are diagnostics, not population AP'},indent=2));d=build();fv=d['fold'].to_numpy();y=np.array([NAMES.index(v) for v in d['behavior_family']]);wt=np.where(d['window'].to_numpy()=='full',1,.5);pred={k:np.zeros(len(d)) for k in ['control','peer','graph']};start=time.time()
 for f in range(4):
  tr=fv!=f;va=fv==f
  for kind,extra in [('control',[]),('peer',PEER),('graph',PEER+GRAPH)]:
   x=d.select(bc+extra).to_numpy()
   if kind=='control':m=CatBoostClassifier();m.load_model(str(BASE/f'residual_only_fold{f}.cbm'))
   else:m=CatBoostClassifier(iterations=600,depth=5,learning_rate=.04,loss_function='MultiClass',l2_leaf_reg=10,thread_count=2,random_seed=991+f,verbose=False,allow_writing_files=False);m.fit(x[tr],y[tr],sample_weight=wt[tr]);m.save_model(str(ROOT/f'{kind}_fold{f}.cbm'))
   pred[kind][va]=1-m.predict_proba(x[va],thread_count=2)[:,0];print('relationship context',f,kind,round(time.time()-start,1),flush=True)
 out=d.select('pair_id','table_id','fold','window','label','behavior_family','n_ev_in').with_columns(*[pl.Series(k,v) for k,v in pred.items()]);out.write_parquet(ROOT/'oof.parquet');report=metrics(out,list(pred));(ROOT/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
