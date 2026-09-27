\
\
import os,json,numpy as np,polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
C=pl.col; S=os.environ.get('POKEML_SCRATCH','artifacts/session4r_cache')+'/'; os.makedirs(S,exist_ok=True)
mem=json.load(open('artifacts/candidate_r34/build_manifest.json'))['members']+json.load(open('artifacts/candidate_r41/build_manifest.json'))['new_pairs']+list(pl.read_parquet(S+'lift_candidates2.parquet')['pair_id'])
ev=pl.read_parquet(S+'card_share_eval.parquet').filter(C('pair_id').is_in(mem))
z1=pl.when(C('n_after').fill_null(0)>=25).then(C('z_after')).otherwise(0.0); z2=pl.when(C('n_after2').fill_null(0)>=25).then(C('z_after2')).otherwise(0.0)
ev=ev.with_columns(z1.alias('z1'),z2.alias('z2'))
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','phase','started_at']).filter(C('phase')=='evaluation')
seats=pl.read_parquet('data/seats.parquet',columns=['hand_id','player_id']); ptab=seats.join(hands.select('hand_id','table_id'),on='hand_id').group_by('player_id').agg(C('table_id').first()); ptab=dict(zip(ptab['player_id'],ptab['table_id']))
FEAT=['equity','made_category','rank_high','rank_low','suited','pocket','style_0','style_1','style_2','style_3','local_style_3','players_active','previous_action','prior_raises','pot_bb','call_bb','stack_bb','pot_odds','position','log_amount_bb','log_bet_ratio','trel']
rows=[]
for pid,p1,p2,za_,zb_ in ev.select('pair_id','player_1','player_2','z1','z2').iter_rows():
    P,Q=(p1,p2) if za_>=zb_ else (p2,p1); t=ptab[P]
    a=pl.read_parquet(f'artifacts/policy/actions/{t}.parquet').filter((C('phase')=='evaluation')&(C('street_no')==0)&(C('player_id')==P)).sort('action_no').group_by('hand_id',maintain_order=True).first()
    st=pl.read_parquet(f'artifacts/policy/states/{t}.parquet',columns=['hand_id','player_id','street_no','equity']).filter((C('street_no')==0)&(C('player_id')==Q)).select('hand_id',C('equity').alias('eQ'))
    j=a.join(st,on='hand_id').filter(C('action_class')==3)                                          
    tmin=a['time_index'].min(); tmax=a['time_index'].max()
    j=j.with_columns(((C('time_index')-tmin)/max(1,tmax-tmin)).alias('trel'),pl.lit(pid).alias('pair'),(C('eQ')>=0.5).cast(pl.Int32).alias('y'))
    rows.append(j.select('pair','hand_id','time_index','eQ','y',*FEAT))
d=pl.concat(rows); print('rows',d.height,'strong-Q raises',int(d['y'].sum()),'pairs',d['pair'].n_unique(),flush=True)
X=d.select(FEAT).to_numpy().astype(np.float32); y=d['y'].to_numpy(); pairs=d['pair'].to_numpy(); up=np.unique(pairs); rng=np.random.default_rng(0); rng.shuffle(up)
folds={p:i%5 for i,p in enumerate(up)}; fid=np.array([folds[p] for p in pairs]); oof=np.zeros(len(y))
for f in range(5):
    tr=fid!=f; m=CatBoostClassifier(iterations=400,depth=4,learning_rate=0.05,l2_leaf_reg=10,verbose=0,random_seed=0); m.fit(X[tr],y[tr]); oof[~tr]=m.predict_proba(X[~tr])[:,1]
from sklearn.metrics import roc_auc_score
print('OOF AUC strong-Q vs weak-Q raises:',round(roc_auc_score(y,oof),4))
m=CatBoostClassifier(iterations=400,depth=4,learning_rate=0.05,l2_leaf_reg=10,verbose=0,random_seed=0); m.fit(X,y)
imp=sorted(zip(m.get_feature_importance(),FEAT),reverse=True)[:10]; print('top features',[(f,round(v,1)) for v,f in imp])
d.with_columns(pl.Series('pu',oof)).write_parquet(S+'pump_pu_scores.parquet'); print('saved')
