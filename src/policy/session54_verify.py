import json
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session54_relationship_context import ROOT,BASE,P,PEER,GRAPH,CHANNELS,WINDOWS,context,folder,C
def main():
 d=pl.read_parquet(ROOT/'features.parquet');ref=pl.read_parquet(ROOT/'oof.parquet');fv=d['fold'].to_numpy();bc=json.load(open(P/'residual_columns.json'));err=0;models=0
 for f in range(4):
  va=fv==f
  for kind,cols in [('control',bc),('peer',bc+PEER),('graph',bc+PEER+GRAPH)]:
   m=CatBoostClassifier();path=BASE/f'residual_only_fold{f}.cbm' if kind=='control' else ROOT/f'{kind}_fold{f}.cbm';m.load_model(str(path));p=1-m.predict_proba(d.select(cols).to_numpy()[va],thread_count=2)[:,0];err=max(err,float(abs(p-ref[kind].to_numpy()[va]).max()));models+=1
 old=ref.join(pl.read_parquet(BASE/'oof.parquet').select('pair_id','window','residual_only'),on=['pair_id','window'],validate='1:1');control_error=float((old['control']-old['residual_only']).abs().max());assert err==0 and control_error==0;feature_error=0;invariance_error=0;checks=0
 for window in WINDOWS:
  wanted=set(d.filter(C('window')==window)['pair_id']);samples=0
  for path in sorted(folder(window).glob('T*.parquet')):
   q=pl.read_parquet(path,columns=['pair_id','phase','player_1','player_2','policy_n_hands',*CHANNELS]).filter(C('phase')=='development');z=context(q);ids=set(q['pair_id'])&wanted
   if ids:
    actual=z.filter(C('pair_id').is_in(list(ids))).join(d.filter(C('window')==window).select('pair_id',*PEER,*GRAPH),on='pair_id',suffix='_ref',validate='1:1');feature_error=max(feature_error,float(abs(actual.select(PEER+GRAPH).to_numpy()-actual.select([c+'_ref' for c in PEER+GRAPH]).to_numpy()).max()))
   if samples<3 and len(q)>20:
    nodes=sorted(set(q['player_1'])|set(q['player_2']));rng=np.random.default_rng(5411+samples);order=rng.permutation(len(nodes));mapping={p:f'renamed_{j:05d}' for p,j in zip(nodes,order)};renamed=q.with_columns(C('player_1').replace_strict(mapping),C('player_2').replace_strict(mapping));swapped=q.with_columns(C('player_2').alias('player_1'),C('player_1').alias('player_2'));labelled=q.with_columns(pl.lit(999).alias('label'),pl.lit('unrelated').alias('behavior_family'))
    for altered in [q.reverse(),renamed,swapped,labelled]:
     zz=z.select('pair_id').join(context(altered),on='pair_id',validate='1:1',maintain_order='left');invariance_error=max(invariance_error,float(abs(z.select(PEER+GRAPH).to_numpy()-zz.select(PEER+GRAPH).to_numpy()).max()));checks+=1
    samples+=1
 assert feature_error==0 and invariance_error<1e-5;report={'models_replayed':models,'saved_prediction_error':err,'original43_control_error':control_error,'all_window_graph_features_rebuilt_error':feature_error,'order_identifier_endpoint_label_invariance_cases':checks,'invariance_max_error':invariance_error,'supervision':'only confirmed labels; every graph formed before label filtering; outer folds isolate entire pools'};(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
