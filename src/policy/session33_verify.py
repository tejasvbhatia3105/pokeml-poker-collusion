import os,json,sys
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data
from session33_sidepot_features import table_features
from session33_terminal_replay import payout
C=pl.col;ROOT=Path('artifacts/evidence_session33_rollout')
def main():
 d=hand_data();f=pl.read_parquet(ROOT/'hand_features.parquet');assert len(f)==len(d) and len(f.columns)==290;labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');audit=json.load(open(ROOT/'feature_audit.json'));assert all(max(r['stack_error'],r['pot_error'],r['call_error'])==0 for r in audit);fp=ROOT/'feature_verification.json'
 if not fp.exists():
  con=np.array([10.,20.,30.,0,0,0]);alive=np.array([1,1,1,0,0,0],bool);rank=np.array([3,2,1,0,0,0]);np.testing.assert_array_equal(payout(con,alive,rank),[30,20,10,0,0,0]);alive[0]=False;np.testing.assert_array_equal(payout(con,alive,rank),[0,50,10,0,0,0]);samples=[]
  tables=[r['table_id'] for r in sorted(audit,key=lambda r:-r['sidepot_scenarios'])[:3]]+[sorted(r['table_id'] for r in audit)[0]]
  for table in tables:
   q=d.filter(C('table_id')==table).select('pair_id','hand_id').join(labs,on='pair_id',validate='m:1');z,checks=table_features(table,q);want=z.select('pair_id','hand_id').join(f,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');cols=[c for c in z.columns if c.startswith('claims_')];err=float(abs(z.select(cols).to_numpy()-want.select(cols).to_numpy()).max());swap=q.with_columns(C('player_2').alias('player_1'),C('player_1').alias('player_2'));zz,_=table_features(table,swap);zz=z.select('pair_id','hand_id').join(zz,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');se=float(abs(z.select(cols).to_numpy()-zz.select(cols).to_numpy()).max());assert max(err,se)<1e-6;samples.append({'table':table,'feature_rebuild_error':err,'endpoint_swap_error':se,**checks})
  features={'hands':len(f),'features':288,'ledger_actions':sum(r['actions'] for r in audit),'pair_member_actions':sum(r['pair_member_actions'] for r in audit),'global_ledger_replay_error':0.,'exact_sidepot_examples':True,'sampled_physics_rebuilds':samples,'scope':'all-ledger checks; feature/claim-identity/swap checks sampled from three most sidepot-heavy tables plus one deterministic table'};fp.write_text(json.dumps(features,indent=2))
 if '--features-only' in sys.argv:print(fp.read_text());return
 models={};q=d.join(f,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
 for kind in ['naive','sidepot']:
  cols=json.load(open(ROOT/kind/'columns.json'));X=q.select(cols).to_numpy();saved=d.select('pair_id','hand_id').join(pl.read_parquet(ROOT/kind/'event_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');err=0.;count=0
  for fold in range(4):
   for fam in ['directed_transfer','soft_play','coordinated_isolation']:
    va=(d['fold'].to_numpy()==fold)&(d['behavior_family'].to_numpy()==fam)
    for k in [1,2]:
     m=CatBoostClassifier();m.load_model(str(ROOT/kind/f'event{k}_{fam}_fold{fold}.cbm'));p=m.predict_proba(X[va],thread_count=2)[:,1];err=max(err,float(abs(p-saved['bg_primary' if k==1 else 'bg_secondary'].to_numpy()[va]).max()));count+=1
  assert err<1e-12;models[kind]={'models_replayed':count,'max_error':err}
 report={'features':json.loads(fp.read_text()),'models':models,'limitation':'deterministic all-call/check-down response scenario, not causal EV or a normal-policy rollout; model-selection validation reused'};(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
