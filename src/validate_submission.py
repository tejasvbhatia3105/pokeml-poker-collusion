from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,duckdb
import argparse
parser=argparse.ArgumentParser();parser.add_argument('submission',nargs='?',default='submission.csv');parser.add_argument('--report',default='artifacts/submission_validation.json');parser.add_argument('--threads',type=int,default=5);parser.add_argument('--memory-limit',default='4GB');args=parser.parse_args()
s=pd.read_csv(args.submission,keep_default_na=False);t=pd.read_csv('data/sample_submission.csv');pairs=pd.read_csv('data/evaluation_pairs.csv')
assert list(s.columns)==list(t.columns)
assert len(s)==len(t)==112540
assert s.pair_id.tolist()==t.pair_id.tolist()
assert s.pair_id.is_unique and set(s.pair_id)==set(pairs.pair_id)
assert not (s.astype(str)=='').any().any()
assert np.isfinite(s.risk_score).all() and s.risk_score.between(0,1).all()
assert set(s.predicted_behavior)<=set(['none','directed_transfer','soft_play','coordinated_isolation','other_coordination'])
ecols=[f'evidence_hand_{i}' for i in range(1,6)]
for row in s[ecols].itertuples(index=False,name=None):
    used=[v for v in row if v!='NO_EVIDENCE'];assert len(used)==len(set(used))
e=s.melt(id_vars=['pair_id'],value_vars=ecols,value_name='hand_id');e=e[e.hand_id!='NO_EVIDENCE'].merge(pairs,on='pair_id')
c=duckdb.connect();assert args.threads>0;c.execute('SET threads=?',[args.threads]);c.execute('SET memory_limit=?',[args.memory_limit]);c.register('submitted_evidence',e)
bad=c.sql("""select count(*) from submitted_evidence e left join read_parquet('data/hands.parquet') h using(hand_id)
where h.hand_id is null or h.phase!='evaluation' or not exists(select 1 from read_parquet('data/seats.parquet') s where s.hand_id=e.hand_id and s.player_id=e.player_1) or not exists(select 1 from read_parquet('data/seats.parquet') s where s.hand_id=e.hand_id and s.player_id=e.player_2)""").fetchone()[0]
assert bad==0,bad
report={'rows':len(s),'evidence_hands':len(e),'invalid_evidence':bad,'empty_cells':0,'sha256':hashlib.sha256(Path(args.submission).read_bytes()).hexdigest(),'behavior_counts':s.predicted_behavior.value_counts().to_dict()}
Path(args.report).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
