import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
import build_outcome_roles as BOR,build_relationship_evidence as BRE
C=pl.col;ROOT=Path('artifacts/evidence_session6')
def main():
    ix=pl.read_parquet('artifacts/evidence_session4/hand_index.parquet');labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2')
    audit=json.loads((ROOT/'chronology_audit.json').read_text());tables=[r['table_id'] for r in sorted((r for r in audit if r['kind']=='outcome'),key=lambda r:-r['changed_rows'])[:3]]
    counts=0;differences=[]
    for table in tables:
        q=ix.filter(C('table_id')==table).select('pair_id','hand_id').join(labs,on='pair_id');a=pl.read_parquet(BOR.root/'actions'/f'{table}.parquet')
        seats=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet');net={(r['hand_id'],r['player_id']):r['net_chips'] for r in seats.select('hand_id','player_id','net_chips').to_dicts()}
        st=pl.read_parquet(BOR.root/'states'/f'{table}.parquet');states=set(st.select('hand_id','street_no','player_id').iter_rows());actions={k:g.sort('action_no','player_id').to_dicts() for k,g in a.group_by('hand_id')}
        fixed=BOR.build(table,q,chronological=True);expected={};fields=['action_class','pot_odds','call_stack','log_bet_ratio','players_active']
        for r in q.to_dicts():
            h=r['hand_id'];p1=r['player_1'];p2=r['player_2'];key=(r['pair_id'],h);v={}
            for act in actions[(h,)]:
                p=act['player_id']
                if p not in (p1,p2):continue
                partner=p2 if p==p1 else p1
                if (h,act['street_no'],partner) not in states:continue
                for role,yes in [('lower',net[h,p]<=net[h,partner]),('higher',net[h,p]>=net[h,partner])]:
                    if yes:
                        for field in fields:v[f'outcome_{role}_s{int(act["street_no"])}_last_{field}']=act[field]
            expected[key]=v
        for r in fixed.to_dicts():
            reference=expected[r['pair_id'],r['hand_id']]
            for col in (c for c in fixed.columns if '_last_' in c):
                assert np.isclose(r[col],reference.get(col,-1),atol=1e-6), (table,r['hand_id'],col,r[col],reference.get(col,-1));counts+=1
        shuffled=q.sample(fraction=1,shuffle=True,seed=66)
        for builder in [BOR,BRE]:
            base=builder.build(table,q,chronological=True).sort('pair_id','hand_id');other=builder.build(table,shuffled,chronological=True).sort('pair_id','hand_id');cols=[c for c in base.columns if c not in ['pair_id','hand_id']]
            delta=float(np.max(np.abs(base.select(cols).to_numpy()-other.select(cols).to_numpy())));assert delta<1e-5;differences.append(delta)
    report={'tables':tables,'raw_last_action_values_checked':counts,'query_shuffle_max_error':max(differences),'status':'passed'};(ROOT/'chronology_checks.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
