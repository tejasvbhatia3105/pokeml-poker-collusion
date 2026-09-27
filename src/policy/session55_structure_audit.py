import json
import numpy as np,polars as pl
from session8_data import hand_data
from session11_conditional_family import template
from session55_current_targets import ROOT
C=pl.col
def main():
 d=hand_data();ap=pl.read_csv('artifacts/evidence_session12/matchup_combined_comparison.csv').select('pair_id',C('current').alias('r32_AP'));rows=[]
 for (pid,),g in d.group_by('pair_id'):
  g=g.sort('time','hand_id');p=g.filter(C('evidence')==1).sort('evidence_rank');t=p['time'].to_numpy();positions={hid:i for i,hid in enumerate(g['hand_id'])};e=np.array([positions[h] for h in p['hand_id']]);rows.append({'pair_id':pid,'family':g['behavior_family'][0],'restarts':int((np.diff(t)<0).sum()),'listed':len(p),'two_tier_compatible':len(template(len(g),e))>0})
 r=pl.DataFrame(rows).join(ap,on='pair_id',validate='1:1');out={'family_and_restart_counts':r.group_by('family','restarts','two_tier_compatible').agg(pl.len(),C('r32_AP').mean()).sort('family','restarts').to_dicts(),'incompatible_pairs':r.filter(~C('two_tier_compatible')).to_dicts(),'scope':'descriptive annotation-order audit; public labels, no inference rule or tuning'};(ROOT/'list_structure_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
