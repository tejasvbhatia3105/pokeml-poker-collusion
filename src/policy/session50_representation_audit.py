import json
import numpy as np,polars as pl
from session50_matchup import ROOT,load,C
from session51_matchup_inference import current_from_design
def main():
 cfg=json.load(open('artifacts/evidence_session37_bet_fold/config.json'));records=[]
 for family in ['directed_transfer','soft_play','coordinated_isolation']:
  d,a,x,cols=load(family)
  if family=='coordinated_isolation':
   folder=ROOT.parent/'evidence_session41_isolation_bet_fold';ex=pl.read_parquet(folder/'action_features.parquet').sort('action_row');x=np.column_stack([a.select(cfg['fold_columns']).to_numpy(),d.select(cfg['hand_columns']).to_numpy()[a['row'].to_numpy()],ex.select(cfg['paired_columns']).to_numpy()])
  rebuilt=current_from_design(a,x,cfg);expected=np.load(ROOT/family/'features.npz')['x'][:,[3,6]];err=float(abs(rebuilt-expected).max());assert err==0;records.append({'family':family,'actions':len(a),'feature_error':err})
 report={'feature_identities':records,'finding':'The winning ablation adds no new raw information: current made-hand ordering and full rank gap are exact nonlinear/linear reexpressions of the already aligned made_category and made_kicker fields. The latter encodes all kickers, not just one. Historical pair-equity features also exist in build_value.py; the outcome arm refines/aligned these, not a wholly new information source.','inference':'Current-only uses verified existing action inputs, requires no new card simulation or raw-card joins.'};(ROOT/'representation_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
