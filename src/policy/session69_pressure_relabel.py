\
\
\
\
\
\
import json
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
import session58_pressure_comparison as training
from session57_isolation_pressure import data
from session67_isolation_types import design,ROOT as TYPE_ROOT
from session59_pressure_equity import COLS,ROOT as OLD
from session6_priority import training_targets
from session8_data import targets as original_targets
ROOT=Path('artifacts/evidence_session69_pressure_relabel');C=pl.col
AUDIT=[]
def targets(full,f,b):
 assert b=='coordinated_isolation'
 d=full.filter(C('behavior_family')==b).drop('row').with_row_index('row')
                                                                             
 m=CatBoostClassifier();m.load_model(str(TYPE_ROOT/f'pressure_fold{f}.cbm'))
 tp=m.predict_proba(TYPE_X,thread_count=2)[:,1];tr=d['fold'].to_numpy()!=f
 groups=[g['row'].to_numpy() for _,g in d.group_by('pair_id',maintain_order=True)]
 local=training_targets(d['evidence'].to_numpy(),d['subtype'].to_numpy(),tp,tr,d['time'].to_numpy(),groups,d['evidence_rank'].fill_null(0).to_numpy())
 mask=full['behavior_family'].to_numpy()==b;out=[]
 old=original_targets(full,f,b)
 for k,v in enumerate(local):
  assert not np.any(v&~tr);q=np.zeros(len(full),bool);q[mask]=v;out.append(q)
  AUDIT.append({'fold':f,'field':['primary','secondary','eligible_primary','eligible_secondary'][k],'old_count':int(old[k].sum()),'new_count':int(q.sum()),'changed_training_rows':int((q!=old[k]).sum()),'heldout_used':False})
 return (*out,(full['fold'].to_numpy()==f)&mask)
def compute(d,a,players=None):
 return np.load(OLD/'features.npz')['x'],{'source':'verified59 precision features; unchanged','subtype_teacher':'67 pressure-only, outer training known labels'}
def main():
 global TYPE_X
 ROOT.mkdir(exist_ok=True);_,d,a,ac=data();TYPE_X,cols=design(d,a,ac)
 np.testing.assert_array_equal(TYPE_X,np.load(TYPE_ROOT/'features.npz')['x'])
 training.ROOT=ROOT;training.COLS=COLS;training.compute=compute;training.targets=targets;training.__doc__=__doc__;training.main()
 (ROOT/'target_changes.json').write_text(json.dumps(AUDIT,indent=2))
if __name__=='__main__':main()
