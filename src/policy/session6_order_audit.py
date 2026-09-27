import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
ROOT=Path('artifacts/evidence_session6')
def build():
    d=pl.read_csv('data/development_evidence.csv').join(pl.read_parquet('artifacts/evidence_session4/hand_index.parquet').select('pair_id','hand_id','time'),on=['pair_id','hand_id'],validate='1:1')
    rows=[];labels=[]
    for (pid,),g in d.sort('pair_id','evidence_rank').group_by('pair_id',maintain_order=True):
        t=g['time'].to_numpy()*5000;br=np.flatnonzero(np.diff(t)<0)+1
        rows.append({'pair_id':pid,'family':g['behavior_family'][0],'n':len(g),'descents':len(br),'first_break':int(br[0]) if len(br) else 0,'chronological_span':float(t.max()-t.min())})
        if len(br)==1:
            labels.extend({'pair_id':pid,'hand_id':h,'subtype':1 if i<br[0] else 2} for i,h in enumerate(g['hand_id']))
    a=pl.DataFrame(rows);lab=pl.DataFrame(labels)
    previous=ROOT/'order_subtype_labels.parquet'
    if previous.exists():assert lab.sort('pair_id','hand_id').equals(pl.read_parquet(previous).sort('pair_id','hand_id'))
    a.write_csv(ROOT/'evidence_order_audit.csv');lab.write_parquet(previous)
    report={'pairs':len(a),'descent_counts':a.group_by('family','descents').len().sort('family','descents').to_dicts(),'known_subtype_hands':len(lab)}
    (ROOT/'evidence_order_summary.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':build()
