import os,json,hashlib
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from threadpoolctl import threadpool_limits
from session7_em import data
from session7_model import load_models,score,ROOT
from session4_evidence_model import FAMILIES
def main():
    method=os.environ['METHOD'];d=data();models,cols=load_models(method);saved=pl.read_parquet(ROOT/f'{method}_oof.parquet');errs=[];mutation=[]
    limiter=threadpool_limits(limits=4)
    for f in range(4):
        for b in FAMILIES:
            q=d.filter((pl.col('fold')==f)&(pl.col('behavior_family')==b));pred=score(models,cols,b,q,[f]);expected=q.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],maintain_order='left',validate='1:1')['score'].to_numpy();errs.append(float(np.max(np.abs(pred-expected))))
            changed=q.sample(fraction=1,shuffle=True,seed=71).with_columns((1-pl.col('evidence')).alias('evidence'),pl.lit(999).alias('evidence_rank'));p=score(models,cols,b,changed,[f]);back=q.select('pair_id','hand_id').join(changed.select('pair_id','hand_id').with_columns(pl.Series('p',p)),on=['pair_id','hand_id'],maintain_order='left',validate='1:1')['p'].to_numpy();mutation.append(float(np.max(np.abs(pred-back))))
    assert max(errs)<1e-12 and max(mutation)<1e-12
    sha=hashlib.sha256(Path('artifacts/candidate_r28/submission.csv').read_bytes()).hexdigest();assert sha=='0094443d12882b2c62296893b45c1bfa359ef080dac358cf1800d69f8c8b99a8'
    out={'method':method,'replayed_hands':len(d),'replay_max_error':max(errs),'row_and_label_mutation_max_error':max(mutation),'r28_sha256':sha};(ROOT/f'{method}_checks.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
