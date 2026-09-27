import json
from pathlib import Path
import polars as pl
ROOT=Path('artifacts/pair_session129_phase_history')
def main():
    ROOT.mkdir(exist_ok=True)
    a=pl.read_parquet('artifacts/evidence_session91_family_uncertainty/evaluation.parquet').select('pair_id',pl.col('risk_score').alias('evaluation_risk'))
    b=pl.read_parquet('artifacts/evidence_session91_family_uncertainty/full.parquet').select('pair_id',pl.col('risk_score').alias('history_risk'))
    z=a.join(b,on='pair_id',how='left',validate='1:1');z.write_parquet(ROOT/'comparison.parquet')
    report=dict(missing_history=z['history_risk'].null_count(),thresholds=[])
    for t in [.1,.5,.9]:
        report['thresholds'].append(dict(threshold=t,evaluation_high=z.filter(pl.col('evaluation_risk')>t).height,
            history_high=z.filter(pl.col('history_risk')>t).height,both_high=z.filter((pl.col('history_risk')>t)&(pl.col('evaluation_risk')>t)).height))
    (ROOT/'report.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
