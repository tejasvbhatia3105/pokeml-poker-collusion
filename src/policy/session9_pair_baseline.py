\
\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
ROOT=Path('artifacts/evidence_session9');C=pl.col
N=['directed_transfer','soft_play','coordinated_isolation'];SEQ=['seq_v6','seq_v6s1','seq_v6s2','seq_v7','seq_v7s1','seq_v7s2','seq_v8']
def normalize(v):
    total=v.sum(1,keepdims=True);return np.divide(v,total,out=np.full_like(v,1/3),where=total>0)
def assemble(g1,g2,sequences):
    q=g1.select('pair_id',*[(1-C('none')).alias('g1')],*[C(n).alias('a_'+n) for n in N]).join(g2.select('pair_id',(1-C('none')).alias('g2'),*[C(n).alias('b_'+n) for n in N]),on='pair_id',validate='1:1')
    for name,p in sequences.items():q=q.join(p.select('pair_id',(1-C('none')).alias(name),*([C(n).alias('seq_'+n) for n in N] if name=='seq_v6' else [])),on='pair_id',validate='1:1')
                                                                               
    g=np.sqrt(np.clip(q['g1'].to_numpy(),1e-6,1)*np.clip(q['g2'].to_numpy(),1e-6,1));gclass=normalize(q.select(['a_'+n for n in N]).to_numpy()+q.select(['b_'+n for n in N]).to_numpy())*g[:,None];fam=normalize(gclass+q.select(['seq_'+n for n in N]).to_numpy());lg=np.log(np.clip(q.select(SEQ).to_numpy(),1e-300,1));s=np.exp((lg[:,:3].mean(1)+lg[:,3:6].mean(1)+.5*lg[:,6])/2.5);risk=np.sqrt(g*s)
    return q.select('pair_id').with_columns(pl.Series('risk_score',risk),pl.Series('gbdt_risk',g),pl.Series('seq_risk',s),pl.Series('family',[N[i] for i in fam.argmax(1)]),*[pl.Series(n,risk*fam[:,i]) for i,n in enumerate(N)])
def main():
    seq={name:pl.read_csv(f'artifacts/{name}/eval_all_lpo.csv') for name in SEQ};check=assemble(pl.read_csv('artifacts/candidate_r12/eval_r4s_all.csv'),pl.read_csv('artifacts/candidate_r12/eval_r4k_all.csv'),seq).join(pl.read_csv('artifacts/candidate_r29/submission.csv').select('pair_id',C('risk_score').alias('expected_risk'),C('predicted_behavior').alias('expected_family')),on='pair_id',validate='1:1');error=(check['risk_score']-check['expected_risk']).abs().max();wrong=check.filter((C('risk_score')>=.01)&(C('family')!=C('expected_family'))).height;assert error<1e-12 and wrong==0;report={'evaluation_risk_max_error':error,'evaluation_family_disagreements_above_none_cutoff':wrong,'windows':[]}
    labs=pl.read_csv('data/development_labels.csv');positive_players=set(labs.filter(C('label')==1)['player_1'])|set(labs.filter(C('label')==1)['player_2']);truth=pl.read_parquet('artifacts/evidence_session4/hand_index.parquet').filter(C('evidence')==1)
    for w,lo,hi in [('full',0,3000),('first_2000',0,2000),('last_2000',1000,3000)]:
        q=assemble(pl.read_csv(ROOT/f'r4s_{w}.csv'),pl.read_csv(ROOT/f'r4k_{w}.csv'),{n:pl.read_csv(f'artifacts/{n}/dev_{w}_lpo.csv') for n in SEQ});meta=pl.read_csv(f'artifacts/seq_v6/dev_{w}.csv').select('pair_id','player_1','player_2','n_hands');q=q.join(meta,on='pair_id',validate='1:1').join(labs.select('pair_id','label',C('behavior_family').alias('true_family')),on='pair_id',how='left',validate='1:1').with_columns(C('label').fill_null(-1));count=truth.filter((C('time')*5000>=lo)&(C('time')*5000<hi)).group_by('pair_id').len().rename({'len':'n_truth'});q=q.join(count,on='pair_id',how='left',validate='1:1').with_columns(C('n_truth').fill_null(0));q.write_parquet(ROOT/f'pair_{w}.parquet')
        r=q.sort(['risk_score','pair_id'],descending=[True,False]).with_row_index('rank');p=r.filter((C('label')==1)&(C('n_truth')>0));bad=r.filter(C('label')==0);unknown=r.filter(C('label')==-1);other=np.sort(r.filter(C('label')!=1)['risk_score'].to_numpy());above=len(other)-np.searchsorted(other,p['risk_score'].to_numpy(),side='right');hard=np.array([(a in positive_players)!=(b in positive_players) for a,b in r.select('player_1','player_2').iter_rows()])&(r['label'].to_numpy()==-1)
        report['windows'].append({'window':w,'pairs':len(q),'positive_pairs_with_listed_truth':len(p),'positive_pairs_below_evidence_gate':p.filter(C('risk_score')<.05).height,'wrong_family_positive_pairs':p.filter(C('family')!=C('true_family')).height,'recall_at_300_other_pairs':float(np.mean(above<300)),'recall_at_1000_other_pairs':float(np.mean(above<1000)),'confirmed_negatives_top1000':bad.filter(C('rank')<1000).height,'assumed_spillover_negatives_top1000':int((hard&(r['rank'].to_numpy()<1000)).sum()),'unknown_pairs_top1000':unknown.filter(C('rank')<1000).height})
    report['caveat']='Whole-pool OOF pair recipe replay; ranking proxies do not label unknown pairs as confirmed negatives. Window truth is cropped public annotation. Archived teachers have historical training/selection limitations.';(ROOT/'pair_baseline.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
