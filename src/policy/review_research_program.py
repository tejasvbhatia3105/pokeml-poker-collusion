\
\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from session8_data import reference
C=pl.col;ROOT=Path('artifacts/research_review')
SEQ=['seq_v6','seq_v6s1','seq_v6s2','seq_v7','seq_v7s1','seq_v7s2','seq_v8']
def loss_accounting():
    d=reference();rows=[]
    for (pid,),g in d.group_by('pair_id'):
        truth=set(g.filter(C('evidence')==1)['hand_id']);den=min(5,len(truth));ranked=g.sort(['r29','hand_id'],descending=[True,False]);hit=np.array([h in truth for h in ranked['hand_id']]);ap=float((hit[:5]*np.cumsum(hit[:5])/np.arange(1,6)).sum()/den);selected=hit[:5].sum()/den
        row={'pair_id':pid,'family':g['behavior_family'][0],'table_id':g['table_id'][0],'fold':int(g['fold'][0]),'n_hands':len(g),'n_truth':den,'map5':ap,'top5_oracle_order':float(selected),'missed_set_loss':float(1-selected),'ordering_loss':float(selected-ap),'worst_true_rank':int(np.flatnonzero(hit).max()+1)}
        for k in [5,10,20,50]:row['recall'+str(k)]=float(hit[:k].sum()/den)
        assert np.isclose(1-ap,row['missed_set_loss']+row['ordering_loss']);rows.append(row)
    r=pl.DataFrame(rows);r.write_csv(ROOT/'loss_accounting.csv');metrics=['map5','top5_oracle_order','missed_set_loss','ordering_loss','recall5','recall10','recall20','recall50'];overall=r.select(C(metrics).mean()).to_dicts()[0];overall['fraction_loss_from_set_membership']=overall['missed_set_loss']/(1-overall['map5'])
    report={'overall':overall,'by_family':r.group_by('family').agg(pl.len(),C(metrics).mean()).to_dicts(),'loss_concentration':{str(n):float((1-r.sort('map5').head(n)['map5']).sum()/(1-r['map5']).sum()) for n in [10,25,50,100]},'by_exposure':r.with_columns(pl.when(C('n_hands')<90).then(pl.lit('<90')).when(C('n_hands')<130).then(pl.lit('90-129')).otherwise(pl.lit('130+')).alias('exposure')).group_by('exposure').agg(pl.len(),C(metrics).mean()).to_dicts()}
    (ROOT/'loss_accounting.json').write_text(json.dumps(report,indent=2));return report
def windows():
    ix=pl.read_parquet('artifacts/evidence_session4/hand_index.parquet');truth=ix.filter(C('evidence')==1);counts=truth.group_by('pair_id').len().rename({'len':'full_truth'});rows=[]
    for name,lo,hi in [('first_2000',0,2000),('last_2000',1000,3000)]:
        z=counts.join(truth.filter((C('time')*5000>=lo)&(C('time')*5000<hi)).group_by('pair_id').len().rename({'len':'window_truth'}),on='pair_id',how='left',validate='1:1').with_columns(C('window_truth').fill_null(0));rows.append({'window':name,'public_positive_pairs':len(z),'retained_truth_nonempty':z.filter(C('window_truth')>0).height,'excluded_empty_truth':z.filter(C('window_truth')==0).height,'full_cap_five_pairs':z.filter(C('full_truth')==5).height,'full_cap_five_loses_listed_hands':z.filter((C('full_truth')==5)&(C('window_truth')<5)).height,'retained_scored_capped_pairs_losing_hands':z.filter((C('full_truth')==5)&(C('window_truth')>0)&(C('window_truth')<5)).height})
    report={'windows':rows,'truth_count_distribution':counts.group_by('full_truth').len().sort('full_truth').to_dicts(),'interpretation':'Cropping a capped list is not re-running the annotation selector. Counts identify potentially incomplete labels, not proved missing private evidence.'};(ROOT/'window_annotation_audit.json').write_text(json.dumps(report,indent=2));return report
def deployment():
    ecols=[f'evidence_hand_{i}' for i in range(1,6)];r29=pl.read_csv('artifacts/candidate_r29/submission.csv');v4=pl.read_csv('artifacts/v4/submission.csv');q=r29.join(v4.select('pair_id',*[C(c).alias('old_'+c) for c in ecols]),on='pair_id',validate='1:1').with_columns(pl.any_horizontal([C(c)!=C('old_'+c) for c in ecols]).alias('changed'));bands=q.with_columns(pl.when(C('risk_score')>=.05).then(pl.lit('R29 gate >=.05')).otherwise(pl.lit('below .05')).alias('band')).group_by('band').agg(pl.len(),C('changed').sum()).to_dicts()
    labels=pl.read_csv('data/development_labels.csv').filter(C('label')==1).select('pair_id','behavior_family');d=labels
    for v in SEQ:d=d.join(pl.read_csv(f'artifacts/{v}/dev_full_lpo.csv').select('pair_id',(1-C('none')).alias(v)),on='pair_id',validate='1:1')
    for name in ['r4s','r4k']:d=d.join(pl.read_csv(f'artifacts/candidate_{name}/pair_oof_allpairs.csv').select('pair_id',(1-C('none')).alias(name)),on='pair_id',validate='1:1')
    lg=np.log(np.clip(d.select(SEQ).to_numpy(),1e-300,1));seq=np.exp((lg[:,:3].mean(1)+lg[:,3:6].mean(1)+.5*lg[:,6])/2.5);ub=np.sqrt(np.sqrt(d['r4s'].to_numpy()*d['r4k'].to_numpy())*seq);d=d.select('pair_id','behavior_family').with_columns(pl.Series('deployment_risk_upper_bound',ub));d.write_csv(ROOT/'positive_gate_upper_bound.csv')
    report={'evaluation_evidence_routing':bands,'full_development_public_positives':len(d),'positives_below_gate_even_before_gbdt_rescoring':int((ub<.05).sum()),'limitation':'Raw r4s/r4k risks yield an upper bound, not an exact replay of R12 all-partner rescoring. No actual pipeline evidence MAP or evaluation-positive recall is inferred from this bound.'};(ROOT/'deployment_routing_audit.json').write_text(json.dumps(report,indent=2));return report
def main():
    ROOT.mkdir(exist_ok=True,parents=True);report={'loss':loss_accounting(),'windows':windows(),'deployment':deployment()};(ROOT/'review_diagnostics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
