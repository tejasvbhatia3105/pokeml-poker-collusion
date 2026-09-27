\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from session8_data import hand_data,reference
from session8_exposure_training import data as windows_data
from session4_evidence_model import load_models as load_base,score as base_score,COLS
from session6_priority_model import load_models as load_priority,score as priority_score
from session7_model import load_models as load_event,score as event_score
ROOT=Path('artifacts/evidence_session9');C=pl.col;FAMILIES=['directed_transfer','soft_play','coordinated_isolation']
def r29_score(models,b,q,f):
    base,prior,event,cols=models
    q=q.with_columns(pl.lit(b).alias('behavior_family'))
    return .25*base_score(base,b,q.select(COLS).to_numpy(),[f])+.25*priority_score(prior,cols,b,q,[f])+.5*event_score(event,cols,b,q,[f])
def main():
    full=hand_data();_,window,_=windows_data();window=window.filter(C('window')!='full').join(full.select('pair_id','hand_id','table_id'),on=['pair_id','hand_id'],validate='m:1');full=full.join(reference().select('pair_id','hand_id','r29'),on=['pair_id','hand_id'],validate='1:1');window=window.join(pl.read_parquet('artifacts/evidence_session8/window_r29_predictions.parquet'),on=['pair_id','hand_id','window'],validate='1:1');base=load_base();prior,cols=load_priority('priority_ordered');event,_=load_event('hist_eventblend');models=(base,prior,event,cols)
    rc=json.loads(Path('artifacts/policy/residual_columns.json').read_text());oc=json.loads(Path('artifacts/rank_columns.json').read_text());sc=json.loads(Path('artifacts/policy/sequence/evidence_columns.json').read_text());v4=pl.read_csv('artifacts/policy/residual_oof.csv');v4=v4.select('pair_id').with_columns(pl.Series('v4_family',[FAMILIES[i] for i in v4.select(FAMILIES).to_numpy().argmax(1)]));oldmodels={};oldpair={};rows=[];parts=[]
    def fallback(q,b,f):
        key=(b,f)
        if key not in oldmodels:
            a=CatBoostClassifier();a.load_model(f'artifacts/rank_{b}_fold{f}.cbm');z=CatBoostClassifier();z.load_model(f'artifacts/policy/sequence/evidence_{b}_fold{f}.cbm');oldmodels[key]=(a,z)
        a,z=oldmodels[key];return .25*a.predict_proba(q.select(oc).to_numpy(),thread_count=4)[:,1]+.75*z.predict_proba(q.select(sc).to_numpy(),thread_count=4)[:,1]
    with threadpool_limits(limits=4):
        for w in ['full','first_2000','last_2000']:
            q=full if w=='full' else window.filter(C('window')==w);pair=pl.read_parquet(ROOT/f'pair_{w}.parquet').filter(C('label')==1).select('pair_id','risk_score',C('family').alias('routed_family'));q=q.join(pair,on='pair_id',validate='m:1');q=q.with_columns(C('r29').alias('routed_score'),C('r29').alias('ungated_score'));local=[]
            for (pid,),g in q.group_by('pair_id'):
                f=int(g['fold'][0]);b=g['routed_family'][0];truth_b=g['behavior_family'][0];risk=g['risk_score'][0];routed=g['r29'].to_numpy().copy();ungated=routed.copy();old_b=None
                if b!=truth_b:ungated=r29_score(models,b,g,f)
                routed=ungated.copy()
                if risk<.05:
                    if w=='full':old_b=v4.filter(C('pair_id')==pid)['v4_family'][0]
                    else:
                        if f not in oldpair:m=CatBoostClassifier();m.load_model(f'artifacts/policy/residual_fold{f}.cbm');oldpair[f]=m
                        table=g['table_id'][0];pf=pl.read_parquet(f'artifacts/policy/full_window_stress/{w}/pair_features/{table}.parquet').filter(C('pair_id')==pid);prob=oldpair[f].predict_proba(pf.select(rc).to_numpy(),thread_count=4)[0,1:];old_b=FAMILIES[int(prob.argmax())]
                    routed=fallback(g,old_b,f)
                local.append(g.select('pair_id','hand_id','evidence','fold','r29').with_columns(pl.lit(w).alias('window'),pl.Series('routed_score',routed),pl.Series('ungated_score',ungated)))
                truth=set(g.filter(C('evidence')==1)['hand_id']);den=min(5,len(truth))
                if not den:continue
                row={'pair_id':pid,'table_id':g['table_id'][0],'window':w,'fold':f,'family':truth_b,'routed_family':b,'risk_score':float(risk),'below_gate':bool(risk<.05),'fallback_family':old_b}
                for name,pred in [('oracle_r29',g['r29'].to_numpy()),('routed_r29',routed),('ungated_r29',ungated)]:
                    order=np.lexsort((np.array(g['hand_id']),-pred))[:5];hit=np.array([h in truth for h in np.array(g['hand_id'])[order]]);row[name]=float((hit*np.cumsum(hit)/np.arange(1,len(hit)+1)).sum()/den)
                rows.append(row)
            parts.extend(local)
    r=pl.DataFrame(rows);r.write_csv(ROOT/'routed_evidence.csv');pl.concat(parts).write_parquet(ROOT/'routed_hand_scores.parquet');names=['oracle_r29','routed_r29','ungated_r29'];report={'windows':r.group_by('window').agg(pl.len(),C(names).mean(),C('below_gate').sum()).sort('window').to_dicts(),'routing_buckets':r.group_by('window','below_gate').agg(pl.len(),C(names).mean()).to_dicts(),'family':r.group_by('window','family').agg(C(names).mean()).to_dicts(),'caveat':__doc__};(ROOT/'routed_evidence.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
