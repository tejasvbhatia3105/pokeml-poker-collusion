\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','3');os.environ.setdefault('OMP_NUM_THREADS','3')
import json,time,hashlib
from pathlib import Path
import numpy as np,polars as pl,joblib
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from session8_data import hand_data
from session46_paired_inference import action_design
from session51_matchup_inference import current_from_design
from session25_persistent_actor import actor_features,pair_features
from session63_pressure_inference import design as pressure_design
from session27_donor_call_witness import noisy_or
from session4_evidence_model import load_models as base_models,score as base_score,COLS
from session6_priority import inclusion
from session8_count_conditioning import conditioned
from session11_conditional_family import template

ROOT=Path('artifacts/evidence_session142_expert_transfer');C=pl.col
FAMILIES=['directed_transfer','soft_play','coordinated_isolation']
CFG=json.load(open('artifacts/evidence_session37_bet_fold/config.json'))
PRESSURE_CFG=json.load(open('artifacts/evidence_session59_pressure_equity/config.json'))

def load_models():
    out={'base':base_models(),'fold':[]};paths=[]
    for f in range(4):
        part={}
        for name,path in {
            'actor':f'artifacts/evidence_session25_persistent_actor/actor_fold{f}.cbm',
            'direct_primary':f'artifacts/evidence_session50_matchup/current/directed_transfer/event1_fold{f}.cbm',
            'direct_secondary':f'artifacts/evidence_session25_persistent_actor/oriented/event2_directed_transfer_fold{f}.cbm',
            'soft_primary':f'artifacts/evidence_session50_matchup/current/soft_play/event1_fold{f}.cbm',
            'soft_secondary':f'artifacts/evidence_session6/priority_ordered_event2_soft_play_fold{f}.cbm',
            'iso_primary':f'artifacts/evidence_session59_pressure_equity/event1_fold{f}_em2.cbm',
            'iso_secondary':f'artifacts/evidence_session59_pressure_equity/event2_fold{f}_em2.cbm',
        }.items():
            m=CatBoostClassifier();m.load_model(path);part[name]=m;paths.append(path)
        for fam in FAMILIES:
            part['hist_'+fam]=[joblib.load(f'artifacts/evidence_session7/hist_event{k}_{fam}_fold{f}.joblib') for k in [1,2]]
            paths.extend([f'artifacts/evidence_session7/hist_event{k}_{fam}_fold{f}.joblib' for k in [1,2]])
            paths.extend([f'artifacts/evidence_session4/base_{fam}_fold{f}.cbm',f'artifacts/evidence_session4/leafwise_separate_{FAMILIES.index(fam)}_fold{f}.joblib'])
        out['fold'].append(part)
    out['hashes']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths};return out

def minimums(d):
    result={}
    for f in range(4):
        result[f]={}
        for fam in FAMILIES:
            counts=[]
            for _,g in d.filter((C('fold')!=f)&(C('behavior_family')==fam)).group_by('pair_id'):
                g=g.sort('time','hand_id');r=g['evidence_rank'].fill_null(0).to_numpy();ix=np.flatnonzero(r>0);ix=ix[np.argsort(r[ix])]
                if len(template(len(g),ix)):counts.append(len(ix))
            result[f][fam]=min(counts)
    return result

def predictions(d,players,models):
    d=d.sort('pair_id','time','hand_id').drop('row',strict=False).with_row_index('row');f=int(d['fold'][0]);assert d['fold'].n_unique()==1;m=models['fold'][f]
                                                                                   
    a,x,_,hx=action_design(d,players,CFG);x=np.column_stack([x,current_from_design(a,x,CFG)])
    AX=actor_features(d,players);groups=[g['row'].to_numpy() for _,g in d.group_by('pair_id',maintain_order=True)];PX=pair_features(AX,groups)
    pr=m['actor'].predict_proba(PX.reshape(-1,PX.shape[-1]),thread_count=2)[:,1].reshape(-1,2);pr/=pr.sum(1)[:,None];dw=np.zeros((len(d),2))
    for ix,p in zip(groups,pr):dw[ix]=p
    rows=a['row'].to_numpy();actors=a['actor'].to_numpy();p=np.zeros((len(d),2));sp=np.zeros(len(d))
    if len(a):
        p[rows,actors]=m['direct_primary'].predict_proba(x,thread_count=2)[:,1]
        sp[rows]=m['soft_primary'].predict_proba(x,thread_count=2)[:,1]
    dp=(p*dw).sum(1);ds=np.column_stack([m['direct_secondary'].predict_proba(np.column_stack([hx,AX[:,r]]),thread_count=2)[:,1] for r in range(2)]);ds=(ds*dw).sum(1)
    ss=m['soft_secondary'].predict_proba(hx,thread_count=2)[:,1]
    pa,px,_=pressure_design(d,players,PRESSURE_CFG);g=pa['row'].to_numpy()
    ip=[noisy_or(m[name].predict_proba(px,thread_count=2)[:,1] if len(pa) else np.empty(0),g,len(d)) for name in ['iso_primary','iso_secondary']]
    cats={'directed_transfer':np.column_stack([dp,ds]),'soft_play':np.column_stack([sp,ss]),'coordinated_isolation':np.column_stack(ip)}
    out={}
    for fam in FAMILIES:
        out[fam]=dict(cat=cats[fam],hist=np.column_stack([h.predict_proba(hx)[:,1] for h in m['hist_'+fam]]),base=base_score(models['base'],fam,hx,[f]))
    return d,out,dict(fold_actions=len(a),pressure_actions=len(pa))

def main(pilot=False):
    ROOT.mkdir(exist_ok=True);d=hand_data();models=load_models();mins=minimums(d);players=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2')
    assert CFG['hand_columns']==COLS
    ref=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet').select('pair_id','hand_id','bg_primary','bg_secondary')
    old=pl.concat([pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').filter(C('fold')==f).select('pair_id','hand_id','base','hist_primary','hist_secondary') for f in range(4)])
    ref=ref.join(old,on=['pair_id','hand_id'],validate='1:1')
    tables=sorted(set(d['table_id']))
    if pilot:tables=[sorted(set(d.filter(C('fold')==f)['table_id']))[0] for f in range(4)]
    start=time.time();audit=[]
    with threadpool_limits(limits=3):
        for table in tables:
            path=ROOT/f'{table}.parquet'
            if path.exists():audit.append(json.load(open(path.with_suffix('.json'))));continue
            local=d.filter(C('table_id')==table);truth=local.select('pair_id','hand_id','behavior_family')
                                                                               
            clean=local.drop('evidence','evidence_rank','subtype','behavior_family');q,prob,counts=predictions(clean,players,models)
            q=q.join(truth,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');f=int(q['fold'][0]);parts=[];errors=[]
            for fam in FAMILIES:
                v=prob[fam];z=q.select('pair_id','hand_id','table_id','fold','time',C('behavior_family').alias('true_family')).with_columns(pl.lit(fam).alias('expert'),pl.Series('primary',v['cat'][:,0]),pl.Series('secondary',v['cat'][:,1]),pl.Series('hist_primary',v['hist'][:,0]),pl.Series('hist_secondary',v['hist'][:,1]),pl.Series('base',v['base']))
                matched=z.filter(C('true_family')==fam).join(ref,on=['pair_id','hand_id'],validate='1:1',suffix='_ref')
                if len(matched):
                    err=max(float((matched[a]-matched[b]).abs().max()) for a,b in [('primary','bg_primary'),('secondary','bg_secondary'),('base','base_ref'),('hist_primary','hist_primary_ref'),('hist_secondary','hist_secondary_ref')]);assert err<1e-10,(table,fam,err);errors.append(dict(family=fam,hands=len(matched),error=err))
                for _,g in z.group_by('pair_id'):
                    g=g.sort('time','hand_id');ca=g.select('primary','secondary').to_numpy();ca=ca/np.maximum(1,ca.sum(1))[:,None];hp=g.select('hist_primary','hist_secondary').to_numpy();hp=hp/np.maximum(1,hp.sum(1))[:,None];jp=.5*(ca+hp);ci=inclusion(*ca.T);ji=inclusion(*jp.T);jc=conditioned(jp,mins[f][fam]);parts.append(g.with_columns(pl.Series('raw_score',.25*g['base'].to_numpy()+.25*ci+.5*ji),pl.Series('conditioned_score',.25*g['base'].to_numpy()+.25*ci+.5*jc)))
            pl.concat(parts).write_parquet(path);record=dict(table=table,fold=f,hands=len(q),expert_rows=len(q)*3,matched_replay=errors,**counts);path.with_suffix('.json').write_text(json.dumps(record,indent=2));audit.append(record)
            if len(audit)%20==0 or pilot:print('EXPERT_TRANSFER',len(audit),table,round(time.time()-start,1),flush=True)
    output='pilot.json' if pilot else 'audit.json';(ROOT/output).write_text(json.dumps(dict(records=audit,minimums=mins,model_hashes=models['hashes'],method=__doc__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2));print('COMPLETE',len(audit),round(time.time()-start,1),flush=True)

if __name__=='__main__':
    import sys
    main('--pilot' in sys.argv)
