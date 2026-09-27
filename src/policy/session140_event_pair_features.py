\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json,time,hashlib
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from sequence_features import augment
import build_outcome_roles as BOR,build_relationship_evidence as BRE
from session46_paired_inference import action_design
from session51_matchup_inference import current_from_design

ROOT=Path('artifacts/pair_session140_event_features');P=Path('artifacts/policy');C=pl.col
CFG=json.load(open('artifacts/evidence_session37_bet_fold/config.json'))
NEST=Path('artifacts/evidence_session55_current_nested')
FOLDS=json.load(open(P/'table_folds.json'))
HALVES=json.load(open('artifacts/evidence_session10/nested6/table_half_split.json'))

def design(table,players):
    ids=players.select('pair_id')
    d=pl.read_parquet(f'artifacts/detail_features/{table}.parquet').filter(C('phase')=='development').join(ids,on='pair_id',how='semi')
    h=pl.read_parquet(P/'hand_features'/f'{table}.parquet').filter(C('phase')=='development').join(ids,on='pair_id',how='semi').sort('pair_id','time_index')
    rc=[c for c in h.columns if c.endswith('_r')]
    h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rc]);hz=[c for c in h.columns if c.endswith('_hz')]
    h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz])
    add=[c for c in h.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']]
    d=d.join(h.select('pair_id','hand_id',*add),on=['pair_id','hand_id'],validate='1:1');d,_=augment(d)
    query=d.select('pair_id','hand_id').join(players,on='pair_id',validate='m:1')
    d=d.join(BOR.build(table,query),on=['pair_id','hand_id'],validate='1:1').join(BRE.build(table,query),on=['pair_id','hand_id'],validate='1:1')
    d=d.with_columns((C('time')/.6).alias('relative_time')).sort('pair_id','time','hand_id').with_row_index('row')
    assert not d.select(CFG['hand_columns']).null_count().to_numpy().any()
    a,x,_,_=action_design(d,players,CFG)
    x=np.column_stack([x,current_from_design(a,x,CFG)])
    assert x.shape[1]==929 and np.isfinite(x).all()
    a=a.select('pair_id','hand_id','action_no','actor').join(h.select('pair_id','hand_id','time_index'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
    return d.select('pair_id','hand_id','time'),a,x

def replay():
    from session55_current_targets import state
    ROOT.mkdir(exist_ok=True);s=state();labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');records=[]
                                                                              
    tables=[sorted(set(s['directed_transfer']['d'].filter(C('fold')==f)['table_id']))[0] for f in range(4)]
    for table in tables:
        ids=pl.concat([v['d'].filter(C('table_id')==table).select('pair_id') for v in s.values()]).unique()
        players=labs.join(ids,on='pair_id',how='semi');_,a,x=design(table,players)
        for fam in ['directed_transfer','soft_play']:
            v=s[fam];d=v['d'];ix=v['a'].join(d.select('row','table_id'),on='row',validate='m:1').filter(C('table_id')==table)
            if not len(ix):continue
            z=a.with_row_index('new_row').join(ix.select('pair_id','hand_id','action_no','action_row'),on=['pair_id','hand_id','action_no'],validate='1:1')
            assert len(z)==len(ix)
            expected=v['x'][z['action_row'].to_numpy()];actual=x[z['new_row'].to_numpy()];err=float(np.max(abs(actual-expected)))
            if err>1e-5:
                i,j=np.unravel_index(np.argmax(abs(actual-expected)),actual.shape)
                names=CFG['fold_columns']+CFG['hand_columns']+CFG['paired_columns']+['current_comparison','rank_gap']
                raise AssertionError((table,fam,err,names[j],float(actual[i,j]),float(expected[i,j])))
            records.append(dict(table=table,family=fam,actions=len(z),feature_max_error=err))
        print('REPLAY',table,records[-2:],flush=True)
    (ROOT/'raw_replay.json').write_text(json.dumps(records,indent=2))

def model(outer,native,half,family):
    if outer==native:
        p=Path('artifacts/evidence_session50_matchup/current')/family/f'event1_fold{outer}.cbm'
    else:p=NEST/f'outer{outer}_parent{native}_half{half}'/('direct_primary.cbm' if family=='directed_transfer' else 'soft_primary.cbm')
    m=CatBoostClassifier();m.load_model(str(p));return m,p

def main():
    from session55_current_targets import state
    ROOT.mkdir(exist_ok=True);assert (ROOT/'raw_replay.json').exists()
    reference=state()
    labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2')
    bags=pl.read_parquet('artifacts/evidence_session115_relationship_data/bags.parquet');start=time.time();seen=set();audit=[];mc={}
    for table in sorted(set(bags['table_id'])):
        players=labs.join(bags.filter(C('table_id')==table).select('pair_id'),on='pair_id',how='semi');path=ROOT/f'{table}.parquet'
        if path.exists():
            old=json.load(open(path.with_suffix('.json')));seen.update(old['pairs']);audit.append(old);continue
        d,a,x=design(table,players);native=FOLDS[table];half=HALVES.get(table,0);pred={};refs=[];errors={}
        for fam in ['directed_transfer','soft_play']:
            v=reference[fam];old=v['a'].join(v['d'].select('row','table_id'),on='row',validate='m:1').filter(C('table_id')==table)
            if not len(old):continue
            z=a.with_row_index('new_row').join(old.select('pair_id','hand_id','action_no','action_row'),on=['pair_id','hand_id','action_no'],validate='1:1')
            assert len(z)==len(old)
            err=float(np.max(abs(x[z['new_row'].to_numpy()]-v['x'][z['action_row'].to_numpy()])))
            assert err<1e-5,(table,fam,err)
            errors[fam]=dict(actions=len(z),max_error=err)
        for outer in range(4):
            for fam,short in [('directed_transfer','direct'),('soft_play','soft')]:
                key=(outer,native,half,fam)
                if key not in mc:mc[key]=model(*key)
                m,p=mc[key];pred[f'{short}_outer{outer}']=m.predict_proba(x,thread_count=2)[:,1] if len(a) else np.zeros(0)
                refs.append(dict(outer=outer,family=fam,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        a.with_columns(*[pl.Series(k,v) for k,v in pred.items()]).write_parquet(path)
        info=dict(table=table,native_fold=native,half=half,half_fallback=table not in HALVES,pairs=players['pair_id'].to_list(),hands=len(d),actions=len(a),models=refs,positive_feature_replay=errors)
        path.with_suffix('.json').write_text(json.dumps(info,indent=2));seen.update(players['pair_id']);audit.append(info)
        if len(audit)%20==0:print('EVENT_FEATURES',len(audit),len(seen),round(time.time()-start,1),flush=True)
    assert seen==set(labs['pair_id'])
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));print('COMPLETE',len(seen),sum(q['actions'] for q in audit),time.time()-start,flush=True)

if __name__=='__main__':
    import sys
    replay() if '--replay' in sys.argv else main()
