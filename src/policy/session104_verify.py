import copy,hashlib,json,sys
from pathlib import Path
import numpy as np
import polars as pl
import session104_allstreet_rollout as s
C=pl.col

def replay():
    q=pl.read_parquet(s.ROOT/'queries.parquet');chosen=[]
    for f in range(4):
        for table in sorted(q.filter(C('fold')==f)['table_id'].unique()):
            paths=[s.ROOT/'tables'/f'{table}_exclude{min(f,h)}{max(f,h)}.json' for h in range(4) if h!=f]
            if all(p.exists() for p in paths):chosen.append((f,table));break
    assert len(chosen)==4
    records=[];outcome_cases=0
    for native,table in chosen:
        local=q.filter(C('table_id')==table).sort('street_no','query_id')
        keys=local.select('hand_id','action_no').unique(maintain_order=True).head(2)
        small=local.join(keys,on=['hand_id','action_no'],how='semi')
        roots=s.roots_for_table(table,small,32)
        original=s.engine.table_data
        def mutate(t,needed):
            raw,meta,seats=original(t,needed);meta=copy.deepcopy(meta)
            inv={v:k for k,v in s.CARD.items()}
            for hid,ss in seats.items():
                st=int(small.filter(C('hand_id')==hid)['street_no'].min());nb=0 if st==0 else st+2
                visible=meta[hid]['board_cards'].split()[:nb]
                used=set(ss['hole_card_1'])|set(ss['hole_card_2'])|set(visible)
                future=[inv[c] for c in range(51,-1,-1) if inv[c] not in used][:5-nb]
                meta[hid]['board_cards']=' '.join(visible+future);meta[hid]['final_pot']+=777
                seats[hid]=ss.with_columns((C('net_chips')+777).alias('net_chips'))
            return raw,meta,seats
        try:
            s.engine.table_data=mutate;altered=s.roots_for_table(table,small,32)
        finally:s.engine.table_data=original
        assert len(roots)==len(altered)
        for a,b in zip(roots,altered):
            for name in ['boards','templates','ranks']:np.testing.assert_array_equal(a[name],b[name])
            for name in ['starting','contribution','committed','alive','pending','raise_right']:np.testing.assert_array_equal(getattr(a['state'],name),getattr(b['state'],name))
            outcome_cases+=1
        for f in range(4):
            if f==native:continue
            action,size,meta=s.models(f,native);value,info=s.simulate(roots,action,size,meta,32);z=s.values(roots,value)
            a,b=sorted([f,native]);saved=pl.read_parquet(s.ROOT/'tables'/f'{table}_exclude{a}{b}.parquet')
            expected=z.select('query_id').join(saved,on='query_id',validate='1:1',maintain_order='left')
            np.testing.assert_array_equal(z.to_numpy(),expected.select(z.columns).to_numpy())
            reverse,_=s.simulate(roots[::-1],action,size,meta,32);np.testing.assert_array_equal(value,reverse[::-1])
            records.append({'native_fold':native,'other_excluded_fold':f,'table':table,'roots':len(roots),'cached_value_error':0,'root_reversal_error':0,**info})
    report={'reference_blocks_replayed':len(records),'future_board_outcome_mutation_roots':outcome_cases,'future_outcome_error':0,'records':records}
    (s.ROOT/'replay_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

def coverage():
    q=pl.read_parquet(s.ROOT/'queries.parquet');np.testing.assert_array_equal(q.to_numpy(),s.queries().to_numpy())
    cfg=json.loads((s.ROOT/'config.json').read_text());assert cfg['provenance_sha256']==s.provenance()
    refs=json.load(open('artifacts/evidence_session84_nested_joint_policy/reference_audit.json'))
    refmap={tuple(r['excluded_folds']):r for r in refs};folds=json.load(open('artifacts/policy/table_folds.json'));records=[];identity_error=0.
    for (table,),local in q.group_by('table_id'):
        native=int(local['fold'][0]);assert native==folds[table]
        for f in range(4):
            if f==native:continue
            a,b=sorted([f,native]);path=s.ROOT/'tables'/f'{table}_exclude{a}{b}.parquet'
            audit=json.load(open(path.with_suffix('.json')));assert audit['config']==cfg and audit['native_fold']==native and audit['other_excluded_fold']==f
            meta=json.load(open(f'artifacts/evidence_session103_nested_sizes/metadata_exclude{a}{b}.json'))
            for source in [meta,refmap[a,b]]:
                assert table not in source['training_tables']
                assert all(folds[t] not in [a,b] for t in source['training_tables'])
            z=pl.read_parquet(path).sort('query_id');np.testing.assert_array_equal(z['query_id'],local.sort('query_id')['query_id'])
            assert np.isfinite(z.drop('query_id').to_numpy()).all()
            for arm in ['learned','checkcall']:
                error=abs(z[arm+'_team_gain'].to_numpy()-z[arm+'_partner_gain'].to_numpy()+z[arm+'_own_loss'].to_numpy())
                identity_error=max(identity_error,float(error.max()))
                assert (z.select([arm+'_'+k for k in ['own_stderr','partner_stderr','team_stderr']]).to_numpy()>=0).all()
            assert audit['chip_conservation_error']<1e-8
            records.append(audit)
    out={'query_rows':len(q),'tables':q['table_id'].n_unique(),'complete_reference_blocks':len(records),
         'reference_query_rows':sum(r['query_rows'] for r in records),'model_fold_exclusions_verified':True,
         'provenance_hashes_verified':len(cfg['provenance_sha256']),'trajectories':sum(r['trajectories'] for r in records),
         'simulated_actions':sum(r['actions'] for r in records),'chip_conservation_max_error':max(r['chip_conservation_error'] for r in records)}
    out['team_gain_identity_max_error']=identity_error;assert identity_error<1e-10
    assert out['complete_reference_blocks']==3*out['tables'] and out['reference_query_rows']==3*len(q)
    (s.ROOT/'coverage_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':{'replay':replay,'coverage':coverage}[sys.argv[1]]()
