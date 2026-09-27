import json,sys
from pathlib import Path
import numpy as np
import polars as pl
import session108_action_frontier as f
C=pl.col

def scalar(root,values,member):
    \
    n=values.shape[2];half=n//2;own=root['own'];other=member['other'];out=[]
    for arm in range(2):
        v=values[:,arm]/max(1.,root['pot']);result=np.zeros(12)
        for choose,evaluate in [(range(half),range(half,n)),(range(half,n),range(half))]:
            objectives=[]
            for mode in range(3):
                scores=[]
                for action in range(len(v)):
                    scores.append(sum((v[action,r,own] if mode==0 else v[action,r,own]+v[action,r,other] if mode==1 else v[action,r,other]) for r in choose)/half)
                maximum=max(scores);best=next(k for k,score in enumerate(scores) if maximum-score<=1e-10);objectives.append(best)
            deltas=[]
            for best in objectives:
                deltas.append([sum(v[0,r,j]-v[best,r,j] for r in evaluate)/half for j in [own,other]])
            a,b,c=deltas
            result+=np.array([-a[0],a[1],sum(a),-sum(b),b[0],b[1],-c[1],c[0],sum(c),*[float(k==0) for k in objectives]])/2
        out.append(result)
    return np.array(out)

def replay(use_cache=True,backend=None):
    q=pl.read_parquet(f.ROOT/'queries.parquet');chosen=[]
    for native in range(4):
        for table in q.filter(C('fold')==native)['table_id'].unique().sort():
            paths=[f.ROOT/'tables'/f'{table}_exclude{min(native,h)}{max(native,h)}.json' for h in range(4) if h!=native]
            complete=all(p.exists() for p in paths)
            matching=complete and (backend is None or all(json.loads(p.read_text()).get('backend','python')==backend for p in paths))
            if not use_cache or matching:
                chosen.append((native,table));break
    assert len(chosen)==4
    reports=[];scalar_error=0.;scalars=0;roots_count=0
    for native,table in chosen:
        local=q.filter(C('table_id')==table);keys=[]
        for street in range(4):
            z=local.filter(C('street_no')==street).select('hand_id','action_no').unique().sort('hand_id','action_no').head(1)
            if len(z):keys.append(z)
        small=local.join(pl.concat(keys),on=['hand_id','action_no'],how='semi');roots=f.s.roots_for_table(table,small,64);roots_count+=len(roots)
        for root in roots:
            menu=f.menu(root);st=root['state'];j=root['own'];call=st.to_call(j)
            assert menu[0]==(root['forced_class'],root['forced_amount']) and len(menu)==len(set(menu))
            assert ((0,0.) if call>0 else (1,0.)) in menu
            if call>0:assert (2,float(call)) in menu
            for k,amount in menu:assert not st.clone().apply(j,k,amount)
        for other in range(4):
            if other==native:continue
            action,size,meta=f.s.models(native,other);z,info,values,offset=f.evaluate(roots,action,size,meta,64)
            if use_cache:
                a,b=sorted([native,other]);cached=pl.read_parquet(f.ROOT/'tables'/f'{table}_exclude{a}{b}.parquet')
                expected=z.select('query_id').join(cached,on='query_id',maintain_order='left',validate='1:1')
                np.testing.assert_array_equal(z.to_numpy(),expected.select(z.columns).to_numpy())
            original,_=f.s.simulate(roots,action,size,meta,64)
            np.testing.assert_array_equal(values[np.array(offset[:-1])],original[:,[0,2]])
            rev,_,_,_=f.evaluate(roots[::-1],action,size,meta,64)
            np.testing.assert_array_equal(z.sort('query_id').to_numpy(),rev.sort('query_id').to_numpy())
            lookup={row['query_id']:row for row in z.to_dicts()}
            for i,root in enumerate(roots):
                for member in root['members']:
                    want=scalar(root,values[offset[i]:offset[i+1]],member)
                    got=np.array([[lookup[member['query_id']][arm+'_'+name] for name in f.FIELDS] for arm in ['learned','checkcall']])
                    scalar_error=max(scalar_error,float(abs(want-got).max()));np.testing.assert_allclose(want,got,rtol=1e-12,atol=1e-12);scalars+=1
            reports.append(dict(native=native,other_excluded_fold=other,table=table,roots=len(roots),cache_error=0 if use_cache else None,forced_arm_parity_error=0,query_reversal_error=0,**info))
    report=dict(reference_blocks=len(reports),roots=roots_count,scalar_query_reference_checks=scalars,scalar_max_error=scalar_error,backend_selection=backend,records=reports)
    filename='native_cache_verification.json' if backend=='native' else 'replay_verification.json' if use_cache else 'prebuild_verification.json'
    (f.ROOT/filename).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

def coverage():
    q=pl.read_parquet(f.ROOT/'queries.parquet');np.testing.assert_array_equal(q.to_numpy(),f.s.queries().to_numpy())
    cfg=json.loads((f.ROOT/'config.json').read_text());assert cfg==dict(f.CONFIG,provenance_sha256=f.provenance())
    refs=json.load(open('artifacts/evidence_session84_nested_joint_policy/reference_audit.json'));refmap={tuple(r['excluded_folds']):r for r in refs}
    folds=json.load(open('artifacts/policy/table_folds.json'));records=[];identity_error=0.
    for (table,),local in q.group_by('table_id'):
        native=int(local['fold'][0]);assert native==folds[table]
        for outer in range(4):
            if outer==native:continue
            a,b=sorted([native,outer]);path=f.ROOT/'tables'/f'{table}_exclude{a}{b}.parquet'
            audit=json.loads(path.with_suffix('.json').read_text());assert audit['config']==cfg
            assert audit['native_fold']==native and audit['other_excluded_fold']==outer
            if audit.get('backend')=='native':
                import session111_native_rollout as backend
                assert audit['backend_provenance']==backend.provenance()
                for name in ['pilot.json','raw_verification.json']:
                    assert json.loads((backend.ROOT/name).read_text())['provenance']==audit['backend_provenance']
            meta=json.load(open(f'artifacts/evidence_session103_nested_sizes/metadata_exclude{a}{b}.json'))
            for source in [meta,refmap[a,b]]:
                assert table not in source['training_tables'];assert all(folds[t] not in [a,b] for t in source['training_tables'])
            z=pl.read_parquet(path).sort('query_id');np.testing.assert_array_equal(z['query_id'],local.sort('query_id')['query_id']);assert np.isfinite(z.drop('query_id').to_numpy()).all()
            for arm in ['learned','checkcall']:
                v=lambda name:z[arm+'_'+name].to_numpy()
                for err in [v('team_gain_own')-v('partner_gain_own')+v('own_regret'),v('team_regret')+v('own_gain_team')+v('partner_gain_team'),v('team_gain_partner')-v('own_gain_partner')+v('partner_regret')]:
                    identity_error=max(identity_error,float(abs(err).max()))
                rates=z.select([arm+'_actual_'+name+'_rate' for name in ['own','team','partner']]).to_numpy()
                assert np.isin(rates,[0,.5,1]).all()
            assert audit['chip_conservation_error']<1e-8;records.append(audit)
    assert identity_error<1e-10 and len(records)==3*q['table_id'].n_unique()
    report=dict(query_rows=len(q),reference_rows=sum(r['query_rows'] for r in records),reference_blocks=len(records),
        provenance_hashes=len(cfg['provenance_sha256']),model_fold_exclusions=True,team_identity_max_error=identity_error,
        trajectories=sum(r['trajectories'] for r in records),simulated_actions=sum(r['actions'] for r in records),
        chip_conservation_max_error=max(r['chip_conservation_error'] for r in records))
    assert report['reference_rows']==3*len(q)
    (f.ROOT/'coverage_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':{'replay':replay,'coverage':coverage,'prebuild':lambda:replay(False),'replay_native':lambda:replay(True,'native')}[sys.argv[1]]()
