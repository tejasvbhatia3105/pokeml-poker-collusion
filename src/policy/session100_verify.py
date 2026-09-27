import gc,hashlib,json
import numpy as np
import polars as pl
import torch
from catboost import CatBoostClassifier
import session100_isolation_bags as s
C=pl.col

def main():
    assert torch.backends.mps.is_available()
    torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.65)
    full,d,a,x,support,count,ax=s.inputs()
    cached=pl.read_parquet('artifacts/evidence_session57_isolation_pressure/pressure_actions.parquet')
    np.testing.assert_array_equal(a.to_numpy(),cached.to_numpy())
    hc=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event']
    g=a['row'].to_numpy();checked=0
                                                                          
    for row in np.flatnonzero(support):
        z=ax[g==row].astype(np.float64)
        expected=np.r_[d.select(hc).row(int(row)),np.log1p(len(z)),z.mean(0),z.min(0),z.max(0)].astype(np.float32)
        np.testing.assert_array_equal(expected,x[row]);checked+=1
    fv=d['fold'].to_numpy();fm=full['behavior_family'].to_numpy()=='coordinated_isolation'
    records=[];pred={k:np.zeros((len(d),2)) for k in s.KINDS}
    audit={(r['fold'],r['head']):r for r in json.loads((s.ROOT/'fit_audit.json').read_text())}
    for f in range(4):
        old=s.targets(full,f,'coordinated_isolation')
        dm=full.with_columns(*[pl.when(C('fold')==f).then(pl.lit(value)).otherwise(C(col)).alias(col)
                              for col,value in [('evidence',0),('evidence_rank',-999),('subtype',-999)]])
        new=s.targets(dm,f,'coordinated_isolation')
        for one,two in zip(old[:4],new[:4]):np.testing.assert_array_equal(one,two)
        for k in range(2):
            y=old[k][fm];tr=old[2+k][fm]&support;va=(fv==f)&support
            assert not(tr&va).any()
            assert not set(d['table_id'].to_numpy()[tr])&set(d['table_id'].to_numpy()[va])
            path=s.ROOT/f'fold{f}_head{k}_context.npz'
            assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==audit[f,k]['context_sha256']
            z=np.load(path)
            np.testing.assert_array_equal(z['training_rows'],np.flatnonzero(tr))
            np.testing.assert_array_equal(z['query_rows'],np.flatnonzero(va))
            np.testing.assert_array_equal(z['y_train'],y[tr])
            for kind in ['cat_full','cat128']:
                m=CatBoostClassifier();m.load_model(str(s.ROOT/kind/f'fold{f}_head{k}.cbm'))
                if kind=='cat_full':
                    imp=m.get_feature_importance(thread_count=2)
                    selected=np.sort(np.lexsort((np.arange(len(imp)),-imp))[:128])
                    np.testing.assert_array_equal(selected,z['selected_columns'])
                    np.testing.assert_array_equal(x[tr][:,selected],z['x_train'])
                    np.testing.assert_array_equal(x[va][:,selected],z['x_query'])
                    inp=x[va]
                else:inp=z['x_query']
                p=m.predict_proba(inp,thread_count=2)[:,1]
                np.testing.assert_array_equal(p,np.load(s.ROOT/kind/f'fold{f}_head{k}_query.npy'))
                np.testing.assert_array_equal(p,m.predict_proba(inp[::-1],thread_count=2)[:,1][::-1])
                pred[kind][va,k]=p
                del m
            gc.collect();torch.mps.empty_cache()
            m=s.tab.estimator(f);m.fit(z['x_train'],z['y_train'])
            p=m.predict_proba(z['x_query'])[:,1]
            saved=np.load(s.ROOT/'tabicl'/f'fold{f}_head{k}_query.npy')
            perm=m.predict_proba(z['x_query'][::-1])[:,1][::-1]
            r={'fold':f,'head':k,'replay_max_error':float(abs(p-saved).max()),
               'reversal_max_error':float(abs(p-perm).max()),'heldout_overlap':0}
            records.append(r);(s.ROOT/'verification_progress.json').write_text(json.dumps(records,indent=2))
            print(r,flush=True)
            assert r['replay_max_error']<1e-5 and r['reversal_max_error']<1e-5
            pred['tabicl'][va,k]=saved
            del m;gc.collect();torch.mps.empty_cache()
    base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet')
    for kind,pp in pred.items():
        q=pl.read_parquet(s.ROOT/kind/'event_oof.parquet')
        expected=d.select('pair_id','hand_id').join(q,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
        np.testing.assert_array_equal(pp,expected.select('bg_primary','bg_secondary').to_numpy())
        assert (pp[~support]==0).all()
        other=q.join(d.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='anti').join(base,on=['pair_id','hand_id'],validate='1:1',suffix='_base')
        np.testing.assert_array_equal(other.select('bg_primary','bg_secondary').to_numpy(),other.select('bg_primary_base','bg_secondary_base').to_numpy())
    report={'pooled_hands_scalar_replayed':checked,'pressure_actions_match_audited57':len(a),
            'contexts_reconstructed':8,'target_exclusions':8,'cat_models_replayed':16,
            'tabicl_contexts_replayed':8,'records':records,'probability_replay_tolerance':1e-5,
            'nonisolation_heads_unchanged':True,'event_aggregation_exact':True}
    (s.ROOT/'verification.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
