import gc, hashlib, json
import numpy as np
import polars as pl
import torch
from catboost import CatBoostClassifier
import session98_tabicl_events as s

C = pl.col

def main(replay_folds=None):
    assert torch.backends.mps.is_available()
    torch.set_num_threads(2)
    torch.mps.set_per_process_memory_fraction(.65)
    assert hashlib.file_digest(s.CHECKPOINT.open('rb'), 'sha256').hexdigest() == s.SHA
    states = s.state()
    feature_count = json.loads((s.ROOT/'config.json').read_text())['features']
    base = pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet')
    audit = {(v['family'], v['fold']): v for v in json.loads((s.ROOT/'fit_audit.json').read_text())}
    results = []
    for fam in ['directed_transfer', 'soft_play']:
        v = states[fam]
        d, a = v['d'], v['a']
        g, actor = a['row'].to_numpy(), a['actor'].to_numpy()
        pp = {k: np.zeros((len(d), 2)) for k in ['original', 'cat128', 'tabicl']}
        for f in range(4):
            z = np.load(s.ROOT/f'{fam}_fold{f}_context.npz')
            dm = d.with_columns(*[pl.when(C('fold')==f).then(pl.lit(val)).otherwise(C(col)).alias(col)
                                  for col,val in [('evidence',0),('evidence_rank',-999),('subtype',-999)]])
            if fam == 'directed_transfer':
                y,tr,va = s.labels(d,a,f)
                ym,tm,vm = s.labels(dm,a,f)
            else:
                yy,e,_ = s.target(d,f)
                yy2,e2,_ = s.target(dm,f)
                y,ym,tr,tm = yy[g],yy2[g],e[g],e2[g]
                va = vm = v['fv'][g]==f
            for one,two in [(y,ym),(tr,tm),(va,vm)]:
                np.testing.assert_array_equal(one,two)
            assert not (tr&va).any()
            assert not set(d['table_id'].to_numpy()[g[tr]]) & set(d['table_id'].to_numpy()[g[va]])
            np.testing.assert_array_equal(z['training_rows'],np.flatnonzero(tr))
            np.testing.assert_array_equal(z['query_rows'],np.flatnonzero(va))
            np.testing.assert_array_equal(z['y_train'],y[tr])
            path = s.Path('artifacts/evidence_session50_matchup/current')/fam/f'event1_fold{f}.cbm'
            assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==audit[fam,f]['selector_model_sha256']
            assert hashlib.file_digest((s.ROOT/f'{fam}_fold{f}_context.npz').open('rb'),'sha256').hexdigest()==audit[fam,f]['context_sha256']
            original = CatBoostClassifier()
            original.load_model(str(path))
            imp = original.get_feature_importance(thread_count=2)
            selected = np.sort(np.lexsort((np.arange(len(imp)),-imp))[:feature_count])
            np.testing.assert_array_equal(selected,z['selected_columns'])
            x = v['x'][:,selected].astype(np.float32)
            np.testing.assert_array_equal(x[tr],z['x_train'])
            np.testing.assert_array_equal(x[va],z['x_query'])
            p = original.predict_proba(v['x'][va],thread_count=2)[:,1]
            np.testing.assert_array_equal(p,np.load(s.ROOT/f'{fam}_fold{f}_original.npy'))
            pp['original'][g[va],actor[va]] = p
            m = CatBoostClassifier()
            m.load_model(str(s.ROOT/'cat128'/f'{fam}_fold{f}.cbm'))
            p = m.predict_proba(x[va],thread_count=2)[:,1]
            np.testing.assert_array_equal(p,np.load(s.ROOT/'cat128'/f'{fam}_fold{f}_actions.npy'))
            np.testing.assert_array_equal(p,m.predict_proba(x[va][::-1],thread_count=2)[:,1][::-1])
            pp['cat128'][g[va],actor[va]] = p
            del original,m
            gc.collect()
            torch.mps.empty_cache()
            saved = np.load(s.ROOT/'tabicl'/f'{fam}_fold{f}_actions.npy')
            assert np.isfinite(saved).all() and ((saved>=0)&(saved<=1)).all()
            if replay_folds is not None and f not in replay_folds:
                pp['tabicl'][g[va],actor[va]] = saved
                continue
            m = s.estimator(f)
            m.fit(z['x_train'],z['y_train'])
            p = m.predict_proba(z['x_query'])[:,1]
            permuted = m.predict_proba(z['x_query'][::-1])[:,1][::-1]
                                                                           
            r = {'family':fam,'fold':f,'queries':len(p),
                 'replay_max_error':float(abs(p-saved).max()),
                 'query_permutation_max_error':float(abs(p-permuted).max()),
                 'context_and_targets_exact':True,'heldout_pool_overlap':0}
            results.append(r)
            (s.ROOT/'verification_progress.json').write_text(json.dumps(results,indent=2))
            print(r,flush=True)
            assert r['replay_max_error'] < 1e-5
            assert r['query_permutation_max_error'] < 1e-5
            np.save(s.ROOT/'tabicl'/f'{fam}_fold{f}_replayed_actions.npy',p)
            pp['tabicl'][g[va],actor[va]] = saved
            del m,x
            gc.collect()
            torch.mps.empty_cache()
        dw = d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if fam=='directed_transfer' else np.ones((len(d),2))
        for kind,p in pp.items():
            saved = base if kind=='original' else pl.read_parquet(s.ROOT/kind/'event_oof.parquet')
            expected = d.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['bg_primary'].to_numpy()
            np.testing.assert_array_equal((p*dw).sum(1),expected)
            paired = saved.join(base,on=['pair_id','hand_id'],validate='1:1',suffix='_base')
            np.testing.assert_array_equal(paired['bg_secondary'],paired['bg_secondary_base'])
            iso = states['coordinated_isolation']['d']['pair_id'].unique()
            q = paired.filter(C('pair_id').is_in(iso.implode()))
            np.testing.assert_array_equal(q['bg_primary'],q['bg_primary_base'])
    report = {'records':results,'all_contexts_reconstructed':8,'target_mutations':8,
              'cat_models_replayed':16,'tabicl_contexts_replayed':len(results),
              'original_primary_aggregation_exact':True,'secondary_isolation_unchanged':True,
              'checkpoint_sha256':s.SHA,'query_permutation_and_replay_errors_measured':True}
    report.update(max_replay_error=max(v['replay_max_error'] for v in results),
                  max_permutation_error=max(v['query_permutation_max_error'] for v in results),
                  replay_tolerance=1e-5,replay_within_tolerance=True)
    (s.ROOT/'verification.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
