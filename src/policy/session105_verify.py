import json
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
import session105_rollout_heads as s
C=pl.col

def main():
    q=pl.read_parquet(s.ROOT/'queries.parquet');cfg=json.load(open(s.ROOT/'config.json'))
    assert cfg['rollout_config']['provenance_sha256']==s.roll.provenance()
                                                                                          
    reference=np.zeros((4,len(q),len(s.KINDS)*len(s.roll.FIELDS)));seen=np.zeros((4,len(q)),int)
    for (table,),local in q.group_by('table_id'):
        h=int(local['fold'][0]);ix=local['query_id'].to_numpy()
        for f in range(4):
            if f==h:continue
            a,b=sorted([f,h]);z=pl.read_parquet(s.roll.ROOT/'tables'/f'{table}_exclude{a}{b}.parquet')
            z=local.select('query_id').join(z,on='query_id',validate='1:1',maintain_order='left');v=z.select(cfg['fields']).to_numpy()
            reference[f,ix]=v;seen[f,ix]+=1;reference[h,ix]+=v/3;seen[h,ix]+=1
    for f in range(4):
        np.testing.assert_array_equal(seen[f],np.where(q['fold'].to_numpy()==f,3,1))
        np.testing.assert_array_equal(reference[f],np.load(s.ROOT/f'features_fold{f}.npz')['x'])
    base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet')
    models=0;mutations=0;scalar_checks=0;scalar_error=0.;aligned=0
    for kind,v in s.inputs().items():
        d,a=v['d'],v['a'];g=a['row'].to_numpy();fv=d['fold'].to_numpy();actor=None if kind=='isolation' else a['actor'].to_numpy()
                                                                                      
        np.testing.assert_array_equal(d.select('pair_id','hand_id').to_numpy()[g],a.select('pair_id','hand_id').to_numpy())
        aligned+=len(a)
        for arm in s.KINDS:
            saved=pl.read_parquet(s.ROOT/arm/'event_oof.parquet');pred=np.zeros((len(d),2))
            for f in range(4):
                dm=d.with_columns(*[pl.when(C('fold')==f).then(pl.lit(val)).otherwise(C(col)).alias(col)
                                    for col,val in [('evidence',0),('evidence_rank',-999),('subtype',-999)]])
                if kind=='direct_secondary':
                    y,tr,donor=s.calls.targets(d,f);ym,tm,donorm=s.calls.targets(dm,f)
                    for x,xx in [(y,ym),(tr,tm),(donor,donorm)]:np.testing.assert_array_equal(x,xx)
                    va=fv==f;assert not(tr&va).any();mutations+=1
                    ex=s.secondary_features(v,f,arm);raw=s.extra(a,kind,f,arm);bags=2*g+actor
                    selected=np.random.default_rng(105).choice(np.unique(bags),min(128,len(np.unique(bags))),replace=False)
                    for bag in selected:
                        z=raw[bags==bag];want=np.r_[z.mean(0),z.max(0)];got=ex[bag//2,bag%2]
                        scalar_error=max(scalar_error,float(abs(want-got).max()));np.testing.assert_allclose(want,got,atol=1e-12,rtol=1e-12);scalar_checks+=1
                    m=CatBoostClassifier();m.load_model(str(s.ROOT/arm/f'{kind}_fold{f}_head0_em0.cbm'));assert m.tree_count_==400
                    for r in range(2):
                        x=np.column_stack([v['x'][va],v['ax'][va,r],ex[va,r]]);p=m.predict_proba(x,thread_count=2)[:,1]
                        np.testing.assert_array_equal(p,m.predict_proba(x[::-1],thread_count=2)[:,1][::-1]);pred[va,r]=p
                    models+=1;continue
                x=np.column_stack([v['x'],s.extra(a,kind,f,arm)]);nh=2 if kind=='isolation' else 1
                vm=dict(v,d=dm)
                if kind=='isolation':
                    vm['full']=v['full'].with_columns(*[pl.when(C('fold')==f).then(pl.lit(val)).otherwise(C(col)).alias(col)
                                                       for col,val in [('evidence',0),('evidence_rank',-999),('subtype',-999)]])
                for head in range(nh):
                    y,tr,va,bag=s.training(v,kind,f,head);ym,tm,vam,bagm=s.training(vm,kind,f,head)
                    for one,two in [(y,ym),(tr,tm),(va,vam)]:np.testing.assert_array_equal(one,two)
                    assert not(tr&va).any();mutations+=1
                    assert not set(d['table_id'].to_numpy()[g[tr]])&set(d['table_id'].to_numpy()[g[va]])
                    em=2 if bag is not None else 0;m=CatBoostClassifier();m.load_model(str(s.ROOT/arm/f'{kind}_fold{f}_head{head}_em{em}.cbm'));assert m.tree_count_==400
                    p=m.predict_proba(x[va],thread_count=2)[:,1];np.testing.assert_array_equal(p,m.predict_proba(x[va][::-1],thread_count=2)[:,1][::-1]);models+=1
                    if kind=='isolation':pred[fv==f,head]=s.iso.noisy_or(p,g[va],len(d))[fv==f]
                    else:pred[g[va],actor[va]]=p
            expected=d.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],maintain_order='left',validate='1:1')
            if kind=='isolation':np.testing.assert_array_equal(pred,expected.select('bg_primary','bg_secondary').to_numpy())
            else:
                p=(pred*(s.donor_weights(d) if kind.startswith('direct') else 1)).sum(1)
                np.testing.assert_array_equal(p,expected['bg_secondary' if kind=='direct_secondary' else 'bg_primary'].to_numpy())
            if kind=='soft_primary':
                old=d.select('pair_id','hand_id').join(base,on=['pair_id','hand_id'],maintain_order='left',validate='1:1')
                np.testing.assert_array_equal(expected['bg_secondary'],old['bg_secondary'])
    report={'outer_feature_matrices_reconstructed':4,'aligned_action_queries':aligned,'final_Cat_models_replayed':models,
            'heldout_target_mutations':mutations,'secondary_scalar_pools_checked':scalar_checks,'secondary_scalar_max_error':scalar_error,
            'event_aggregation_error':0,'Cat_query_reversal_error':0,'soft_secondary_unchanged':True,
            'intermediate_EM_models':'saved for reproduction; final40 predictors audited through actual aggregation'}
    assert models==40 and mutations==40
    (s.ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
