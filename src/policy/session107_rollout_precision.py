\
\
\
\
\
\
import json,time,sys
from pathlib import Path
import numpy as np
import polars as pl
from catboost import CatBoostClassifier
import session104_allstreet_rollout as s
from session106_focal_integration import integrate

ROOT=Path('artifacts/evidence_session107_rollout_precision');C=pl.col

def main():
    ROOT.mkdir(exist_ok=True);q=pl.read_parquet(s.ROOT/'queries.parquet')
    rng=np.random.default_rng(10701);selected=[]
    for f in range(4):
        tables=q.filter(C('fold')==f)['table_id'].unique().sort().to_list()
        for table in rng.choice(tables,3,replace=False):
            local=q.filter(C('table_id')==table);keys=[]
            for street in range(4):
                k=local.filter(C('street_no')==street).select('hand_id','action_no').unique().sort('hand_id','action_no')
                if len(k):keys.append(k[rng.choice(len(k),min(2,len(k)),replace=False)])
            z=local.join(pl.concat(keys),on=['hand_id','action_no'],how='semi')
            selected.append(z)
    selection=pl.concat(selected);selection.write_parquet(ROOT/'selection.parquet')
    recover='recover-report' in sys.argv
    records=pl.read_parquet(ROOT/'precision.parquet').to_dicts() if recover else []
    audits=[];start=time.time()
    if recover:
        assert len(records)==6*len(selection)
        assert set(r['query_id'] for r in records)==set(selection['query_id'].to_list())
        for local in selected:
            audits.append(dict(table_id=local['table_id'][0],roots=local.select('hand_id','action_no').n_unique(),queries=len(local),
                note='Recovered report from all 12 completed pool outputs; original exact-prefix checks passed, see session107_precision.log. Detailed trajectory counters were not persisted before the reporting failure.'))
    for local in ([] if recover else selected):
        table=local['table_id'][0];f=int(local['fold'][0]);h=(f+1)%4
        roots=s.roots_for_table(table,local,256);action,size,meta=s.models(f,h)
        raw,rb,info=integrate(roots,action,size,meta,256)
        a,b=sorted([f,h]);cached=pl.read_parquet(s.ROOT/'tables'/f'{table}_exclude{a}{b}.parquet')
        small=s.values(roots,raw[:,:,:32]).sort('query_id')
        expected=small.select('query_id').join(cached,on='query_id',maintain_order='left',validate='1:1')
        np.testing.assert_array_equal(small.drop('query_id').to_numpy(),expected.select(small.columns[1:]).to_numpy())
        for i,r in enumerate(roots):
            for member in r['members']:
                own,other=r['own'],member['other'];den=max(1.,r['pot'])
                for arm,j,k in [('learned',0,1),('checkcall',2,3)]:
                    x=(raw[i,j]-raw[i,k])/den;y=(rb[i,j]-rb[i,k])/den
                    xx=np.column_stack([-x[:,own],x[:,other],x[:,own]+x[:,other]])
                    yy=np.column_stack([-y[:,own],y[:,other],y[:,own]+y[:,other]])
                    for c,name in enumerate(['own_loss','partner_gain','team_gain']):
                        v,w=xx[:,c],yy[:,c]
                        records.append(dict(query_id=member['query_id'],table_id=table,fold=f,street=r['state'].street,arm=arm,field=name,
                            sampled32=float(v[:32].mean()),sampled256=float(v.mean()),integrated256=float(w.mean()),
                            sampled32_variance=float(v.var(ddof=1)/32),integrated256_variance=float(w.var(ddof=1)/256),
                            focal_variance_ratio=float(w.var(ddof=1)/v.var(ddof=1)) if v.var()>1e-20 else None))
        audits.append(dict(table_id=table,fold=f,reference=[a,b],roots=len(roots),queries=len(local),cached32_replay_error=0,**info))
        pl.DataFrame(records).write_parquet(ROOT/'precision.parquet')
        print('PRECISION_POOL',table,len(roots),time.time()-start,flush=True)
    z=pl.DataFrame(records);report=[]
    for group in [[],['arm'],['street','arm'],['arm','field']]:
        groups=[((),z)] if not group else list(z.group_by(group))
        for key,d in groups:
            x=d['sampled32'].to_numpy();y=d['integrated256'].to_numpy()
            report.append(dict(group=dict(zip(group,key)),n=len(d),
                rmse32_vs_integrated256=float(np.sqrt(np.mean((x-y)**2))),
                correlation32_vs_integrated256=float(np.corrcoef(x,y)[0,1]),
                mean_sampled32_variance=d['sampled32_variance'].mean(),
                mean_integrated256_variance=d['integrated256_variance'].mean(),
                median_focal_variance_ratio=d['focal_variance_ratio'].median(),
                sign_disagreement=float(np.mean(x*y<0))))
    importance=[]
    root=Path('artifacts/evidence_session105_rollout_heads')
    for arm in ['checkcall','learned']:
        for kind in ['direct_primary','soft_primary','direct_secondary','isolation']:
            for f in range(4):
                for h in range(2 if kind=='isolation' else 1):
                    em=2 if kind=='isolation' else 0;m=CatBoostClassifier()
                    m.load_model(str(root/arm/f'{kind}_fold{f}_head{h}_em{em}.cbm'))
                    imp=m.feature_importances_;n=12 if kind=='direct_secondary' else 6
                    available=imp is not None and np.ndim(imp)==1
                    importance.append(dict(arm=arm,kind=kind,fold=f,head=h,
                        new_feature_importance_percent=float(imp[-n:].sum()) if available else None,
                        individual=imp[-n:].tolist() if available else None,
                        note='Stored importance unavailable for this checkpoint' if not available else 'Stored prediction-value-change importance'))
    out=dict(method=__doc__,pools=len(audits),roots=sum(a['roots'] for a in audits),queries=len(selection),replicates=256,
             reporting_recovery=recover,seconds=time.time()-start,summary=report,pool_audits=audits,model_importance=importance,
             caveat='Correlated query/field records are descriptive only. 256 integrated paths are not exact EV; models may be misspecified.')
    (ROOT/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
