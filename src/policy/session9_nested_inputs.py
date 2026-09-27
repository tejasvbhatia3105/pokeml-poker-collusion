\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from session8_data import hand_data
from session6_priority import training_targets,inclusion
ROOT=Path('artifacts/evidence_session9');C=pl.col
FAMILIES=['directed_transfer','soft_play','coordinated_isolation']
def main():
    d=hand_data();cfg=json.loads(Path('artifacts/evidence_session6/priority_ordered_columns.json').read_text());X=d.select(cfg['event']).to_numpy();XT=d.select(cfg['type']).to_numpy();fv=d['fold'].to_numpy();fam=d['behavior_family'].to_numpy();sub=d['subtype'].to_numpy();y=d['evidence'].to_numpy();rank=d['evidence_rank'].fill_null(0).to_numpy();t=d['time'].to_numpy();groups=[g['row'].to_numpy() for _,g in d.group_by('pair_id',maintain_order=True)];start=time.time();audit=[]
    hist=pl.read_parquet('artifacts/evidence_session7/hist_events_oof.parquet');cat=pl.read_parquet('artifacts/evidence_session6/priority_ordered_oof.parquet')
    with threadpool_limits(limits=4):
        for outer in range(4):
            path=ROOT/f'nested_outer{outer}.parquet'
            if path.exists():continue
            pred=np.full((len(d),2),np.nan)
            for inner in range(4):
                if inner==outer:continue
                for b in FAMILIES:
                    tr=(fv!=outer)&(fv!=inner)&(fam==b);va=(fv==inner)&(fam==b);known=tr&(sub>0)
                    typ=CatBoostClassifier(iterations=200,depth=3,learning_rate=.04,l2_leaf_reg=10,thread_count=4,random_seed=6210+inner,verbose=False,allow_writing_files=False);typ.fit(XT[known],(sub[known]==1).astype(int));tp=typ.predict_proba(XT,thread_count=4)[:,1];a,bt,ea,eb=training_targets(y,sub,tp,tr,t,groups,rank)
                    for head,target,eligible in [(1,a,ea),(2,bt,eb)]:
                        assert not np.any(eligible&((fv==outer)|(fv==inner)))
                        m=HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=10,max_bins=127,early_stopping=False,random_state=4710+inner);m.fit(X[eligible],target[eligible]);pred[va,head-1]=m.predict_proba(X[va])[:,1]
                        audit.append({'outer':outer,'inner':inner,'family':b,'head':head,'training_hands':int(eligible.sum()),'training_folds':sorted(np.unique(fv[eligible]).tolist())})
                print('nested R29 outer',outer,'inner',inner,'seconds',round(time.time()-start,1),flush=True)
            ph=d.select('pair_id','hand_id','fold').with_columns(pl.Series('primary',pred[:,0]),pl.Series('secondary',pred[:,1])).filter(C('fold')!=outer).drop('fold');ph=pl.concat([ph,hist.join(d.select('pair_id','hand_id','fold'),on=['pair_id','hand_id'],validate='1:1').filter(C('fold')==outer).select('pair_id','hand_id','primary','secondary')])
            pc=pl.read_parquet(f'artifacts/evidence_session7/calibration_nested_outer{outer}.parquet').drop('fold');pc=pl.concat([pc,cat.join(d.select('pair_id','hand_id','fold'),on=['pair_id','hand_id'],validate='1:1').filter(C('fold')==outer).select('pair_id','hand_id','primary','secondary')])
            q=d.select('pair_id','hand_id','fold','time').join(pl.read_parquet(f'artifacts/evidence_session4/nested_blend/nested_outer{outer}.parquet').rename({'base_score':'base'}),on=['pair_id','hand_id'],validate='1:1').join(pc.rename({'primary':'cat_primary','secondary':'cat_secondary'}),on=['pair_id','hand_id'],validate='1:1').join(ph.rename({'primary':'hist_primary','secondary':'hist_secondary'}),on=['pair_id','hand_id'],validate='1:1');assert len(q)==len(d) and np.isfinite(q.select(pl.selectors.numeric()).to_numpy()).all();parts=[]
            for _,g in q.group_by('pair_id'):
                g=g.sort('time','hand_id');ca=g['cat_primary'].to_numpy();cb=g['cat_secondary'].to_numpy();ha=g['hist_primary'].to_numpy();hb=g['hist_secondary'].to_numpy();cs=np.maximum(1,ca+cb);hs=np.maximum(1,ha+hb);cp=inclusion(ca,cb);hp=inclusion(.5*ca/cs+.5*ha/hs,.5*cb/cs+.5*hb/hs);parts.append(g.with_columns(pl.Series('cat_inclusion',cp),pl.Series('joint_inclusion',hp),pl.Series('r29',.25*g['base'].to_numpy()+.25*cp+.5*hp)))
            pl.concat(parts).write_parquet(path);(ROOT/f'nested_outer{outer}_audit.json').write_text(json.dumps([r for r in audit if r['outer']==outer],indent=2))
    print('nested R29 inputs complete',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
