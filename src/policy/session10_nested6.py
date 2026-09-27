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
ROOT=Path('artifacts/evidence_session10/nested6');ROOT.mkdir(exist_ok=True);C=pl.col
def cat(seed,**kwargs):return CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False,**kwargs)
def hist(seed):return HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=10,max_bins=127,early_stopping=False,random_state=seed)
def main():
    d=hand_data();cfg=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'));X=d.select(cfg['event']).to_numpy();XT=d.select(cfg['type']).to_numpy();fv=d['fold'].to_numpy();family=d['behavior_family'].to_numpy();y=d['evidence'].to_numpy();sub=d['subtype'].to_numpy();t=d['time'].to_numpy();ranks=d['evidence_rank'].fill_null(0).to_numpy();groups=[g['row'].to_numpy() for _,g in d.group_by('pair_id',maintain_order=True)];half={}
    for f in range(4):
        tables=sorted(d.filter(C('fold')==f)['table_id'].unique());order=np.random.default_rng(1110+f).permutation(len(tables));half.update({tables[k]:i%2 for i,k in enumerate(order)})
    hv=np.array([half[t] for t in d['table_id']]);(ROOT/'table_half_split.json').write_text(json.dumps(half,indent=2));lex=np.lexsort((d['hand_id'].to_numpy(),d['pair_id'].to_numpy()));start=time.time()
    with threadpool_limits(limits=4):
        for outer in map(int,os.environ.get('NESTED_OUTERS','0,1,2,3').split(',')):
            final=ROOT/f'nested_outer{outer}.parquet'
            if final.exists():continue
            pieces=[]
            for parent in map(int,os.environ.get('NESTED_PARENTS','0,1,2,3').split(',')):
                if parent==outer:continue
                for h in range(2):
                    path=ROOT/f'teacher_outer{outer}_parent{parent}_half{h}.parquet';audit=[];held=(fv==parent)&(hv==h);eligible=(fv!=outer)&~held
                    if path.exists():pieces.append(pl.read_parquet(path));continue
                    pred=np.full((len(d),5),np.nan)
                    for b in ['directed_transfer','soft_play','coordinated_isolation']:
                        tr=eligible&(family==b);va=held&(family==b);trsort=lex[tr[lex]];m=cat(1710+outer*10+parent,loss_function='Logloss');m.fit(X[trsort],y[trsort]);bc=m.predict_proba(X[va],thread_count=4)[:,1];m=hist(4710+outer*10+parent);m.fit(X[tr],y[tr]);pred[va,0]=.5*(bc+m.predict_proba(X[va])[:,1]);known=tr&(sub>0);typ=CatBoostClassifier(iterations=200,depth=3,learning_rate=.04,l2_leaf_reg=10,random_seed=6210+parent,thread_count=4,verbose=False,allow_writing_files=False);typ.fit(XT[known],(sub[known]==1).astype(int));tp=typ.predict_proba(XT,thread_count=4)[:,1];a,bb,ea,eb=training_targets(y,sub,tp,tr,t,groups,ranks)
                        for k,target,mask in [(1,a,ea),(2,bb,eb)]:
                            assert not np.any(mask&((fv==outer)|held));m=cat(6310+11*parent+k);m.fit(X[mask],target[mask]);pred[va,k]=m.predict_proba(X[va],thread_count=4)[:,1];m=hist(4710+parent);m.fit(X[mask],target[mask]);pred[va,2+k]=m.predict_proba(X[va])[:,1]
                            audit.append({'outer':outer,'inner_parent':parent,'inner_half':h,'family':b,'head':k,'training_hands':int(mask.sum()),'training_tables':sorted(set(d['table_id'].to_numpy()[mask])),'prediction_tables':sorted(set(d['table_id'].to_numpy()[va]))})
                    assert np.isfinite(pred[held]).all();part=d.filter(pl.Series(held)).select('pair_id','hand_id','fold','time').with_columns(*[pl.Series(n,pred[held,i]) for i,n in enumerate(['base','cat_primary','cat_secondary','hist_primary','hist_secondary'])]);part.write_parquet(path);pieces.append(part);path.with_suffix('.json').write_text(json.dumps(audit,indent=2));print('nested6',outer,parent,h,'seconds',round(time.time()-start,1),flush=True)
            training=pl.concat(pieces);parts=[]
            if len(training)!=int((fv!=outer).sum()):
                print('nested6 selected inner blocks complete for outer',outer,flush=True)
                continue
            for _,g in training.group_by('pair_id'):
                g=g.sort('time','hand_id');ca=g['cat_primary'].to_numpy();cb=g['cat_secondary'].to_numpy();ha=g['hist_primary'].to_numpy();hb=g['hist_secondary'].to_numpy();cs=np.maximum(1,ca+cb);hs=np.maximum(1,ha+hb);p=inclusion(ca,cb);hp=inclusion(.5*(ca/cs+ha/hs),.5*(cb/cs+hb/hs));parts.append(g.with_columns(pl.Series('cat_inclusion',p),pl.Series('joint_inclusion',hp),pl.Series('r29',.25*g['base'].to_numpy()+.25*p+.5*hp)))
            val=pl.read_parquet(f'artifacts/evidence_session9/nested_outer{outer}.parquet').filter(C('fold')==outer);combined=pl.concat(parts+[val]);assert len(combined)==len(d) and combined.select('pair_id','hand_id').n_unique()==len(d);combined.write_parquet(final)
    print('nested6 complete',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
