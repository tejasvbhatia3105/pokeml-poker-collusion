\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostRanker,Pool
from session8_data import hand_data
from session10_list_learning import features,ROOT
CONFIG={'loss_function':'YetiRank:mode=MAP;top=5','iterations':500,'depth':5,'learning_rate':.035,'l2_leaf_reg':8,'thread_count':4,'seed_base':1010,'shortlist':20,'baseline':'frozen nested R29 log odds','selection':'fixed compact and full input comparison; no early stopping or grid'}
def main():
    path=ROOT/'ranker_config.json'
    if path.exists():assert json.loads(path.read_text())==CONFIG
    else:path.write_text(json.dumps(CONFIG,indent=2))
    d=hand_data();basecols=json.loads(Path('artifacts/evidence_session6/priority_ordered_columns.json').read_text())['event'];parts=[];start=time.time();audit=[]
    for f in range(4):
        q=d.join(pl.read_parquet(f'artifacts/evidence_session9/nested_outer{f}.parquet').drop('fold','time'),on=['pair_id','hand_id'],validate='1:1');bags=[]
        for (pid,),g in q.group_by('pair_id'):
            g=g.sort('time','hand_id');x,_=features(g);s=g['r29'].to_numpy();ix=np.lexsort((g['hand_id'].to_numpy(),-s))[:20];bags.append((pid,g[ix],x[ix]))
        bags.sort(key=lambda z:z[0]);g=pl.concat([z[1] for z in bags]);compact=np.concatenate([z[2] for z in bags]);full=np.column_stack([compact,g.select(basecols).to_numpy()]);s=np.clip(g['r29'].to_numpy(),1e-5,1-1e-5);prior=np.log(s)-np.log1p(-s);y=g['evidence'].to_numpy();tr=g['fold'].to_numpy()!=f;va=~tr;group=g['pair_id'].to_numpy();out=g.filter(pl.Series(va)).select('pair_id','hand_id','fold','evidence','r29').with_columns(pl.Series('r29_logit',prior[va]))
        for name,X in [('compact',compact),('full',full)]:
            assert np.isfinite(X).all();cfg={k:CONFIG[k] for k in ['loss_function','iterations','depth','learning_rate','l2_leaf_reg','thread_count']};m=CatBoostRanker(**cfg,random_seed=CONFIG['seed_base']+f,allow_writing_files=False,verbose=False);pool=Pool(X[tr],y[tr],group_id=group[tr],baseline=prior[tr]);m.fit(pool);pred=prior[va]+m.predict(X[va]);m.save_model(str(ROOT/f'ranker_{name}_fold{f}.cbm'));out=out.with_columns(pl.Series('ranker_'+name,pred));audit.append({'fold':f,'kind':name,'features':X.shape[1],'train_pairs':len(set(group[tr])),'valid_pairs':len(set(group[va]))});print('ranker',f,name,round(time.time()-start,1),flush=True)
        parts.append(out)
    pl.concat(parts).write_parquet(ROOT/'ranker_oof.parquet');(ROOT/'ranker_audit.json').write_text(json.dumps(audit,indent=2));(ROOT/'ranker_hand_columns.json').write_text(json.dumps(basecols))
if __name__=='__main__':main()
