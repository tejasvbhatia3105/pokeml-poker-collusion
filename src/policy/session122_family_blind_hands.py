\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json,hashlib
from pathlib import Path
import numpy as np
import polars as pl
import session119_private_partner_data as source
ROOT=Path('artifacts/evidence_session122_family_blind_hands');C=pl.col
GROUPS=['all','fold','postflop_fold','call','postflop_call','raise','postflop_raise']

def aggregate(x,bag,selected,n):
    xx=x[selected];bb=bag[selected];counts=np.bincount(bb,minlength=n);s=np.zeros((n,x.shape[1]),np.float64)
    lo=np.full_like(s,np.inf);hi=np.full_like(s,-np.inf)
    np.add.at(s,bb,xx);np.minimum.at(lo,bb,xx);np.maximum.at(hi,bb,xx)
    mean=s/np.maximum(counts,1)[:,None];mean[counts==0]=-2;lo[counts==0]=-2;hi[counts==0]=-2
    return np.column_stack([np.log1p(counts),mean,lo,hi]).astype(np.float32),counts

def main():
    ROOT.mkdir(exist_ok=True);d=source.source().filter(C('label')==1).sort('pair_id','hand_id').with_row_index('hand_index')
    assert len(d)==45129 and d['pair_id'].n_unique()==372
    frames=[];arrays=[]
    for a in json.load(open(source.ROOT/'audit.json'))['tables']:
        table=a['table'];meta=pl.read_parquet(source.ROOT/f'{table}.parquet').with_row_index('local')
        z=meta.join(d.select(C('row').alias('hand_row'),'hand_index'),on='hand_row',how='inner',validate='m:1')
        if not len(z):continue
        x=np.load(source.ROOT/f'{table}.npz')['x'][z['local'].to_numpy()];frames.append(z);arrays.append(x)
    meta=pl.concat(frames);x=np.concatenate(arrays);bag=meta['hand_index'].to_numpy();cls=meta['action_class'].to_numpy();post=x[:,source.PUBLIC.index('street_no')]>0
    masks=[np.ones(len(x),bool),cls==0,(cls==0)&post,cls==2,(cls==2)&post,cls==3,(cls==3)&post]
    cols=[];parts=[];rawcols=source.PUBLIC+source.PRIVATE;checks=0
    for name,mask in zip(GROUPS,masks):
        block,counts=aggregate(x,bag,mask,len(d));parts.append(block)
        cols+=[name+'_log_count']+[name+'_'+stat+'_'+c for stat in ['mean','min','max'] for c in rawcols]
                                                                 
        for i in np.random.default_rng(12201).choice(len(d),128,replace=False):
            z=x[mask&(bag==i)]
            expected=np.r_[np.log1p(len(z)),z.mean(0,dtype=np.float64),z.min(0),z.max(0)] if len(z) else np.r_[0,np.full(3*x.shape[1],-2)]
            np.testing.assert_allclose(block[i],expected,rtol=1e-6,atol=1e-5);checks+=1
    h=pl.read_parquet('artifacts/evidence_session115_relationship_data/hands.parquet').select('pair_id','hand_id','time_index')
    d=d.join(h,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')
    chronology=np.zeros((len(d),3),np.float32)
    for _,g in d.group_by('pair_id'):
        g=g.sort('time_index','hand_id');ix=g['hand_index'].to_numpy();chronology[ix]=np.column_stack([g['time_index'].to_numpy()/3000,np.arange(len(g))/max(1,len(g)-1),np.full(len(g),np.log1p(len(g)))])
    parts.append(chronology);cols+=['phase_position','pair_hand_position','log_pair_hand_count'];out=np.column_stack(parts)
    assert out.shape==(len(d),len(cols)) and np.isfinite(out).all() and len(set(cols))==len(cols)
                                                                    
    truth=pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id','evidence_rank')
    d=d.join(truth,on=['pair_id','hand_id'],how='left',validate='1:1',maintain_order='left').with_columns(C('evidence_rank').is_not_null().cast(pl.Int8).alias('evidence'))
    assert d['evidence'].sum()==1817
    np.save(ROOT/'x.npy',out);d.write_parquet(ROOT/'hands.parquet')
    config=dict(method=__doc__,columns=cols,source_config=json.load(open(source.ROOT/'config.json')),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        x_sha256=hashlib.sha256((ROOT/'x.npy').read_bytes()).hexdigest(),teacher_predictions=False,ids_as_features=False,fold_specific_normalization=False)
    (ROOT/'config.json').write_text(json.dumps(config,indent=2));(ROOT/'verification.json').write_text(json.dumps(dict(hand_rows=len(d),action_rows=len(meta),features=len(cols),scalar_group_checks=checks,all_truth_preserved=True),indent=2))
    print('complete',out.shape,'action_rows',len(meta),'checks',checks)

if __name__=='__main__':main()
