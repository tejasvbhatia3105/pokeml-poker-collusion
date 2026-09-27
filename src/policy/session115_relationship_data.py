\
\
\
\
\
\
\
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import polars as pl

ROOT=Path('artifacts/evidence_session115_relationship_data');C=pl.col

def swapname(c):
    m={'surp_sum_1':'surp_sum_2','surp_alive_1':'surp_alive_2','surp_facing_1':'surp_facing_2','n_act_1':'n_act_2','n_facing_1':'n_facing_2','surp_max_1':'surp_max_2','net1_bb':'net2_bb','put1_l':'put2_l','f1':'f2','sd1':'sd2','stk1_l':'stk2_l','eq1':'eq2'}
    m.update({v:k for k,v in list(m.items())})
    if c.startswith('p1_'):return 'p2_'+c[3:]
    if c.startswith('p2_'):return 'p1_'+c[3:]
    return m.get(c,c)

def main():
    ROOT.mkdir(exist_ok=True);start=time.time();cfg=json.loads(Path('artifacts/seq_v7/config.json').read_text())
    t2=Path(cfg['tokens2']);t1=t2.parent/'seq_tokens'
    cols=json.loads((t1/'columns.json').read_text())+json.loads((t2/'columns2.json').read_text())
    perm=[cols.index(swapname(c)) for c in cols];assert [perm[i] for i in perm]==list(range(len(cols)))
    assert not set(cols)&{'pair_id','hand_id','player_id','table_id','label','evidence','evidence_rank','subtype','behavior_family'}
    labels=pl.read_csv('data/development_labels.csv');lm={r['pair_id']:r for r in labels.to_dicts()}
    evidence=pl.read_csv('data/development_evidence.csv');truth=set(evidence.select('pair_id','hand_id').iter_rows());oldhands=set(evidence['hand_id'])
    folds=json.loads(Path('artifacts/policy/table_folds.json').read_text());arrays=[];rows=[];bags=[];total=0;seen=set();audit=[]
    for table in sorted(folds):
        with np.load(t1/f'{table}.npz') as z:
            offset=np.r_[0,np.cumsum(z['n'])];x1=np.load(t1/f'{table}_X.npy',mmap_mode='r');x2=np.load(t2/f'{table}.npy',mmap_mode='r')
            assert x1.shape[0]==x2.shape[0]==len(z['hand_id'])
            for i,(pid,phase) in enumerate(zip(z['pair_id'],z['phase'])):
                if phase!='development' or pid not in lm:continue
                assert pid not in seen;seen.add(pid);a,b=map(int,offset[i:i+2]);lab=lm[pid]
                assert lab['player_1']==z['player_1'][i] and lab['player_2']==z['player_2'][i]
                x=np.column_stack([x1[a:b],x2[a:b]]).astype(np.float32);assert np.isfinite(x).all() and len(x)>0
                hid=z['hand_id'][a:b];times=z['time_index'][a:b];assert len(set(hid))==len(hid) and (np.diff(times)>=0).all()
                y=np.array([(pid,h) in truth for h in hid],np.uint8);old=np.array([h in oldhands for h in hid],np.uint8)
                assert not (y>old).any()
                if lab['label']==0:assert not y.any()
                else:assert int(y.sum())==sum(p==pid for p,h in truth)
                arrays.append(x)
                rows.append(pl.DataFrame({'row':np.arange(total,total+len(x)), 'pair_id':[str(pid)]*len(x),'hand_id':hid,'time_index':times,
                    'evidence':y,'old_hand_only_evidence':old}))
                bags.append(dict(pair_id=str(pid),table_id=table,fold=folds[table],label=int(lab['label']),behavior_family=lab['behavior_family'],
                    player_1=lab['player_1'],player_2=lab['player_2'],offset=total,n_hands=len(x),n_evidence=int(y.sum()),
                    old_false_positives=int(((old==1)&(y==0)).sum())))
                total+=len(x)
        if len(audit)%50==0:print('RELATIONSHIP_DATA',len(audit),len(bags),total,time.time()-start,flush=True)
        audit.append(dict(table_id=table,source_pool_fold=folds[table]))
    assert seen==set(lm) and len(bags)==1860
    x=np.concatenate(arrays);r=pl.concat(rows);b=pl.DataFrame(bags).with_row_index('bag')
    assert r.select('pair_id','hand_id').n_unique()==len(r) and int(r['evidence'].sum())==len(truth)==1817
                                                                           
    sample=x[::97].copy();sw=sample[:,perm].copy();seat=cols.index('seatd');sw[:,seat]=6-sw[:,seat]
    back=sw[:,perm].copy();back[:,seat]=6-back[:,seat];np.testing.assert_array_equal(sample,back)
    np.save(ROOT/'x.npy',x);r.write_parquet(ROOT/'hands.parquet');b.write_parquet(ROOT/'bags.parquet')
    references=[]
    for f,h in itertools.combinations(range(4),2):
        tr=b.filter(~C('fold').is_in([f,h]));assert not set(tr['table_id'])&set(b.filter(C('fold').is_in([f,h]))['table_id'])
        references.append(dict(excluded_folds=[f,h],training_pair_ids=tr['pair_id'].to_list(),training_tables=sorted(set(tr['table_id'])),
            training_pairs=len(tr),training_positive_pairs=int(tr['label'].sum()),training_evidence_hands=int(tr['n_evidence'].sum())))
    config=dict(method=__doc__,columns=cols,role_permutation=perm,seat_distance_column=seat,source_tokens=[str(t1),str(t2)],
        source_policy='Original ordinary action/size policies exclude the source pool fold. They use gameplay outcomes, not competition labels. Raw outer-fold gameplay can enter a different pool policy; this is not the84 two-fold-excluded ordinary-policy protocol.',
        relationship_labels='Trusted positive/confirmed non-target only; no pseudo labels or hard/background unknown negatives',
        truncation=False,normalization='Fit on each reference model training pools; raw role swap before scaling',
        future_training='Six reference models excluding two whole pool folds each. No old supervised relationship weights or attribution tokens.',
        array_sha256=hashlib.file_digest((ROOT/'x.npy').open('rb'),'sha256').hexdigest())
    (ROOT/'config.json').write_text(json.dumps(config,indent=2));(ROOT/'reference_plan.json').write_text(json.dumps(references,indent=2))
    report=dict(pairs=len(b),positive_pairs=int(b['label'].sum()),confirmed_negative_pairs=int((b['label']==0).sum()),hand_rows=len(r),features=x.shape[1],
        all1817_evidence_hands_preserved=True,max_hands=int(b['n_hands'].max()),old160_hand_truncation_pairs=int((b['n_hands']>160).sum()),
        old_hand_only_false_positive_rows=int(b['old_false_positives'].sum()),old_false_positive_rows_in_positive_pairs=int(b.filter(C('label')==1)['old_false_positives'].sum()),
        old_false_positive_rows_in_confirmed_negative_pairs=int(b.filter(C('label')==0)['old_false_positives'].sum()),
        role_swap_involution_error=0,nested_reference_plans=len(references),seconds=time.time()-start,
        caveat='Old-label counts describe this newly selected trusted-pair dataset, not the historical mixture of training samples. No model fitted yet.')
    (ROOT/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
