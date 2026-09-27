\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
import sys
sys.path.insert(0,'src')
from features import build as build_basic
from detail_features import build as build_detail
from threadpoolctl import threadpool_limits
from sequence_features import augment
import build_outcome_roles as BOR,build_relationship_evidence as BRE
from session4_evidence_model import load_models,score,COLS,FAMILIES
from session6_priority_model import load_models as load_priority
from session7_model import load_models as load_event
ROOT=Path(os.environ.get('PAIR_EVENT_ROOT','artifacts/evidence_session12/pair_events'));C=pl.col
def tail(p,k):
    z=np.zeros(k+1);z[0]=1
    for s in p:z=np.r_[z[0]*(1-s),z[1:k]*(1-s)+z[:k-1]*s,z[k]+z[k-1]*s]
    return float(z[-1])
def main():
    ROOT.mkdir(parents=True,exist_ok=True);pair=pl.read_parquet('artifacts/evidence_session9/pair_full.parquet');selected=set(pair.sort('risk_score','pair_id',descending=[True,False]).head(2000)['pair_id'])|set(pair.filter(C('label')>=0)['pair_id']);sel=pair.filter(C('pair_id').is_in(list(selected)));sel.write_parquet(ROOT/'selected_pairs.parquet');families=sel.select('pair_id',C('family').alias('behavior_family'));players=sel.select('pair_id','player_1','player_2');base=load_models();cat,cols=load_priority('priority_ordered');hist,_=load_event('hist_eventblend');assert cols==COLS;folds=json.load(open('artifacts/policy/table_folds.json'));root=Path('artifacts/policy');start=time.time();done=0;reference=pl.concat([pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').filter(C('fold')==f) for f in range(4)]);errors=[]
    with threadpool_limits(limits=3):
        for path in sorted(Path('artifacts/detail_features').glob('T*.parquet')):
            out=ROOT/path.name
            if out.exists():continue
            if os.environ.get('PAIR_EVENT_EXPAND')=='1':
                raw=build_basic(path.stem,players,players.head(0));d=build_detail(path.stem,base=raw,pairs=players).join(families,on='pair_id')
            else:d=pl.read_parquet(path).filter(C('phase')=='development').join(families,on='pair_id')
            if not len(d):continue
            d=d.with_columns((C('time')/.6).alias('relative_time'));h=pl.read_parquet((Path(os.environ['PAIR_ALL_HANDS']) if os.environ.get('PAIR_ALL_HANDS') else root/'hand_features')/path.name).select(list(pl.read_parquet_schema(root/'hand_features'/path.name))).filter(C('phase')=='development').join(families.select('pair_id'),on='pair_id').sort('pair_id','time_index');rc=[c for c in h.columns if c.endswith('_r')];h=h.with_columns(*[(C(c)/(C(c[:-2]+'_v')+1).sqrt()).alias(c[:-2]+'_hz') for c in rc]);hz=[c for c in h.columns if c.endswith('_hz')];h=h.with_columns(*[C(c).rolling_mean(5,min_samples=1,center=True).over('pair_id').alias(c+'_near5') for c in hz]);add=[c for c in h.columns if c not in ['pair_id','hand_id','phase','time_index','player_1','player_2']];d=d.join(h.select('pair_id','hand_id',*add),on=['pair_id','hand_id']);d,_=augment(d);query=d.select('pair_id','hand_id').join(players,on='pair_id');d=d.join(BOR.build(path.stem,query),on=['pair_id','hand_id']).join(BRE.build(path.stem,query),on=['pair_id','hand_id']);f=folds[path.stem];rows=[]
            for b in FAMILIES:
                q=d.filter(C('behavior_family')==b).sort('pair_id','time','hand_id')
                if not len(q):continue
                x=q.select(COLS).to_numpy();assert np.isfinite(x).all();bp=score(base,b,x,[f]);ca=cat[b][f][0].predict_proba(x,thread_count=3)[:,1];cb=cat[b][f][1].predict_proba(x,thread_count=3)[:,1];ha=hist[b][f][1][0].predict_proba(x)[:,1];hb=hist[b][f][1][1].predict_proba(x)[:,1];q=q.with_columns(*[pl.Series(n,p) for n,p in [('base',bp),('cat_primary',ca),('cat_secondary',cb),('hist_primary',ha),('hist_secondary',hb)]]);check=q.select('pair_id','hand_id','base','cat_primary','cat_secondary','hist_primary','hist_secondary').join(reference.select('pair_id','hand_id','base','cat_primary','cat_secondary','hist_primary','hist_secondary'),on=['pair_id','hand_id'],suffix='_ref',validate='1:1');err={n:float((check[n]-check[n+'_ref']).abs().max()) for n in ['base','cat_primary','cat_secondary','hist_primary','hist_secondary']} if len(check) else {};errors.append({'table_id':path.stem,'positive_hands_replayed':len(check),'errors':err})
                for (pid,),g in q.group_by('pair_id'):
                    ca=g['cat_primary'].to_numpy();cb=g['cat_secondary'].to_numpy();ha=g['hist_primary'].to_numpy();hb=g['hist_secondary'].to_numpy();cs=np.maximum(1,ca+cb);hs=np.maximum(1,ha+hb);a=.5*(ca/cs+ha/hs);bb=.5*(cb/cs+hb/hs);s=a+bb;row={'pair_id':pid,'table_id':path.stem,'fold':f,'event_family':b,'shared_hands':len(g),'prob_at_least3':tail(s,3),'prob_at_least5':tail(s,5),'joint_primary_mass':float(a.sum()),'joint_secondary_mass':float(bb.sum()),'joint_event_mass':float(s.sum()),'base_mass':float(g['base'].sum())}
                    for name,v in [('joint',s),('base',g['base'].to_numpy()),('primary',a),('secondary',bb)]:
                        z=np.sort(v)[::-1]
                        for k in [1,3,5,10]:row[f'{name}_top{k}_sum']=float(z[:k].sum())
                    rows.append(row)
            pl.DataFrame(rows).write_parquet(out);out.with_suffix('.json').write_text(json.dumps(errors[-len(FAMILIES):],indent=2));done+=1
            if done%40==0:print('pair event tables',done,'seconds',round(time.time()-start,1),flush=True)
    allscores=pl.read_parquet(list(ROOT.glob('T*.parquet')));assert set(allscores['pair_id'])==selected;allscores.join(sel,on='pair_id',validate='1:1').write_parquet(ROOT/'pair_event_scores.parquet');audit=[]
    for path in ROOT.glob('T*.json'):audit.extend(json.load(open(path)))
    maximum=max([v for r in audit for v in r['errors'].values()] or [0]);(ROOT/'audit.json').write_text(json.dumps({'selected_pairs':len(selected),'positive_reference_max_error':maximum,'target_policy':'labels plus baseline top2000; unknown remains unknown'},indent=2));print('pair event complete',len(selected),'positive replay error',maximum,flush=True)
if __name__=='__main__':main()
