\
import os,sys,json,argparse,time; os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl,torch
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('tokens'); ap.add_argument('model'); ap.add_argument('phase'); ap.add_argument('base'); ap.add_argument('out'); ap.add_argument('--window'); ap.add_argument('--mode',default='all'); ap.add_argument('--thr',type=float,default=0.3); ap.add_argument('--min-risk',type=float,default=0.02); ap.add_argument('--min-kept',type=int,default=10)
ARGS=ap.parse_args(); TOK=Path(ARGS.tokens); MD=Path(ARGS.model)
sys.argv=['seq_train.py',ARGS.tokens,str(MD/'_tmp'),str(MD/'config.json')]
src=open('src/policy/seq_train.py').read().split("EPOCHS=cfg.get")[0]                                             
exec(src)
a=ARGS; C=pl.col
names=['none','directed_transfer','soft_play','coordinated_isolation']
nz=np.load(MD/'norm.npz'); MU[:]=nz['mu']; SD[:]=nz['sd']
if ACT: AMU[:]=nz['amu']; ASD[:]=nz['asd']
if HSDIR: HMU[:]=nz['hmu']; HSD[:]=nz['hsd']
models=[]
for f in range(4):
    m=Net(F_).to(dev); load_checked_state(m,torch.load(MD/f'seq_fold{f}.pt',map_location=dev,weights_only=True),cfg.get('aux_evidence',False)); m.eval(); models.append(m)
win={'first_2000':(0,2000),'last_2000':(1000,3000)}.get(a.window) if a.window else None
base=pl.read_csv(a.base).select('pair_id','player_1','player_2',(1-C('none')).alias('risk'),*names[1:])
long=pl.concat([base.select(C('player_1').alias('p'),C('player_2').alias('q'),'risk'),base.select(C('player_2').alias('p'),C('player_1').alias('q'),'risk')])
strong=long.filter(C('risk')>a.thr).select('p','q',C('risk').alias('rq'))
top=long.sort('risk',descending=True).group_by('p').agg(C('q').first().alias('q1'),C('risk').first().alias('r1'))
b=base.join(top.rename({'p':'player_1','q1':'q1_1','r1':'r1_1'}),on='player_1').join(top.rename({'p':'player_2','q1':'q1_2','r1':'r1_2'}),on='player_2')
b=b.with_columns((C('r1_1')>pl.max_horizontal(pl.lit(a.thr),C('risk'))).alias('ex1'),(C('r1_2')>pl.max_horizontal(pl.lit(a.thr),C('risk'))).alias('ex2'))
cand=b.filter((C('risk')>=a.min_risk)&(C('ex1')|C('ex2'))); print('candidates',cand.height,'of',b.height,flush=True)
                                      
qq=cand.select('pair_id','player_1','player_2','risk')
cond=(C('rq')>pl.max_horizontal(pl.lit(a.thr),C('risk')))
s1=qq.join(strong.rename({'p':'player_1','q':'qx'}),on='player_1').filter(cond).filter(C('qx')!=C('player_2')).select('pair_id','qx')
s2=qq.join(strong.rename({'p':'player_2','q':'qx'}),on='player_2').filter(cond).filter(C('qx')!=C('player_1')).select('pair_id','qx')
if a.mode=='top':
    s1=cand.filter(C('ex1')).select('pair_id',C('q1_1').alias('qx')); s2=cand.filter(C('ex2')).select('pair_id',C('q1_2').alias('qx'))
ex=pl.concat([s1,s2]).unique(); exmap={}
for pid,qx in zip(ex['pair_id'],ex['qx']): exmap.setdefault(pid,set()).add(qx)
                                                 
t0=time.time(); rows=[]
cand_set=set(cand['pair_id'])
pf=pl.concat([pl.read_parquet(p).filter(C('phase')==a.phase).select('pair_id','table_id') for p in sorted(Path('artifacts/policy/pair_features').glob('*.parquet'))]) if False else None
for t in tables:
    d=data[t]; off=np.concatenate([[0],np.cumsum(d['n'])]); z=np.load(TOK/f'{t}.npz'); hid=z['hand_id']
    roster=None; items=[]; meta=[]
    for k in range(len(d['pair'])):
        pid=d['pair'][k]
        if d['phase'][k]!=a.phase or pid not in cand_set: continue
        if roster is None:
            st=pl.read_parquet(f'artifacts/policy/states/{t}.parquet').select('hand_id','player_id').unique(); roster={}
            for h_,p_ in zip(st['hand_id'],st['player_id']): roster.setdefault(h_,set()).add(p_)
        lo,hi=off[k],off[k+1]; X=_cat(d,slice(lo,hi)); ti=d['ti'][lo:hi]; hh=hid[lo:hi]
        keep=np.array([not (roster.get(h_,set())&exmap[pid]) for h_ in hh])
        if win is not None: keep&=(ti>=win[0])&(ti<win[1])
        nk=int(keep.sum()); meta.append((pid,nk))
        items.append((pid,d['p1'][k],d['p2'][k],X[keep] if nk>0 else X[:1],None,t,lo,hi,None,nk,k,keep))
    if not items: continue
    if a.phase=='development': sc=score(models[tf[t]],items)
    else: sc=np.mean([score(m,items) for m in models],axis=0)
    for (pid,nk),s in zip(meta,sc): rows.append((pid,nk,float(s[0]) if nk>=a.min_kept else 0.0))
print('rescored',len(rows),round(time.time()-t0),flush=True)
new=pl.DataFrame({'pair_id':[r[0] for r in rows],'n_kept':[r[1] for r in rows],'risk_new':[r[2] for r in rows]})
out=base.join(new,on='pair_id',how='left').with_columns(C('n_kept').fill_null(0),C('risk_new').fill_null(C('risk')))
r=np.minimum(out['risk'].to_numpy(),out['risk_new'].to_numpy()); fam=out.select(names[1:]).to_numpy(); fam=fam/np.maximum(fam.sum(1,keepdims=True),1e-12)
pl.DataFrame({'pair_id':out['pair_id'],'none':1-r,**{n:r*fam[:,k] for k,n in enumerate(names[1:])},'risk_orig':out['risk'],'risk_new':out['risk_new'],'n_kept':out['n_kept']}).sort('pair_id').write_csv(a.out)
print('saved',a.out,'>0.5 before',int((out['risk']>0.5).sum()),'after',int((r>0.5).sum()),flush=True)
