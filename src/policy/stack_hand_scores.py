\
import os,sys,glob; os.environ.setdefault('POLARS_MAX_THREADS','6')
import json,numpy as np,polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
C=pl.col; root=Path('artifacts/policy'); S='cache/'
base_dir=sys.argv[1]; base_dev=sys.argv[2]; base_eval=sys.argv[3]; out=Path(sys.argv[4]); out.mkdir(exist_ok=True,parents=True)
hs=pl.read_parquet(os.environ.get('HS_FEATURES','artifacts/hand_scores_cheap/hand_score_features.parquet')); HS=[c for c in hs.columns if c.startswith('hs_')]
cols=json.loads((root/'residual_columns.json').read_text()); names=['none','directed_transfer','soft_play','coordinated_isolation']
folds=json.loads(Path('artifacts/folds.json').read_text()); tf={t:f['fold'] for f in folds for t in f['valid_tables']}
lab=pl.read_csv('data/development_labels.csv'); pos=lab.filter(C('label')==1); partner={}
for a,b in zip(pos['player_1'],pos['player_2']): partner.setdefault(a,set()).add(b); partner.setdefault(b,set()).add(a)
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','started_at']).sort('table_id','started_at').with_columns(pl.int_range(pl.len()).over('table_id').alias('idx'))
evd=pl.read_csv('data/development_evidence.csv').join(hands.select('hand_id','idx'),on='hand_id')
models={f:[] for f in range(4)}
for f in range(4):
    for p in sorted(glob.glob(f'{base_dir}/pair_seed*_fold{f}.cbm')) or [f'{base_dir}/pair_fold{f}.cbm']:
        m=CatBoostClassifier(); m.load_model(p); models[f].append(m)
def frame(window):
    if window=='full':
        d=pl.read_csv(base_dev).select('pair_id',(1-C('none')).alias('base')).join(pl.read_parquet(S+'allpairs_full.parquet').select('pair_id','player_1','player_2','table_id','policy_n_hands'),on='pair_id')
    else:
        parts=[]
        for p in sorted((root/'full_window_stress'/window/'pair_features').glob('*.parquet')):
            z=pl.read_parquet(p).filter(C('phase')=='development'); t=p.stem
            if t not in tf: continue
            r=1-np.mean([m.predict_proba(z.select(cols).to_numpy(),thread_count=6) for m in models[tf[t]]],axis=0)[:,0]
            parts.append(z.select('pair_id','player_1','player_2','table_id','policy_n_hands').with_columns(pl.Series('base',r)))
        d=pl.concat(parts)
    d=d.join(hs.filter(C('window')==window).drop('window'),on='pair_id',how='left').fill_null(0).join(lab.select('pair_id','label'),on='pair_id',how='left').with_columns(C('label').fill_null(-1))
    lo,hi={'full':(0,3000),'first_2000':(0,2000),'last_2000':(1000,3000)}[window]
    gw=evd.filter((C('idx')>=lo)&(C('idx')<hi)).group_by('pair_id').agg(pl.len().alias('n_ev_in')); d=d.join(gw,on='pair_id',how='left').with_columns(C('n_ev_in').fill_null(0))
    p1=d['player_1'].to_list(); p2=d['player_2'].to_list(); L=d['label'].to_numpy()
    d=d.with_columns(pl.Series('hardneg',[(l==-1) and ((a in partner)!=(b in partner)) for a,b,l in zip(p1,p2,L)]),pl.Series('clean',[(l==-1) and (a not in partner) and (b not in partner) for a,b,l in zip(p1,p2,L)]),pl.Series('fold',[tf.get(t,-1) for t in d['table_id']]))
    return d.filter(C('fold')>=0)
W={w:frame(w) for w in ['full','first_2000','last_2000']}; print('frames',{w:d.height for w,d in W.items()},flush=True)
def X_of(d): 
    b=d['base'].to_numpy().clip(1e-6,1-1e-6); return np.column_stack([np.log(b/(1-b)),np.log(d['policy_n_hands'].to_numpy()+1),d.select(HS).to_numpy()])
FEAT=['logit_base','log_n']+HS
def report(tag,d,s):
    L=d['label'].to_numpy(); nev=d['n_ev_in'].to_numpy(); hn=d['hardneg'].to_numpy(); cl=d['clean'].to_numpy(); pos_=(L==1)&(nev>=1)
    order=np.argsort(-s); rank=np.empty(len(s),int); rank[order]=np.arange(len(s)); negs=np.sort(s[L!=1]); above=len(negs)-np.searchsorted(negs,s[pos_],side='right')
    print(f"{tag:22s} pos>0.5={int((pos_&(s>0.5)).sum())}/{int(pos_.sum())} rec@300={np.mean(above<300):.4f} rec@1000={np.mean(above<1000):.4f} | confneg>0.5={int(((L==0)&(s>0.5)).sum())} hardneg>0.5={int((hn&(s>0.5)).sum())} top700={int((hn&(rank<700)).sum())} top1000={int((hn&(rank<1000)).sum())} | clean top700={int((cl&(rank<700)).sum())}",flush=True)
rng=np.random.default_rng(11); oof={w:np.zeros(len(d)) for w,d in W.items()}
for f in range(4):
    Xs=[];ys=[];ws=[]
    for w,d in W.items():
        L=d['label'].to_numpy(); fold=d['fold'].to_numpy(); nev=d['n_ev_in'].to_numpy(); hn=d['hardneg'].to_numpy(); base=d['base'].to_numpy()
        tr=fold!=f; posm=tr&(L==1)&(nev>=1); negm=tr&(L==0); hnm=tr&hn; um=tr&(L==-1)&~hn&(base<0.5); ui=rng.choice(np.flatnonzero(um),min(30000,um.sum()),replace=False); umask=np.zeros(len(d),bool); umask[ui]=True
        X=X_of(d); wf=1.0 if w=='full' else 0.5
        for m,y,wt in [(posm,1,1.0),(negm,0,1.0),(hnm,0,0.3),(umask,0,0.05)]:
            Xs.append(X[m]); ys.append(np.full(m.sum(),y)); ws.append(np.full(m.sum(),wt*wf))
    Xt=np.vstack(Xs); yt=np.concatenate(ys); wt=np.concatenate(ws)
    m=CatBoostClassifier(iterations=400,depth=3,learning_rate=.05,l2_leaf_reg=20,loss_function='Logloss',thread_count=6,random_seed=31+f,verbose=False,allow_writing_files=False,monotone_constraints={0:1})
    m.fit(Xt,yt,sample_weight=wt); m.save_model(str(out/f'stack_fold{f}.cbm'))
    for w,d in W.items():
        va=(d['fold']==f).to_numpy(); oof[w][va]=m.predict_proba(X_of(d)[va])[:,1]
    print('fold',f,'rows',len(Xt),flush=True)
for w,d in W.items():
    report(f'{w} base',d,d['base'].to_numpy()); report(f'{w} stacked',d,oof[w]); report(f'{w} geo(base,stack)',d,np.sqrt(d['base'].to_numpy()*oof[w]))
full=W['full']; pl.DataFrame({'pair_id':full['pair_id'],'none':1-oof['full'],'label':full['label']}).write_csv(out/'stack_oof_full.csv')
            
ev=pl.read_csv(base_eval).select('pair_id',(1-C('none')).alias('base'),*names[1:]).join(pl.read_csv('data/evaluation_pairs.csv').select('pair_id',C('shared_hands').alias('policy_n_hands')),on='pair_id').join(hs.filter(C('window')=='evaluation').drop('window'),on='pair_id',how='left').fill_null(0)
ms=[CatBoostClassifier() for _ in range(4)]
for f in range(4): ms[f].load_model(str(out/f'stack_fold{f}.cbm'))
s=np.mean([m.predict_proba(X_of(ev))[:,1] for m in ms],axis=0); base=ev['base'].to_numpy()
for tag,r in [('stack',s),('geo',np.sqrt(base*s))]:
    fam=ev.select(names[1:]).to_numpy(); fam=fam/np.maximum(fam.sum(1,keepdims=True),1e-12)
    pl.DataFrame({'pair_id':ev['pair_id'],'none':1-r,**{n:r*fam[:,k] for k,n in enumerate(names[1:])}}).sort('pair_id').write_csv(out/f'pair_eval_{tag}.csv'); print(tag,'eval >0.5',int((r>0.5).sum()),'>0.9',int((r>0.9).sum()),flush=True)
