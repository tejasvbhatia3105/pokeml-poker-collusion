\
\
\
import os,sys; os.environ.setdefault('POLARS_MAX_THREADS','6')
import json,time,numpy as np,polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
root=Path('artifacts/policy'); C=pl.col; out=Path(sys.argv[1]); out.mkdir(exist_ok=True,parents=True); cfg=json.loads(Path(sys.argv[2]).read_text())
cols=json.loads((root/'residual_columns.json').read_text()); names=['none','directed_transfer','soft_play','coordinated_isolation']
folds=json.loads(Path('artifacts/folds.json').read_text()); tf={t:f['fold'] for f in folds for t in f['valid_tables']}
lab=pl.read_csv('data/development_labels.csv')
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','started_at']).sort('table_id','started_at').with_columns(pl.int_range(pl.len()).over('table_id').alias('idx'))
evd=pl.read_csv('data/development_evidence.csv').join(hands.select('hand_id','idx'),on='hand_id')
def bounds(w): return {'full':(0,3000),'first_2000':(0,2000),'last_2000':(1000,3000)}.get(w) or tuple(map(int,w[1:].split('_')))
GRID=cfg.get('grid')                                                                               
gridcols=None
def gridframe(key):
    global gridcols
    src=GRID[key]; g=pl.read_parquet(sorted(Path(src).glob('*.parquet'))) if Path(src).is_dir() else pl.read_parquet(src)
    if key=='eval': g=g.filter(C('phase')=='evaluation')
    elif key=='full': g=g.filter(C('phase')=='development')
    zc=[c for c in g.columns if c.endswith(('_z','_w10_max','_w10_min'))]
    if gridcols is None: gridcols=zc
    return g.select('pair_id',*gridcols)
def load(window):
    paths=sorted((root/'pair_features').glob('*.parquet')) if window=='full' else sorted((root/'full_window_stress'/window/'pair_features').glob('*.parquet'))
    d=pl.concat([pl.read_parquet(p).filter(C('phase')=='development') for p in paths]).join(lab.select('pair_id','label','behavior_family'),on='pair_id',how='left').with_columns(C('label').fill_null(-1))
    if GRID: d=d.join(gridframe(window),on='pair_id',how='left').fill_null(0)
    a,b=bounds(window); gw=evd.filter((C('idx')>=a)&(C('idx')<b)).group_by('pair_id').agg(pl.len().alias('n_ev_in'))
    return d.join(gw,on='pair_id',how='left').with_columns(C('n_ev_in').fill_null(0),pl.Series('fold',[tf.get(t,-1) for t in d['table_id']]))
full=load('full')
if cfg.get('clean_features'):
    cf=pl.read_parquet(cfg['clean_features']).filter(C('n_kept')>=10).drop('n_kept'); keep=full.join(cf.select('pair_id'),on='pair_id',how='anti'); rep=full.select([c for c in full.columns if c not in cols]).join(cf,on='pair_id')
    full=pl.concat([keep,rep.select(full.columns)]); print('cleaned rows',rep.height,flush=True)
wins={w:load(w) for w in cfg.get('windows',[])}
if GRID: cols=cols+gridcols; print('features',len(cols),flush=True)
src=pl.read_csv(cfg.get('pseudo_src','artifacts/policy/history/evaluation_history.csv')) if cfg.get('pseudo_src') else None
if src is not None: risk=full.select('pair_id').join(src.select('pair_id',(1-C('none')).alias('r')),on='pair_id',how='left')['r'].fill_null(0).to_numpy()
else:
    v4=[CatBoostClassifier() for _ in range(4)]
    for k in range(4): v4[k].load_model(str(root/f'residual_fold{k}.cbm'))
    parts=[]
    for f in range(4):
        m=(full['fold']==f).to_numpy(); parts.append((np.flatnonzero(m),1-v4[f].predict_proba(full.filter(pl.Series(m)).select(cols).to_numpy(),thread_count=6)[:,0]))
    risk=np.zeros(len(full)); [risk.__setitem__(i,r) for i,r in parts]
pm=((full['label']==-1).to_numpy())&(risk>cfg.get('pseudo',0.9))
v4=[CatBoostClassifier() for _ in range(4)]
for k in range(4): v4[k].load_model(str(root/f'residual_fold{k}.cbm'))
Xp=full.filter(pl.Series(pm)).select(cols).to_numpy(); yp=np.mean([v4[k].predict_proba(Xp,thread_count=6) for k in range(4)],axis=0)[:,1:].argmax(1)+1
print('pseudo-positives',int(pm.sum()),'windows',list(wins),flush=True)
seeds=cfg.get('seeds',[991]); t0=time.time(); models={s:[] for s in seeds}; oof=np.zeros((len(full),4))
for f in range(4):
    rng=np.random.default_rng(7+f); parts=[];ys=[];ws=[]
    for w,d in [('full',full)]+list(wins.items()):
        tr=(d['fold']!=f).to_numpy(); lm=(d['label']>=0).to_numpy()&tr
        if w!='full': lm&=((d['label']==0)|(d['n_ev_in']>=1)).to_numpy()
        z=d.filter(pl.Series(lm)); yl=np.array([names.index(x) if x in names else 0 for x in z['behavior_family'].fill_null('none')])
        parts.append(z.select(cols).to_numpy()); ys.append(yl); ws.append(np.full(len(yl),1.0 if w=='full' else cfg.get('w_win',0.5)))
    ftr=(full['fold']!=f).to_numpy(); keep=ftr[pm]; parts.append(Xp[keep]); ys.append(yp[keep]); ws.append(np.full(int(keep.sum()),cfg.get('w_pseudo',1.0)))
    if cfg.get('w_hardneg'):
        lab_pos=lab.filter(C('label')==1); pp=set(lab_pos['player_1'])|set(lab_pos['player_2']); partner={}
        for x,y in zip(lab_pos['player_1'],lab_pos['player_2']): partner.setdefault(x,set()).add(y); partner.setdefault(y,set()).add(x)
        hn=np.array([(l==-1) and ((x in pp)!=(y in pp)) for x,y,l in zip(full['player_1'],full['player_2'],full['label'])])&ftr&~pm
        parts.append(full.filter(pl.Series(hn)).select(cols).to_numpy()); ys.append(np.zeros(int(hn.sum()),int)); ws.append(np.full(int(hn.sum()),cfg['w_hardneg']))
    if cfg.get('eval_pseudo'):
        ep=cfg['eval_pseudo']; src=pl.read_csv(ep['path']).with_columns((1-C('none')).alias('r'))
        posq=src.filter(C('r')>ep.get('thr',0.9)).select('pair_id'); fam_=src.filter(C('r')>ep.get('thr',0.9)).select(names[1:]).to_numpy().argmax(1)+1
        famq=dict(zip(posq['pair_id'],fam_))
        if 'EVALX' not in globals():
            globals()['EVALX']=pl.concat([pl.read_parquet(p).filter(C('phase')=='evaluation').select('pair_id',*cols) for p in sorted((root/'pair_features').glob('*.parquet'))])
        z=EVALX.join(posq,on='pair_id'); parts.append(z.select(cols).to_numpy()); ys.append(np.array([famq[x] for x in z['pair_id']])); ws.append(np.full(z.height,ep.get('w',1.0)))
        if ep.get('neg_path'):
            ng=pl.read_csv(ep['neg_path']); ng=ng.filter((C('risk_orig')>0.5)&(C('risk_new')<0.1)).select('pair_id'); z=EVALX.join(ng,on='pair_id')
            parts.append(z.select(cols).to_numpy()); ys.append(np.zeros(z.height,int)); ws.append(np.full(z.height,ep.get('w_neg',0.3)))
        if f==0: print('eval pseudo-positives',posq.height,'eval spillover negatives',(z.height if ep.get('neg_path') else 0),flush=True)
    nbg=cfg.get('bg',12000)
    if nbg:
        um=(full['label']==-1).to_numpy()&ftr&~(risk>0.5); idx=rng.choice(np.flatnonzero(um),nbg,replace=False)
        parts.append(full[idx.tolist()].select(cols).to_numpy()); ys.append(np.zeros(nbg,int)); ws.append(np.full(nbg,cfg.get('w_bg',0.1)))
    X=np.vstack(parts); y=np.concatenate(ys); wt=np.concatenate(ws); va=(full['fold']==f).to_numpy(); Xv=full.filter(pl.Series(va)).select(cols).to_numpy()
    for s in seeds:
        m=CatBoostClassifier(iterations=cfg.get('iters',900),depth=cfg.get('depth',5),learning_rate=.04,loss_function='MultiClass',l2_leaf_reg=10,thread_count=6,random_seed=s+f,verbose=False,allow_writing_files=False)
        m.fit(X,y,sample_weight=wt); m.save_model(str(out/f'pair_seed{s}_fold{f}.cbm')); models[s].append(m); oof[va]+=m.predict_proba(Xv,thread_count=6)/len(seeds)
    print('fold',f,'rows',len(X),'seconds',round(time.time()-t0),flush=True)
o=pl.DataFrame(oof,schema=names).with_columns(full['pair_id'],full['label'],full['table_id']).filter(C('table_id').is_in(set(tf))).drop('table_id'); o.write_csv(out/'pair_oof_allpairs.csv')
r=1-o['none']; print('OOF: pos<0.5',int(((o['label']==1)&(r<0.5)).sum()),'neg>0.1',int(((o['label']==0)&(r>0.1)).sum()),'neg>0.5',int(((o['label']==0)&(r>0.5)).sum()),'unlisted>0.5',int(((o['label']==-1)&(r>0.5)).sum()),flush=True)
ev=pl.read_csv('data/evaluation_pairs.csv').select('pair_id'); parts=[]
EVG=gridframe('eval') if GRID else None
for p in sorted((root/'pair_features').glob('*.parquet')):
    z=pl.read_parquet(p).filter(C('phase')=='evaluation').join(ev,on='pair_id')
    if GRID: z=z.join(EVG,on='pair_id',how='left').fill_null(0)
    X=z.select(cols).to_numpy()
    pr=np.mean([m.predict_proba(X,thread_count=6) for s in seeds for m in models[s]],axis=0)
    parts.append(z.select('pair_id').with_columns(*[pl.Series(n,pr[:,k]) for k,n in enumerate(names)]))
pl.concat(parts).sort('pair_id').write_csv(out/'pair_eval.csv'); json.dump(cfg,open(out/'config.json','w'),indent=2); print('eval predicted',flush=True)
