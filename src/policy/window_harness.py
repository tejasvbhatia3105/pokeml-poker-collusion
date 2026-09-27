\
import os,sys,glob; os.environ.setdefault('POLARS_MAX_THREADS','6')
import json,numpy as np,polars as pl
from pathlib import Path
from catboost import CatBoostClassifier
root=Path('artifacts/policy'); C=pl.col
cols=json.loads((root/'residual_columns.json').read_text()); folds=json.loads(Path('artifacts/folds.json').read_text()); tf={t:f['fold'] for f in folds for t in f['valid_tables']}
lab=pl.read_csv('data/development_labels.csv'); pos=lab.filter(C('label')==1); partner={}
for a,b in zip(pos['player_1'],pos['player_2']): partner.setdefault(a,set()).add(b); partner.setdefault(b,set()).add(a)
hands=pl.read_parquet('data/hands.parquet',columns=['hand_id','table_id','started_at']).sort('table_id','started_at').with_columns(pl.int_range(pl.len()).over('table_id').alias('idx'))
evd=pl.read_csv('data/development_evidence.csv').join(hands.select('hand_id','idx'),on='hand_id')
def models_for(spec):
    if spec=='v4': paths={f:[str(root/f'residual_fold{f}.cbm')] for f in range(4)}
    else: paths={f:sorted(glob.glob(f'{spec}/pair_seed*_fold{f}.cbm')) or [f'{spec}/pair_fold{f}.cbm'] for f in range(4)}
    out={}
    for f,ps in paths.items():
        out[f]=[]
        for p in ps: m=CatBoostClassifier(); m.load_model(p); out[f].append(m)
    return out
WINDOWS={'first_2000':(0,2000),'last_2000':(1000,3000),'w500_2500':(500,2500),'w0_1500':(0,1500),'w1500_3000':(1500,3000)}
data={}
for w,(a,b) in WINDOWS.items():
    d=pl.concat([pl.read_parquet(p).filter(C('phase')=='development') for p in sorted((root/'full_window_stress'/w/'pair_features').glob('*.parquet'))])
    d=d.join(lab.select('pair_id','label'),on='pair_id',how='left').with_columns(C('label').fill_null(-1))
    gw=evd.filter((C('idx')>=a)&(C('idx')<b)).group_by('pair_id').agg(pl.len().alias('n_ev_in')); d=d.join(gw,on='pair_id',how='left').with_columns(C('n_ev_in').fill_null(0),pl.Series('fold',[tf.get(t,-1) for t in d['table_id']])).filter(C('fold')>=0)
    p1=d['player_1'].to_list(); p2=d['player_2'].to_list(); l=d['label'].to_numpy()
    d=d.with_columns(pl.Series('hardneg',[(x==-1) and ((p in partner)!=(q in partner)) for p,q,x in zip(p1,p2,l)]),pl.Series('clean',[(x==-1) and (p not in partner) and (q not in partner) for p,q,x in zip(p1,p2,l)]))
    data[w]=d
print('windows loaded',flush=True)
rows=[]
for spec in sys.argv[1:]:
    name,path=spec.split('=') if '=' in spec else (spec,spec); ms=models_for(path)
    agg={}
    for w,d in data.items():
        s=np.zeros(len(d)); X=d.select(cols).to_numpy(); fold=d['fold'].to_numpy()
        for f in range(4):
            va=fold==f; s[va]=1-np.mean([m.predict_proba(X[va],thread_count=6) for m in ms[f]],axis=0)[:,0]
        l=d['label'].to_numpy(); nev=d['n_ev_in'].to_numpy(); hn=d['hardneg'].to_numpy(); cl=d['clean'].to_numpy()
        pos_=(l==1)&(nev>=1); weak=pos_&(nev<=3); ineligible=(l==1)&(nev==0)
        order=np.argsort(-s); rank=np.empty(len(s),int); rank[order]=np.arange(len(s))
        negs=np.sort(s[(l!=1)]); above=len(negs)-np.searchsorted(negs,s[pos_],side='right'); wabove=len(negs)-np.searchsorted(negs,s[weak],side='right')
        r=dict(window=w,pos=int(pos_.sum()),rec100=float(np.mean(above<100)),rec300=float(np.mean(above<300)),rec1000=float(np.mean(above<1000)),weak_rec300=float(np.mean(wabove<300)),
               hardneg_top600=int((hn&(rank<600)).sum()),hardneg_top1000=int((hn&(rank<1000)).sum()),clean_top600=int((cl&(rank<600)).sum()),pos_top600=int((pos_&(rank<600)).sum()),confneg_top1000=int(((l==0)&(rank<1000)).sum()))
        rows.append(dict(model=name,**r))
        print(name,json.dumps(r),flush=True)
pl.DataFrame(rows).write_csv('artifacts/window_harness_'+'_'.join(s.split('=')[0] for s in sys.argv[1:])+'.csv')
