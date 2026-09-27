\
import os,sys,json; os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl,torch
from pathlib import Path
ARGS=sys.argv[1:]; TOK=Path(ARGS[0]); MD=Path(ARGS[1]); OUTP=ARGS[2]
sys.argv=['seq_train.py',ARGS[0],str(MD/'_tmp'),str(MD/'config.json')]
src=open('src/policy/seq_train.py').read().split("EPOCHS=cfg.get")[0]; exec(src)
C=pl.col; nz=np.load(MD/'norm.npz'); MU[:]=nz['mu']; SD[:]=nz['sd']
if ACT: AMU[:]=nz['amu']; ASD[:]=nz['asd']
models=[]
for f in range(4):
    m=Net(F_).to(dev); m.load_state_dict(torch.load(MD/f'seq_fold{f}.pt',map_location=dev),strict=False); m.eval(); models.append(m)
posids=set(lab.filter(C('label')==1)['pair_id']); rows=[]
with torch.no_grad():
    for t in tables:
        if t not in tf: continue
        d=data[t]; off=np.concatenate([[0],np.cumsum(d['n'])]); hid=np.load(TOK/f'{t}.npz')['hand_id']
        items=[];meta=[]
        for k in range(len(d['pair'])):
            if d['phase'][k]!='development' or d['pair'][k] not in posids: continue
            a,b=off[k],off[k+1]; X=_cat(d,slice(a,b))
            items.append((d['pair'][k],d['p1'][k],d['p2'][k],X)); meta.append((d['pair'][k],hid[a:b]))
        if not items: continue
        m=models[tf[t]]
        for i in range(0,len(items),64):
            b=items[i:i+64]; acc=None
            for sw in [False,True]:
                X,M=batchify(b,swap=sw); m(X.to(dev),M.to(dev)); tl=torch.sigmoid(m.last_tok).cpu().numpy(); acc=tl if acc is None else (acc+tl)/2
            for j,(pid,hh) in enumerate(meta[i:i+64]):
                n=len(hh); L=acc.shape[1]; sc=acc[j,:min(n,L)]
                if n>L: sc=np.concatenate([np.zeros(n-L),sc])                                   
                for h_,s_ in zip(hh,sc): rows.append((pid,h_,float(s_)))
o=pl.DataFrame({'pair_id':[r[0] for r in rows],'hand_id':[r[1] for r in rows],'tok':[r[2] for r in rows]})
rel=pl.read_parquet('artifacts/policy/relationship_evidence/oof.parquet').select('pair_id','hand_id','behavior_family','evidence','score')
o=o.join(rel,on=['pair_id','hand_id'],how='inner'); o.write_parquet(OUTP)
def map5(df,col):
    r=[]
    for pid,g in df.sort(['pair_id',col,'hand_id'],descending=[False,True,False]).group_by('pair_id',maintain_order=True):
        e=g['evidence'].to_numpy()[:5]; r.append(float((np.cumsum(e)/np.arange(1,len(e)+1)*e).sum()/min(5,g['evidence'].sum())))
    return np.mean(r)
from scipy.stats import rankdata
o=o.with_columns(((C('tok').rank().over('pair_id')+C('score').rank().over('pair_id'))/2).alias('rankblend'),((C('tok')*C('score')).sqrt()).alias('geo'))
print('hands',o.height,'pairs',o['pair_id'].n_unique())
for c in ['score','tok','rankblend','geo']: print(f'MAP@5 {c:10s} {map5(o,c):.4f}  '+' '.join(f'{b[:4]}={map5(o.filter(C("behavior_family")==b),c):.4f}' for b in ['directed_transfer','soft_play','coordinated_isolation']))
