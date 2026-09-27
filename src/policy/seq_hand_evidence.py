\
\
\
import os,sys,json; os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl,torch
from pathlib import Path
TOK=Path(sys.argv[1]); MD=Path(sys.argv[2]); OUTP=sys.argv[3]; ML=int(sys.argv[4]) if len(sys.argv)>4 else 400
if not json.loads((MD/'config.json').read_text()).get('aux_evidence'):
    raise ValueError('This model did not train an auxiliary evidence head. Use a validated attribution method or an evidence-trained checkpoint.')
sys.argv=['seq_train.py',str(TOK),str(MD/'_tmp'),str(MD/'config.json')]
src=open('src/policy/seq_train.py').read().split("EPOCHS=cfg.get")[0]; exec(src)
MAXL=ML
nz=np.load(MD/'norm.npz'); MU[:]=nz['mu']; SD[:]=nz['sd']
if ACT: AMU[:]=nz['amu']; ASD[:]=nz['asd']
if HSDIR: HMU[:]=nz['hmu']; HSD[:]=nz['hsd']
models=[]
for f in range(4):
    m=Net(F_).to(dev); load_checked_state(m,torch.load(MD/f'seq_fold{f}.pt',map_location=dev,weights_only=True),aux_evidence=True); m.eval(); models.append(m)
posl=set(pl.read_csv('data/development_labels.csv').filter(C('label')==1)['pair_id']); rows=[]
with torch.no_grad():
    for t in tables:
        f=tf.get(t); 
        if f is None: continue
        if 'hid' not in data[t]: data[t]['hid']=np.load(TOK/f'{t}.npz')['hand_id']
        its=[it for it in seqs_of(t,'development',None) if it[0] in posl]
        if not its: continue
        m=models[f]
        for i in range(0,len(its),64):
            b=its[i:i+64]; acc=None
            for sw in [False,True]:
                bt=batchify(b,swap=sw); X,M=bt[0].to(dev),bt[1].to(dev); m(X,M,*bt[2:]); tl=m.last_tok.cpu().numpy()
                acc=tl if acc is None else (acc+tl)/2
            for k,it in enumerate(b):
                a,bb=it[6],it[7]; hid=data[t]['hid'][a:bb]; n=min(len(hid),acc.shape[1]); hid=hid[-n:]
                rows.append(pl.DataFrame({'pair_id':[it[0]]*n,'hand_id':list(hid),'tok':acc[k,:n].astype(np.float64)}))
out=pl.concat(rows); out.write_parquet(OUTP); print('rows',out.height,'pairs',out['pair_id'].n_unique(),flush=True)
ev=pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id',pl.lit(1).alias('evidence'))
d=out.join(ev,on=['pair_id','hand_id'],how='left').with_columns(C('evidence').fill_null(0))
truth_counts=dict(ev.group_by('pair_id').len().iter_rows())
def map5(df,col):
    res={}
    for (pid,),g in df.group_by('pair_id',maintain_order=True):
        g=g.sort([col,'hand_id'],descending=[True,False]); e=g['evidence'].to_numpy(); nrel=truth_counts[pid]
        hits=0; ps=0.0
        for r,h in enumerate(e[:5],1):
            if h: hits+=1; ps+=hits/r
        res[pid]=ps/min(nrel,5)
    return round(float(np.mean([res.get(pid,0.0) for pid in truth_counts])),4),len(truth_counts)
print('token head alone MAP@5 (all shared hands):',map5(d,'tok'))
s1=pl.read_parquet('artifacts/policy/evidence_rerank_oof.parquet').select('pair_id','hand_id','s1')
j=d.join(s1,on=['pair_id','hand_id'],how='inner')
j=j.with_columns(((C('tok')-C('tok').mean())/(C('tok').std()+1e-6)).over('pair_id').alias('zt'),((C('s1').log()-C('s1').log().mean())/(C('s1').log().std()+1e-6)).over('pair_id').alias('zs'))
print('within top-20 candidates: s1 alone',map5(j,'s1'),' tok alone',map5(j,'tok'))
for w in [0.25,0.5,0.75,1.0,1.5]: print(f'  blend zs + {w}*zt:',map5(j.with_columns((C('zs')+w*C('zt')).alias('b')),'b'))
