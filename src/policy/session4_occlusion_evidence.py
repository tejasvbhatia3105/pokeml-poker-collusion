\
\
\
\
\
import os,ast,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl,torch
from torch import nn
torch.set_num_threads(4)
OUT=Path(os.environ.get('SEQ_REPLAY_OUT','artifacts/evidence_session4'));OUT.mkdir(exist_ok=True,parents=True)
MD=Path(os.environ.get('SEQ_REPLAY_MODEL','artifacts/seq_v6'));cfg=json.loads((MD/'config.json').read_text())
local_cfg=json.loads(Path('artifacts/seq_v6/config.json').read_text())
TOK2=Path(local_cfg['tokens2']);TOK3=Path(local_cfg['tokens3']);TOK=TOK2.parent/'seq_tokens'
cols=json.loads((TOK/'columns.json').read_text())+json.loads((TOK2/'columns2.json').read_text())+json.loads((TOK3/'columns3.json').read_text())
ns=dict(cfg=cfg,nn=nn,torch=torch,ACT=False,F_IN=len(cols),FS=0,TREL=cols.index('trel'))
tree=ast.parse(Path('src/policy/seq_train.py').read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['Net','swapname']]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<frozen_seq_v6_definitions>','exec'),ns)
perm=np.array([cols.index(ns['swapname'](c)) for c in cols]);seatd=cols.index('seatd')
nz=np.load(MD/'norm.npz');mu=nz['mu'];sd=nz['sd'];models=[]
for f in range(4):
    m=ns['Net'](len(cols));m.load_state_dict(torch.load(MD/f'seq_fold{f}.pt',map_location='cpu',weights_only=True),strict=True);m.eval();models.append(m)
tf={t:f['fold'] for f in json.loads(Path('artifacts/folds.json').read_text()) for t in f['valid_tables']}
ix=pl.read_parquet('artifacts/evidence_session4/hand_index.parquet');N=['directed_transfer','soft_play','coordinated_isolation'];parts=[];t0=time.time();comparisons=[]
existing=pl.read_csv(MD/'dev_full.csv');risk=dict(zip(existing['pair_id'],existing['risk']))
with torch.no_grad():
    for ti,((table,),q) in enumerate(ix.group_by('table_id')):
        with np.load(TOK/f'{table}.npz') as z:
            offsets=np.r_[0,np.cumsum(z['n'])];x1=np.load(TOK/f'{table}_X.npy',mmap_mode='r');x2=np.load(TOK2/f'{table}.npy',mmap_mode='r');x3=np.load(TOK3/f'{table}.npy',mmap_mode='r')
            pmap={p:i for i,p in enumerate(z['pair_id']) if z['phase'][i]=='development'}
            for (pid,),bag in q.group_by('pair_id'):
                j=pmap[pid];a,b=offsets[j:j+2];raw=np.concatenate([x1[a:b],x2[a:b],x3[a:b]],1).astype('float32');x=(raw-mu)/sd;hids=z['hand_id'][a:b];n=len(x)
                assert set(bag['hand_id'])==set(hids)
                family=N.index(bag['behavior_family'][0]);model=models[tf[table]];acc=np.zeros((n,2));pbase=[]
                for swap in [False,True]:
                    xx=x.copy()
                    if swap:
                        xx=xx[:,perm];xx[:,seatd]=(6-xx[:,seatd]*sd[seatd]-mu[seatd])/sd[seatd]
                    X=torch.tensor(xx);base=model(X[None],torch.ones((1,n),dtype=torch.bool))[0]
                    pbase.append(float(torch.sigmoid(base[0])))
                    for start in ([] if os.environ.get('OCCLUSION_AUDIT_ONLY') else range(0,n,32)):
                        ids=np.arange(start,min(n,start+32));mask=torch.ones((len(ids),n),dtype=torch.bool);mask[np.arange(len(ids)),ids]=False
                        pred=model(X[None].expand(len(ids),-1,-1).contiguous(),mask)
                        acc[ids,0]+=(base[0]-pred[:,0]).numpy()/2
                        acc[ids,1]+=(base[family+1]-pred[:,family+1]).numpy()/2
                if n<=cfg['maxlen']:comparisons.append({'pair_id':pid,'replayed_risk':float(np.mean(pbase)),'stored_risk':float(risk[pid]),'difference':float(abs(np.mean(pbase)-risk[pid]))})
                parts.append(pl.DataFrame({'pair_id':[pid]*n,'hand_id':hids,'occlusion':acc[:,0],'family_occlusion':acc[:,1]}))
        if ti%40==0:print('occlusion tables',ti,'seconds',round(time.time()-t0,1),flush=True)
out=pl.concat(parts)
if not os.environ.get('OCCLUSION_AUDIT_ONLY'):out.write_parquet(OUT/'occlusion_oof.parquet')
report={'rows':len(out),'pairs':out['pair_id'].n_unique(),'baseline_matching_pairs':len(comparisons),'max_pair_risk_difference':max(c['difference'] for c in comparisons),'full_bags_used':True}
report['reproduction_passed']=report['max_pair_risk_difference']<2e-5
(OUT/'occlusion_audit.json').write_text(json.dumps(report,indent=2));pl.DataFrame(comparisons).write_csv(OUT/'occlusion_pair_replay.csv');print(report,flush=True)
assert report['reproduction_passed'], 'Frozen detector replay differs from archived predictions; do not promote these explanations as a verified R26 decomposition.'
