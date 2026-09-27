import sys
from pathlib import Path
import polars as pl,numpy as np
names=['none','directed_transfer','soft_play','coordinated_isolation']
def geo(paths,out,ws,template=None):
 fr=None
 for k,p in paths.items():
  d=pl.read_csv(p).select('pair_id',*names).sort('pair_id').rename({n:f'{n}_{k}' for n in names});fr=d if fr is None else fr.join(d,on='pair_id')
 if template is not None:fr=fr.join(template,on='pair_id')
 tot=sum(ws.values());r=np.exp(sum(ws[k]*np.log(np.clip(1-fr[f'none_{k}'].to_numpy(),1e-6,1)) for k in paths)/tot)
 fam=sum(ws[k]*fr.select([f'{n}_{k}' for n in names[1:]]).to_numpy() for k in paths)/tot;fam=fam/np.maximum(fam.sum(1,keepdims=True),1e-12)
 pl.DataFrame({'pair_id':fr['pair_id'],'none':1-r,**{n:r*fam[:,j] for j,n in enumerate(names[1:])}}).write_csv(out);return int((r>0.5).sum())
if __name__=='__main__':
 out=Path(sys.argv[1] if len(sys.argv)>1 else 'artifacts/candidate_r12');out.mkdir(exist_ok=True,parents=True)
 n=geo({'a':str(out/'eval_r4s_all.csv'),'b':str(out/'eval_r4k_all.csv')},str(out/'pair_eval.csv'),{'a':1,'b':1},pl.read_csv('data/evaluation_pairs.csv').select('pair_id'))
 print('R12 eval >0.5',n)
