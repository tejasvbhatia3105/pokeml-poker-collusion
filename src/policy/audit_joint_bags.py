import ast,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.metrics import average_precision_score as ap

root=Path('artifacts/policy/joint_bags');bags=pd.read_csv(root/'bags.csv');ev=np.load(root/'evidence.npy');ids=pd.read_parquet(root/'hand_ids.parquet').hand_id.to_numpy()
folds=json.loads(Path('artifacts/folds.json').read_text());valid=bags.table_id.isin(folds[0]['valid_tables'])
records=[]
for name,folder in [('all_known',root),('withheld_soft_play',root/'withheld_soft_play')]:
    p=pd.read_csv(folder/'pair_oof_fold0.csv');assert set(p.pair_id)==set(bags[valid].pair_id)
    score=np.load(folder/'hand_scores_fold0.npy');maps=[]
    for row in bags.itertuples():
        values=score[row.start:row.end]
        if not valid.iloc[row.Index]:assert np.isnan(values).all();continue
        assert np.isfinite(values).all()
        if row.label:
            order=np.lexsort((ids[row.start:row.end],-values))[:5];rel=ev[row.start:row.end][order];value=float((np.cumsum(rel)*rel/np.arange(1,len(rel)+1)).sum()/min(5,ev[row.start:row.end].sum()));maps.append(dict(pair_id=row.pair_id,behavior=row.behavior,map5=value))
    pd.DataFrame(maps).to_csv(folder/'reference_order_evidence_fold0.csv',index=False)
    p=p.merge(bags[['pair_id','behavior']],on='pair_id');keep=np.ones(len(p),bool) if name=='all_known' else (p.truth==0)|(p.behavior==2)
    records.append(dict(model=name,pair_AP=ap(p.truth,p.risk),weighted_AP=ap(p.truth[keep],p.risk[keep],sample_weight=np.where(p.truth[keep]>0,1,50)),evidence_MAP5=float(np.mean([r['map5'] for r in maps])),heldout_evidence_MAP5=float(np.mean([r['map5'] for r in maps if r['behavior']==2])) if name!='all_known' else None))

                                                                                   
tree=ast.parse(Path('src/policy/train_joint_bags.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='JointBags');scope={'nn':nn,'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'JointBags','exec'),scope)
raw=np.load(root/'features.npy',mmap_mode='r');scale=np.load(root/'scaler_fold0.npz');q=bags[valid].iloc[0];z=(raw[q.start:q.end]-scale['median'])/scale['scale'];x=(np.sign(z)*np.log1p(abs(z))).clip(-8,8).astype(np.float32)
model=scope['JointBags'](x.shape[1]);model.load_state_dict(torch.load(root/'model_fold0.pt',weights_only=True));model.eval();rng=np.random.default_rng(10);perm=rng.permutation(len(x))
with torch.no_grad():
    a=model(torch.from_numpy(x[None]),torch.ones((1,len(x)),dtype=torch.bool));b=model(torch.from_numpy(x[perm][None]),torch.ones((1,len(x)),dtype=torch.bool))
    padded=np.concatenate([x,rng.normal(size=(17,x.shape[1])).astype(np.float32)]);mask=torch.arange(len(padded))[None]<len(x);c=model(torch.from_numpy(padded[None]),mask)
    assert torch.max(abs(a[0]-b[0]))<1e-5 and torch.max(abs(a[0]-c[0]))<1e-5
    assert torch.max(abs(a[2][:,perm]-b[2]))<1e-5
out=dict(pilot_results=records,masking_and_permutation_checks='passed',validation='One preselected player-pool fold only; fixed 40 epochs. No four-fold generalization claim.')
(root/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
