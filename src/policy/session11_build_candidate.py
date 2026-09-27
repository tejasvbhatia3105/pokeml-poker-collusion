\
\
\
\
import os,json,csv,hashlib
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl,torch,joblib
from session10_list_learning import Model,features
from session11_list_boost import infer
from session8_count_conditioning import conditioned
from session6_priority import inclusion
C=pl.col
ROOT=Path('artifacts/evidence_session11')
SOURCE=Path('artifacts/candidate_r29/submission.csv')
SHA='0fe7747d01d1c64099b803fb830f995ae1b4ca055dd1333fd260475732d10c18'
NAMES=['base','cat_primary','cat_secondary','hist_primary','hist_secondary','cat_inclusion','joint_inclusion','r29']
def models(method):
    if method=='coverage':return None
    if method in ['ensemble','equal_tree_conditional','equal_tree_family']:
        members={'ensemble':['tree_full','conditional3','nested6'],'equal_tree_conditional':['tree_full','conditional3'],'equal_tree_family':['tree_full','conditional_family']}[method]
        return {k:models(k) for k in members}
    if method.startswith('tree_'):return [joblib.load(ROOT/f'list_boost_{method[5:]}_fold{f}.joblib') for f in range(4)]
    folder=ROOT/method if method.startswith('conditional') else Path('artifacts/evidence_session10/nested6_model');out=[]
    for f in range(4):
        seeds=[]
        for seed in [1010,2020]:
            s=torch.load(folder/f'list_independent_fold{f}_seed{seed}.pt',weights_only=False);m=Model(s['feature_count'],'independent');m.load_state_dict(s['state_dict']);m.eval();seeds.append((m,s['mu'],s['sd'],s.get('minimums',{})))
        out.append(seeds)
    return out
def score(g,f,method,model):
    q=g.with_columns(*[C(f'{n}_{f}').alias(n) for n in NAMES]);x,prior=features(q)
    if method=='coverage':return q['r29'].to_numpy()
    if method in ['ensemble','equal_tree_conditional','equal_tree_family']:return np.mean([score(g,f,k,m) for k,m in model.items()],axis=0)
    if method.startswith('tree_'):
        m=model[f];X=x if method=='tree_compact' else np.column_stack([x,q.select(m['columns']).to_numpy().astype('float32')]);X=np.nan_to_num(X,nan=0,posinf=1e6,neginf=-1e6);deltas=[infer(m,X)]
    else:
        deltas=[]
        for m,mu,sd,minimums in model[f]:
            X=torch.tensor(np.clip((x-mu)/sd,-6,6))[None];mask=torch.ones((1,len(g)),dtype=torch.bool)
            with torch.no_grad():deltas.append(m(X,mask)[0].numpy())
    vals=[]
    for delta in deltas:
        z=torch.tensor(prior+delta);p=torch.softmax(torch.cat([torch.zeros_like(z[:,:1]),z],1),1).numpy()
        vals.append(conditioned(p[:,1:],model[f][0][3].get(g['behavior_family'][0],3)) if method.startswith('conditional') else inclusion(p[:,1],p[:,2]))
    return .25*q['base'].to_numpy()+.25*q['cat_inclusion'].to_numpy()+.5*np.mean(vals,axis=0)
def main():
    method=os.environ.get('CANDIDATE_METHOD','coverage');assert method in ['coverage','tree_full','tree_compact','conditional3','conditional_family','nested6','ensemble','equal_tree_conditional','equal_tree_family'];out=Path(os.environ.get('CANDIDATE_OUTPUT','artifacts/candidate_r30'));out.mkdir(exist_ok=True);assert not (out/'submission.csv').exists();assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
    with SOURCE.open(newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
    lookup={r['pair_id']:r for r in rows};model=models(method);parts=[];choices={};replay=0.
    old=pl.read_parquet('artifacts/candidate_r29/evidence_scores.parquet').select('pair_id','hand_id','base_score')
    for folder in [ROOT/'eval_cache',ROOT/'tail_cache']:
        assert (folder/'audit.json').exists()
        for path in sorted(folder.glob('T*.parquet')):
            d=pl.read_parquet(path)
            for (pid,),g in d.group_by('pair_id'):
                assert pid not in choices;g=g.sort('time','hand_id');risk=float(lookup[pid]['risk_score']);assert risk>=.01 and (g['time']>=.6).all();m=method if risk>=.05 else 'coverage';s=np.mean([score(g,f,m,model) for f in range(4)],axis=0);assert np.isfinite(s).all();base=np.mean([g[f'r29_{f}'].to_numpy() for f in range(4)],axis=0);z=g.select('pair_id','hand_id').with_columns(pl.Series('score',s),pl.Series('r29',base));parts.append(z);choices[pid]=z.sort('score','hand_id',descending=[True,False])['hand_id'].to_list()[:5]
    pred=pl.concat(parts);z=pred.join(old,on=['pair_id','hand_id'],validate='1:1');replay=float((z['r29']-z['base_score']).abs().max());assert replay<1e-10
    assert set(choices)=={r['pair_id'] for r in rows if r['predicted_behavior'] in ['directed_transfer','soft_play','coordinated_isolation']}
    ecols=[f'evidence_hand_{i}' for i in range(1,6)];changed=0;tailchanged=0
    for r in rows:
        pid=r['pair_id']
        if pid not in choices:continue
        h=choices[pid]+['NO_EVIDENCE']*(5-len(choices[pid]));diff=any(r[k]!=v for k,v in zip(ecols,h));changed+=diff;tailchanged+=diff and float(r['risk_score'])<.05
        for k,v in zip(ecols,h):r[k]=v
    with (out/'submission.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    with SOURCE.open(newline='') as f:
        for oldrow,newrow in zip(csv.DictReader(f),rows):assert all(oldrow[k]==newrow[k] for k in fields if k not in ecols)
    pred.write_parquet(out/'evidence_scores.parquet');report={'status':'unscored; requires user upload','method':method,'coverage':'every already-classified pair, risk >= .01; established R29 for .01-.05','rows':len(rows),'rescored_pairs':len(choices),'changed_evidence_rows':changed,'changed_tail_rows':tailchanged,'source_sha256':SHA,'r29_replay_max_error':replay,'non_evidence_strings_unchanged':True,'sha256':hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()};(out/'build_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
