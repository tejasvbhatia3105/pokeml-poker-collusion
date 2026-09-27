import json
import numpy as np,polars as pl,torch,joblib
from session12_conditional_boost import ROOT,pack,objective
from session11_list_boost import infer
from session8_count_conditioning import conditioned
from session8_data import hand_data
from session6_priority import inclusion
def main():
    d=hand_data();cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];saved=pl.read_parquet(ROOT/'conditional_boost_oof.parquet');error=0.;permutation=0.
    assert not set(cols)&{'fold','evidence','evidence_rank','subtype','pair_id','hand_id'}
    for f in range(4):
        groups,x,big,P,M,T,V,D,fv,bid,pos=pack(d,f,cols);train=torch.tensor(np.flatnonzero((fv!=f)&V.any(1).numpy()));valid=np.flatnonzero(fv==f);va=np.isin(bid,valid);A=torch.zeros_like(P,requires_grad=True);original=objective(P,A,M,T,V,D,train);original.backward();assert torch.count_nonzero(A.grad[valid])==0;changed=T.clone();changed[valid]=1;DD=D.clone();DD[valid]=999;assert float(original.detach())==float(objective(P,A.detach(),M,changed,V,DD,train))
        for kind in ['full']:
            X=x if kind=='compact' else np.column_stack([x,big]);X=np.nan_to_num(X,nan=0,posinf=1e6,neginf=-1e6);m=joblib.load(ROOT/f'list_boost_{kind}_fold{f}.joblib');delta=infer(m,X);perm=np.random.default_rng(f).permutation(int(va.sum()));reversed=infer(m,X[va][perm]);permutation=max(permutation,float(np.max(abs(reversed-delta[va][perm]))))
            z=P.numpy().copy();z[bid,pos]+=delta
            for i in valid:
                g=groups[i];n=len(g);p=torch.softmax(torch.tensor(np.column_stack([np.zeros(n,np.float32),z[i,:n]])),1).numpy();s=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*conditioned(p[:,1:],m['minimums'][g['behavior_family'][0]]);expected=g.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],maintain_order='left',validate='1:1')['conditional_boost'].to_numpy();error=max(error,float(np.max(abs(s-expected))))
    assert error<1e-6 and permutation<1e-6
    audit=json.load(open(ROOT/'list_boost_audit.json'))
    for r in audit:assert np.max(np.diff([r['initial_loss']]+r['training_loss']))<1e-5
    report={'saved_prediction_replay_max_error':error,'permutation_max_error':permutation,'validation_objective_gradient':'zero','validation_label_mutation':'training loss invariant','training_line_search':'monotone','models_checked':4,'raw_feature_target_columns':'absent'};(ROOT/'conditional_boost_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
