import hashlib,json
import numpy as np
import polars as pl
import torch
import session116_relationship_encoder as s
C=pl.col

def main():
    cfg,x,b,h=s.load();device=torch.device('mps');assert torch.backends.mps.is_available()
    plans=json.loads((s.DATA/'reference_plan.json').read_text());records=[]
    output_cols=['local','context','pair_risk','family_0','family_1','family_2']
    for plan in plans:
        f,g=plan['excluded_folds'];name=f'exclude{f}{g}';path=s.ROOT/(name+'.pt');audit=json.loads((s.ROOT/(name+'.json')).read_text())
        assert audit['reference_plan']==plan and audit['manifest']['config']==s.CONFIG
        assert audit['manifest']['source_sha256']==hashlib.file_digest(open(s.__file__,'rb'),'sha256').hexdigest()
        assert audit['checkpoint_sha256']==hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        state=torch.load(path,map_location='cpu',weights_only=False);assert state['config']==s.CONFIG and state['data_config']==cfg and state['excluded_folds']==[f,g]
        tr=np.flatnonzero(~b['fold'].is_in([f,g]).to_numpy());va=np.flatnonzero(b['fold'].is_in([f,g]).to_numpy())
        mu,sd,rows=s.normalization(x,b,tr);np.testing.assert_array_equal(mu,state['mu']);np.testing.assert_array_equal(sd,state['sd'])
        bm=b.with_columns(pl.when(C('fold').is_in([f,g])).then(1-C('label')).otherwise(C('label')).alias('label'),
            pl.when(C('fold').is_in([f,g])).then(pl.lit('mutated')).otherwise(C('behavior_family')).alias('behavior_family'))
        np.testing.assert_array_equal(b[tr].to_numpy(),bm[tr].to_numpy())
        held=set(b[va]['pair_id']);hm=h.with_columns(pl.when(C('pair_id').is_in(list(held))).then(1-C('evidence')).otherwise(C('evidence')).alias('evidence'))
        np.testing.assert_array_equal(h[rows].to_numpy(),hm[rows].to_numpy())
        model=s.Model(x.shape[1],cfg['columns'].index('trel')).to(device);model.load_state_dict(state['state_dict'])
        got=s.predict(model,x,b,h,va,mu,sd,cfg,device);saved=pl.read_parquet(s.ROOT/(name+'.parquet'))
        np.testing.assert_array_equal(got.select('row','pair_id','hand_id').to_numpy(),saved.select('row','pair_id','hand_id').to_numpy())
        err=float(abs(got.select(output_cols).to_numpy()-saved.select(output_cols).to_numpy()).max());assert err<1e-5,err
        chosen=va[:2];xx,mask,*_=s.batch(x,b,h,chosen,mu,sd,cfg);xx=xx.to(device);mask=mask.to(device)
        model.eval()
        with torch.no_grad():
            one=model(xx,mask);changed=xx.clone();changed[:,1:]*=-1;two=model(changed,mask)
            localerr=float(abs(torch.sigmoid(one[1][:,0])-torch.sigmoid(two[1][:,0])).max().cpu())
            contexterr=float(abs(torch.sigmoid(one[2][:,0])-torch.sigmoid(two[2][:,0])).max().cpu());assert localerr<1e-6
                                                                                      
        single=s.predict(model,x,b,h,va[:1],mu,sd,cfg,device);expected=single.select('row').join(saved,on='row',maintain_order='left',validate='1:1')
        paddingerr=float(abs(single.select(output_cols).to_numpy()-expected.select(output_cols).to_numpy()).max());assert paddingerr<1e-5
        records.append(dict(reference=name,training_pairs=len(tr),heldout_pairs=len(va),scored_hands=len(got),normalization_error=0,
            heldout_pair_and_hand_target_mutations=True,model_replay_max_error=err,padding_max_error=paddingerr,
            local_context_mutation_error=localerr,context_mutation_difference=contexterr))
        print('RELATIONSHIP_VERIFY',name,err,paddingerr,flush=True);del model;torch.mps.empty_cache()
    report=dict(references=len(records),records=records,max_prediction_replay_error=max(r['model_replay_max_error'] for r in records),
        max_padding_error=max(r['padding_max_error'] for r in records),all_hand_queries_preserved=True,
        limitation='Local head precedes the transformer; context head also has additional depth. This comparison does not isolate attention from model depth.')
    (s.ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
