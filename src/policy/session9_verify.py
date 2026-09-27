import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
import hashlib,itertools,json
import numpy as np,polars as pl,torch
from session9_selector import ROOT,CONFIG,Selector,pack,set_nll
from session8_data import hand_data,reference
C=pl.col
def main():
    d=hand_data();ref=reference().select('pair_id','hand_id','r29');saved=pl.read_parquet(ROOT/'selector_oof.parquet');checks={'saved_replay_max_error':0.,'permutation_max_error':0.,'nested_reference_max_error':0.}
    for f in range(4):
        nested=pl.read_parquet(ROOT/f'nested_outer{f}.parquet')
        check=nested.filter(C('fold')==f).join(ref,on=['pair_id','hand_id'],suffix='_reference',validate='1:1')
        checks['nested_reference_max_error']=max(checks['nested_reference_max_error'],float((check['r29']-check['r29_reference']).abs().max()))
        for a in json.loads((ROOT/f'nested_outer{f}_audit.json').read_text()):
            assert f not in a['training_folds'] and a['inner'] not in a['training_folds']
        q=d.join(nested.drop('fold','time'),on=['pair_id','hand_id'],validate='1:1');bags=[pack(g) for _,g in q.group_by('pair_id')];bags.sort(key=lambda b:b['pair_id']);X=np.stack([b['X'] for b in bags]);tr=np.array([b['fold']!=f for b in bags]);mu=X[tr].reshape(-1,X.shape[-1]).mean(0);sd=np.maximum(.05,X[tr].reshape(-1,X.shape[-1]).std(0));valid=[b for b in bags if b['fold']==f];XX=torch.tensor(np.clip((X[~tr]-mu)/sd,-6,6));prior=np.stack([b['prior'] for b in valid]);perm=np.arange(19,-1,-1)
        g=q.filter(C('fold')==f).filter(C('pair_id')==valid[0]['pair_id']);b1=pack(g);b2=pack(g.reverse().with_columns((1-C('evidence')).alias('evidence'),pl.lit(999).alias('evidence_rank'),pl.lit(99).alias('subtype')))
        for key in ['hand','X','prior']:np.testing.assert_array_equal(b1[key],b2[key])
        for kind in ['pointwise','set']:
            predictions=[]
            for seed in CONFIG['seeds']:
                state=torch.load(ROOT/f'selector_{kind}_fold{f}_seed{seed}.pt',weights_only=False);assert state['config']==CONFIG;np.testing.assert_array_equal(mu,state['mu']);np.testing.assert_array_equal(sd,state['sd']);m=Selector(state['feature_count'],kind);m.load_state_dict(state['state_dict']);m.eval()
                with torch.no_grad():
                    delta=m(XX).numpy();reverse=m(XX[:,perm]).numpy()[:,perm]
                checks['permutation_max_error']=max(checks['permutation_max_error'],float(abs(delta-reverse).max()));predictions.append(prior+delta)
            pred=np.mean(predictions,axis=0)
            for i,b in enumerate(valid):
                expected=pl.DataFrame({'hand_id':b['hand']}).join(saved.filter(C('pair_id')==b['pair_id']),on='hand_id',validate='1:1',maintain_order='left')[kind+'_logit'].to_numpy();checks['saved_replay_max_error']=max(checks['saved_replay_max_error'],float(abs(pred[i]-expected).max()))
    assert checks['saved_replay_max_error']<1e-6 and checks['permutation_max_error']<2e-6 and checks['nested_reference_max_error']<1e-12
    score=torch.tensor([[.8,-.2,1.1,-1.3,.1,.5]],dtype=torch.float64)
    for k in range(1,6):
        y=torch.zeros_like(score);y[:,:k]=1;terms=torch.stack([score[0,list(s)].sum() for s in itertools.combinations(range(6),k)]);expected=(torch.logsumexp(terms,0)-score[:,:k].sum())/k;torch.testing.assert_close(set_nll(score,y)[0],expected)
    checks.update({'exact_set_likelihood':'passed exhaustive k=1..5 on six candidates','input_label_mutation':'features and predictions unchanged','training_normalization':'exactly training folds only','nested_teacher_fold_exclusions':'passed all recorded heads'})
    r=pl.read_csv(ROOT/'routed_evidence.csv').filter(C('window')=='full').join(pl.read_csv(ROOT/'selector_comparison.csv').select('pair_id','r29','pointwise','set'),on='pair_id',validate='1:1')
    for kind in ['pointwise','set']:r=r.with_columns(pl.when(C('below_gate')).then(C('routed_r29')).otherwise(C(kind)).alias(kind+'_routed'))
    assert (r['family']==r['routed_family']).all();names=['routed_r29','pointwise_routed','set_routed'];pool=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(991).integers(0,len(pool),(3000,len(pool)));result={}
    for name in names:
        diff=pool[name].to_numpy()-pool['routed_r29'].to_numpy();boot=diff[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);result[name]={'map5':r[name].mean(),'gain':r[name].mean()-r['routed_r29'].mean(),'fixed_predictions_pool_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),'improved_pairs':int((r[name]>r['routed_r29']+1e-12).sum()),'worse_pairs':int((r[name]<r['routed_r29']-1e-12).sum())}
    r.write_csv(ROOT/'full_pipeline_comparison.csv');(ROOT/'full_pipeline_comparison.json').write_text(json.dumps(result,indent=2))
                                                                          
    denominators=dict(d.group_by('pair_id').agg(C('evidence').sum()).iter_rows());loss_rows=[]
    for (pid,),g in saved.group_by('pair_id'):
        den=min(5,denominators[pid]);hands=g['hand_id'].to_numpy();y=g['evidence'].to_numpy().astype('float64');row={'pair_id':pid}
        for name in ['r29','pointwise','set']:
            order=np.lexsort((hands,-g[name+'_logit'].to_numpy()))[:5];hit=y[order];recall=float(hit.sum()/den);ap=float((hit*np.cumsum(hit)/np.arange(1,6)).sum()/den);row.update({name+'_recall5':recall,name+'_selection_loss':1-recall,name+'_ordering_loss':recall-ap})
            frozen=order[np.lexsort((hands[order],-g['r29_logit'].to_numpy()[order]))];hit=y[frozen];row[name+'_selection_r29_order']=float((hit*np.cumsum(hit)/np.arange(1,6)).sum()/den)
        loss_rows.append(row)
    loss=pl.DataFrame(loss_rows);loss.write_csv(ROOT/'selector_loss_accounting.csv');(ROOT/'selector_loss_accounting.json').write_text(json.dumps(loss.select(pl.selectors.numeric().mean()).to_dicts()[0],indent=2))
    r=r.join(loss.select('pair_id',*[n+'_selection_r29_order' for n in ['pointwise','set']]),on='pair_id',validate='1:1');followup={}
    for name in ['pointwise','set']:
        key=name+'_selection_r29_order';r=r.with_columns(pl.when(C('below_gate')).then(C('routed_r29')).otherwise(C(key)).alias(key));p=r.group_by('table_id').agg(C(key).sum(),C('routed_r29').sum(),pl.len().alias('n')).sort('table_id');diff=p[key].to_numpy()-p['routed_r29'].to_numpy();boot=diff[ix].sum(1)/p['n'].to_numpy()[ix].sum(1);followup[key]={'routed_map5':r[key].mean(),'gain':r[key].mean()-r['routed_r29'].mean(),'fixed_predictions_pool_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist()}
    (ROOT/'frozen_order_followup.json').write_text(json.dumps({'status':'Post-result diagnostic, not predeclared and not independent validation','comparison':followup},indent=2))
    sha=hashlib.sha256(open('artifacts/candidate_r29/submission.csv','rb').read()).hexdigest();assert sha=='0fe7747d01d1c64099b803fb830f995ae1b4ca055dd1333fd260475732d10c18';checks['r29_sha256']=sha;(ROOT/'verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps({'checks':checks,'full_pipeline':result},indent=2))
if __name__=='__main__':main()
