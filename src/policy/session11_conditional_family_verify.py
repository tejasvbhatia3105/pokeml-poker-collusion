import os,json,hashlib
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl,torch
from session8_data import hand_data,reference
from session11_conditional_family import ROOT,Model,features,template,CONFIG,conditional_nll,log_count_at_least
from session8_count_conditioning import conditioned
from pathlib import Path
import itertools
from session6_priority import inclusion
C=pl.col
def enumeration():
    error=0.;count=0
    for minimum in [3,5]:
        p=np.random.default_rng(1111).dirichlet([3,1,1],size=6);found={};mass=0
        for cat in itertools.product(range(3),repeat=6):
            e=tuple(([i for i,c in enumerate(cat) if c==1]+[i for i,c in enumerate(cat) if c==2])[:5]);prob=np.prod(p[np.arange(6),cat])
            if len(e)>=minimum:found[e]=found.get(e,0)+prob;mass+=prob
        logits=torch.tensor(np.log(p[:,1:]/p[:,:1]))[None];mask=torch.ones((1,6),dtype=torch.bool);K=torch.tensor([minimum])
        for e,prob in found.items():
            z=template(6,np.array(e));loss=conditional_nll(logits,torch.tensor(z)[None],torch.ones((1,len(z)),dtype=torch.bool),torch.tensor([len(e)]),mask,K);error=max(error,abs(np.exp(-loss.item()*len(e))-prob/mass));count+=1
        pad=torch.cat([logits,torch.randn(1,3,2,dtype=logits.dtype)],1);paddedmask=torch.cat([mask,torch.zeros((1,3),dtype=torch.bool)],1);assert torch.allclose(log_count_at_least(logits,mask,K),log_count_at_least(pad,paddedmask,K),atol=1e-12)
        x=logits.clone().requires_grad_(True);assert torch.autograd.gradcheck(lambda x:log_count_at_least(x,mask,K),(x,))
    assert error<1e-12
    print('family conditional probability enumeration',count,'error',error,flush=True)
    return {'observed_lists':count,'max_probability_error':error,'padding':'invariant','gradient':'finite-difference check passed'}

def main():
    checks=enumeration()
    folder=ROOT;d=hand_data();saved=pl.read_parquet(folder/'list_learning_oof.parquet');ref=reference().select('pair_id','hand_id','r29');maxerr=0.;permutation_error=0.;validation_errors=[]
    for f in range(4):
        nested=pl.read_parquet(Path('artifacts/evidence_session10/nested6')/f'nested_outer{f}.parquet');val=nested.filter(C('fold')==f).join(ref,on=['pair_id','hand_id'],validate='1:1',suffix='_expected');err=float((val['r29']-val['r29_expected']).abs().max());assert err==0;validation_errors.append(err);q=d.join(nested.drop('fold','time'),on=['pair_id','hand_id'],validate='1:1');bags=[]
        for (pid,),g in q.group_by('pair_id'):
            g=g.sort('time','hand_id');x,p=features(g);e=np.flatnonzero(g['evidence_rank'].is_not_null());e=e[np.argsort(g['evidence_rank'].to_numpy()[e])];bags.append({'pid':pid,'g':g,'X':x,'prior':p,'fold':g['fold'][0],'compatible':len(template(len(g),e))>0})
        bags.sort(key=lambda b:b['pid']);n=max(len(b['g']) for b in bags);N=len(bags);nf=bags[0]['X'].shape[1];X=np.zeros((N,n,nf),dtype='float32');P=np.zeros((N,n,2),dtype='float32');M=np.zeros((N,n),bool)
        for i,b in enumerate(bags):k=len(b['g']);X[i,:k]=b['X'];P[i,:k]=b['prior'];M[i,:k]=True
        tr=np.array([b['fold']!=f and b['compatible'] for b in bags]);va=np.array([b['fold']==f for b in bags]);mu=X[tr][M[tr]].mean(0);sd=np.maximum(.05,X[tr][M[tr]].std(0));XX=torch.tensor(np.clip((X[va]-mu)/sd,-6,6));PP=torch.tensor(P[va]);MM=torch.tensor(M[va]);pred=[]
        for seed in CONFIG['seeds']:
            state=torch.load(folder/f'list_independent_fold{f}_seed{seed}.pt',weights_only=False);assert state['config']==CONFIG;minimums={fam:min(int(b['g']['evidence'].sum()) for b in bags if b['fold']!=f and b['compatible'] and b['g']['behavior_family'][0]==fam) for fam in ['directed_transfer','soft_play','coordinated_isolation']};assert state['minimums']==minimums;np.testing.assert_array_equal(mu,state['mu']);np.testing.assert_array_equal(sd,state['sd']);m=Model(nf,'independent');m.load_state_dict(state['state_dict']);m.eval()
            with torch.no_grad():
                delta=m(XX,MM);rev=m(XX.flip(1),MM.flip(1)).flip(1);permutation_error=max(permutation_error,float(abs(delta-rev).max()));logits=PP+delta;pred.append(torch.softmax(torch.cat([torch.zeros_like(logits[:,:,:1]),logits],2),2).numpy())
        for k,b in enumerate([b for b in bags if b['fold']==f]):
            g=b['g'];count=len(g);inc=np.mean([conditioned(p[k,:count,1:3],minimums[g['behavior_family'][0]]) for p in pred],axis=0);score=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc;expected=g.select('pair_id','hand_id').join(saved,on=['pair_id','hand_id'],validate='1:1',maintain_order='left')['independent'].to_numpy();maxerr=max(maxerr,float(abs(score-expected).max()));mutated=g.with_columns((1-C('evidence')).alias('evidence'),pl.lit(999).alias('evidence_rank'),pl.lit(99).alias('subtype'));mx,mp=features(mutated);np.testing.assert_array_equal(mx,b['X']);np.testing.assert_array_equal(mp,b['prior'])
    inp=json.load(open(Path('artifacts/evidence_session10/nested6/input_audit.json')));assert inp['heads_audited']==144 and len(inp['validation_replay_max_errors'])==4;assert maxerr<1e-6 and permutation_error<3e-6;sha=hashlib.sha256(open('artifacts/candidate_r29/submission.csv','rb').read()).hexdigest();assert sha=='0fe7747d01d1c64099b803fb830f995ae1b4ca055dd1333fd260475732d10c18';result={'conditional_likelihood_checks':checks,'saved_replay_max_error':maxerr,'permutation_error':permutation_error,'validation_reference_errors':validation_errors,'training_normalization':'exact replay, outer validation excluded','input_labels':'mutation leaves features unchanged','teacher_event_heads_audited':inp['heads_audited'],'r29_sha256':sha};(folder/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
