\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','3')
import json,itertools,time
from pathlib import Path
import numpy as np,polars as pl,torch,joblib
from catboost import CatBoostClassifier
from session55_current_targets import state
from session8_data import hand_data
from session11_conditional_family import Model,features
from session65_list_pressure_inference import tree_delta
from session6_priority import inclusion
from session8_count_conditioning import conditioned

ROOT=Path('artifacts/evidence_session149_persistent_donor');C=pl.col;torch.set_num_threads(2)

def probabilities(ca,hp,delta):
    jp=.5*(ca+hp);prior=np.log(np.maximum(jp,1e-6))-np.log(np.maximum(1-jp.sum(1),1e-6))[:,None]
    z=torch.tensor(prior,dtype=torch.float32)+torch.as_tensor(delta,dtype=torch.float32)
    return torch.softmax(torch.cat([torch.zeros_like(z[:,:1]),z],1),1).numpy()[:,1:]

def math_check():
    rng=np.random.default_rng(14901);p=rng.dirichlet([4,1,1],size=(2,6));w=np.array([.7,.3]);result=np.zeros(6);mass=np.zeros(2)
    for actor in range(2):
        num=np.zeros(6)
        for path in itertools.product(range(3),repeat=6):
            selected=([i for i,c in enumerate(path) if c==1]+[i for i,c in enumerate(path) if c==2])[:5]
            if len(selected)<3:continue
            v=float(np.prod(p[actor,np.arange(6),path]));mass[actor]+=v;num[selected]+=v
        result+=w[actor]*num/mass[actor]
    actual=sum(w[k]*conditioned(p[k,:,1:],3) for k in range(2));err=float(abs(actual-result).max());assert err<1e-12
    return dict(paths_per_actor=3**6,actors=2,minimum=3,max_probability_error=err,weights_interpretation='fixed donor probabilities conditional on the ascertained positive-pair population; no second count-tail update to donor weights')

def main():
    ROOT.mkdir(exist_ok=True);(ROOT/'PROTOCOL.md').write_text('''# Persistent-donor inference ablation

Keep the current R33 models and all donor probabilities fixed. Current donor weights are mixed into per-hand Cat event probabilities before the capped-list calculation. Compare with a coherent mixture that calculates each donor's conditional inclusion first, then averages it with those same weights. Do not update donor weights by the count-tail probability: the donor classifier was trained on ascertained positive relationships.

For both R33 branches retain their ORIGINAL frozen correction vectors, computed on their existing averaged inputs. Change only the Cat event probabilities supplied to inclusion and corrected joint Cat/HGB priors. The histogram probabilities stay unchanged. No teacher retraining, new model, post-result splice, seed or weight search. Other families retain R33. Replay current donor-weighted Cat probabilities, both R33 branch scores and the final R33 mixture before measuring new results.

Earlier93 separately conditioned donors with a newly trained, weaker joint action model. It did not test this inference change on the current R33 models. This ablation is therefore narrower; success or failure does not prove a general statement about all persistent-donor models.
''');(ROOT/'math_check.json').write_text(json.dumps(math_check(),indent=2))
    full=hand_data();v=state()['directed_transfer'];d=v['d'];a=v['a'];x=v['x'];g=a['row'].to_numpy();actor=a['actor'].to_numpy();af=d['fold'].to_numpy()[g];primary=np.zeros((len(d),2))
    for f in range(4):
        m=CatBoostClassifier();m.load_model(f'artifacts/evidence_session50_matchup/current/directed_transfer/event1_fold{f}.cbm');use=af==f;primary[g[use],actor[use]]=m.predict_proba(x[use],thread_count=2)[:,1]
    sec=d.select('pair_id','hand_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/conditional_oof.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left').select('actor0_event1','actor1_event1').to_numpy()
    byhand=d.select('pair_id','hand_id').with_columns(*[pl.Series(f'actor{k}_{name}',arr[:,k]) for k in range(2) for name,arr in [('primary',primary),('secondary',sec)]]);dw=pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1')
    gc=json.load(open('artifacts/evidence_session62_grounded_list_boost/grounded_columns.json'));gx=np.load('artifacts/evidence_session62_grounded_list_boost/grounded_features.npz')['x'];assert len(gx)==len(full)
    full=full.with_columns(*[pl.Series(c,gx[:,j]) for j,c in enumerate(gc)]);baseline=pl.read_parquet('artifacts/evidence_session64_equal_list_pressure/oof.parquet').select('pair_id','hand_id','equal','pressure59','full')
    native=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet').select('pair_id','hand_id','bg_primary','bg_secondary')
    parts=[];audit=[];start=time.time()
    for f in range(4):
        metadata=full.filter((C('fold')==f)&(C('behavior_family')=='directed_transfer'))
        current=pl.read_parquet(f'artifacts/evidence_session55_current_nested/nested_outer{f}.parquet').drop('time','fold');old=pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').select('pair_id','hand_id',*[C(c).alias('old_'+c) for c in ['base','cat_primary','cat_secondary','hist_primary','hist_secondary','cat_inclusion','joint_inclusion','r29']])
        q=metadata.join(current,on=['pair_id','hand_id'],validate='1:1').join(old,on=['pair_id','hand_id'],validate='1:1').join(byhand,on=['pair_id','hand_id'],validate='1:1').join(dw,on='pair_id',validate='m:1').join(baseline,on=['pair_id','hand_id'],validate='1:1').join(native,on=['pair_id','hand_id'],validate='1:1')
        corrections=[]
        for seed in [1010,2020]:
            checkpoint=torch.load(f'artifacts/evidence_session11/conditional_family/list_independent_fold{f}_seed{seed}.pt',weights_only=False);m=Model(35,'independent');m.load_state_dict(checkpoint['state_dict']);m.eval();corrections.append((m,checkpoint))
        tree=joblib.load(f'artifacts/evidence_session62_grounded_list_boost/list_boost_full_fold{f}.joblib');errors=[]
        for (pid,),z in q.group_by('pair_id'):
            z=z.sort('time','hand_id');w=z.select('actor0','actor1').to_numpy()[0];cc=np.stack([z.select(f'actor{k}_primary',f'actor{k}_secondary').to_numpy() for k in range(2)]);raw=(cc*w[:,None,None]).sum(0)
            event_error=float(abs(raw-z.select('bg_primary','bg_secondary').to_numpy()).max());assert event_error<1e-12
            ca=raw/np.maximum(1,raw.sum(1))[:,None];hp=z.select('hist_primary','hist_secondary').to_numpy();hp=hp/np.maximum(1,hp.sum(1))[:,None];ci=inclusion(*ca.T)
            original=z.with_columns(*[C('old_'+c).alias(c) for c in ['base','cat_primary','cat_secondary','hist_primary','hist_secondary','cat_inclusion','joint_inclusion','r29']]);xo,_=features(original);deltas=[]
            with torch.no_grad():
                for m,st in corrections:deltas.append(m(torch.tensor(np.clip((xo-st['mu'])/st['sd'],-6,6))[None],torch.ones((1,len(z)),dtype=torch.bool))[0].numpy())
            minimum=corrections[0][1]['minimums']['directed_transfer'];xf,_=features(z);xx=np.nan_to_num(np.column_stack([xf,z.select(gc).to_numpy()]),nan=0,posinf=1e6,neginf=-1e6);td=tree_delta(tree,xx)
            p_inc=np.mean([conditioned(probabilities(ca,hp,delta),minimum) for delta in deltas],0);t_inc=conditioned(probabilities(ca,hp,td),tree['minimums']['directed_transfer']);base=z['base'].to_numpy();bp=.25*base+.25*ci+.5*p_inc;bt=.25*base+.25*ci+.5*t_inc
            error=max(float(abs(bp-z['pressure59'].to_numpy()).max()),float(abs(bt-z['full'].to_numpy()).max()),float(abs(.5*(bp+bt)-z['equal'].to_numpy()).max()));assert error<1e-7,(f,pid,error)
            new_ci=np.zeros(len(z));new_pi=np.zeros(len(z));new_ti=np.zeros(len(z))
            for k in range(2):
                cp=cc[k]/np.maximum(1,cc[k].sum(1))[:,None];new_ci+=w[k]*inclusion(*cp.T);new_pi+=w[k]*np.mean([conditioned(probabilities(cp,hp,delta),minimum) for delta in deltas],0);new_ti+=w[k]*conditioned(probabilities(cp,hp,td),tree['minimums']['directed_transfer'])
            new=.25*base+.25*new_ci+.25*new_pi+.25*new_ti
            parts.append(z.select('pair_id','hand_id').with_columns(pl.Series('donor_integrated',new)));errors.append(dict(pair_id=pid,cat_event_replay_error=event_error,R33_branch_replay_error=error,maximum_donor_weight=float(w.max()),score_max_change=float(abs(new-z['equal'].to_numpy()).max())))
        audit.append(dict(fold=f,pairs=errors));print('DONOR_INFERENCE',f,round(time.time()-start,1),flush=True)
    out=baseline.join(pl.concat(parts),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(C('donor_integrated').fill_null(C('equal'))).rename({'equal':'r33'});out.write_parquet(ROOT/'oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2));print('COMPLETE',time.time()-start,flush=True)

if __name__=='__main__':main()
