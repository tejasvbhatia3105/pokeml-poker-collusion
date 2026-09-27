\
\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json,hashlib,time
from pathlib import Path
import numpy as np,polars as pl
import session163_generic_action_data as base
ROOT=Path('artifacts/evidence_session188_relationship_action_context');C=pl.col
FIELDS=['equity','partner_private_equity','partner_equity_minus_own','partner_alive','partner_last_aggressor','players_active','pot_odds','call_stack',
    'response_log_amount_bb','response_amount_pot_fraction','response_raise_increment_pot_fraction','response_call_coverage',
    'mw_own','mw_partner','mw_team','mw_call_edge','mw_information_gap','mw_partner_fold_gain','mw_fold_value',*[f'observed_class_{i}' for i in range(4)]]
FLOW=['net_bb','contribution_bb','won_share','folded','showdown','start_bb']

def stats(x,pair,hand,actor):
    x=np.asarray(x,np.float64);g=pair*2+actor;h=hand*2+actor;ng=int(pair.max()+1)*2;nh=int(hand.max()+1)*2
    gc=np.bincount(g,minlength=ng);hc=np.bincount(h,minlength=nh)
    gs=np.zeros((ng,x.shape[1]));hs=np.zeros((nh,x.shape[1]));gq=np.zeros_like(gs);hq=np.zeros_like(hs)
    for out,ids,val in [(gs,g,x),(hs,h,x),(gq,g,x*x),(hq,h,x*x)]:np.add.at(out,ids,val)
    result=[]
    for flip in [0,1]:
        gg=g^flip;hh=h^flip;count=gc[gg]-hc[hh];assert np.all(count>=0)
        mean=(gs[gg]-hs[hh])/np.maximum(count,1)[:,None];second=(gq[gg]-hq[hh])/np.maximum(count,1)[:,None]
        sd=np.sqrt(np.maximum(0,second-mean*mean));mean[count==0]=-2;sd[count==0]=0
        result.extend([mean,sd,count])
    return result

def check_stats(x,pair,hand,actor):
    result=stats(x,pair,hand,actor);swapped=stats(x,pair,hand,1-actor)
    for a,b in zip(result,swapped):np.testing.assert_array_equal(a,b)
    order=np.arange(len(x))[::-1];back=stats(x[order],pair[order],hand[order],actor[order])
    for a,b in zip(result,back):np.testing.assert_allclose(a,b[::-1],atol=2e-5,rtol=1e-6)
    samples=np.random.default_rng(18801).choice(len(x),min(64,len(x)),replace=False)
    for i in samples:
        for role in [0,1]:
            z=x[(pair==pair[i])&(hand!=hand[i])&(actor==(actor[i]^role))].astype(np.float64);mean,sd,count=result[role*3:role*3+3]
            expected=z.mean(0) if len(z) else np.full(x.shape[1],-2);std=z.std(0) if len(z) else np.zeros(x.shape[1])
            np.testing.assert_allclose(mean[i],expected,atol=2e-5,rtol=1e-6);np.testing.assert_allclose(sd[i],std,atol=2e-5,rtol=1e-6);assert count[i]==len(z)
                                                                         
    xx=np.arange(48,dtype=float).reshape(12,4);pp=np.repeat([0,1],6);hh=np.repeat(np.arange(6),2);aa=np.tile([0,1],6)
    old=stats(xx,pp,hh,aa);xx[hh==1]+=1000;new=stats(xx,pp,hh,aa)
    for a,b in zip(old,new):np.testing.assert_allclose(a[hh==1],b[hh==1],atol=1e-10,rtol=0)
    return result,dict(scalar_rows=len(samples),endpoint_role_reversal_exact=True,row_reversal_within_tolerance=True,current_hand_subtraction_exact=True)

def flows(d):
    out=np.zeros((len(d),2,6),np.float32)
    for table in sorted(d['table_id'].unique()):
        q=d.filter(C('table_id')==table)
        seats=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet');hands=pl.read_parquet(f'artifacts/compact/hands/table_id={table}/*.parquet',columns=['hand_id','big_blind'])
        for actor,player in enumerate(['player_1','player_2']):
            z=q.select('hand_index','hand_id',C(player).alias('player_id')).join(seats,on=['hand_id','player_id'],validate='m:1',maintain_order='left').join(hands,on='hand_id',validate='m:1',maintain_order='left')
            assert len(z)==len(q) and not z['net_chips'].null_count()
            out[z['hand_index'].to_numpy(),actor]=z.select(C('net_chips')/C('big_blind'),C('total_contribution')/C('big_blind'),C('won_share'),C('folded').cast(pl.Float32),C('went_to_showdown').cast(pl.Float32),C('starting_stack')/C('big_blind')).to_numpy()
    return out

def main():
    ROOT.mkdir(exist_ok=True);start=time.time();cfg=json.load(open(base.ROOT/'config.json'));d=pl.read_parquet(base.base.ROOT/'hands.parquet');m=pl.read_parquet(base.ROOT/'actions.parquet');x=np.load(base.ROOT/'x.npy',mmap_mode='r')
    _,pid=np.unique(d['pair_id'].to_numpy(),return_inverse=True);hand=m['hand_index'].to_numpy();pair=pid[hand];actor=m['actor'].to_numpy().astype(np.int64)
    v=x[:,[cfg['columns'].index(c) for c in FIELDS]];r,proof=check_stats(v,pair,hand,actor);own,sd,oc,other,other_sd,pc=r
    parts=[own,other,own-other,v-own,np.clip((v-own)/np.maximum(.1,sd),-10,10),np.log1p(oc)[:,None],np.log1p(pc)[:,None]]
    names=[f'relation_{role}_{c}' for role in ['own_mean','partner_mean','own_minus_partner','current_minus_own','current_z_own'] for c in FIELDS]+['relation_log_otherhand_own_actions','relation_log_otherhand_partner_actions']
    f=flows(d);fp=np.repeat(pid,2);fh=np.repeat(np.arange(len(d)),2);fa=np.tile([0,1],len(d));fr,fproof=check_stats(f.reshape(-1,6),fp,fh,fa)
                                                                            
    ownf=fr[0].reshape(len(d),2,6);otherf=fr[3].reshape(len(d),2,6);a=ownf[hand,actor];b=otherf[hand,actor]
    parts.extend([a,b,a-b]);names += [f'relation_flow_{role}_{c}' for role in ['own_mean','partner_mean','own_minus_partner'] for c in FLOW]
    extra=np.column_stack(parts).astype(np.float32);assert extra.shape==(159748,135) and np.isfinite(extra).all() and len(names)==135
    np.save(ROOT/'extra.npy',extra);np.save(ROOT/'hand_actor_flow.npy',f)
    config=dict(method=__doc__,columns=names,features=135,base_config_sha256=hashlib.sha256((base.ROOT/'config.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),extra_sha256=hashlib.sha256((ROOT/'extra.npy').read_bytes()).hexdigest(),query_labels_used=False,fitted_teachers=False)
    (ROOT/'config.json').write_text(json.dumps(config,indent=2));(ROOT/'verification.json').write_text(json.dumps(dict(actions=len(m),hands=len(d),pairs=len(np.unique(pid)),action_checks=proof,flow_checks=fproof,all_current_hand_actions_and_both_players_excluded=True),indent=2));print('COMPLETE',extra.shape,round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
