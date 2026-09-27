\
\
\
\
\
import os,json,time,itertools
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl,torch
from torch import nn
from session8_data import hand_data
ROOT=Path('artifacts/evidence_session9');C=pl.col;torch.set_num_threads(4)
FAMILIES=['directed_transfer','soft_play','coordinated_isolation']
PHYSICAL=['fold_partner','call_partner','check_hu','both_showdown','strong_fold_partner','weak_call_partner','strong_check_hu','weak_agg_out','team_net','weak_net','strong_net','min_contrib','max_contrib','pot_fraction']
CONFIG={'shortlist':20,'seeds':[991,1991,2991],'epochs':80,'batch_size':32,'learning_rate':.001,'weight_decay':.01,'residual_l2':.03,'max_log_odds_adjustment':1.5,'hidden_dim':32,'attention_heads':4,'dropout':.1,'loss':'conditional Bernoulli exact-set negative log likelihood divided by number of shortlist positives','selection':'No early stopping, parameter sweep, or held-out checkpoint selection'}
def logit(p):p=np.clip(p,1e-5,1-1e-5);return np.log(p)-np.log1p(-p)
def pack(g):
    g=g.sort('time','hand_id');n=len(g);assert n>=20
    scores=g.select('base','cat_primary','cat_secondary','hist_primary','hist_secondary','cat_inclusion','joint_inclusion','r29').to_numpy();a=scores[:,1];b=scores[:,2];h=scores[:,3];j=scores[:,4];ca=a/np.maximum(1,a+b);ha=.5*ca+.5*h/np.maximum(1,h+j);other=g.select(PHYSICAL).to_numpy();other=np.sign(other)*np.log1p(abs(other));rank=np.lexsort((np.array(g['hand_id']),-scores[:,7]));rr=np.empty(n);rr[rank]=np.arange(n)/max(n-1,1)
    extra=np.column_stack([g['relative_time'].to_numpy(),np.arange(n)/max(n-1,1),np.full(n,np.log1p(n)),rr,(np.cumsum(ca)-ca)/5,(ca.sum()-np.cumsum(ca))/5,(np.cumsum(ha)-ha)/5,(ha.sum()-np.cumsum(ha))/5,np.full(n,ca.sum()/5),np.full(n,ha.sum()/5)])
    fam=np.tile(np.eye(3)[FAMILIES.index(g['behavior_family'][0])],(n,1));X=np.column_stack([logit(scores),extra,other,fam]).astype('float32');ix=rank[:20]
    return {'pair_id':g['pair_id'][0],'table_id':g['table_id'][0],'family':g['behavior_family'][0],'fold':int(g['fold'][0]),'hand':np.array(g['hand_id'])[ix],'X':X[ix],'y':g['evidence'].to_numpy()[ix].astype('float32'),'prior':logit(scores[ix,7]).astype('float32'),'den':int(g['evidence'].sum())}
class Selector(nn.Module):
    def __init__(self,nfeatures,kind):
        super().__init__();self.kind=kind;self.local=nn.Sequential(nn.Linear(nfeatures,48),nn.GELU(),nn.Dropout(.1),nn.Linear(48,32),nn.GELU())
        if kind=='set':self.attention=nn.MultiheadAttention(32,4,dropout=.1,batch_first=True);self.norm=nn.LayerNorm(32)
        self.out=nn.Linear(32,1);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x):
        h=self.local(x)
        if self.kind=='set':a,_=self.attention(h,h,h,need_weights=False);h=self.norm(h+a)
        return CONFIG['max_log_odds_adjustment']*torch.tanh(self.out(h).squeeze(-1))
def set_nll(score,y):
                                                                                 
    z=torch.cat([torch.zeros_like(score[:,:1]),torch.full_like(score[:,:5],-10000.)],1)
    for i in range(score.shape[1]):z=torch.cat([z[:,:1],torch.logaddexp(z[:,1:],z[:,:-1]+score[:,i:i+1])],1)
    k=y.sum(1).long();assert int(k.max())<=5
    return (z.gather(1,k[:,None]).squeeze(1)-(score*y).sum(1))/k.clamp_min(1)
def metric(hand,y,den,score):
    ix=np.lexsort((hand,-score))[:5];v=y[ix];return float((v*np.cumsum(v)/np.arange(1,len(v)+1)).sum()/min(5,den))
def main():
    config=ROOT/'selector_config.json'
    if config.exists():assert json.loads(config.read_text())==CONFIG
    else:config.write_text(json.dumps(CONFIG,indent=2))
    d=hand_data();rows=[];audit=[];allpred=[];start=time.time()
    for f in range(4):
        q=d.join(pl.read_parquet(ROOT/f'nested_outer{f}.parquet').drop('fold','time'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');bags=[pack(g) for _,g in q.group_by('pair_id')];bags.sort(key=lambda b:b['pair_id']);X=np.stack([b['X'] for b in bags]);Y=np.stack([b['y'] for b in bags]);prior=np.stack([b['prior'] for b in bags]);fv=np.array([b['fold'] for b in bags]);tr=fv!=f;va=~tr;mu=X[tr].reshape(-1,X.shape[-1]).mean(0);sd=np.maximum(.05,X[tr].reshape(-1,X.shape[-1]).std(0));XX=torch.tensor(np.clip((X-mu)/sd,-6,6));YY=torch.tensor(Y);PP=torch.tensor(prior);train_ix=np.flatnonzero(tr);valid_ix=np.flatnonzero(va);preds={}
        assert np.isfinite(X).all() and np.all(fv[train_ix]!=f)
        audit.append({'outer':f,'train_pairs':int(tr.sum()),'validation_pairs':int(va.sum()),'train_top20_recall':float(np.mean([b['y'].sum()/b['den'] for b in bags if b['fold']!=f])),'validation_top20_recall':float(np.mean([b['y'].sum()/b['den'] for b in bags if b['fold']==f]))})
        for kind in ['pointwise','set']:
            predictions=[]
            for seed in CONFIG['seeds']:
                torch.manual_seed(seed+f);rng=np.random.default_rng(seed+f);model=Selector(X.shape[-1],kind);opt=torch.optim.AdamW(model.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay']);model.eval()
                with torch.no_grad():assert torch.count_nonzero(model(XX[valid_ix[:2]])).item()==0
                for epoch in range(CONFIG['epochs']):
                    model.train();order=rng.permutation(train_ix)
                    for off in range(0,len(order),CONFIG['batch_size']):
                        ix=order[off:off+CONFIG['batch_size']];delta=model(XX[ix]);loss=set_nll(PP[ix]+delta,YY[ix]).mean()+CONFIG['residual_l2']*delta.square().mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                model.eval()
                with torch.no_grad():p=(PP[valid_ix]+model(XX[valid_ix])).numpy();predictions.append(p)
                torch.save({'state_dict':model.state_dict(),'kind':kind,'feature_count':X.shape[-1],'mu':mu,'sd':sd,'config':CONFIG,'outer':f,'seed':seed},ROOT/f'selector_{kind}_fold{f}_seed{seed}.pt')
            preds[kind]=np.mean(predictions,axis=0);print('selector',f,kind,'seconds',round(time.time()-start,1),flush=True)
        for k,ix in enumerate(valid_ix):
            b=bags[ix];row={x:b[x] for x in ['pair_id','table_id','family','fold']};row['den']=b['den'];row['shortlist_recall']=float(b['y'].sum()/b['den'])
            for name,p in [('r29',b['prior']),('pointwise',preds['pointwise'][k]),('set',preds['set'][k])]:row[name]=metric(b['hand'],b['y'],b['den'],p)
            rows.append(row);allpred.append(pl.DataFrame({'pair_id':[b['pair_id']]*20,'hand_id':b['hand'],'fold':[f]*20,'evidence':b['y'],'r29_logit':b['prior'],'pointwise_logit':preds['pointwise'][k],'set_logit':preds['set'][k]}))
    pl.DataFrame(rows).write_csv(ROOT/'selector_comparison.csv');pl.concat(allpred).write_parquet(ROOT/'selector_oof.parquet');(ROOT/'selector_training_audit.json').write_text(json.dumps(audit,indent=2));r=pl.DataFrame(rows);names=['r29','pointwise','set'];pool=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(991).integers(0,len(pool),(3000,len(pool)));report={}
    for name in names:
        delta=pool[name].to_numpy()-pool['r29'].to_numpy();boot=delta[ix].sum(1)/pool['n'].to_numpy()[ix].sum(1);report[name]={'map5':r[name].mean(),'gain':r[name].mean()-r['r29'].mean(),'ci95_fixed_predictions':np.quantile(boot,[.025,.975]).tolist(),'folds':r.group_by('fold').agg(C(name).mean()).sort('fold')[name].to_list(),'families':dict(r.group_by('family').agg(C(name).mean()).iter_rows())}
    (ROOT/'selector_comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
