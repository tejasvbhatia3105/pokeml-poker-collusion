import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl,torch
from torch import nn
from session11_conditional_family import Model,features,template,conditional_nll
from session8_data import hand_data
from session8_count_conditioning import conditioned
ROOT=Path('artifacts/evidence_session12/frozen_adapter');C=pl.col;torch.set_num_threads(3)
CONFIG={'steps':150,'learning_rate':.01,'bound':.75,'residual_l2':.1,'weight_decay':.05,'seeds':[1010,2020],'architecture':'zero-initialized linear logit adapter; R30 network frozen','features':'compact control vs compact plus frozen action embeddings interacted with family','schedule':'fixed full-batch steps, no held-out checkpoint selection'}
def main():
    ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));d=hand_data();parts=[];audit=[];start=time.time()
    for f in range(4):
        raw=pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').drop('fold','time');emb=pl.read_parquet(f'artifacts/evidence_session12/action_embeddings/trained_outer{f}.parquet');q=d.join(raw,on=['pair_id','hand_id'],validate='1:1').join(emb,on=['pair_id','hand_id'],validate='1:1');bags=[]
        for (pid,),g in q.group_by('pair_id'):
            g=g.sort('time','hand_id');x,p=features(g);e=np.flatnonzero(g['evidence_rank'].is_not_null());e=e[np.argsort(g['evidence_rank'].to_numpy()[e])];b=g['behavior_family'][0];family=['directed_transfer','soft_play','coordinated_isolation'].index(b);v=g.select([f'emb_{i}' for i in range(128)]).to_numpy();full=np.zeros((len(g),384),np.float32);full[:,family*128:(family+1)*128]=v;bags.append({'pid':pid,'g':g,'x':x,'p':p,'emb':full,'template':template(len(g),e),'den':len(e),'fold':g['fold'][0],'family':b})
        bags.sort(key=lambda b:b['pid']);N=len(bags);n=max(len(b['g']) for b in bags);X=np.zeros((N,n,35),np.float32);E=np.zeros((N,n,384),np.float32);P=np.zeros((N,n,2),np.float32);M=np.zeros((N,n),bool);T=np.zeros((N,6,n),np.int64);V=np.zeros((N,6),bool);D=np.array([b['den'] for b in bags],np.float32);fv=np.array([b['fold'] for b in bags])
        for i,b in enumerate(bags):k=len(b['g']);X[i,:k]=b['x'];E[i,:k]=b['emb'];P[i,:k]=b['p'];M[i,:k]=True;t=b['template'];T[i,:len(t),:k]=t;V[i,:len(t)]=True
        tr=(fv!=f)&V.any(1);va=fv==f;train=np.flatnonzero(tr);valid=np.flatnonzero(va);MM=torch.tensor(M);TT=torch.tensor(T);VV=torch.tensor(V);DD=torch.tensor(D);PP=torch.tensor(P);pred={};initial_errors=[]
        for kind in ['compact','action']:
            A=X if kind=='compact' else np.concatenate([X,E],2);mu=A[tr][M[tr]].mean(0);sd=np.maximum(.05,A[tr][M[tr]].std(0));AA=torch.tensor(np.clip((A-mu)/sd,-6,6));seedpred=[]
            for seed in CONFIG['seeds']:
                torch.manual_seed(seed+f);state=torch.load(f'artifacts/evidence_session11/conditional_family/list_independent_fold{f}_seed{seed}.pt',weights_only=False);base=Model(35,'independent');base.load_state_dict(state['state_dict']);base.eval();XX=torch.tensor(np.clip((X-state['mu'])/state['sd'],-6,6));K=torch.tensor([state['minimums'][b['family']] for b in bags])
                with torch.no_grad():prior=PP+base(XX,MM)
                adapter=nn.Linear(A.shape[-1],2);nn.init.zeros_(adapter.weight);nn.init.zeros_(adapter.bias);opt=torch.optim.AdamW(adapter.parameters(),lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay']);assert torch.count_nonzero(adapter(AA[valid[:2]]))==0
                for step in range(CONFIG['steps']):
                    delta=CONFIG['bound']*torch.tanh(adapter(AA[train]));loss=conditional_nll(prior[train]+delta,TT[train],VV[train],DD[train],MM[train],K[train]).mean()+CONFIG['residual_l2']*(delta.square()*MM[train,:,None]).sum()/MM[train].sum()/2;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(adapter.parameters(),5);opt.step()
                with torch.no_grad():logits=prior[valid]+CONFIG['bound']*torch.tanh(adapter(AA[valid]));prob=torch.softmax(torch.cat([torch.zeros_like(logits[:,:,:1]),logits],2),2).numpy();seedpred.append(prob)
                torch.save({'state_dict':adapter.state_dict(),'mu':mu,'sd':sd,'kind':kind,'fold':f,'seed':seed,'minimums':state['minimums'],'config':CONFIG},ROOT/f'adapter_{kind}_fold{f}_seed{seed}.pt')
            pred[kind]=seedpred;audit.append({'fold':f,'kind':kind,'train_pairs':len(train),'validation_pairs':len(valid),'final_training_loss':float(loss.detach())});print('frozen adapter',f,kind,'seconds',round(time.time()-start,1),flush=True)
        for k,i in enumerate(valid):
            b=bags[i];g=b['g'];z=g.select('pair_id','hand_id')
            for kind,pp in pred.items():
                inc=np.mean([conditioned(p[k,:len(g),1:],state['minimums'][b['family']]) for p in pp],axis=0);score=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc;z=z.with_columns(pl.Series(kind,score))
            parts.append(z)
    pl.concat(parts).write_parquet(ROOT/'adapter_oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
