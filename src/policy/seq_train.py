\
\
\
import os,sys,json,time,math; os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl,torch,torch.nn as nn,torch.nn.functional as F
from pathlib import Path
SEED=int(os.environ.get("SEED","0")); torch.manual_seed(SEED); np.random.seed(SEED)
TOK=Path(sys.argv[1]); OUT=Path(sys.argv[2]); OUT.mkdir(exist_ok=True,parents=True); cfg=json.loads(Path(sys.argv[3]).read_text())
dev=torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() and cfg.get('mps',True) else 'cpu')); print('device',dev,flush=True)
C=pl.col; S=os.environ.get('POKEML_SCRATCH','cache')+'/'
cols=json.load(open(TOK/'columns.json'))
TOK2=Path(cfg['tokens2']) if cfg.get('tokens2') else None
if TOK2: cols=cols+json.load(open(TOK2/'columns2.json'))
TOK3=Path(cfg['tokens3']) if cfg.get('tokens3') else None
if TOK3: cols=cols+json.load(open(TOK3/'columns3.json'))
ACT=Path(cfg['actions']) if cfg.get('actions') else None
FA=len(json.load(open(ACT/'action_columns.json'))) if ACT else 0; NA=24
ACOLS=json.load(open(ACT/'action_columns.json')) if ACT else []; AST=ACOLS.index('street') if ACT else 0
HSDIR=Path(cfg['hand_states']) if cfg.get('hand_states') else None                                                             
SCOLS=json.load(open(HSDIR/'state_columns.json')) if HSDIR else []; FS=len(SCOLS); FA2=FA+(4 if HSDIR else 0)
if HSDIR: assert ACT, 'hand_states requires actions'; SEQ=SCOLS.index('eq0'); SFOLD=SCOLS.index('fold_street')
F_=len(cols); MAXL=cfg.get('maxlen',160); MINH=cfg.get('min_hands',15)
RAW=bool(cfg.get('raw_only',False)); KEEP=(['trel']+list(cfg.get('keep_cols',[]))) if RAW else cols                                
KEEPI=np.array([cols.index(c) for c in KEEP]); TREL=KEEP.index('trel'); F_IN=len(KEEP)
                                                                
def swapname(c):
    for a,b in [('_1','_2'),('1_','2_')]:
        pass
    m={'surp_sum_1':'surp_sum_2','surp_alive_1':'surp_alive_2','surp_facing_1':'surp_facing_2','n_act_1':'n_act_2','n_facing_1':'n_facing_2','surp_max_1':'surp_max_2','net1_bb':'net2_bb','put1_l':'put2_l','f1':'f2','sd1':'sd2','stk1_l':'stk2_l','eq1':'eq2'}
    m.update({v:k for k,v in list(m.items())})
    if c.startswith('p1_'): return 'p2_'+c[3:]
    if c.startswith('p2_'): return 'p1_'+c[3:]
    if c.startswith('q1_'): return 'q2_'+c[3:]
    if c.startswith('q2_'): return 'q1_'+c[3:]
    return m.get(c,c)
PERM=np.array([cols.index(swapname(c)) for c in cols]); SEATD=cols.index('seatd')
folds=json.loads(Path('artifacts/folds.json').read_text()); tf={t:f['fold'] for f in folds for t in f['valid_tables']}
EVH=set(pl.read_csv('data/development_evidence.csv').select('pair_id','hand_id').iter_rows()) if cfg.get('aux_evidence') else set()
lab=pl.read_csv('data/development_labels.csv'); names=['none','directed_transfer','soft_play','coordinated_isolation']; labm=dict(zip(lab['pair_id'],lab['label'])); famm=dict(zip(lab['pair_id'],lab['behavior_family']))
pos=lab.filter(C('label')==1); partner={}
for a,b in zip(pos['player_1'],pos['player_2']): partner.setdefault(a,set()).add(b); partner.setdefault(b,set()).add(a)
                                              
t0=time.time(); tables=sorted(p.stem for p in TOK.glob('*.npz')); data={}
for t in tables:
    z=np.load(TOK/f'{t}.npz'); xp=TOK/f'{t}_X.npy'
    if not xp.exists(): np.save(xp,z['X'])                                     
    data[t]=dict(X=np.load(xp,mmap_mode='r'),ti=z['time_index'],pair=z['pair_id'],phase=z['phase'],n=z['n'],p1=z['player_1'],p2=z['player_2'])
    if TOK2: data[t]['X2']=np.load(TOK2/f'{t}.npy',mmap_mode='r')
    if TOK3: data[t]['X3']=np.load(TOK3/f'{t}.npy',mmap_mode='r')
    if ACT:
        za=np.load(ACT/f'{t}.npz'); xa=ACT/f'{t}_XA.npy'
        if not xa.exists() or os.path.getsize(xa)<1000: np.save(xa,za['XA'])
        data[t].update(XA=np.load(xa,mmap_mode='r'),ACT=za['ACT'],LAG=za['LAG'],hidx=za['hidx'],pi1=za['p1'],pi2=za['p2'])
    if HSDIR:
        zh=np.load(HSDIR/f'{t}.npz'); assert (zh['hidx']==data[t]['hidx']).all(); data[t].update(HS=zh['HS'],ROST=zh['ROST'])
print('loaded',len(tables),'tables',round(time.time()-t0),flush=True)
                                     
def _cat(d,sl): return np.concatenate([np.asarray(d['X'][sl])]+([np.asarray(d['X2'][sl])] if TOK2 else [])+([np.asarray(d['X3'][sl])] if TOK3 else []),1)
samp=np.concatenate([_cat(data[t],slice(None,None,20)).astype(np.float32) for t in tables[::4]]); MU=samp.mean(0); SD=samp.std(0)+1e-3
if ACT:
    sa=np.concatenate([np.asarray(data[t]['XA'][::40]).reshape(-1,FA).astype(np.float32) for t in tables[::8]]); sa=sa[np.abs(sa).sum(1)>0]; AMU=sa.mean(0); ASD=sa.std(0)+1e-3
else: AMU=ASD=None
if HSDIR:
    sh=np.concatenate([data[t]['HS'].reshape(-1,FS)[data[t]['ROST'].reshape(-1)>=0].astype(np.float32) for t in tables[::8]]); HMU=sh.mean(0); HSD=sh.std(0)+1e-3
else: HMU=HSD=None
from seq_contract import save_or_check_normalization, load_checked_state
save_or_check_normalization(OUT,dict(mu=MU,sd=SD,amu=AMU if ACT else np.zeros(1),asd=ASD if ACT else np.ones(1),hmu=HMU if HSDIR else np.zeros(1),hsd=HSD if HSDIR else np.ones(1)),cols,cfg.get('aux_evidence',False))
def getX(it):
    if it[3] is not None: return it[3]
    t,a,b,window=it[5],it[6],it[7],it[8]; d=data[t]; X=_cat(d,slice(a,b))
    if window: X=X[(d['ti'][a:b]>=window[0])&(d['ti'][a:b]<window[1])]
    return X
def seqs_of(t,phase,window=None,lazy=False):
    \
    d=data[t]; off=np.concatenate([[0],np.cumsum(d['n'])]); out=[]
    if EVH and 'hid' not in d: d['hid']=np.load(TOK/f'{t}.npz')['hand_id']
    for k in range(len(d['pair'])):
        if d['phase'][k]!=phase: continue
        a,b=off[k],off[k+1]; ti=d['ti'][a:b]
        m=((ti>=window[0])&(ti<window[1])) if window else None
        if lazy:
            n=int(m.sum()) if window else b-a
            out.append((d['pair'][k],d['p1'][k],d['p2'][k],None,None,t,a,b,window,n)); continue
        X=_cat(d,slice(a,b))
        ev=np.array([(d['pair'][k],h_) in EVH for h_ in d['hid'][a:b]]) if EVH else None
        if window:
            X=X[m]
            if ev is not None: ev=ev[m]
        out.append((d['pair'][k],d['p1'][k],d['p2'][k],X,ev,t,a,b,window,b-a,k))
    return out
def batchify(items,swap=False,with_ev=False):
    Xs=[getX(it) for it in items]; B=len(items); L=min(MAXL,max(len(x) for x in Xs)); Xb=np.zeros((B,L,F_),np.float32); M=np.zeros((B,L),bool); E=np.zeros((B,L),np.float32); EM=np.zeros((B,L),bool)
    for i,it in enumerate(items):
        X=Xs[i].astype(np.float32); ev=it[4] if len(it)>4 else None
        if len(X)>L:
            X=X[-L:]
            if ev is not None: ev=ev[-L:]
        n=len(X); Xb[i,:n]=(X-MU)/SD; M[i,:n]=True
        if with_ev and ev is not None and ev.any(): E[i,:n]=ev; EM[i,:n]=True
    if swap: Xb=Xb[:,:,PERM]; Xb[:,:,SEATD]=(6-Xb[:,:,SEATD]*SD[SEATD]-MU[SEATD])/SD[SEATD]                          
    if RAW: Xb=Xb[:,:,KEEPI]
    if ACT:
        XA=np.zeros((B,L,NA,FA2),np.float32); RO=np.full((B,L,NA),3,np.int64); LR=np.zeros((B,L,NA),np.int64); HQ=np.zeros((B,L,2*FS),np.float32) if HSDIR else None
        for i,it in enumerate(items):
            t,a,b,window=it[5],it[6],it[7],it[8]; d=data[t]; hid=d['hidx'][a:b]
            if len(it)>11 and it[11] is not None: hid=hid[it[11]] if it[11].any() else hid[:1]
            elif window: hid=hid[(d['ti'][a:b]>=window[0])&(d['ti'][a:b]<window[1])]
            if len(hid)>L: hid=hid[-L:]
            n=len(hid); k=it[10] if len(it)>10 else None
            pi1=d['pi1'][k] if k is not None else -2; pi2=d['pi2'][k] if k is not None else -2
            if k is None:
                pl_=list(d['pair']); k=pl_.index(it[0]); pi1=d['pi1'][k]; pi2=d['pi2'][k]
            if swap: pi1,pi2=pi2,pi1
            xa=np.asarray(d['XA'][hid]).astype(np.float32); ac=d['ACT'][hid]; la=d['LAG'][hid]
            xa=(xa-AMU)/ASD
            if HSDIR and n:
                ro_=d['ROST'][hid]; hs_=d['HS'][hid].astype(np.float32); P=np.zeros((n,FS),np.float32); Q=np.zeros((n,FS),np.float32)
                r_,c_=np.where(ro_==pi1); P[r_]=hs_[r_,c_]; r_,c_=np.where(ro_==pi2); Q[r_]=hs_[r_,c_]
                st=np.clip(xa[:,:,AST]*ASD[AST]+AMU[AST],0,3).round().astype(int); rr=np.arange(n)[:,None]
                ex_=np.stack([P[:,SEQ:SEQ+4][rr,st],(P[:,SFOLD][:,None]>st).astype(np.float32),Q[:,SEQ:SEQ+4][rr,st],(Q[:,SFOLD][:,None]>st).astype(np.float32)],-1); ex_[ac<0]=0
                xa=np.concatenate([xa,ex_],-1); HQ[i,:n]=np.concatenate([(P-HMU)/HSD,(Q-HMU)/HSD],1)
            XA[i,:n]=xa; ro=np.zeros_like(ac,dtype=np.int64); ro[ac==pi1]=1; ro[ac==pi2]=2; ro[ac<0]=3; RO[i,:n]=ro
            lr=np.ones_like(la,dtype=np.int64); lr[la<0]=0; lr[la==pi1]=2; lr[la==pi2]=3; LR[i,:n]=lr
        extra=(torch.from_numpy(XA),torch.from_numpy(RO),torch.from_numpy(LR))+((torch.from_numpy(HQ),) if HSDIR else ())
    else: extra=()
    if with_ev: return (torch.from_numpy(Xb),torch.from_numpy(M),torch.from_numpy(E),torch.from_numpy(EM))+extra
    return (torch.from_numpy(Xb),torch.from_numpy(M))+extra
class ActEnc(nn.Module):
    def __init__(s,FA,da=cfg.get('d_act',64),drop=0.1):
        super().__init__(); s.role=nn.Embedding(4,da); s.lrole=nn.Embedding(4,da); s.pos=nn.Embedding(NA,da); s.inp=nn.Linear(FA,da)
        enc=nn.TransformerEncoderLayer(da,4,dim_feedforward=2*da,dropout=drop,batch_first=True,norm_first=True); s.enc=nn.TransformerEncoder(enc,cfg.get('act_layers',1)); s.att=nn.Linear(da,1)
    def forward(s,XA,RO,LR):
        B,L,A_,F=XA.shape; x=XA.view(B*L,A_,F); ro=RO.view(B*L,A_); lr=LR.view(B*L,A_); pad=ro==3
        h=s.inp(x)+s.role(ro)+s.lrole(lr)+s.pos(torch.arange(A_,device=x.device))[None]
        allpad=pad.all(1); pad2=pad.clone(); pad2[allpad,0]=False
        h=s.enc(h,src_key_padding_mask=pad2); a=s.att(h).squeeze(-1).masked_fill(pad2,-1e4); w=torch.softmax(a,1).unsqueeze(-1)
        out=(w*h).sum(1); out[allpad]=0; return out.view(B,L,-1)
class Net(nn.Module):
    def __init__(s,F_,d=cfg.get('d',96),L=cfg.get('layers',2),H=cfg.get('heads',4),drop=cfg.get('dropout',0.1)):
        super().__init__(); s.act=ActEnc(FA2) if ACT else None; da=cfg.get('d_act',64) if ACT else 0
        s.inp=nn.Sequential(nn.Linear(F_IN+da+2*FS,d),nn.GELU(),nn.LayerNorm(d),nn.Dropout(drop))
        s.pos=nn.Linear(1,d)
        enc=nn.TransformerEncoderLayer(d,H,dim_feedforward=2*d,dropout=drop,batch_first=True,norm_first=True); s.enc=nn.TransformerEncoder(enc,L)
        s.att=nn.Linear(d,1); s.head=nn.Sequential(nn.Linear(3*d+1,d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,4)); s.tok=nn.Linear(d,1)
    def forward(s,X,M,XA=None,RO=None,LR=None,HQ=None):
        tr=X[:,:,TREL][:,:,None]
        if s.act is not None: X=torch.cat([X,s.act(XA.to(X.device),RO.to(X.device),LR.to(X.device))]+([HQ.to(X.device)] if HQ is not None else []),-1)
        h=s.inp(X)+s.pos(tr)                    
        h=s.enc(h,src_key_padding_mask=~M)
        a=s.att(h).squeeze(-1).masked_fill(~M,-1e4); w=torch.softmax(a,1).unsqueeze(-1)
        pooled=(w*h).sum(1); mean=(h*M.unsqueeze(-1)).sum(1)/M.sum(1,keepdim=True).clamp(min=1); mx=h.masked_fill(~M.unsqueeze(-1),-1e4).max(1).values
        n=torch.log1p(M.sum(1,keepdim=True).float())
        out=s.head(torch.cat([pooled,mean,mx,n],-1))                                              
        s.last_tok=s.tok(h).squeeze(-1)
        return out
def risk_from(logits): return torch.sigmoid(logits[:,0])
def build_train(f):
    items=[];w=[];y=[];fam=[]
    rng=np.random.default_rng(100+f)
    pseudo=set(pl.read_csv(cfg['pseudo_src']).filter((1-C('none'))>cfg.get('pseudo_thr',0.9))['pair_id']) if cfg.get('pseudo_src') else set()
    pfam=None
    if pseudo:
        pp=pl.read_csv(cfg['pseudo_src']).filter((1-C('none'))>cfg.get('pseudo_thr',0.9)); pfam=dict(zip(pp['pair_id'],pp.select(names[1:]).to_numpy().argmax(1)+1))
    hn_budget=cfg.get('hardneg',6000); bg_budget=cfg.get('background',20000)
    hn_pool=[];bg_pool=[]
    for t in tables:
        if t not in tf or tf[t]==f: continue
        for window in [None,(0,2000),(1000,3000)]+[tuple(x) for x in cfg.get('extra_windows',[])]:
            for it in seqs_of(t,'development',window,lazy=True):
                pid,p1,p2=it[:3]; nlen=it[9]
                if nlen<MINH: continue
                l=labm.get(pid,-1); ww=1.0 if window is None else cfg.get('w_win',0.5)
                if l==1 or l==0 or pid in pseudo:
                    ev=None
                    if EVH: ev=np.array([(it[0],h_) in EVH for h_ in data[t]['hid'][it[6]:it[7]]]); ti=data[t]['ti'][it[6]:it[7]]; ev=ev[(ti>=window[0])&(ti<window[1])] if window else ev
                    full_it=(pid,p1,p2,getX(it),ev,it[5],it[6],it[7],it[8],it[9])
                if l==1:
                    items.append(full_it); y.append(1); fam.append(names.index(famm[pid])); w.append(ww)
                elif l==0:
                    items.append(full_it); y.append(0); fam.append(0); w.append(ww)
                elif pid in pseudo:
                    items.append(full_it); y.append(1); fam.append(pfam[pid]); w.append(ww*cfg.get('w_pseudo',1.0))
                elif window is None:
                    if (p1 in partner)!=(p2 in partner): hn_pool.append(it)
                    elif (p1 not in partner) and (p2 not in partner): bg_pool.append(it)
    idx=rng.choice(len(hn_pool),min(hn_budget,len(hn_pool)),replace=False)
    for i in idx: items.append(hn_pool[i]); y.append(0); fam.append(0); w.append(cfg.get('w_hardneg',0.5))
    items=[it if it[3] is not None else (it[0],it[1],it[2],getX(it),None,it[5],it[6],it[7],it[8],it[9]) for it in items]
    nfix=len(items)
    if cfg.get('bg_resample'):
        print('fold',f,'fixed seqs',nfix,'pos',sum(y),'bg pool',len(bg_pool),'(resampled per epoch)',flush=True)
        return items,np.array(y,np.float32),np.array(fam),np.array(w,np.float32),bg_pool
    idx=rng.choice(len(bg_pool),min(bg_budget,len(bg_pool)),replace=False)
    for i in idx: items.append(bg_pool[i]); y.append(0); fam.append(0); w.append(cfg.get('w_bg',0.1))
                                                                                
    items=[it if it[3] is not None else (it[0],it[1],it[2],getX(it),None,it[5],it[6],it[7],it[8],it[9]) for it in items]
    print('fold',f,'train seqs',len(items),'pos',sum(y),'hardneg',min(hn_budget,len(hn_pool)),'bg',min(bg_budget,len(bg_pool)),flush=True)
    return items,np.array(y,np.float32),np.array(fam),np.array(w,np.float32),None
def score(model,items,bs=256):
    model.eval(); out=np.zeros((len(items),4),np.float32)
    with torch.no_grad():
        for i in range(0,len(items),bs):
            b=items[i:i+bs]; acc=0
            for sw in [False,True]:
                bt=batchify(b,swap=sw); X,M=bt[0],bt[1]; lg=model(X.to(dev),M.to(dev),*bt[2:]); r=torch.sigmoid(lg[:,0]); fm=torch.softmax(lg[:,1:],-1)
                acc=acc+torch.cat([r.unsqueeze(1),fm],1).cpu().numpy()/2
            out[i:i+len(b)]=acc
    return out
EPOCHS=cfg.get('epochs',8); BS=cfg.get('batch',64); LR=cfg.get('lr',3e-4); bg_budget_cfg=cfg.get('background',20000); AUXW=cfg.get('aux_evidence',0.0)
dev_scores={w:[] for w in ['full','first_2000','last_2000']}; models=[]
WIN={'full':None,'first_2000':(0,2000),'last_2000':(1000,3000)}
def score_tables(model,tabs,phase,win):
    frames=[]
    for t in tabs:
        its=[it for it in seqs_of(t,phase,win) if len(it[3])>=1]
        if not its: continue
        sc=score(model,its); frames.append(pl.DataFrame({'pair_id':[it[0] for it in its],'player_1':[it[1] for it in its],'player_2':[it[2] for it in its],'n_hands':[len(it[3]) for it in its],'risk':sc[:,0],**{n:sc[:,k+1] for k,n in enumerate(names[1:])}}))
    return pl.concat(frames) if frames else None
for f in range(4):
    done=all((OUT/f'dev_{w}_fold{f}.csv').exists() for w in WIN) and (OUT/f'eval_fold{f}.csv').exists() and (OUT/f'seq_fold{f}.pt').exists()
    model=Net(F_).to(dev)
    if done:
        load_checked_state(model,torch.load(OUT/f'seq_fold{f}.pt',map_location=dev,weights_only=True),cfg.get('aux_evidence',False)); models.append(model)
        for w in WIN: dev_scores[w].append(pl.read_csv(OUT/f'dev_{w}_fold{f}.csv'))
        print('fold',f,'resumed from disk',flush=True); continue
    items0,y0,fam0,w0,bgpool=build_train(f); items,y,fam,w=items0,y0,fam0,w0; opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01)
    nb_per=math.ceil((len(items0)+(min(bg_budget_cfg,len(bgpool)) if bgpool is not None else 0))/BS); steps=EPOCHS*nb_per; sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,total_steps=steps,pct_start=0.1)
    order=np.arange(len(items)); t1=time.time()
    for ep in range(EPOCHS):
        if bgpool is not None:
            bi=np.random.choice(len(bgpool),min(bg_budget_cfg,len(bgpool)),replace=False); items=items0+[bgpool[i] for i in bi]; y=np.concatenate([y0,np.zeros(len(bi),np.float32)]); fam=np.concatenate([fam0,np.zeros(len(bi),int)]); w=np.concatenate([w0,np.full(len(bi),cfg.get('w_bg',0.1),np.float32)]); order=np.arange(len(items))
        model.train(); np.random.shuffle(order); tot=0;nb=0
        for i in range(0,len(order),BS):
            idx=order[i:i+BS]; b=[items[k] for k in idx]; bt=batchify(b,swap=bool(np.random.rand()<0.5),with_ev=True); X,M,E,EM=[x.to(dev) for x in bt[:4]]; ex=bt[4:]
            yy=torch.from_numpy(y[idx]).to(dev); ww=torch.from_numpy(w[idx]).to(dev); ff=torch.from_numpy(fam[idx]).to(dev)
            lg=model(X,M,*ex); loss=(F.binary_cross_entropy_with_logits(lg[:,0],yy,reduction='none')*ww).sum()/ww.sum()
            pm=yy>0
            if pm.any(): loss=loss+0.3*F.cross_entropy(lg[pm][:,1:],ff[pm]-1)
            RW=cfg.get('rank_w',0.0)
            if RW and pm.any() and (~pm).any():
                dif=lg[pm][:,0][:,None]-lg[~pm][:,0][None,:]; wn=ww[~pm][None,:]
                loss=loss+RW*(F.softplus(-dif)*wn).sum()/(wn.sum()*pm.sum()).clamp(min=1)
            if AUXW and EM.any():
                tl=model.last_tok; tw=torch.where(E>0,1.0,0.1)*EM.float()
                loss=loss+AUXW*(F.binary_cross_entropy_with_logits(tl,E,reduction='none')*tw).sum()/tw.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step(); sched.step(); tot+=loss.item(); nb+=1
        print(f' fold {f} epoch {ep} loss {tot/nb:.4f} {round(time.time()-t1)}s',flush=True)
    torch.save(model.state_dict(),OUT/f'seq_fold{f}.pt'); models.append(model); del items,items0,bgpool
    ftabs=[t for t in tables if tf.get(t)==f]
    for w,win in WIN.items():
        fr=score_tables(model,ftabs,'development',win)
        if fr is not None: fr.write_csv(OUT/f'dev_{w}_fold{f}.csv'); dev_scores[w].append(fr)
    fe=score_tables(model,tables,'evaluation',None); fe.write_csv(OUT/f'eval_fold{f}.csv')
    print('fold',f,'scored',round(time.time()-t1),flush=True)
for w in dev_scores:
    if not dev_scores[w]: continue
    d=pl.concat(dev_scores[w],how='vertical_relaxed'); d=d.with_columns((1-C('risk')).alias('none'),*[(C('risk')*C(n)).alias(n) for n in names[1:]]); d.write_csv(OUT/f'dev_{w}.csv')
                                          
fe=[pl.read_csv(OUT/f'eval_fold{f}.csv').sort('pair_id') for f in range(4)]
e=fe[0].select('pair_id','player_1','player_2','n_hands').with_columns(*[pl.Series(c,np.mean([x[c].to_numpy() for x in fe],axis=0)) for c in ['risk']+names[1:]])
e=e.with_columns((1-C('risk')).alias('none'),*[(C('risk')*C(n)).alias(n) for n in names[1:]]); e.write_csv(OUT/'eval_all.csv'); print('done',round(time.time()-t0),flush=True)
