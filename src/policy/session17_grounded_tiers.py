\
\
\
\
\
\
import os,json,time,itertools
os.environ.setdefault('POLARS_MAX_THREADS','3');os.environ.setdefault('LOKY_MAX_CPU_COUNT','3')
from pathlib import Path
import numpy as np,polars as pl,joblib
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from session8_data import hand_data
from session7_likelihood import multilevel_inclusion
ROOT=Path('artifacts/evidence_session17_grounded');C=pl.col
CONFIG={'hypothesis':'soft-play evidence has chronological fold/call/check priority tiers','support':'fold_partner >0, call_partner >0, check_hu >0; training assignment support only; no inference hard gate','type':'unique outer-training feasible lists supervise 200-tree depth3 multiclass subtype model; ambiguous lists maximize subtype likelihood over feasible assignments','events':'original Cat and HGB schedules with three censored heads; no outer stopping','inference':'compare third-tier selection with collapsed call/check control; no R30 neural correction in new joint three-tier arm','limitation':'physical support was explored on all public lists before these outer-isolated fits; validation is reused and hypothesis-selected'}
def assignments(g):
 pos=g.filter(C('evidence')==1).sort('evidence_rank');ix=pos['local_row'].to_numpy();t=pos['time'].to_numpy();support=np.column_stack([pos['fold_partner']>0,pos['call_partner']>0,pos['check_hu']>0]);valid=[];n=len(ix)
 for k in range(n+1):
  for j in range(k,n+1):
   c=np.r_[np.zeros(k,int),np.ones(j-k,int),np.full(n-j,2)]
   if all(np.all(np.diff(t[c==r])>0) for r in range(3)) and support[np.arange(n),c].all():valid.append(c)
 return ix,valid
def distribution(a):
 a=np.asarray(a,float);a=a/np.maximum(1,a.sum(1))[:,None];return np.column_stack([1-a.sum(1),a])
def conditional(p,k):
 p=np.asarray(p,float);p=p/p.sum(1)[:,None];ordinary=multilevel_inclusion(p);s=1-p[:,0]
 def scan(v):
  z=np.zeros((len(v)+1,k));z[0,0]=1
  for i,vv in enumerate(v):z[i+1]=np.r_[z[i,0]*(1-vv),z[i,1:]*(1-vv)+z[i,:-1]*vv]
  return z
 pre=scan(s);suf=scan(s[::-1])[::-1];den=1-pre[-1].sum()
 if den<1e-12:return ordinary
 sub=np.array([s[i]*np.convolve(pre[i],suf[i+1])[:k-1].sum() for i in range(len(s))]);v=(ordinary-sub)/den;assert v.min()>-1e-8 and v.max()<1+1e-8;return np.clip(v,0,1)
def check():
 p=np.random.default_rng(1717).dirichlet([4,1,1,1],size=6);m=np.zeros(6);u=np.zeros(6);den=0.
 for cc in itertools.product(range(4),repeat=6):
  w=np.prod(p[np.arange(6),cc]);e=sum(([i for i,c in enumerate(cc) if c==k] for k in range(1,4)),[]);u[e[:5]]+=w
  if len(e)>=3:m[e[:5]]+=w;den+=w
 a=float(abs(multilevel_inclusion(p)-u).max());b=float(abs(conditional(p,3)-m/den).max());assert max(a,b)<1e-12;return {'paths':4**6,'ordinary_error':a,'conditional_error':b}
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps(CONFIG,indent=2));(ROOT/'math_checks.json').write_text(json.dumps(check(),indent=2));full=hand_data();d=full.filter(C('behavior_family')=='soft_play').drop('row').with_row_index('local_row');cols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];typecols=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['type'];X=d.select(cols).to_numpy();XT=d.select(typecols).to_numpy();fv=d['fold'].to_numpy();groups=[g for _,g in d.group_by('pair_id')];predcat=np.zeros((len(d),3));predhist=np.zeros_like(predcat);audit=[];mins={};start=time.time();(ROOT/'columns.json').write_text(json.dumps({'event':cols,'type':typecols},indent=2))
 with threadpool_limits(limits=2):
  for f in range(4):
   tr=fv!=f;va=~tr;sub=np.full(len(d),-1);pending=[];truthcounts=[]
   for g in groups:
    if g['fold'][0]==f:continue
    ix,valid=assignments(g);assert valid;truthcounts.append(len(ix));pending.append((g,ix,valid))
    if len(valid)==1:sub[ix]=valid[0]
   known=(sub>=0)&tr;assert set(sub[known])=={0,1,2};typ=CatBoostClassifier(iterations=200,depth=3,learning_rate=.04,l2_leaf_reg=10,loss_function='MultiClass',thread_count=2,random_seed=6210+f,verbose=False,allow_writing_files=False);typ.fit(XT[known],sub[known]);typ.save_model(str(ROOT/f'type_fold{f}.cbm'));tp=typ.predict_proba(XT,thread_count=2);y=np.zeros((len(d),3));eligible=np.tile(tr[:,None],(1,3));records=[]
   for g,ix,valid in pending:
    ll=[np.log(tp[ix,c].clip(1e-8,1)).sum() for c in valid];chosen=valid[int(np.argmax(ll))];y[ix,chosen]=1;rows=g['local_row'].to_numpy();records.append({'pair_id':g['pair_id'][0],'feasible_assignments':len(valid),'chosen':chosen.tolist()})
    if len(ix)==5:
     final=int(chosen[-1]);eligible[rows,final+1:]=False;after=rows[d['time'].to_numpy()[rows]>d['time'].to_numpy()[ix[-1]]];eligible[after,final]=False
   mins[f]=min(truthcounts);assert not np.any(eligible[va]);assert y[va].sum()==0
   for k in range(3):
    e=eligible[:,k];m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,thread_count=2,random_seed=6310+11*f+k+1,verbose=False,allow_writing_files=False);m.fit(X[e],y[e,k]);m.save_model(str(ROOT/f'event{k}_fold{f}.cbm'));predcat[va,k]=m.predict_proba(X[va],thread_count=2)[:,1];h=HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=10,max_bins=127,early_stopping=False,random_state=4710+f);h.fit(X[e],y[e,k]);joblib.dump(h,ROOT/f'event{k}_fold{f}.joblib',compress=3);predhist[va,k]=h.predict_proba(X[va])[:,1];audit.append({'fold':f,'tier':k,'eligible_rows':int(e.sum()),'positive_rows':int(y[e,k].sum()),'unique_subtype_rows':int(known.sum()),'validation_overlap':int((e&va).sum())})
   (ROOT/f'assignments_fold{f}.json').write_text(json.dumps(records,indent=2));print('grounded tiers',f,round(time.time()-start,1),flush=True)
 d.select('pair_id','hand_id','time','fold').with_columns(*[pl.Series(f'{name}_{k}',p[:,k]) for name,p in [('cat',predcat),('hist',predhist)] for k in range(3)]).write_parquet(ROOT/'heads_oof.parquet');(ROOT/'audit.json').write_text(json.dumps({'heads':audit,'minimums':mins},indent=2));assemble()
def assemble():
 d=hand_data();new=pl.read_parquet(ROOT/'heads_oof.parquet');r30=pl.read_parquet('artifacts/evidence_session11/conditional_family_deployed_oof.parquet').select('pair_id','hand_id','conditional_family');raw=pl.concat([pl.read_parquet(f'artifacts/evidence_session10/nested6/nested_outer{f}.parquet').filter(C('fold')==f).select('pair_id','hand_id','base','cat_inclusion') for f in range(4)]);q=d.join(r30,on=['pair_id','hand_id'],validate='1:1').join(raw,on=['pair_id','hand_id'],validate='1:1').join(new.drop('time','fold'),on=['pair_id','hand_id'],how='left',validate='1:1');mins=json.load(open(ROOT/'audit.json'))['minimums'];parts=[]
 for (pid,),g in q.group_by('pair_id'):
  g=g.sort('time','hand_id');score=g['conditional_family'].to_numpy();values={n:score.copy() for n in ['cat_only','collapsed_control','joint','conditional_joint']}
  if g['behavior_family'][0]=='soft_play':
   ca=distribution(g.select('cat_0','cat_1','cat_2').to_numpy());ha=distribution(g.select('hist_0','hist_1','hist_2').to_numpy());jp=.5*(ca+ha);ci=multilevel_inclusion(ca);collapsed=np.column_stack([ca[:,0],ca[:,1],ca[:,2:].sum(1)]);values['cat_only']=score+.25*(ci-g['cat_inclusion'].to_numpy());values['collapsed_control']=score+.25*(multilevel_inclusion(collapsed)-g['cat_inclusion'].to_numpy());values['joint']=.25*g['base'].to_numpy()+.25*ci+.5*multilevel_inclusion(jp);values['conditional_joint']=.25*g['base'].to_numpy()+.25*ci+.5*conditional(jp,mins[str(g['fold'][0])])
  parts.append(g.select('pair_id','hand_id').with_columns(*[pl.Series(n,v) for n,v in values.items()]))
 pl.concat(parts).write_parquet(ROOT/'grounded_oof.parquet')
if __name__=='__main__':main()
