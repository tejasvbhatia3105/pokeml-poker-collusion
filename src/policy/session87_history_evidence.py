\
\
\
\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl,torch
from catboost import CatBoostClassifier,CatBoostRegressor
import session86_action_history as enc
import session83_joint_policy as old
from session37_bet_fold import paired_features
from session12_event_replacement import assemble
ROOT=Path('artifacts/evidence_session87_history_evidence');C=pl.col
def prepare():
 ROOT.mkdir(exist_ok=True);states=old.state();pc=json.load(open('artifacts/policy/feature_columns.json'))
 for fam in ['directed_transfer','soft_play']:
  v=states[fam];d,a=v['d'],v['a'];_,align=paired_features(d,a);pieces={'own':[],'bet':[]};lengths={'own':[],'bet':[]};raw={'own':[],'bet':[]}
  for (table,),q in d.group_by('table_id'):
   z=align.join(q.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi');src=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').filter(C('phase')=='development').sort('hand_id','action_no').with_row_index('source_row');keys=src.select('hand_id',C('action_no').cast(pl.Int64),'source_row')
   for role,col in [('own','fold_action_no'),('bet','bet_action_no')]:
    qr=z.select('action_row','hand_id',C(col).alias('action_no')).join(keys,on=['hand_id','action_no'],validate='m:1',maintain_order='left');ix=qr['source_row'].to_numpy();h,l=enc.history(src,ix);pieces[role].append((qr['action_row'].to_numpy(),h));lengths[role].append(l);raw[role].append(src[ix].select(pc).to_numpy())
  out={};cached=np.load(old.ROOT/f'{fam}_raw.npz')
  for role in ['own','bet']:
   rows=np.concatenate([r for r,h in pieces[role]]);order=np.argsort(rows);np.testing.assert_array_equal(rows[order],np.arange(len(a)));out[role+'_history']=np.concatenate([h for r,h in pieces[role]])[order];out[role+'_length']=np.concatenate(lengths[role])[order];x=np.concatenate(raw[role])[order];np.testing.assert_array_equal(x,cached[role]);out[role]=x
  out['size']=cached['size'];np.savez_compressed(ROOT/f'{fam}_actions.npz',**out);print('evidence history prepared',fam,len(a),flush=True)
def representation(kind,f,raw,h,length,native):
 pc=json.load(open('artifacts/policy/feature_columns.json'));ck=torch.load(enc.ROOT/f'{kind}_fold{f}.pt',map_location='cpu',weights_only=False);m=enc.Policy(kind,len(pc));m.load_state_dict(ck['state']);m.eval();x=((enc.current_transform(raw,pc)-ck['mean'])/ck['scale']).clip(-20,20).astype(np.float32);prior=enc.reference(raw,native,f);call=raw[:,pc.index('call_bb')]>0;pieces=[]
 with torch.no_grad():
  for st in range(0,len(x),4096):
   sl=slice(st,st+4096);xx=torch.from_numpy(x[sl]);ll=torch.from_numpy(length[sl])
   if kind=='history':
    z,_=m.gru(torch.from_numpy(h[sl].astype(np.float32)));z=z[torch.arange(len(xx)),(ll-1).clamp_min(0)]*(ll>0)[:,None];xx=torch.cat([xx,z],1)
   hidden=m.net[:4](xx);delta=3*torch.tanh(m.net[4](hidden)/3);pp=torch.softmax(enc.legal_logits(torch.from_numpy(prior[sl])+delta,torch.from_numpy(call[sl])),1);pieces.append(torch.cat([pp,hidden],1).numpy())
 return np.concatenate(pieces)
def features(fam,v,f,kind):
 z=np.load(ROOT/f'{fam}_actions.npz');native=v['fv'][v['a']['row'].to_numpy()];a=representation(kind,f,z['own'],z['own_history'],z['own_length'],native);b=representation(kind,f,z['bet'],z['bet_history'],z['bet_length'],native);p,q=a[:,:4],b[:,:4];u=-np.log(p[:,0].clip(1e-7));w=-np.log(q[:,3].clip(1e-7));m=CatBoostRegressor();m.load_model(f'artifacts/policy/size_fold{f}.cbm');res=z['size']-m.predict(z['bet'],thread_count=2);return np.column_stack([p,q,u,w,u+w,np.minimum(u,w),u-w,res,abs(res),a[:,4:],b[:,4:]]).astype(np.float32)
def train():
 torch.set_num_threads(3);states=old.state();base=pl.read_parquet('artifacts/evidence_session59_pressure_equity/event_oof.parquet');parts={k:[] for k in ['summary','history']};audit=[];start=time.time();(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'arms':['summary','history'],'event_schedule':'Cat400 D5 lr.035 L2=8 seed6311+11fold; unchanged R32 targets/features plus143 policy/representation fields','feature_dimensions':143,'fixed_recipe':'preserve all other heads; equal mixture with R33 grounded-list tree','no_family_or_blend_selection':True},indent=2))
 for fam in ['directed_transfer','soft_play']:
  v=states[fam];d,a=v['d'],v['a'];g=a['row'].to_numpy();actor=a['actor'].to_numpy();fv=v['fv'];dw=d.select('pair_id').join(pl.read_parquet('artifacts/evidence_session25_persistent_actor/actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if fam=='directed_transfer' else np.ones((len(d),2));pred={k:np.zeros((len(d),2)) for k in parts}
  for f in range(4):
   if fam=='directed_transfer':y,tr,va=old.labels(d,a,f)
   else:yy,e,_=old.target(d,f);y=yy[g];tr=e[g];va=fv[g]==f
   assert not(tr&va).any()
   for kind in parts:
    root=ROOT/kind;root.mkdir(exist_ok=True);ex=features(fam,v,f,kind);np.savez_compressed(root/f'{fam}_extra_fold{f}.npz',x=ex);x=np.column_stack([v['x'],ex]);m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6311+11*f,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[tr],y[tr]);m.save_model(str(root/f'{fam}_primary_fold{f}.cbm'));pred[kind][g[va],actor[va]]=m.predict_proba(x[va],thread_count=2)[:,1];audit.append({'family':fam,'fold':f,'kind':kind,'training_actions':int(tr.sum()),'positive_actions':int(y[tr].sum()),'validation_overlap':0});print('history evidence',fam,f,kind,round(time.time()-start,1),flush=True)
  for kind,p in pred.items():parts[kind].append(d.select('pair_id','hand_id').with_columns(pl.Series('replacement',(p*dw).sum(1))))
 for kind in parts:
  root=ROOT/kind;base.join(pl.concat(parts[kind]),on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('replacement','bg_primary').alias('bg_primary')).drop('replacement').write_parquet(root/'event_oof.parquet');assemble(root)
 (ROOT/'audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':
 import sys
 if sys.argv[1]=='prepare':prepare()
 else:train()
