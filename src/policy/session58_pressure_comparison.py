\
\
\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','3')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from cards import CARD,rank
from session57_isolation_pressure import data,BASE,targets,noisy_or,assemble,C
ROOT=Path('artifacts/evidence_session58_pressure_comparison')
COLS=['partner_category','partner_kicker','pair_order','pair_rank_gap']
def compute(d,a,players=None):
 people=(pl.read_csv('data/development_labels.csv') if players is None else players).select('pair_id','player_1','player_2');z=a.with_row_index('action_row');out=np.full((len(a),4),-2,np.float32);own_error=0;post=0
 for (table,),q in d.group_by('table_id'):
  local=z.join(q.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi');need=local.select('hand_id').unique();s=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').join(need,on='hand_id',how='semi');h=pl.read_parquet(f'artifacts/compact/hands/table_id={table}/*.parquet').join(need,on='hand_id',how='semi');raw=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').select('hand_id','street_no','action_no','player_id');local=local.join(raw,on=['hand_id','street_no','action_no'],validate='m:1').join(people,on='pair_id',validate='m:1');holes={(hid,p):[CARD[c1],CARD[c2]] for hid,p,c1,c2 in s.select('hand_id','player_id','hole_card_1','hole_card_2').iter_rows()};boards={hid:[CARD[c] for c in b.split()] for hid,b in h.select('hand_id','board_cards').iter_rows()};cache={}
  def strength(hid,p,nb):
   key=(hid,p,nb)
   if key not in cache:cache[key]=rank(holes[hid,p]+boards[hid][:nb])
   return cache[key]
  for r in local.to_dicts():
   st=int(r['street_no']);own=r['player_id'];assert own in [r['player_1'],r['player_2']]
   if st==0:continue
   partner=r['player_2'] if own==r['player_1'] else r['player_1'];ar=strength(r['hand_id'],own,st+2);br=strength(r['hand_id'],partner,st+2);own_error=max(own_error,abs(ar/2**24-float(r['made_category'])-float(r['made_kicker'])));out[r['action_row']]=[br>>24,(br&0xFFFFFF)/2**24,int(ar>br)-int(ar<br),(ar-br)/2**24];post+=1
 assert own_error==0;return out,{'actions':len(a),'postflop_actions':post,'own_encoded_rank_error':own_error}
def main():
 ROOT.mkdir(exist_ok=True);full,d,a,ac=data();ex,rawaudit=compute(d,a);np.savez_compressed(ROOT/'features.npz',x=ex);(ROOT/'feature_audit.json').write_text(json.dumps(rawaudit,indent=2));hc=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'))['event'];g=a['row'].to_numpy();cnt=np.bincount(g,minlength=len(d));x=np.column_stack([a.select(ac).to_numpy(),d.select(hc).to_numpy()[g],ex]);fv=d['fold'].to_numpy();mask=full['behavior_family'].to_numpy()=='coordinated_isolation';pp=np.zeros((len(d),2));audit=[];start=time.time();(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'extra_columns':COLS,'action_columns':ac,'hand_columns':hc,'schedule':'exact57 action_hand 3 EM Cat400 D5 lr.035 L2=8, seeds6311+11fold+head','support_and_targets':'unchanged57; original two censored isolation heads'},indent=2))
 for f in range(4):
  p1,p2,e1,e2,_=targets(full,f,'coordinated_isolation');ys=np.column_stack([p1,p2])[mask];es=np.column_stack([e1,e2])[mask];va=fv[g]==f
  for k in range(2):
   y=ys[:,k];tr=es[g,k];assert not(tr&va).any();gt=g[tr];yt=y[gt];resp=yt/cnt[gt];trace=[];bags=np.flatnonzero(es[:,k]&(cnt>0))
   for step in range(3):
    m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,loss_function='CrossEntropy',random_seed=6311+11*f+k,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[tr],resp);m.save_model(str(ROOT/f'event{k+1}_fold{f}_em{step}.cbm'));p=m.predict_proba(x[tr],thread_count=2)[:,1];hp=noisy_or(p,gt,len(d));resp=np.where(yt,p/np.maximum(hp[gt],1e-8),0).clip(0,1);v=hp[bags].clip(1e-8,1-1e-8);trace.append(float(-(y[bags]*np.log(v)+(1-y[bags])*np.log1p(-v)).mean()))
   pred=noisy_or(m.predict_proba(x[va],thread_count=2)[:,1],g[va],len(d));pp[fv==f,k]=pred[fv==f];audit.append({'fold':f,'head':k+1,'training_actions':int(tr.sum()),'positive_hands':int(y[bags].sum()),'validation_overlap':0,'training_bag_nll':trace});print('pressure comparison',f,k+1,round(time.time()-start,1),flush=True)
 base=pl.read_parquet(BASE/'event_oof.parquet');q=d.select('pair_id','hand_id').with_columns(pl.Series('new_primary',pp[:,0]),pl.Series('new_secondary',pp[:,1]));base.join(q,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('new_primary','bg_primary').alias('bg_primary'),pl.coalesce('new_secondary','bg_secondary').alias('bg_secondary')).drop('new_primary','new_secondary').write_parquet(ROOT/'event_oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2));assemble(ROOT)
if __name__=='__main__':main()
