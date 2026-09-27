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
from session50_matchup import load,ROOT as PAIR_ROOT,hand_data,targets,labels,target,OLD,C,assemble
ROOT=Path('artifacts/evidence_session53_outsider_rank')
COLS=['outsider_count','own_vs_best_order','own_vs_best_gap','partner_vs_best_order','partner_vs_best_gap','outsiders_beating_own','outsiders_tying_own','outsiders_beating_partner','outsiders_tying_partner','team_at_least_best','both_partners_beaten']
def compute(d,a,players=None):
 people=(pl.read_csv('data/development_labels.csv') if players is None else players).select('pair_id','player_1','player_2');z=a.join(people,on='pair_id',validate='m:1',maintain_order='left');out=np.full((len(a),len(COLS)),-2,np.float32);records=[]
 for (table,),q in d.group_by('table_id'):
  local=z.join(q.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi');need=local.select('hand_id').unique();s=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').join(need,on='hand_id',how='semi');h=pl.read_parquet(f'artifacts/compact/hands/table_id={table}/*.parquet').join(need,on='hand_id',how='semi');raw=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').join(need,on='hand_id',how='semi');holes={(hid,p):(CARD[c1],CARD[c2]) for hid,p,c1,c2 in s.select('hand_id','player_id','hole_card_1','hole_card_2').iter_rows()};boards={hid:[CARD[c] for c in b.split()] for hid,b in h.select('hand_id','board_cards').iter_rows()};rosters={hid:g['player_id'].to_list() for (hid,),g in s.group_by('hand_id')};folds={(hid,p):int(n) for hid,p,n in raw.filter(C('action_class')==0).select('hand_id','player_id','action_no').iter_rows()};rank_cache={}
  def strength(hid,p,nb):
   key=(hid,p,nb)
   if key not in rank_cache:rank_cache[key]=rank(list(holes[hid,p])+boards[hid][:nb])
   return rank_cache[key]
  for r in local.to_dicts():
   hid=r['hand_id'];own=r['player_1'] if r['actor']==0 else r['player_2'];partner=r['player_2'] if r['actor']==0 else r['player_1'];active=[p for p in rosters[hid] if folds.get((hid,p),999)>=r['action_no']];assert own in active and partner in active;outsiders=[p for p in active if p not in [own,partner]];v=out[r['action_row']];v[0]=len(outsiders);street=int(r['street_no']);assert len(active)==int(r['players_active'])
   if street>0 and outsiders:
    nb=street+2;ar=strength(hid,own,nb);br=strength(hid,partner,nb);other=np.array([strength(hid,p,nb) for p in outsiders],dtype=np.int64);best=int(other.max());v[1:]=[int(ar>best)-int(ar<best),(ar-best)/2**24,int(br>best)-int(br<best),(br-best)/2**24,int((other>ar).sum()),int((other==ar).sum()),int((other>br).sum()),int((other==br).sum()),max(ar,br)>=best,max(ar,br)<best]
   records.append({'action_row':r['action_row'],'active':len(active),'outsiders':len(outsiders),'street':street})
 assert len(records)==len(a) and np.isfinite(out).all();return out,records
def main():
 ROOT.mkdir(exist_ok=True);(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'columns':COLS,'missing':'-2 for no postflop made hand or no outsiders, explicit outsider count','schedule':'same Cat400 depth5 lr.035 L2 8 seeds6311+11fold+head','baseline':'R32 current-only'},indent=2));full=hand_data();base=pl.read_parquet(PAIR_ROOT/'current/event_oof.parquet');parts=[];audit=[];start=time.time()
 for family in ['directed_transfer','soft_play','coordinated_isolation']:
  root=ROOT/family;root.mkdir(exist_ok=True);d,a,x,cols=load(family);ex,records=compute(d,a);np.savez_compressed(root/'features.npz',x=ex);(root/'feature_audit.json').write_text(json.dumps(records));current=np.load(PAIR_ROOT/family/'features.npz')['x'][:,[3,6]];extra=np.column_stack([current,ex]);g=a['row'].to_numpy();r=a['actor'].to_numpy();iso=family=='coordinated_isolation';xx=np.full((len(d),extra.shape[1]),-2.) if iso else extra
  if iso:xx[g]=extra
  x=np.column_stack([x,xx]);pp=np.zeros((len(d),2));fv=d['fold'].to_numpy();fam=full['behavior_family'].to_numpy()==family
  for f in range(4):
   if iso:p1,p2,e1,e2,_=targets(full,f,family);ys=np.column_stack([p1,p2])[fam];es=np.column_stack([e1,e2])[fam];va=fv==f
   elif family=='directed_transfer':y,tr,va=labels(d,a,f)
   else:y0,e,_=target(d,f);y=y0[g];tr=e[g];va=fv[g]==f
   for k in range(2 if iso else 1):
    if iso:y=ys[:,k];tr=es[:,k]
    assert not(tr&va).any();m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6311+11*f+k,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[tr],y[tr]);m.save_model(str(root/f'event{k+1}_fold{f}.cbm'));p=m.predict_proba(x[va],thread_count=2)[:,1]
    if iso:pp[va,k]=p
    else:pp[g[va],r[va]]=p
    audit.append({'family':family,'fold':f,'head':k+1,'training_count':int(tr.sum()),'positive_count':int(y[tr].sum()),'validation_overlap':0})
   print('outsider rank',family,f,round(time.time()-start,1),flush=True)
  if iso:new=pp
  else:
   dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy() if family=='directed_transfer' else np.ones_like(pp);new=(pp*dw).sum(1)[:,None]
  parts.append(d.select('pair_id','hand_id').with_columns(pl.Series('new_primary',new[:,0]),pl.Series('new_secondary',new[:,1] if iso else np.full(len(d),np.nan))))
 q=base.join(pl.concat(parts),on=['pair_id','hand_id'],validate='1:1').with_columns(C('new_primary').alias('bg_primary'),pl.when(C('new_secondary').is_nan()).then(C('bg_secondary')).otherwise(C('new_secondary')).alias('bg_secondary')).drop('new_primary','new_secondary');q.write_parquet(ROOT/'event_oof.parquet');(ROOT/'audit.json').write_text(json.dumps(audit,indent=2));assemble(ROOT)
if __name__=='__main__':main()
