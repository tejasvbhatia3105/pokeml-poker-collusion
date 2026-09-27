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
from session8_data import hand_data
from session35_fold_likelihood import labels,OLD,EXACT
from session12_event_replacement import assemble
C=pl.col;ROOT=Path('artifacts/evidence_session37_bet_fold')

def paired_features(d,a,response_class=0,players=None):
 pc=json.load(open('artifacts/policy/feature_columns.json'));people=(pl.read_csv('data/development_labels.csv') if players is None else players).select('pair_id','player_1','player_2');a=a.join(people,on='pair_id',validate='m:1',maintain_order='left');rows=[];align=[]
 diff=[c for c in pc if c not in ['big_blind','street_no','action_no','previous_action','prior_raises','self_last_aggressor']]
 for (table,),q in d.group_by('table_id'):
  z=a.join(q.select('pair_id','hand_id'),on=['pair_id','hand_id'],how='semi');raw=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet').join(z.select('hand_id').unique(),on='hand_id',how='semi');hands={hid:g.sort('action_no').to_dicts() for (hid,),g in raw.group_by('hand_id')}
  for r in z.to_dicts():
   actor=r['player_1'] if r['actor']==0 else r['player_2'];partner=r['player_2'] if r['actor']==0 else r['player_1'];hr=hands[r['hand_id']];fold=next(t for t in hr if t['action_no']==r['action_no']);assert fold['player_id']==actor and fold['action_class']==response_class and fold['last_aggressor']==partner;before=[t for t in hr if t['action_no']<r['action_no'] and t['street_no']==r['street_no']];bet=next(t for t in reversed(before) if t['player_id']==partner and t['action_class']==3);between=[t for t in before if t['action_no']>bet['action_no']];v={f'bet_{c}':bet[c] for c in pc};v.update({f'fold_minus_bet_{c}':fold[c]-bet[c] for c in diff});v.update({'bet_log_amount_bb':bet['log_amount_bb'],'bet_log_bet_ratio':bet['log_bet_ratio'],'action_gap':fold['action_no']-bet['action_no'],'pot_growth_bb':fold['pot_bb']-bet['pot_bb'],'bet_stack_fraction':bet['amount']/max(1,bet['stack_bb']*bet['big_blind']),'bet_allin':float(abs(bet['amount']-bet['stack_bb']*bet['big_blind'])<1e-4),'between_fold_count':sum(t['action_class']==0 for t in between),'between_call_count':sum(t['action_class']==2 for t in between),'between_raise_count':sum(t['action_class']==3 for t in between),'between_other_amount_bb':sum(t['amount'] for t in between if t['player_id'] not in [actor,partner])/fold['big_blind']});assert v['between_raise_count']==0;rows.append({'action_row':r['action_row'],**v});align.append({'action_row':r['action_row'],'pair_id':r['pair_id'],'hand_id':r['hand_id'],'fold_action_no':int(fold['action_no']),'bet_action_no':int(bet['action_no']),'actor':actor,'partner':partner})
 out=pl.DataFrame(rows).sort('action_row');assert np.array_equal(out['action_row'],np.arange(len(a)));return out,pl.DataFrame(align).sort('action_row')

def main():
 ROOT.mkdir(exist_ok=True);d=hand_data().filter(C('behavior_family')=='directed_transfer').drop('row').with_row_index('row');a=pl.read_parquet(EXACT/'fold_actions.parquet').with_row_index('action_row');extra,alignment=paired_features(d,a);extra.write_parquet(ROOT/'action_features.parquet');alignment.write_parquet(ROOT/'alignment.parquet');cfg=json.load(open(EXACT/'columns.json'));cols=[c for c in extra.columns if c!='action_row'];ex=extra.select(cols).to_numpy();g=a['row'].to_numpy();actor=a['actor'].to_numpy();ax=a.select(cfg['action']).to_numpy();hx=d.select(cfg['hand']).to_numpy()[g];dw=d.select('pair_id').join(pl.read_parquet(OLD/'actor_oof.parquet').select('pair_id','actor0','actor1'),on='pair_id',validate='m:1',maintain_order='left').select('actor0','actor1').to_numpy();pred={k:np.zeros((len(d),2)) for k in ['paired_action','paired_hand']};audit=[];start=time.time();(ROOT/'config.json').write_text(json.dumps({'method':__doc__,'paired_columns':cols,'fold_columns':cfg['action'],'hand_columns':cfg['hand'],'arms':['paired_action','paired_hand'],'schedule':'original Cat400 depth5 lr.035 L2 8 primary seed','targets':'same session26 donor-fold labels and censoring','limitation':'all features describe public gameplay; actor inferred on heldout pools; no new certified action-level labels'},indent=2))
 for f in range(4):
  y,tr,va=labels(d,a,f)
  for kind in pred:
   root=ROOT/kind;root.mkdir(exist_ok=True);x=np.column_stack([ax,ex] if kind=='paired_action' else [ax,hx,ex]);m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6311+11*f,thread_count=2,verbose=False,allow_writing_files=False);m.fit(x[tr],y[tr]);m.save_model(str(root/f'primary_fold{f}.cbm'));pred[kind][g[va],actor[va]]=m.predict_proba(x[va],thread_count=2)[:,1];audit.append({'fold':f,'kind':kind,'training_actions':int(tr.sum()),'positives':int(y[tr].sum()),'heldout_actions':int(va.sum()),'overlap':0});print('bet fold',f,kind,round(time.time()-start,1),flush=True)
 (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));base=pl.read_parquet(OLD/'oriented/event_oof.parquet')
 for kind,pp in pred.items():
  root=ROOT/kind;d.select('pair_id','hand_id').with_columns(pl.Series('actor0_primary',pp[:,0]),pl.Series('actor1_primary',pp[:,1])).write_parquet(root/'conditional_primary.parquet');q=d.select('pair_id','hand_id').with_columns(pl.Series('new_primary',(pp*dw).sum(1)));base.join(q,on=['pair_id','hand_id'],how='left',validate='1:1').with_columns(pl.coalesce('new_primary','bg_primary').alias('bg_primary')).drop('new_primary').write_parquet(root/'event_oof.parquet');assemble(root)
if __name__=='__main__':main()
