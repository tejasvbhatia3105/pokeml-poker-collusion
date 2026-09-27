import os,json
os.environ.setdefault('POLARS_MAX_THREADS','3')
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session36_nested_cascade import ROOT,OLD
from session8_data import hand_data
from session6_priority import training_targets,inclusion
C=pl.col
def main():
 d=hand_data();cfg=json.load(open('artifacts/evidence_session6/priority_ordered_columns.json'));x=d.select(cfg['event']).to_numpy();xt=d.select(cfg['type']).to_numpy();fv=d['fold'].to_numpy();fam=d['behavior_family'].to_numpy();y=d['evidence'].to_numpy();sub=d['subtype'].to_numpy();tv=d['time'].to_numpy();ranks=d['evidence_rank'].fill_null(0).to_numpy();groups=[g['row'].to_numpy() for _,g in d.group_by('pair_id',maintain_order=True)];half=json.load(open(OLD/'table_half_split.json'));hv=np.array([half[t] for t in d['table_id']]);maximum=0.;heads=0;checks=0;valerrors=[]
 for outer in range(4):
  for parent in range(4):
   if parent==outer:continue
   for h in range(2):
    held=(fv==parent)&(hv==h);eligible=(fv!=outer)&~held;path=ROOT/f'conditional_outer{outer}_parent{parent}_half{h}.parquet';audit=json.load(open(path.with_suffix('.json')));q=d.select('pair_id','hand_id').join(pl.read_parquet(path),on=['pair_id','hand_id'],how='left',validate='1:1',maintain_order='left');expected=q['conditional_secondary'].to_numpy()
    for family in ['directed_transfer','soft_play','coordinated_isolation']:
     tr=eligible&(fam==family);va=held&(fam==family);known=tr&(sub>0);stem=f'outer{outer}_parent{parent}_half{h}_{family}';typ=CatBoostClassifier();typ.load_model(str(ROOT/(stem+'_type.cbm')));tp=typ.predict_proba(xt,thread_count=2)[:,1];a,b,ea,eb=training_targets(y,sub,tp,tr,tv,groups,ranks);mask=ea&eb&~a;assert not (mask&((fv==outer)|held)).any();assert not (known&((fv==outer)|held)).any();entry=next(z for z in audit if z['family']==family);assert entry['training_tables']==sorted(set(d['table_id'].to_numpy()[mask]));assert entry['training_hands']==int(mask.sum());assert entry['positive_hands']==int(b[mask].sum());m=CatBoostClassifier();m.load_model(str(ROOT/(stem+'.cbm')));pred=m.predict_proba(x[va],thread_count=2)[:,1];maximum=max(maximum,float(abs(pred-expected[va]).max()));heads+=1
                                                                       
     ym=y.copy();sm=sub.copy();rm=ranks.copy();tm=tv.copy();ym[~tr]=1-ym[~tr];sm[~tr]=2;rm[~tr]=999;tm[~tr]=-100;mut=training_targets(ym,sm,tp,tr,tm,groups,rm)
     for u,v in zip((a,b,ea,eb),mut):assert np.array_equal(u[tr],v[tr])
     assert not (mut[2]&mut[3]&~mut[0]&~tr).any();checks+=1
  new=pl.read_parquet(ROOT/f'nested_outer{outer}.parquet');old=pl.read_parquet(OLD/f'nested_outer{outer}.parquet');unchanged=['base','cat_primary','hist_primary','hist_secondary'];q=new.join(old.select('pair_id','hand_id',*unchanged),on=['pair_id','hand_id'],validate='1:1',suffix='_old');assert max(float((q[c]-q[c+'_old']).abs().max()) for c in unchanged)==0;val=new.filter(C('fold')==outer).join(pl.read_parquet('artifacts/evidence_session31_conditional_events/event_oof.parquet').select('pair_id','hand_id','bg_primary','bg_secondary'),on=['pair_id','hand_id'],validate='1:1');v=max(float((val['cat_primary']-val['bg_primary']).abs().max()),float((val['cat_secondary']-val['bg_secondary']).abs().max()));valerrors.append(v)
  for _,g in new.group_by('pair_id'):
   g=g.sort('time','hand_id');ca=g.select('cat_primary','cat_secondary').to_numpy();hp=g.select('hist_primary','hist_secondary').to_numpy();hp=hp/np.maximum(1,hp.sum(1))[:,None];assert np.max(ca.sum(1))<=1+1e-12;ci=inclusion(*ca.T);ji=inclusion(*((ca+hp)/2).T);assert max(abs(ci-g['cat_inclusion'].to_numpy()).max(),abs(ji-g['joint_inclusion'].to_numpy()).max())==0;assert abs(.25*g['base'].to_numpy()+.25*ci+.5*ji-g['r29'].to_numpy()).max()==0
 assert maximum==0 and max(valerrors)==0;report={'conditional_heads_replayed':heads,'max_probability_error':maximum,'type_models_reloaded_and_target_masks_rebuilt':checks,'outside_training_label_time_mutation_checks':checks,'outer_validation_event_errors':valerrors,'old_base_primary_hist_inputs_exact':True,'derived_scores_exact':True};(ROOT/'input_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
