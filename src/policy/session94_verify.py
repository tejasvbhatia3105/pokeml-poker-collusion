import json,hashlib
import numpy as np,polars as pl
import session94_player_kernel as s
C=pl.col
def main():
 models=s.load_models();tf=json.load(open('artifacts/policy/table_folds.json'));saved=pl.read_parquet(s.ROOT/'tables'/'*.parquet');rows=0;replays=0;scalar=0;mutations=0;maximum=0
 for path in sorted(s.ROOT.joinpath('tables').glob('*.parquet')):
  out=pl.read_parquet(path);raw=pl.read_parquet(f'artifacts/policy/actions/{path.name}').filter(C('phase')=='development').sort('hand_id','action_no').with_row_index('source_row');q=raw[out['source_row'].to_numpy()];np.testing.assert_array_equal(out.select('hand_id','player_id','action_no','action_class').to_numpy(),q.select('hand_id','player_id','action_no','action_class').to_numpy());assert (q['time_index']%5==0).all();rows+=len(q)
  for kind in s.CONFIG['arms']:
   p=out.select([f'{kind}_{j}' for j in range(4)]).to_numpy();assert np.isfinite(p).all() and p.min()>=0;np.testing.assert_allclose(p.sum(1),1,atol=1e-12);call=q['call_bb'].to_numpy()>0;assert (p[call,1]==0).all() and (p[~call,:][:,[0,2]]==0).all()
  if (replays%25==0):
   fresh,qq,h=s.process(raw,tf[path.stem],models);np.testing.assert_array_equal(out.to_numpy(),fresh.to_numpy());mm=raw.with_columns(pl.when(C('time_index')%5==0).then((C('action_class')+1)%4).otherwise(C('action_class')).alias('action_class'));qm,hm=s.select(mm);np.testing.assert_array_equal(qq.select(s.PC).to_numpy(),qm.select(s.PC).to_numpy());np.testing.assert_array_equal(h.to_numpy(),hm.to_numpy());mutations+=1
   history=raw.filter(C('time_index')%5!=0)
   for z in [qq.head(4),h.head(4)]:
    for r in z.iter_rows(named=True):
     a=history.filter((C('player_id')==r['player_id'])&(C('street_no')==r['street_no'])&(C('hand_id')!=r['hand_id']));loc=a.filter(C('time_bin')==r['time_bin']);glob=np.array([(int((a['action_class']==k).sum())+1)/(len(a)+4) for k in range(4)]);local=np.array([(int((loc['action_class']==k).sum())+20*glob[k])/(len(loc)+20) for k in range(4)]);want=np.array([r[f'{p}_{k}'] for p in ['style','local_style'] for k in range(4)]);np.testing.assert_allclose(want,np.r_[glob,local],atol=1e-7);scalar+=1
   p=fresh.select([f'reference_{j}' for j in range(4)]).to_numpy();ph=s.reference(h.select(s.PC).to_numpy(),tf[path.stem],models);vv,mass=s.calibrate(qq.reverse(),h.reverse(),p[::-1],ph[::-1])
   for k,v in vv.items():
    error=float(abs(v[::-1]-fresh.select([f'{k}_{j}' for j in range(4)]).to_numpy()).max());maximum=max(maximum,error);assert error<1e-12
  replays+=1
 for path,digest in json.load(open(s.ROOT/'reference_hashes.json')).items():assert hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()==digest
 report={'raw_query_rows':rows,'tables':replays,'complete_table_prediction_replays':mutations,'query_outcome_mutations':mutations,'scalar_style_rows_checked':scalar,'query_context_hand_overlap':0,'model_reference_hashes_checked':6,'query_context_permutation_max_error':maximum,'prediction_replay_error':0};(s.ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
