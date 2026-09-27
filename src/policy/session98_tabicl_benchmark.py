import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import json,time,hashlib,gc
from pathlib import Path
import numpy as np,torch,psutil
from tabicl import TabICLClassifier
ROOT=Path('artifacts/evidence_session98_tabicl')
def main():
 torch.set_num_threads(2);assert torch.backends.mps.is_available();torch.mps.set_per_process_memory_fraction(.65);checkpoint=ROOT/'tabicl-classifier-v2-20260212.ckpt';assert hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()=='bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0';results=[]
 for features,train,query in [(128,2300,512),(929,2300,512)]:
  rng=np.random.default_rng(9801);x=rng.normal(size=(train+query,features)).astype(np.float32);y=(x[:train,0]+x[:train,1]*x[:train,2]+rng.normal(size=train)>.5).astype(int);start=time.time();print('START',features,train,query,flush=True);m=TabICLClassifier(n_estimators=1,batch_size=1,model_path=checkpoint,allow_auto_download=False,device='mps',n_jobs=2,offload_mode='cpu',random_state=9801,verbose=True,inference_config={key:{'safety_factor':.02} for key in ['COL_CONFIG','ROW_CONFIG','ICL_CONFIG']});m.fit(x[:train],y);p=m.predict_proba(x[train:]);assert np.isfinite(p).all();np.testing.assert_allclose(p.sum(1),1,atol=1e-6);r={'features':features,'training_rows':train,'query_rows':query,'seconds':time.time()-start,'rss_bytes':psutil.Process().memory_info().rss,'mps_allocated':torch.mps.current_allocated_memory(),'mps_driver':torch.mps.driver_allocated_memory(),'valid_probabilities':True};results.append(r);print('BENCHMARK',r,flush=True);(ROOT/'synthetic_benchmark.json').write_text(json.dumps(results,indent=2));del m,x,p;gc.collect();torch.mps.empty_cache()
if __name__=='__main__':main()
