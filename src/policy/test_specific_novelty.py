import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import json,numpy as np,pandas as pd,polars as pl
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap
root=Path('artifacts/policy');dest=root/'specific_novelty';dest.mkdir(exist_ok=True)
C=pl.col;labs=pl.read_csv('data/development_labels.csv').select('pair_id','label','behavior_family')
a=pl.scan_parquet(str(root/'pair_features/*.parquet')).filter(C('phase')=='development').join(pl.scan_parquet(str(root/'relationship_context/features/*.parquet')),on=['pair_id','phase']).collect().sort('pair_id');d=a.join(labs,on='pair_id').sort('pair_id')
cols=[c for c in json.loads((root/'residual_columns.json').read_text()) if c.endswith(('_z','_w10_max','_w10_min'))];peercols=[c for c in a.columns if c.startswith('peer_') and '_excess_' in c]
folds=json.loads(Path('artifacts/folds.json').read_text());score={k:np.zeros(len(d)) for k in ['peer_excess','gated','sqrt_gated','half_gated']};g=d['table_id'].to_numpy();y=d['label'].to_numpy()
for f in folds:
    va=np.isin(g,f['valid_tables']);normal=~va&(y==0);av=a['table_id'].is_in(f['valid_tables']).to_numpy();v=a.filter(pl.Series(av));NX=d.filter(pl.Series(normal)).select(cols).to_numpy();med=np.median(NX,axis=0);scale=np.maximum(np.quantile(NX,.9,axis=0)-np.quantile(NX,.1,axis=0),.01);z=(v.select(cols).to_numpy()-med)/scale;az=np.log1p(abs(z))
    low=v.select(['peer_'+c+'_pct_min' for c in cols]).to_numpy();high=v.select(['peer_'+c+'_pct_max' for c in cols]).to_numpy();percentile=np.where(z>=0,low,1-high);gate=np.clip((percentile-.5)*2,0,1)
    NP=d.filter(pl.Series(normal)).select(peercols).to_numpy();pm=np.median(NP,axis=0);ps=np.maximum(np.quantile(NP,.9,axis=0)-np.quantile(NP,.1,axis=0),.01);pz=np.log1p(abs((v.select(peercols).to_numpy()-pm)/ps))
    for name,feat in [('peer_excess',pz),('gated',az*gate),('sqrt_gated',az*np.sqrt(gate)),('half_gated',az*(.5+.5*gate))]:
        s=np.sort(feat,axis=1)[:,-3:].mean(1);rarity=-np.log((len(s)+1-rankdata(s,method='average'))/(len(s)+1));frame=pd.Series(rarity,index=v['pair_id'].to_list());score[name][va]=frame.loc[d.filter(pl.Series(va))['pair_id'].to_list()]
out=pd.DataFrame(dict(pair_id=d['pair_id'].to_list(),**score));out.to_csv(dest/'scores.csv',index=False)
base=pd.read_csv(root/'population_rank_fusion/oof.csv');reports=[]
assert base.withheld_family.nunique()==4,'Wait for all population tests to finish.'
for family,z in base.merge(out,on='pair_id').groupby('withheld_family'):
    yy=z.truth.to_numpy();keep=np.ones(len(z),bool) if family=='all_known' else (yy==0)|(z.behavior.to_numpy()==family)
    for method in score:
        for weight in [.75,1]:
            p=np.maximum(z.base_rarity.to_numpy(),weight*z[method].to_numpy());reports.append(dict(withheld_family=family,method=method,weight=weight,weighted_AP=ap(yy[keep],p[keep],sample_weight=np.where(yy[keep]>0,1,50))))
(dest/'metrics.json').write_text(json.dumps(reports,indent=2));print(pd.DataFrame(reports).pivot_table(index=['method','weight'],columns='withheld_family',values='weighted_AP').to_string())
