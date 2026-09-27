from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.metrics import average_precision_score as ap
root=Path('artifacts/policy/population_rank_fusion')
d=pd.read_csv(root/'oof.csv');metrics=json.loads((root/'metrics.json').read_text())
assert len(d.withheld_family.unique())==4,'Wait for all family tests to finish.'
rows=[]
for family,z in d[d.withheld_family!='all_known'].groupby('withheld_family'):
    y=z.truth.to_numpy();hidden=z.behavior.to_numpy()==family;known=(y>0)&~hidden
    for share in [0,.05,.1,.2,.25,1/3]:
        w=np.where(y>0,1.,50.)
        w[hidden]=0 if share==0 else share/(1-share)*known.sum()/hidden.sum()
        for method in ['simple','monotone']:
            for weight in [0,.5,.75,1]:
                p=np.maximum(z.base_rarity.to_numpy(),weight*z[method+'_rarity'].to_numpy())
                rows.append(dict(withheld_family=family,unknown_share=share,method=method,weight=weight,weighted_AP=ap(y,p,sample_weight=w)))
df=pd.DataFrame(rows);df.to_csv(root/'mixtures.csv',index=False)
summary=df.groupby(['unknown_share','method','weight']).weighted_AP.mean().reset_index();summary.to_csv(root/'mixture_summary.csv',index=False)
choices=[]
for method in ['simple','monotone']:
    for weight in [.5,.75,1]:
        baseline=next(r['weighted_AP'] for r in metrics if r['withheld_family']=='all_known' and r['novelty']==method and r['weight']==0)
        known=next(r['weighted_AP'] for r in metrics if r['withheld_family']=='all_known' and r['novelty']==method and r['weight']==weight)
        unseen=[r['weighted_AP'] for r in metrics if r['withheld_family']!='all_known' and r['novelty']==method and r['weight']==weight]
        choices.append(dict(method=method,weight=weight,known_loss=baseline-known,mean_withheld_AP=float(np.mean(unseen)),worst_withheld_AP=float(min(unseen)),eligible=baseline-known<=.005))
(root/'comparison.json').write_text(json.dumps(dict(choices=choices,caveat='Exploratory comparisons on the same public labels; unknown shares are sensitivity scenarios, not measured test proportions.'),indent=2))
print(json.dumps(choices,indent=2));print(summary.to_string(index=False))
