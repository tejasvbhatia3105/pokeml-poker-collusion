\
\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl
from session9_pair_baseline import ROOT
C=pl.col
def ap(y):
    y=np.asarray(y);return float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/y.sum())
def main():
    names=['risk_score','gbdt_risk','seq_risk'];rows=[];failures=[]
    for w in ['full','first_2000','last_2000']:
        q=pl.read_parquet(ROOT/f'pair_{w}.parquet')
        for name in names:
            r=q.sort([name,'pair_id'],descending=[True,False]);known=r.filter(C('label')>=0);known_y=known['label'].to_numpy();other=np.cumsum(r['label'].to_numpy()!=1);p=r.filter(C('label')==1);above=other[r['label'].to_numpy()==1];row={'window':w,'component':name,'known_label_ap':ap(known_y),'known_positives':p.height,'known_negatives':known.height-p.height,'all_positive_recall_at_300_other_pairs':float(np.mean(above<300)),'all_positive_recall_at_1000_other_pairs':float(np.mean(above<1000)),'confirmed_negatives_top1000':r.head(1000).filter(C('label')==0).height}
            has_truth=p['n_truth'].to_numpy()>0
            for budget in [300,1000]:row[f'listed_truth_positive_recall_at_{budget}_other_pairs']=float(np.mean(above[has_truth]<budget))
            rows.append(row)
            if name=='risk_score':failures.append(p.with_columns(pl.Series('other_pairs_above',above)).sort('other_pairs_above',descending=True).head(20).with_columns(pl.lit(w).alias('window')))
    pl.DataFrame(rows).write_csv(ROOT/'ranking_controls.csv');pl.concat(failures).write_csv(ROOT/'ranking_tail_positives.csv');(ROOT/'ranking_controls.json').write_text(json.dumps({'caveats':__doc__,'controls':rows},indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
