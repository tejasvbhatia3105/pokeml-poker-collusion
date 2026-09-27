import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl
from session8_data import reference
from session10_list_learning import ROOT
C=pl.col
def main():
    meta=reference().select('pair_id','table_id').unique();small=pl.read_csv(ROOT/'teacher_shift.csv').group_by('pair_id').agg(C('gain').mean().alias('small_teacher_gain'));large=pl.read_csv(ROOT/'comparison.csv').select('pair_id',(C('independent')-C('r29')).alias('large_teacher_gain'));r=small.join(large,on='pair_id',validate='1:1').join(meta,on='pair_id',validate='1:1').with_columns((C('small_teacher_gain')-C('large_teacher_gain')).alias('gain_difference'));names=['small_teacher_gain','large_teacher_gain','gain_difference'];p=r.group_by('table_id').agg(C(names).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(10101).integers(0,len(p),(4000,len(p)));out={}
    for n in names:
        boot=p[n].to_numpy()[ix].sum(1)/p['n'].to_numpy()[ix].sum(1);out[n]={'mean':r[n].mean(),'fixed_predictions_pool_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist()}
    out['caveat']='Post-result diagnostic. Pair repeats are averaged before pool resampling. Differential gains can reflect larger error headroom as well as teacher-distribution mismatch; they do not identify causality or include model-search uncertainty.';(ROOT/'teacher_shift_uncertainty.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
