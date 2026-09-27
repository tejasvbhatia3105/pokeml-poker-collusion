import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl
from session8_data import reference
from session10_list_learning import ROOT
C=pl.col
def main():
    ref=reference();meta=ref.group_by('pair_id').agg(C('table_id').first(),C('fold').first(),C('behavior_family').first().alias('family'),C('evidence').sum().alias('den'));truth={pid:set(g.filter(C('evidence')==1)['hand_id']) for (pid,),g in ref.group_by('pair_id')};sources={'r29':(ref,'r29')}
    for file,names in [('list_learning_oof.parquet',['independent','contextual']),('ranker_oof.parquet',['ranker_compact','ranker_full']),('activity_oof.parquet',['activity']),('matched_teacher_ensemble_oof.parquet',['matched_base','matched_corrected'])]:
        if (ROOT/file).exists():
            d=pl.read_parquet(ROOT/file)
            for name in names:sources[name]=(d,name)
    cat=pl.read_parquet('artifacts/evidence_session6/priority_ordered_oof.parquet').select('pair_id','hand_id',C('score').alias('cat_inclusion'))
    if (ROOT/'nested6_model/list_learning_oof.parquet').exists():sources['nested6']=(pl.read_parquet(ROOT/'nested6_model/list_learning_oof.parquet'),'independent')
    if (ROOT/'witness_gate_oof.parquet').exists():sources['witness_gate']=(pl.read_parquet(ROOT/'witness_gate_oof.parquet'),'witness_gate')
    for kind in ['outcome','history','policy']:
        file=ROOT/f'role_{kind}_oof.parquet'
        if file.exists():
            name='role_'+kind;d=ref.join(cat,on=['pair_id','hand_id'],validate='1:1').join(pl.read_parquet(file).select('pair_id','hand_id','inclusion'),on=['pair_id','hand_id'],validate='1:1').with_columns((C('r29')+.25*(C('inclusion')-C('cat_inclusion'))).alias(name));sources[name]=(d,name)
    results={};hands={};recalls={}
    for name,(d,col) in sources.items():
        results[name]={};hands[name]={};recalls[name]={}
        for (pid,),g in d.group_by('pair_id'):
            chosen=g.sort([col,'hand_id'],descending=[True,False])['hand_id'].to_list()[:5];y=np.array([h in truth[pid] for h in chosen]);den=min(5,len(truth[pid]));results[name][pid]=float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/den);recalls[name][pid]=float(y.sum()/den);hands[name][pid]=' '.join(chosen)
    r=meta.sort('pair_id');ids=r['pair_id'].to_list();route=pl.read_csv('artifacts/evidence_session9/routed_evidence.csv').filter(C('window')=='full').select('pair_id','below_gate',C('routed_r29').alias('deployed_r29'));r=r.join(route,on='pair_id',validate='1:1',maintain_order='left');names=list(sources)
    for name in names:r=r.with_columns(pl.Series(name,[results[name][p] for p in ids]),pl.Series(name+'_recall',[recalls[name][p] for p in ids]),pl.Series(name+'_hands',[hands[name][p] for p in ids])).with_columns(pl.when(C('below_gate')).then(C('deployed_r29')).otherwise(C(name)).alias(name+'_routed'))
    r.write_csv(ROOT/'comparison.csv');names2=[n+'_routed' for n in names];p=r.group_by('table_id').agg(C(names2).sum(),pl.len().alias('n')).sort('table_id');ix=np.random.default_rng(1010).integers(0,len(p),(4000,len(p)));report={}
    for name in names:
        n=name+'_routed';delta=p[n].to_numpy()-p['r29_routed'].to_numpy();boot=delta[ix].sum(1)/p['n'].to_numpy()[ix].sum(1);report[name]={'oracle_family_map':r[name].mean(),'routed_map':r[n].mean(),'routed_gain':r[n].mean()-r['r29_routed'].mean(),'routed_ci95_fixed_predictions':np.quantile(boot,[.025,.975]).tolist(),'recall5':r[name+'_recall'].mean(),'routed_folds':r.group_by('fold').agg(C(n).mean()).sort('fold')[n].to_list(),'routed_families':dict(r.group_by('family').agg(C(n).mean()).iter_rows()),'improved_pairs':int((r[n]>r['r29_routed']+1e-12).sum()),'worse_pairs':int((r[n]<r['r29_routed']-1e-12).sum())}
    (ROOT/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
