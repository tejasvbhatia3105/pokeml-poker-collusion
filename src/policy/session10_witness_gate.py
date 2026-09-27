\
\
\
\
\
import os,json,itertools
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl
from session8_data import hand_data,targets,reference
from session10_list_learning import ROOT
from session6_priority import inclusion
C=pl.col
FLAGS=['fold_partner','call_partner','check_hu','agg_partner','agg_out','strong_check_hu','strong_fold_partner','weak_call_partner']
def main():
    d=hand_data();ref=reference().select('pair_id','hand_id','r29');cat=pl.read_parquet('artifacts/evidence_session6/priority_ordered_oof.parquet').select('pair_id','hand_id','primary','secondary');hist=pl.read_parquet('artifacts/evidence_session7/hist_events_oof.parquet').select('pair_id','hand_id',C('primary').alias('hist_primary'),C('secondary').alias('hist_secondary'));q=d.join(cat,on=['pair_id','hand_id'],validate='1:1',maintain_order='left').join(hist,on=['pair_id','hand_id'],validate='1:1',maintain_order='left').join(ref,on=['pair_id','hand_id'],validate='1:1',maintain_order='left');assert q.select('pair_id','hand_id').equals(d.select('pair_id','hand_id'));flags=d.select(FLAGS).to_numpy()>0;combinations=[c for size in (1,2) for c in itertools.combinations(range(len(FLAGS)),size)];rules=[flags[:,c].any(1) for c in combinations];gates=np.ones((len(d),2));audit=[]
    (ROOT/'witness_gate_config.json').write_text(json.dumps({'flags':FLAGS,'max_union':2,'selection':'100% training-positive recall, maximize eligible-negative exclusions, prefer fewer flags and fixed lexical flag order'},indent=2))
    for f in range(4):
        for b in ['directed_transfer','soft_play','coordinated_isolation']:
            t1,t2,e1,e2,va=targets(d,f,b)
            for head,y,tr in [(0,t1,e1),(1,t2,e2)]:
                pos=tr&y;neg=tr&~y;valid=[i for i,mask in enumerate(rules) if mask[pos].all()];best=max(valid,key=lambda i:(int((~rules[i]&neg).sum()),-len(combinations[i]),-i)) if valid else None;rule=rules[best] if best is not None else np.ones(len(d),bool);gates[va,head]=rule[va];audit.append({'fold':f,'family':b,'head':head+1,'flags':[FLAGS[i] for i in combinations[best]] if best is not None else [],'training_positive_hands':int(pos.sum()),'training_negative_exclusions':int((neg&~rule).sum())})
    q=q.with_columns(pl.Series('gate1',gates[:,0]),pl.Series('gate2',gates[:,1]));parts=[]
    for _,g in q.group_by('pair_id'):
        g=g.sort('time','hand_id');a=g['primary'].to_numpy();b=g['secondary'].to_numpy();ha=g['hist_primary'].to_numpy();hb=g['hist_secondary'].to_numpy();cs=np.maximum(1,a+b);hs=np.maximum(1,ha+hb);oldp=inclusion(a,b);oldh=inclusion(.5*(a/cs+ha/hs),.5*(b/cs+hb/hs));g1=g['gate1'].to_numpy();g2=g['gate2'].to_numpy();a=a*g1;b=b*g2;ha=ha*g1;hb=hb*g2;cs=np.maximum(1,a+b);hs=np.maximum(1,ha+hb);p=inclusion(a,b);h=inclusion(.5*(a/cs+ha/hs),.5*(b/cs+hb/hs));score=g['r29'].to_numpy()+.25*(p-oldp)+.5*(h-oldh);parts.append(g.select('pair_id','hand_id','fold','evidence').with_columns(pl.Series('witness_gate',score)))
    pl.concat(parts).write_parquet(ROOT/'witness_gate_oof.parquet');(ROOT/'witness_gate_audit.json').write_text(json.dumps(audit,indent=2))
if __name__=='__main__':main()
