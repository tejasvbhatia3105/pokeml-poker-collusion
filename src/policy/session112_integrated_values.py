\
\
\
\
\
\
import gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import polars as pl
import session104_allstreet_rollout as s
import session108_action_frontier as frontier
import session111_native_rollout as native
from session106_focal_integration import choices,integrate

ROOT=Path('artifacts/evidence_session112_integrated_values');C=pl.col
FIELDS=s.FIELDS
CONFIG={'method':__doc__,'replicates':256,'root_batch':4,
    'focal':'exact sum of all legal action classes and categorical raise bins; clipped duplicate payments merged',
    'responses':'fixed learned policy and matched check/call; same models and random streams as104',
    'outer_mapping':'three references excluding native h and outer f; train f uses(h,f), validation h mean3',
    'scaling':'same six value/standard-error fields per arm as104, divided by root pot'}

def provenance():
    out=frontier.provenance();out.update(native.provenance())
    for p in [Path(__file__),Path('src/policy/session106_focal_integration.py')]:
        out[str(p.resolve().relative_to(Path.cwd()))]=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
    return out

def evaluate(roots,action,size,meta,reps=256):
    expanded=[];offset=[];weights=[]
    for root in roots:
        start=len(expanded);expanded.append(root);w=[]
        for (k,amount),weight in choices(root,action,size,meta).items():
            expanded.append(dict(root,forced_class=k,forced_amount=amount));w.append(weight)
        assert abs(sum(w)-1)<1e-12;offset.append((start,len(expanded)));weights.append(w)
    result,info=native.simulate(expanded,action,size,meta,reps)
    out=np.zeros((len(roots),4,reps,6))
    for i,((start,end),w) in enumerate(zip(offset,weights)):
        out[i,0]=result[start,0];out[i,2]=result[start,1]
        for j,weight in enumerate(w,start+1):
            out[i,1]+=weight*result[j,0];out[i,3]+=weight*result[j,1]
    return s.values(roots,out),dict(focal_cases=len(expanded),**info),out

def initialize():
    ROOT.mkdir(exist_ok=True);(ROOT/'tables').mkdir(exist_ok=True)
    hashes=native.provenance()
    for name in ['pilot.json','raw_verification.json']:
        assert json.loads((native.ROOT/name).read_text())['provenance']==hashes
    cfg=dict(CONFIG,provenance_sha256=provenance());p=ROOT/'config.json'
    if p.exists():assert json.loads(p.read_text())==cfg
    else:p.write_text(json.dumps(cfg,indent=2))
    q=pl.read_parquet(s.ROOT/'queries.parquet');q.write_parquet(ROOT/'queries.parquet');return q,cfg

def pilot():
    q,cfg=initialize();records=[];start=time.time()
    for f in range(4):
        table=q.filter(C('fold')==f)['table_id'].unique().sort()[0];local=q.filter(C('table_id')==table);keys=[]
        for street in range(4):
            z=local.filter(C('street_no')==street).select('hand_id','action_no').unique().sort('hand_id','action_no').head(1)
            if len(z):keys.append(z)
        local=local.join(pl.concat(keys),on=['hand_id','action_no'],how='semi');roots=s.roots_for_table(table,local,256);action,size,meta=s.models(f,(f+1)%4)
        z,info,result=evaluate(roots,action,size,meta,256)
                                                                              
        _,expected,_=integrate(roots,action,size,meta,32);np.testing.assert_array_equal(result[:,:,:32],expected)
        small,_,prefix=evaluate(roots,action,size,meta,32);np.testing.assert_array_equal(prefix,result[:,:,:32])
        rev,_,rv=evaluate(roots[::-1],action,size,meta,256);np.testing.assert_array_equal(result,rv[::-1]);np.testing.assert_array_equal(z.sort('query_id').to_numpy(),rev.sort('query_id').to_numpy())
        records.append(dict(table=table,native_fold=f,roots=len(roots),mixture_reference_error=0,replicate_prefix_error=0,query_reversal_error=0,**info))
    report=dict(config=cfg,records=records,seconds=time.time()-start)
    (ROOT/'pilot.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

def worker(index,count):
    cfg=json.loads((ROOT/'config.json').read_text());assert cfg==dict(CONFIG,provenance_sha256=provenance())
    assert json.loads((ROOT/'pilot.json').read_text())['config']==cfg
    q=pl.read_parquet(ROOT/'queries.parquet');tables=q['table_id'].unique().sort().to_list()[index::count];audits=[];start=time.time()
    for table in tables:
        local=q.filter(C('table_id')==table);native_fold=int(local['fold'][0]);pending=[]
        for f in range(4):
            if f==native_fold:continue
            a,b=sorted([f,native_fold]);path=ROOT/'tables'/f'{table}_exclude{a}{b}.parquet'
            if path.exists() and path.with_suffix('.json').exists():
                audit=json.loads(path.with_suffix('.json').read_text());assert audit['config']==cfg
                np.testing.assert_array_equal(pl.read_parquet(path).sort('query_id')['query_id'],local.sort('query_id')['query_id']);audits.append(audit)
            else:pending.append((f,path))
        if not pending:continue
        roots=s.roots_for_table(table,local,cfg['replicates'])
        for f,path in pending:
            action,size,meta=s.models(native_fold,f);assert table not in meta['training_tables'];parts=[];infos=[]
            for pos in range(0,len(roots),cfg['root_batch']):
                z,info,_=evaluate(roots[pos:pos+cfg['root_batch']],action,size,meta,cfg['replicates']);parts.append(z);infos.append(info)
            out=pl.concat(parts).sort('query_id');np.testing.assert_array_equal(out['query_id'],local.sort('query_id')['query_id']);assert np.isfinite(out.drop('query_id').to_numpy()).all();out.write_parquet(path)
            audit=dict(table_id=table,native_fold=native_fold,other_excluded_fold=f,query_rows=len(local),roots=len(roots),worker=index,config=cfg,
                trajectories=sum(v['trajectories'] for v in infos),actions=sum(v['actions'] for v in infos),focal_cases=sum(v['focal_cases'] for v in infos),
                chip_conservation_error=max(v['chip_conservation_error'] for v in infos),elapsed_seconds=time.time()-start)
            path.with_suffix('.json').write_text(json.dumps(audit,indent=2));audits.append(audit)
            print('INTEGRATED_POOL',index,table,'exclude',native_fold,f,'roots',len(roots),'seconds',time.time()-start,flush=True)
        del roots;gc.collect()
    (ROOT/f'worker{index}_audit.json').write_text(json.dumps(audits,indent=2));print('INTEGRATED_WORKER_COMPLETE',index,len(audits),time.time()-start,flush=True)

if __name__=='__main__':
    if sys.argv[1]=='pilot':pilot()
    else:worker(int(sys.argv[1]),int(sys.argv[2]))
