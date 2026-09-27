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
import session108_action_frontier as f

ROOT=Path('artifacts/evidence_session110_frontier_workers');C=pl.col

def worker(index,count,backend='python'):
    ROOT.mkdir(exist_ok=True);cfg=json.loads((f.ROOT/'config.json').read_text())
    assert cfg==dict(f.CONFIG,provenance_sha256=f.provenance())
    backend_hashes=None
    if backend=='native':
        import session111_native_rollout as native
        backend_hashes=native.provenance()
        for name in ['pilot.json','raw_verification.json']:
            verification=json.loads((native.ROOT/name).read_text());assert verification['provenance']==backend_hashes
        f.simulate=native.simulate
    else:assert backend=='python'
    q=pl.read_parquet(f.ROOT/'queries.parquet');tables=q['table_id'].unique().sort().to_list()
    assigned=tables[index::count];start=time.time();audits=[]
    p=Path(__file__);(ROOT/f'worker{index}_{backend}_config.json').write_text(json.dumps(dict(worker=index,count=count,tables=assigned,
        orchestration_source_sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),feature_config=cfg,backend_provenance=backend_hashes),indent=2))
    for table in assigned:
        local=q.filter(C('table_id')==table);native=int(local['fold'][0]);pending=[]
        for outer in range(4):
            if outer==native:continue
            a,b=sorted([native,outer]);path=f.ROOT/'tables'/f'{table}_exclude{a}{b}.parquet'
            if path.exists() and path.with_suffix('.json').exists():
                old=json.loads(path.with_suffix('.json').read_text());assert old['config']==cfg
                np.testing.assert_array_equal(pl.read_parquet(path).sort('query_id')['query_id'],local.sort('query_id')['query_id']);audits.append(old)
            else:pending.append((outer,path))
        if not pending:continue
        roots=f.s.roots_for_table(table,local,cfg['replicates'])
        for outer,path in pending:
            action,size,meta=f.s.models(native,outer);assert table not in meta['training_tables'];parts=[];infos=[]
            for pos in range(0,len(roots),cfg['root_batch']):
                z,info,_,_=f.evaluate(roots[pos:pos+cfg['root_batch']],action,size,meta,cfg['replicates']);parts.append(z);infos.append(info)
            out=pl.concat(parts).sort('query_id');np.testing.assert_array_equal(out['query_id'],local.sort('query_id')['query_id'])
            assert np.isfinite(out.drop('query_id').to_numpy()).all();out.write_parquet(path)
            audit=dict(table_id=table,native_fold=native,other_excluded_fold=outer,query_rows=len(local),roots=len(roots),config=cfg,
                actions=sum(x['actions'] for x in infos),trajectories=sum(x['trajectories'] for x in infos),menu_cases=sum(x['menu_cases'] for x in infos),
                chip_conservation_error=max(x['chip_conservation_error'] for x in infos),elapsed_seconds=time.time()-start,worker=index,
                backend=backend,backend_provenance=backend_hashes)
            path.with_suffix('.json').write_text(json.dumps(audit,indent=2));audits.append(audit)
            print('FRONTIER_WORKER',index,table,'exclude',native,outer,'roots',len(roots),'seconds',time.time()-start,flush=True)
        del roots;gc.collect()
    (ROOT/f'worker{index}_audit.json').write_text(json.dumps(audits,indent=2));print('WORKER_COMPLETE',index,len(assigned),len(audits),time.time()-start,flush=True)

def collect():
    q=pl.read_parquet(f.ROOT/'queries.parquet');cfg=json.loads((f.ROOT/'config.json').read_text());audits=[]
    assert cfg==dict(f.CONFIG,provenance_sha256=f.provenance())
    for (table,),local in q.group_by('table_id'):
        native=int(local['fold'][0])
        for outer in range(4):
            if native==outer:continue
            a,b=sorted([native,outer]);path=f.ROOT/'tables'/f'{table}_exclude{a}{b}.parquet';z=pl.read_parquet(path)
            audit=json.loads(path.with_suffix('.json').read_text());assert audit['config']==cfg
            np.testing.assert_array_equal(z.sort('query_id')['query_id'],local.sort('query_id')['query_id']);audits.append(audit)
    assert len(audits)==3*q['table_id'].n_unique()
    (f.ROOT/'build_audit.json').write_text(json.dumps(audits,indent=2));print('COLLECTED',len(audits))

if __name__=='__main__':
    if sys.argv[1]=='collect':collect()
    else:worker(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3] if len(sys.argv)>3 else 'python')
