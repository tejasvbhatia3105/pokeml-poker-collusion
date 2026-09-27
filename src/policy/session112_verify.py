import copy,json,sys
import numpy as np
import polars as pl
import session112_integrated_values as v
C=pl.col

def coverage():
    import session104_verify as verify
    v.queries=v.s.queries;verify.s=v;verify.coverage()
    audits=[json.loads(p.read_text()) for p in sorted((v.ROOT/'tables').glob('*.json'))]
    assert len(audits)==735
    (v.ROOT/'build_audit.json').write_text(json.dumps(audits,indent=2))

def replay():
    q=pl.read_parquet(v.ROOT/'queries.parquet');cfg=json.loads((v.ROOT/'config.json').read_text())
    assert cfg['provenance_sha256']==v.provenance();records=[];mutations=0
    for native in range(4):
        table=q.filter(C('fold')==native)['table_id'].unique().sort()[0];local=q.filter(C('table_id')==table);keys=[]
        for street in range(4):
            z=local.filter(C('street_no')==street).select('hand_id','action_no').unique().sort('hand_id','action_no').head(1)
            if len(z):keys.append(z)
        small=local.join(pl.concat(keys),on=['hand_id','action_no'],how='semi');roots=v.s.roots_for_table(table,small,256)
        original=v.s.engine.table_data;rootmap={(r['hand_id'],r['action_no']):r for r in roots}
                                                                             
                                                                            
                                                                              
        for street in range(4):
            subset=small.filter(C('street_no')==street)
            if not len(subset):continue
            def mutate(t,needed):
                raw,meta,seats=original(t,needed);meta=copy.deepcopy(meta);inverse={i:c for c,i in v.s.CARD.items()}
                for hid,ss in seats.items():
                    nb=0 if street==0 else street+2
                    visible=meta[hid]['board_cards'].split()[:nb];used=set(ss['hole_card_1'])|set(ss['hole_card_2'])|set(visible)
                    later=[inverse[c] for c in range(51,-1,-1) if inverse[c] not in used][:5-nb]
                    meta[hid]['board_cards']=' '.join(visible+later);meta[hid]['final_pot']+=777
                    seats[hid]=ss.with_columns((C('net_chips')+777).alias('net_chips'))
                return raw,meta,seats
            try:v.s.engine.table_data=mutate;altered=v.s.roots_for_table(table,subset,256)
            finally:v.s.engine.table_data=original
            for b in altered:
                a=rootmap[b['hand_id'],b['action_no']]
                for name in ['boards','templates','ranks']:np.testing.assert_array_equal(a[name],b[name])
                np.testing.assert_array_equal(v.native.state(a['state']),v.native.state(b['state']));mutations+=1
        for outer in range(4):
            if native==outer:continue
            action,size,meta=v.s.models(native,outer);z,info,result=v.evaluate(roots,action,size,meta,256)
            a,b=sorted([native,outer]);cached=pl.read_parquet(v.ROOT/'tables'/f'{table}_exclude{a}{b}.parquet')
            expected=z.select('query_id').join(cached,on='query_id',maintain_order='left',validate='1:1')
            np.testing.assert_array_equal(z.to_numpy(),expected.select(z.columns).to_numpy())
                                                                                  
            _,ref,_=v.integrate(roots,action,size,meta,32);np.testing.assert_array_equal(result[:,:,:32],ref)
            reverse,_,rv=v.evaluate(roots[::-1],action,size,meta,256)
            np.testing.assert_array_equal(result,rv[::-1]);np.testing.assert_array_equal(z.sort('query_id').to_numpy(),reverse.sort('query_id').to_numpy())
            records.append(dict(native=native,outer=outer,roots=len(roots),cache_error=0,independent_mixture_prefix_error=0,root_reversal_error=0,**info))
    report=dict(reference_blocks=len(records),future_outcome_mutation_roots=mutations,future_outcome_error=0,records=records)
    (v.ROOT/'replay_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:a for k,a in report.items() if k!='records'},indent=2))

if __name__=='__main__':{'coverage':coverage,'replay':replay}[sys.argv[1]]()
