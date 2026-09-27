import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import polars as pl
from session8_data import hand_data,reference
C=pl.col;ROOT=Path('artifacts/evidence_session10');ROOT.mkdir(exist_ok=True)
def main():
    d=hand_data().join(reference().select('pair_id','hand_id','r29'),on=['pair_id','hand_id']);labs=pl.read_csv('data/development_labels.csv');out=[]
    for b in ['directed_transfer','soft_play','coordinated_isolation']:
        cases=[]
        for (pid,),g in d.filter(C('behavior_family')==b).group_by('pair_id'):
            g=g.sort('r29',descending=True).with_row_index('rr',offset=1);fp=g.filter((C('rr')<=5)&(C('evidence')==0));fn=g.filter((C('rr')>5)&(C('evidence')==1))
            if fp.height and fn.height:cases.append((abs(fp['r29'][0]-fn['r29'][0]),pid,g,fp.head(1),fn.head(1)))
        cases.sort(key=lambda x:(x[0],x[1]))
        for _,pid,g,fp,fn in cases[:3]:
            table=g['table_id'][0];hands=pl.concat([fp,fn]);a=pl.read_parquet(f'artifacts/policy/actions/{table}.parquet');lab=labs.filter(C('pair_id')==pid).row(0,named=True);raw=pl.read_parquet(f'artifacts/compact/actions/table_id={table}/*.parquet');seats=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet');board=pl.read_parquet(f'artifacts/compact/hands/table_id={table}/*.parquet');out.append({'family':b,'pair_id':pid,'player_1':lab['player_1'],'player_2':lab['player_2'],'hands':[]})
            for r in hands.to_dicts():
                hid=r['hand_id'];ar=a.filter(C('hand_id')==hid).with_columns(C('action_no').cast(pl.Int64)).sort('action_no');rr=raw.filter(C('hand_id')==hid).sort('action_no');su=seats.filter(C('hand_id')==hid);out[-1]['hands'].append({'hand_id':hid,'score':r['r29'],'rank':r['rr'],'truth':r['evidence'],'truth_rank':r['evidence_rank'],'time':r['time'],'board':board.filter(C('hand_id')==hid)['board_cards'][0],'seats':su.to_dicts(),'actions':rr.join(ar.select('hand_id','action_no','equity','made_category','last_aggressor'),on=['hand_id','action_no'],validate='1:1').to_dicts()})
    (ROOT/'cutoff_cases.json').write_text(json.dumps(out,indent=2))
    for q in out:
        print('\nPAIR',q['family'],q['pair_id'],q['player_1'],q['player_2'])
        for h in q['hands']:
            print('HAND',h['hand_id'],'truth',h['truth'],'rank',h['rank'],'score',round(h['score'],3),'board',h['board'],[(s['player_id'],s['hole_card_1']+s['hole_card_2'],s['net_chips']) for s in h['seats'] if s['player_id'] in [q['player_1'],q['player_2']]])
            for a in h['actions']:print('*' if a['player_id'] in [q['player_1'],q['player_2']] else ' ',a['action_no'],a['player_id'],a['street'],a['action'],'amount',a['amount'],'pot',a['pot_before'],'call',a['to_call'],'eq',round(a['equity'],2),'active',a['players_active'])
if __name__=='__main__':main()
