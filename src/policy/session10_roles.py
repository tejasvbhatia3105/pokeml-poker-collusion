\
\
\
\
import os,json,time
os.environ.setdefault('POLARS_MAX_THREADS','4')
from pathlib import Path
import numpy as np,polars as pl
from catboost import CatBoostClassifier
from session8_data import hand_data,targets
from session6_priority import inclusion
from session10_list_learning import ROOT
C=pl.col
ATTR=['strength_0','strength_1','strength_2','strength_3','category_1','category_2','category_3','draw_1','draw_2','draw_3','holematch_1','holematch_2','holematch_3','high','low','pocket','suited','fold_no','net','position','contribution']
ATTR += [f'{street}_{dec}_{field}' for street in ['preflop','flop','turn','river'] for dec in range(2) for field in ['action_code','facing_partner','amount_bb','call_bb','to_bb','pot_ratio','raise_ratio','players_active']]
def pf(c1,c2):
    rank={r:i+2 for i,r in enumerate('23456789TJQKA')};a=np.array([rank[c[0]] for c in c1]);b=np.array([rank[c[0]] for c in c2]);return np.clip((a+b-4)/24*.65+(a==b)*.35+np.array([x[1]==y[1] for x,y in zip(c1,c2)])*.06+(abs(a-b)<=2)*.04,0,1)
def transform(g,weak_first,kind):
    net=g['net_direction'].to_numpy().astype(float);v=np.sign(net)*np.log1p(abs(net));ref=(v.sum()-v)/np.sqrt(1+(v*v).sum()-v*v) if kind=='history' else v
    if kind=='policy':
                                                                               
                                                                                
        ref=-np.sign(net)*(g['relationship_fold_payoff_alignment'].to_numpy()+g['relationship_call_payoff_alignment'].to_numpy()-g['relationship_raise_payoff_alignment'].to_numpy())
    orientation=np.sign(ref);weight=(1+orientation*np.where(weak_first,1,-1))/2;weak=g.select(['weak_'+c for c in ATTR]).to_numpy();strong=g.select(['strong_'+c for c in ATTR]).to_numpy();higher=weight[:,None]*weak+(1-weight[:,None])*strong;lower=weight[:,None]*strong+(1-weight[:,None])*weak;X=np.column_stack([higher,lower,abs(ref),orientation*v]);assert np.isfinite(X).all();return X
def build(d,kinds=('outcome','history')):
    labs=pl.read_csv('data/development_labels.csv').select('pair_id','player_1','player_2');meta=[];start=time.time()
    for (table,),g in d.group_by('table_id'):
        s=pl.read_parquet(f'artifacts/compact/seats/table_id={table}/*.parquet').join(g.select('hand_id').unique(),on='hand_id',how='semi');s=s.with_columns(pl.Series('pf',pf(s['hole_card_1'],s['hole_card_2'])));q=g.select('pair_id','hand_id','weak_strength_0','strong_strength_0').join(labs,on='pair_id');q=q.join(s.select('hand_id',C('player_id').alias('player_1'),C('pf').alias('pf1')),on=['hand_id','player_1'],validate='m:1').join(s.select('hand_id',C('player_id').alias('player_2'),C('pf').alias('pf2')),on=['hand_id','player_2'],validate='m:1');assert (q['weak_strength_0']-q.select(pl.min_horizontal('pf1','pf2')).to_series()).abs().max()<1e-6;meta.append(q.select('pair_id','hand_id',(C('pf1')<=C('pf2')).alias('weak_first')))
    d=d.join(pl.concat(meta),on=['pair_id','hand_id'],validate='1:1');names=[f'role_{r}_{c}' for r in ['reference_high','reference_low'] for c in ATTR]+['role_reference_strength','role_payoff_alignment'];parts={k:[] for k in kinds};audit=[]
    for (pid,),g in d.group_by('pair_id'):
        g=g.sort('time','hand_id');w=g['weak_first'].to_numpy();net=g['net_direction'].to_numpy();v=np.sign(net)*np.log1p(abs(net));ref=(v.sum()-v)/np.sqrt(1+(v*v).sum()-v*v);audit.append({'pair_id':pid,'n':len(g),'history_vs_outcome_direction_disagreements':int((np.sign(ref)!=np.sign(net)).sum()),'mean_reference_strength':float(abs(ref).mean())})
        for k in parts:
            X=transform(g,w,k);swapped=g.with_columns((-C('net_direction')).alias('net_direction'));np.testing.assert_allclose(X,transform(swapped,~w,k),atol=1e-7)
            if k=='history':
                modified=net.copy();modified[0]+=10000;z=np.sign(modified)*np.log1p(abs(modified));before=(v[1:].sum()/np.sqrt(1+(v[1:]**2).sum()));after=z[1:].sum()/np.sqrt(1+(z[1:]**2).sum());assert before==after
            parts[k].append(g.select('pair_id','hand_id').hstack(pl.DataFrame(X,schema=names)))
    for k in parts:pl.concat(parts[k]).write_parquet(ROOT/f'role_{k}_features.parquet')
    audit_name='role_feature_audit.json' if tuple(kinds)==('outcome','history') else 'role_'+'_'.join(kinds)+'_feature_audit.json';(ROOT/audit_name).write_text(json.dumps(audit,indent=2));(ROOT/'role_feature_columns.json').write_text(json.dumps(names));print('role features',round(time.time()-start,1),flush=True)
def main():
    arms=os.environ.get('ROLE_ARMS','outcome,history').split(',');assert all(k in ['outcome','history','policy'] for k in arms);suffix='' if arms==['outcome','history'] else '_'+'_'.join(arms);cfg={'arms':arms,'role_attributes':ATTR,'reference':'leave-current-hand-out signed log1p net difference / root sum squares' if 'policy' not in arms else 'signed off-hand call plus fold minus raise policy residuals; recovered from payoff alignment, average roles at net ties','iterations':400,'depth':5,'learning_rate':.035,'l2_leaf_reg':8,'prediction':'replace .25 R29 Cat priority component with corresponding new model inclusion; keep other components frozen'};path=ROOT/f'roles{suffix}_config.json'
    if path.exists():assert json.loads(path.read_text())==cfg
    else:path.write_text(json.dumps(cfg,indent=2))
    d=hand_data()
    missing=[k for k in arms if not (ROOT/f'role_{k}_features.parquet').exists()]
    if missing:build(d,missing)
    bc=json.loads(Path('artifacts/evidence_session6/priority_ordered_columns.json').read_text())['event'];newcols=json.loads((ROOT/'role_feature_columns.json').read_text());cols=bc+newcols;start=time.time();audit=[]
    for kind in cfg['arms']:
        q=d.join(pl.read_parquet(ROOT/f'role_{kind}_features.parquet'),on=['pair_id','hand_id'],validate='1:1',maintain_order='left');X=q.select(cols).to_numpy();pred=np.zeros((len(q),2))
        for f in range(4):
            for b in ['directed_transfer','soft_play','coordinated_isolation']:
                t1,t2,e1,e2,va=targets(d,f,b)
                for k,y,tr in [(1,t1,e1),(2,t2,e2)]:
                    assert not (tr&va).any();m=CatBoostClassifier(iterations=400,depth=5,learning_rate=.035,l2_leaf_reg=8,random_seed=6310+11*f+k,thread_count=4,verbose=False,allow_writing_files=False);m.fit(X[tr],y[tr]);pred[va,k-1]=m.predict_proba(X[va],thread_count=4)[:,1];m.save_model(str(ROOT/f'role_{kind}_event{k}_{b}_fold{f}.cbm'));audit.append({'kind':kind,'fold':f,'family':b,'head':k,'train_hands':int(tr.sum()),'validation_hands':int(va.sum())})
            print('roles',kind,f,round(time.time()-start,1),flush=True)
        q=d.select('pair_id','hand_id','fold','evidence','time').with_columns(pl.Series('primary',pred[:,0]),pl.Series('secondary',pred[:,1]));parts=[]
        for _,g in q.group_by('pair_id'):
            g=g.sort('time','hand_id');parts.append(g.with_columns(pl.Series('inclusion',inclusion(g['primary'].to_numpy(),g['secondary'].to_numpy()))))
        pl.concat(parts).write_parquet(ROOT/f'role_{kind}_oof.parquet')
    (ROOT/f'roles{suffix}_training_audit.json').write_text(json.dumps(audit,indent=2));(ROOT/'roles_columns.json').write_text(json.dumps(cols))
if __name__=='__main__':main()
