import os,json,hashlib
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl,torch
from catboost import CatBoostRanker,CatBoostClassifier
from session8_data import hand_data,reference
from session10_list_learning import ROOT,features,Model,CONFIG,template
from session10_activity import correlated_selection
from session6_priority import inclusion
C=pl.col
def main():
    d=hand_data();rank_saved=pl.read_parquet(ROOT/'ranker_oof.parquet');list_saved=pl.read_parquet(ROOT/'list_learning_oof.parquet');activity_saved=pl.read_parquet(ROOT/'activity_oof.parquet');bc=json.loads((ROOT/'ranker_hand_columns.json').read_text());fit={(x['fold'],x['family']):x['parameters'] for x in json.loads((ROOT/'activity_audit.json').read_text())};report={'ranker_max_replay_error':0.,'list_max_replay_error':0.,'list_permutation_error':0.,'activity_max_replay_error':0.,'list_initial_score_max_error':0.,'role_max_replay_error':0.}
    for f in range(4):
        q=d.join(pl.read_parquet(f'artifacts/evidence_session9/nested_outer{f}.parquet').drop('fold','time'),on=['pair_id','hand_id'],validate='1:1');bags=[]
        for (pid,),g in q.group_by('pair_id'):
            g=g.sort('time','hand_id');x,prior=features(g);e=np.flatnonzero(g['evidence_rank'].is_not_null());e=e[np.argsort(g['evidence_rank'].to_numpy()[e])];bags.append({'g':g,'pid':pid,'fold':g['fold'][0],'X':x,'prior':prior,'compatible':len(template(len(g),e))>0})
        bags.sort(key=lambda b:b['pid']);n=max(len(b['g']) for b in bags);N=len(bags);nf=bags[0]['X'].shape[1];X=np.zeros((N,n,nf),dtype='float32');P=np.zeros((N,n,2),dtype='float32');M=np.zeros((N,n),bool)
        for i,b in enumerate(bags):k=len(b['g']);X[i,:k]=b['X'];P[i,:k]=b['prior'];M[i,:k]=True
        tr=np.array([b['fold']!=f and b['compatible'] for b in bags]);va=np.array([b['fold']==f for b in bags]);mu=X[tr][M[tr]].mean(0);sd=np.maximum(.05,X[tr][M[tr]].std(0));XX=torch.tensor(np.clip((X[va]-mu)/sd,-6,6));PP=torch.tensor(P[va]);MM=torch.tensor(M[va]);valid=[b for b in bags if b['fold']==f];pred={}
        for kind in ['independent','contextual']:
            pred[kind]=[]
            for seed in CONFIG['seeds']:
                state=torch.load(ROOT/f'list_{kind}_fold{f}_seed{seed}.pt',weights_only=False);np.testing.assert_array_equal(mu,state['mu']);np.testing.assert_array_equal(sd,state['sd']);m=Model(nf,kind);m.load_state_dict(state['state_dict']);m.eval()
                with torch.no_grad():
                    delta=m(XX,MM);rev=m(XX.flip(1),MM.flip(1)).flip(1);report['list_permutation_error']=max(report['list_permutation_error'],float(abs(delta-rev).max()));logits=PP+delta;pred[kind].append(torch.softmax(torch.cat([torch.zeros_like(logits[:,:,:1]),logits],2),2).numpy())
        rankers={}
        for kind in ['compact','full']:m=CatBoostRanker();m.load_model(str(ROOT/f'ranker_{kind}_fold{f}.cbm'));rankers[kind]=m
        for k,b in enumerate(valid):
            g=b['g'];pid=b['pid'];count=len(g);x,prior=features(g);mutated=g.with_columns((1-C('evidence')).alias('evidence'),pl.lit(999).alias('evidence_rank'),pl.lit(99).alias('subtype'));mx,mp=features(mutated);np.testing.assert_array_equal(x,mx);np.testing.assert_array_equal(prior,mp)
            expected=g.select('pair_id','hand_id').join(list_saved,on=['pair_id','hand_id'],maintain_order='left',validate='1:1')
            for kind in pred:
                inc=np.mean([inclusion(p[k,:count,1],p[k,:count,2]) for p in pred[kind]],axis=0);s=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inc;report['list_max_replay_error']=max(report['list_max_replay_error'],float(abs(s-expected[kind].to_numpy()).max()))
            ip=torch.softmax(torch.cat([torch.zeros((count,1)),torch.tensor(prior)],1),1).numpy();s=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inclusion(ip[:,1],ip[:,2]);report['list_initial_score_max_error']=max(report['list_initial_score_max_error'],float(abs(s-g['r29'].to_numpy()).max()))
            ix=np.lexsort((g['hand_id'].to_numpy(),-g['r29'].to_numpy()))[:20];gg=g[ix];rr=np.clip(gg['r29'].to_numpy(),1e-5,1-1e-5);pr=np.log(rr)-np.log1p(-rr);er=gg.select('pair_id','hand_id').join(rank_saved,on=['pair_id','hand_id'],maintain_order='left',validate='1:1')
            for kind,xx in [('compact',x[ix]),('full',np.column_stack([x[ix],gg.select(bc).to_numpy()]))]:
                s=pr+rankers[kind].predict(xx);np.testing.assert_allclose(s,pr+rankers[kind].predict(xx[::-1])[::-1],atol=1e-12);report['ranker_max_replay_error']=max(report['ranker_max_replay_error'],float(abs(s-er['ranker_'+kind].to_numpy()).max()))
            a=g.select('cat_primary','cat_secondary').to_numpy();h=g.select('hist_primary','hist_secondary').to_numpy();p=.5*(a/np.maximum(1,a.sum(1))[:,None]+h/np.maximum(1,h.sum(1))[:,None]);p=np.column_stack([np.maximum(1e-8,1-p.sum(1)),p]);p/=p.sum(1,keepdims=True);s=.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*correlated_selection(p,g['time'].to_numpy()*5000,fit[f,g['behavior_family'][0]]);expected=g.select('pair_id','hand_id').join(activity_saved,on=['pair_id','hand_id'],maintain_order='left',validate='1:1');report['activity_max_replay_error']=max(report['activity_max_replay_error'],float(abs(s-expected['activity'].to_numpy()).max()))
    rc=json.loads((ROOT/'roles_columns.json').read_text())
    for kind in ['outcome','history','policy']:
        q=d.join(pl.read_parquet(ROOT/f'role_{kind}_features.parquet'),on=['pair_id','hand_id'],validate='1:1').join(pl.read_parquet(ROOT/f'role_{kind}_oof.parquet').select('pair_id','hand_id','primary','secondary'),on=['pair_id','hand_id'],validate='1:1')
        for f in range(4):
            for b in ['directed_transfer','soft_play','coordinated_isolation']:
                g=q.filter((C('fold')==f)&(C('behavior_family')==b));X=g.select(rc).to_numpy()
                for k,col in [(1,'primary'),(2,'secondary')]:
                    m=CatBoostClassifier();m.load_model(str(ROOT/f'role_{kind}_event{k}_{b}_fold{f}.cbm'));s=m.predict_proba(X,thread_count=4)[:,1];report['role_max_replay_error']=max(report['role_max_replay_error'],float(abs(s-g[col].to_numpy()).max()))
    assert report['list_permutation_error']<3e-6
    for key in ['ranker_max_replay_error','list_max_replay_error','activity_max_replay_error','role_max_replay_error']:assert report[key]<1e-6,(key,report[key])
    for name in ['list_likelihood_checks','activity_checks']:report[name]=json.loads((ROOT/f'{name}.json').read_text())
    report['label_mutation']='Feature tensors unchanged';report['normalization']='Matches outer training compatible pairs only';sha=hashlib.sha256(open('artifacts/candidate_r29/submission.csv','rb').read()).hexdigest();assert sha=='0fe7747d01d1c64099b803fb830f995ae1b4ca055dd1333fd260475732d10c18';report['r29_sha256']=sha;(ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
