\
\
\
\
\
import os,json
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np,polars as pl,torch
from session8_data import hand_data
from session10_list_learning import ROOT,Model,CONFIG,features
from session6_priority import inclusion
C=pl.col
def ap(g,s):
    ix=np.lexsort((g['hand_id'].to_numpy(),-s))[:5];y=g['evidence'].to_numpy()[ix];return float((y*np.cumsum(y)/np.arange(1,len(y)+1)).sum()/min(5,g['evidence'].sum()))
def main():
    d=hand_data();rows=[];predictions=[]
    for f in range(4):
        models=[]
        for seed in CONFIG['seeds']:
            state=torch.load(ROOT/f'list_independent_fold{f}_seed{seed}.pt',weights_only=False);m=Model(state['feature_count'],'independent');m.load_state_dict(state['state_dict']);m.eval();models.append((m,state))
        for j in range(4):
            if f==j:continue
            nested=pl.read_parquet(f'artifacts/evidence_session9/nested_outer{j}.parquet').filter(C('fold')==f);q=d.filter(C('fold')==f).join(nested.drop('fold','time'),on=['pair_id','hand_id'],validate='1:1');audit=json.load(open(f'artifacts/evidence_session9/nested_outer{j}_audit.json'))
            for r in audit:
                if r['inner']==f:assert f not in r['training_folds'] and j not in r['training_folds']
            for (pid,),g in q.group_by('pair_id'):
                g=g.sort('time','hand_id');X,prior=features(g);scores=[]
                for m,state in models:
                    x=torch.tensor(np.clip((X-state['mu'])/state['sd'],-6,6))[None];mask=torch.ones((1,len(g)),dtype=torch.bool)
                    with torch.no_grad():logits=torch.tensor(prior)[None]+m(x,mask);p=torch.softmax(torch.cat([torch.zeros_like(logits[:,:,:1]),logits],2),2)[0].numpy()
                    scores.append(.25*g['base'].to_numpy()+.25*g['cat_inclusion'].to_numpy()+.5*inclusion(p[:,1],p[:,2]))
                corrected=np.mean(scores,0);rows.append({'pair_id':pid,'outer_fold':f,'additional_excluded_teacher_fold':j,'family':g['behavior_family'][0],'small_teacher_baseline':ap(g,g['r29'].to_numpy()),'small_teacher_corrected':ap(g,corrected)});predictions.append(g.select('pair_id','hand_id','fold','evidence').with_columns(pl.lit(j).alias('additional_excluded_fold'),C('fold').alias('outer_fold'),pl.Series('matched_base',g['r29'].to_numpy()),pl.Series('matched_corrected',corrected)))
    raw=pl.concat(predictions);raw.write_parquet(ROOT/'teacher_shift_hand_predictions.parquet');ensemble=raw.group_by('pair_id','hand_id','fold','evidence').agg(C('matched_base','matched_corrected').mean(),pl.len().alias('members'));assert ensemble['members'].min()==3 and ensemble['members'].max()==3;ensemble.write_parquet(ROOT/'matched_teacher_ensemble_oof.parquet')
    r=pl.DataFrame(rows).with_columns((C('small_teacher_corrected')-C('small_teacher_baseline')).alias('gain'));r.write_csv(ROOT/'teacher_shift.csv');report={'status':'post-result diagnostic, not candidate validation','teacher_training_pools':'two original folds for both training-score and diagnostic validation-score teachers','cases':len(r),'pairs':r['pair_id'].n_unique(),'mean_small_teacher_baseline':r['small_teacher_baseline'].mean(),'mean_small_teacher_corrected':r['small_teacher_corrected'].mean(),'mean_gain':r['gain'].mean(),'families':r.group_by('family').agg(C('small_teacher_baseline','small_teacher_corrected','gain').mean()).to_dicts(),'folds':r.group_by('outer_fold').agg(C('small_teacher_baseline','small_teacher_corrected','gain').mean()).sort('outer_fold').to_dicts()};(ROOT/'teacher_shift.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
