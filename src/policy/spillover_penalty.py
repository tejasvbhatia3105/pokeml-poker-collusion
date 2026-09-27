\
import sys,collections,numpy as np,polars as pl
inp,outp=sys.argv[1],sys.argv[2]; gamma=float(sys.argv[3]) if len(sys.argv)>3 else 1.0; pairs_file=sys.argv[4] if len(sys.argv)>4 else 'data/evaluation_pairs.csv'
names=['none','directed_transfer','soft_play','coordinated_isolation']
pp=pl.read_csv(pairs_file) if pairs_file.endswith('.csv') else pl.read_parquet(pairs_file)
p=pl.read_csv(inp).join(pp.select('pair_id','player_1','player_2'),on='pair_id')
s=(1-p['none']).to_numpy(); p1=p['player_1'].to_list(); p2=p['player_2'].to_list()
best=collections.defaultdict(lambda:[0.0,0.0])
for a,b,v in zip(p1,p2,s):
    for x in (a,b):
        m=best[x]
        if v>m[0]: m[1]=m[0]; m[0]=v
        elif v>m[1]: m[1]=v
adj=np.empty(len(s))
for i,(a,b,v) in enumerate(zip(p1,p2,s)):
    ma=best[a][1] if v>=best[a][0] else best[a][0]; mb=best[b][1] if v>=best[b][0] else best[b][0]; m=max(ma,mb)
    adj[i]=v if v>=m or m<=0 else v*(v/m)**gamma
fam=p.select(names[1:]).to_numpy(); fam=fam/np.maximum(fam.sum(1,keepdims=True),1e-12)
out=pl.DataFrame({'pair_id':p['pair_id'],'none':1-adj,**{n:adj*fam[:,k] for k,n in enumerate(names[1:])}}).sort('pair_id')
out.write_csv(outp); print('penalized',int((adj<s-1e-9).sum()),'pairs; >0.5 before',int((s>0.5).sum()),'after',int((adj>0.5).sum()),'; >0.9 before',int((s>0.9).sum()),'after',int((adj>0.9).sum()))
