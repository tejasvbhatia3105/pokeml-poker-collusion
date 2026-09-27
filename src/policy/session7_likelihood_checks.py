import itertools,json
from pathlib import Path
import numpy as np
from session7_likelihood import posterior,multilevel_inclusion
def main():
    rng=np.random.default_rng(7101);errors=[];cases=0
    for n,levels in [(3,2),(5,2),(6,2),(7,2),(4,3),(6,3)]:
        p=rng.dirichlet([2]+[1]*levels,size=n);groups={};marginal=np.zeros(n)
        for state in itertools.product(range(levels+1),repeat=n):
            chosen=tuple([i for tier in range(1,levels+1) for i,k in enumerate(state) if k==tier][:5]);weight=np.prod(p[np.arange(n),state]);marginal[list(chosen)]+=weight
            if chosen not in groups:groups[chosen]=[0.,np.zeros((n,levels+1))]
            groups[chosen][0]+=weight;groups[chosen][1][np.arange(n),state]+=weight
        total=0.
        for chosen,(mass,counts) in groups.items():
            post,ll,_=posterior(p,chosen);errors.extend([abs(np.exp(ll)-mass),float(np.max(np.abs(post-counts/mass)))]);total+=np.exp(ll);cases+=1
        assert abs(total-1)<1e-12
        errors.append(float(np.max(np.abs(marginal-multilevel_inclusion(p)))))
    assert max(errors)<1e-12
    assert posterior(np.ones((6,3))/3,[2,0,4,1,5])[0] is None
    report={'observed_lists_checked':cases,'max_error':max(errors),'impossible_three_run_list_rejected':True};Path('artifacts/evidence_session7/likelihood_checks.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
