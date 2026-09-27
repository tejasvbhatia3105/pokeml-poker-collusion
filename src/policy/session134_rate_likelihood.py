\
\
\
\
\
import json,sys,time
import numpy as np,torch
from pathlib import Path
from scipy.optimize import minimize
import session130_hierarchical_list as h

ROOT=Path('artifacts/evidence_session134_rate_likelihood')

def objective(P,A,M,T,V,D,K,fam,train,sigma,kind):
    z=P[train]+A[train]
    nll=h.mixture_nll(z,M[train],T[train],V[train],D[train],K[train],sigma[fam[train]])
    reg=(A[train].square().sum(2)*M[train]).sum(1)/M[train].sum(1)
    return nll.sum()+h.CONFIG['ridge']*reg.sum()

def rate_opt(P,A,M,T,V,D,K,fam,tr,sigma):
    def f(v):
        s=torch.tensor(np.sqrt(v),dtype=P.dtype,requires_grad=True)
        loss=objective(P,A,M,T,V,D,K,fam,tr,s,'random_rate')
        gs,=torch.autograd.grad(loss,s);value=float(loss.detach());g=np.zeros(3)
        for j in range(3):
            if v[j]>1e-8:g[j]=float(gs[j])/(2*np.sqrt(v[j]))
            else:
                                                                          
                u=s.detach().clone();u[j]=.01
                with torch.no_grad():g[j]=(float(objective(P,A,M,T,V,D,K,fam,tr,u,'random_rate'))-value)/1e-4
        return value,g
    initial=sigma.numpy()**2
    result=minimize(f,initial,jac=True,method='L-BFGS-B',bounds=[(0,h.CONFIG['rate_bound']**2)]*3,
                    options=dict(maxiter=h.CONFIG['rate_maxiter'],ftol=1e-9,gtol=1e-5,maxls=20))
    return torch.tensor(np.sqrt(result.x),dtype=P.dtype),dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit))

def setup():
    h.ROOT=ROOT;h.CONFIG=dict(h.CONFIG,kinds=['random_rate'],quadrature=17,
                            rate_parameterization='Variance; analytic chain-rule gradients above zero, one-sided derivative at zero',
                            regularization='Original deterministic hand-correction penalty only; variance estimated by likelihood.',
                            followup='Session133 showed the per-pair variance penalty binds even where training likelihood favours variance. One fixed follow-up; no held-out weight search.')
    h.QX,h.QW=np.polynomial.hermite.hermgauss(17);h.QX*=np.sqrt(2);h.QW/=np.sqrt(np.pi)
    h.objective=objective;h.rate_opt=rate_opt

if __name__=='__main__':
    setup()
    {'train':h.train,'verify':h.verify,'check':h.check}[sys.argv[1]]()
