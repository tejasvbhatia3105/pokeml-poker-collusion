import ctypes,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('artifacts/evidence_session117_fast_count')
LIB=ctypes.CDLL(str((ROOT/'count_tail.dylib').resolve()))
D=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');B=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS');I=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
LIB.count_tail.argtypes=[D,B,I,ctypes.c_int,ctypes.c_int,D,D];LIB.count_tail.restype=ctypes.c_int

def provenance():
    paths=[Path(__file__),Path('src/policy/session117_count_tail.cpp'),ROOT/'count_tail.dylib']
    return {str(p.resolve().relative_to(Path.cwd())):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths}

class CountTail(torch.autograd.Function):
    @staticmethod
    def forward(ctx,logits,mask,minimum):
        a=logits.detach().cpu().double().numpy();z=np.concatenate([np.zeros_like(a[:,:,:1]),a],2);z-=z.max(2,keepdims=True)
        p=np.exp(z);p/=p.sum(2,keepdims=True);m=np.ascontiguousarray(mask.detach().cpu().numpy(),dtype=np.uint8);k=np.ascontiguousarray(minimum.detach().cpu().numpy(),dtype=np.int32)
        out=np.empty(len(a));gradient=np.empty_like(a)
        assert LIB.count_tail(p,m,k,len(a),a.shape[1],out,gradient)==0
        g=torch.from_numpy(gradient).to(device=logits.device,dtype=logits.dtype);ctx.save_for_backward(g)
        return torch.from_numpy(out).to(device=logits.device,dtype=logits.dtype)
    @staticmethod
    def backward(ctx,grad_output):
        gradient,=ctx.saved_tensors
        return grad_output[:,None,None]*gradient,None,None

def log_count_at_least(logits,mask,minimum):return CountTail.apply(logits,mask,minimum)

def verify():
    from session12_frozen_action import log_count_at_least as reference
    torch.manual_seed(11701);records=[]
    for length in [1,3,6,19,465]:
        x=torch.randn(6,length,2,dtype=torch.float64,requires_grad=True)*2
        mask=torch.rand(6,length)>.15;minimum=torch.arange(6)
        old=reference(x,mask,minimum);new=log_count_at_least(x,mask,minimum)
        go,=torch.autograd.grad(old.sum(),x,retain_graph=True);gn,=torch.autograd.grad(new.sum(),x)
        fe=float(abs(old-new).max());ge=float(abs(go-gn).max());assert fe<1e-10 and ge<1e-10,(fe,ge)
        records.append(dict(length=length,forward_max_error=fe,gradient_max_error=ge))
    x=torch.randn(2,6,2,dtype=torch.float64,requires_grad=True);mask=torch.ones(2,6,dtype=torch.bool);minimum=torch.tensor([2,5])
    assert torch.autograd.gradcheck(lambda z:log_count_at_least(z,mask,minimum),x,eps=1e-6,atol=1e-5,rtol=1e-4)
    xx=torch.randn(16,465,2,requires_grad=True);mm=torch.ones(16,465,dtype=torch.bool);kk=torch.full((16,),3);timing={}
    for name,fn in [('reference',reference),('native',log_count_at_least)]:
        start=time.time()
        for i in range(10):y=fn(xx,mm,kk);torch.autograd.grad(y.sum(),xx)
        timing[name]=time.time()-start
    report=dict(records=records,finite_difference_gradcheck=True,ten_forward_backward_seconds=timing,provenance=provenance(),
        method='Same cardinality tail, computed in double precision; derivative uses independent prefix/suffix count probabilities. This is not a new evidence model.')
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':verify()
