\
\
\
\
\
import os
os.environ.setdefault('POLARS_MAX_THREADS', '2')
import itertools, json, time
from pathlib import Path
import numpy as np
import torch
from session8_count_conditioning import conditioned

ROOT = Path('artifacts/evidence_session193_selection_marginals')
torch.set_num_threads(2)

def scan(p):
    z = torch.cat([torch.ones((len(p), 1), dtype=p.dtype), torch.zeros((len(p), 4), dtype=p.dtype)], 1)
    result = [z]
    for i in range(p.shape[1]):
        v = p[:, i:i+1]
        z = z*(1-v) + torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], 1)*v
        result.append(z)
    return torch.stack(result, 1)

def selected_probabilities(logits, mask, minimum):
    z = logits.to(torch.float64)
    p = torch.softmax(torch.cat([torch.zeros_like(z[:, :, :1]), z], 2), 2)
    a, b = p[:, :, 1]*mask, p[:, :, 2]*mask
    ap = scan(a); ar = scan(a.flip(1)).flip(1)
    ep = scan(a+b); er = scan((a+b).flip(1)).flip(1)
    degrees = torch.arange(5)[:, None] + torch.arange(5)[None, :]
    ordinary = a*ap[:, :-1].sum(2) + b*torch.einsum('bnk,kl,bnl->bn', ep[:, :-1], (degrees < 5).to(z.dtype), ar[:, 1:])
    low = (degrees[None] < minimum[:, None, None]-1).to(z.dtype)
    subtract = (a+b)*torch.einsum('bnk,bkl,bnl->bn', ep[:, :-1], low, er[:, 1:])
    den = 1-(ep[:, -1]*(torch.arange(5)[None] < minimum[:, None])).sum(1)
    result = (ordinary-subtract)/den.clamp_min(1e-12)[:, None]
    result = torch.where(den[:, None] < 1e-12, ordinary, result)
    return result.clamp(0, 1)*mask

def check():
    rng = np.random.default_rng(193)
    p = rng.dirichlet([4, 1, 1], size=(3, 6))
    z = torch.tensor(np.log(p[:, :, 1:]/p[:, :, :1]), requires_grad=True)
    m = torch.ones((3, 6), dtype=torch.bool)
    k = torch.tensor([1, 3, 5])
    got = selected_probabilities(z, m, k)
    expected = np.zeros((3, 6)); den = np.zeros(3)
    for path in itertools.product(range(3), repeat=6):
        chosen = [i for i, c in enumerate(path) if c == 1] + [i for i, c in enumerate(path) if c == 2]
        for j in range(3):
            if len(chosen) >= k[j]:
                w = np.prod(p[j, np.arange(6), path]); den[j] += w
                expected[j, chosen[:5]] += w
    expected /= den[:, None]
    err = float(abs(got.detach().numpy()-expected).max())
    assert err < 1e-11
    for j in range(3):
        np.testing.assert_allclose(got[j].detach().numpy(), conditioned(p[j, :, 1:], int(k[j])), atol=1e-12, rtol=0)
    assert torch.autograd.gradcheck(lambda zz: selected_probabilities(zz, m, k), (z,), eps=1e-6, atol=1e-5)
    padded = torch.cat([z.detach(), torch.randn(3, 3, 2, dtype=z.dtype)], 1).requires_grad_(True)
    pm = torch.cat([m, torch.zeros(3, 3, dtype=torch.bool)], 1)
    out = selected_probabilities(padded, pm, k)
    np.testing.assert_allclose(out[:, :6].detach(), got.detach(), atol=1e-12, rtol=0)
    out.sum().backward(); assert torch.count_nonzero(padded.grad[:, 6:]) == 0
    ROOT.mkdir(exist_ok=True)
    report = dict(enumerated_paths=3**6, minimum_counts=k.tolist(), probability_error=err,
                  independent_numpy_match=True, full_logit_gradcheck=True, padding_values_and_gradients_invariant=True)
    (ROOT/'math_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    t = time.time()
    large = torch.tensor(rng.normal(-3, 1, size=(280, 465, 2)), requires_grad=True)
    inc = selected_probabilities(large, torch.ones((280, 465), dtype=torch.bool), torch.full((280,), 3))
    inc.square().sum().backward()
    print('BENCHMARK_FORWARD_BACKWARD_SECONDS', time.time()-t, flush=True)

if __name__ == '__main__':
    check()
